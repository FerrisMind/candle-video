// Universal video generation CLI (Wan 2.1 T2V 1.3B MVP).
//
// LTX users should continue using `cargo run --example ltx-video`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use candle_core::DType;
use candle_video::profiling::init_tracing;
use candle_video::utils::latents_fixture::load_latents_json;
use candle_video::utils::video_export::{
    PixelRange, export_video_output_with_mp4_options_and_observer,
};
use candle_video::{
    GenerateRequest, GenerationEvent, GenerationStage, MemoryOptions, ProgressObserver,
    RunManifest, VideoPipeline, WanDevicePlan, WanPipeline, WanPipelineAdapter,
    WanSchedulerProfile, write_run_manifest,
};
use clap::Parser;

#[path = "../common/progress.rs"]
mod progress;

use progress::{CliProgressObserver, CliProgressOptions, LogFormat, ProgressMode};

#[derive(Parser, Debug)]
#[command(
    name = "video-generate",
    about = "Generate video with candle-video backends"
)]
struct Args {
    #[arg(long, default_value = "wan:2.1-t2v-1.3b")]
    model: String,

    #[arg(long, visible_alias = "model-path")]
    weights: PathBuf,

    #[arg(long, default_value = "A cat playing with a ball of yarn")]
    prompt: String,

    #[arg(long, default_value = "")]
    negative_prompt: String,

    #[arg(long, default_value_t = 480)]
    height: usize,

    #[arg(long, default_value_t = 832)]
    width: usize,

    #[arg(long, default_value_t = 81)]
    num_frames: usize,

    #[arg(long, default_value_t = 50)]
    steps: usize,

    #[arg(long, default_value_t = 5.0)]
    guidance_scale: f32,

    /// Compatibility profile: diffusers, official-wan, or flow-match-euler.
    #[arg(long, default_value = "diffusers")]
    scheduler_profile: WanSchedulerProfile,

    /// Optional explicit flow/sample shift for the selected scheduler.
    #[arg(long)]
    scheduler_shift: Option<f32>,

    #[arg(long)]
    seed: Option<u64>,

    /// PyTorch-exported initial latents (`scripts/wan/export_latents.py`) for seed parity with Diffusers.
    #[arg(long)]
    latents_json: Option<PathBuf>,

    #[arg(long, default_value = "output")]
    output_dir: String,

    /// Explicit MP4 output path. Parent directories are created automatically.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Output frame rate used by GIF/MP4 encoding.
    #[arg(long, default_value_t = 16)]
    fps: usize,

    /// Save individual PNG frames (exclusive with default GIF output).
    #[arg(long)]
    frames: bool,

    /// Save GIF animation (default when `--frames` is not set).
    #[arg(long)]
    gif: bool,

    #[arg(long)]
    cpu: bool,

    /// Force F32 transformer weights (default on CUDA is F16).
    #[arg(long)]
    transformer_f32: bool,

    /// Transformer compute dtype (`f16`, `bf16`, or `f32`).
    #[arg(long)]
    dtype: Option<String>,

    #[arg(long)]
    vae_tiling: bool,

    #[arg(long)]
    vae_slicing: bool,

    #[arg(long)]
    cpu_offload: bool,

    /// Alias for the explicit CPU-offload/sequential-loading path.
    #[arg(long)]
    low_vram: bool,

    /// Progress renderer: auto uses a terminal bar and plain lines when piped.
    #[arg(long, value_enum, default_value_t = ProgressMode::Auto)]
    progress: ProgressMode,

    /// Human terminal events or JSONL events on stderr.
    #[arg(long, value_enum, default_value_t = LogFormat::Human)]
    log_format: LogFormat,

    /// Increase diagnostic detail (repeat for more verbosity).
    #[arg(short = 'v', long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Suppress human progress output while retaining machine events.
    #[arg(short = 'q', long)]
    quiet: bool,

    /// Heartbeat interval for long model operations; zero disables heartbeats.
    #[arg(long, default_value_t = 5)]
    heartbeat_seconds: u64,

    /// Include elapsed timings in human output.
    #[arg(long)]
    timings: bool,

    /// Show detailed configuration and stage diagnostics.
    #[arg(long, visible_alias = "trace-stages")]
    diagnostics: bool,

    /// Add best-effort nvidia-smi snapshots to heartbeat lines.
    #[arg(long)]
    gpu_stats: bool,

    /// Write structured events to a JSONL file as well as stderr.
    #[arg(long, visible_alias = "log-file")]
    events_jsonl: Option<PathBuf>,

    /// Write a run manifest beside the generated output.
    #[arg(long)]
    dump_run_manifest: bool,

    /// Retain an incomplete `.partial` MP4 when export fails.
    #[arg(long)]
    keep_partial: bool,
}

fn main() -> anyhow::Result<()> {
    init_tracing();
    let args = Args::parse();

    let renderer = CliProgressObserver::new(CliProgressOptions {
        progress: args.progress,
        log_format: args.log_format,
        quiet: args.quiet,
        verbose: args.verbose,
        timings: args.timings,
        diagnostics: args.diagnostics,
        gpu_stats: args.gpu_stats,
        heartbeat_seconds: args.heartbeat_seconds,
        events_jsonl: args.events_jsonl.clone(),
    })?;
    CliProgressObserver::install_ctrlc(&renderer)?;
    let observer: Arc<dyn ProgressObserver> = renderer.clone();

    match args.model.as_str() {
        "wan:2.1-t2v-1.3b" | "wan2.1-t2v-1.3b" => {}
        "ltx" | "ltx:0.9.8-2b-distilled" => {
            anyhow::bail!("use `cargo run --example ltx-video` for LTX models");
        }
        other => anyhow::bail!("unsupported model {other}"),
    }

    if !args.weights.join("model_index.json").exists()
        && !args.weights.join("text_encoder_gguf").exists()
    {
        anyhow::bail!(
            "weights path must be a Diffusers Wan folder (model_index.json) or consolidated bundle (text_encoder_gguf/): {}",
            args.weights.display()
        );
    }

    #[cfg(all(feature = "cuda", not(feature = "flash-attn")))]
    if !args.cpu {
        eprintln!(
            "WARNING: built with `cuda` but without `flash-attn`. Wan will OOM on 12 GB GPUs. \
             Rebuild: cargo build --release --example video-generate --features flash-attn,cuda"
        );
    }

    let memory = MemoryOptions {
        vae_tiling: args.vae_tiling,
        vae_slicing: args.vae_slicing,
        cpu_offload: args.cpu_offload || args.low_vram,
    };

    let mut plan = WanDevicePlan::for_wan(
        args.cpu,
        if args.transformer_f32 {
            Some(false)
        } else {
            None
        },
        &memory,
    )?;
    if let Some(dtype) = args.dtype.as_deref() {
        plan.transformer_dtype = parse_dtype(dtype)?;
    }

    if !args.cpu {
        eprintln!(
            "Device plan: compute={:?} text_encoder={:?} vae={:?} transformer={:?}",
            plan.compute, plan.text_encoder, plan.vae, plan.transformer_dtype
        );
    }

    let load_started = Instant::now();
    observer.on_event(&GenerationEvent::StageStarted {
        stage: GenerationStage::LoadComponents,
        message: format!("Loading Wan components from {}", args.weights.display()),
    });
    let pipeline = WanPipeline::load_with_devices_and_scheduler(
        &args.weights,
        &plan.compute,
        &plan.text_encoder,
        &plan.vae,
        plan.transformer_dtype,
        plan.vae_dtype,
        plan.sequential_gpu,
        plan.vae_tiling,
        args.scheduler_profile,
        args.scheduler_shift,
    )?;
    observer.on_event(&GenerationEvent::StageFinished {
        stage: GenerationStage::LoadComponents,
        elapsed_secs: load_started.elapsed().as_secs_f64(),
    });
    let mut adapter = WanPipelineAdapter::new(pipeline);

    let initial_latents = match &args.latents_json {
        Some(path) => {
            eprintln!("Loading initial latents from {} …", path.display());
            Some(load_latents_json(path, &plan.compute)?)
        }
        None => None,
    };

    let req = GenerateRequest {
        prompt: args.prompt,
        negative_prompt: if args.negative_prompt.is_empty() {
            None
        } else {
            Some(args.negative_prompt)
        },
        width: args.width,
        height: args.height,
        frames: args.num_frames,
        steps: args.steps,
        guidance_scale: args.guidance_scale,
        guidance_rescale: 0.0,
        seed: args.seed,
        frame_rate: args.fps,
        task: candle_video::VideoTask::TextToVideo,
        memory,
        output: Default::default(),
        initial_latents,
        prompt_embeds: None,
        prompt_attention_mask: None,
        negative_prompt_embeds: None,
        negative_prompt_attention_mask: None,
    };

    eprintln!(
        "Generating {}x{} video, {} frames, {} steps …",
        req.width, req.height, req.frames, req.steps
    );
    let generation_started = Instant::now();
    let out = adapter.generate_with_observer(req, &plan.compute, Arc::clone(&observer))?;
    let dims = out.frames.dims();
    if args.diagnostics {
        eprintln!("Output tensor shape: {dims:?}");
    }

    let output_dir = args
        .output
        .as_ref()
        .and_then(|path| {
            path.parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .map(Path::to_path_buf)
        })
        .unwrap_or_else(|| PathBuf::from(&args.output_dir));
    if !output_dir.exists() {
        std::fs::create_dir_all(&output_dir)?;
    }

    let save_png = args.frames;
    let save_gif = args.gif || (args.output.is_none() && !args.frames);
    let output_path = args.output.clone();
    export_video_output_with_mp4_options_and_observer(
        &out,
        &output_dir,
        PixelRange::NegOneToOne,
        save_png,
        save_gif,
        output_path.as_deref(),
        args.fps,
        args.keep_partial,
        observer.as_ref(),
    )?;

    if save_png {
        println!("Saved PNG frames to {}", output_dir.display());
    }
    if save_gif {
        println!("Saved GIF to {}", output_dir.join("video.gif").display());
    }
    let generated_output = output_path
        .as_ref()
        .map(|path| path.display().to_string())
        .or_else(|| {
            if save_png {
                Some(output_dir.display().to_string())
            } else if save_gif {
                Some(output_dir.join("video.gif").display().to_string())
            } else {
                None
            }
        });

    if let Some(path) = output_path.as_ref() {
        println!("Saved MP4 to {}", path.display());
    }

    if args.dump_run_manifest {
        let manifest_path = output_path.as_ref().map_or_else(
            || output_dir.join("run.json"),
            |path| path.with_extension("json"),
        );
        let manifest = RunManifest {
            model: "Wan2.1-T2V-1.3B".to_string(),
            backend: if args.cpu { "cpu" } else { "cuda" }.to_string(),
            device: format!("{:?}", plan.compute),
            dtype: format!("{:?}", plan.transformer_dtype),
            width: out.width,
            height: out.height,
            frames: out.num_frames()?,
            fps: args.fps,
            steps: args.steps,
            guidance_scale: args.guidance_scale,
            seed: args.seed,
            scheduler: args.scheduler_profile.to_string(),
            total_seconds: generation_started.elapsed().as_secs_f64(),
            peak_vram_bytes: renderer.peak_vram_bytes(),
            output: generated_output.clone().unwrap_or_default(),
        };
        observer.on_event(&GenerationEvent::StageStarted {
            stage: GenerationStage::Finalize,
            message: format!("Writing run manifest to {}", manifest_path.display()),
        });
        let finalize_started = Instant::now();
        write_run_manifest(&manifest_path, &manifest)?;
        observer.on_event(&GenerationEvent::StageFinished {
            stage: GenerationStage::Finalize,
            elapsed_secs: finalize_started.elapsed().as_secs_f64(),
        });
        println!("Saved run manifest to {}", manifest_path.display());
    }

    observer.on_event(&GenerationEvent::GenerationFinished {
        output: generated_output,
        elapsed_secs: generation_started.elapsed().as_secs_f64(),
    });

    Ok(())
}

fn parse_dtype(value: &str) -> anyhow::Result<DType> {
    match value.trim().to_ascii_lowercase().as_str() {
        "f32" | "float32" => Ok(DType::F32),
        "f16" | "float16" | "half" => Ok(DType::F16),
        "bf16" | "bfloat16" => Ok(DType::BF16),
        other => {
            anyhow::bail!("unsupported transformer dtype `{other}`; expected f16, bf16, or f32")
        }
    }
}
