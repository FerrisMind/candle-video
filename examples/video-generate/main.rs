// Universal video generation CLI (Wan 2.1 T2V 1.3B MVP).
//
// LTX users should continue using `cargo run --example ltx-video`.

use std::path::{Path, PathBuf};

use candle_core::DType;
use candle_video::profiling::init_tracing;
use candle_video::utils::latents_fixture::load_latents_json;
use candle_video::utils::video_export::{PixelRange, export_video_output_with_mp4};
use candle_video::{
    GenerateRequest, MemoryOptions, VideoPipeline, WanDevicePlan, WanPipeline, WanPipelineAdapter,
    WanSchedulerProfile,
};
use clap::Parser;

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
}

fn main() -> anyhow::Result<()> {
    init_tracing();
    let args = Args::parse();

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
        println!(
            "Device plan: compute={:?} text_encoder={:?} vae={:?} transformer={:?}",
            plan.compute, plan.text_encoder, plan.vae, plan.transformer_dtype
        );
    }

    println!("Loading Wan pipeline from {} …", args.weights.display());
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
    let mut adapter = WanPipelineAdapter::new(pipeline);

    let initial_latents = match &args.latents_json {
        Some(path) => {
            println!("Loading initial latents from {} …", path.display());
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

    println!(
        "Generating {}x{} video, {} frames, {} steps …",
        req.width, req.height, req.frames, req.steps
    );
    let out = adapter.generate(req, &plan.compute)?;
    let dims = out.frames.dims();
    println!("Output tensor shape: {dims:?}");

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
    export_video_output_with_mp4(
        &out,
        &output_dir,
        PixelRange::NegOneToOne,
        save_png,
        save_gif,
        args.output.as_deref(),
        args.fps,
    )?;

    if save_png {
        println!("Saved PNG frames to {}", output_dir.display());
    }
    if save_gif {
        println!("Saved GIF to {}", output_dir.join("video.gif").display());
    }
    if let Some(path) = args.output {
        println!("Saved MP4 to {}", path.display());
    }

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
