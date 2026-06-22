//! Universal video generation CLI (Wan 2.1 T2V 1.3B MVP).
//!
//! LTX users should continue using `cargo run --example ltx-video`.

use std::path::{Path, PathBuf};

use candle_video::profiling::init_tracing;
use candle_video::utils::latents_fixture::load_latents_json;
use candle_video::utils::video_export::{PixelRange, export_video_output};
use candle_video::{
    GenerateRequest, MemoryOptions, VideoPipeline, WanDevicePlan, WanPipeline, WanPipelineAdapter,
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

    #[arg(long)]
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

    #[arg(long)]
    seed: Option<u64>,

    /// PyTorch-exported initial latents (`scripts/wan/export_latents.py`) for seed parity with Diffusers.
    #[arg(long)]
    latents_json: Option<PathBuf>,

    #[arg(long, default_value = "output")]
    output_dir: String,

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

    #[arg(long)]
    vae_tiling: bool,

    #[arg(long)]
    vae_slicing: bool,

    #[arg(long)]
    cpu_offload: bool,
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
        cpu_offload: args.cpu_offload,
    };

    let plan = WanDevicePlan::for_wan(
        args.cpu,
        if args.transformer_f32 {
            Some(false)
        } else {
            None
        },
        &memory,
    )?;

    if !args.cpu {
        println!(
            "Device plan: compute={:?} text_encoder={:?} vae={:?} transformer={:?}",
            plan.compute, plan.text_encoder, plan.vae, plan.transformer_dtype
        );
    }

    println!("Loading Wan pipeline from {} …", args.weights.display());
    let pipeline = WanPipeline::load_with_devices(
        &args.weights,
        &plan.compute,
        &plan.text_encoder,
        &plan.vae,
        plan.transformer_dtype,
        plan.vae_dtype,
        plan.sequential_gpu,
        plan.vae_tiling,
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
        frame_rate: 16,
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

    if !Path::new(&args.output_dir).exists() {
        std::fs::create_dir_all(&args.output_dir)?;
    }

    let save_png = args.frames;
    let save_gif = !args.frames || args.gif;
    export_video_output(
        &out,
        &args.output_dir,
        PixelRange::NegOneToOne,
        save_png,
        save_gif,
    )?;

    if save_png {
        println!("Saved PNG frames to {}", args.output_dir);
    }
    if save_gif {
        println!("Saved GIF to {}/video.gif", args.output_dir);
    }

    Ok(())
}
