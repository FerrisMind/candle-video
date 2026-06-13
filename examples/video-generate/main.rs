//! Universal video generation CLI (Wan 2.1 T2V 1.3B MVP).
//!
//! LTX users should continue using `cargo run --example ltx-video`.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_video::{
    GenerateRequest, VideoPipeline, WanPipeline, WanPipelineAdapter,
};
use candle_video::utils::latents_fixture::load_latents_json;
use candle_video::utils::video_export::{export_video_output, PixelRange};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "video-generate", about = "Generate video with candle-video backends")]
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

    /// Run transformer in F16 on CUDA (saves VRAM; needed for 480p on 12 GB GPUs).
    #[arg(long)]
    transformer_f16: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    match args.model.as_str() {
        "wan:2.1-t2v-1.3b" | "wan2.1-t2v-1.3b" => {}
        "ltx" | "ltx:0.9.8-2b-distilled" => {
            anyhow::bail!("use `cargo run --example ltx-video` for LTX models");
        }
        other => anyhow::bail!("unsupported model {other}"),
    }

    if !args.weights.join("model_index.json").exists() {
        anyhow::bail!(
            "weights path must be a Diffusers Wan folder with model_index.json: {}",
            args.weights.display()
        );
    }

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0)?
    };
    let text_encoder_device = if args.cpu {
        Device::Cpu
    } else {
        println!("Text encoder on CPU (UMT5-XXL does not fit in VRAM alongside transformer on 12 GB GPUs).");
        Device::Cpu
    };
    if !args.cpu {
        println!("VAE decode on CPU (frees VRAM for transformer denoise).");
    }

    println!("Loading Wan pipeline from {} …", args.weights.display());
    let transformer_dtype = if args.cpu {
        DType::F32
    } else if args.transformer_f16 {
        println!("Transformer in F16.");
        DType::F16
    } else {
        DType::F32
    };
    let vae_device = if args.cpu {
        Device::Cpu
    } else {
        // Keep VAE off GPU during denoise to save VRAM on 12 GB cards.
        Device::Cpu
    };
    let pipeline = WanPipeline::load_with_devices(
        &args.weights,
        &device,
        &text_encoder_device,
        &vae_device,
        transformer_dtype,
        DType::F32,
    )?;
    let mut adapter = WanPipelineAdapter::new(pipeline);

    let initial_latents = match &args.latents_json {
        Some(path) => {
            println!("Loading initial latents from {} …", path.display());
            Some(load_latents_json(path, &device)?)
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
        memory: Default::default(),
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
    let out = adapter.generate(req, &device)?;
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
