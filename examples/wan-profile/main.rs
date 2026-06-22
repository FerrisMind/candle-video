//! Reproducible Wan profiling fixtures for flamegraph / Tracy / heaptrack.
//!
//! All scenarios use fixed inputs under `tests/fixtures/wan/` for comparable before/after runs.

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_video::profiling::{init_tracing, vram};
use candle_video::{
    WanDenoiseStack, WanDevicePlan, ensure_wan_cuda_requirements, latent_shape_from_config,
    wan_patch_token_count,
};
use clap::{Parser, Subcommand};
use serde_json::Value;

#[derive(Parser, Debug)]
#[command(name = "wan-profile", about = "Wan profiling fixtures (fixed inputs)")]
struct Args {
    #[arg(
        long,
        default_value = "/home/mod479711/Downloads/Wan2.1-T2V-1.3B-Diffusers"
    )]
    weights: PathBuf,

    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    transformer_f32: bool,

    #[command(subcommand)]
    scenario: Scenario,
}

#[derive(Subcommand, Debug)]
enum Scenario {
    /// Synthetic matmul + softmax hot loop (no model weights).
    SyntheticHotOp {
        #[arg(long, default_value_t = 512)]
        seq_len: usize,
        #[arg(long, default_value_t = 200)]
        iters: usize,
    },
    /// Load transformer + VAE + scheduler from disk.
    ModelLoad,
    /// One transformer forward (prefill / step-0 denoise).
    Prefill {
        #[arg(long, default_value_t = 256)]
        height: usize,
        #[arg(long, default_value_t = 448)]
        width: usize,
        #[arg(long, default_value_t = 9)]
        num_frames: usize,
    },
    /// VAE decode only (latents from denoise fixture or synthetic).
    VaeDecodeOnly {
        #[arg(long, default_value_t = 480)]
        height: usize,
        #[arg(long, default_value_t = 832)]
        width: usize,
        #[arg(long, default_value_t = 81)]
        num_frames: usize,
        #[arg(long)]
        vae_tiling: bool,
    },
    /// Denoise loop without VAE decode.
    DenoiseOnly {
        #[arg(long, default_value_t = 256)]
        height: usize,
        #[arg(long, default_value_t = 448)]
        width: usize,
        #[arg(long, default_value_t = 9)]
        num_frames: usize,
        #[arg(long, default_value_t = 5)]
        steps: usize,
        #[arg(long, default_value_t = 5.0)]
        guidance_scale: f32,
    },
    /// Denoise + VAE decode (end-to-end minus text encoder).
    Decode {
        #[arg(long, default_value_t = 256)]
        height: usize,
        #[arg(long, default_value_t = 448)]
        width: usize,
        #[arg(long, default_value_t = 9)]
        num_frames: usize,
        #[arg(long, default_value_t = 5)]
        steps: usize,
        #[arg(long, default_value_t = 5.0)]
        guidance_scale: f32,
    },
    /// Full denoise + VAE decode (end-to-end minus text encoder).
    FullInference {
        #[arg(long, default_value_t = 256)]
        height: usize,
        #[arg(long, default_value_t = 448)]
        width: usize,
        #[arg(long, default_value_t = 9)]
        num_frames: usize,
        #[arg(long, default_value_t = 3)]
        steps: usize,
    },
}

fn main() -> anyhow::Result<()> {
    init_tracing();
    let mut args = Args::parse();
    if let Ok(w) = std::env::var("WAN_WEIGHTS") {
        if !w.is_empty() {
            args.weights = PathBuf::from(w);
        }
    }

    if !args.weights.join("model_index.json").exists() {
        anyhow::bail!("missing model_index.json in {}", args.weights.display());
    }

    let plan = WanDevicePlan::for_wan(
        args.cpu,
        if args.transformer_f32 {
            Some(false)
        } else {
            Some(true)
        },
        &Default::default(),
    )?;

    match args.scenario {
        Scenario::SyntheticHotOp { seq_len, iters } => run_synthetic(&plan.compute, seq_len, iters),
        Scenario::ModelLoad => run_model_load(&args.weights, &plan),
        Scenario::Prefill {
            height,
            width,
            num_frames,
        } => run_prefill(&args.weights, &plan, height, width, num_frames),
        Scenario::VaeDecodeOnly {
            height,
            width,
            num_frames,
            vae_tiling,
        } => run_vae_decode_only(&args.weights, &plan, height, width, num_frames, vae_tiling),
        Scenario::DenoiseOnly {
            height,
            width,
            num_frames,
            steps,
            guidance_scale,
        } => run_denoise_only(
            &args.weights,
            &plan,
            height,
            width,
            num_frames,
            steps,
            guidance_scale,
        ),
        Scenario::Decode {
            height,
            width,
            num_frames,
            steps,
            guidance_scale,
        } => run_decode(
            &args.weights,
            &plan,
            height,
            width,
            num_frames,
            steps,
            guidance_scale,
        ),
        Scenario::FullInference {
            height,
            width,
            num_frames,
            steps,
        } => run_full(&args.weights, &plan, height, width, num_frames, steps),
    }
}

fn run_synthetic(device: &Device, seq_len: usize, iters: usize) -> anyhow::Result<()> {
    vram::log_vram("synthetic_start");
    let t0 = Instant::now();
    let heads = 12usize;
    let head_dim = 128usize;
    let b = 1usize;

    for i in 0..iters {
        candle_video::profile_zone!("synthetic_hot_op", iter = i);
        let q = Tensor::randn(0f32, 1f32, (b, seq_len, heads, head_dim), device)?;
        let k = q.clone();
        let v = q.clone();
        let q_m = q.transpose(1, 2)?.contiguous()?.to_dtype(DType::F16)?;
        let k_m = k.transpose(1, 2)?.contiguous()?.to_dtype(DType::F16)?;
        let v_m = v.transpose(1, 2)?.contiguous()?.to_dtype(DType::F16)?;
        let k_t = k_m.transpose(candle_core::D::Minus1, candle_core::D::Minus2)?;
        let att = q_m.matmul(&k_t.contiguous()?)?;
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;
        let _ = att.matmul(&v_m.contiguous()?)?;
    }

    vram::log_vram("synthetic_end");
    println!(
        "synthetic_hot_op: seq_len={seq_len} iters={iters} elapsed={:.3}s",
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn run_model_load(weights: &Path, plan: &WanDevicePlan) -> anyhow::Result<()> {
    vram::log_vram("model_load_start");
    let t0 = Instant::now();
    candle_video::profile_zone!("wan_model_load");
    let _stack = WanDenoiseStack::load_with_vae_device(
        weights,
        &plan.compute,
        &plan.vae,
        &plan.compute,
        plan.transformer_dtype,
        plan.vae_dtype,
        false,
        false,
    )?;
    vram::log_vram("model_load_end");
    println!("model_load: elapsed={:.3}s", t0.elapsed().as_secs_f64());
    Ok(())
}

fn load_prompt_fixture(device: &Device) -> anyhow::Result<Tensor> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wan/step0_reference.json");
    let raw = std::fs::read_to_string(&path)?;
    let fixture: Value = serde_json::from_str(&raw)?;
    let p_shape: Vec<usize> = fixture["prompt_embeds_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    load_f32_tensor(
        fixture["prompt_embeds"].as_array().unwrap(),
        &p_shape,
        device,
    )
}

fn latent_tensor(
    stack: &WanDenoiseStack,
    height: usize,
    width: usize,
    num_frames: usize,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let shape = latent_shape_from_config(stack.config(), 1, height, width, num_frames);
    Ok(Tensor::randn(0f32, 1f32, shape.as_slice(), device)?)
}

fn load_f32_tensor(data: &[Value], shape: &[usize], device: &Device) -> anyhow::Result<Tensor> {
    let v: Vec<f32> = data.iter().map(|x| x.as_f64().unwrap() as f32).collect();
    Ok(Tensor::from_vec(v, shape, device)?)
}

fn run_prefill(
    weights: &Path,
    plan: &WanDevicePlan,
    height: usize,
    width: usize,
    num_frames: usize,
) -> anyhow::Result<()> {
    let tokens = wan_patch_token_count(height, width, num_frames);
    ensure_wan_cuda_requirements(&plan.compute, tokens)?;

    vram::log_vram("prefill_start");
    let t0 = Instant::now();
    let mut stack = WanDenoiseStack::load_with_vae_device(
        weights,
        &plan.compute,
        &plan.vae,
        &plan.compute,
        plan.transformer_dtype,
        plan.vae_dtype,
        false,
        false,
    )?;

    let latents = latent_tensor(&stack, height, width, num_frames, &plan.compute)?;
    let prompt_embeds = load_prompt_fixture(&plan.compute)?;
    let timestep = Tensor::full(999_i64, 1, &plan.compute)?;
    candle_video::profile_zone!("wan_prefill_forward");
    let _ = stack.transformer()?.forward(
        &latents.to_dtype(plan.transformer_dtype)?,
        &timestep,
        &prompt_embeds.to_dtype(plan.transformer_dtype)?,
    )?;

    vram::log_vram("prefill_end");
    println!(
        "prefill: tokens={tokens} elapsed={:.3}s",
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn run_vae_decode_only(
    weights: &Path,
    plan: &WanDevicePlan,
    height: usize,
    width: usize,
    num_frames: usize,
    vae_tiling: bool,
) -> anyhow::Result<()> {
    vram::log_vram("vae_decode_start");
    let t0 = Instant::now();
    let mut stack = WanDenoiseStack::load_with_vae_device(
        weights,
        &plan.compute,
        &plan.compute,
        &plan.compute,
        plan.transformer_dtype,
        plan.vae_dtype,
        true,
        vae_tiling,
    )?;
    let latents = latent_tensor(&stack, height, width, num_frames, &plan.compute)?
        .to_dtype(plan.vae_dtype)?;
    let vae = stack
        .vae_mut()
        .ok_or_else(|| anyhow::anyhow!("vae not loaded"))?;
    let _frames = vae.decode(&latents)?;
    vram::log_vram("vae_decode_end");
    println!(
        "vae_decode_only: tiling={vae_tiling} elapsed={:.3}s",
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn run_denoise_only(
    weights: &Path,
    plan: &WanDevicePlan,
    height: usize,
    width: usize,
    num_frames: usize,
    steps: usize,
    guidance_scale: f32,
) -> anyhow::Result<()> {
    let tokens = wan_patch_token_count(height, width, num_frames);
    ensure_wan_cuda_requirements(&plan.compute, tokens)?;

    vram::log_vram("denoise_start");
    let t0 = Instant::now();
    let mut stack = WanDenoiseStack::load_with_vae_device(
        weights,
        &plan.compute,
        &plan.vae,
        &plan.compute,
        plan.transformer_dtype,
        plan.vae_dtype,
        false,
        false,
    )?;

    let latents = latent_tensor(&stack, height, width, num_frames, &plan.compute)?;
    let prompt_embeds = load_prompt_fixture(&plan.compute)?;
    let neg = prompt_embeds.zeros_like()?;

    let _ = stack.denoise(
        height,
        width,
        num_frames,
        steps,
        guidance_scale,
        Some(42),
        Some(latents),
        prompt_embeds.to_dtype(plan.transformer_dtype)?,
        Some(neg.to_dtype(plan.transformer_dtype)?),
    )?;

    vram::log_vram("denoise_end");
    println!(
        "denoise_only: steps={steps} tokens={tokens} elapsed={:.3}s",
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn run_decode(
    weights: &Path,
    plan: &WanDevicePlan,
    height: usize,
    width: usize,
    num_frames: usize,
    steps: usize,
    guidance_scale: f32,
) -> anyhow::Result<()> {
    let tokens = wan_patch_token_count(height, width, num_frames);
    ensure_wan_cuda_requirements(&plan.compute, tokens)?;

    vram::log_vram("decode_start");
    let t0 = Instant::now();
    let mut stack = WanDenoiseStack::load_with_vae_device(
        weights,
        &plan.compute,
        &plan.vae,
        &plan.compute,
        plan.transformer_dtype,
        plan.vae_dtype,
        false,
        false,
    )?;

    let latents = latent_tensor(&stack, height, width, num_frames, &plan.compute)?;
    let prompt_embeds = load_prompt_fixture(&plan.compute)?;
    let neg = prompt_embeds.zeros_like()?;

    let _ = stack.denoise_and_decode(
        height,
        width,
        num_frames,
        steps,
        guidance_scale,
        Some(42),
        Some(latents),
        prompt_embeds.to_dtype(plan.transformer_dtype)?,
        Some(neg.to_dtype(plan.transformer_dtype)?),
    )?;

    vram::log_vram("decode_end");
    println!(
        "decode: steps={steps} tokens={tokens} elapsed={:.3}s",
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn run_full(
    weights: &Path,
    plan: &WanDevicePlan,
    height: usize,
    width: usize,
    num_frames: usize,
    steps: usize,
) -> anyhow::Result<()> {
    run_decode(weights, plan, height, width, num_frames, steps, 5.0)
}
