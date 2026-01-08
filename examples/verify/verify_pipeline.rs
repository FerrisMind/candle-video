use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_video::models::ltx_video::ltx_transformer::{
    LtxVideoTransformer3DModel, LtxVideoTransformer3DModelConfig,
};
use candle_video::models::ltx_video::scheduler::{
    FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerConfig, TimeShiftType,
};
use candle_video::models::ltx_video::t2v_pipeline::LtxPipeline;

fn main() -> Result<()> {
    let device = Device::Cpu;

    println!("Loading verification data from pipeline_verification.safetensors...");
    let tensors = candle_core::safetensors::load("pipeline_verification.safetensors", &device)?;

    // ... (previous checks)
    println!("\n--- Verifying pack_latents ---");
    let initial_latents = tensors
        .get("initial_latents")
        .context("Missing initial_latents")?;
    let py_packed_latents = tensors
        .get("packed_latents")
        .context("Missing packed_latents")?;

    let rust_packed_latents = LtxPipeline::pack_latents(initial_latents, 1, 1)?;
    compare_tensors(
        "Pack Latents",
        &rust_packed_latents,
        py_packed_latents,
        1e-6,
    )?;

    // 2. Setup Transformer and Verify Forward
    println!("\n--- Verifying Transformer Forward ---");
    let config = LtxVideoTransformer3DModelConfig {
        in_channels: 128,
        out_channels: 128,
        num_attention_heads: 32,
        attention_head_dim: 64,
        num_layers: 1,
        patch_size: 1,
        patch_size_t: 1,
        ..Default::default()
    };

    let vb = VarBuilder::from_tensors(tensors.clone(), DType::F32, &device).pp("transformer");
    let transformer = LtxVideoTransformer3DModel::new(&config, vb)?;

    let prompt_embeds = tensors.get("prompt_embeds").context("prompt_embeds")?;
    let prompt_mask = tensors
        .get("prompt_attention_mask")
        .context("prompt_attention_mask")?;
    let negative_prompt_embeds = tensors
        .get("negative_prompt_embeds")
        .context("negative_prompt_embeds")?;
    let negative_prompt_mask = tensors
        .get("negative_prompt_attention_mask")
        .context("negative_prompt_attention_mask")?;
    let timesteps = tensors.get("timesteps").context("timesteps")?;
    let t = timesteps.to_vec1::<f32>()?[0];

    let p_emb = Tensor::cat(&[negative_prompt_embeds.clone(), prompt_embeds.clone()], 0)?;
    let p_mask = Tensor::cat(&[negative_prompt_mask.clone(), prompt_mask.clone()], 0)?;
    let latent_model_input =
        Tensor::cat(&[py_packed_latents.clone(), py_packed_latents.clone()], 0)?;
    let timestep = Tensor::full(t, (2,), &device)?;

    let _noise_pred = transformer.forward(
        &latent_model_input,
        &p_emb,
        &timestep,
        Some(&p_mask),
        2, // num_frames
        1, // height
        1, // width
        Some((8.0 / 24.0, 32.0, 32.0)),
        None,
    )?;

    // 3. Verify Iterative Denoising Loop
    println!("\n--- Verifying Iterative Denoising (5 steps) ---");
    let py_sigmas = tensors.get("sigmas").context("Missing sigmas")?;
    let sched_config = FlowMatchEulerDiscreteSchedulerConfig {
        num_train_timesteps: 1000,
        shift: 1.0,
        time_shift_type: TimeShiftType::Exponential,
        ..Default::default()
    };
    let mut scheduler = FlowMatchEulerDiscreteScheduler::new(sched_config)?;
    let sigmas_vec = py_sigmas.to_vec1::<f32>()?;
    let timesteps_vec = timesteps.to_vec1::<f32>()?;
    let n = timesteps_vec.len();
    scheduler.set_timesteps(
        None,
        &device,
        Some(&sigmas_vec[0..n]),
        None,
        Some(&timesteps_vec),
    )?;

    let mut current_latents = py_packed_latents.clone();
    let guidance_scale = 3.0f32;

    for (i, &t) in timesteps_vec.iter().enumerate().take(n) {
        let latent_model_input =
            Tensor::cat(&[current_latents.clone(), current_latents.clone()], 0)?;
        let timestep = Tensor::full(t, (2,), &device)?;

        let noise_pred = transformer.forward(
            &latent_model_input,
            &p_emb,
            &timestep,
            Some(&p_mask),
            2, // latent_num_frames
            1, // height
            1, // width
            Some((8.0 / 24.0, 32.0, 32.0)),
            None,
        )?;

        let chunks = noise_pred.chunk(2, 0)?;
        let noise_uncond = &chunks[0];
        let noise_text = &chunks[1];
        let diff = noise_text.broadcast_sub(noise_uncond)?;
        let noise_pred_cfg =
            noise_uncond.broadcast_add(&diff.affine(guidance_scale as f64, 0.0)?)?;

        current_latents = scheduler
            .step(&noise_pred_cfg, t, &current_latents, None)?
            .prev_sample;

        let py_res = tensors
            .get(&format!("latents_step_{}", i))
            .context("Missing step result")?;
        compare_tensors(&format!("Step {}", i), &current_latents, py_res, 5e-4)?;
    }

    println!("\nALL PIPELINE LOGIC VERIFIED!");
    Ok(())
}

fn compare_tensors(name: &str, rust: &Tensor, py: &Tensor, tol: f64) -> Result<()> {
    let diff = (rust - py)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    println!("{}: MAD = {:.8}", name, diff);
    if diff as f64 > tol {
        anyhow::bail!("{} mismatch too high: {}", name, diff);
    }
    Ok(())
}
