use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_video::models::ltx_video::ltx_transformer::{
    LtxVideoTransformer3DModel, LtxVideoTransformer3DModelConfig,
};
use candle_video::models::ltx_video::scheduler::{
    FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerConfig,
};
use std::path::PathBuf;

fn main() -> Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32; // Using F32 for verification

    println!("Device: {:?}", device);

    // 1. Load Step 0 Reference Data
    println!("Loading reference data...");
    let ref_path = "reference_output/pipeline_step0.safetensors";
    let ref_tensors = candle_core::safetensors::load(ref_path, &Device::Cpu)?;

    // Helper to get and move
    let get_tensor = |name: &str| -> Result<Tensor> {
        ref_tensors
            .get(name)
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", name)))?
            .to_device(&device)?
            .to_dtype(dtype)
    };

    let prompt_embeds = get_tensor("prompt_embeds")?; // [2, 11, 4096] if CFG=3
    let prompt_attention_mask = get_tensor("prompt_attention_mask")?;
    let initial_latents = get_tensor("initial_latents")?; // [1, 128, 9, 4, 4] -> wait, python LTX is different per version
    // Check shapes
    println!("Initial Latents Shape: {:?}", initial_latents.dims());
    // Python output says: [1, 128, F_lat, H_lat, W_lat]?
    // Wait, capture script printed: Initial Latents: torch.Size([1, 128, 2, 4, 4]) (example)

    let timestep_tens = get_tensor("timestep")?;
    // Rust transformer expects timestep as [B]
    // Python capture: tensor([999.0]) -> scalar or [1]

    let py_transformer_output = get_tensor("transformer_output")?;
    let py_next_latents = get_tensor("next_latents")?;

    let py_sigmas = get_tensor("scheduler_sigmas")?;
    let py_timesteps = get_tensor("scheduler_timesteps")?;

    // 2. Load Model
    println!("Loading Model...");
    let model_path = "c:/candle-video/models/models--Lightricks--LTX-Video-0.9.5/transformer";
    let weights_path = PathBuf::from(model_path).join("diffusion_pytorch_model.safetensors");
    let config_path = PathBuf::from(model_path).join("config.json");

    let config: LtxVideoTransformer3DModelConfig = {
        let file = std::fs::File::open(config_path)?;
        serde_json::from_reader(file).map_err(candle_core::Error::wrap)?
    };

    // We compare in F32 to be sure
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)? };
    let transformer = LtxVideoTransformer3DModel::new(&config, vb)?;

    // 3. Run Transformer Forward
    println!("Running Transformer...");

    // Input preparation: CFG expansion
    // prompt_embeds already has CFG expansion from capture script IF scale > 1
    // But capture script with scale=1.0 will have no expansion.
    // latents need expansion ONLY if scale > 1

    let guidance_scale = 1.0;

    let latents_input = if guidance_scale > 1.0 {
        Tensor::cat(&[&initial_latents, &initial_latents], 0)?
    } else {
        initial_latents.clone()
    };

    // Timestep expansion
    let t_val = timestep_tens.to_vec1::<f32>()?[0];
    let timestep_input = if guidance_scale > 1.0 {
        Tensor::new(&[t_val, t_val], &device)?
    } else {
        Tensor::new(&[t_val], &device)?
    };

    // RoPE scale matches capture script
    // Python script:
    // height = 128, width = 128, num_frames = 9
    // rope = (161/9, 480/128, 704/128)
    // rope = (17.888..., 3.75, 5.5)
    let rope_scale = Some((161.0 / 9.0, 480.0 / 128.0, 704.0 / 128.0));

    // Latents are [B, S, C] packed
    // Dimensions for RoPE derived from config/pipeline logic
    let f = 2; // (9-1)/8 + 1
    let h = 4; // 128/32
    let w = 4; // 128/32

    // Using attention mask from python directly
    let attention_mask_arg = Some(&prompt_attention_mask);

    let transformer_output = transformer.forward(
        &latents_input,
        &prompt_embeds,
        &timestep_input,
        attention_mask_arg,
        f,
        h,
        w,
        rope_scale,
        None, // video_coords
        None, // skip_layer_mask
    )?;

    // 4. CFG
    println!("Performing CFG...");
    let guidance_scale = 1.0;

    let noise_pred = if guidance_scale > 1.0 {
        let chunks = transformer_output.chunk(2, 0)?;
        let noise_pred_uncond = &chunks[0];
        let noise_pred_text = &chunks[1];
        (noise_pred_uncond + (noise_pred_text - noise_pred_uncond)? * guidance_scale)?
    } else {
        transformer_output
    };

    // Compare Transformer Output
    let diff_tr = (&noise_pred - &py_transformer_output)?.abs()?;
    println!(
        "Transformer Output Diff Max: {}",
        diff_tr.flatten_all()?.max(0)?.to_scalar::<f32>()?
    );
    println!(
        "Transformer Output Diff Avg: {}",
        diff_tr.flatten_all()?.mean(0)?.to_scalar::<f32>()?
    );

    // 5. Scheduler Step
    println!("Running Scheduler...");
    let scheduler_config = FlowMatchEulerDiscreteSchedulerConfig::default();
    let mut scheduler = FlowMatchEulerDiscreteScheduler::new(scheduler_config)?;

    // Set Timesteps to match Python
    // Python script set num_inference_steps=2
    // We can manually inject sigmas/timesteps from Python capture to match EXACTLY
    // But let's try standard set logic first to see if it matches.
    // LTX: shift=1.0? No, checking standard config.
    // Python default is 1.0 shift.

    // Instead of full set_timesteps, let's just create scheduler state matching step 0
    // We need 'sigmas' array.

    // Verify sigmas match
    // Python sigmas: [sigma_0, sigma_1, ..., sigma_last, 0.0]
    // Let's load them
    println!("Python Sigmas: {:?}", py_sigmas.to_vec1::<f32>()?);

    // Force set sigmas to match python perfectly for this test
    // scheduler.set_timesteps ...
    // Using internal fields if possible or create a new method?
    // Or just use set_timesteps with explicit sigmas.
    let py_sigmas_vec = py_sigmas.to_vec1::<f32>()?;
    let py_timesteps_vec = py_timesteps.to_vec1::<f32>()?;

    // Note: Rust scheduler uses [f32]
    scheduler.set_timesteps(
        None,
        &device,
        Some(&py_sigmas_vec[..py_sigmas_vec.len() - 1]), // python ends with 0.0, we pass non-terminal usually?
        // Wait, set_timesteps(sigmas=...) expects full list?
        // Rust implementation appends terminal if not present?
        // Let's check impl.
        None,
        Some(&py_timesteps_vec),
    )?;

    // Check if generated sigmas match
    let _rust_sigmas = scheduler.sigmas().to_vec1::<f32>()?;
    // println!("Rust Sigmas: {:?}", rust_sigmas);

    // Step
    // Python: step(noise_pred, t, latents)
    // Rust: step(&noise_pred, timestep_f32, &latents, None)

    let scheduler_output = scheduler.step(&noise_pred, t_val, &initial_latents, None)?;

    let rust_next_latents = scheduler_output.prev_sample;

    // Compare Scheduler Output
    let diff_sch = (&rust_next_latents - &py_next_latents)?.abs()?;
    println!(
        "Scheduler Output Diff Max: {}",
        diff_sch.flatten_all()?.max(0)?.to_scalar::<f32>()?
    );
    println!(
        "Scheduler Output Diff Avg: {}",
        diff_sch.flatten_all()?.mean(0)?.to_scalar::<f32>()?
    );

    if diff_sch.flatten_all()?.max(0)?.to_scalar::<f32>()? < 0.01 {
        println!("SUCCESS: Pipeline Step matches.");
    } else {
        println!("FAILURE: Pipeline Step mismatch.");
    }

    Ok(())
}
