use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_video::models::ltx_video::scheduler::{
    FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerConfig, TimeShiftType,
};

fn main() -> Result<()> {
    let device = Device::Cpu;

    println!("Loading verification data from scheduler_verification.safetensors...");
    let tensors = candle_core::safetensors::load("scheduler_verification.safetensors", &device)?;

    let py_timesteps = tensors.get("timesteps").context("Missing timesteps")?;
    let py_sigmas = tensors.get("sigmas").context("Missing sigmas")?;
    let sample_in = tensors.get("sample_in").context("Missing sample_in")?;
    let model_output = tensors
        .get("model_output")
        .context("Missing model_output")?;
    let py_sample_out = tensors.get("sample_out").context("Missing sample_out")?;

    println!("Initializing Rust Scheduler...");
    let config = FlowMatchEulerDiscreteSchedulerConfig {
        num_train_timesteps: 1000,
        shift: 2.0,
        use_dynamic_shifting: false,
        base_shift: Some(0.5),
        max_shift: Some(1.15),
        base_image_seq_len: Some(256),
        max_image_seq_len: Some(4096),
        invert_sigmas: false,
        shift_terminal: None,
        use_karras_sigmas: false,
        use_exponential_sigmas: false,
        use_beta_sigmas: false,
        time_shift_type: TimeShiftType::Exponential,
        stochastic_sampling: false,
    };

    let mut scheduler = FlowMatchEulerDiscreteScheduler::new(config)?;

    println!("Setting timesteps (50 steps)...");
    scheduler.set_timesteps(Some(50), &device, None, None, None)?;

    let rust_timesteps = scheduler.timesteps();
    let rust_sigmas = scheduler.sigmas();

    // Compare Sigmas and Timesteps
    compare_tensors("Timesteps", rust_timesteps, py_timesteps, 1e-4)?;
    compare_tensors("Sigmas", rust_sigmas, py_sigmas, 1e-4)?;

    // Test Step
    println!("Testing scheduler.step()...");
    let ts_vec = py_timesteps.to_vec1::<f32>()?;
    let rust_step_out = scheduler.step(model_output, ts_vec[0], sample_in, None)?;

    compare_tensors(
        "Step Output",
        &rust_step_out.prev_sample,
        py_sample_out,
        1e-5,
    )?;

    println!("\nALL TESTS PASSED!");
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
