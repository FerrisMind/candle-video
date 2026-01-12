use candle_core::{Device, Result};
use candle_video::models::ltx_video::scheduler::{
    FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerConfig,
};

fn main() -> Result<()> {
    let device = Device::Cpu;

    // 1. Load reference
    let ref_path = "reference_output/scheduler_step.safetensors";
    let ref_tensors = candle_core::safetensors::load(ref_path, &device)?;

    let sample = ref_tensors.get("sample").unwrap();
    let model_output = ref_tensors.get("model_output").unwrap();
    let timestep = ref_tensors.get("timestep").unwrap().to_scalar::<f32>()?;
    let py_prev_sample = ref_tensors.get("prev_sample").unwrap();

    println!("Timestep: {}", timestep);

    // 2. Setup scheduler
    let config = FlowMatchEulerDiscreteSchedulerConfig {
        num_train_timesteps: 1000,
        shift: 1.0,
        ..Default::default()
    };
    let mut scheduler = FlowMatchEulerDiscreteScheduler::new(config)?;

    // Set timesteps (required for step to work)
    scheduler.set_timesteps(Some(30), &device, None, None, None)?;

    // 3. Run step (no RNG needed, last arg is per_token_timesteps)
    let rust_prev_sample = scheduler
        .step(model_output, timestep, sample, None)?
        .prev_sample;

    // 4. Compare
    let diff = (&rust_prev_sample - py_prev_sample)?.abs()?;
    let max_diff = diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    let avg_diff = diff.flatten_all()?.mean(0)?.to_scalar::<f32>()?;

    println!("Scheduler Step Max Diff: {}", max_diff);
    println!("Scheduler Step Avg Diff: {}", avg_diff);

    if max_diff < 1e-4 {
        println!("SUCCESS: Scheduler step matches.");
    } else {
        println!("FAILURE: Scheduler step mismatch.");
    }

    Ok(())
}
