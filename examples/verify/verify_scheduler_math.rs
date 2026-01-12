use candle_core::{Device, Result};
use candle_video::models::ltx_video::scheduler::{
    FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerConfig,
};
// Remove unused import

fn main() -> Result<()> {
    let device = Device::Cpu;

    println!("Loading reference scheduler data...");
    let ref_path = "reference_output/scheduler_ref.safetensors";
    let ref_tensors = candle_core::safetensors::load(ref_path, &device)?;

    let py_timesteps = ref_tensors.get("timesteps").unwrap();
    let py_sigmas = ref_tensors.get("sigmas").unwrap();

    println!("Initializing Rust Scheduler...");
    let config = FlowMatchEulerDiscreteSchedulerConfig {
        num_train_timesteps: 1000,
        shift: 1.0,
        use_dynamic_shifting: false,
        ..Default::default()
    };

    let mut scheduler = FlowMatchEulerDiscreteScheduler::new(config)?;

    println!("Setting timesteps (50)...");

    scheduler.set_timesteps(Some(50), &device, None, None, None)?;

    let rust_timesteps = scheduler.timesteps();
    let rust_sigmas = scheduler.sigmas();

    println!("Comparing Timesteps...");
    let diff_ts = (rust_timesteps - py_timesteps)?.abs()?;
    let max_diff_ts = diff_ts.max(0)?.to_scalar::<f32>()?;
    println!("Timesteps Max Diff: {}", max_diff_ts);

    println!("Comparing Sigmas...");
    let diff_sig = (rust_sigmas - py_sigmas)?.abs()?;
    let max_diff_sig = diff_sig.max(0)?.to_scalar::<f32>()?;
    println!("Sigmas Max Diff: {}", max_diff_sig);

    if max_diff_ts < 1e-4 && max_diff_sig < 1e-4 {
        println!("SUCCESS: Scheduler logic matches.");
    } else {
        println!("FAILURE: Scheduler logic mismatch.");
    }

    Ok(())
}
