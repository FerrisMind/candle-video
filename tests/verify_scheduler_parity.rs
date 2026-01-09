#[cfg(test)]
mod tests {
    use candle_core::Device;
    use candle_video::models::ltx_video::scheduler::{
        FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerConfig,
    };
    use std::path::Path;

    #[test]
    fn test_scheduler_parity() -> anyhow::Result<()> {
        let path = Path::new("gen_scheduler_ref.safetensors");
        if !path.exists() {
            println!("Skipping test_scheduler_parity: gen_scheduler_ref.safetensors not found");
            return Ok(());
        }

        let device = Device::Cpu;
        println!("Running on device: {:?}", device);

        let tensors = candle_core::safetensors::load(path, &device)?;

        // Get reference data
        let ref_timesteps = tensors.get("timesteps").unwrap();
        let ref_sigmas = tensors.get("sigmas").unwrap();
        let ref_mu = tensors.get("mu").unwrap().to_vec1::<f32>()?[0];

        println!("Reference mu: {}", ref_mu);
        println!("Reference timesteps shape: {:?}", ref_timesteps.shape());
        println!("Reference sigmas shape: {:?}", ref_sigmas.shape());

        // Initialize Rust scheduler with same config
        let config = FlowMatchEulerDiscreteSchedulerConfig::default();
        let mut scheduler = FlowMatchEulerDiscreteScheduler::new(config)?;

        // Set timesteps with correct API: set_timesteps(num_steps, device, sigmas, mu, timesteps)
        scheduler.set_timesteps(Some(40), &device, None, Some(ref_mu), None)?;

        let rust_timesteps = scheduler.timesteps().to_vec1::<f32>()?;
        println!("\nRust timesteps (first 10): {:?}", &rust_timesteps[..10]);
        println!(
            "Rust timesteps (last 10): {:?}",
            &rust_timesteps[rust_timesteps.len() - 10..]
        );

        let rust_sigmas = scheduler.sigmas().to_vec1::<f32>()?;
        println!("\nRust sigmas (first 10): {:?}", &rust_sigmas[..10]);
        println!(
            "Rust sigmas (last 10): {:?}",
            &rust_sigmas[rust_sigmas.len() - 10..]
        );

        // Compare timesteps
        let ref_ts = ref_timesteps.to_vec1::<f32>()?;
        println!("\nRef timesteps (first 10): {:?}", &ref_ts[..10]);

        let mut max_ts_diff: f32 = 0.0;
        for (i, (&rust, &py)) in rust_timesteps.iter().zip(ref_ts.iter()).enumerate() {
            let diff = (rust - py).abs();
            if diff > max_ts_diff {
                max_ts_diff = diff;
            }
            if diff > 1.0 {
                println!(
                    "Timestep {} mismatch: Rust={}, Python={}, diff={}",
                    i, rust, py, diff
                );
            }
        }
        println!("\nMax timestep difference: {}", max_ts_diff);

        // Compare sigmas
        let ref_sig = ref_sigmas.to_vec1::<f32>()?;
        let mut max_sig_diff: f32 = 0.0;
        for (i, (&rust, &py)) in rust_sigmas.iter().zip(ref_sig.iter()).enumerate() {
            let diff = (rust - py).abs();
            if diff > max_sig_diff {
                max_sig_diff = diff;
            }
            if diff > 0.01 {
                println!(
                    "Sigma {} mismatch: Rust={}, Python={}, diff={}",
                    i, rust, py, diff
                );
            }
        }
        println!("Max sigma difference: {}", max_sig_diff);

        // Test step() output
        let sample = tensors.get("sample").unwrap();
        let model_output = tensors.get("model_output").unwrap();
        let ref_step_output = tensors.get("step_output").unwrap();
        let step_t = tensors.get("step_t").unwrap().to_vec1::<f32>()?[0];

        let rust_step_result = scheduler.step(&model_output, step_t, &sample, None)?;
        let rust_step_output = rust_step_result.prev_sample;

        let step_diff = (rust_step_output.sub(ref_step_output)?).abs()?.max_all()?;
        let step_diff_val = step_diff.to_vec0::<f32>()?;
        println!("\nStep output max difference: {}", step_diff_val);

        // Assertions
        assert!(
            max_ts_diff < 2.0,
            "Timestep difference too large: {}",
            max_ts_diff
        );
        assert!(
            max_sig_diff < 0.01,
            "Sigma difference too large: {}",
            max_sig_diff
        );
        assert!(
            step_diff_val < 0.01,
            "Step output difference too large: {}",
            step_diff_val
        );

        println!("\n✓ Scheduler parity test passed!");

        Ok(())
    }
}
