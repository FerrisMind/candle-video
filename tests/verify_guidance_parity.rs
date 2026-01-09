#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};
    use std::path::Path;

    #[test]
    fn test_guidance_parity() -> anyhow::Result<()> {
        let path = Path::new("gen_guidance_ref.safetensors");
        if !path.exists() {
            println!("Skipping test_guidance_parity: gen_guidance_ref.safetensors not found");
            return Ok(());
        }

        let device = Device::Cpu;
        println!("Running on device: {:?}", device);

        let tensors = candle_core::safetensors::load(path, &device)?;

        // Load inputs
        let noise_uncond = tensors.get("noise_pred_uncond").unwrap();
        let noise_text = tensors.get("noise_pred_text").unwrap();
        let noise_perturb = tensors.get("noise_pred_perturb").unwrap();
        let guidance_scale = tensors.get("guidance_scale").unwrap().to_vec1::<f32>()?[0] as f64;
        let stg_scale = tensors.get("stg_scale").unwrap().to_vec1::<f32>()?[0] as f64;

        // Load expected outputs
        let ref_combined_cfg = tensors.get("combined_cfg").unwrap();
        let ref_combined_final = tensors.get("combined_final").unwrap();

        println!("Guidance scale: {}", guidance_scale);
        println!("STG scale: {}", stg_scale);

        // Replicate Rust formula from t2v_pipeline.rs lines 859-876:
        // combined = uncond + cfg_scale * (text - uncond)
        // combined = combined + stg_scale * (text - perturb)

        let noise_text_f32 = noise_text.to_dtype(DType::F32)?;
        let uncond_f32 = noise_uncond.to_dtype(DType::F32)?;
        let perturb_f32 = noise_perturb.to_dtype(DType::F32)?;

        // Step 1: CFG
        let diff_cfg = noise_text_f32.broadcast_sub(&uncond_f32)?;
        let combined_cfg = uncond_f32.broadcast_add(&diff_cfg.affine(guidance_scale, 0.0)?)?;

        // Step 2: STG
        let diff_stg = noise_text_f32.broadcast_sub(&perturb_f32)?;
        let combined_final = combined_cfg.broadcast_add(&diff_stg.affine(stg_scale, 0.0)?)?;

        // Compare CFG result
        let diff_cfg_result = (combined_cfg.sub(ref_combined_cfg)?).abs()?.max_all()?;
        let diff_cfg_val = diff_cfg_result.to_vec0::<f32>()?;
        println!("CFG max difference: {}", diff_cfg_val);

        // Compare final result
        let diff_final_result = (combined_final.sub(ref_combined_final)?).abs()?.max_all()?;
        let diff_final_val = diff_final_result.to_vec0::<f32>()?;
        println!("Final max difference: {}", diff_final_val);

        // Assertions
        assert!(
            diff_cfg_val < 1e-5,
            "CFG difference too large: {}",
            diff_cfg_val
        );
        assert!(
            diff_final_val < 1e-5,
            "Final difference too large: {}",
            diff_final_val
        );

        println!("\n✓ Guidance parity test passed!");

        Ok(())
    }
}
