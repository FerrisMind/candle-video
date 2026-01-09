#[cfg(test)]
mod tests {
    use candle_core::Device;
    use candle_video::models::ltx_video::t2v_pipeline::LtxPipeline;
    use std::path::Path;

    #[test]
    fn test_latent_norm_parity() -> anyhow::Result<()> {
        let path = Path::new("gen_latent_norm_ref.safetensors");
        if !path.exists() {
            println!("Skipping test_latent_norm_parity: gen_latent_norm_ref.safetensors not found");
            return Ok(());
        }

        let device = Device::Cpu;
        println!("Running on device: {:?}", device);

        let tensors = candle_core::safetensors::load(path, &device)?;

        // Load inputs
        let latents = tensors.get("latents").unwrap();
        let latents_mean = tensors.get("latents_mean").unwrap();
        let latents_std = tensors.get("latents_std").unwrap();
        let scaling_factor = tensors.get("scaling_factor").unwrap().to_vec1::<f32>()?[0];

        // Load expected outputs
        let ref_normalized = tensors.get("normalized").unwrap();
        let ref_denormalized = tensors.get("denormalized").unwrap();

        println!("Latents shape: {:?}", latents.shape());
        println!("Scaling factor: {}", scaling_factor);

        // Test normalize
        let rust_normalized =
            LtxPipeline::normalize_latents(&latents, &latents_mean, &latents_std, scaling_factor)?;

        let norm_diff = (rust_normalized.sub(ref_normalized)?).abs()?.max_all()?;
        let norm_diff_val = norm_diff.to_vec0::<f32>()?;
        println!("Normalize max difference: {}", norm_diff_val);

        // Test denormalize
        let rust_denormalized = LtxPipeline::denormalize_latents(
            &rust_normalized,
            &latents_mean,
            &latents_std,
            scaling_factor,
        )?;

        let denorm_diff = (rust_denormalized.sub(ref_denormalized)?)
            .abs()?
            .max_all()?;
        let denorm_diff_val = denorm_diff.to_vec0::<f32>()?;
        println!("Denormalize max difference: {}", denorm_diff_val);

        // Test round-trip
        let roundtrip_diff = (rust_denormalized.sub(&latents)?).abs()?.max_all()?;
        let roundtrip_diff_val = roundtrip_diff.to_vec0::<f32>()?;
        println!("Round-trip max difference: {}", roundtrip_diff_val);

        // Assertions
        assert!(
            norm_diff_val < 1e-5,
            "Normalize difference too large: {}",
            norm_diff_val
        );
        assert!(
            denorm_diff_val < 1e-5,
            "Denormalize difference too large: {}",
            denorm_diff_val
        );
        assert!(
            roundtrip_diff_val < 1e-5,
            "Round-trip difference too large: {}",
            roundtrip_diff_val
        );

        println!("\n✓ Latent normalization parity test passed!");

        Ok(())
    }
}
