#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};
    use std::path::Path;

    /// Sinusoidal timestep embeddings matching Python diffusers
    /// Parameters: flip_sin_to_cos=True, downscale_freq_shift=0
    fn get_timestep_embedding(
        timesteps: &Tensor,
        embedding_dim: usize,
    ) -> candle_core::Result<Tensor> {
        let half_dim = embedding_dim / 2;
        let device = timesteps.device();

        // Python: exponent = -math.log(max_period) * torch.arange(0, half_dim) / (half_dim - downscale_freq_shift)
        // With downscale_freq_shift=0: exponent / half_dim
        let max_period = 10000f64;
        let downscale_freq_shift = 0.0;

        let exponent_coef = -(max_period.ln()) / (half_dim as f64 - downscale_freq_shift);
        let emb = (Tensor::arange(0u32, half_dim as u32, device)?
            .to_dtype(DType::F32)?
            .affine(exponent_coef, 0.0))?
        .exp()?;

        // timesteps: (B,) -> (B, 1) * emb -> (B, half_dim)
        let timesteps_f = timesteps.to_dtype(DType::F32)?.unsqueeze(1)?;
        let emb = timesteps_f.broadcast_mul(&emb.unsqueeze(0)?)?;

        // flip_sin_to_cos=True means [cos, sin] order
        let sin_emb = emb.sin()?;
        let cos_emb = emb.cos()?;
        Tensor::cat(&[&cos_emb, &sin_emb], 1)
    }

    #[test]
    fn test_timestep_embedding_parity() -> anyhow::Result<()> {
        let path = Path::new("gen_vae_parity.safetensors");
        if !path.exists() {
            println!("Skipping test: gen_vae_parity.safetensors not found");
            return Ok(());
        }

        let device = Device::Cpu;
        let tensors = candle_core::safetensors::load(path, &device)?;

        // Get embedding dimension from reference
        let embedding_dim = tensors
            .get("sinusoidal_embedding_dim")
            .unwrap()
            .to_vec1::<i64>()?[0] as usize;
        println!("Embedding dimension: {}", embedding_dim);

        // Test various timestep values
        let test_cases = vec![
            ("sinusoidal_t0.0_embedding", 0.0f32),
            ("sinusoidal_t0.05_embedding", 0.05f32),
            ("sinusoidal_t0.1_embedding", 0.1f32),
            ("sinusoidal_t0.5_embedding", 0.5f32),
            ("sinusoidal_t1.0_embedding", 1.0f32),
        ];

        for (ref_key, t_val) in test_cases {
            let ref_emb = tensors.get(ref_key).unwrap().to_dtype(DType::F32)?;

            let timesteps = Tensor::new(&[t_val], &device)?;
            let rust_emb = get_timestep_embedding(&timesteps, embedding_dim)?;

            let diff = (rust_emb.sub(&ref_emb)?).abs()?;
            let max_diff = diff.max_all()?.to_vec0::<f32>()?;
            let mse = diff.sqr()?.mean_all()?.to_vec0::<f32>()?;

            println!("t={}: max_diff={:.2e}, mse={:.2e}", t_val, max_diff, mse);

            assert!(
                max_diff < 1e-5,
                "Timestep embedding t={} max_diff {} exceeds threshold",
                t_val,
                max_diff
            );
        }

        println!("\n✓ Timestep embedding parity test passed!");
        Ok(())
    }
}
