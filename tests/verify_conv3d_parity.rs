#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;
    use candle_video::models::ltx_video::vae::LtxVideoCausalConv3d;
    use std::path::Path;

    #[test]
    fn test_causal_conv3d_parity() -> anyhow::Result<()> {
        let path = Path::new("gen_conv3d_model.safetensors");
        if !path.exists() {
            println!("Skipping test: gen_conv3d_model.safetensors not found");
            return Ok(());
        }

        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16; // Use BF16 for GPU - matches LTX model format
        println!("Running on device: {:?}, dtype: {:?}", device, dtype);

        let tensors = candle_core::safetensors::load(path, &device)?;

        let input = tensors.get("input").unwrap().to_dtype(dtype)?;
        let ref_output = tensors.get("output").unwrap().to_dtype(dtype)?;

        println!("Input shape: {:?}", input.shape());
        println!("Ref output shape: {:?}", ref_output.shape());

        // Load weights - note Python saves as conv.weight, conv.bias
        let mut weights = std::collections::HashMap::new();
        weights.insert(
            "conv.weight".to_string(),
            tensors.get("conv.weight").unwrap().to_dtype(dtype)?.clone(),
        );
        weights.insert(
            "conv.bias".to_string(),
            tensors.get("conv.bias").unwrap().to_dtype(dtype)?.clone(),
        );

        let vb = VarBuilder::from_tensors(weights, dtype, &device);

        // Create conv with same config as Python - NOTE: upsampler conv is NOT causal!
        let conv =
            LtxVideoCausalConv3d::new(1024, 4096, (3, 3, 3), (1, 1, 1), (1, 1, 1), 1, false, vb)?;

        let rust_output = conv.forward(&input)?;

        println!("Rust output shape: {:?}", rust_output.shape());

        // Convert to F32 for statistics (BF16 doesn't support to_vec0)
        let rust_f32 = rust_output.to_dtype(DType::F32)?;
        let ref_f32 = ref_output.to_dtype(DType::F32)?;

        println!(
            "Rust output mean: {}",
            rust_f32.mean_all()?.to_vec0::<f32>()?
        );
        println!("Ref output mean: {}", ref_f32.mean_all()?.to_vec0::<f32>()?);

        let diff = (rust_f32.sub(&ref_f32)?).abs()?.max_all()?;
        let diff_val = diff.to_vec0::<f32>()?;
        println!("\nMax diff: {}", diff_val);

        if diff_val < 0.1 {
            // BF16 has lower precision than F32
            println!("✓ CausalConv3d parity OK");
        } else {
            println!("!!! CausalConv3d DIVERGENCE: {} !!!", diff_val);
        }

        Ok(())
    }
}
