//! Minimal Conv3d tests for debugging parity issues.

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use candle_video::ops::conv3d::{Conv3d, Conv3dConfig, PaddingMode};
    use std::collections::HashMap;
    use std::path::Path;

    const TOLERANCE: f32 = 1e-4;

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> anyhow::Result<f32> {
        let a_f32 = a.to_dtype(DType::F32)?;
        let b_f32 = b.to_dtype(DType::F32)?;
        let diff = a_f32.sub(&b_f32)?.abs()?.max_all()?;
        Ok(diff.to_vec0::<f32>()?)
    }

    #[test]
    fn test_simple_conv3d() -> anyhow::Result<()> {
        let ref_path = Path::new("gen_conv3d_minimal.safetensors");
        if !ref_path.exists() {
            println!("Skipping: gen_conv3d_minimal.safetensors not found");
            println!("Run: python scripts/debug_conv3d_minimal.py");
            return Ok(());
        }

        let device = Device::Cpu;
        let tensors = candle_core::safetensors::load(ref_path, &device)?;

        println!("\n=== Simple 2x2x2 Conv3d Test ===");
        
        let input = tensors.get("simple_input").unwrap();
        let weight = tensors.get("simple_weight").unwrap();
        let bias = tensors.get("simple_bias").unwrap();
        let ref_output = tensors.get("simple_output").unwrap();

        println!("Input shape: {:?}", input.dims());
        println!("Weight shape: {:?}", weight.dims());
        println!("Ref output shape: {:?}", ref_output.dims());

        // Create Conv3d
        let config = Conv3dConfig::new((2, 2, 2))
            .with_padding((0, 0, 0));

        let mut weights_map = HashMap::new();
        weights_map.insert("weight".to_string(), weight.clone());
        weights_map.insert("bias".to_string(), bias.clone());

        let vb = VarBuilder::from_tensors(weights_map, DType::F32, &device);
        let conv = Conv3d::new(1, 1, config, vb)?;

        let rust_output = conv.forward(input)?;
        
        println!("Rust output shape: {:?}", rust_output.dims());
        println!("Rust output: {:?}", rust_output.flatten_all()?.to_vec1::<f32>()?);
        println!("Python output: {:?}", ref_output.flatten_all()?.to_vec1::<f32>()?);

        let diff = max_abs_diff(&rust_output, ref_output)?;
        println!("Max diff: {:.6e}", diff);

        assert!(diff < TOLERANCE, "Simple conv3d failed: diff={}", diff);
        println!("✅ PASS");

        Ok(())
    }

    #[test]
    fn test_padded_conv3d() -> anyhow::Result<()> {
        let ref_path = Path::new("gen_conv3d_minimal.safetensors");
        if !ref_path.exists() {
            println!("Skipping: gen_conv3d_minimal.safetensors not found");
            return Ok(());
        }

        let device = Device::Cpu;
        let tensors = candle_core::safetensors::load(ref_path, &device)?;

        println!("\n=== Padded 3x3x3 Conv3d Test ===");
        
        let input = tensors.get("padded_input").unwrap();
        let weight = tensors.get("padded_weight").unwrap();
        let bias = tensors.get("padded_bias").unwrap();
        let ref_output = tensors.get("padded_output").unwrap();

        println!("Input shape: {:?}", input.dims());
        println!("Weight shape: {:?}", weight.dims());
        println!("Ref output shape: {:?}", ref_output.dims());

        // Test manual padding to debug
        println!("\n--- Debug: Manual padding test ---");
        let padded = input.pad_with_zeros(2, 1, 1)?; // temporal
        let padded = padded.pad_with_zeros(3, 1, 1)?; // height
        let padded = padded.pad_with_zeros(4, 1, 1)?; // width
        println!("After manual padding: {:?}", padded.dims());

        // Create Conv3d with padding=1
        let config = Conv3dConfig::new((3, 3, 3))
            .with_padding((1, 1, 1));

        let mut weights_map = HashMap::new();
        weights_map.insert("weight".to_string(), weight.clone());
        weights_map.insert("bias".to_string(), bias.clone());

        let vb = VarBuilder::from_tensors(weights_map, DType::F32, &device);
        let conv = Conv3d::new(2, 3, config, vb)?;

        let rust_output = conv.forward(input)?;
        
        println!("Rust output shape: {:?}", rust_output.dims());
        
        let rust_flat = rust_output.flatten_all()?.to_vec1::<f32>()?;
        let ref_flat = ref_output.flatten_all()?.to_vec1::<f32>()?;
        
        println!("Rust first 8: {:?}", &rust_flat[..8]);
        println!("Python first 8: {:?}", &ref_flat[..8]);

        let diff = max_abs_diff(&rust_output, ref_output)?;
        println!("Max diff: {:.6e}", diff);

        assert!(diff < TOLERANCE, "Padded conv3d failed: diff={}", diff);
        println!("✅ PASS");

        Ok(())
    }

    #[test]
    fn test_pointwise_conv3d() -> anyhow::Result<()> {
        let ref_path = Path::new("gen_conv3d_minimal.safetensors");
        if !ref_path.exists() {
            println!("Skipping: gen_conv3d_minimal.safetensors not found");
            return Ok(());
        }

        let device = Device::Cpu;
        let tensors = candle_core::safetensors::load(ref_path, &device)?;

        println!("\n=== Pointwise 1x1x1 Conv3d Test ===");
        
        let input = tensors.get("pointwise_input").unwrap();
        let weight = tensors.get("pointwise_weight").unwrap();
        let bias = tensors.get("pointwise_bias").unwrap();
        let ref_output = tensors.get("pointwise_output").unwrap();

        println!("Input shape: {:?}", input.dims());
        println!("Weight shape: {:?}", weight.dims());
        println!("Ref output shape: {:?}", ref_output.dims());

        // Create Conv3d pointwise
        let config = Conv3dConfig::new((1, 1, 1));

        let mut weights_map = HashMap::new();
        weights_map.insert("weight".to_string(), weight.clone());
        weights_map.insert("bias".to_string(), bias.clone());

        let vb = VarBuilder::from_tensors(weights_map, DType::F32, &device);
        let conv = Conv3d::new(4, 8, config, vb)?;

        let rust_output = conv.forward(input)?;
        
        println!("Rust output shape: {:?}", rust_output.dims());
        
        let rust_flat = rust_output.flatten_all()?.to_vec1::<f32>()?;
        let ref_flat = ref_output.flatten_all()?.to_vec1::<f32>()?;
        
        println!("Rust first 8: {:?}", &rust_flat[..8]);
        println!("Python first 8: {:?}", &ref_flat[..8]);

        let diff = max_abs_diff(&rust_output, ref_output)?;
        println!("Max diff: {:.6e}", diff);

        assert!(diff < TOLERANCE, "Pointwise conv3d failed: diff={}", diff);
        println!("✅ PASS");

        Ok(())
    }

    /// Test im2col directly
    #[test]
    fn test_im2col_manual() -> anyhow::Result<()> {
        use candle_video::ops::conv3d::cpu::{im2col_3d, Im2ColConfig};
        
        let device = Device::Cpu;
        
        // Simple input: (1, 1, 2, 3, 3) with sequential values
        let input_data: Vec<f32> = (0..18).map(|x| x as f32).collect();
        let input = Tensor::from_vec(input_data, (1, 1, 2, 3, 3), &device)?;
        
        println!("\n=== Im2col Manual Test ===");
        println!("Input shape: {:?}", input.dims());
        // Config for 2x2x2 kernel, no padding, stride 1
        let config = Im2ColConfig::new((2, 2, 2), (1, 1, 1), (1, 1, 1));
        
        // Output dims: t_out=1, h_out=2, w_out=2
        let cols = im2col_3d(&input, &config, 1, 2, 2)?;
        
        println!("Im2col output shape: {:?}", cols.dims());
        // Expected: (1, 1*2*2, 1*2*2*2) = (1, 4, 8)
        
        let cols_vec = cols.to_vec3::<f32>()?;
        println!("Im2col output:\n{:?}", cols_vec);
        
        // For position (0,0,0) in output, we should extract:
        // x[0, 0, 0:2, 0:2, 0:2] = [0,1,3,4,9,10,12,13]
        let expected_first_patch = vec![0.0, 1.0, 3.0, 4.0, 9.0, 10.0, 12.0, 13.0];
        println!("Expected first patch: {:?}", expected_first_patch);
        println!("Actual first patch: {:?}", &cols_vec[0][0]);
        
        // Verify
        for (i, (e, a)) in expected_first_patch.iter().zip(cols_vec[0][0].iter()).enumerate() {
            assert!((e - a).abs() < 1e-6, "Mismatch at {}: expected {}, got {}", i, e, a);
        }
        
        println!("✅ Im2col PASS");
        
        Ok(())
    }
}
