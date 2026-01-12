//! Exact parity tests for Conv3d against Python reference.
//!
//! These tests use reference data generated with the exact same padding semantics
//! as the Rust implementation.

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

    fn run_test(
        name: &str,
        tensors: &HashMap<String, Tensor>,
        kernel: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
        groups: usize,
        is_causal: bool,
        padding_mode: PaddingMode,
        device: &Device,
    ) -> anyhow::Result<(bool, String)> {
        let input_key = format!("{}_input", name);
        let weight_key = format!("{}_weight", name);
        let bias_key = format!("{}_bias", name);
        let output_key = format!("{}_output", name);

        let input = tensors.get(&input_key)
            .ok_or_else(|| anyhow::anyhow!("Missing {}", input_key))?;
        let weight = tensors.get(&weight_key)
            .ok_or_else(|| anyhow::anyhow!("Missing {}", weight_key))?;
        let bias = tensors.get(&bias_key)
            .ok_or_else(|| anyhow::anyhow!("Missing {}", bias_key))?;
        let ref_output = tensors.get(&output_key)
            .ok_or_else(|| anyhow::anyhow!("Missing {}", output_key))?;

        // Get dimensions
        let weight_dims = weight.dims();
        let out_channels = weight_dims[0];
        let in_channels_per_group = weight_dims[1];
        let in_channels = in_channels_per_group * groups;

        // Build config
        let config = Conv3dConfig::new(kernel)
            .with_stride(stride)
            .with_padding(padding)
            .with_dilation(dilation)
            .with_groups(groups)
            .with_causal(is_causal)
            .with_padding_mode(padding_mode);

        // Create VarBuilder with weights
        let weight_dev = weight.to_device(device)?;
        let bias_dev = bias.to_device(device)?;
        
        let mut weights_map = HashMap::new();
        weights_map.insert("weight".to_string(), weight_dev);
        weights_map.insert("bias".to_string(), bias_dev);

        let vb = VarBuilder::from_tensors(weights_map, DType::F32, device);
        let conv = Conv3d::new(in_channels, out_channels, config, vb)?;

        // Run forward
        let input_dev = input.to_device(device)?;
        let rust_output = conv.forward(&input_dev)?;

        // Compare shapes
        let rust_shape = rust_output.dims();
        let ref_shape = ref_output.dims();
        if rust_shape != ref_shape {
            return Ok((
                false,
                format!(
                    "[FAIL] {} - Shape mismatch: Rust {:?} vs Python {:?}",
                    name, rust_shape, ref_shape
                ),
            ));
        }

        // Compare values
        let ref_output_dev = ref_output.to_device(device)?;
        let diff = max_abs_diff(&rust_output, &ref_output_dev)?;

        if diff > TOLERANCE {
            // Get more details
            let rust_flat = rust_output.flatten_all()?.to_vec1::<f32>()?;
            let ref_flat = ref_output_dev.flatten_all()?.to_vec1::<f32>()?;
            
            let mut max_idx = 0;
            let mut max_diff_val = 0.0f32;
            for (i, (r, p)) in rust_flat.iter().zip(ref_flat.iter()).enumerate() {
                let d = (r - p).abs();
                if d > max_diff_val {
                    max_diff_val = d;
                    max_idx = i;
                }
            }
            
            return Ok((
                false,
                format!(
                    "[FAIL] {} - Max diff {:.6} at idx {} (rust={:.6}, python={:.6})",
                    name, diff, max_idx, rust_flat[max_idx], ref_flat[max_idx]
                ),
            ));
        }

        Ok((true, format!("[PASS] {} - max diff: {:.6e}", name, diff)))
    }

    #[test]
    fn test_conv3d_exact_parity() -> anyhow::Result<()> {
        let ref_path = Path::new("gen_conv3d_exact.safetensors");
        if !ref_path.exists() {
            println!("Skipping test: gen_conv3d_exact.safetensors not found");
            println!("Run: python scripts/verify_conv3d_exact.py to generate");
            return Ok(());
        }

        let device = Device::Cpu;
        let tensors = candle_core::safetensors::load(ref_path, &device)?;
        
        println!("\nRunning exact Conv3d parity tests");
        println!("Tolerance: {}", TOLERANCE);
        println!();

        let mut passed = 0;
        let mut failed = 0;

        // Test 1: Basic 3x3x3 non-causal
        let (success, msg) = run_test(
            "basic_3x3x3",
            &tensors,
            (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1),
            1, false, PaddingMode::Zeros,
            &device,
        )?;
        println!("  {}", msg);
        if success { passed += 1; } else { failed += 1; }

        // Test 2: Causal 3x3x3
        let (success, msg) = run_test(
            "causal_3x3x3",
            &tensors,
            (3, 3, 3), (1, 1, 1), (0, 1, 1), (1, 1, 1),
            1, true, PaddingMode::Replicate,
            &device,
        )?;
        println!("  {}", msg);
        if success { passed += 1; } else { failed += 1; }

        // Test 3: Pointwise
        let (success, msg) = run_test(
            "pointwise",
            &tensors,
            (1, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1),
            1, false, PaddingMode::Zeros,
            &device,
        )?;
        println!("  {}", msg);
        if success { passed += 1; } else { failed += 1; }

        // Test 4: Temporal-only
        let (success, msg) = run_test(
            "temporal",
            &tensors,
            (3, 1, 1), (1, 1, 1), (1, 0, 0), (1, 1, 1),
            1, false, PaddingMode::Zeros,
            &device,
        )?;
        println!("  {}", msg);
        if success { passed += 1; } else { failed += 1; }

        // Test 5: Spatial-only
        let (success, msg) = run_test(
            "spatial",
            &tensors,
            (1, 3, 3), (1, 1, 1), (0, 1, 1), (1, 1, 1),
            1, false, PaddingMode::Zeros,
            &device,
        )?;
        println!("  {}", msg);
        if success { passed += 1; } else { failed += 1; }

        // Test 6: Grouped
        let (success, msg) = run_test(
            "grouped",
            &tensors,
            (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1),
            2, false, PaddingMode::Zeros,
            &device,
        )?;
        println!("  {}", msg);
        if success { passed += 1; } else { failed += 1; }

        // Test 7: Strided
        let (success, msg) = run_test(
            "strided",
            &tensors,
            (3, 3, 3), (2, 2, 2), (1, 1, 1), (1, 1, 1),
            1, false, PaddingMode::Zeros,
            &device,
        )?;
        println!("  {}", msg);
        if success { passed += 1; } else { failed += 1; }

        // Test 8: Wan VAE style
        let (success, msg) = run_test(
            "wan_vae",
            &tensors,
            (3, 3, 3), (1, 1, 1), (0, 1, 1), (1, 1, 1),
            1, true, PaddingMode::Replicate,
            &device,
        )?;
        println!("  {}", msg);
        if success { passed += 1; } else { failed += 1; }

        // Test 9: Single frame causal
        let (success, msg) = run_test(
            "single_frame_causal",
            &tensors,
            (3, 3, 3), (1, 1, 1), (0, 1, 1), (1, 1, 1),
            1, true, PaddingMode::Replicate,
            &device,
        )?;
        println!("  {}", msg);
        if success { passed += 1; } else { failed += 1; }

        println!();
        println!("Results: {} passed, {} failed", passed, failed);
        
        assert_eq!(failed, 0, "Some parity tests failed");
        
        Ok(())
    }
}
