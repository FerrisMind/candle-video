//! Causal Conv3d parity test.

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, IndexOp, Tensor};
    use candle_nn::VarBuilder;
    use candle_video::ops::conv3d::{Conv3d, Conv3dConfig, PaddingMode};
    use candle_video::ops::conv3d::cpu::{im2col_3d, Im2ColConfig};
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
    fn test_causal_conv3d() -> anyhow::Result<()> {
        let ref_path = Path::new("gen_conv3d_causal.safetensors");
        if !ref_path.exists() {
            println!("Skipping: gen_conv3d_causal.safetensors not found");
            println!("Run: python scripts/debug_conv3d_causal.py");
            return Ok(());
        }

        let device = Device::Cpu;
        let tensors = candle_core::safetensors::load(ref_path, &device)?;

        println!("\n=== Causal Conv3d Parity Test ===");
        
        let input = tensors.get("causal_input").unwrap();
        let weight = tensors.get("causal_weight").unwrap();
        let bias = tensors.get("causal_bias").unwrap();
        let ref_padded = tensors.get("causal_input_padded").unwrap();
        let ref_full_padded = tensors.get("causal_full_padded").unwrap();
        let ref_output = tensors.get("causal_output").unwrap();
        let ref_im2col = tensors.get("causal_im2col_cols").unwrap();

        println!("Input shape: {:?}", input.dims());
        println!("Weight shape: {:?}", weight.dims());
        println!("Ref padded shape: {:?}", ref_padded.dims());
        println!("Ref full padded shape: {:?}", ref_full_padded.dims());
        println!("Ref output shape: {:?}", ref_output.dims());
        println!("Ref im2col shape: {:?}", ref_im2col.dims());

        // Step 1: Test causal temporal padding (replicate first frame)
        println!("\n--- Step 1: Test causal temporal padding ---");
        let (b, c, _t, h, w) = input.dims5()?;
        let kt = 3;
        let first_frame = input.narrow(2, 0, 1)?;
        let pad_left = first_frame.broadcast_as((b, c, kt - 1, h, w))?;
        let rust_padded = Tensor::cat(&[&pad_left, input], 2)?;
        println!("Rust padded shape: {:?}", rust_padded.dims());
        
        let pad_diff = max_abs_diff(&rust_padded, ref_padded)?;
        println!("Temporal padding diff: {:.6e}", pad_diff);
        assert!(pad_diff < 1e-6, "Temporal padding mismatch");
        println!("✅ Temporal padding PASS");

        // Step 2: Test full padding (temporal + spatial)
        println!("\n--- Step 2: Test full padding ---");
        // Add spatial padding (1, 1) on each side
        let rust_full_padded = rust_padded.pad_with_zeros(3, 1, 1)?; // height
        let rust_full_padded = rust_full_padded.pad_with_zeros(4, 1, 1)?; // width
        println!("Rust full padded shape: {:?}", rust_full_padded.dims());
        
        let full_pad_diff = max_abs_diff(&rust_full_padded, ref_full_padded)?;
        println!("Full padding diff: {:.6e}", full_pad_diff);
        assert!(full_pad_diff < 1e-6, "Full padding mismatch");
        println!("✅ Full padding PASS");

        // Step 3: Test im2col
        println!("\n--- Step 3: Test im2col ---");
        let config = Im2ColConfig::new((3, 3, 3), (1, 1, 1), (1, 1, 1));
        let rust_im2col = im2col_3d(&rust_full_padded, &config, 4, 4, 4)?;
        println!("Rust im2col shape: {:?}", rust_im2col.dims());
        
        let im2col_diff = max_abs_diff(&rust_im2col, ref_im2col)?;
        println!("Im2col diff: {:.6e}", im2col_diff);
        
        if im2col_diff > TOLERANCE {
            println!("⚠️ Im2col has differences");
            // Print first patch
            let rust_first = rust_im2col.i((0, 0, ..))?.to_vec1::<f32>()?;
            let ref_first = ref_im2col.i((0, 0, ..))?.to_vec1::<f32>()?;
            println!("Rust first patch (first 8): {:?}", &rust_first[..8.min(rust_first.len())]);
            println!("Ref first patch (first 8): {:?}", &ref_first[..8.min(ref_first.len())]);
        } else {
            println!("✅ Im2col PASS");
        }

        // Step 4: Test full Conv3d with causal mode
        println!("\n--- Step 4: Test full causal Conv3d ---");
        
        // First, let's manually do what Conv3d should do
        println!("Manual causal conv3d:");
        
        // 1. Apply causal temporal padding (replicate first frame kt-1 times)
        let (b, c, t, h, w) = input.dims5()?;
        let kt = 3;
        let first_frame = input.narrow(2, 0, 1)?;
        let pad_left = first_frame.broadcast_as((b, c, kt - 1, h, w))?;
        let manual_padded = Tensor::cat(&[&pad_left, input], 2)?;
        println!("  After temporal padding: {:?}", manual_padded.dims());
        
        // 2. Apply spatial padding (zeros, ph=1, pw=1)
        let manual_full_padded = manual_padded.pad_with_zeros(3, 1, 1)?;
        let manual_full_padded = manual_full_padded.pad_with_zeros(4, 1, 1)?;
        println!("  After spatial padding: {:?}", manual_full_padded.dims());
        
        // 3. Im2col
        let config_im2col = Im2ColConfig::new((3, 3, 3), (1, 1, 1), (1, 1, 1));
        let manual_cols = im2col_3d(&manual_full_padded, &config_im2col, 4, 4, 4)?;
        println!("  Im2col shape: {:?}", manual_cols.dims());
        
        // 4. Matmul
        let out_c = 3;
        let patch_size = 2 * 3 * 3 * 3; // in_c * kt * kh * kw = 54
        let weight_2d = weight.reshape((out_c, patch_size))?.t()?.contiguous()?;
        let cols_2d = manual_cols.reshape((1 * 64, patch_size))?;
        let y_2d = cols_2d.matmul(&weight_2d)?;
        let manual_output = y_2d.reshape((1, 4, 4, 4, 3))?.permute((0, 4, 1, 2, 3))?;
        
        // 5. Add bias
        let bias_reshaped = bias.reshape((1, 3, 1, 1, 1))?;
        let manual_output = manual_output.broadcast_add(&bias_reshaped)?;
        
        println!("  Manual output shape: {:?}", manual_output.dims());
        let manual_flat = manual_output.flatten_all()?.to_vec1::<f32>()?;
        println!("  Manual first 8: {:?}", &manual_flat[..8]);
        
        let manual_diff = max_abs_diff(&manual_output, ref_output)?;
        println!("  Manual vs ref diff: {:.6e}", manual_diff);
        
        // Now test Conv3d
        println!("\nConv3d forward:");
        let config = Conv3dConfig::new((3, 3, 3))
            .with_padding((0, 1, 1))  // Spatial padding only
            .with_causal(true)
            .with_padding_mode(PaddingMode::Replicate);

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
        println!("Ref first 8: {:?}", &ref_flat[..8]);

        let output_diff = max_abs_diff(&rust_output, ref_output)?;
        println!("Output diff: {:.6e}", output_diff);

        assert!(output_diff < TOLERANCE, "Causal Conv3d output mismatch: diff={}", output_diff);
        println!("✅ Causal Conv3d PASS");

        Ok(())
    }
}
