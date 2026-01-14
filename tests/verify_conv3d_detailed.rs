//! Detailed Conv3d parity test with intermediate values.

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, IndexOp, Tensor};
    use candle_nn::VarBuilder;
    use candle_video::ops::conv3d::cpu::{Im2ColConfig, im2col_3d};
    use candle_video::ops::conv3d::{Conv3d, Conv3dConfig};
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
    fn test_detailed_conv3d() -> anyhow::Result<()> {
        let ref_path = Path::new("gen_conv3d_detailed.safetensors");
        if !ref_path.exists() {
            println!("Skipping: gen_conv3d_detailed.safetensors not found");
            println!("Run: python scripts/debug_conv3d_detailed.py");
            return Ok(());
        }

        let device = Device::Cpu;
        let tensors = candle_core::safetensors::load(ref_path, &device)?;

        println!("\n=== Detailed Conv3d Parity Test ===");

        let input = tensors.get("input").unwrap();
        let weight = tensors.get("weight").unwrap();
        let bias = tensors.get("bias").unwrap();
        let input_padded = tensors.get("input_padded").unwrap();
        let ref_output = tensors.get("output_pytorch").unwrap();
        let ref_im2col = tensors.get("im2col_cols").unwrap();

        println!("Input shape: {:?}", input.dims());
        println!("Weight shape: {:?}", weight.dims());
        println!("Input padded shape: {:?}", input_padded.dims());
        println!("Ref output shape: {:?}", ref_output.dims());
        println!("Ref im2col shape: {:?}", ref_im2col.dims());

        // Step 1: Test padding
        println!("\n--- Step 1: Test padding ---");
        let rust_padded = input.pad_with_zeros(2, 1, 1)?; // temporal
        let rust_padded = rust_padded.pad_with_zeros(3, 1, 1)?; // height
        let rust_padded = rust_padded.pad_with_zeros(4, 1, 1)?; // width
        println!("Rust padded shape: {:?}", rust_padded.dims());

        let pad_diff = max_abs_diff(&rust_padded, input_padded)?;
        println!("Padding diff: {:.6e}", pad_diff);
        assert!(pad_diff < 1e-6, "Padding mismatch");
        println!("✅ Padding PASS");

        // Step 2: Test im2col
        println!("\n--- Step 2: Test im2col ---");
        let config = Im2ColConfig::new((3, 3, 3), (1, 1, 1), (1, 1, 1));
        let rust_im2col = im2col_3d(&rust_padded, &config, 4, 4, 4)?;
        println!("Rust im2col shape: {:?}", rust_im2col.dims());

        let im2col_diff = max_abs_diff(&rust_im2col, ref_im2col)?;
        println!("Im2col diff: {:.6e}", im2col_diff);

        // Print first patch comparison
        let rust_first = rust_im2col.i((0, 0, ..))?.to_vec1::<f32>()?;
        let ref_first = ref_im2col.i((0, 0, ..))?.to_vec1::<f32>()?;
        println!(
            "Rust first patch (first 8): {:?}",
            &rust_first[..8.min(rust_first.len())]
        );
        println!(
            "Ref first patch (first 8): {:?}",
            &ref_first[..8.min(ref_first.len())]
        );

        if im2col_diff > TOLERANCE {
            println!("⚠️ Im2col has differences, investigating...");
            // Find where differences are
            for i in 0..4 {
                let rust_patch = rust_im2col.i((0, i, ..))?.to_vec1::<f32>()?;
                let ref_patch = ref_im2col.i((0, i, ..))?.to_vec1::<f32>()?;
                let patch_diff: f32 = rust_patch
                    .iter()
                    .zip(ref_patch.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                if patch_diff > 1e-6 {
                    println!("Patch {} diff: {:.6e}", i, patch_diff);
                    println!("  Rust: {:?}", &rust_patch[..8.min(rust_patch.len())]);
                    println!("  Ref:  {:?}", &ref_patch[..8.min(ref_patch.len())]);
                }
            }
        } else {
            println!("✅ Im2col PASS");
        }

        // Step 3: Test full convolution
        println!("\n--- Step 3: Test full Conv3d ---");
        let config = Conv3dConfig::new((3, 3, 3)).with_padding((1, 1, 1));

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

        assert!(
            output_diff < TOLERANCE,
            "Conv3d output mismatch: diff={}",
            output_diff
        );
        println!("✅ Conv3d PASS");

        Ok(())
    }
}
