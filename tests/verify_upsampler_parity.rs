#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, IndexOp};
    use candle_nn::VarBuilder;
    use candle_video::models::ltx_video::vae::LtxVideoCausalConv3d;
    use std::path::Path;

    #[test]
    fn test_upsampler_detailed() -> anyhow::Result<()> {
        let path = Path::new("gen_upsampler_detailed.safetensors");
        if !path.exists() {
            println!("Skipping test: gen_upsampler_detailed.safetensors not found");
            return Ok(());
        }

        let device = Device::new_cuda(0)?;
        let dtype = DType::F32;
        println!("Running on device: {:?}, dtype: {:?}", device, dtype);

        let tensors = candle_core::safetensors::load(path, &device)?;

        // Load inputs and references
        let input = tensors.get("input").unwrap().to_dtype(dtype)?;
        let ref_conv_out = tensors.get("conv_out").unwrap().to_dtype(dtype)?;
        let ref_main_path = tensors.get("main_path").unwrap().to_dtype(dtype)?;
        let ref_residual = tensors.get("residual").unwrap().to_dtype(dtype)?;
        let ref_output = tensors.get("output").unwrap().to_dtype(dtype)?;

        println!("Input shape: {:?}", input.shape());
        println!("Ref conv_out shape: {:?}", ref_conv_out.shape());
        println!("Ref main_path shape: {:?}", ref_main_path.shape());
        println!("Ref residual shape: {:?}", ref_residual.shape());
        println!("Ref output shape: {:?}", ref_output.shape());

        // Load conv weights
        let mut conv_weights = std::collections::HashMap::new();
        for (k, v) in tensors.iter() {
            if k.starts_with("upsampler.conv.") {
                let new_key = k.strip_prefix("upsampler.conv.").unwrap();
                println!(
                    "Loading weight: {} -> {}, shape: {:?}",
                    k,
                    new_key,
                    v.shape()
                );
                conv_weights.insert(new_key.to_string(), v.to_dtype(dtype)?.clone());
            }
        }

        println!("Loaded {} weights", conv_weights.len());
        let vb = VarBuilder::from_tensors(conv_weights, dtype, &device);

        // Test conv - IMPORTANT: upsampler conv is NOT causal!
        let conv =
            LtxVideoCausalConv3d::new(1024, 4096, (3, 3, 3), (1, 1, 1), (1, 1, 1), 1, false, vb)?;
        let rust_conv_out = conv.forward(&input)?;
        let conv_diff = (rust_conv_out.sub(&ref_conv_out)?).abs()?.max_all()?;
        println!("\nConv output diff: {}", conv_diff.to_vec0::<f32>()?);

        // Test pixel shuffle on main path using ref_conv_out to isolate shuffle logic
        let (b, c, t, h, w) = ref_conv_out.dims5()?;
        let (st, sh, sw) = (2usize, 2usize, 2usize);
        let c_out = c / (st * sh * sw); // 512

        // Reshape to [B, C', st, sh, sw, T, H, W]
        let reshaped = ref_conv_out.reshape(&[b, c_out, st, sh, sw, t, h, w])?;
        println!("After reshape: {:?}", reshaped.shape());

        // Permute(0, 1, 5, 2, 6, 3, 7, 4) -> [B, C', T, st, H, sh, W, sw]
        let permuted = reshaped
            .permute(vec![0, 1, 5, 2, 6, 3, 7, 4])?
            .contiguous()?;
        println!("After permute: {:?}", permuted.shape());

        // flatten(6,7).flatten(4,5).flatten(2,3)
        let step1 = permuted.reshape(&[b, c_out, t, st, h, sh, w * sw])?;
        let step2 = step1.reshape(&[b, c_out, t, st, h * sh, w * sw])?;
        let step3 = step2.reshape(&[b, c_out, t * st, h * sh, w * sw])?;

        // Slice [:, :, st-1:]
        let main_path = step3.i((.., .., (st - 1).., .., ..))?;
        println!("Main path shape: {:?}", main_path.shape());

        // Compare main_path
        let main_diff = (main_path.sub(&ref_main_path)?).abs()?.max_all()?;
        println!(
            "\nMain path diff (using ref_conv_out as input): {}",
            main_diff.to_vec0::<f32>()?
        );

        // Test residual path using input
        let (b, c_in, t, h, w) = input.dims5()?;
        let c_res_out = c_in / (st * sh * sw); // 128

        let res_reshaped = input.reshape(&[b, c_res_out, st, sh, sw, t, h, w])?;
        let res_permuted = res_reshaped
            .permute(vec![0, 1, 5, 2, 6, 3, 7, 4])?
            .contiguous()?;

        let res_step1 = res_permuted.reshape(&[b, c_res_out, t, st, h, sh, w * sw])?;
        let res_step2 = res_step1.reshape(&[b, c_res_out, t, st, h * sh, w * sw])?;
        let res_step3 = res_step2.reshape(&[b, c_res_out, t * st, h * sh, w * sw])?;

        // Repeat channels: 128 * 4 = 512
        let channel_repeats = 4usize;
        let res_repeated = res_step3.repeat((1, channel_repeats, 1, 1, 1))?;
        println!("Residual after repeat: {:?}", res_repeated.shape());

        // Slice
        let residual = res_repeated.i((.., .., (st - 1).., .., ..))?;
        println!("Residual shape: {:?}", residual.shape());

        // Compare residual
        let res_diff = (residual.sub(&ref_residual)?).abs()?.max_all()?;
        println!("\nResidual diff: {}", res_diff.to_vec0::<f32>()?);

        // Final output
        let output = main_path.add(&residual)?;
        let out_diff = (output.sub(&ref_output)?).abs()?.max_all()?;
        println!("\nFinal output diff: {}", out_diff.to_vec0::<f32>()?);

        println!("\n=== Summary ===");
        println!("Conv diff: {}", conv_diff.to_vec0::<f32>()?);
        println!("Main path diff: {}", main_diff.to_vec0::<f32>()?);
        println!("Residual diff: {}", res_diff.to_vec0::<f32>()?);
        println!("Output diff: {}", out_diff.to_vec0::<f32>()?);

        Ok(())
    }
}
