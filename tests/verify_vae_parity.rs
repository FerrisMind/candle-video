#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, IndexOp, Module};
    use candle_nn::VarBuilder;
    use candle_video::models::ltx_video::vae::{
        AutoencoderKLLtxVideo, AutoencoderKLLtxVideoConfig,
    };
    use std::path::Path;

    #[test]
    fn test_vae_full_parity() -> anyhow::Result<()> {
        let path = Path::new("gen_vae_ref.safetensors");
        if !path.exists() {
            println!("Skipping test: gen_vae_ref.safetensors not found");
            return Ok(());
        }

        // Use GPU with F32 for accurate parity comparison
        let device = Device::new_cuda(0)?;
        let dtype = DType::F32;
        println!("Running on device: {:?}, dtype: {:?}", device, dtype);

        let tensors = candle_core::safetensors::load(path, &device)?;

        // Load model weights
        let mut model_weights = std::collections::HashMap::new();
        for (k, v) in tensors.iter() {
            if k.starts_with("vae.") {
                let new_key = k.strip_prefix("vae.").unwrap();
                model_weights.insert(new_key.to_string(), v.to_dtype(dtype)?.clone());
            }
        }

        let vb = VarBuilder::from_tensors(model_weights, dtype, &device);
        let config = AutoencoderKLLtxVideoConfig::default();
        let model = AutoencoderKLLtxVideo::new(config, vb)?;
        let decoder = &model.decoder;

        // Load inputs (convert to BF16)
        let latents = tensors.get("latents").unwrap().to_dtype(dtype)?;
        let temb = tensors.get("temb").unwrap().to_dtype(dtype)?;

        println!("Latents shape: {:?}", latents.shape());

        // Prepare temb_scaled
        let temb_scaled = if let Some(tsm) = &decoder.timestep_scale_multiplier {
            let t_flat = temb.flatten_all()?;
            Some(t_flat.broadcast_mul(&tsm.to_dtype(dtype)?)?)
        } else {
            Some(temb.flatten_all()?)
        };

        // Step 1: conv_in
        let h = decoder.conv_in.forward(&latents)?;
        println!("ConvIn output shape: {:?}", h.shape());

        // Step 2: mid_block
        let mid_out = decoder.mid_block.forward(&h, temb_scaled.as_ref(), false)?;
        println!("MidBlock output shape: {:?}", mid_out.shape());

        if let Some(ref_mid) = tensors.get("mid_block_out") {
            let ref_mid = ref_mid.to_dtype(dtype)?;
            let diff = (mid_out
                .to_dtype(DType::F32)?
                .sub(&ref_mid.to_dtype(DType::F32)?)?)
            .abs()?
            .max_all()?;
            println!("MidBlock max diff: {}", diff.to_vec0::<f32>()?);
        }

        // Step 3: up_blocks
        let ref_keys = vec![
            ("up_block_0_out", "UpBlock 0"),
            ("up_block_1_out", "UpBlock 1"),
            ("up_block_2_out", "UpBlock 2"),
        ];

        let mut h_up = mid_out;
        for (i, ub) in decoder.up_blocks.iter().enumerate() {
            h_up = ub.forward(&h_up, temb_scaled.as_ref(), false)?;
            println!("{} output shape: {:?}", ref_keys[i].1, h_up.shape());

            if let Some(ref_tensor) = tensors.get(ref_keys[i].0) {
                let ref_tensor = ref_tensor.to_dtype(dtype)?;
                let diff = (h_up
                    .to_dtype(DType::F32)?
                    .sub(&ref_tensor.to_dtype(DType::F32)?)?)
                .abs()?
                .max_all()?;
                let diff_val = diff.to_vec0::<f32>()?;
                println!("{} max diff: {}", ref_keys[i].1, diff_val);
            }
        }

        // Step 4: norm_out + time conditioning + conv_act + conv_out
        let mut h_norm = h_up;

        // Apply norm_out
        if let Some(ref norm) = decoder.norm_out {
            h_norm = norm
                .forward(&h_norm.permute((0, 2, 3, 4, 1))?)?
                .permute((0, 4, 1, 2, 3))?;
        }

        // Apply global time conditioning
        if let (Some(te), Some(sst), Some(temb_s)) = (
            &decoder.time_embedder,
            &decoder.scale_shift_table,
            &temb_scaled,
        ) {
            let temb_proj = te.forward(temb_s, h_norm.dtype())?;
            let batch_size = h_norm.dims5()?.0;
            let c = sst.dims2()?.1;

            let temb_shaped = temb_proj
                .reshape((batch_size, 2, c))?
                .broadcast_add(&sst.unsqueeze(0)?)?
                .unsqueeze(3)?
                .unsqueeze(4)?
                .unsqueeze(5)?;

            let shift = temb_shaped.i((.., 0, .., .., .., ..))?.squeeze(1)?;
            let scale = temb_shaped.i((.., 1, .., .., .., ..))?.squeeze(1)?;

            let h_shape = h_norm.shape();
            let scale_b = scale.broadcast_as(h_shape)?;
            let shift_b = shift.broadcast_as(h_shape)?;

            h_norm = h_norm
                .broadcast_mul(&scale_b.affine(1.0, 1.0)?)?
                .broadcast_add(&shift_b)?;
        }

        // conv_act
        let h_act = h_norm.apply(&decoder.conv_act)?;

        // conv_out
        let conv_out = decoder.conv_out.forward(&h_act)?;
        println!("ConvOut output shape: {:?}", conv_out.shape());

        if let Some(ref_conv_out) = tensors.get("conv_out_out") {
            let ref_conv_out = ref_conv_out.to_dtype(dtype)?;
            let diff = (conv_out
                .to_dtype(DType::F32)?
                .sub(&ref_conv_out.to_dtype(DType::F32)?)?)
            .abs()?
            .max_all()?;
            let diff_val = diff.to_vec0::<f32>()?;
            println!("ConvOut max diff: {}", diff_val);
        }

        // Step 5: unpatchify
        let output = decoder.unpatchify(&conv_out)?;
        println!("Final output shape: {:?}", output.shape());

        if let Some(ref_out) = tensors.get("out") {
            let ref_out = ref_out.to_dtype(dtype)?;
            let diff = (output
                .to_dtype(DType::F32)?
                .sub(&ref_out.to_dtype(DType::F32)?)?)
            .abs()?
            .max_all()?;
            let diff_val = diff.to_vec0::<f32>()?;
            println!("Final output max diff: {}", diff_val);
        }

        println!("\nTest completed.");
        Ok(())
    }
}
