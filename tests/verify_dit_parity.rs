#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;
    use candle_video::models::ltx_video::ltx_transformer::{
        LtxVideoTransformer3DModel, LtxVideoTransformer3DModelConfig,
    };
    use std::path::Path;

    #[test]
    fn test_dit_parity() -> anyhow::Result<()> {
        // 1. Load reference tensors
        let path = Path::new("gen_dit_ref.safetensors");
        if !path.exists() {
            println!("Skipping test_dit_parity: gen_dit_ref.safetensors not found");
            return Ok(());
        }

        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        println!("Running on device: {:?}", device);

        let tensors = candle_core::safetensors::load(path, &device)?;

        // 2. Setup Config (must match python script)
        let config = LtxVideoTransformer3DModelConfig {
            in_channels: 32,
            out_channels: 32,
            patch_size: 1,
            patch_size_t: 1,
            num_attention_heads: 2,
            attention_head_dim: 16,
            cross_attention_dim: 32,
            num_layers: 2,
            caption_channels: 32,
            qk_norm: "rms_norm_across_heads".to_string(),
            norm_elementwise_affine: false,
            norm_eps: 1e-6,
            attention_bias: true,
            attention_out_bias: true,
        };

        // 3. Load Weights
        // The python script saved model weights with prefix "model." e.g. "model.proj_in.weight"
        // But VarBuilder usually expects strict hierarchical matching or we can use pp.
        // Let's filter and remap keys if necessary.
        // Actually, python script did: tensors[f"model.{k}"] = v
        // So keys are like "model.proj_in.weight".
        // We can create a VB from the tensors subset.

        let mut model_weights = std::collections::HashMap::new();
        for (k, v) in tensors.iter() {
            if k.starts_with("model.") {
                let new_key = k.strip_prefix("model.").unwrap();
                model_weights.insert(new_key.to_string(), v.clone());
            }
        }

        let vb = VarBuilder::from_tensors(model_weights, DType::F32, &device);

        // 4. Initialize Model
        let model = LtxVideoTransformer3DModel::new(&config, vb)?;

        // 5. runs forward
        let hidden_states = tensors.get("hidden_states").unwrap();
        let encoder_hidden_states = tensors.get("encoder_hidden_states").unwrap();
        let timestep = tensors.get("timestep").unwrap();

        // Python used:
        // b = 1
        // f = 8
        // h = 32
        // w = 32
        // rope_interpolation_scale=(1.0, 1.0, 1.0)

        let out = model.forward(
            hidden_states,
            encoder_hidden_states,
            timestep,
            None,                  // encoder_attention_mask (was None in python script)
            8,                     // num_frames
            32,                    // height
            32,                    // width
            Some((1.0, 1.0, 1.0)), // rope_interpolation_scale
            None,                  // video_coords
            None,                  // skip_layer_mask
        )?;

        // 6. Compare
        let ref_out = tensors.get("output").unwrap();

        println!("Output shape: {:?}", out.shape());
        println!("Ref output shape: {:?}", ref_out.shape());

        let diff = (out - ref_out)?.abs()?.max_all()?;
        let diff_val = diff.to_vec0::<f32>()?;

        println!("Max difference: {}", diff_val);

        assert!(diff_val < 2e-3, "Difference too large: {}", diff_val);

        Ok(())
    }
}
