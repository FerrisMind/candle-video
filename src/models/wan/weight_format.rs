//! Native Wan checkpoint key remapping (ComfyUI / single-file → Diffusers).
//!
//! Ported from `diffusers.loaders.single_file_utils::{convert_wan_transformer_to_diffusers, convert_wan_vae_to_diffusers}`.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Result, Tensor};

/// Remap native Wan transformer keys to Diffusers `WanTransformer3DModel` names.
pub fn remap_transformer_key(key: &str) -> String {
    let key = key.strip_prefix("model.diffusion_model.").unwrap_or(key);

    let mut result = key.to_string();
    const REPLACEMENTS: &[(&str, &str)] = &[
        (
            "time_embedding.0",
            "condition_embedder.time_embedder.linear_1",
        ),
        (
            "time_embedding.2",
            "condition_embedder.time_embedder.linear_2",
        ),
        (
            "text_embedding.0",
            "condition_embedder.text_embedder.linear_1",
        ),
        (
            "text_embedding.2",
            "condition_embedder.text_embedder.linear_2",
        ),
        ("time_projection.1", "condition_embedder.time_proj"),
        ("cross_attn", "attn2"),
        ("self_attn", "attn1"),
        (".o.", ".to_out.0."),
        (".q.", ".to_q."),
        (".k.", ".to_k."),
        (".v.", ".to_v."),
        ("head.modulation", "scale_shift_table"),
        ("head.head", "proj_out"),
        ("modulation", "scale_shift_table"),
        ("ffn.0", "ffn.net.0.proj"),
        ("ffn.2", "ffn.net.2"),
        ("norm2", "norm__placeholder"),
        ("norm3", "norm2"),
        ("norm__placeholder", "norm3"),
    ];
    for (from, to) in REPLACEMENTS {
        result = result.replace(from, to);
    }
    result
}

/// Remap native Wan VAE keys to Diffusers `AutoencoderKLWan` names.
pub fn remap_vae_state_dict(native: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    let mut out = HashMap::new();
    for (key, value) in native {
        if let Some(mapped) = remap_vae_key(&key) {
            out.insert(mapped, value);
        }
    }
    out
}

fn remap_vae_key(key: &str) -> Option<String> {
    const MIDDLE: &[(&str, &str)] = &[
        (
            "encoder.middle.0.residual.0.gamma",
            "encoder.mid_block.resnets.0.norm1.gamma",
        ),
        (
            "encoder.middle.0.residual.2.bias",
            "encoder.mid_block.resnets.0.conv1.bias",
        ),
        (
            "encoder.middle.0.residual.2.weight",
            "encoder.mid_block.resnets.0.conv1.weight",
        ),
        (
            "encoder.middle.0.residual.3.gamma",
            "encoder.mid_block.resnets.0.norm2.gamma",
        ),
        (
            "encoder.middle.0.residual.6.bias",
            "encoder.mid_block.resnets.0.conv2.bias",
        ),
        (
            "encoder.middle.0.residual.6.weight",
            "encoder.mid_block.resnets.0.conv2.weight",
        ),
        (
            "encoder.middle.2.residual.0.gamma",
            "encoder.mid_block.resnets.1.norm1.gamma",
        ),
        (
            "encoder.middle.2.residual.2.bias",
            "encoder.mid_block.resnets.1.conv1.bias",
        ),
        (
            "encoder.middle.2.residual.2.weight",
            "encoder.mid_block.resnets.1.conv1.weight",
        ),
        (
            "encoder.middle.2.residual.3.gamma",
            "encoder.mid_block.resnets.1.norm2.gamma",
        ),
        (
            "encoder.middle.2.residual.6.bias",
            "encoder.mid_block.resnets.1.conv2.bias",
        ),
        (
            "encoder.middle.2.residual.6.weight",
            "encoder.mid_block.resnets.1.conv2.weight",
        ),
        (
            "decoder.middle.0.residual.0.gamma",
            "decoder.mid_block.resnets.0.norm1.gamma",
        ),
        (
            "decoder.middle.0.residual.2.bias",
            "decoder.mid_block.resnets.0.conv1.bias",
        ),
        (
            "decoder.middle.0.residual.2.weight",
            "decoder.mid_block.resnets.0.conv1.weight",
        ),
        (
            "decoder.middle.0.residual.3.gamma",
            "decoder.mid_block.resnets.0.norm2.gamma",
        ),
        (
            "decoder.middle.0.residual.6.bias",
            "decoder.mid_block.resnets.0.conv2.bias",
        ),
        (
            "decoder.middle.0.residual.6.weight",
            "decoder.mid_block.resnets.0.conv2.weight",
        ),
        (
            "decoder.middle.2.residual.0.gamma",
            "decoder.mid_block.resnets.1.norm1.gamma",
        ),
        (
            "decoder.middle.2.residual.2.bias",
            "decoder.mid_block.resnets.1.conv1.bias",
        ),
        (
            "decoder.middle.2.residual.2.weight",
            "decoder.mid_block.resnets.1.conv1.weight",
        ),
        (
            "decoder.middle.2.residual.3.gamma",
            "decoder.mid_block.resnets.1.norm2.gamma",
        ),
        (
            "decoder.middle.2.residual.6.bias",
            "decoder.mid_block.resnets.1.conv2.bias",
        ),
        (
            "decoder.middle.2.residual.6.weight",
            "decoder.mid_block.resnets.1.conv2.weight",
        ),
        (
            "encoder.middle.1.norm.gamma",
            "encoder.mid_block.attentions.0.norm.gamma",
        ),
        (
            "encoder.middle.1.to_qkv.weight",
            "encoder.mid_block.attentions.0.to_qkv.weight",
        ),
        (
            "encoder.middle.1.to_qkv.bias",
            "encoder.mid_block.attentions.0.to_qkv.bias",
        ),
        (
            "encoder.middle.1.proj.weight",
            "encoder.mid_block.attentions.0.proj.weight",
        ),
        (
            "encoder.middle.1.proj.bias",
            "encoder.mid_block.attentions.0.proj.bias",
        ),
        (
            "decoder.middle.1.norm.gamma",
            "decoder.mid_block.attentions.0.norm.gamma",
        ),
        (
            "decoder.middle.1.to_qkv.weight",
            "decoder.mid_block.attentions.0.to_qkv.weight",
        ),
        (
            "decoder.middle.1.to_qkv.bias",
            "decoder.mid_block.attentions.0.to_qkv.bias",
        ),
        (
            "decoder.middle.1.proj.weight",
            "decoder.mid_block.attentions.0.proj.weight",
        ),
        (
            "decoder.middle.1.proj.bias",
            "decoder.mid_block.attentions.0.proj.bias",
        ),
        ("encoder.head.0.gamma", "encoder.norm_out.gamma"),
        ("encoder.head.2.bias", "encoder.conv_out.bias"),
        ("encoder.head.2.weight", "encoder.conv_out.weight"),
        ("decoder.head.0.gamma", "decoder.norm_out.gamma"),
        ("decoder.head.2.bias", "decoder.conv_out.bias"),
        ("decoder.head.2.weight", "decoder.conv_out.weight"),
        ("conv1.weight", "quant_conv.weight"),
        ("conv1.bias", "quant_conv.bias"),
        ("conv2.weight", "post_quant_conv.weight"),
        ("conv2.bias", "post_quant_conv.bias"),
    ];
    for (from, to) in MIDDLE {
        if key == *from {
            return Some((*to).to_string());
        }
    }

    if key == "encoder.conv1.weight" {
        return Some("encoder.conv_in.weight".into());
    }
    if key == "encoder.conv1.bias" {
        return Some("encoder.conv_in.bias".into());
    }
    if key == "decoder.conv1.weight" {
        return Some("decoder.conv_in.weight".into());
    }
    if key == "decoder.conv1.bias" {
        return Some("decoder.conv_in.bias".into());
    }

    if key.starts_with("encoder.downsamples.") {
        let mut new_key = key.replace("encoder.downsamples.", "encoder.down_blocks.");
        new_key = new_key.replace(".residual.0.gamma", ".norm1.gamma");
        new_key = new_key.replace(".residual.2.bias", ".conv1.bias");
        new_key = new_key.replace(".residual.2.weight", ".conv1.weight");
        new_key = new_key.replace(".residual.3.gamma", ".norm2.gamma");
        new_key = new_key.replace(".residual.6.bias", ".conv2.bias");
        new_key = new_key.replace(".residual.6.weight", ".conv2.weight");
        new_key = new_key.replace(".shortcut.bias", ".conv_shortcut.bias");
        new_key = new_key.replace(".shortcut.weight", ".conv_shortcut.weight");
        return Some(new_key);
    }

    if key.starts_with("decoder.upsamples.") {
        return remap_decoder_upsample_key(key);
    }

    Some(key.to_string())
}

fn remap_decoder_upsample_key(key: &str) -> Option<String> {
    let parts: Vec<&str> = key.split('.').collect();
    if parts.len() < 4 {
        return Some(key.to_string());
    }
    let block_idx: usize = parts[2].parse().unwrap_or(0);

    if key.contains("residual") {
        let (new_block_idx, resnet_idx) = match block_idx {
            0..=2 => (0, block_idx),
            4..=6 => (1, block_idx - 4),
            8..=10 => (2, block_idx - 8),
            12..=14 => (3, block_idx - 12),
            _ => return Some(key.to_string()),
        };
        let new_key = if key.contains(".residual.0.gamma") {
            format!("decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.norm1.gamma")
        } else if key.contains(".residual.2.bias") {
            format!("decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv1.bias")
        } else if key.contains(".residual.2.weight") {
            format!("decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv1.weight")
        } else if key.contains(".residual.3.gamma") {
            format!("decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.norm2.gamma")
        } else if key.contains(".residual.6.bias") {
            format!("decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv2.bias")
        } else if key.contains(".residual.6.weight") {
            format!("decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv2.weight")
        } else {
            key.to_string()
        };
        return Some(new_key);
    }

    if key.contains(".shortcut.") {
        let new_key = if block_idx == 4 {
            key.replace(".shortcut.", ".resnets.0.conv_shortcut.")
                .replace("decoder.upsamples.4", "decoder.up_blocks.1")
        } else {
            key.replace("decoder.upsamples.", "decoder.up_blocks.")
                .replace(".shortcut.", ".conv_shortcut.")
        };
        return Some(new_key);
    }

    if key.contains(".resample.") || key.contains(".time_conv.") {
        let new_key = match block_idx {
            3 => key.replace("decoder.upsamples.3", "decoder.up_blocks.0.upsamplers.0"),
            7 => key.replace("decoder.upsamples.7", "decoder.up_blocks.1.upsamplers.0"),
            11 => key.replace("decoder.upsamples.11", "decoder.up_blocks.2.upsamplers.0"),
            _ => key.replace("decoder.upsamples.", "decoder.up_blocks."),
        };
        return Some(new_key);
    }

    Some(key.replace("decoder.upsamples.", "decoder.up_blocks."))
}

/// Load a safetensors file and remap transformer keys in memory.
pub fn load_transformer_tensors(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let native = candle_core::safetensors::load(path, device)?;
    Ok(native
        .into_iter()
        .map(|(k, v)| (remap_transformer_key(&k), v))
        .collect())
}

/// Load a safetensors file and remap VAE keys in memory.
pub fn load_vae_tensors(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let native = candle_core::safetensors::load(path, device)?;
    Ok(remap_vae_state_dict(native))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transformer_cross_attn_maps_to_attn2() {
        let k = "model.diffusion_model.blocks.0.cross_attn.q.weight";
        assert_eq!(remap_transformer_key(k), "blocks.0.attn2.to_q.weight");
    }

    #[test]
    fn vae_conv2_maps_to_post_quant_conv() {
        let mut m = HashMap::new();
        m.insert(
            "conv2.weight".into(),
            Tensor::zeros((1,), candle_core::DType::F32, &Device::Cpu).unwrap(),
        );
        let out = remap_vae_state_dict(m);
        assert!(out.contains_key("post_quant_conv.weight"));
    }
}
