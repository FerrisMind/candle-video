use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

use super::config::{AutoencoderKLWanConfig, WanTransformer3DConfig};
use super::text_encoder::{QuantizedUMT5Encoder, UMT5TextEncoder, UMT5Tokenizer};
use super::transformer_wan::WanTransformer3DModel;
use super::vae::AutoencoderKLWan;

pub use crate::interfaces::t5_encoder::T5EncoderConfig;

#[derive(Debug, thiserror::Error)]
pub enum WanLoaderError {
    #[error("Failed to read file: {path}")]
    FileRead {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("Failed to parse JSON config: {path}")]
    JsonParse {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("Model file not found: {path}")]
    ModelNotFound { path: String },
}

const TRANSFORMER_KEY_RENAMES: &[(&str, &str)] = &[
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
    ("head.modulation", "scale_shift_table"),
    ("head.head", "proj_out"),
    ("modulation", "scale_shift_table"),
    ("ffn.0", "ffn.net.0.proj"),
    ("ffn.2", "ffn.net.2"),
    ("self_attn.q", "attn1.to_q"),
    ("self_attn.k", "attn1.to_k"),
    ("self_attn.v", "attn1.to_v"),
    ("self_attn.o", "attn1.to_out.0"),
    ("self_attn.norm_q", "attn1.norm_q"),
    ("self_attn.norm_k", "attn1.norm_k"),
    ("cross_attn.q", "attn2.to_q"),
    ("cross_attn.k", "attn2.to_k"),
    ("cross_attn.v", "attn2.to_v"),
    ("cross_attn.o", "attn2.to_out.0"),
    ("cross_attn.norm_q", "attn2.norm_q"),
    ("cross_attn.norm_k", "attn2.norm_k"),
];

fn needs_prefix_stripping(keys: &[String]) -> bool {
    keys.iter().any(|k| k.starts_with("model.diffusion_model."))
}

fn is_original_official_format(keys: &[String]) -> bool {
    keys.iter()
        .any(|k| k.contains("self_attn.") || k.contains("cross_attn."))
}

fn convert_key_to_diffusers(key: &str, is_original_format: bool) -> String {
    let key = key.strip_prefix("model.diffusion_model.").unwrap_or(key);

    if !is_original_format {
        return key.to_string();
    }

    let mut result = key.to_string();

    if result.contains(".norm2.") {
        result = result.replace(".norm2.", ".__norm_placeholder__.");
    }
    if result.contains(".norm3.") {
        result = result.replace(".norm3.", ".norm2.");
    }
    if result.contains(".__norm_placeholder__.") {
        result = result.replace(".__norm_placeholder__.", ".norm3.");
    }

    for (from, to) in TRANSFORMER_KEY_RENAMES {
        if result.contains(from) {
            result = result.replace(from, to);
        }
    }

    result
}

fn load_transformer_with_conversion(
    path: &Path,
    config: WanTransformer3DConfig,
    device: &Device,
    dtype: DType,
) -> std::result::Result<WanTransformer3DModel, WanLoaderError> {
    let file_data = std::fs::read(path).map_err(|e| WanLoaderError::FileRead {
        path: path.display().to_string(),
        source: e,
    })?;

    let tensors = SafeTensors::deserialize(&file_data)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to parse safetensors: {}", e)))?;

    let keys: Vec<String> = tensors.names().into_iter().map(|s| s.to_string()).collect();

    let needs_stripping = needs_prefix_stripping(&keys);
    let is_original = is_original_official_format(&keys);

    if needs_stripping || is_original {
        if is_original {
            println!(
                "  Detected original official Wan format, converting keys to diffusers format..."
            );
        } else {
            println!(
                "  Detected official Wan weights with model.diffusion_model. prefix, stripping prefix..."
            );
        }

        let original_tensors = candle_core::safetensors::load_buffer(&file_data, device)?;

        let mut converted_tensors: HashMap<String, Tensor> = HashMap::new();

        for (key, tensor) in original_tensors {
            let new_key = convert_key_to_diffusers(&key, is_original);

            let tensor = if tensor.dtype() != dtype {
                tensor.to_dtype(dtype)?
            } else {
                tensor
            };

            converted_tensors.insert(new_key, tensor);
        }

        let vb = VarBuilder::from_tensors(converted_tensors, dtype, device);

        WanTransformer3DModel::new(config, vb).map_err(WanLoaderError::from)
    } else {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device)? };
        WanTransformer3DModel::new(config, vb).map_err(WanLoaderError::from)
    }
}

pub fn load_transformer(
    weights_path: impl AsRef<Path>,
    config: WanTransformer3DConfig,
    device: &Device,
    dtype: DType,
) -> std::result::Result<WanTransformer3DModel, WanLoaderError> {
    let path = weights_path.as_ref();

    if !path.exists() {
        return Err(WanLoaderError::ModelNotFound {
            path: path.display().to_string(),
        });
    }

    load_transformer_with_conversion(path, config, device, dtype)
}

pub fn load_transformer_from_dir(
    dir: impl AsRef<Path>,
    config: WanTransformer3DConfig,
    device: &Device,
    dtype: DType,
) -> std::result::Result<WanTransformer3DModel, WanLoaderError> {
    let dir = dir.as_ref();

    let candidates = [
        "diffusion_pytorch_model.safetensors",
        "model.safetensors",
        "transformer.safetensors",
    ];

    for candidate in &candidates {
        let path = dir.join(candidate);
        if path.exists() {
            return load_transformer(&path, config, device, dtype);
        }
    }

    Err(WanLoaderError::ModelNotFound {
        path: dir.display().to_string(),
    })
}

fn is_official_vae_format(keys: &[String]) -> bool {
    keys.iter().any(|k| {
        k.starts_with("encoder.conv1")
            || k.starts_with("decoder.conv1")
            || k.contains(".downsamples.")
            || k.contains(".upsamples.")
            || k.contains(".middle.")
            || k.contains(".head.")
    })
}

fn convert_vae_key_to_diffusers(key: &str) -> String {
    let mut result = key.to_string();

    if result == "conv1.weight" {
        return "quant_conv.weight".to_string();
    }
    if result == "conv1.bias" {
        return "quant_conv.bias".to_string();
    }
    if result == "conv2.weight" {
        return "post_quant_conv.weight".to_string();
    }
    if result == "conv2.bias" {
        return "post_quant_conv.bias".to_string();
    }

    if result == "encoder.conv1.weight" {
        return "encoder.conv_in.weight".to_string();
    }
    if result == "encoder.conv1.bias" {
        return "encoder.conv_in.bias".to_string();
    }
    if result == "decoder.conv1.weight" {
        return "decoder.conv_in.weight".to_string();
    }
    if result == "decoder.conv1.bias" {
        return "decoder.conv_in.bias".to_string();
    }

    result = result.replace("encoder.head.0.gamma", "encoder.norm_out.gamma");
    result = result.replace("encoder.head.2.weight", "encoder.conv_out.weight");
    result = result.replace("encoder.head.2.bias", "encoder.conv_out.bias");
    result = result.replace("decoder.head.0.gamma", "decoder.norm_out.gamma");
    result = result.replace("decoder.head.2.weight", "decoder.conv_out.weight");
    result = result.replace("decoder.head.2.bias", "decoder.conv_out.bias");

    if result.contains(".middle.") {
        result = result.replace(
            ".middle.0.residual.0.gamma",
            ".mid_block.resnets.0.norm1.gamma",
        );
        result = result.replace(
            ".middle.0.residual.2.weight",
            ".mid_block.resnets.0.conv1.weight",
        );
        result = result.replace(
            ".middle.0.residual.2.bias",
            ".mid_block.resnets.0.conv1.bias",
        );
        result = result.replace(
            ".middle.0.residual.3.gamma",
            ".mid_block.resnets.0.norm2.gamma",
        );
        result = result.replace(
            ".middle.0.residual.6.weight",
            ".mid_block.resnets.0.conv2.weight",
        );
        result = result.replace(
            ".middle.0.residual.6.bias",
            ".mid_block.resnets.0.conv2.bias",
        );

        result = result.replace(
            ".middle.2.residual.0.gamma",
            ".mid_block.resnets.1.norm1.gamma",
        );
        result = result.replace(
            ".middle.2.residual.2.weight",
            ".mid_block.resnets.1.conv1.weight",
        );
        result = result.replace(
            ".middle.2.residual.2.bias",
            ".mid_block.resnets.1.conv1.bias",
        );
        result = result.replace(
            ".middle.2.residual.3.gamma",
            ".mid_block.resnets.1.norm2.gamma",
        );
        result = result.replace(
            ".middle.2.residual.6.weight",
            ".mid_block.resnets.1.conv2.weight",
        );
        result = result.replace(
            ".middle.2.residual.6.bias",
            ".mid_block.resnets.1.conv2.bias",
        );

        result = result.replace(".middle.1.norm.gamma", ".mid_block.attentions.0.norm.gamma");
        result = result.replace(
            ".middle.1.to_qkv.weight",
            ".mid_block.attentions.0.to_qkv.weight",
        );
        result = result.replace(
            ".middle.1.to_qkv.bias",
            ".mid_block.attentions.0.to_qkv.bias",
        );
        result = result.replace(
            ".middle.1.proj.weight",
            ".mid_block.attentions.0.proj.weight",
        );
        result = result.replace(".middle.1.proj.bias", ".mid_block.attentions.0.proj.bias");
    }

    if result.contains("encoder.downsamples.") {
        result = result.replace("encoder.downsamples.", "encoder.down_blocks.");

        result = result.replace(".residual.0.gamma", ".norm1.gamma");
        result = result.replace(".residual.2.weight", ".conv1.weight");
        result = result.replace(".residual.2.bias", ".conv1.bias");
        result = result.replace(".residual.3.gamma", ".norm2.gamma");
        result = result.replace(".residual.6.weight", ".conv2.weight");
        result = result.replace(".residual.6.bias", ".conv2.bias");
        result = result.replace(".shortcut.weight", ".conv_shortcut.weight");
        result = result.replace(".shortcut.bias", ".conv_shortcut.bias");
    }

    if result.contains("decoder.upsamples.") {
        if let Some(idx_start) = result.find("decoder.upsamples.") {
            let after_prefix = &result[idx_start + "decoder.upsamples.".len()..];
            if let Some(dot_pos) = after_prefix.find('.')
                && let Ok(block_idx) = after_prefix[..dot_pos].parse::<usize>()
            {
                let rest = &after_prefix[dot_pos..];

                let (new_block_idx, resnet_idx) = match block_idx {
                    0 => (0, Some(0)),
                    1 => (0, Some(1)),
                    2 => (0, Some(2)),
                    3 => (0, None),
                    4 => (1, Some(0)),
                    5 => (1, Some(1)),
                    6 => (1, Some(2)),
                    7 => (1, None),
                    8 => (2, Some(0)),
                    9 => (2, Some(1)),
                    10 => (2, Some(2)),
                    11 => (2, None),
                    12 => (3, Some(0)),
                    13 => (3, Some(1)),
                    14 => (3, Some(2)),
                    _ => (block_idx, None),
                };

                if rest.contains(".residual.") {
                    if let Some(r_idx) = resnet_idx {
                        let mut new_rest = rest.to_string();
                        new_rest = new_rest.replace(".residual.0.gamma", ".norm1.gamma");
                        new_rest = new_rest.replace(".residual.2.weight", ".conv1.weight");
                        new_rest = new_rest.replace(".residual.2.bias", ".conv1.bias");
                        new_rest = new_rest.replace(".residual.3.gamma", ".norm2.gamma");
                        new_rest = new_rest.replace(".residual.6.weight", ".conv2.weight");
                        new_rest = new_rest.replace(".residual.6.bias", ".conv2.bias");
                        return format!(
                            "decoder.up_blocks.{}.resnets.{}{}",
                            new_block_idx, r_idx, new_rest
                        );
                    }
                } else if rest.contains(".shortcut.") {
                    if let Some(r_idx) = resnet_idx {
                        let new_rest = rest.replace(".shortcut.", ".conv_shortcut.");
                        return format!(
                            "decoder.up_blocks.{}.resnets.{}{}",
                            new_block_idx, r_idx, new_rest
                        );
                    }
                } else if rest.contains(".resample.") || rest.contains(".time_conv.") {
                    let new_rest = rest.to_string();
                    return format!(
                        "decoder.up_blocks.{}.upsamplers.0{}",
                        new_block_idx, new_rest
                    );
                }
            }
        }

        result = result.replace("decoder.upsamples.", "decoder.up_blocks.");
    }

    result
}

fn load_vae_with_conversion(
    path: &Path,
    config: AutoencoderKLWanConfig,
    device: &Device,
    dtype: DType,
) -> std::result::Result<AutoencoderKLWan, WanLoaderError> {
    let file_data = std::fs::read(path).map_err(|e| WanLoaderError::FileRead {
        path: path.display().to_string(),
        source: e,
    })?;

    let tensors = SafeTensors::deserialize(&file_data)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to parse safetensors: {}", e)))?;

    let keys: Vec<String> = tensors.names().into_iter().map(|s| s.to_string()).collect();

    if is_official_vae_format(&keys) {
        println!("  Detected official Wan VAE format, converting keys to diffusers format...");

        let original_tensors = candle_core::safetensors::load_buffer(&file_data, device)?;

        let mut converted_tensors: HashMap<String, Tensor> = HashMap::new();

        for (key, tensor) in original_tensors {
            let new_key = convert_vae_key_to_diffusers(&key);

            let tensor = if tensor.dtype() != dtype {
                tensor.to_dtype(dtype)?
            } else {
                tensor
            };

            converted_tensors.insert(new_key, tensor);
        }

        let vb = VarBuilder::from_tensors(converted_tensors, dtype, device);

        AutoencoderKLWan::new(config, vb).map_err(WanLoaderError::from)
    } else {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device)? };
        AutoencoderKLWan::new(config, vb).map_err(WanLoaderError::from)
    }
}

pub fn load_vae(
    weights_path: impl AsRef<Path>,
    config: AutoencoderKLWanConfig,
    device: &Device,
    dtype: DType,
) -> std::result::Result<AutoencoderKLWan, WanLoaderError> {
    let path = weights_path.as_ref();

    if !path.exists() {
        return Err(WanLoaderError::ModelNotFound {
            path: path.display().to_string(),
        });
    }

    load_vae_with_conversion(path, config, device, dtype)
}

pub fn load_vae_from_dir(
    dir: impl AsRef<Path>,
    config: AutoencoderKLWanConfig,
    device: &Device,
    dtype: DType,
) -> std::result::Result<AutoencoderKLWan, WanLoaderError> {
    let dir = dir.as_ref();

    let candidates = [
        "diffusion_pytorch_model.safetensors",
        "model.safetensors",
        "vae.safetensors",
    ];

    for candidate in &candidates {
        let path = dir.join(candidate);
        if path.exists() {
            return load_vae(&path, config, device, dtype);
        }
    }

    Err(WanLoaderError::ModelNotFound {
        path: dir.display().to_string(),
    })
}

pub fn load_transformer_config(
    path: impl AsRef<Path>,
) -> std::result::Result<WanTransformer3DConfig, WanLoaderError> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path).map_err(|e| WanLoaderError::FileRead {
        path: path.display().to_string(),
        source: e,
    })?;

    serde_json::from_str(&content).map_err(|e| WanLoaderError::JsonParse {
        path: path.display().to_string(),
        source: e,
    })
}

pub fn load_vae_config(
    path: impl AsRef<Path>,
) -> std::result::Result<AutoencoderKLWanConfig, WanLoaderError> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path).map_err(|e| WanLoaderError::FileRead {
        path: path.display().to_string(),
        source: e,
    })?;

    serde_json::from_str(&content).map_err(|e| WanLoaderError::JsonParse {
        path: path.display().to_string(),
        source: e,
    })
}

pub fn load_wan_t2v_1_3b_transformer(
    weights_path: impl AsRef<Path>,
    device: &Device,
    dtype: DType,
) -> std::result::Result<WanTransformer3DModel, WanLoaderError> {
    load_transformer(
        weights_path,
        WanTransformer3DConfig::wan_t2v_1_3b(),
        device,
        dtype,
    )
}

pub fn load_wan_t2v_14b_transformer(
    weights_path: impl AsRef<Path>,
    device: &Device,
    dtype: DType,
) -> std::result::Result<WanTransformer3DModel, WanLoaderError> {
    load_transformer(
        weights_path,
        WanTransformer3DConfig::wan_t2v_14b(),
        device,
        dtype,
    )
}

pub fn load_wan_2_1_vae(
    weights_path: impl AsRef<Path>,
    device: &Device,
    dtype: DType,
) -> std::result::Result<AutoencoderKLWan, WanLoaderError> {
    load_vae(
        weights_path,
        AutoencoderKLWanConfig::wan_2_1(),
        device,
        dtype,
    )
}

pub fn load_text_encoder(
    dir: impl AsRef<Path>,
    device: &Device,
    dtype: DType,
) -> std::result::Result<UMT5TextEncoder, WanLoaderError> {
    let dir = dir.as_ref();

    let candidates = [
        "model.safetensors",
        "diffusion_pytorch_model.safetensors",
        "pytorch_model.safetensors",
    ];

    for candidate in &candidates {
        let path = dir.join(candidate);
        if path.exists() {
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&path], dtype, device)? };
            let encoder = UMT5TextEncoder::new_umt5_xxl(vb)?;
            return Ok(encoder);
        }
    }

    Err(WanLoaderError::ModelNotFound {
        path: dir.display().to_string(),
    })
}

pub fn load_text_encoder_from_file(
    path: impl AsRef<Path>,
    device: &Device,
    dtype: DType,
) -> std::result::Result<UMT5TextEncoder, WanLoaderError> {
    let path = path.as_ref();

    if !path.exists() {
        return Err(WanLoaderError::ModelNotFound {
            path: path.display().to_string(),
        });
    }

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device)? };
    let encoder = UMT5TextEncoder::new_umt5_xxl(vb)?;
    Ok(encoder)
}

pub fn load_quantized_text_encoder(
    gguf_path: impl AsRef<Path>,
    device: &Device,
) -> std::result::Result<QuantizedUMT5Encoder, WanLoaderError> {
    let path = gguf_path.as_ref();

    if !path.exists() {
        return Err(WanLoaderError::ModelNotFound {
            path: path.display().to_string(),
        });
    }

    let encoder = QuantizedUMT5Encoder::load(path, device)?;
    Ok(encoder)
}

pub fn load_tokenizer(
    path: impl AsRef<Path>,
    max_length: Option<usize>,
) -> std::result::Result<UMT5Tokenizer, WanLoaderError> {
    let path = path.as_ref();

    if !path.exists() {
        return Err(WanLoaderError::ModelNotFound {
            path: path.display().to_string(),
        });
    }

    UMT5Tokenizer::load(path, max_length.unwrap_or(512)).map_err(|e| WanLoaderError::FileRead {
        path: path.display().to_string(),
        source: std::io::Error::new(std::io::ErrorKind::InvalidData, e),
    })
}

pub fn load_tokenizer_from_dir(
    dir: impl AsRef<Path>,
    max_length: Option<usize>,
) -> std::result::Result<UMT5Tokenizer, WanLoaderError> {
    let dir = dir.as_ref();
    let path = dir.join("tokenizer.json");

    if !path.exists() {
        return Err(WanLoaderError::ModelNotFound {
            path: path.display().to_string(),
        });
    }

    load_tokenizer(&path, max_length)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_presets() {
        let t_cfg = WanTransformer3DConfig::wan_t2v_1_3b();
        assert_eq!(t_cfg.num_layers, 30);
        assert_eq!(t_cfg.num_attention_heads, 12);
        assert_eq!(t_cfg.inner_dim(), 1536);

        let v_cfg = AutoencoderKLWanConfig::wan_2_1();
        assert_eq!(v_cfg.z_dim, 16);
    }

    #[test]
    fn test_umt5_config() {
        let config = T5EncoderConfig::umt5_xxl();
        assert_eq!(config.vocab_size, 256384);
        assert_eq!(config.d_model, 4096);
        assert_eq!(config.max_seq_len, 512);
    }
}
