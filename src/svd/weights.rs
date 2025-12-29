//! Weight Loading for SVD Models
//!
//! Utilities for loading weights from safetensors files and mapping HuggingFace keys to Candle.

use candle_core::{DType, Device, Result};
use candle_nn::VarBuilder;
use std::path::Path;

/// Load weights from a safetensors file
pub fn load_safetensors<P: AsRef<Path>>(
    path: P,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let path = path.as_ref();

    if !path.exists() {
        return Err(candle_core::Error::Msg(format!(
            "Weights file not found: {}",
            path.display()
        )));
    }

    unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device) }
}

/// Load weights from multiple safetensors files (sharded model)
pub fn load_sharded_safetensors<P: AsRef<Path>>(
    paths: &[P],
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let paths: Vec<_> = paths.iter().map(|p| p.as_ref()).collect();

    for path in &paths {
        if !path.exists() {
            return Err(candle_core::Error::Msg(format!(
                "Weights file not found: {}",
                path.display()
            )));
        }
    }

    unsafe { VarBuilder::from_mmaped_safetensors(&paths, dtype, device) }
}

/// Key mapping for UNet weights (HuggingFace -> Candle)
pub struct UNetKeyMapper;

impl UNetKeyMapper {
    /// Map HuggingFace key to Candle path
    pub fn map_key(hf_key: &str) -> String {
        // Most keys map directly, with some transformations
        let key = hf_key
            // Remove model prefix if present
            .strip_prefix("unet.")
            .unwrap_or(hf_key);

        // Convert to Candle conventions - only actual transformations
        key.replace("to_out.0.weight", "to_out.weight")
            .replace("to_out.0.bias", "to_out.bias")
    }
}

/// Key mapping for VAE weights
pub struct VaeKeyMapper;

impl VaeKeyMapper {
    /// Map HuggingFace key to Candle path
    pub fn map_key(hf_key: &str) -> String {
        let key = hf_key.strip_prefix("vae.").unwrap_or(hf_key);

        key.to_string()
    }
}

/// Key mapping for CLIP weights
pub struct ClipKeyMapper;

impl ClipKeyMapper {
    /// Map HuggingFace key to Candle path
    pub fn map_key(hf_key: &str) -> String {
        let key = hf_key.strip_prefix("image_encoder.").unwrap_or(hf_key);

        key.to_string()
    }
}

/// Validate that all expected keys are present in the weights
pub fn validate_weights(vb: &VarBuilder, expected_keys: &[&str]) -> Result<()> {
    // This is a placeholder - actual implementation would check tensor names
    // For now, we trust the VarBuilder to handle missing keys
    let _ = (vb, expected_keys);
    Ok(())
}

/// Get a list of all tensor names in a safetensors file
pub fn list_tensor_names<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
    let data = std::fs::read(path.as_ref())
        .map_err(|e| candle_core::Error::Msg(format!("Failed to read file: {}", e)))?;

    let tensors = safetensors::SafeTensors::deserialize(&data)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to parse safetensors: {}", e)))?;

    Ok(tensors.names().into_iter().map(|s| s.to_string()).collect())
}

/// Weight loader with path prefix support
pub struct WeightLoader<'a> {
    vb: VarBuilder<'a>,
}

impl<'a> WeightLoader<'a> {
    pub fn new(vb: VarBuilder<'a>) -> Self {
        Self { vb }
    }

    /// Get VarBuilder for a specific component
    pub fn unet(&self) -> VarBuilder<'a> {
        self.vb.pp("unet")
    }

    /// Get VarBuilder for VAE
    pub fn vae(&self) -> VarBuilder<'a> {
        self.vb.pp("vae")
    }

    /// Get VarBuilder for image encoder
    pub fn image_encoder(&self) -> VarBuilder<'a> {
        self.vb.pp("image_encoder")
    }
}

/// Statistics about loaded weights
#[derive(Debug, Default)]
pub struct WeightStats {
    pub total_parameters: usize,
    pub total_bytes: usize,
    pub tensor_count: usize,
}

impl WeightStats {
    pub fn from_tensor_names(names: &[String], dtype: DType) -> Self {
        let tensor_count = names.len();
        // Rough estimate - actual sizes would need tensor shapes
        let _bytes_per_param = match dtype {
            DType::F16 | DType::BF16 => 2,
            DType::F32 => 4,
            DType::F64 => 8,
            _ => 4,
        };

        Self {
            total_parameters: 0, // Would need tensor shapes
            total_bytes: 0,
            tensor_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unet_key_mapping() {
        let key = "unet.down_blocks.0.resnets.0.norm1.weight";
        let mapped = UNetKeyMapper::map_key(key);
        assert_eq!(mapped, "down_blocks.0.resnets.0.norm1.weight");
    }

    #[test]
    fn test_vae_key_mapping() {
        let key = "vae.encoder.down_blocks.0.resnets.0.conv1.weight";
        let mapped = VaeKeyMapper::map_key(key);
        assert_eq!(mapped, "encoder.down_blocks.0.resnets.0.conv1.weight");
    }

    #[test]
    fn test_clip_key_mapping() {
        let key = "image_encoder.vision_model.embeddings.patch_embedding.weight";
        let mapped = ClipKeyMapper::map_key(key);
        assert_eq!(mapped, "vision_model.embeddings.patch_embedding.weight");
    }
}
