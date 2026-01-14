use candle_core::{DType, Device, Result};
use candle_nn::VarBuilder;
use std::path::Path;

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

pub struct UNetKeyMapper;

impl UNetKeyMapper {
    pub fn map_key(hf_key: &str) -> String {
        let key = hf_key.strip_prefix("unet.").unwrap_or(hf_key);

        key.replace("to_out.0.weight", "to_out.weight")
            .replace("to_out.0.bias", "to_out.bias")
    }
}

pub struct VaeKeyMapper;

impl VaeKeyMapper {
    pub fn map_key(hf_key: &str) -> String {
        let key = hf_key.strip_prefix("vae.").unwrap_or(hf_key);

        key.to_string()
    }
}

pub struct ClipKeyMapper;

impl ClipKeyMapper {
    pub fn map_key(hf_key: &str) -> String {
        let key = hf_key.strip_prefix("image_encoder.").unwrap_or(hf_key);

        key.to_string()
    }
}

pub fn validate_weights(vb: &VarBuilder, expected_keys: &[&str]) -> Result<()> {
    let _ = (vb, expected_keys);
    Ok(())
}

pub fn list_tensor_names<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
    let data = std::fs::read(path.as_ref())
        .map_err(|e| candle_core::Error::Msg(format!("Failed to read file: {}", e)))?;

    let tensors = safetensors::SafeTensors::deserialize(&data)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to parse safetensors: {}", e)))?;

    Ok(tensors.names().into_iter().map(|s| s.to_string()).collect())
}

pub struct WeightLoader<'a> {
    vb: VarBuilder<'a>,
}

impl<'a> WeightLoader<'a> {
    pub fn new(vb: VarBuilder<'a>) -> Self {
        Self { vb }
    }

    pub fn unet(&self) -> VarBuilder<'a> {
        self.vb.pp("unet")
    }

    pub fn vae(&self) -> VarBuilder<'a> {
        self.vb.pp("vae")
    }

    pub fn image_encoder(&self) -> VarBuilder<'a> {
        self.vb.pp("image_encoder")
    }
}

#[derive(Debug, Default)]
pub struct WeightStats {
    pub total_parameters: usize,
    pub total_bytes: usize,
    pub tensor_count: usize,
}

impl WeightStats {
    pub fn from_tensor_names(names: &[String], dtype: DType) -> Self {
        let tensor_count = names.len();

        let _bytes_per_param = match dtype {
            DType::F16 | DType::BF16 => 2,
            DType::F32 => 4,
            DType::F64 => 8,
            _ => 4,
        };

        Self {
            total_parameters: 0,
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
