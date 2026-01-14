use regex::Regex;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightFormat {
    Diffusers,

    Official,
}

pub fn detect_format(path: &Path) -> WeightFormat {
    if path.is_file() {
        WeightFormat::Official
    } else {
        WeightFormat::Diffusers
    }
}

#[derive(Debug, Clone)]
pub struct KeyRemapper {
    encoder_block_re: Regex,
    decoder_block_re: Regex,
}

impl Default for KeyRemapper {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyRemapper {
    pub fn new() -> Self {
        Self {
            encoder_block_re: Regex::new(r"encoder\.down_blocks\.(\d+)").unwrap(),
            decoder_block_re: Regex::new(r"decoder\.up_blocks\.(\d+)").unwrap(),
        }
    }

    pub fn remap_key(&self, key: &str) -> String {
        let mut result = key.to_string();

        result = result.replace("patchify_proj", "proj_in");
        result = result.replace("adaln_single", "time_embed");
        result = result.replace("q_norm", "norm_q");
        result = result.replace("k_norm", "norm_k");

        result = result.replace("res_blocks", "resnets");

        result = self.remap_encoder_blocks_095(&result);

        result = self.remap_decoder_blocks_095(&result);

        result = result.replace("last_time_embedder", "time_embedder");
        result = result.replace("last_scale_shift_table", "scale_shift_table");
        result = result.replace("norm3.norm", "norm3");
        result = result.replace("per_channel_statistics.mean-of-means", "latents_mean");
        result = result.replace("per_channel_statistics.std-of-means", "latents_std");

        result
    }

    fn remap_encoder_blocks_095(&self, key: &str) -> String {
        self.encoder_block_re
            .replace_all(key, |caps: &regex::Captures| {
                let native_idx: usize = caps[1].parse().unwrap_or(0);
                match native_idx {
                    0 => "encoder.down_blocks.0".to_string(),
                    1 => "encoder.down_blocks.0.downsamplers.0".to_string(),
                    2 => "encoder.down_blocks.1".to_string(),
                    3 => "encoder.down_blocks.1.downsamplers.0".to_string(),
                    4 => "encoder.down_blocks.2".to_string(),
                    5 => "encoder.down_blocks.2.downsamplers.0".to_string(),
                    6 => "encoder.down_blocks.3".to_string(),
                    7 => "encoder.down_blocks.3.downsamplers.0".to_string(),
                    8 => "encoder.mid_block".to_string(),
                    _ => format!("encoder.down_blocks.{}", native_idx),
                }
            })
            .to_string()
    }

    fn remap_decoder_blocks_095(&self, key: &str) -> String {
        self.decoder_block_re
            .replace_all(key, |caps: &regex::Captures| {
                let native_idx: usize = caps[1].parse().unwrap_or(0);
                match native_idx {
                    0 => "decoder.mid_block".to_string(),
                    1 => "decoder.up_blocks.0.upsamplers.0".to_string(),
                    2 => "decoder.up_blocks.0".to_string(),
                    3 => "decoder.up_blocks.1.upsamplers.0".to_string(),
                    4 => "decoder.up_blocks.1".to_string(),
                    5 => "decoder.up_blocks.2.upsamplers.0".to_string(),
                    6 => "decoder.up_blocks.2".to_string(),
                    7 => "decoder.up_blocks.3.upsamplers.0".to_string(),
                    8 => "decoder.up_blocks.3".to_string(),
                    _ => format!("decoder.up_blocks.{}", native_idx),
                }
            })
            .to_string()
    }

    pub fn is_transformer_key(key: &str) -> bool {
        key.starts_with("transformer.")
            || key.starts_with("model.diffusion_model.")
            || key.contains("transformer_blocks")
            || key.contains("patchify_proj")
            || key.contains("proj_in")
            || key.contains("adaln_single")
            || key.contains("time_embed")
    }

    pub fn is_vae_key(key: &str) -> bool {
        key.starts_with("vae.")
            || key.starts_with("encoder.")
            || key.starts_with("decoder.")
            || key.contains("per_channel_statistics")
            || key.contains("latents_mean")
            || key.contains("latents_std")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remap_transformer_key() {
        let remapper = KeyRemapper::new();
        assert_eq!(
            remapper.remap_key("transformer.patchify_proj.weight"),
            "transformer.proj_in.weight"
        );
        assert_eq!(
            remapper.remap_key("transformer.adaln_single.linear.weight"),
            "transformer.time_embed.linear.weight"
        );
    }

    #[test]
    fn test_remap_encoder_blocks_095() {
        let remapper = KeyRemapper::new();

        assert_eq!(
            remapper.remap_key("encoder.down_blocks.0.res_blocks.0.conv1.weight"),
            "encoder.down_blocks.0.resnets.0.conv1.weight"
        );

        assert_eq!(
            remapper.remap_key("encoder.down_blocks.1.conv.weight"),
            "encoder.down_blocks.0.downsamplers.0.conv.weight"
        );

        assert_eq!(
            remapper.remap_key("encoder.down_blocks.2.res_blocks.0.conv1.weight"),
            "encoder.down_blocks.1.resnets.0.conv1.weight"
        );

        assert_eq!(
            remapper.remap_key("encoder.down_blocks.6.res_blocks.0.weight"),
            "encoder.down_blocks.3.resnets.0.weight"
        );

        assert_eq!(
            remapper.remap_key("encoder.down_blocks.8.res_blocks.0.weight"),
            "encoder.mid_block.resnets.0.weight"
        );
    }

    #[test]
    fn test_remap_decoder_blocks_095() {
        let remapper = KeyRemapper::new();

        assert_eq!(
            remapper.remap_key("decoder.up_blocks.0.res_blocks.0.weight"),
            "decoder.mid_block.resnets.0.weight"
        );

        assert_eq!(
            remapper.remap_key("decoder.up_blocks.1.conv.weight"),
            "decoder.up_blocks.0.upsamplers.0.conv.weight"
        );

        assert_eq!(
            remapper.remap_key("decoder.up_blocks.2.res_blocks.0.weight"),
            "decoder.up_blocks.0.resnets.0.weight"
        );

        assert_eq!(
            remapper.remap_key("decoder.up_blocks.8.res_blocks.0.weight"),
            "decoder.up_blocks.3.resnets.0.weight"
        );
    }

    #[test]
    fn test_remap_time_embedder() {
        let remapper = KeyRemapper::new();
        assert_eq!(
            remapper.remap_key("decoder.last_time_embedder.weight"),
            "decoder.time_embedder.weight"
        );
    }

    #[test]
    fn test_remap_latents_stats() {
        let remapper = KeyRemapper::new();
        assert_eq!(
            remapper.remap_key("per_channel_statistics.mean-of-means"),
            "latents_mean"
        );
        assert_eq!(
            remapper.remap_key("per_channel_statistics.std-of-means"),
            "latents_std"
        );
    }
}
