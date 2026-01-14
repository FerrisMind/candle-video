use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WanTransformer3DConfig {
    pub patch_size: (usize, usize, usize),

    pub num_attention_heads: usize,

    pub attention_head_dim: usize,

    pub in_channels: usize,

    pub out_channels: usize,

    pub text_dim: usize,

    pub freq_dim: usize,

    pub ffn_dim: usize,

    pub num_layers: usize,

    pub cross_attn_norm: bool,

    pub qk_norm: Option<String>,

    pub eps: f64,

    pub image_dim: Option<usize>,

    pub added_kv_proj_dim: Option<usize>,

    pub rope_max_seq_len: usize,
}

impl Default for WanTransformer3DConfig {
    fn default() -> Self {
        Self::wan_t2v_1_3b()
    }
}

impl WanTransformer3DConfig {
    pub fn wan_t2v_1_3b() -> Self {
        Self {
            patch_size: (1, 2, 2),
            num_attention_heads: 12,
            attention_head_dim: 128,
            in_channels: 16,
            out_channels: 16,
            text_dim: 4096,
            freq_dim: 256,
            ffn_dim: 8960,
            num_layers: 30,
            cross_attn_norm: true,
            qk_norm: Some("rms_norm_across_heads".to_string()),
            eps: 1e-6,
            image_dim: None,
            added_kv_proj_dim: None,
            rope_max_seq_len: 1024,
        }
    }

    pub fn wan_t2v_14b() -> Self {
        Self {
            patch_size: (1, 2, 2),
            num_attention_heads: 40,
            attention_head_dim: 128,
            in_channels: 16,
            out_channels: 16,
            text_dim: 4096,
            freq_dim: 256,
            ffn_dim: 13824,
            num_layers: 40,
            cross_attn_norm: true,
            qk_norm: Some("rms_norm_across_heads".to_string()),
            eps: 1e-6,
            image_dim: None,
            added_kv_proj_dim: None,
            rope_max_seq_len: 1024,
        }
    }

    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoencoderKLWanConfig {
    pub base_dim: usize,

    pub decoder_base_dim: Option<usize>,

    pub z_dim: usize,

    pub dim_mult: Vec<usize>,

    pub num_res_blocks: usize,

    pub attn_scales: Vec<f32>,

    pub temporal_downsample: Vec<bool>,

    pub dropout: f64,

    pub in_channels: usize,

    pub out_channels: usize,

    pub is_residual: bool,

    pub scale_factor_spatial: usize,

    pub scale_factor_temporal: usize,

    pub latents_mean: Vec<f32>,

    pub latents_std: Vec<f32>,
}

impl Default for AutoencoderKLWanConfig {
    fn default() -> Self {
        Self::wan_2_1()
    }
}

impl AutoencoderKLWanConfig {
    pub fn wan_2_1() -> Self {
        Self {
            base_dim: 96,
            decoder_base_dim: None,
            z_dim: 16,
            dim_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            attn_scales: vec![],
            temporal_downsample: vec![false, true, true],
            dropout: 0.0,
            in_channels: 3,
            out_channels: 3,
            is_residual: false,
            scale_factor_spatial: 8,
            scale_factor_temporal: 4,
            latents_mean: vec![
                -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134,
                -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
            ],
            latents_std: vec![
                2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526,
                2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
            ],
        }
    }

    pub fn wan_2_2() -> Self {
        Self {
            is_residual: true,
            ..Self::wan_2_1()
        }
    }

    pub fn get_decoder_base_dim(&self) -> usize {
        self.decoder_base_dim.unwrap_or(self.base_dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wan_t2v_1_3b_config() {
        let cfg = WanTransformer3DConfig::wan_t2v_1_3b();
        assert_eq!(cfg.num_layers, 30);
        assert_eq!(cfg.num_attention_heads, 12);
        assert_eq!(cfg.inner_dim(), 1536);
    }

    #[test]
    fn test_wan_t2v_14b_config() {
        let cfg = WanTransformer3DConfig::wan_t2v_14b();
        assert_eq!(cfg.num_layers, 40);
        assert_eq!(cfg.inner_dim(), 5120);
    }

    #[test]
    fn test_wan_vae_config() {
        let cfg = AutoencoderKLWanConfig::wan_2_1();
        assert_eq!(cfg.z_dim, 16);
        assert_eq!(cfg.latents_mean.len(), 16);
        assert_eq!(cfg.latents_std.len(), 16);
    }
}
