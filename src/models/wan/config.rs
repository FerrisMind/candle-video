//! Configuration for Wan T2V models.
//!
//! Preset configurations for Wan2.1-T2V-1.3B and Wan2.1-T2V-14B models.

use serde::{Deserialize, Serialize};

// =============================================================================
// WanTransformer3DConfig
// =============================================================================

/// Configuration for WanTransformer3DModel.
///
/// Matches diffusers WanTransformer3DModel config.json structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WanTransformer3DConfig {
    /// 3D patch dimensions (t, h, w)
    pub patch_size: (usize, usize, usize),
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Dimension per attention head
    pub attention_head_dim: usize,
    /// Input channels (latent dim)
    pub in_channels: usize,
    /// Output channels
    pub out_channels: usize,
    /// Text embedding dimension (UMT5: 4096)
    pub text_dim: usize,
    /// Timestep frequency dimension
    pub freq_dim: usize,
    /// Feed-forward hidden dimension
    pub ffn_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Enable cross-attention normalization
    pub cross_attn_norm: bool,
    /// Query/Key normalization mode
    pub qk_norm: Option<String>,
    /// Epsilon for layer norms
    pub eps: f64,
    /// Image embedding dimension (for I2V, None for T2V)
    pub image_dim: Option<usize>,
    /// Added KV projection dimension
    pub added_kv_proj_dim: Option<usize>,
    /// Max sequence length for RoPE
    pub rope_max_seq_len: usize,
}

impl Default for WanTransformer3DConfig {
    fn default() -> Self {
        Self::wan_t2v_1_3b()
    }
}

impl WanTransformer3DConfig {
    /// Wan2.1-T2V-1.3B configuration (480p/720p).
    ///
    /// - 1.3B parameters
    /// - 20 layers, 20 heads × 128 dim = 2560 inner_dim
    pub fn wan_t2v_1_3b() -> Self {
        Self {
            patch_size: (1, 2, 2),
            num_attention_heads: 20,
            attention_head_dim: 128,
            in_channels: 16,
            out_channels: 16,
            text_dim: 4096,
            freq_dim: 256,
            ffn_dim: 8960,  // ~3.5 * inner_dim
            num_layers: 20,
            cross_attn_norm: true,
            qk_norm: Some("rms_norm_across_heads".to_string()),
            eps: 1e-6,
            image_dim: None,
            added_kv_proj_dim: None,
            rope_max_seq_len: 1024,
        }
    }

    /// Wan2.1-T2V-14B configuration.
    ///
    /// - 14B parameters
    /// - 40 layers, 40 heads × 128 dim = 5120 inner_dim
    pub fn wan_t2v_14b() -> Self {
        Self {
            patch_size: (1, 2, 2),
            num_attention_heads: 40,
            attention_head_dim: 128,
            in_channels: 16,
            out_channels: 16,
            text_dim: 4096,
            freq_dim: 256,
            ffn_dim: 13824,  // ~2.7 * inner_dim
            num_layers: 40,
            cross_attn_norm: true,
            qk_norm: Some("rms_norm_across_heads".to_string()),
            eps: 1e-6,
            image_dim: None,
            added_kv_proj_dim: None,
            rope_max_seq_len: 1024,
        }
    }

    /// Get inner dimension (num_heads × head_dim).
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }
}

// =============================================================================
// AutoencoderKLWanConfig
// =============================================================================

/// Configuration for AutoencoderKLWan (Wan VAE).
///
/// Matches diffusers AutoencoderKLWan config.json structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoencoderKLWanConfig {
    /// Base channel dimension
    pub base_dim: usize,
    /// Decoder base dimension (None = same as base_dim)
    pub decoder_base_dim: Option<usize>,
    /// Latent channel dimension
    pub z_dim: usize,
    /// Channel multipliers per stage
    pub dim_mult: Vec<usize>,
    /// Number of residual blocks per stage
    pub num_res_blocks: usize,
    /// Attention scales (empty for no attention)
    pub attn_scales: Vec<f32>,
    /// Temporal downsampling flags per stage
    pub temporal_downsample: Vec<bool>,
    /// Dropout rate
    pub dropout: f64,
    /// Input channels (3 for RGB)
    pub in_channels: usize,
    /// Output channels (3 for RGB)
    pub out_channels: usize,
    /// Use residual architecture (Wan 2.2)
    pub is_residual: bool,
    /// Spatial compression factor
    pub scale_factor_spatial: usize,
    /// Temporal compression factor
    pub scale_factor_temporal: usize,
    /// Latent mean for normalization (16 channels)
    pub latents_mean: Vec<f32>,
    /// Latent std for normalization (16 channels)
    pub latents_std: Vec<f32>,
}

impl Default for AutoencoderKLWanConfig {
    fn default() -> Self {
        Self::wan_2_1()
    }
}

impl AutoencoderKLWanConfig {
    /// Wan 2.1 VAE configuration.
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
                -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
            ],
            latents_std: vec![
                2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
            ],
        }
    }

    /// Wan 2.2 VAE configuration (residual architecture).
    pub fn wan_2_2() -> Self {
        Self {
            is_residual: true,
            ..Self::wan_2_1()
        }
    }

    /// Get decoder base dimension.
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
        assert_eq!(cfg.num_layers, 20);
        assert_eq!(cfg.inner_dim(), 2560);
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
