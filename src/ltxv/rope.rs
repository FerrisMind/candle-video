//! Normalized Fractional RoPE (Rotary Position Embedding) implementation
//!
//! LTX-Video uses a 3D fractional RoPE where positions are normalized
//! to [0, 1] for temporal (T), height (H), and width (W) dimensions.
//! The head dimension is split into 3 parts for each coordinate type.

use candle_core::{D, DType, Device, Result, Tensor};
use std::f64::consts::PI;

pub const DIFFUSERS_BASE_NUM_FRAMES: usize = 20;
pub const DIFFUSERS_BASE_HEIGHT: usize = 2048;
pub const DIFFUSERS_BASE_WIDTH: usize = 2048;
pub const DIFFUSERS_THETA: f64 = 10000.0;
pub const DEFAULT_FRAME_RATE: f64 = 25.0;
pub const VAE_TEMPORAL_COMPRESSION: f64 = 8.0;
pub const VAE_SPATIAL_COMPRESSION: f64 = 32.0;

/// Normalized Fractional RoPE for 3D spatiotemporal coordinates
pub struct FractionalRoPE {
    /// Base frequency for rotary embeddings
    base: f64,
    /// Head dimension
    head_dim: usize,
    /// Maximum sequence length (for normalization)
    #[allow(dead_code)]
    max_seq_len: usize,
}

impl FractionalRoPE {
    /// Create a new FractionalRoPE
    pub fn new(head_dim: usize, base: f64, max_seq_len: usize) -> Self {
        Self {
            base,
            head_dim,
            max_seq_len,
        }
    }

    /// Compute RoPE frequencies from indices grid
    ///
    /// # Arguments
    /// * `indices_grid` - Grid of normalized positions (B, 3, L) where 3 = (t, h, w)
    /// * `device` - Device to create tensors on
    ///
    /// # Returns
    /// Tuple of (cos_freqs, sin_freqs) each of shape (B, L, head_dim)
    pub fn compute_freqs_cis(
        &self,
        indices_grid: &Tensor,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let batch = indices_grid.dim(0)?;
        let seq_len = indices_grid.dim(2)?;
        let dim = self.head_dim;
        let dtype = indices_grid.dtype();

        // Get fractional positions: (B, 3, L) -> (B, L, 3)
        let fractional_positions = indices_grid.permute((0, 2, 1))?;

        // Generate frequency indices using exponential spacing
        // indices = theta^(linspace(log_theta(1), log_theta(theta), dim/6))
        let freq_dim = dim / 6;
        let theta = self.base;

        let indices: Vec<f32> = (0..freq_dim)
            .map(|i| {
                let t = i as f64 / (freq_dim - 1).max(1) as f64;
                let log_start = 1.0f64.log(theta);
                let log_end = theta.log(theta);
                let val = theta.powf(log_start + t * (log_end - log_start));
                (val * PI / 2.0) as f32
            })
            .collect();

        let indices = Tensor::from_vec(indices, freq_dim, device)?.to_dtype(dtype)?;

        // Compute frequencies for each position
        // freqs = (indices * (fractional_positions * 2 - 1)).transpose(-1, -2).flatten(2)
        // Shape: (B, L, 3) @ (freq_dim,) -> (B, L, 3, freq_dim)

        // fractional_positions * 2 - 1 to map [0, 1] to [-1, 1]
        let scaled_positions = fractional_positions.affine(2.0, -1.0)?;

        // (B, L, 3) -> (B, L, 3, 1) * (freq_dim,) -> (B, L, 3, freq_dim)
        let scaled_positions = scaled_positions.unsqueeze(D::Minus1)?;
        let indices_expanded = indices.reshape((1, 1, 1, freq_dim))?;
        let freqs = scaled_positions.broadcast_mul(&indices_expanded)?;

        // Transpose and flatten: (B, L, 3, freq_dim) -> (B, L, freq_dim, 3) -> (B, L, 3*freq_dim)
        // This matches diffusers: freqs.transpose(-1, -2).flatten(2)
        let freqs = freqs.transpose(2, 3)?;
        let freqs = freqs.flatten_from(2)?;

        // Compute cos and sin, then repeat_interleave(2) to match head_dim
        let cos_freq = freqs.cos()?;
        let sin_freq = freqs.sin()?;

        // Repeat interleave by 2: [a, b, c] -> [a, a, b, b, c, c]
        let cos_freq = repeat_interleave_2(&cos_freq)?;
        let sin_freq = repeat_interleave_2(&sin_freq)?;

        // Handle padding if dim % 6 != 0
        let current_dim = 3 * freq_dim * 2;
        let (cos_freq, sin_freq) = if current_dim < dim {
            let pad_size = dim - current_dim;
            let cos_pad = Tensor::ones((batch, seq_len, pad_size), dtype, device)?;
            let sin_pad = Tensor::zeros((batch, seq_len, pad_size), dtype, device)?;
            (
                Tensor::cat(&[cos_pad, cos_freq], D::Minus1)?,
                Tensor::cat(&[sin_pad, sin_freq], D::Minus1)?,
            )
        } else {
            (cos_freq, sin_freq)
        };

        Ok((cos_freq, sin_freq))
    }

    /// Apply rotary embeddings to query and key tensors (legacy 3D API)
    ///
    /// # Arguments
    /// * `q` - Query tensor of shape (B, H, L, D) where D is head_dim
    /// * `k` - Key tensor of shape (B, H, L, D)
    /// * `t_coords` - Normalized temporal coordinates [0, 1] of shape (L,)
    /// * `h_coords` - Normalized height coordinates [0, 1] of shape (L,)
    /// * `w_coords` - Normalized width coordinates [0, 1] of shape (L,)
    #[allow(clippy::too_many_arguments)]
    pub fn apply_3d(
        &self,
        q: &Tensor,
        k: &Tensor,
        t_coords: &Tensor,
        h_coords: &Tensor,
        w_coords: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let device = q.device();
        let batch = q.dim(0)?;
        let seq_len = q.dim(2)?;

        // Generate full position grid
        let mut t_indices = Vec::with_capacity(seq_len);
        let mut h_indices = Vec::with_capacity(seq_len);
        let mut w_indices = Vec::with_capacity(seq_len);

        let t_vec = t_coords.to_vec1::<f32>()?;
        let h_vec = h_coords.to_vec1::<f32>()?;
        let w_vec = w_coords.to_vec1::<f32>()?;

        for t_val in &t_vec {
            for h_val in &h_vec {
                for w_val in &w_vec {
                    t_indices.push(*t_val);
                    h_indices.push(*h_val);
                    w_indices.push(*w_val);
                }
            }
        }

        // Create indices_grid: (1, 3, seq_len)
        let t_tensor = Tensor::from_vec(t_indices, seq_len, device)?;
        let h_tensor = Tensor::from_vec(h_indices, seq_len, device)?;
        let w_tensor = Tensor::from_vec(w_indices, seq_len, device)?;

        let indices_grid = Tensor::stack(&[t_tensor, h_tensor, w_tensor], 0)?.unsqueeze(0)?;

        // Expand to batch size
        let indices_grid = indices_grid.broadcast_as((batch, 3, seq_len))?;

        // Compute frequencies
        let freqs_cis = self.compute_freqs_cis(&indices_grid.contiguous()?, device)?;

        // Apply rotation
        let q_rotated = self.apply_rotary_emb(q, &freqs_cis)?;
        let k_rotated = self.apply_rotary_emb(k, &freqs_cis)?;

        Ok((q_rotated, k_rotated))
    }

    /// Apply rotary embedding to a tensor
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape (B, H, L, D)
    /// * `freqs_cis` - Tuple of (cos, sin) each of shape (B, L, D)
    pub fn apply_rotary_emb(&self, x: &Tensor, freqs_cis: &(Tensor, Tensor)) -> Result<Tensor> {
        let (cos_freqs, sin_freqs) = freqs_cis;

        // x: (B, H, L, D)
        // freqs: (B, L, D)
        let x_shape = x.dims();
        let batch = x_shape[0];
        let heads = x_shape[1];
        let seq_len = x_shape[2];
        let head_dim = x_shape[3];

        // Reshape x for rotation: pair up consecutive elements
        // (B, H, L, D) -> (B, H, L, D/2, 2)
        let x_reshaped = x.reshape((batch, heads, seq_len, head_dim / 2, 2))?;

        // Split into x1, x2
        let x1 = x_reshaped.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
        let x2 = x_reshaped.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;

        // Rotated version: (-x2, x1)
        let x_rot = Tensor::stack(&[x2.neg()?, x1], D::Minus1)?;
        let x_rot = x_rot.reshape((batch, heads, seq_len, head_dim))?;

        // Flatten x back
        let x = x_reshaped.reshape((batch, heads, seq_len, head_dim))?;

        // Expand freqs to match attention shape: (B, 1, L, D)
        let cos_expanded = cos_freqs.unsqueeze(1)?;
        let sin_expanded = sin_freqs.unsqueeze(1)?;

        // Apply rotation: x * cos + x_rot * sin
        x.broadcast_mul(&cos_expanded)?
            .add(&x_rot.broadcast_mul(&sin_expanded)?)
    }
}

/// Repeat interleave by factor of 2: [a, b, c] -> [a, a, b, b, c, c]
fn repeat_interleave_2(x: &Tensor) -> Result<Tensor> {
    let shape = x.dims();
    let _last_dim = shape[shape.len() - 1];

    // (B, L, D) -> (B, L, D, 1) -> (B, L, D, 2) -> (B, L, D*2)
    let x = x.unsqueeze(D::Minus1)?;
    let x = Tensor::cat(&[x.clone(), x], D::Minus1)?;
    x.flatten_from(D::Minus2)
}

/// Generate normalized coordinate grids for 3D spatiotemporal data
pub fn generate_coord_grids(
    t_len: usize,
    h_len: usize,
    w_len: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    // Generate normalized coordinates [0, 1]
    let t_coords = if t_len > 1 {
        Tensor::arange(0u32, t_len as u32, device)?
            .to_dtype(DType::F32)?
            .broadcast_div(&Tensor::new(&[(t_len - 1) as f32], device)?)?
    } else {
        Tensor::zeros(1, DType::F32, device)?
    };

    let h_coords = if h_len > 1 {
        Tensor::arange(0u32, h_len as u32, device)?
            .to_dtype(DType::F32)?
            .broadcast_div(&Tensor::new(&[(h_len - 1) as f32], device)?)?
    } else {
        Tensor::zeros(1, DType::F32, device)?
    };

    let w_coords = if w_len > 1 {
        Tensor::arange(0u32, w_len as u32, device)?
            .to_dtype(DType::F32)?
            .broadcast_div(&Tensor::new(&[(w_len - 1) as f32], device)?)?
    } else {
        Tensor::zeros(1, DType::F32, device)?
    };

    Ok((t_coords, h_coords, w_coords))
}

/// Generate indices grid for RoPE from spatial dimensions
///
/// This version matches the original Rust implementation with [0,1] normalization.
/// For diffusers compatibility, use `generate_indices_grid_for_diffusers`.
pub fn generate_indices_grid(
    batch_size: usize,
    t: usize,
    h: usize,
    w: usize,
    device: &Device,
) -> Result<Tensor> {
    let seq_len = t * h * w;
    let mut indices = Vec::with_capacity(batch_size * 3 * seq_len);

    for _b in 0..batch_size {
        // T coordinates (normalized to [0, 1])
        for ti in 0..t {
            let t_coord = if t > 1 {
                ti as f32 / (t - 1) as f32
            } else {
                0.0
            };
            for _hi in 0..h {
                for _wi in 0..w {
                    indices.push(t_coord);
                }
            }
        }
    }

    for _b in 0..batch_size {
        // H coordinates
        for _ti in 0..t {
            for hi in 0..h {
                let h_coord = if h > 1 {
                    hi as f32 / (h - 1) as f32
                } else {
                    0.0
                };
                for _wi in 0..w {
                    indices.push(h_coord);
                }
            }
        }
    }

    for _b in 0..batch_size {
        // W coordinates
        for _ti in 0..t {
            for _hi in 0..h {
                for wi in 0..w {
                    let w_coord = if w > 1 {
                        wi as f32 / (w - 1) as f32
                    } else {
                        0.0
                    };
                    indices.push(w_coord);
                }
            }
        }
    }

    Tensor::from_vec(indices, (batch_size, 3, seq_len), device)
}

/// Generate indices grid matching diffusers LTXVideoRotaryPosEmbed._prepare_video_coords
///
/// Uses raw indices normalized by base dimensions:
/// - t_coord = t_idx / base_num_frames
/// - h_coord = h_idx / base_height  
/// - w_coord = w_idx / base_width
///
/// Default base values are from diffusers LTXVideoRotaryPosEmbed:
/// base_num_frames=20, base_height=2048, base_width=2048
#[allow(clippy::too_many_arguments)]
pub fn generate_indices_grid_for_diffusers(
    batch_size: usize,
    t: usize,
    h: usize,
    w: usize,
    base_num_frames: usize,
    base_height: usize,
    base_width: usize,
    patch_size: usize,
    patch_size_t: usize,
    rope_scale_t: f32, // rope_interpolation_scale[0] = vae_t_compression / fps
    rope_scale_h: f32, // rope_interpolation_scale[1] = vae_s_compression
    rope_scale_w: f32, // rope_interpolation_scale[2] = vae_s_compression
    device: &Device,
) -> Result<Tensor> {
    let seq_len = t * h * w;
    let mut indices = Vec::with_capacity(batch_size * 3 * seq_len);

    // Following diffusers formula exactly:
    // grid[:, 0:1] = grid[:, 0:1] * rope_scale[0] * patch_size_t / base_num_frames
    // grid[:, 1:2] = grid[:, 1:2] * rope_scale[1] * patch_size / base_height
    // grid[:, 2:3] = grid[:, 2:3] * rope_scale[2] * patch_size / base_width

    for _b in 0..batch_size {
        // T coordinates
        for ti in 0..t {
            let t_coord =
                (ti as f32) * rope_scale_t * (patch_size_t as f32) / (base_num_frames as f32);
            for _hi in 0..h {
                for _wi in 0..w {
                    indices.push(t_coord);
                }
            }
        }
    }

    for _b in 0..batch_size {
        // H coordinates
        for _ti in 0..t {
            for hi in 0..h {
                let h_coord =
                    (hi as f32) * rope_scale_h * (patch_size as f32) / (base_height as f32);
                for _wi in 0..w {
                    indices.push(h_coord);
                }
            }
        }
    }

    for _b in 0..batch_size {
        // W coordinates
        for _ti in 0..t {
            for _hi in 0..h {
                for wi in 0..w {
                    let w_coord =
                        (wi as f32) * rope_scale_w * (patch_size as f32) / (base_width as f32);
                    indices.push(w_coord);
                }
            }
        }
    }

    Tensor::from_vec(indices, (batch_size, 3, seq_len), device)
}

/// Compute rope_interpolation_scale for diffusers LTX pipelines.
///
/// Returns (temporal, height, width) scales as f32.
pub fn diffusers_rope_scales(frame_rate: Option<f64>) -> (f32, f32, f32) {
    let fr = frame_rate.unwrap_or(DEFAULT_FRAME_RATE);
    let temporal = (VAE_TEMPORAL_COMPRESSION / fr) as f32;
    let spatial = VAE_SPATIAL_COMPRESSION as f32;
    (temporal, spatial, spatial)
}

/// Generate indices grid using diffusers default constants and scaling.
pub fn generate_indices_grid_diffusers(
    batch_size: usize,
    t: usize,
    h: usize,
    w: usize,
    frame_rate: Option<f64>,
    device: &Device,
) -> Result<Tensor> {
    let (rope_scale_t, rope_scale_h, rope_scale_w) = diffusers_rope_scales(frame_rate);
    generate_indices_grid_for_diffusers(
        batch_size,
        t,
        h,
        w,
        DIFFUSERS_BASE_NUM_FRAMES,
        DIFFUSERS_BASE_HEIGHT,
        DIFFUSERS_BASE_WIDTH,
        1,
        1,
        rope_scale_t,
        rope_scale_h,
        rope_scale_w,
        device,
    )
}

/// Generate raw indices grid matching diffusers when rope_interpolation_scale=None
///
/// This produces RAW integer indices (as float32):
/// - t_coord = t_idx (0, 1, 2, ...)
/// - h_coord = h_idx (0, 1, 2, ..., height-1)
/// - w_coord = w_idx (0, 1, 2, ..., width-1)
///
/// This is exactly what diffusers _prepare_video_coords produces when
/// rope_interpolation_scale is None (no normalization applied).
pub fn generate_indices_grid_raw(
    batch_size: usize,
    t: usize,
    h: usize,
    w: usize,
    device: &Device,
) -> Result<Tensor> {
    let seq_len = t * h * w;
    let mut indices = Vec::with_capacity(batch_size * 3 * seq_len);

    // Raw T coordinates (0, 1, 2, ...)
    for _b in 0..batch_size {
        for ti in 0..t {
            let t_coord = ti as f32;
            for _hi in 0..h {
                for _wi in 0..w {
                    indices.push(t_coord);
                }
            }
        }
    }

    // Raw H coordinates (0, 1, 2, ..., h-1)
    for _b in 0..batch_size {
        for _ti in 0..t {
            for hi in 0..h {
                let h_coord = hi as f32;
                for _wi in 0..w {
                    indices.push(h_coord);
                }
            }
        }
    }

    // Raw W coordinates (0, 1, 2, ..., w-1)
    for _b in 0..batch_size {
        for _ti in 0..t {
            for _hi in 0..h {
                for wi in 0..w {
                    let w_coord = wi as f32;
                    indices.push(w_coord);
                }
            }
        }
    }

    Tensor::from_vec(indices, (batch_size, 3, seq_len), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fractional_rope_creation() {
        let rope = FractionalRoPE::new(96, 10000.0, 1024);
        assert_eq!(rope.head_dim, 96);
        assert_eq!(rope.base, 10000.0);
    }

    #[test]
    fn test_coord_grid_generation() -> Result<()> {
        let device = Device::Cpu;
        let (t, h, w) = generate_coord_grids(8, 16, 16, &device)?;

        assert_eq!(t.dims1()?, 8);
        assert_eq!(h.dims1()?, 16);
        assert_eq!(w.dims1()?, 16);
        Ok(())
    }

    #[test]
    fn test_indices_grid_generation() -> Result<()> {
        let device = Device::Cpu;
        let grid = generate_indices_grid(2, 4, 8, 8, &device)?;

        assert_eq!(grid.dim(0)?, 2); // batch
        assert_eq!(grid.dim(1)?, 3); // t, h, w coordinates
        assert_eq!(grid.dim(2)?, 4 * 8 * 8); // seq_len
        Ok(())
    }

    #[test]
    fn test_indices_grid_diffusers_shape() -> Result<()> {
        let device = Device::Cpu;
        let grid = generate_indices_grid_diffusers(1, 2, 3, 4, None, &device)?;

        assert_eq!(grid.dims(), &[1, 3, 2 * 3 * 4]);
        Ok(())
    }

    #[test]
    fn test_compute_freqs_cis() -> Result<()> {
        let device = Device::Cpu;
        let rope = FractionalRoPE::new(72, 10000.0, 1024);

        let batch = 2;
        let seq_len = 16;
        let indices = Tensor::randn(0f32, 0.5, (batch, 3, seq_len), &device)?;

        let (cos, sin) = rope.compute_freqs_cis(&indices, &device)?;

        assert_eq!(cos.dim(0)?, batch);
        assert_eq!(cos.dim(1)?, seq_len);
        assert_eq!(cos.dim(2)?, 72);

        assert_eq!(sin.dim(0)?, batch);
        assert_eq!(sin.dim(1)?, seq_len);
        assert_eq!(sin.dim(2)?, 72);
        Ok(())
    }

    #[test]
    fn test_apply_rotary_emb() -> Result<()> {
        let device = Device::Cpu;
        let rope = FractionalRoPE::new(72, 10000.0, 1024);

        let batch = 2;
        let heads = 4;
        let seq_len = 16;
        let head_dim = 72;

        let x = Tensor::randn(0f32, 1.0, (batch, heads, seq_len, head_dim), &device)?;
        let indices = Tensor::randn(0f32, 0.5, (batch, 3, seq_len), &device)?;

        let freqs_cis = rope.compute_freqs_cis(&indices, &device)?;
        let rotated = rope.apply_rotary_emb(&x, &freqs_cis)?;

        assert_eq!(rotated.dims(), x.dims());
        Ok(())
    }

    #[test]
    fn test_repeat_interleave_2() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 1, 3), &device)?;
        let repeated = repeat_interleave_2(&x)?;

        assert_eq!(repeated.dim(2)?, 6);

        let vals = repeated.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(vals, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        Ok(())
    }
}
