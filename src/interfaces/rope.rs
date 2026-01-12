//! Rotary Position Embeddings (RoPE) for transformer models.
//!
//! Provides the apply_rotary_emb function and trait for RoPE implementations.

use candle_core::{D, DType, IndexOp, Result, Tensor};

/// Apply rotary position embeddings to a tensor.
///
/// # Arguments
/// * `x` - Input tensor of shape [B, S, C] where C must be even
/// * `cos` - Cosine frequencies of shape [B, S, C]
/// * `sin` - Sine frequencies of shape [B, S, C]
///
/// # Returns
/// Tensor with rotary embeddings applied, same shape as input.
pub fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    // Upcast to F32 for rotation math stability
    let x_f32 = x.to_dtype(DType::F32)?;
    let cos = cos.to_dtype(DType::F32)?;
    let sin = sin.to_dtype(DType::F32)?;

    let (b, s, c) = x_f32.dims3()?;
    if !c.is_multiple_of(2) {
        candle_core::bail!("apply_rotary_emb expects last dim even, got {c}");
    }
    let half = c / 2;

    // x -> [B, S, half, 2]
    let x2 = x_f32.reshape((b, s, half, 2))?;
    let x_real = x2.i((.., .., .., 0))?;
    let x_imag = x2.i((.., .., .., 1))?;

    // [-imag, real] interleave back.
    let x_rot = Tensor::stack(&[x_imag.neg()?, x_real.clone()], D::Minus1)?.reshape((b, s, c))?;

    let out = x_f32
        .broadcast_mul(&cos)?
        .broadcast_add(&x_rot.broadcast_mul(&sin)?)?;
    out.to_dtype(dtype)
}

/// Apply rotary position embeddings to a 4D tensor.
///
/// Used by Wan transformer where tensors are [B, S, H, D].
///
/// # Arguments
/// * `x` - Input tensor of shape [B, S, H, D] where D must be even
/// * `cos` - Cosine frequencies of shape [1, S, 1, D] or broadcastable
/// * `sin` - Sine frequencies of shape [1, S, 1, D] or broadcastable
///
/// # Returns
/// Tensor with rotary embeddings applied, same shape as input.
pub fn apply_rotary_emb_4d(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let cos = cos.to_dtype(DType::F32)?;
    let sin = sin.to_dtype(DType::F32)?;

    let (b, s, h, d) = x_f32.dims4()?;
    if !d.is_multiple_of(2) {
        candle_core::bail!("apply_rotary_emb_4d expects last dim even, got {d}");
    }

    // Split even/odd: x_even = x[..., 0::2], x_odd = x[..., 1::2]
    let half = d / 2;
    let x2 = x_f32.reshape((b, s, h, half, 2))?;
    let x_even = x2.i((.., .., .., .., 0))?; // [b, s, h, half]
    let x_odd = x2.i((.., .., .., .., 1))?; // [b, s, h, half]

    // cos/sin also need splitting for even/odd positions
    let cos_dims = cos.dims();
    let (cs1, cs2, cs3, _cs4) = (cos_dims[0], cos_dims[1], cos_dims[2], cos_dims[3]);
    let cos2 = cos.reshape((cs1, cs2, cs3, half, 2))?;
    let sin2 = sin.reshape((cs1, cs2, cs3, half, 2))?;
    let cos_even = cos2.i((.., .., .., .., 0))?;
    let sin_even = sin2.i((.., .., .., .., 1))?;

    // Rotation: out_even = x_even * cos - x_odd * sin
    //           out_odd = x_even * sin + x_odd * cos
    let out_even = x_even
        .broadcast_mul(&cos_even)?
        .broadcast_sub(&x_odd.broadcast_mul(&sin_even)?)?;
    let out_odd = x_even
        .broadcast_mul(&sin_even)?
        .broadcast_add(&x_odd.broadcast_mul(&cos_even)?)?;

    // Interleave even/odd back to [b, s, h, d]
    let out_even = out_even.unsqueeze(D::Minus1)?; // [b, s, h, half, 1]
    let out_odd = out_odd.unsqueeze(D::Minus1)?; // [b, s, h, half, 1]
    let stacked = Tensor::cat(&[&out_even, &out_odd], 4)?; // [b, s, h, half, 2]
    let out = stacked.reshape((b, s, h, d))?;

    out.to_dtype(dtype)
}

/// Generate 1D rotary position embeddings for a sequence.
///
/// Implementation matches diffusers `get_1d_rotary_pos_embed` with:
/// - `use_real=True` (returns cos/sin tensors, not complex)
/// - `repeat_interleave_real=True` (repeats each frequency twice)
///
/// # Arguments
/// * `dim` - Embedding dimension (must be even)
/// * `max_seq_len` - Maximum sequence length
/// * `theta` - Base frequency (typically 10000.0)
/// * `device` - Device to create tensors on
///
/// # Returns
/// Tuple of (cos, sin) tensors, each of shape [max_seq_len, dim]
pub fn get_1d_rotary_pos_embed(
    dim: usize,
    max_seq_len: usize,
    theta: f64,
    device: &candle_core::Device,
) -> Result<(Tensor, Tensor)> {
    if !dim.is_multiple_of(2) {
        candle_core::bail!("rotary dim must be even, got {dim}");
    }

    let half = dim / 2;

    // inv_freq: [half] = 1.0 / (theta ^ (2i / dim))
    let mut inv_freq = Vec::with_capacity(half);
    for i in 0..half {
        let exponent = (2.0 * i as f64) / (dim as f64);
        inv_freq.push((1.0 / theta.powf(exponent)) as f32);
    }
    let inv_freq = Tensor::from_vec(inv_freq, (half,), device)?;

    // positions: [max_seq_len]
    let pos: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
    let pos = Tensor::from_vec(pos, (max_seq_len, 1), device)?; // [L, 1]

    // angles: [L, half] = pos @ inv_freq.T
    let angles = pos.matmul(&inv_freq.unsqueeze(0)?)?; // [L, half]

    // repeat_interleave_real=True => expand to [L, dim] by cat
    let angles2 = Tensor::cat(&[&angles, &angles], 1)?; // [L, dim]

    let cos = angles2.cos()?;
    let sin = angles2.sin()?;
    Ok((cos, sin))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_apply_rotary_emb() -> Result<()> {
        let device = Device::Cpu;
        let b = 2;
        let s = 4;
        let c = 8;

        let x = Tensor::randn(0.0f32, 1.0, (b, s, c), &device)?;
        let cos = Tensor::ones((b, s, c), DType::F32, &device)?;
        let sin = Tensor::zeros((b, s, c), DType::F32, &device)?;

        let out = apply_rotary_emb(&x, &cos, &sin)?;
        assert_eq!(out.dims(), &[b, s, c]);

        // With sin=0, cos=1, output should equal input
        let x_vec: Vec<f32> = x.flatten_all()?.to_vec1()?;
        let out_vec: Vec<f32> = out.flatten_all()?.to_vec1()?;
        for (a, b) in x_vec.iter().zip(out_vec.iter()) {
            assert!((a - b).abs() < 1e-5, "Expected {} == {}", a, b);
        }
        Ok(())
    }

    #[test]
    fn test_apply_rotary_emb_odd_dim_fails() {
        let device = Device::Cpu;
        let x = Tensor::zeros((1, 2, 5), DType::F32, &device).unwrap();
        let cos = Tensor::ones((1, 2, 5), DType::F32, &device).unwrap();
        let sin = Tensor::zeros((1, 2, 5), DType::F32, &device).unwrap();

        let result = apply_rotary_emb(&x, &cos, &sin);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_rotary_emb_4d() -> Result<()> {
        let device = Device::Cpu;
        let b = 2;
        let s = 4;
        let h = 8;
        let d = 16;

        let x = Tensor::randn(0.0f32, 1.0, (b, s, h, d), &device)?;
        let cos = Tensor::ones((1, s, 1, d), DType::F32, &device)?;
        let sin = Tensor::zeros((1, s, 1, d), DType::F32, &device)?;

        let out = apply_rotary_emb_4d(&x, &cos, &sin)?;
        assert_eq!(out.dims(), &[b, s, h, d]);
        Ok(())
    }

    #[test]
    fn test_get_1d_rotary_pos_embed() -> Result<()> {
        let device = Device::Cpu;
        let dim = 64;
        let max_seq = 128;
        let theta = 10000.0;

        let (cos, sin) = get_1d_rotary_pos_embed(dim, max_seq, theta, &device)?;
        assert_eq!(cos.dims(), &[max_seq, dim]);
        assert_eq!(sin.dims(), &[max_seq, dim]);

        // Check first position has cos=1, sin=0 for all dims
        let cos0: Vec<f32> = cos.i(0)?.to_vec1()?;
        for c in cos0.iter() {
            assert!((c - 1.0).abs() < 1e-5, "cos(0) should be 1");
        }
        Ok(())
    }
}
