//! Rotary Position Embeddings (RoPE) for transformer models.
//!
//! Provides the apply_rotary_emb function and trait for RoPE implementations.

use candle_core::{DType, IndexOp, Result, Tensor, D};

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
    if c % 2 != 0 {
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
}
