use candle_core::{D, DType, Result, Tensor};
use candle_nn::VarBuilder;

#[derive(Clone, Debug)]
pub struct RmsNorm {
    weight: Option<Tensor>,
    eps: f64,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f64, elementwise_affine: bool, vb: VarBuilder) -> Result<Self> {
        let weight = if elementwise_affine {
            Some(vb.get(dim, "weight")?)
        } else {
            None
        };
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let dim = xs_f32.dim(D::Minus1)? as f64;
        let ms = xs_f32
            .sqr()?
            .sum_keepdim(D::Minus1)?
            .affine(1.0 / dim, 0.0)?;
        let denom = ms.affine(1.0, self.eps)?.sqrt()?;
        let ys_f32 = xs_f32.broadcast_div(&denom)?;
        let mut ys = ys_f32.to_dtype(dtype)?;
        if let Some(w) = &self.weight {
            let rank = ys.rank();
            let mut shape = vec![1usize; rank];
            shape[rank - 1] = w.dims1()?;
            let w = w.reshape(shape)?;
            ys = ys.broadcast_mul(&w)?;
        }
        Ok(ys)
    }
}

#[derive(Clone, Debug)]
pub struct LayerNormNoParams {
    eps: f64,
}

impl LayerNormNoParams {
    pub fn new(eps: f64) -> Self {
        Self { eps }
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let last_dim = xs.dim(D::Minus1)?;
        let mean = (xs.sum_keepdim(D::Minus1)? / (last_dim as f64))?;
        let xc = xs.broadcast_sub(&mean)?;
        let var = (xc.sqr()?.sum_keepdim(D::Minus1)? / (last_dim as f64))?;
        let denom = (var + self.eps)?.sqrt()?;
        xc.broadcast_div(&denom)
    }
}

pub fn rmsnorm_channels_first(norm: &candle_nn::RmsNorm, x: &Tensor) -> Result<Tensor> {
    x.permute((0, 2, 3, 4, 1))?
        .apply(norm)?
        .permute((0, 4, 1, 2, 3))
}

pub fn layernorm_channels_first(norm: &candle_nn::LayerNorm, x: &Tensor) -> Result<Tensor> {
    x.permute((0, 2, 3, 4, 1))?
        .apply(norm)?
        .permute((0, 4, 1, 2, 3))
}

pub fn rmsnorm_channels_first_custom(norm: &RmsNorm, x: &Tensor) -> Result<Tensor> {
    let permuted = x.permute((0, 2, 3, 4, 1))?;
    let normed = norm.forward(&permuted)?;
    normed.permute((0, 4, 1, 2, 3))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_rms_norm() -> Result<()> {
        let device = Device::Cpu;
        let xs = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device)?;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let norm = RmsNorm::new(3, 1e-5, false, vb)?;
        let out = norm.forward(&xs)?;
        assert_eq!(out.dims(), xs.dims());
        Ok(())
    }

    #[test]
    fn test_layer_norm_no_params() -> Result<()> {
        let device = Device::Cpu;
        let xs = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device)?;
        let norm = LayerNormNoParams::new(1e-5);
        let out = norm.forward(&xs)?;
        assert_eq!(out.dims(), xs.dims());

        let mean = out.sum_keepdim(D::Minus1)?.squeeze(D::Minus1)?;
        let mean_vals: Vec<f32> = mean.to_vec1()?;
        for m in mean_vals {
            assert!(m.abs() < 1e-5, "Mean should be close to 0, got {}", m);
        }
        Ok(())
    }
}
