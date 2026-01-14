use candle_core::{DType, Result, Tensor};
use candle_nn::{self as nn, Linear, Module, VarBuilder};

pub fn gelu_approximate(x: &Tensor) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let x_cube = x_f32.sqr()?.broadcast_mul(&x_f32)?;
    let inner = x_f32.broadcast_add(&x_cube.affine(0.044715, 0.0)?)?;
    let scale = (2.0f64 / std::f64::consts::PI).sqrt() as f32;
    let tanh_input = inner.affine(scale as f64, 0.0)?;
    let tanh_out = tanh_input.tanh()?;
    let gelu = x_f32
        .broadcast_mul(&tanh_out.affine(1.0, 1.0)?)?
        .affine(0.5, 0.0)?;
    gelu.to_dtype(x.dtype())
}

#[derive(Clone, Debug)]
pub struct GeluProjection {
    proj: Linear,
}

impl GeluProjection {
    pub fn new(dim_in: usize, dim_out: usize, vb: VarBuilder) -> Result<Self> {
        let proj = nn::linear(dim_in, dim_out, vb.pp("proj"))?;
        Ok(Self { proj })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.proj.forward(xs)?;
        gelu_approximate(&x)
    }
}

impl Module for GeluProjection {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}

#[derive(Clone, Debug)]
pub struct GeGluProjection {
    proj: Linear,
    dim_out: usize,
}

impl GeGluProjection {
    pub fn new(dim_in: usize, dim_out: usize, vb: VarBuilder) -> Result<Self> {
        let proj = nn::linear(dim_in, dim_out * 2, vb.pp("proj"))?;
        Ok(Self { proj, dim_out })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        use candle_core::D;
        let hidden = self.proj.forward(xs)?;

        let gate = hidden.narrow(D::Minus1, 0, self.dim_out)?;
        let value = hidden.narrow(D::Minus1, self.dim_out, self.dim_out)?;

        &gate.gelu_erf()? * value
    }
}

impl Module for GeGluProjection {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_gelu_approximate() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[[-1.0f32, 0.0, 1.0], [2.0, -2.0, 0.5]], &device)?;
        let out = gelu_approximate(&x)?;
        assert_eq!(out.dims(), x.dims());

        let out_vec: Vec<Vec<f32>> = out.to_vec2()?;
        assert!(out_vec[0][1].abs() < 1e-5, "GELU(0) should be ~0");
        Ok(())
    }
}
