#![allow(clippy::too_many_arguments)]
use candle_core::{Result, Tensor};

#[derive(Clone, Copy, Debug)]
pub struct Im2ColConfig {
    pub kernel: (usize, usize, usize),

    pub stride: (usize, usize, usize),

    pub dilation: (usize, usize, usize),
}

impl Im2ColConfig {
    pub fn new(
        kernel: (usize, usize, usize),
        stride: (usize, usize, usize),
        dilation: (usize, usize, usize),
    ) -> Self {
        Self {
            kernel,
            stride,
            dilation,
        }
    }
}

pub fn im2col_3d(
    x: &Tensor,
    config: &Im2ColConfig,
    t_out: usize,
    h_out: usize,
    w_out: usize,
) -> Result<Tensor> {
    let (kt, kh, kw) = config.kernel;
    let (st, sh, sw) = config.stride;
    let (dt, dh, dw) = config.dilation;

    let (batch, in_c, _t_pad, _h_pad, _w_pad) = x.dims5()?;

    let spatial_out = t_out * h_out * w_out;

    let patch_size = in_c * kt * kh * kw;

    let mut patches = Vec::with_capacity(spatial_out);

    for to in 0..t_out {
        for ho in 0..h_out {
            for wo in 0..w_out {
                let patch = extract_patch_3d(x, to, ho, wo, kt, kh, kw, st, sh, sw, dt, dh, dw)?;

                patches.push(patch.unsqueeze(1)?);
            }
        }
    }

    let result = Tensor::cat(&patches.iter().collect::<Vec<_>>(), 1)?;
    debug_assert_eq!(result.dims(), &[batch, spatial_out, patch_size]);

    Ok(result)
}

fn extract_patch_3d(
    x: &Tensor,
    to: usize,
    ho: usize,
    wo: usize,
    kt: usize,
    kh: usize,
    kw: usize,
    st: usize,
    sh: usize,
    sw: usize,
    dt: usize,
    dh: usize,
    dw: usize,
) -> Result<Tensor> {
    let (batch, in_c, _t, _h, _w) = x.dims5()?;

    let mut slices = Vec::with_capacity(in_c * kt * kh * kw);

    for c in 0..in_c {
        for ki_t in 0..kt {
            let ti = to * st + ki_t * dt;
            for ki_h in 0..kh {
                let hi = ho * sh + ki_h * dh;
                for ki_w in 0..kw {
                    let wi = wo * sw + ki_w * dw;

                    let slice = x
                        .narrow(1, c, 1)?
                        .narrow(2, ti, 1)?
                        .narrow(3, hi, 1)?
                        .narrow(4, wi, 1)?
                        .reshape((batch, 1))?;
                    slices.push(slice);
                }
            }
        }
    }

    Tensor::cat(&slices.iter().collect::<Vec<_>>(), 1)
}

pub fn conv3d_cpu_forward(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    config: &Im2ColConfig,
    groups: usize,
    t_out: usize,
    h_out: usize,
    w_out: usize,
) -> Result<Tensor> {
    let (kt, kh, kw) = config.kernel;
    let (batch, in_c, _t_pad, _h_pad, _w_pad) = x.dims5()?;
    let out_c = weight.dims()[0];
    let in_c_per_group = in_c / groups;
    let out_c_per_group = out_c / groups;

    let col = im2col_3d(x, config, t_out, h_out, w_out)?;

    let spatial = t_out * h_out * w_out;
    let patch_size = in_c_per_group * kt * kh * kw;

    let y = if groups == 1 {
        let weight_2d = weight.reshape((out_c, patch_size))?.t()?.contiguous()?;

        let col_2d = col.reshape((batch * spatial, patch_size))?;

        let y_2d = col_2d.matmul(&weight_2d)?;

        y_2d.reshape((batch, t_out, h_out, w_out, out_c))?
            .permute((0, 4, 1, 2, 3))?
    } else {
        let mut outputs = Vec::with_capacity(groups);

        for g in 0..groups {
            let col_g = col.narrow(2, g * patch_size, patch_size)?;
            let col_g_2d = col_g.reshape((batch * spatial, patch_size))?;

            let weight_g = weight
                .narrow(0, g * out_c_per_group, out_c_per_group)?
                .reshape((out_c_per_group, patch_size))?
                .t()?
                .contiguous()?;

            let y_g = col_g_2d.matmul(&weight_g)?;
            outputs.push(y_g.reshape((batch, spatial, out_c_per_group))?);
        }

        let y = Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 2)?;
        y.reshape((batch, t_out, h_out, w_out, out_c))?
            .permute((0, 4, 1, 2, 3))?
    };

    if let Some(bias) = bias {
        let bias = bias.reshape((1, out_c, 1, 1, 1))?;
        y.broadcast_add(&bias)
    } else {
        Ok(y)
    }
}

pub fn conv3d_pointwise(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let (batch, in_c, t, h, w) = x.dims5()?;
    let out_c = weight.dims()[0];

    let x_2d = x
        .permute((0, 2, 3, 4, 1))?
        .reshape((batch * t * h * w, in_c))?;

    let weight_2d = weight.reshape((out_c, in_c))?;

    let y = x_2d.matmul(&weight_2d.t()?)?;

    let y = y
        .reshape((batch, t, h, w, out_c))?
        .permute((0, 4, 1, 2, 3))?;

    if let Some(bias) = bias {
        let bias = bias.reshape((1, out_c, 1, 1, 1))?;
        y.broadcast_add(&bias)
    } else {
        Ok(y)
    }
}

pub fn conv3d_temporal_only(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    kt: usize,
    st: usize,
    dt: usize,
    t_out: usize,
    sh: usize,
    sw: usize,
    h_out: usize,
    w_out: usize,
) -> Result<Tensor> {
    let (batch, in_c, _t, h, w) = x.dims5()?;
    let out_c = weight.dims()[0];

    let (x_strided, h_actual, w_actual) = if sh > 1 || sw > 1 {
        let mut rows = Vec::with_capacity(h_out);
        for hi in 0..h_out {
            let h_idx = hi * sh;
            let row = x.narrow(3, h_idx, 1)?;
            rows.push(row);
        }
        let x_h = Tensor::cat(&rows.iter().collect::<Vec<_>>(), 3)?;

        let mut cols = Vec::with_capacity(w_out);
        for wi in 0..w_out {
            let w_idx = wi * sw;
            let col = x_h.narrow(4, w_idx, 1)?;
            cols.push(col);
        }
        let x_hw = Tensor::cat(&cols.iter().collect::<Vec<_>>(), 4)?;
        (x_hw, h_out, w_out)
    } else {
        (x.clone(), h, w)
    };

    let x_3d = x_strided.permute((0, 3, 4, 1, 2))?.reshape((
        batch * h_actual * w_actual,
        in_c,
        x_strided.dims()[2],
    ))?;

    let weight_3d = weight.reshape((out_c, in_c, kt))?;

    let col = im2col_1d(&x_3d, kt, st, dt, t_out)?;

    let weight_2d = weight_3d.reshape((out_c, in_c * kt))?.t()?.contiguous()?;

    let col_2d = col.reshape((batch * h_actual * w_actual * t_out, in_c * kt))?;
    let y_2d = col_2d.matmul(&weight_2d)?;

    let y = y_2d
        .reshape((batch, h_actual, w_actual, t_out, out_c))?
        .permute((0, 4, 3, 1, 2))?;

    if let Some(bias) = bias {
        let bias = bias.reshape((1, out_c, 1, 1, 1))?;
        y.broadcast_add(&bias)
    } else {
        Ok(y)
    }
}

fn im2col_1d(x: &Tensor, k: usize, s: usize, d: usize, out_len: usize) -> Result<Tensor> {
    let (batch, c, _t) = x.dims3()?;

    let mut patches = Vec::with_capacity(out_len);

    for o in 0..out_len {
        let mut slices = Vec::with_capacity(c * k);
        for ci in 0..c {
            for ki in 0..k {
                let ti = o * s + ki * d;
                let slice = x.narrow(1, ci, 1)?.narrow(2, ti, 1)?.reshape((batch, 1))?;
                slices.push(slice);
            }
        }
        let patch = Tensor::cat(&slices.iter().collect::<Vec<_>>(), 1)?;
        patches.push(patch.unsqueeze(1)?);
    }

    Tensor::cat(&patches.iter().collect::<Vec<_>>(), 1)
}

pub fn conv3d_spatial_only(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    dh: usize,
    dw: usize,
    h_out: usize,
    w_out: usize,
    st: usize,
    t_out: usize,
) -> Result<Tensor> {
    let (batch, in_c, t, _h, _w) = x.dims5()?;
    let out_c = weight.dims()[0];

    let x_strided = if st > 1 {
        let mut frames = Vec::with_capacity(t_out);
        for i in 0..t_out {
            let frame_idx = i * st;
            let frame = x.narrow(2, frame_idx, 1)?;
            frames.push(frame);
        }
        Tensor::cat(&frames.iter().collect::<Vec<_>>(), 2)?
    } else {
        x.clone()
    };

    let t_actual = if st > 1 { t_out } else { t };

    let x_4d = x_strided.permute((0, 2, 1, 3, 4))?.reshape((
        batch * t_actual,
        in_c,
        x.dims()[3],
        x.dims()[4],
    ))?;

    let weight_4d = weight.reshape((out_c, in_c, kh, kw))?;

    let col = im2col_2d(&x_4d, kh, kw, sh, sw, dh, dw, h_out, w_out)?;

    let weight_2d = weight_4d
        .reshape((out_c, in_c * kh * kw))?
        .t()?
        .contiguous()?;

    let spatial = h_out * w_out;
    let col_2d = col.reshape((batch * t_actual * spatial, in_c * kh * kw))?;
    let y_2d = col_2d.matmul(&weight_2d)?;

    let y = y_2d
        .reshape((batch, t_actual, h_out, w_out, out_c))?
        .permute((0, 4, 1, 2, 3))?;

    if let Some(bias) = bias {
        let bias = bias.reshape((1, out_c, 1, 1, 1))?;
        y.broadcast_add(&bias)
    } else {
        Ok(y)
    }
}

fn im2col_2d(
    x: &Tensor,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    dh: usize,
    dw: usize,
    h_out: usize,
    w_out: usize,
) -> Result<Tensor> {
    let (batch, c, _h, _w) = x.dims4()?;

    let mut patches = Vec::with_capacity(h_out * w_out);

    for ho in 0..h_out {
        for wo in 0..w_out {
            let mut slices = Vec::with_capacity(c * kh * kw);
            for ci in 0..c {
                for ki_h in 0..kh {
                    let hi = ho * sh + ki_h * dh;
                    for ki_w in 0..kw {
                        let wi = wo * sw + ki_w * dw;
                        let slice = x
                            .narrow(1, ci, 1)?
                            .narrow(2, hi, 1)?
                            .narrow(3, wi, 1)?
                            .reshape((batch, 1))?;
                        slices.push(slice);
                    }
                }
            }
            let patch = Tensor::cat(&slices.iter().collect::<Vec<_>>(), 1)?;
            patches.push(patch.unsqueeze(1)?);
        }
    }

    Tensor::cat(&patches.iter().collect::<Vec<_>>(), 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn create_test_tensor_5d(shape: (usize, usize, usize, usize, usize)) -> Result<Tensor> {
        let (b, c, t, h, w) = shape;
        let device = Device::Cpu;
        let data: Vec<f32> = (0..(b * c * t * h * w) as u32).map(|x| x as f32).collect();
        Tensor::from_vec(data, (b, c, t, h, w), &device)
    }

    fn create_random_tensor_5d(shape: (usize, usize, usize, usize, usize)) -> Result<Tensor> {
        let device = Device::Cpu;
        Tensor::randn(0f32, 1.0, shape, &device)
    }

    #[test]
    fn test_im2col_3d_basic() -> Result<()> {
        let x = create_test_tensor_5d((1, 2, 4, 4, 4))?;
        let config = Im2ColConfig::new((2, 2, 2), (1, 1, 1), (1, 1, 1));

        let col = im2col_3d(&x, &config, 3, 3, 3)?;

        assert_eq!(col.dims(), &[1, 27, 16]);

        Ok(())
    }

    #[test]
    fn test_im2col_3d_stride() -> Result<()> {
        let x = create_test_tensor_5d((1, 2, 6, 6, 6))?;
        let config = Im2ColConfig::new((2, 2, 2), (2, 2, 2), (1, 1, 1));

        let col = im2col_3d(&x, &config, 3, 3, 3)?;

        assert_eq!(col.dims(), &[1, 27, 16]);

        Ok(())
    }

    #[test]
    fn test_conv3d_cpu_forward_basic() -> Result<()> {
        let device = Device::Cpu;
        let x = create_random_tensor_5d((2, 4, 4, 8, 8))?;

        let weight = Tensor::randn(0f32, 0.1, (8, 4, 3, 3, 3), &device)?;
        let bias = Tensor::zeros(8, candle_core::DType::F32, &device)?;

        let config = Im2ColConfig::new((3, 3, 3), (1, 1, 1), (1, 1, 1));

        let y = conv3d_cpu_forward(&x, &weight, Some(&bias), &config, 1, 2, 6, 6)?;

        assert_eq!(y.dims(), &[2, 8, 2, 6, 6]);

        Ok(())
    }

    #[test]
    fn test_conv3d_pointwise() -> Result<()> {
        let device = Device::Cpu;
        let x = create_random_tensor_5d((2, 4, 8, 16, 16))?;

        let weight = Tensor::randn(0f32, 0.1, (8, 4, 1, 1, 1), &device)?;
        let bias = Tensor::zeros(8, candle_core::DType::F32, &device)?;

        let y = conv3d_pointwise(&x, &weight, Some(&bias))?;

        assert_eq!(y.dims(), &[2, 8, 8, 16, 16]);

        Ok(())
    }

    #[test]
    fn test_conv3d_temporal_only() -> Result<()> {
        let device = Device::Cpu;
        let x = create_random_tensor_5d((2, 4, 8, 8, 8))?;

        let weight = Tensor::randn(0f32, 0.1, (8, 4, 3, 1, 1), &device)?;
        let bias = Tensor::zeros(8, candle_core::DType::F32, &device)?;

        let y = conv3d_temporal_only(&x, &weight, Some(&bias), 3, 1, 1, 6, 1, 1, 8, 8)?;

        assert_eq!(y.dims(), &[2, 8, 6, 8, 8]);

        Ok(())
    }

    #[test]
    fn test_conv3d_spatial_only() -> Result<()> {
        let device = Device::Cpu;
        let x = create_random_tensor_5d((2, 4, 8, 8, 8))?;

        let weight = Tensor::randn(0f32, 0.1, (8, 4, 1, 3, 3), &device)?;
        let bias = Tensor::zeros(8, candle_core::DType::F32, &device)?;

        let y = conv3d_spatial_only(&x, &weight, Some(&bias), 3, 3, 1, 1, 1, 1, 6, 6, 1, 8)?;

        assert_eq!(y.dims(), &[2, 8, 8, 6, 6]);

        Ok(())
    }

    #[test]
    fn test_conv3d_grouped() -> Result<()> {
        let device = Device::Cpu;
        let x = create_random_tensor_5d((1, 4, 4, 4, 4))?;

        let weight = Tensor::randn(0f32, 0.1, (8, 2, 2, 2, 2), &device)?;
        let bias = Tensor::zeros(8, candle_core::DType::F32, &device)?;

        let config = Im2ColConfig::new((2, 2, 2), (1, 1, 1), (1, 1, 1));

        let y = conv3d_cpu_forward(&x, &weight, Some(&bias), &config, 2, 3, 3, 3)?;

        assert_eq!(y.dims(), &[1, 8, 3, 3, 3]);

        Ok(())
    }
}
