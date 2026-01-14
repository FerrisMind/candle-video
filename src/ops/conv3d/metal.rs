#![allow(clippy::too_many_arguments)]
use candle_core::{DType, Device, Result, Tensor};

use super::cpu::Im2ColConfig;

pub fn is_metal_tensor(tensor: &Tensor) -> bool {
    matches!(tensor.device(), Device::Metal(_))
}

pub fn is_metal_device(device: &Device) -> bool {
    matches!(device, Device::Metal(_))
}

pub fn is_supported_metal_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F16)
}

pub fn supported_metal_dtypes() -> Vec<DType> {
    vec![DType::F32, DType::F16]
}

pub fn conv3d_metal_forward(
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

    let col = im2col_3d_metal(x, config, t_out, h_out, w_out)?;

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

fn im2col_3d_metal(
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
                let patch =
                    extract_patch_3d_metal(x, to, ho, wo, kt, kh, kw, st, sh, sw, dt, dh, dw)?;

                patches.push(patch.unsqueeze(1)?);
            }
        }
    }

    let result = Tensor::cat(&patches.iter().collect::<Vec<_>>(), 1)?;
    debug_assert_eq!(result.dims(), &[batch, spatial_out, patch_size]);

    Ok(result)
}

fn extract_patch_3d_metal(
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

    let mut slices = Vec::with_capacity(kt * kh * kw);

    for ki_t in 0..kt {
        let ti = to * st + ki_t * dt;
        for ki_h in 0..kh {
            let hi = ho * sh + ki_h * dh;
            for ki_w in 0..kw {
                let wi = wo * sw + ki_w * dw;

                let slice = x
                    .narrow(2, ti, 1)?
                    .narrow(3, hi, 1)?
                    .narrow(4, wi, 1)?
                    .reshape((batch, in_c))?;
                slices.push(slice);
            }
        }
    }

    Tensor::cat(&slices.iter().collect::<Vec<_>>(), 1)
}

pub fn conv3d_pointwise_metal(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
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

pub fn conv3d_spatial_only_metal(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    kh: usize,
    kw: usize,
    sh: usize,
    _sw: usize,
    dh: usize,
    _dw: usize,
    h_out: usize,
    w_out: usize,
    st: usize,
    t_out: usize,
    groups: usize,
) -> Result<Tensor> {
    let (batch, in_c, t, h, w) = x.dims5()?;
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

    let x_4d = x_strided
        .permute((0, 2, 1, 3, 4))?
        .reshape((batch * t_actual, in_c, h, w))?;

    let weight_4d = weight.reshape((out_c, in_c / groups, kh, kw))?;

    let y_4d = x_4d.conv2d(&weight_4d, 0, sh, dh, groups)?;

    let y = y_4d
        .reshape((batch, t_actual, out_c, h_out, w_out))?
        .permute((0, 2, 1, 3, 4))?;

    if let Some(bias) = bias {
        let bias = bias.reshape((1, out_c, 1, 1, 1))?;
        y.broadcast_add(&bias)
    } else {
        Ok(y)
    }
}

pub fn conv3d_temporal_only_metal(
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
    groups: usize,
) -> Result<Tensor> {
    let (batch, in_c, t, h, w) = x.dims5()?;
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

    let x_3d =
        x_strided
            .permute((0, 3, 4, 1, 2))?
            .reshape((batch * h_actual * w_actual, in_c, t))?;

    let weight_3d = weight.reshape((out_c, in_c / groups, kt))?;

    let y_3d = x_3d.conv1d(&weight_3d, 0, st, dt, groups)?;

    let y = y_3d
        .reshape((batch, h_actual, w_actual, out_c, t_out))?
        .permute((0, 3, 4, 1, 2))?;

    if let Some(bias) = bias {
        let bias = bias.reshape((1, out_c, 1, 1, 1))?;
        y.broadcast_add(&bias)
    } else {
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_supported_metal_dtype() {
        assert!(is_supported_metal_dtype(DType::F32));
        assert!(is_supported_metal_dtype(DType::F16));

        assert!(!is_supported_metal_dtype(DType::BF16));

        assert!(!is_supported_metal_dtype(DType::F64));
    }

    #[test]
    fn test_supported_metal_dtypes() {
        let dtypes = supported_metal_dtypes();
        assert_eq!(dtypes.len(), 2);
        assert!(dtypes.contains(&DType::F32));
        assert!(dtypes.contains(&DType::F16));
    }

    #[test]
    fn test_is_metal_device() {
        let cpu = Device::Cpu;
        assert!(!is_metal_device(&cpu));
    }

    #[test]
    fn test_is_metal_tensor() {
        let device = Device::Cpu;
        let tensor = Tensor::zeros((2, 3), DType::F32, &device).unwrap();
        assert!(!is_metal_tensor(&tensor));
    }

    #[cfg(test)]
    mod cpu_algorithm_tests {
        use super::*;
        use crate::ops::conv3d::cpu::Im2ColConfig;

        fn create_test_tensor_5d(shape: (usize, usize, usize, usize, usize)) -> Result<Tensor> {
            let (b, c, t, h, w) = shape;
            let device = Device::Cpu;
            Tensor::randn(0f32, 1.0, (b, c, t, h, w), &device)
        }

        #[test]
        fn test_conv3d_metal_forward_basic() -> Result<()> {
            let device = Device::Cpu;
            let x = create_test_tensor_5d((2, 4, 4, 8, 8))?;

            let weight = Tensor::randn(0f32, 0.1, (8, 4, 3, 3, 3), &device)?;
            let bias = Tensor::zeros(8, DType::F32, &device)?;

            let config = Im2ColConfig::new((3, 3, 3), (1, 1, 1), (1, 1, 1));

            let y = conv3d_metal_forward(&x, &weight, Some(&bias), &config, 1, 2, 6, 6)?;

            assert_eq!(y.dims(), &[2, 8, 2, 6, 6]);

            Ok(())
        }

        #[test]
        fn test_conv3d_pointwise_metal() -> Result<()> {
            let device = Device::Cpu;
            let x = create_test_tensor_5d((2, 4, 8, 16, 16))?;

            let weight = Tensor::randn(0f32, 0.1, (8, 4, 1, 1, 1), &device)?;
            let bias = Tensor::zeros(8, DType::F32, &device)?;

            let y = conv3d_pointwise_metal(&x, &weight, Some(&bias))?;

            assert_eq!(y.dims(), &[2, 8, 8, 16, 16]);

            Ok(())
        }

        #[test]
        fn test_conv3d_temporal_only_metal() -> Result<()> {
            let device = Device::Cpu;
            let x = create_test_tensor_5d((2, 4, 8, 8, 8))?;

            let weight = Tensor::randn(0f32, 0.1, (8, 4, 3, 1, 1), &device)?;
            let bias = Tensor::zeros(8, DType::F32, &device)?;

            let y =
                conv3d_temporal_only_metal(&x, &weight, Some(&bias), 3, 1, 1, 6, 1, 1, 8, 8, 1)?;

            assert_eq!(y.dims(), &[2, 8, 6, 8, 8]);

            Ok(())
        }

        #[test]
        fn test_conv3d_spatial_only_metal() -> Result<()> {
            let device = Device::Cpu;
            let x = create_test_tensor_5d((2, 4, 8, 8, 8))?;

            let weight = Tensor::randn(0f32, 0.1, (8, 4, 1, 3, 3), &device)?;
            let bias = Tensor::zeros(8, DType::F32, &device)?;

            let y = conv3d_spatial_only_metal(
                &x,
                &weight,
                Some(&bias),
                3,
                3,
                1,
                1,
                1,
                1,
                6,
                6,
                1,
                8,
                1,
            )?;

            assert_eq!(y.dims(), &[2, 8, 8, 6, 6]);

            Ok(())
        }

        #[test]
        fn test_conv3d_metal_grouped() -> Result<()> {
            let device = Device::Cpu;
            let x = create_test_tensor_5d((1, 4, 4, 4, 4))?;

            let weight = Tensor::randn(0f32, 0.1, (8, 2, 2, 2, 2), &device)?;
            let bias = Tensor::zeros(8, DType::F32, &device)?;

            let config = Im2ColConfig::new((2, 2, 2), (1, 1, 1), (1, 1, 1));

            let y = conv3d_metal_forward(&x, &weight, Some(&bias), &config, 2, 3, 3, 3)?;

            assert_eq!(y.dims(), &[1, 8, 3, 3, 3]);

            Ok(())
        }

        #[test]
        fn test_conv3d_metal_f16() -> Result<()> {
            let device = Device::Cpu;

            let x = Tensor::randn(0f32, 1.0, (1, 2, 4, 4, 4), &device)?.to_dtype(DType::F16)?;
            let weight =
                Tensor::randn(0f32, 0.1, (4, 2, 2, 2, 2), &device)?.to_dtype(DType::F16)?;
            let bias = Tensor::zeros(4, DType::F16, &device)?;

            let config = Im2ColConfig::new((2, 2, 2), (1, 1, 1), (1, 1, 1));

            let y = conv3d_metal_forward(&x, &weight, Some(&bias), &config, 1, 3, 3, 3)?;

            assert_eq!(y.dims(), &[1, 4, 3, 3, 3]);
            assert_eq!(y.dtype(), DType::F16);

            Ok(())
        }

        #[test]
        fn test_conv3d_metal_f32() -> Result<()> {
            let device = Device::Cpu;

            let x = Tensor::randn(0f32, 1.0, (1, 2, 4, 4, 4), &device)?;
            let weight = Tensor::randn(0f32, 0.1, (4, 2, 2, 2, 2), &device)?;
            let bias = Tensor::zeros(4, DType::F32, &device)?;

            let config = Im2ColConfig::new((2, 2, 2), (1, 1, 1), (1, 1, 1));

            let y = conv3d_metal_forward(&x, &weight, Some(&bias), &config, 1, 3, 3, 3)?;

            assert_eq!(y.dims(), &[1, 4, 3, 3, 3]);
            assert_eq!(y.dtype(), DType::F32);

            Ok(())
        }

        #[test]
        fn test_conv3d_pointwise_metal_f16() -> Result<()> {
            let device = Device::Cpu;
            let x = Tensor::randn(0f32, 1.0, (2, 4, 8, 16, 16), &device)?.to_dtype(DType::F16)?;
            let weight =
                Tensor::randn(0f32, 0.1, (8, 4, 1, 1, 1), &device)?.to_dtype(DType::F16)?;
            let bias = Tensor::zeros(8, DType::F16, &device)?;

            let y = conv3d_pointwise_metal(&x, &weight, Some(&bias))?;

            assert_eq!(y.dims(), &[2, 8, 8, 16, 16]);
            assert_eq!(y.dtype(), DType::F16);

            Ok(())
        }

        #[test]
        fn test_conv3d_temporal_only_metal_f16() -> Result<()> {
            let device = Device::Cpu;
            let x = Tensor::randn(0f32, 1.0, (2, 4, 8, 8, 8), &device)?.to_dtype(DType::F16)?;
            let weight =
                Tensor::randn(0f32, 0.1, (8, 4, 3, 1, 1), &device)?.to_dtype(DType::F16)?;
            let bias = Tensor::zeros(8, DType::F16, &device)?;

            let y =
                conv3d_temporal_only_metal(&x, &weight, Some(&bias), 3, 1, 1, 6, 1, 1, 8, 8, 1)?;

            assert_eq!(y.dims(), &[2, 8, 6, 8, 8]);
            assert_eq!(y.dtype(), DType::F16);

            Ok(())
        }

        #[test]
        fn test_conv3d_spatial_only_metal_f16() -> Result<()> {
            let device = Device::Cpu;
            let x = Tensor::randn(0f32, 1.0, (2, 4, 8, 8, 8), &device)?.to_dtype(DType::F16)?;
            let weight =
                Tensor::randn(0f32, 0.1, (8, 4, 1, 3, 3), &device)?.to_dtype(DType::F16)?;
            let bias = Tensor::zeros(8, DType::F16, &device)?;

            let y = conv3d_spatial_only_metal(
                &x,
                &weight,
                Some(&bias),
                3,
                3,
                1,
                1,
                1,
                1,
                6,
                6,
                1,
                8,
                1,
            )?;

            assert_eq!(y.dims(), &[2, 8, 8, 6, 6]);
            assert_eq!(y.dtype(), DType::F16);

            Ok(())
        }
    }
}
