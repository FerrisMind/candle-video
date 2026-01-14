#![allow(clippy::too_many_arguments)]
use candle_core::{DType, Device, Result, Tensor};

use super::cpu::Im2ColConfig;

pub fn is_cuda_tensor(tensor: &Tensor) -> bool {
    matches!(tensor.device(), Device::Cuda(_))
}

pub fn is_cuda_device(device: &Device) -> bool {
    matches!(device, Device::Cuda(_))
}

pub fn conv3d_cuda_forward(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    config: &Im2ColConfig,
    groups: usize,
    t_out: usize,
    h_out: usize,
    w_out: usize,
) -> Result<Tensor> {
    let out_c = weight.dims()[0];

    let y = conv3d_via_conv2d_slices(x, weight, config, groups, t_out, h_out, w_out)?;

    if let Some(bias) = bias {
        let bias = bias.reshape((1, out_c, 1, 1, 1))?;
        y.broadcast_add(&bias)
    } else {
        Ok(y)
    }
}

fn conv3d_via_conv2d_slices(
    x: &Tensor,
    weight: &Tensor,
    config: &Im2ColConfig,
    groups: usize,
    t_out: usize,
    h_out: usize,
    w_out: usize,
) -> Result<Tensor> {
    let (kt, kh, kw) = config.kernel;
    let (st, sh, _sw) = config.stride;
    let (dt, dh, dw) = config.dilation;

    let (batch, in_c, _t_pad, _h_pad, _w_pad) = x.dims5()?;
    let out_c = weight.dims()[0];

    if dt != 1 || dh != 1 || dw != 1 {
        return conv3d_cuda_forward_im2col(x, weight, config, groups, t_out, h_out, w_out);
    }

    let _ = (kh, kw);

    let mut result: Option<Tensor> = None;

    for ki_t in 0..kt {
        let weight_2d = weight.narrow(2, ki_t, 1)?.squeeze(2)?;

        let x_frames = if st == 1 {
            x.narrow(2, ki_t, t_out)?
        } else {
            let indices: Vec<u32> = (0..t_out).map(|to| (to * st + ki_t) as u32).collect();
            let indices_tensor = Tensor::new(&indices[..], x.device())?;
            x.index_select(&indices_tensor, 2)?
        };

        let (_, _, t_actual, h_in, w_in) = x_frames.dims5()?;
        let x_2d =
            x_frames
                .permute((0, 2, 1, 3, 4))?
                .reshape((batch * t_actual, in_c, h_in, w_in))?;

        let y_2d = x_2d.conv2d(&weight_2d, 0, sh, 1, groups)?;

        let y_5d = y_2d
            .reshape((batch, t_actual, out_c, h_out, w_out))?
            .permute((0, 2, 1, 3, 4))?;
        result = Some(match result {
            Some(acc) => acc.add(&y_5d)?,
            None => y_5d,
        });
    }

    result.ok_or_else(|| candle_core::Error::Msg("No temporal slices processed".to_string()))
}

fn conv3d_cuda_forward_im2col(
    x: &Tensor,
    weight: &Tensor,
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

    let col = im2col_3d_cuda(x, config, t_out, h_out, w_out)?;

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

    Ok(y)
}

fn im2col_3d_cuda(
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

    let mut all_patches = Vec::with_capacity(kt);

    for ki_t in 0..kt {
        let mut spatial_patches = Vec::with_capacity(kh * kw);

        for ki_h in 0..kh {
            for ki_w in 0..kw {
                let mut output_patches = Vec::with_capacity(spatial_out);

                for to in 0..t_out {
                    let ti = to * st + ki_t * dt;
                    for ho in 0..h_out {
                        let hi = ho * sh + ki_h * dh;
                        for wo in 0..w_out {
                            let wi = wo * sw + ki_w * dw;

                            let slice = x
                                .narrow(2, ti, 1)?
                                .narrow(3, hi, 1)?
                                .narrow(4, wi, 1)?
                                .reshape((batch, in_c))?;
                            output_patches.push(slice.unsqueeze(1)?);
                        }
                    }
                }

                let kernel_elem_patches =
                    Tensor::cat(&output_patches.iter().collect::<Vec<_>>(), 1)?;
                spatial_patches.push(kernel_elem_patches);
            }
        }

        let temporal_slice_patches = Tensor::cat(&spatial_patches.iter().collect::<Vec<_>>(), 2)?;
        all_patches.push(temporal_slice_patches);
    }

    let result = Tensor::cat(&all_patches.iter().collect::<Vec<_>>(), 2)?;
    debug_assert_eq!(result.dims(), &[batch, spatial_out, patch_size]);

    Ok(result)
}

pub fn conv3d_pointwise_cuda(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
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

pub fn conv3d_spatial_only_cuda(
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

pub fn conv3d_temporal_only_cuda(
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

pub fn is_supported_cuda_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F64 | DType::BF16 | DType::F16)
}

pub fn supported_cuda_dtypes() -> Vec<DType> {
    vec![DType::F32, DType::F64, DType::BF16, DType::F16]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_supported_cuda_dtype() {
        assert!(is_supported_cuda_dtype(DType::F32));
        assert!(is_supported_cuda_dtype(DType::F64));
        assert!(is_supported_cuda_dtype(DType::BF16));
        assert!(is_supported_cuda_dtype(DType::F16));
    }

    #[test]
    fn test_supported_cuda_dtypes() {
        let dtypes = supported_cuda_dtypes();
        assert_eq!(dtypes.len(), 4);
        assert!(dtypes.contains(&DType::F32));
        assert!(dtypes.contains(&DType::F64));
        assert!(dtypes.contains(&DType::BF16));
        assert!(dtypes.contains(&DType::F16));
    }

    #[test]
    fn test_is_cuda_device() {
        let cpu = Device::Cpu;
        assert!(!is_cuda_device(&cpu));
    }
}
