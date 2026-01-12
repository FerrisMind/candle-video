//! CPU backend for Conv3d using im2col + GEMM algorithm.
//!
//! This module provides an efficient CPU implementation of 3D convolution
//! using the im2col (image to column) transformation followed by matrix
//! multiplication (GEMM).
//!
//! The algorithm works as follows:
//! 1. Extract patches from the input tensor and arrange them as columns
//! 2. Reshape the weight tensor to a 2D matrix
//! 3. Perform matrix multiplication
//! 4. Reshape the result to the output tensor shape
//!
//! This approach is efficient because it leverages highly optimized BLAS
//! routines for matrix multiplication.

use candle_core::{Result, Tensor};

/// Configuration for im2col operation.
#[derive(Clone, Copy, Debug)]
pub struct Im2ColConfig {
    /// Kernel size (temporal, height, width)
    pub kernel: (usize, usize, usize),
    /// Stride (temporal, height, width)
    pub stride: (usize, usize, usize),
    /// Dilation (temporal, height, width)
    pub dilation: (usize, usize, usize),
}

impl Im2ColConfig {
    /// Create a new Im2ColConfig.
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

/// Perform im2col transformation for 3D convolution.
///
/// Extracts patches from the input tensor and arranges them as columns
/// for efficient matrix multiplication.
///
/// # Arguments
/// * `x` - Input tensor of shape (batch, in_channels, T_padded, H_padded, W_padded)
/// * `config` - Im2col configuration (kernel, stride, dilation)
/// * `t_out` - Output temporal dimension
/// * `h_out` - Output height dimension
/// * `w_out` - Output width dimension
///
/// # Returns
/// Column tensor of shape (batch, t_out * h_out * w_out, in_channels * kt * kh * kw)
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

    // Total number of output spatial positions
    let spatial_out = t_out * h_out * w_out;
    // Size of each patch (flattened)
    let patch_size = in_c * kt * kh * kw;

    // Collect all patches
    let mut patches = Vec::with_capacity(spatial_out);

    for to in 0..t_out {
        for ho in 0..h_out {
            for wo in 0..w_out {
                // Extract patch at this output position
                let patch = extract_patch_3d(x, to, ho, wo, kt, kh, kw, st, sh, sw, dt, dh, dw)?;
                // patch shape: (batch, in_c * kt * kh * kw)
                patches.push(patch.unsqueeze(1)?); // (batch, 1, patch_size)
            }
        }
    }

    // Concatenate all patches: (batch, spatial_out, patch_size)
    let result = Tensor::cat(&patches.iter().collect::<Vec<_>>(), 1)?;
    debug_assert_eq!(result.dims(), &[batch, spatial_out, patch_size]);

    Ok(result)
}

/// Extract a single patch from the input tensor.
///
/// # Arguments
/// * `x` - Input tensor of shape (batch, in_channels, T, H, W)
/// * `to`, `ho`, `wo` - Output position indices
/// * `kt`, `kh`, `kw` - Kernel sizes
/// * `st`, `sh`, `sw` - Strides
/// * `dt`, `dh`, `dw` - Dilations
///
/// # Returns
/// Patch tensor of shape (batch, in_channels * kt * kh * kw)
///
/// The patch is flattened in C, T, H, W order (channel-first) to match
/// PyTorch's weight layout: (out_c, in_c, kt, kh, kw).
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

    // Extract the 3D patch: x[:, :, t:t+kt, h:h+kh, w:w+kw]
    // We need to handle dilation and stride properly
    
    // Collect slices in C, T, H, W order to match PyTorch's weight layout
    let mut slices = Vec::with_capacity(in_c * kt * kh * kw);

    for c in 0..in_c {
        for ki_t in 0..kt {
            let ti = to * st + ki_t * dt;
            for ki_h in 0..kh {
                let hi = ho * sh + ki_h * dh;
                for ki_w in 0..kw {
                    let wi = wo * sw + ki_w * dw;
                    // x[:, c, ti, hi, wi] -> (batch,)
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

    // Concatenate along last dimension: (batch, in_c * kt * kh * kw)
    Tensor::cat(&slices.iter().collect::<Vec<_>>(), 1)
}

/// Perform 3D convolution using im2col + GEMM on CPU.
///
/// # Arguments
/// * `x` - Padded input tensor of shape (batch, in_channels, T_padded, H_padded, W_padded)
/// * `weight` - Weight tensor of shape (out_channels, in_channels/groups, kt, kh, kw)
/// * `bias` - Optional bias tensor of shape (out_channels,)
/// * `config` - Im2col configuration
/// * `groups` - Number of groups for grouped convolution
/// * `t_out`, `h_out`, `w_out` - Output dimensions
///
/// # Returns
/// Output tensor of shape (batch, out_channels, t_out, h_out, w_out)
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

    // Perform im2col transformation
    let col = im2col_3d(x, config, t_out, h_out, w_out)?;
    // col shape: (batch, t_out * h_out * w_out, in_c * kt * kh * kw)

    let spatial = t_out * h_out * w_out;
    let patch_size = in_c_per_group * kt * kh * kw;

    let y = if groups == 1 {
        // Simple case: no groups
        // Reshape weight to (out_c, in_c * kt * kh * kw) then transpose
        let weight_2d = weight
            .reshape((out_c, patch_size))?
            .t()?
            .contiguous()?; // (patch_size, out_c)

        // col: (batch, spatial, patch_size)
        // Reshape to 2D for matmul: (batch * spatial, patch_size)
        let col_2d = col.reshape((batch * spatial, patch_size))?;

        // Matmul: (batch * spatial, patch_size) @ (patch_size, out_c) = (batch * spatial, out_c)
        let y_2d = col_2d.matmul(&weight_2d)?;

        // Reshape to output: (batch, t_out, h_out, w_out, out_c) -> (batch, out_c, t_out, h_out, w_out)
        y_2d.reshape((batch, t_out, h_out, w_out, out_c))?
            .permute((0, 4, 1, 2, 3))?
    } else {
        // Grouped convolution
        let mut outputs = Vec::with_capacity(groups);

        for g in 0..groups {
            // Extract group's columns
            let col_g = col.narrow(2, g * patch_size, patch_size)?;
            let col_g_2d = col_g.reshape((batch * spatial, patch_size))?;

            // Extract group's weights
            let weight_g = weight
                .narrow(0, g * out_c_per_group, out_c_per_group)?
                .reshape((out_c_per_group, patch_size))?
                .t()?
                .contiguous()?;

            // Matmul for this group
            let y_g = col_g_2d.matmul(&weight_g)?; // (batch * spatial, out_c_per_group)
            outputs.push(y_g.reshape((batch, spatial, out_c_per_group))?);
        }

        // Concatenate groups along channel dimension
        let y = Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 2)?;
        y.reshape((batch, t_out, h_out, w_out, out_c))?
            .permute((0, 4, 1, 2, 3))?
    };

    // Add bias if present
    if let Some(bias) = bias {
        let bias = bias.reshape((1, out_c, 1, 1, 1))?;
        y.broadcast_add(&bias)
    } else {
        Ok(y)
    }
}

// =============================================================================
// Optimized Special Cases
// =============================================================================

/// Optimized pointwise (1x1x1) convolution.
///
/// For 1x1x1 kernels with stride 1, we can use a simple reshape + matmul
/// which is more efficient than the general im2col approach.
///
/// # Arguments
/// * `x` - Input tensor of shape (batch, in_channels, T, H, W)
/// * `weight` - Weight tensor of shape (out_channels, in_channels, 1, 1, 1)
/// * `bias` - Optional bias tensor of shape (out_channels,)
///
/// # Returns
/// Output tensor of shape (batch, out_channels, T, H, W)
pub fn conv3d_pointwise(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    let (batch, in_c, t, h, w) = x.dims5()?;
    let out_c = weight.dims()[0];

    // Reshape input: (batch, in_c, t, h, w) -> (batch * t * h * w, in_c)
    let x_2d = x
        .permute((0, 2, 3, 4, 1))?
        .reshape((batch * t * h * w, in_c))?;

    // Reshape weight: (out_c, in_c, 1, 1, 1) -> (out_c, in_c)
    let weight_2d = weight.reshape((out_c, in_c))?;

    // Matmul: (batch * t * h * w, in_c) @ (in_c, out_c) = (batch * t * h * w, out_c)
    let y = x_2d.matmul(&weight_2d.t()?)?;

    // Reshape output: (batch * t * h * w, out_c) -> (batch, out_c, t, h, w)
    let y = y
        .reshape((batch, t, h, w, out_c))?
        .permute((0, 4, 1, 2, 3))?;

    // Add bias if present
    if let Some(bias) = bias {
        let bias = bias.reshape((1, out_c, 1, 1, 1))?;
        y.broadcast_add(&bias)
    } else {
        Ok(y)
    }
}

/// Optimized temporal-only (kt, 1, 1) convolution.
///
/// For kernels with kh=kw=1, we can treat this as a 1D convolution
/// along the temporal dimension, which is more efficient.
///
/// # Arguments
/// * `x` - Input tensor of shape (batch, in_channels, T, H, W)
/// * `weight` - Weight tensor of shape (out_channels, in_channels, kt, 1, 1)
/// * `bias` - Optional bias tensor of shape (out_channels,)
/// * `kt` - Kernel size in temporal dimension
/// * `st` - Stride in temporal dimension
/// * `dt` - Dilation in temporal dimension
/// * `t_out` - Output temporal dimension
/// * `sh`, `sw` - Strides in spatial dimensions (for subsampling)
/// * `h_out`, `w_out` - Output spatial dimensions
///
/// # Returns
/// Output tensor of shape (batch, out_channels, t_out, h_out, w_out)
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

    // Handle spatial stride by selecting positions
    let (x_strided, h_actual, w_actual) = if sh > 1 || sw > 1 {
        // Select spatial positions at stride intervals
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

    // Reshape input: (batch, in_c, t, h, w) -> (batch * h * w, in_c, t)
    let x_3d = x_strided
        .permute((0, 3, 4, 1, 2))? // (batch, h, w, in_c, t)
        .reshape((batch * h_actual * w_actual, in_c, x_strided.dims()[2]))?;

    // Reshape weight: (out_c, in_c, kt, 1, 1) -> (out_c, in_c, kt)
    let weight_3d = weight.reshape((out_c, in_c, kt))?;

    // Perform 1D convolution along temporal dimension using im2col
    let col = im2col_1d(&x_3d, kt, st, dt, t_out)?;
    // col shape: (batch * h * w, t_out, in_c * kt)

    // Reshape weight for matmul: (out_c, in_c * kt) -> (in_c * kt, out_c)
    let weight_2d = weight_3d
        .reshape((out_c, in_c * kt))?
        .t()?
        .contiguous()?;

    // Matmul: (batch * h * w * t_out, in_c * kt) @ (in_c * kt, out_c)
    let col_2d = col.reshape((batch * h_actual * w_actual * t_out, in_c * kt))?;
    let y_2d = col_2d.matmul(&weight_2d)?;

    // Reshape output: (batch * h * w, t_out, out_c) -> (batch, out_c, t_out, h, w)
    let y = y_2d
        .reshape((batch, h_actual, w_actual, t_out, out_c))?
        .permute((0, 4, 3, 1, 2))?;

    // Add bias if present
    if let Some(bias) = bias {
        let bias = bias.reshape((1, out_c, 1, 1, 1))?;
        y.broadcast_add(&bias)
    } else {
        Ok(y)
    }
}

/// Im2col for 1D convolution (helper for temporal-only conv).
/// 
/// Extracts patches in C, T order to match weight layout.
fn im2col_1d(
    x: &Tensor,
    k: usize,
    s: usize,
    d: usize,
    out_len: usize,
) -> Result<Tensor> {
    let (batch, c, _t) = x.dims3()?;

    let mut patches = Vec::with_capacity(out_len);

    for o in 0..out_len {
        // Collect slices in C, T order
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

/// Optimized spatial-only (1, kh, kw) convolution.
///
/// For kernels with kt=1, we can treat this as a batched 2D convolution
/// where each temporal frame is processed independently.
///
/// # Arguments
/// * `x` - Input tensor of shape (batch, in_channels, T, H, W)
/// * `weight` - Weight tensor of shape (out_channels, in_channels, 1, kh, kw)
/// * `bias` - Optional bias tensor of shape (out_channels,)
/// * `kh`, `kw` - Kernel sizes in spatial dimensions
/// * `sh`, `sw` - Strides in spatial dimensions
/// * `dh`, `dw` - Dilations in spatial dimensions
/// * `h_out`, `w_out` - Output spatial dimensions
/// * `st` - Stride in temporal dimension (for subsampling)
/// * `t_out` - Output temporal dimension
///
/// # Returns
/// Output tensor of shape (batch, out_channels, t_out, h_out, w_out)
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

    // Handle temporal stride by selecting frames at stride positions
    let x_strided = if st > 1 {
        // Select frames at positions 0, st, 2*st, ...
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

    // Reshape input: (batch, in_c, t, h, w) -> (batch * t, in_c, h, w)
    let x_4d = x_strided
        .permute((0, 2, 1, 3, 4))? // (batch, t, in_c, h, w)
        .reshape((batch * t_actual, in_c, x.dims()[3], x.dims()[4]))?;

    // Reshape weight: (out_c, in_c, 1, kh, kw) -> (out_c, in_c, kh, kw)
    let weight_4d = weight.reshape((out_c, in_c, kh, kw))?;

    // Perform 2D convolution using im2col
    let col = im2col_2d(&x_4d, kh, kw, sh, sw, dh, dw, h_out, w_out)?;
    // col shape: (batch * t, h_out * w_out, in_c * kh * kw)

    // Reshape weight for matmul: (out_c, in_c * kh * kw) -> (in_c * kh * kw, out_c)
    let weight_2d = weight_4d
        .reshape((out_c, in_c * kh * kw))?
        .t()?
        .contiguous()?;

    // Matmul
    let spatial = h_out * w_out;
    let col_2d = col.reshape((batch * t_actual * spatial, in_c * kh * kw))?;
    let y_2d = col_2d.matmul(&weight_2d)?;

    // Reshape output: (batch * t, h_out, w_out, out_c) -> (batch, out_c, t, h_out, w_out)
    let y = y_2d
        .reshape((batch, t_actual, h_out, w_out, out_c))?
        .permute((0, 4, 1, 2, 3))?;

    // Add bias if present
    if let Some(bias) = bias {
        let bias = bias.reshape((1, out_c, 1, 1, 1))?;
        y.broadcast_add(&bias)
    } else {
        Ok(y)
    }
}

/// Im2col for 2D convolution (helper for spatial-only conv).
///
/// Extracts patches in C, H, W order to match weight layout.
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
            // Collect slices in C, H, W order
            let mut slices = Vec::with_capacity(c * kh * kw);
            for ci in 0..c {
                for ki_h in 0..kh {
                    let hi = ho * sh + ki_h * dh;
                    for ki_w in 0..kw {
                        let wi = wo * sw + ki_w * dw;
                        let slice = x.narrow(1, ci, 1)?.narrow(2, hi, 1)?.narrow(3, wi, 1)?.reshape((batch, 1))?;
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

// =============================================================================
// Tests
// =============================================================================

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

        // Output dimensions: (4-2)/1+1 = 3 for each dimension
        let col = im2col_3d(&x, &config, 3, 3, 3)?;

        // Expected shape: (batch, 3*3*3, 2*2*2*2) = (1, 27, 16)
        assert_eq!(col.dims(), &[1, 27, 16]);

        Ok(())
    }

    #[test]
    fn test_im2col_3d_stride() -> Result<()> {
        let x = create_test_tensor_5d((1, 2, 6, 6, 6))?;
        let config = Im2ColConfig::new((2, 2, 2), (2, 2, 2), (1, 1, 1));

        // Output dimensions: (6-2)/2+1 = 3 for each dimension
        let col = im2col_3d(&x, &config, 3, 3, 3)?;

        assert_eq!(col.dims(), &[1, 27, 16]);

        Ok(())
    }

    #[test]
    fn test_conv3d_cpu_forward_basic() -> Result<()> {
        let device = Device::Cpu;
        let x = create_random_tensor_5d((2, 4, 4, 8, 8))?;

        // Create weight: (out_c, in_c, kt, kh, kw)
        let weight = Tensor::randn(0f32, 0.1, (8, 4, 3, 3, 3), &device)?;
        let bias = Tensor::zeros(8, candle_core::DType::F32, &device)?;

        let config = Im2ColConfig::new((3, 3, 3), (1, 1, 1), (1, 1, 1));

        // With padding applied externally, output dims would be:
        // t_out = 4-3+1 = 2, h_out = 8-3+1 = 6, w_out = 8-3+1 = 6
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

        // t_out = (8-3)/1+1 = 6, h_out = w_out = 8 (stride=1)
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

        // h_out = w_out = (8-3)/1+1 = 6, t_out = 8 (stride=1)
        let y = conv3d_spatial_only(&x, &weight, Some(&bias), 3, 3, 1, 1, 1, 1, 6, 6, 1, 8)?;

        assert_eq!(y.dims(), &[2, 8, 8, 6, 6]);

        Ok(())
    }

    #[test]
    fn test_conv3d_grouped() -> Result<()> {
        let device = Device::Cpu;
        let x = create_random_tensor_5d((1, 4, 4, 4, 4))?;

        // Grouped conv: 4 input channels, 8 output channels, 2 groups
        // Each group: 2 in_c -> 4 out_c
        let weight = Tensor::randn(0f32, 0.1, (8, 2, 2, 2, 2), &device)?;
        let bias = Tensor::zeros(8, candle_core::DType::F32, &device)?;

        let config = Im2ColConfig::new((2, 2, 2), (1, 1, 1), (1, 1, 1));

        // t_out = h_out = w_out = (4-2)/1+1 = 3
        let y = conv3d_cpu_forward(&x, &weight, Some(&bias), &config, 2, 3, 3, 3)?;

        assert_eq!(y.dims(), &[1, 8, 3, 3, 3]);

        Ok(())
    }
}
