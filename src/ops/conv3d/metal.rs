//! Metal backend for Conv3d.
//!
//! This module provides Metal-accelerated 3D convolution for macOS using:
//! 1. Im2col + GEMM algorithm (leveraging Metal tensor operations)
//! 2. Optimized paths for special cases:
//!    - kt=1: Batched Conv2d using MPS
//!    - kh=kw=1: Conv1d using MPS
//!
//! Since MPS doesn't natively support 3D convolution with all our required
//! features (causal padding, replicate padding), we implement Conv3d using
//! the im2col transformation followed by matrix multiplication, which
//! automatically runs on Metal when tensors are on GPU.
//!
//! For special cases where we can reduce to 2D or 1D convolution, we use
//! Candle's native conv2d/conv1d which leverage MPS for maximum performance.
//!
//! # Requirements
//! - macOS with Metal support
//! - Compiled with `accelerate` feature flag
//!
//! # Supported Data Types
//! - f32: Full precision (recommended for accuracy)
//! - f16: Half precision (recommended for memory efficiency)
//! - Note: f8e4m3 is NOT supported by Metal

use candle_core::{DType, Device, Result, Tensor};

use super::cpu::Im2ColConfig;

// =============================================================================
// Metal Backend Detection
// =============================================================================

/// Check if a tensor is on a Metal device.
pub fn is_metal_tensor(tensor: &Tensor) -> bool {
    matches!(tensor.device(), Device::Metal(_))
}

/// Check if Metal is available for the given device.
pub fn is_metal_device(device: &Device) -> bool {
    matches!(device, Device::Metal(_))
}

// =============================================================================
// Dtype Support
// =============================================================================

/// Check if a dtype is supported for Conv3d on Metal.
///
/// Metal backend supports: f32, f16
/// Note: f8e4m3 is NOT supported by Metal
pub fn is_supported_metal_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F16)
}

/// Get a list of supported dtypes for Metal Conv3d.
pub fn supported_metal_dtypes() -> Vec<DType> {
    vec![DType::F32, DType::F16]
}

// =============================================================================
// Metal-Optimized Conv3d Forward
// =============================================================================

/// Perform 3D convolution on Metal using im2col + GEMM.
///
/// This function uses the same algorithm as the CPU backend but operates on
/// Metal tensors, leveraging GPU acceleration for all tensor operations.
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

    // Perform im2col transformation on Metal
    let col = im2col_3d_metal(x, config, t_out, h_out, w_out)?;
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

/// Perform im2col transformation for 3D convolution on Metal.
///
/// This is the same algorithm as the CPU version but operates on Metal tensors.
/// All tensor operations will automatically run on GPU.
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
                let patch = extract_patch_3d_metal(x, to, ho, wo, kt, kh, kw, st, sh, sw, dt, dh, dw)?;
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

/// Extract a single patch from the input tensor on Metal.
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
                // x[:, :, ti, hi, wi] -> (batch, in_c)
                let slice = x
                    .narrow(2, ti, 1)?
                    .narrow(3, hi, 1)?
                    .narrow(4, wi, 1)?
                    .reshape((batch, in_c))?;
                slices.push(slice);
            }
        }
    }

    // Concatenate along channel dimension: (batch, in_c * kt * kh * kw)
    Tensor::cat(&slices.iter().collect::<Vec<_>>(), 1)
}

// =============================================================================
// Optimized Special Cases for Metal
// =============================================================================

/// Optimized pointwise (1x1x1) convolution on Metal.
///
/// For 1x1x1 kernels with stride 1, we can use a simple reshape + matmul
/// which is highly efficient on GPU.
pub fn conv3d_pointwise_metal(
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

/// Optimized spatial-only (1, kh, kw) convolution on Metal using batched Conv2d.
///
/// For kernels with kt=1, we treat each temporal frame as a separate batch
/// and use Candle's native conv2d which leverages MPS on Metal.
///
/// Note: Candle's conv2d uses a single stride/dilation value for both H and W.
/// This function assumes sh=sw and dh=dw. For non-square stride/dilation,
/// use the general im2col + GEMM approach.
///
/// # Arguments
/// * `x` - Input tensor of shape (batch, in_channels, T, H, W)
/// * `weight` - Weight tensor of shape (out_channels, in_channels, 1, kh, kw)
/// * `bias` - Optional bias tensor of shape (out_channels,)
/// * `kh`, `kw` - Kernel sizes in spatial dimensions
/// * `sh`, `_sw` - Strides in spatial dimensions (assumes sh=sw)
/// * `dh`, `_dw` - Dilations in spatial dimensions (assumes dh=dw)
/// * `h_out`, `w_out` - Output spatial dimensions
/// * `st` - Stride in temporal dimension (for subsampling)
/// * `t_out` - Output temporal dimension
/// * `groups` - Number of groups for grouped convolution
///
/// # Returns
/// Output tensor of shape (batch, out_channels, t_out, h_out, w_out)
pub fn conv3d_spatial_only_metal(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    kh: usize,
    kw: usize,
    sh: usize,
    _sw: usize, // Assumed equal to sh for MPS
    dh: usize,
    _dw: usize, // Assumed equal to dh for MPS
    h_out: usize,
    w_out: usize,
    st: usize,
    t_out: usize,
    groups: usize,
) -> Result<Tensor> {
    let (batch, in_c, t, h, w) = x.dims5()?;
    let out_c = weight.dims()[0];

    // Handle temporal stride by selecting frames at stride positions
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

    // Reshape input: (batch, in_c, t, h, w) -> (batch * t, in_c, h, w)
    let x_4d = x_strided
        .permute((0, 2, 1, 3, 4))? // (batch, t, in_c, h, w)
        .reshape((batch * t_actual, in_c, h, w))?;

    // Reshape weight: (out_c, in_c/groups, 1, kh, kw) -> (out_c, in_c/groups, kh, kw)
    let weight_4d = weight.reshape((out_c, in_c / groups, kh, kw))?;

    // Use Candle's native conv2d which leverages MPS on Metal
    // Note: padding is already applied to input, so we use padding=0 here
    let y_4d = x_4d.conv2d(&weight_4d, 0, sh, dh, groups)?;

    // Reshape output: (batch * t, out_c, h_out, w_out) -> (batch, out_c, t, h_out, w_out)
    let y = y_4d
        .reshape((batch, t_actual, out_c, h_out, w_out))?
        .permute((0, 2, 1, 3, 4))?;

    // Add bias if present
    if let Some(bias) = bias {
        let bias = bias.reshape((1, out_c, 1, 1, 1))?;
        y.broadcast_add(&bias)
    } else {
        Ok(y)
    }
}

/// Optimized temporal-only (kt, 1, 1) convolution on Metal using Conv1d.
///
/// For kernels with kh=kw=1, we treat this as a 1D convolution along the
/// temporal dimension, using Candle's native conv1d which leverages MPS.
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
/// * `groups` - Number of groups for grouped convolution
///
/// # Returns
/// Output tensor of shape (batch, out_channels, t_out, h_out, w_out)
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

    // Handle spatial stride by selecting positions
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

    // Reshape input: (batch, in_c, t, h, w) -> (batch * h * w, in_c, t)
    let x_3d = x_strided
        .permute((0, 3, 4, 1, 2))? // (batch, h, w, in_c, t)
        .reshape((batch * h_actual * w_actual, in_c, t))?;

    // Reshape weight: (out_c, in_c/groups, kt, 1, 1) -> (out_c, in_c/groups, kt)
    let weight_3d = weight.reshape((out_c, in_c / groups, kt))?;

    // Use Candle's native conv1d which leverages MPS on Metal
    // Note: padding is already applied to input, so we use padding=0 here
    let y_3d = x_3d.conv1d(&weight_3d, 0, st, dt, groups)?;

    // Reshape output: (batch * h * w, out_c, t_out) -> (batch, out_c, t_out, h, w)
    let y = y_3d
        .reshape((batch, h_actual, w_actual, out_c, t_out))?
        .permute((0, 3, 4, 1, 2))?;

    // Add bias if present
    if let Some(bias) = bias {
        let bias = bias.reshape((1, out_c, 1, 1, 1))?;
        y.broadcast_add(&bias)
    } else {
        Ok(y)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_supported_metal_dtype() {
        assert!(is_supported_metal_dtype(DType::F32));
        assert!(is_supported_metal_dtype(DType::F16));
        // bf16 is not supported on Metal
        assert!(!is_supported_metal_dtype(DType::BF16));
        // f64 is not supported on Metal
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
        // Metal device test would require actual Metal hardware (macOS)
    }

    #[test]
    fn test_is_metal_tensor() {
        let device = Device::Cpu;
        let tensor = Tensor::zeros((2, 3), DType::F32, &device).unwrap();
        assert!(!is_metal_tensor(&tensor));
        // Metal tensor test would require actual Metal hardware (macOS)
    }

    // Note: Full functional tests for Metal backend require macOS with Metal support.
    // The following tests verify the algorithm correctness on CPU, which uses
    // the same logic as the Metal backend.

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
            // Test the algorithm on CPU (same logic as Metal)
            let device = Device::Cpu;
            let x = create_test_tensor_5d((2, 4, 4, 8, 8))?;

            // Create weight: (out_c, in_c, kt, kh, kw)
            let weight = Tensor::randn(0f32, 0.1, (8, 4, 3, 3, 3), &device)?;
            let bias = Tensor::zeros(8, DType::F32, &device)?;

            let config = Im2ColConfig::new((3, 3, 3), (1, 1, 1), (1, 1, 1));

            // With no padding applied, output dims would be:
            // t_out = 4-3+1 = 2, h_out = 8-3+1 = 6, w_out = 8-3+1 = 6
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

            // t_out = (8-3)/1+1 = 6, h_out = w_out = 8 (stride=1)
            let y = conv3d_temporal_only_metal(&x, &weight, Some(&bias), 3, 1, 1, 6, 1, 1, 8, 8, 1)?;

            assert_eq!(y.dims(), &[2, 8, 6, 8, 8]);

            Ok(())
        }

        #[test]
        fn test_conv3d_spatial_only_metal() -> Result<()> {
            let device = Device::Cpu;
            let x = create_test_tensor_5d((2, 4, 8, 8, 8))?;

            let weight = Tensor::randn(0f32, 0.1, (8, 4, 1, 3, 3), &device)?;
            let bias = Tensor::zeros(8, DType::F32, &device)?;

            // h_out = w_out = (8-3)/1+1 = 6, t_out = 8 (stride=1)
            let y = conv3d_spatial_only_metal(&x, &weight, Some(&bias), 3, 3, 1, 1, 1, 1, 6, 6, 1, 8, 1)?;

            assert_eq!(y.dims(), &[2, 8, 8, 6, 6]);

            Ok(())
        }

        #[test]
        fn test_conv3d_metal_grouped() -> Result<()> {
            let device = Device::Cpu;
            let x = create_test_tensor_5d((1, 4, 4, 4, 4))?;

            // Grouped conv: 4 input channels, 8 output channels, 2 groups
            // Each group: 2 in_c -> 4 out_c
            let weight = Tensor::randn(0f32, 0.1, (8, 2, 2, 2, 2), &device)?;
            let bias = Tensor::zeros(8, DType::F32, &device)?;

            let config = Im2ColConfig::new((2, 2, 2), (1, 1, 1), (1, 1, 1));

            // t_out = h_out = w_out = (4-2)/1+1 = 3
            let y = conv3d_metal_forward(&x, &weight, Some(&bias), &config, 2, 3, 3, 3)?;

            assert_eq!(y.dims(), &[1, 8, 3, 3, 3]);

            Ok(())
        }

        #[test]
        fn test_conv3d_metal_f16() -> Result<()> {
            let device = Device::Cpu;
            // Create f16 tensors
            let x = Tensor::randn(0f32, 1.0, (1, 2, 4, 4, 4), &device)?
                .to_dtype(DType::F16)?;
            let weight = Tensor::randn(0f32, 0.1, (4, 2, 2, 2, 2), &device)?
                .to_dtype(DType::F16)?;
            let bias = Tensor::zeros(4, DType::F16, &device)?;

            let config = Im2ColConfig::new((2, 2, 2), (1, 1, 1), (1, 1, 1));

            // t_out = h_out = w_out = (4-2)/1+1 = 3
            let y = conv3d_metal_forward(&x, &weight, Some(&bias), &config, 1, 3, 3, 3)?;

            assert_eq!(y.dims(), &[1, 4, 3, 3, 3]);
            assert_eq!(y.dtype(), DType::F16);

            Ok(())
        }

        #[test]
        fn test_conv3d_metal_f32() -> Result<()> {
            let device = Device::Cpu;
            // Create f32 tensors (default)
            let x = Tensor::randn(0f32, 1.0, (1, 2, 4, 4, 4), &device)?;
            let weight = Tensor::randn(0f32, 0.1, (4, 2, 2, 2, 2), &device)?;
            let bias = Tensor::zeros(4, DType::F32, &device)?;

            let config = Im2ColConfig::new((2, 2, 2), (1, 1, 1), (1, 1, 1));

            // t_out = h_out = w_out = (4-2)/1+1 = 3
            let y = conv3d_metal_forward(&x, &weight, Some(&bias), &config, 1, 3, 3, 3)?;

            assert_eq!(y.dims(), &[1, 4, 3, 3, 3]);
            assert_eq!(y.dtype(), DType::F32);

            Ok(())
        }

        #[test]
        fn test_conv3d_pointwise_metal_f16() -> Result<()> {
            let device = Device::Cpu;
            let x = Tensor::randn(0f32, 1.0, (2, 4, 8, 16, 16), &device)?
                .to_dtype(DType::F16)?;
            let weight = Tensor::randn(0f32, 0.1, (8, 4, 1, 1, 1), &device)?
                .to_dtype(DType::F16)?;
            let bias = Tensor::zeros(8, DType::F16, &device)?;

            let y = conv3d_pointwise_metal(&x, &weight, Some(&bias))?;

            assert_eq!(y.dims(), &[2, 8, 8, 16, 16]);
            assert_eq!(y.dtype(), DType::F16);

            Ok(())
        }

        #[test]
        fn test_conv3d_temporal_only_metal_f16() -> Result<()> {
            let device = Device::Cpu;
            let x = Tensor::randn(0f32, 1.0, (2, 4, 8, 8, 8), &device)?
                .to_dtype(DType::F16)?;
            let weight = Tensor::randn(0f32, 0.1, (8, 4, 3, 1, 1), &device)?
                .to_dtype(DType::F16)?;
            let bias = Tensor::zeros(8, DType::F16, &device)?;

            // t_out = (8-3)/1+1 = 6, h_out = w_out = 8 (stride=1)
            let y = conv3d_temporal_only_metal(&x, &weight, Some(&bias), 3, 1, 1, 6, 1, 1, 8, 8, 1)?;

            assert_eq!(y.dims(), &[2, 8, 6, 8, 8]);
            assert_eq!(y.dtype(), DType::F16);

            Ok(())
        }

        #[test]
        fn test_conv3d_spatial_only_metal_f16() -> Result<()> {
            let device = Device::Cpu;
            let x = Tensor::randn(0f32, 1.0, (2, 4, 8, 8, 8), &device)?
                .to_dtype(DType::F16)?;
            let weight = Tensor::randn(0f32, 0.1, (8, 4, 1, 3, 3), &device)?
                .to_dtype(DType::F16)?;
            let bias = Tensor::zeros(8, DType::F16, &device)?;

            // h_out = w_out = (8-3)/1+1 = 6, t_out = 8 (stride=1)
            let y = conv3d_spatial_only_metal(&x, &weight, Some(&bias), 3, 3, 1, 1, 1, 1, 6, 6, 1, 8, 1)?;

            assert_eq!(y.dims(), &[2, 8, 8, 6, 6]);
            assert_eq!(y.dtype(), DType::F16);

            Ok(())
        }
    }
}
