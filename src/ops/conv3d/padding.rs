//! Padding utilities for Conv3d.
//!
//! Provides functions for applying temporal and spatial padding
//! with support for different padding modes (zeros, replicate).

use candle_core::{Result, Tensor};

use super::PaddingMode;

// =============================================================================
// Temporal Padding
// =============================================================================

/// Apply temporal padding to a 5D tensor.
///
/// # Arguments
/// * `x` - Input tensor of shape (B, C, T, H, W)
/// * `kt` - Kernel size in temporal dimension
/// * `is_causal` - If true, pad only left side (for autoregressive models)
/// * `mode` - Padding mode (Zeros or Replicate)
///
/// # Returns
/// Padded tensor with shape:
/// - Causal: (B, C, T + kt - 1, H, W)
/// - Non-causal: (B, C, T + 2 * ((kt - 1) / 2), H, W)
///
/// # Causal Padding
/// For causal convolution, we pad only the left side with (kt - 1) frames.
/// This ensures that output at time t depends only on inputs at times <= t.
///
/// # Non-Causal Padding
/// For non-causal convolution, we pad symmetrically with (kt - 1) / 2 frames
/// on each side to maintain the temporal dimension.
pub fn apply_temporal_padding(
    x: &Tensor,
    kt: usize,
    is_causal: bool,
    mode: PaddingMode,
) -> Result<Tensor> {
    if kt <= 1 {
        return Ok(x.clone());
    }

    if is_causal {
        apply_causal_temporal_padding(x, kt, mode)
    } else {
        let pad = (kt - 1) / 2;
        if pad == 0 {
            return Ok(x.clone());
        }
        apply_symmetric_temporal_padding(x, pad, mode)
    }
}

/// Apply causal temporal padding (left side only).
///
/// Pads the left side of the temporal dimension with (kt - 1) frames.
/// This ensures causality: output[t] depends only on input[0..=t].
///
/// # Arguments
/// * `x` - Input tensor of shape (B, C, T, H, W)
/// * `kt` - Kernel size in temporal dimension
/// * `mode` - Padding mode (Zeros or Replicate)
///
/// # Returns
/// Padded tensor of shape (B, C, T + kt - 1, H, W)
pub fn apply_causal_temporal_padding(
    x: &Tensor,
    kt: usize,
    mode: PaddingMode,
) -> Result<Tensor> {
    if kt <= 1 {
        return Ok(x.clone());
    }

    let pad_frames = kt - 1;
    match mode {
        PaddingMode::Zeros => {
            // Pad with zeros on the left side of temporal dimension (dim 2)
            x.pad_with_zeros(2, pad_frames, 0)
        }
        PaddingMode::Replicate => {
            // Replicate the first frame
            let (b, c, _t, h, w) = x.dims5()?;
            let first_frame = x.narrow(2, 0, 1)?;
            let padding = first_frame.broadcast_as((b, c, pad_frames, h, w))?;
            Tensor::cat(&[&padding, x], 2)
        }
    }
}

/// Apply symmetric temporal padding (both sides).
///
/// Pads both sides of the temporal dimension with `pad` frames each.
///
/// # Arguments
/// * `x` - Input tensor of shape (B, C, T, H, W)
/// * `pad` - Number of frames to pad on each side
/// * `mode` - Padding mode (Zeros or Replicate)
///
/// # Returns
/// Padded tensor of shape (B, C, T + 2 * pad, H, W)
pub fn apply_symmetric_temporal_padding(
    x: &Tensor,
    pad: usize,
    mode: PaddingMode,
) -> Result<Tensor> {
    if pad == 0 {
        return Ok(x.clone());
    }

    match mode {
        PaddingMode::Zeros => {
            // Pad with zeros on both sides of temporal dimension (dim 2)
            x.pad_with_zeros(2, pad, pad)
        }
        PaddingMode::Replicate => {
            // Replicate first and last frames
            let (b, c, t, h, w) = x.dims5()?;
            let first_frame = x.narrow(2, 0, 1)?;
            let last_frame = x.narrow(2, t - 1, 1)?;
            let pad_left = first_frame.broadcast_as((b, c, pad, h, w))?;
            let pad_right = last_frame.broadcast_as((b, c, pad, h, w))?;
            Tensor::cat(&[&pad_left, x, &pad_right], 2)
        }
    }
}

// =============================================================================
// Spatial Padding
// =============================================================================

/// Apply spatial padding to a 5D tensor.
///
/// Pads the height and width dimensions symmetrically.
///
/// # Arguments
/// * `x` - Input tensor of shape (B, C, T, H, W)
/// * `kh` - Kernel size in height dimension (used to compute padding if ph not provided)
/// * `kw` - Kernel size in width dimension (used to compute padding if pw not provided)
/// * `ph` - Padding for height dimension
/// * `pw` - Padding for width dimension
/// * `mode` - Padding mode (Zeros or Replicate)
///
/// # Returns
/// Padded tensor of shape (B, C, T, H + 2 * ph, W + 2 * pw)
pub fn apply_spatial_padding(
    x: &Tensor,
    _kh: usize,
    _kw: usize,
    ph: usize,
    pw: usize,
    mode: PaddingMode,
) -> Result<Tensor> {
    if ph == 0 && pw == 0 {
        return Ok(x.clone());
    }

    match mode {
        PaddingMode::Zeros => apply_spatial_padding_zeros(x, ph, pw),
        PaddingMode::Replicate => apply_spatial_padding_replicate(x, ph, pw),
    }
}

/// Apply spatial padding with zeros.
fn apply_spatial_padding_zeros(x: &Tensor, ph: usize, pw: usize) -> Result<Tensor> {
    // Pad height (dim 3) then width (dim 4)
    let x = if ph > 0 {
        x.pad_with_zeros(3, ph, ph)?
    } else {
        x.clone()
    };

    if pw > 0 {
        x.pad_with_zeros(4, pw, pw)
    } else {
        Ok(x)
    }
}

/// Apply spatial padding with replicate mode.
fn apply_spatial_padding_replicate(x: &Tensor, ph: usize, pw: usize) -> Result<Tensor> {
    let (b, c, t, h, w) = x.dims5()?;

    // Pad height (dim 3)
    let x = if ph > 0 {
        let top = x.narrow(3, 0, 1)?.broadcast_as((b, c, t, ph, w))?;
        let bottom = x.narrow(3, h - 1, 1)?.broadcast_as((b, c, t, ph, w))?;
        Tensor::cat(&[&top, x, &bottom], 3)?
    } else {
        x.clone()
    };

    // Pad width (dim 4)
    if pw > 0 {
        let (b, c, t, h_new, _) = x.dims5()?;
        let left = x.narrow(4, 0, 1)?.broadcast_as((b, c, t, h_new, pw))?;
        let right = x.narrow(4, w - 1, 1)?.broadcast_as((b, c, t, h_new, pw))?;
        Tensor::cat(&[&left, &x, &right], 4)
    } else {
        Ok(x)
    }
}

// =============================================================================
// Combined Padding
// =============================================================================

/// Apply full 3D padding (temporal + spatial) to a 5D tensor.
///
/// This is a convenience function that applies both temporal and spatial padding.
///
/// # Arguments
/// * `x` - Input tensor of shape (B, C, T, H, W)
/// * `kernel` - Kernel size (kt, kh, kw)
/// * `padding` - Explicit padding (pt, ph, pw) - used for spatial, temporal uses kernel
/// * `is_causal` - If true, use causal temporal padding
/// * `mode` - Padding mode for temporal dimension (Zeros or Replicate)
///
/// # Returns
/// Padded tensor
///
/// # Note
/// For causal mode, temporal padding uses the specified mode (typically Replicate),
/// but spatial padding always uses Zeros to match PyTorch nn.Conv3d behavior.
pub fn apply_full_padding(
    x: &Tensor,
    kernel: (usize, usize, usize),
    padding: (usize, usize, usize),
    is_causal: bool,
    mode: PaddingMode,
) -> Result<Tensor> {
    let (kt, kh, kw) = kernel;
    let (_pt, ph, pw) = padding;

    // Apply temporal padding with specified mode
    let x = apply_temporal_padding(x, kt, is_causal, mode)?;

    // Apply spatial padding with zeros (matches PyTorch nn.Conv3d behavior)
    // Even in causal mode, spatial padding should be zeros
    apply_spatial_padding(&x, kh, kw, ph, pw, PaddingMode::Zeros)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn create_test_tensor(shape: (usize, usize, usize, usize, usize)) -> Result<Tensor> {
        let (b, c, t, h, w) = shape;
        let device = Device::Cpu;
        // Create tensor with sequential values for easy verification
        let data: Vec<f32> = (0..(b * c * t * h * w) as u32).map(|x| x as f32).collect();
        Tensor::from_vec(data, (b, c, t, h, w), &device)
    }

    // =========================================================================
    // Temporal Padding Tests
    // =========================================================================

    #[test]
    fn test_causal_temporal_padding_zeros() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_causal_temporal_padding(&x, 3, PaddingMode::Zeros)?;

        // kt=3 means pad_frames = 2
        assert_eq!(padded.dims(), &[1, 2, 6, 3, 3]); // T: 4 + 2 = 6

        // Verify first 2 frames are zeros
        let first_two = padded.narrow(2, 0, 2)?;
        let sum: f32 = first_two.sum_all()?.to_scalar()?;
        assert_eq!(sum, 0.0);

        Ok(())
    }

    #[test]
    fn test_causal_temporal_padding_replicate() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_causal_temporal_padding(&x, 3, PaddingMode::Replicate)?;

        // kt=3 means pad_frames = 2
        assert_eq!(padded.dims(), &[1, 2, 6, 3, 3]); // T: 4 + 2 = 6

        // Verify first 2 frames equal the original first frame
        let first_frame_orig = x.narrow(2, 0, 1)?;
        let first_frame_padded = padded.narrow(2, 0, 1)?;
        let second_frame_padded = padded.narrow(2, 1, 1)?;

        let diff1: f32 = (first_frame_orig.clone() - first_frame_padded)?
            .abs()?
            .sum_all()?
            .to_scalar()?;
        let diff2: f32 = (first_frame_orig - second_frame_padded)?
            .abs()?
            .sum_all()?
            .to_scalar()?;

        assert_eq!(diff1, 0.0);
        assert_eq!(diff2, 0.0);

        Ok(())
    }

    #[test]
    fn test_causal_temporal_padding_kt1() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_causal_temporal_padding(&x, 1, PaddingMode::Zeros)?;

        // kt=1 means no padding needed
        assert_eq!(padded.dims(), x.dims());

        Ok(())
    }

    #[test]
    fn test_symmetric_temporal_padding_zeros() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_symmetric_temporal_padding(&x, 2, PaddingMode::Zeros)?;

        // pad=2 on each side
        assert_eq!(padded.dims(), &[1, 2, 8, 3, 3]); // T: 4 + 2 + 2 = 8

        // Verify first 2 and last 2 frames are zeros
        let first_two = padded.narrow(2, 0, 2)?;
        let last_two = padded.narrow(2, 6, 2)?;
        let sum_first: f32 = first_two.sum_all()?.to_scalar()?;
        let sum_last: f32 = last_two.sum_all()?.to_scalar()?;
        assert_eq!(sum_first, 0.0);
        assert_eq!(sum_last, 0.0);

        Ok(())
    }

    #[test]
    fn test_symmetric_temporal_padding_replicate() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_symmetric_temporal_padding(&x, 2, PaddingMode::Replicate)?;

        // pad=2 on each side
        assert_eq!(padded.dims(), &[1, 2, 8, 3, 3]); // T: 4 + 2 + 2 = 8

        // Verify first 2 frames equal original first frame
        let first_frame_orig = x.narrow(2, 0, 1)?;
        let first_frame_padded = padded.narrow(2, 0, 1)?;
        let diff: f32 = (first_frame_orig - first_frame_padded)?
            .abs()?
            .sum_all()?
            .to_scalar()?;
        assert_eq!(diff, 0.0);

        // Verify last 2 frames equal original last frame
        let last_frame_orig = x.narrow(2, 3, 1)?;
        let last_frame_padded = padded.narrow(2, 7, 1)?;
        let diff: f32 = (last_frame_orig - last_frame_padded)?
            .abs()?
            .sum_all()?
            .to_scalar()?;
        assert_eq!(diff, 0.0);

        Ok(())
    }

    #[test]
    fn test_apply_temporal_padding_causal() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_temporal_padding(&x, 3, true, PaddingMode::Zeros)?;

        // Causal with kt=3: pad_frames = 2 on left
        assert_eq!(padded.dims(), &[1, 2, 6, 3, 3]);

        Ok(())
    }

    #[test]
    fn test_apply_temporal_padding_non_causal() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_temporal_padding(&x, 3, false, PaddingMode::Zeros)?;

        // Non-causal with kt=3: pad = (3-1)/2 = 1 on each side
        assert_eq!(padded.dims(), &[1, 2, 6, 3, 3]); // T: 4 + 1 + 1 = 6

        Ok(())
    }

    // =========================================================================
    // Spatial Padding Tests
    // =========================================================================

    #[test]
    fn test_spatial_padding_zeros() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_spatial_padding(&x, 3, 3, 1, 1, PaddingMode::Zeros)?;

        // ph=1, pw=1 on each side
        assert_eq!(padded.dims(), &[1, 2, 4, 5, 5]); // H: 3+2=5, W: 3+2=5

        Ok(())
    }

    #[test]
    fn test_spatial_padding_replicate() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_spatial_padding(&x, 3, 3, 1, 1, PaddingMode::Replicate)?;

        // ph=1, pw=1 on each side
        assert_eq!(padded.dims(), &[1, 2, 4, 5, 5]); // H: 3+2=5, W: 3+2=5

        Ok(())
    }

    #[test]
    fn test_spatial_padding_height_only() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_spatial_padding(&x, 3, 1, 1, 0, PaddingMode::Zeros)?;

        assert_eq!(padded.dims(), &[1, 2, 4, 5, 3]); // H: 3+2=5, W: 3

        Ok(())
    }

    #[test]
    fn test_spatial_padding_width_only() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_spatial_padding(&x, 1, 3, 0, 1, PaddingMode::Zeros)?;

        assert_eq!(padded.dims(), &[1, 2, 4, 3, 5]); // H: 3, W: 3+2=5

        Ok(())
    }

    #[test]
    fn test_spatial_padding_no_padding() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_spatial_padding(&x, 1, 1, 0, 0, PaddingMode::Zeros)?;

        assert_eq!(padded.dims(), x.dims());

        Ok(())
    }

    // =========================================================================
    // Full Padding Tests
    // =========================================================================

    #[test]
    fn test_full_padding_causal() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_full_padding(
            &x,
            (3, 3, 3),  // kernel
            (0, 1, 1),  // padding (pt ignored for temporal, ph=1, pw=1)
            true,       // causal
            PaddingMode::Zeros,
        )?;

        // Causal temporal: T + kt - 1 = 4 + 2 = 6
        // Spatial: H + 2*ph = 3 + 2 = 5, W + 2*pw = 3 + 2 = 5
        assert_eq!(padded.dims(), &[1, 2, 6, 5, 5]);

        Ok(())
    }

    #[test]
    fn test_full_padding_non_causal() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_full_padding(
            &x,
            (3, 3, 3),  // kernel
            (1, 1, 1),  // padding
            false,      // non-causal
            PaddingMode::Zeros,
        )?;

        // Non-causal temporal: T + 2 * ((kt-1)/2) = 4 + 2 = 6
        // Spatial: H + 2*ph = 3 + 2 = 5, W + 2*pw = 3 + 2 = 5
        assert_eq!(padded.dims(), &[1, 2, 6, 5, 5]);

        Ok(())
    }
}
