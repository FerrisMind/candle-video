use candle_core::{Result, Tensor};

use super::PaddingMode;

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

pub fn apply_causal_temporal_padding(x: &Tensor, kt: usize, mode: PaddingMode) -> Result<Tensor> {
    if kt <= 1 {
        return Ok(x.clone());
    }

    let pad_frames = kt - 1;
    match mode {
        PaddingMode::Zeros => x.pad_with_zeros(2, pad_frames, 0),
        PaddingMode::Replicate => {
            let (b, c, _t, h, w) = x.dims5()?;
            let first_frame = x.narrow(2, 0, 1)?;
            let padding = first_frame.broadcast_as((b, c, pad_frames, h, w))?;
            Tensor::cat(&[&padding, x], 2)
        }
    }
}

pub fn apply_symmetric_temporal_padding(
    x: &Tensor,
    pad: usize,
    mode: PaddingMode,
) -> Result<Tensor> {
    if pad == 0 {
        return Ok(x.clone());
    }

    match mode {
        PaddingMode::Zeros => x.pad_with_zeros(2, pad, pad),
        PaddingMode::Replicate => {
            let (b, c, t, h, w) = x.dims5()?;
            let first_frame = x.narrow(2, 0, 1)?;
            let last_frame = x.narrow(2, t - 1, 1)?;
            let pad_left = first_frame.broadcast_as((b, c, pad, h, w))?;
            let pad_right = last_frame.broadcast_as((b, c, pad, h, w))?;
            Tensor::cat(&[&pad_left, x, &pad_right], 2)
        }
    }
}

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

fn apply_spatial_padding_zeros(x: &Tensor, ph: usize, pw: usize) -> Result<Tensor> {
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

fn apply_spatial_padding_replicate(x: &Tensor, ph: usize, pw: usize) -> Result<Tensor> {
    let (b, c, t, h, w) = x.dims5()?;

    let x = if ph > 0 {
        let top = x.narrow(3, 0, 1)?.broadcast_as((b, c, t, ph, w))?;
        let bottom = x.narrow(3, h - 1, 1)?.broadcast_as((b, c, t, ph, w))?;
        Tensor::cat(&[&top, x, &bottom], 3)?
    } else {
        x.clone()
    };

    if pw > 0 {
        let (b, c, t, h_new, _) = x.dims5()?;
        let left = x.narrow(4, 0, 1)?.broadcast_as((b, c, t, h_new, pw))?;
        let right = x.narrow(4, w - 1, 1)?.broadcast_as((b, c, t, h_new, pw))?;
        Tensor::cat(&[&left, &x, &right], 4)
    } else {
        Ok(x)
    }
}

pub fn apply_full_padding(
    x: &Tensor,
    kernel: (usize, usize, usize),
    padding: (usize, usize, usize),
    is_causal: bool,
    mode: PaddingMode,
) -> Result<Tensor> {
    let (kt, kh, kw) = kernel;
    let (_pt, ph, pw) = padding;

    let x = apply_temporal_padding(x, kt, is_causal, mode)?;

    apply_spatial_padding(&x, kh, kw, ph, pw, PaddingMode::Zeros)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn create_test_tensor(shape: (usize, usize, usize, usize, usize)) -> Result<Tensor> {
        let (b, c, t, h, w) = shape;
        let device = Device::Cpu;

        let data: Vec<f32> = (0..(b * c * t * h * w) as u32).map(|x| x as f32).collect();
        Tensor::from_vec(data, (b, c, t, h, w), &device)
    }

    #[test]
    fn test_causal_temporal_padding_zeros() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_causal_temporal_padding(&x, 3, PaddingMode::Zeros)?;

        assert_eq!(padded.dims(), &[1, 2, 6, 3, 3]);

        let first_two = padded.narrow(2, 0, 2)?;
        let sum: f32 = first_two.sum_all()?.to_scalar()?;
        assert_eq!(sum, 0.0);

        Ok(())
    }

    #[test]
    fn test_causal_temporal_padding_replicate() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_causal_temporal_padding(&x, 3, PaddingMode::Replicate)?;

        assert_eq!(padded.dims(), &[1, 2, 6, 3, 3]);

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

        assert_eq!(padded.dims(), x.dims());

        Ok(())
    }

    #[test]
    fn test_symmetric_temporal_padding_zeros() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_symmetric_temporal_padding(&x, 2, PaddingMode::Zeros)?;

        assert_eq!(padded.dims(), &[1, 2, 8, 3, 3]);

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

        assert_eq!(padded.dims(), &[1, 2, 8, 3, 3]);

        let first_frame_orig = x.narrow(2, 0, 1)?;
        let first_frame_padded = padded.narrow(2, 0, 1)?;
        let diff: f32 = (first_frame_orig - first_frame_padded)?
            .abs()?
            .sum_all()?
            .to_scalar()?;
        assert_eq!(diff, 0.0);

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

        assert_eq!(padded.dims(), &[1, 2, 6, 3, 3]);

        Ok(())
    }

    #[test]
    fn test_apply_temporal_padding_non_causal() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_temporal_padding(&x, 3, false, PaddingMode::Zeros)?;

        assert_eq!(padded.dims(), &[1, 2, 6, 3, 3]);

        Ok(())
    }

    #[test]
    fn test_spatial_padding_zeros() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_spatial_padding(&x, 3, 3, 1, 1, PaddingMode::Zeros)?;

        assert_eq!(padded.dims(), &[1, 2, 4, 5, 5]);

        Ok(())
    }

    #[test]
    fn test_spatial_padding_replicate() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_spatial_padding(&x, 3, 3, 1, 1, PaddingMode::Replicate)?;

        assert_eq!(padded.dims(), &[1, 2, 4, 5, 5]);

        Ok(())
    }

    #[test]
    fn test_spatial_padding_height_only() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_spatial_padding(&x, 3, 1, 1, 0, PaddingMode::Zeros)?;

        assert_eq!(padded.dims(), &[1, 2, 4, 5, 3]);

        Ok(())
    }

    #[test]
    fn test_spatial_padding_width_only() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_spatial_padding(&x, 1, 3, 0, 1, PaddingMode::Zeros)?;

        assert_eq!(padded.dims(), &[1, 2, 4, 3, 5]);

        Ok(())
    }

    #[test]
    fn test_spatial_padding_no_padding() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_spatial_padding(&x, 1, 1, 0, 0, PaddingMode::Zeros)?;

        assert_eq!(padded.dims(), x.dims());

        Ok(())
    }

    #[test]
    fn test_full_padding_causal() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_full_padding(&x, (3, 3, 3), (0, 1, 1), true, PaddingMode::Zeros)?;

        assert_eq!(padded.dims(), &[1, 2, 6, 5, 5]);

        Ok(())
    }

    #[test]
    fn test_full_padding_non_causal() -> Result<()> {
        let x = create_test_tensor((1, 2, 4, 3, 3))?;
        let padded = apply_full_padding(&x, (3, 3, 3), (1, 1, 1), false, PaddingMode::Zeros)?;

        assert_eq!(padded.dims(), &[1, 2, 6, 5, 5]);

        Ok(())
    }
}
