use candle_core::{Error, Result, Tensor};

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoLayout {
    /// [batch, frames, channels, height, width]
    BFCHW,
    /// [batch*frames, channels, height, width]
    BfCHW,
}

#[derive(Debug, Clone)]
pub struct VideoLatents {
    pub tensor: Tensor,
    pub layout: VideoLayout,
    pub batch: usize,
    pub frames: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

impl VideoLatents {
    /// Convert to canonical BFCHW layout.
    pub fn to_canonical(&self) -> Result<Self> {
        match self.layout {
            VideoLayout::BFCHW => Ok(self.clone()),
            VideoLayout::BfCHW => {
                let (bf, c, h, w) = self.tensor.dims4()?;
                let expected_bf = self.batch * self.frames;
                if bf != expected_bf || c != self.channels || h != self.height || w != self.width {
                    return Err(Error::Msg(format!(
                        "VideoLatents shape mismatch: expected [{}*{}, {}, {}, {}], got [{}, {}, {}, {}]",
                        self.batch,
                        self.frames,
                        self.channels,
                        self.height,
                        self.width,
                        bf,
                        c,
                        h,
                        w
                    )));
                }
                let tensor = self.tensor.reshape((
                    self.batch,
                    self.frames,
                    self.channels,
                    self.height,
                    self.width,
                ))?;
                Ok(Self {
                    tensor,
                    layout: VideoLayout::BFCHW,
                    batch: self.batch,
                    frames: self.frames,
                    channels: self.channels,
                    height: self.height,
                    width: self.width,
                })
            }
        }
    }

    pub fn from_canonical(
        tensor: Tensor,
        batch: usize,
        frames: usize,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Self {
        Self {
            tensor,
            layout: VideoLayout::BFCHW,
            batch,
            frames,
            channels,
            height,
            width,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_to_canonical_from_flattened() {
        let (batch, frames, channels, height, width) = (2, 3, 4, 5, 6);
        let flat = Tensor::zeros(
            (batch * frames, channels, height, width),
            candle_core::DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let latents = VideoLatents {
            tensor: flat,
            layout: VideoLayout::BfCHW,
            batch,
            frames,
            channels,
            height,
            width,
        };
        let canonical = latents.to_canonical().unwrap();
        assert_eq!(canonical.layout, VideoLayout::BFCHW);
        assert_eq!(
            canonical.tensor.dims(),
            vec![batch, frames, channels, height, width]
        );
    }

    #[test]
    fn test_to_canonical_noop() {
        let (batch, frames, channels, height, width) = (1, 2, 3, 4, 5);
        let tensor = Tensor::zeros(
            (batch, frames, channels, height, width),
            candle_core::DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let latents = VideoLatents::from_canonical(tensor, batch, frames, channels, height, width);
        let canonical = latents.to_canonical().unwrap();
        assert_eq!(canonical.layout, VideoLayout::BFCHW);
        assert_eq!(
            canonical.tensor.dims(),
            vec![batch, frames, channels, height, width]
        );
    }
}
