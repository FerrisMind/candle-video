//! Probability distributions for VAE latent spaces.

use candle_core::{Result, Tensor};

// Re-export from candle-transformers for compatibility
pub type DiagonalGaussian =
    candle_transformers::models::stable_diffusion::vae::DiagonalGaussianDistribution;

/// Diagonal Gaussian Distribution for VAE latent spaces.
///
/// Used in LTX-Video and Wan VAE for sampling from latent distributions.
/// moments tensor has shape (B, 2*C, T, H, W) where first C channels are mean
/// and last C channels are log variance.
#[derive(Clone, Debug)]
pub struct DiagonalGaussianDistribution {
    pub mean: Tensor,
    pub logvar: Tensor,
}

impl DiagonalGaussianDistribution {
    /// Create from moments tensor (B, 2*C, T, H, W)
    pub fn new(moments: &Tensor) -> Result<Self> {
        let dims = moments.dims();
        if dims.len() != 5 {
            candle_core::bail!(
                "DiagonalGaussianDistribution expects 5D tensor (B, 2*C, T, H, W), got {:?}",
                dims
            );
        }
        let ch2 = dims[1];
        if ch2 % 2 != 0 {
            candle_core::bail!("moments channels must be even, got {}", ch2);
        }
        let ch = ch2 / 2;
        let mean = moments.narrow(1, 0, ch)?;
        let logvar = moments.narrow(1, ch, ch)?;
        Ok(Self { mean, logvar })
    }

    /// Create from moments tensor (B, 2*C, H, W) — 4D version for images
    pub fn new_4d(moments: &Tensor) -> Result<Self> {
        let dims = moments.dims();
        if dims.len() != 4 {
            candle_core::bail!(
                "DiagonalGaussianDistribution::new_4d expects 4D tensor (B, 2*C, H, W), got {:?}",
                dims
            );
        }
        let ch2 = dims[1];
        if ch2 % 2 != 0 {
            candle_core::bail!("moments channels must be even, got {}", ch2);
        }
        let ch = ch2 / 2;
        let mean = moments.narrow(1, 0, ch)?;
        let logvar = moments.narrow(1, ch, ch)?;
        Ok(Self { mean, logvar })
    }

    /// Return the mode (mean) of the distribution
    pub fn mode(&self) -> Result<Tensor> {
        Ok(self.mean.clone())
    }

    /// Sample from the distribution: z = mean + exp(0.5 * logvar) * eps
    pub fn sample(&self) -> Result<Tensor> {
        let eps = Tensor::randn(0f32, 1f32, self.mean.shape(), self.mean.device())?
            .to_dtype(self.mean.dtype())?;
        let std = self.logvar.affine(0.5, 0.0)?.exp()?;
        self.mean.add(&std.mul(&eps)?)
    }

    /// Sample with external noise tensor
    pub fn sample_with_noise(&self, noise: &Tensor) -> Result<Tensor> {
        let std = self.logvar.affine(0.5, 0.0)?.exp()?;
        self.mean.add(&std.mul(noise)?)
    }

    /// KL divergence from standard normal N(0, 1)
    pub fn kl(&self) -> Result<Tensor> {
        // KL = 0.5 * sum(mu^2 + exp(logvar) - 1 - logvar)
        let mu_sq = self.mean.sqr()?;
        let exp_logvar = self.logvar.exp()?;
        let one = Tensor::ones_like(&self.logvar)?;
        
        let kl = mu_sq
            .add(&exp_logvar)?
            .sub(&one)?
            .sub(&self.logvar)?
            .affine(0.5, 0.0)?;
        
        // Sum over all dimensions except batch
        let dims: Vec<usize> = (1..kl.rank()).collect();
        kl.sum(dims.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_diagonal_gaussian_5d() {
        let device = Device::Cpu;
        let moments = Tensor::zeros((1, 8, 2, 4, 4), DType::F32, &device).unwrap();
        let dist = DiagonalGaussianDistribution::new(&moments).unwrap();
        assert_eq!(dist.mean.dims(), &[1, 4, 2, 4, 4]);
        assert_eq!(dist.logvar.dims(), &[1, 4, 2, 4, 4]);
    }

    #[test]
    fn test_diagonal_gaussian_4d() {
        let device = Device::Cpu;
        let moments = Tensor::zeros((1, 8, 4, 4), DType::F32, &device).unwrap();
        let dist = DiagonalGaussianDistribution::new_4d(&moments).unwrap();
        assert_eq!(dist.mean.dims(), &[1, 4, 4, 4]);
    }

    #[test]
    fn test_sample() {
        let device = Device::Cpu;
        let moments = Tensor::zeros((1, 4, 2, 2, 2), DType::F32, &device).unwrap();
        let dist = DiagonalGaussianDistribution::new(&moments).unwrap();
        let sample = dist.sample().unwrap();
        assert_eq!(sample.dims(), dist.mean.dims());
    }
}
