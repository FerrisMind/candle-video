//! Euler Discrete Scheduler for SVD
//!
//! Implements the Euler discrete scheduler with v-prediction and karras sigmas
//! for Stable Video Diffusion inference.

use candle_core::{Device, Result, Tensor};

use crate::svd::config::EulerSchedulerConfig;

/// Output from a scheduler step
#[derive(Debug)]
pub struct EulerSchedulerOutput {
    pub prev_sample: Tensor,
    pub pred_original_sample: Option<Tensor>,
}

/// Euler Discrete Scheduler with v-prediction support
///
/// Implements the sampling process for diffusion models trained with
/// v-prediction objective. Supports Karras sigma schedule for improved
/// sample quality.
#[derive(Debug)]
pub struct EulerDiscreteScheduler {
    config: EulerSchedulerConfig,
    timesteps: Vec<f64>,
    sigmas: Vec<f64>,
    init_noise_sigma: f64,
    step_index: Option<usize>,
    num_inference_steps: Option<usize>,
}

impl EulerDiscreteScheduler {
    pub fn new(config: EulerSchedulerConfig) -> Self {
        let (sigmas, init_noise_sigma) = Self::compute_sigmas(&config);

        Self {
            config,
            timesteps: Vec::new(),
            sigmas,
            init_noise_sigma,
            step_index: None,
            num_inference_steps: None,
        }
    }

    fn compute_sigmas(config: &EulerSchedulerConfig) -> (Vec<f64>, f64) {
        let num_train_timesteps = config.num_train_timesteps;

        // Compute betas based on schedule
        let betas: Vec<f64> = match config.beta_schedule.as_str() {
            "scaled_linear" => {
                let start = config.beta_start.sqrt();
                let end = config.beta_end.sqrt();
                (0..num_train_timesteps)
                    .map(|i| {
                        let t = i as f64 / (num_train_timesteps - 1) as f64;
                        let beta_sqrt = start + t * (end - start);
                        beta_sqrt * beta_sqrt
                    })
                    .collect()
            }
            "linear" => (0..num_train_timesteps)
                .map(|i| {
                    let t = i as f64 / (num_train_timesteps - 1) as f64;
                    config.beta_start + t * (config.beta_end - config.beta_start)
                })
                .collect(),
            _ => {
                // Default to linear
                (0..num_train_timesteps)
                    .map(|i| {
                        let t = i as f64 / (num_train_timesteps - 1) as f64;
                        config.beta_start + t * (config.beta_end - config.beta_start)
                    })
                    .collect()
            }
        };

        // Compute alphas and alpha_cumprod
        let alphas: Vec<f64> = betas.iter().map(|b| 1.0 - b).collect();
        let mut alpha_cumprod = Vec::with_capacity(num_train_timesteps);
        let mut cumprod = 1.0;
        for alpha in alphas.iter() {
            cumprod *= alpha;
            alpha_cumprod.push(cumprod);
        }

        // Compute sigmas from alpha_cumprod: sigma = sqrt((1 - alpha_cumprod) / alpha_cumprod)
        let sigmas: Vec<f64> = alpha_cumprod
            .iter()
            .map(|&a| ((1.0 - a) / a).sqrt())
            .collect();

        // init_noise_sigma for "leading" spacing
        let init_noise_sigma = (sigmas[0].powi(2) + 1.0).sqrt();

        (sigmas, init_noise_sigma)
    }

    /// Set timesteps for inference
    pub fn set_timesteps(&mut self, num_inference_steps: usize, _device: &Device) -> Result<()> {
        self.num_inference_steps = Some(num_inference_steps);
        let num_train_timesteps = self.config.num_train_timesteps;

        // Compute timesteps based on spacing type
        let timesteps: Vec<f64> = match self.config.timestep_spacing.as_str() {
            "leading" => {
                // diffusers uses integer division and reverses the result
                let step_ratio = num_train_timesteps / num_inference_steps;
                (0..num_inference_steps)
                    .map(|i| (i * step_ratio) as f64 + self.config.steps_offset as f64)
                    .rev()
                    .collect()
            }
            "trailing" => {
                let step_ratio = num_train_timesteps as f64 / num_inference_steps as f64;
                (1..=num_inference_steps)
                    .rev()
                    .map(|i| (num_train_timesteps as f64 - i as f64 * step_ratio).round())
                    .collect()
            }
            _ => {
                // "linspace" default
                (0..num_inference_steps)
                    .map(|i| {
                        let t = i as f64 / (num_inference_steps - 1).max(1) as f64;
                        ((1.0 - t) * (num_train_timesteps - 1) as f64).round()
                    })
                    .collect()
            }
        };

        // Get sigmas at timestep indices
        let mut step_sigmas: Vec<f64> = timesteps
            .iter()
            .map(|&t| {
                let idx = (t as usize).min(self.sigmas.len() - 1);
                self.sigmas[idx]
            })
            .collect();

        // Apply Karras sigmas if enabled
        if self.config.use_karras_sigmas {
            step_sigmas =
                Self::convert_to_karras(&step_sigmas, self.config.sigma_min, self.config.sigma_max);
        }

        // Add terminal sigma (0)
        step_sigmas.push(0.0);

        // Compute timesteps from sigmas for continuous timestep_type
        // For v_prediction: timesteps = 0.25 * ln(sigma)
        // See: diffusers scheduling_euler_discrete.py:255
        self.timesteps = if self.config.timestep_type == "continuous" {
            // Exclude the last sigma (which is 0.0) from timesteps exposed to pipeline
            step_sigmas[..step_sigmas.len() - 1]
                .iter()
                .map(|&s| 0.25 * s.ln())
                .collect()
        } else {
            timesteps
        };

        self.sigmas = step_sigmas;
        self.step_index = None;

        Ok(())
    }

    /// Convert sigmas to Karras schedule
    fn convert_to_karras(sigmas: &[f64], sigma_min: f64, sigma_max: f64) -> Vec<f64> {
        let n = sigmas.len();
        let rho = 7.0; // Karras et al. recommend rho=7

        let min_inv_rho = sigma_min.powf(1.0 / rho);
        let max_inv_rho = sigma_max.powf(1.0 / rho);

        (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1).max(1) as f64;
                (max_inv_rho + t * (min_inv_rho - max_inv_rho)).powf(rho)
            })
            .collect()
    }

    /// Scale model input by sigma for Euler step
    pub fn scale_model_input(&self, sample: &Tensor, timestep_idx: usize) -> Result<Tensor> {
        let sigma = self.sigmas[timestep_idx];
        let scale = (sigma.powi(2) + 1.0).sqrt();
        sample / scale
    }

    /// Perform one Euler step
    pub fn step(
        &mut self,
        model_output: &Tensor,
        timestep_idx: usize,
        sample: &Tensor,
    ) -> Result<EulerSchedulerOutput> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        // For v-prediction: x0 = sigma * sample - (sigma^2 + 1)^0.5 * model_output
        // Then: noise = (sample - x0) / sigma
        // For epsilon prediction: noise = model_output directly
        let (pred_original_sample, derivative) = match self.config.prediction_type.as_str() {
            "v_prediction" => {
                // x0 = sample * c_out + model_output * c_skip
                // where c_out = -sigma / sqrt(sigma^2 + 1), c_skip = 1 / sqrt(sigma^2 + 1)
                let sigma_sq_plus_1_sqrt = (sigma.powi(2) + 1.0).sqrt();
                let c_skip = 1.0 / sigma_sq_plus_1_sqrt;
                let c_out = -sigma / sigma_sq_plus_1_sqrt;

                let pred_x0 = (sample * c_skip)?.add(&(model_output * c_out)?)?;
                // derivative (dx/dt) = (sample - pred_x0) / sigma
                let deriv = ((sample - &pred_x0)? / sigma)?;
                (Some(pred_x0), deriv)
            }
            "epsilon" => {
                // pred_x0 = sample - sigma * model_output
                let pred_x0 = (sample - &(model_output * sigma)?)?;
                let deriv = model_output.clone();
                (Some(pred_x0), deriv)
            }
            _ => {
                // sample prediction
                let deriv = ((sample - model_output)? / sigma)?;
                (Some(model_output.clone()), deriv)
            }
        };

        // Euler step: x_{t-1} = x_t + (sigma_next - sigma) * derivative
        let dt = sigma_next - sigma;
        let prev_sample = (sample + &(derivative * dt)?)?;

        self.step_index = Some(timestep_idx + 1);

        Ok(EulerSchedulerOutput {
            prev_sample,
            pred_original_sample,
        })
    }

    /// Add noise to sample for initialization
    pub fn add_noise(
        &self,
        original_samples: &Tensor,
        noise: &Tensor,
        timestep_idx: usize,
    ) -> Result<Tensor> {
        let sigma = self.sigmas[timestep_idx];
        let noisy = (original_samples + noise * sigma)?;
        Ok(noisy)
    }

    /// Get initial noise sigma for scaling random noise
    pub fn init_noise_sigma(&self) -> f64 {
        if self.sigmas.is_empty() {
            return self.init_noise_sigma;
        }

        let max_sigma = self.sigmas[0];
        // diffusers: for linspace/trailing, return max_sigma; for leading, return (max_sigma^2+1)^0.5
        match self.config.timestep_spacing.as_str() {
            "linspace" | "trailing" => max_sigma,
            _ => (max_sigma.powi(2) + 1.0).sqrt(),
        }
    }

    /// Get timesteps
    pub fn timesteps(&self) -> &[f64] {
        &self.timesteps
    }

    /// Get sigmas
    pub fn sigmas(&self) -> &[f64] {
        &self.sigmas
    }

    /// Get number of inference steps
    pub fn num_inference_steps(&self) -> Option<usize> {
        self.num_inference_steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_creation() {
        let config = EulerSchedulerConfig::default();
        let scheduler = EulerDiscreteScheduler::new(config);

        assert!(scheduler.init_noise_sigma() > 0.0);
        assert!(!scheduler.sigmas.is_empty());
    }

    #[test]
    fn test_set_timesteps() {
        let config = EulerSchedulerConfig::default();
        let mut scheduler = EulerDiscreteScheduler::new(config);

        scheduler.set_timesteps(25, &Device::Cpu).unwrap();

        assert_eq!(scheduler.num_inference_steps(), Some(25));
        // Sigmas should have num_steps + 1 entries (including terminal 0)
        assert_eq!(scheduler.sigmas().len(), 26);
        // Last sigma should be 0
        assert_eq!(*scheduler.sigmas().last().unwrap(), 0.0);
    }

    #[test]
    fn test_karras_sigmas() {
        let sigmas = vec![14.6, 5.0, 1.0, 0.1];
        let karras = EulerDiscreteScheduler::convert_to_karras(&sigmas, 0.002, 700.0);

        // Should be monotonically decreasing
        for i in 1..karras.len() {
            assert!(karras[i] < karras[i - 1], "Karras sigmas should decrease");
        }
        // First should be close to sigma_max
        assert!(karras[0] <= 700.0);
        // Last should be positive and small (approaching sigma_min)
        assert!(*karras.last().unwrap() > 0.0);
        assert!(*karras.last().unwrap() < 1.0);
    }
}
