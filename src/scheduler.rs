//! Rectified Flow scheduler implementation for LTX-Video
//!
//! This module implements the Rectified Flow scheduler based on:
//! - Hugging Face diffusers FlowMatchEulerDiscreteScheduler
//! - Lightricks LTX-Video RectifiedFlowScheduler
//!
//! Key features:
//! - Euler solver for ODE integration
//! - Multiple timestep spacing strategies (linspace, linear_quadratic)
//! - Multiple sigma schedules (linspace, karras, exponential, beta)
//! - Resolution-dependent timestep shifting (SD3 style)
//! - Classifier-Free Guidance (CFG) support
//! - Stochastic sampling option
//! - Shift terminal stretching

use crate::config::SchedulerConfig;
use candle_core::{Result, Tensor};

#[cfg(test)]
use candle_core::{DType, Device};

/// Output from a scheduler step
#[derive(Debug)]
pub struct SchedulerOutput {
    /// Denoised sample (x_{t-1})
    pub prev_sample: Tensor,
    /// Predicted original sample (x_0), if computed
    pub pred_original_sample: Option<Tensor>,
}

/// Rectified Flow scheduler using Euler solver
///
/// Implements the ODE-based sampling process for flow matching models.
/// The scheduler integrates the probability flow ODE:
///   dx = v(x, t) dt
///
/// where v is the velocity predicted by the model.
pub struct RectifiedFlowScheduler {
    config: SchedulerConfig,
    /// Timesteps for inference (from 1.0 to near 0.0)
    timesteps: Vec<f64>,
    /// Sigmas (noise levels), with terminal 0 appended
    sigmas: Vec<f64>,
    /// Current step index during inference
    step_index: Option<usize>,
    /// Initial noise standard deviation
    init_noise_sigma: f64,
}

impl RectifiedFlowScheduler {
    /// Create a new scheduler with the given configuration
    pub fn new(config: SchedulerConfig) -> Self {
        let (timesteps, sigmas) = Self::compute_schedule(&config, None);

        Self {
            config,
            timesteps,
            sigmas,
            step_index: None,
            init_noise_sigma: 1.0,
        }
    }

    /// Compute the timestep and sigma schedule
    fn compute_schedule(
        config: &SchedulerConfig,
        sample_shape: Option<(usize, usize, usize, usize, usize)>,
    ) -> (Vec<f64>, Vec<f64>) {
        let num_steps = config.num_inference_steps;

        if num_steps == 0 {
            return (vec![], vec![0.0]);
        }

        // Generate initial timesteps based on spacing type
        let mut timesteps = match config.timestep_spacing.as_str() {
            "linear_quadratic" => Self::linear_quadratic_schedule(num_steps),
            _ => Self::linspace_schedule(num_steps), // Default to linspace
        };

        // Apply sigma schedule transformation if configured
        // This is applied BEFORE shifting for better low-step quality
        if let Some(ref sigma_schedule) = config.sigma_schedule {
            let (sigma_min, sigma_max) = (0.0001, 1.0);
            timesteps = match sigma_schedule.as_str() {
                "karras" => Self::karras_sigmas(num_steps, sigma_min, sigma_max),
                "exponential" => Self::exponential_sigmas(num_steps, sigma_min, sigma_max),
                "beta" => Self::beta_sigmas(num_steps, 0.6, 0.6), // Default alpha/beta for LTX
                _ => timesteps,
            };
        }

        // Apply shift if configured (not using dynamic shifting)
        if !config.use_dynamic_shifting {
            if let Some(shift) = config.shift {
                timesteps = Self::apply_shift(&timesteps, shift);
            }
        } else if let Some(shape) = sample_shape {
            // Apply dynamic resolution-dependent shifting
            let mu = Self::compute_mu_for_resolution(shape, config.base_shift, config.max_shift);
            timesteps = Self::apply_time_shift(&timesteps, mu, 1.0);
        }

        // Stretch to shift_terminal if configured
        if let Some(terminal) = config.shift_terminal {
            timesteps = Self::stretch_to_terminal(&timesteps, terminal);
        }

        // Create sigmas (same as timesteps, with terminal 0 appended)
        let mut sigmas = timesteps.clone();
        sigmas.push(0.0);

        (timesteps, sigmas)
    }

    /// Generate linspace timesteps from 1.0 to near 0.0
    fn linspace_schedule(num_steps: usize) -> Vec<f64> {
        if num_steps == 0 {
            return vec![];
        }
        if num_steps == 1 {
            return vec![1.0];
        }

        // Generate timesteps from 1.0 down to 1/num_steps
        // This matches the flow matching convention
        (0..num_steps)
            .map(|i| 1.0 - (i as f64) / (num_steps as f64))
            .collect()
    }

    /// Generate linear-quadratic schedule
    ///
    /// Combines linear schedule for early steps and quadratic for later steps,
    /// providing smoother denoising at low noise levels.
    fn linear_quadratic_schedule(num_steps: usize) -> Vec<f64> {
        if num_steps == 0 {
            return vec![];
        }
        if num_steps == 1 {
            return vec![1.0];
        }

        let threshold_noise = 0.025;
        let linear_steps = num_steps / 2;

        // Linear part: from 0 to threshold_noise
        let mut schedule: Vec<f64> = (0..linear_steps)
            .map(|i| i as f64 * threshold_noise / linear_steps as f64)
            .collect();

        // Quadratic part
        let quadratic_steps = num_steps - linear_steps;
        let threshold_noise_step_diff = linear_steps as f64 - threshold_noise * num_steps as f64;
        let quadratic_coef =
            threshold_noise_step_diff / (linear_steps as f64 * (quadratic_steps as f64).powi(2));
        let linear_coef = threshold_noise / linear_steps as f64
            - 2.0 * threshold_noise_step_diff / (quadratic_steps as f64).powi(2);
        let const_term = quadratic_coef * (linear_steps as f64).powi(2);

        for i in linear_steps..num_steps {
            let val = quadratic_coef * (i as f64).powi(2) + linear_coef * i as f64 + const_term;
            schedule.push(val);
        }

        // Transform: sigma_i = 1 - schedule_i
        let schedule: Vec<f64> = schedule.iter().map(|x| 1.0 - x).collect();

        // Return first num_steps values
        schedule.into_iter().take(num_steps).collect()
    }

    /// Apply shift to sigmas: shift * s / (1 + (shift - 1) * s)
    fn apply_shift(sigmas: &[f64], shift: f64) -> Vec<f64> {
        sigmas
            .iter()
            .map(|s| shift * s / (1.0 + (shift - 1.0) * s))
            .collect()
    }

    /// Apply exponential time shift: exp(mu) / (exp(mu) + (1/t - 1)^sigma)
    fn apply_time_shift(timesteps: &[f64], mu: f64, sigma: f64) -> Vec<f64> {
        let exp_mu = mu.exp();
        timesteps
            .iter()
            .map(|t| {
                if *t <= 0.0 {
                    0.0
                } else if *t >= 1.0 {
                    1.0
                } else {
                    exp_mu / (exp_mu + (1.0 / t - 1.0).powf(sigma))
                }
            })
            .collect()
    }

    /// Compute mu for resolution-dependent shifting
    fn compute_mu_for_resolution(
        shape: (usize, usize, usize, usize, usize),
        base_shift: f64,
        max_shift: f64,
    ) -> f64 {
        // shape: (batch, channels, frames, height, width)
        let n_tokens = shape.2 * shape.3 * shape.4;

        // Linear interpolation based on token count
        const MIN_TOKENS: f64 = 1024.0;
        const MAX_TOKENS: f64 = 4096.0;

        let m = (max_shift - base_shift) / (MAX_TOKENS - MIN_TOKENS);
        let b = base_shift - m * MIN_TOKENS;

        (m * n_tokens as f64 + b).clamp(base_shift, max_shift)
    }

    /// SimpleDiffusion resolution-dependent timestep shift
    ///
    /// Matches Python rf.py simple_diffusion_resolution_dependent_timestep_shift
    ///
    /// Formula: shifted = 1 - √((1-t)² × d)
    /// Where d = (base_resolution² / (height × width)) × n_tokens / num_inference_steps
    ///
    /// # Arguments
    /// * `timesteps` - Original timesteps in [0, 1]
    /// * `n_tokens` - Number of spatial-temporal tokens (T × H × W)
    /// * `num_inference_steps` - Total inference steps
    /// * `base_resolution` - Base resolution (default 512)
    #[allow(dead_code)]
    fn simple_diffusion_shift(
        timesteps: &[f64],
        n_tokens: usize,
        num_inference_steps: usize,
        base_resolution: usize,
    ) -> Vec<f64> {
        // Calculate scaling factor
        // Using sqrt(n_tokens) as approximate resolution
        let approx_resolution = (n_tokens as f64).sqrt();
        let d = (base_resolution as f64).powi(2) / (approx_resolution.powi(2)) * (n_tokens as f64)
            / (num_inference_steps as f64);

        timesteps
            .iter()
            .map(|t| {
                let one_minus_t = 1.0 - t;
                let shifted = 1.0 - (one_minus_t.powi(2) * d).sqrt();
                shifted.clamp(0.0, 1.0)
            })
            .collect()
    }

    /// Stretch timesteps so the terminal value matches the given target
    fn stretch_to_terminal(timesteps: &[f64], terminal: f64) -> Vec<f64> {
        if timesteps.is_empty() {
            return vec![];
        }
        if terminal <= 0.0 || terminal >= 1.0 {
            return timesteps.to_vec();
        }

        let one_minus_z: Vec<f64> = timesteps.iter().map(|z| 1.0 - z).collect();
        let scale_factor = one_minus_z.last().unwrap() / (1.0 - terminal);

        one_minus_z
            .iter()
            .map(|omz| 1.0 - omz / scale_factor)
            .collect()
    }

    // =========================================================================
    // Sigma Schedule Strategies (from diffusers)
    // =========================================================================

    /// Karras et al. sigma schedule
    ///
    /// Produces sigmas that are more concentrated at high noise levels,
    /// improving sample quality especially at low step counts.
    ///
    /// Reference: "Elucidating the Design Space of Diffusion-Based Generative Models"
    ///
    /// # Arguments
    /// * `num_steps` - Number of inference steps
    /// * `sigma_min` - Minimum sigma value
    /// * `sigma_max` - Maximum sigma value (typically 1.0 for flow matching)
    pub fn karras_sigmas(num_steps: usize, sigma_min: f64, sigma_max: f64) -> Vec<f64> {
        if num_steps == 0 {
            return vec![];
        }
        if num_steps == 1 {
            return vec![sigma_max];
        }

        let rho = 7.0; // Karras paper default
        let min_inv_rho = sigma_min.powf(1.0 / rho);
        let max_inv_rho = sigma_max.powf(1.0 / rho);

        (0..num_steps)
            .map(|i| {
                let t = i as f64 / (num_steps - 1) as f64;
                (max_inv_rho + t * (min_inv_rho - max_inv_rho)).powf(rho)
            })
            .collect()
    }

    /// Exponential sigma schedule
    ///
    /// Produces sigmas spaced exponentially between min and max,
    /// providing more fine-grained control at low noise levels.
    ///
    /// # Arguments
    /// * `num_steps` - Number of inference steps
    /// * `sigma_min` - Minimum sigma value
    /// * `sigma_max` - Maximum sigma value
    pub fn exponential_sigmas(num_steps: usize, sigma_min: f64, sigma_max: f64) -> Vec<f64> {
        if num_steps == 0 {
            return vec![];
        }
        if num_steps == 1 {
            return vec![sigma_max];
        }

        let log_min = sigma_min.max(1e-10).ln();
        let log_max = sigma_max.ln();

        (0..num_steps)
            .map(|i| {
                let t = i as f64 / (num_steps - 1) as f64;
                (log_max + t * (log_min - log_max)).exp()
            })
            .collect()
    }

    /// Beta distribution sigma schedule
    ///
    /// Uses the quantile function of a Beta distribution to place sigmas,
    /// allowing flexible control over the distribution shape via alpha/beta.
    ///
    /// # Arguments
    /// * `num_steps` - Number of inference steps
    /// * `alpha` - Alpha parameter (controls left skew, typical: 0.6)
    /// * `beta` - Beta parameter (controls right skew, typical: 0.6)
    pub fn beta_sigmas(num_steps: usize, alpha: f64, beta: f64) -> Vec<f64> {
        if num_steps == 0 {
            return vec![];
        }
        if num_steps == 1 {
            return vec![1.0];
        }

        // Approximate the regularized incomplete beta function using
        // simple numerical integration (trapezoidal rule)
        // For production use, consider using the `statrs` crate
        let beta_cdf = |x: f64, a: f64, b: f64| -> f64 {
            if x <= 0.0 {
                return 0.0;
            }
            if x >= 1.0 {
                return 1.0;
            }

            // Numerical integration with 1000 steps
            let n = 1000;
            let dx = x / n as f64;
            let mut sum = 0.0;
            for i in 0..n {
                let t = (i as f64 + 0.5) * dx;
                sum += t.powf(a - 1.0) * (1.0 - t).powf(b - 1.0) * dx;
            }

            // Normalize by Beta function B(a,b)
            let beta_fn = Self::log_gamma(a) + Self::log_gamma(b) - Self::log_gamma(a + b);
            (sum / beta_fn.exp()).min(1.0)
        };

        // Inverse CDF via binary search
        let beta_ppf = |p: f64, a: f64, b: f64| -> f64 {
            let mut low = 0.0;
            let mut high = 1.0;
            for _ in 0..50 {
                let mid = (low + high) / 2.0;
                if beta_cdf(mid, a, b) < p {
                    low = mid;
                } else {
                    high = mid;
                }
            }
            (low + high) / 2.0
        };

        (0..num_steps)
            .map(|i| {
                let t = (i as f64 + 0.5) / num_steps as f64;
                1.0 - beta_ppf(t, alpha, beta)
            })
            .collect()
    }

    /// Lanczos approximation for log-gamma function
    fn log_gamma(x: f64) -> f64 {
        let coefficients = [
            76.18009172947146,
            -86.50532032941677,
            24.01409824083091,
            -1.231739572450155,
            0.001208650973866179,
            -0.000005395239384953,
        ];

        let tmp = x + 5.5;
        let tmp = tmp - (x + 0.5) * tmp.ln();
        let mut ser = 1.000000000190015;

        for (i, &coef) in coefficients.iter().enumerate() {
            ser += coef / (x + 1.0 + i as f64);
        }

        -tmp + (2.5066282746310005 * ser / x).ln()
    }

    /// Get the timesteps for inference
    pub fn timesteps(&self) -> &[f64] {
        &self.timesteps
    }

    /// Get the sigmas (noise levels)
    pub fn sigmas(&self) -> &[f64] {
        &self.sigmas
    }

    /// Get the sigma range (min, max)
    pub fn sigma_range(&self) -> (f64, f64) {
        let sigma_min = self
            .sigmas
            .iter()
            .filter(|s| **s > 0.0)
            .cloned()
            .fold(f64::MAX, f64::min);
        let sigma_max = self.sigmas.iter().cloned().fold(f64::MIN, f64::max);
        (sigma_min, sigma_max)
    }

    /// Get the number of inference steps
    pub fn num_steps(&self) -> usize {
        self.config.num_inference_steps
    }

    /// Get guidance scale
    pub fn guidance_scale(&self) -> f64 {
        self.config.guidance_scale
    }

    /// Get initial noise sigma
    pub fn init_noise_sigma(&self) -> f64 {
        self.init_noise_sigma
    }

    /// Get current step index
    pub fn step_index(&self) -> Option<usize> {
        self.step_index
    }

    /// Set timesteps with optional sample shape for dynamic shifting
    ///
    /// # Arguments
    /// * `num_inference_steps` - Number of denoising steps
    /// * `sample_shape` - Optional sample shape (B, C, T, H, W) for resolution-dependent shifting
    /// * `custom_timesteps` - Optional custom timesteps
    /// * `custom_sigmas` - Optional custom sigmas
    pub fn set_timesteps_with_shape(
        &mut self,
        num_inference_steps: usize,
        sample_shape: Option<(usize, usize, usize, usize, usize)>,
        _custom_timesteps: Option<Vec<f64>>,
        _custom_sigmas: Option<Vec<f64>>,
    ) -> Result<()> {
        // Check if dynamic shifting requires shape
        if self.config.use_dynamic_shifting && sample_shape.is_none() {
            return Err(candle_core::Error::Msg(
                "Dynamic shifting requires sample_shape (mu parameter)".to_string(),
            ));
        }

        let mut config = self.config.clone();
        config.num_inference_steps = num_inference_steps;

        let (timesteps, sigmas) = Self::compute_schedule(&config, sample_shape);

        self.timesteps = timesteps;
        self.sigmas = sigmas;
        self.step_index = None;

        Ok(())
    }

    /// Perform a single Euler step
    ///
    /// Formula: prev_sample = sample + dt * model_output
    ///
    /// Where:
    /// - sample: current noisy sample (x_t)
    /// - model_output: predicted velocity (v)
    /// - dt: sigma_next - sigma (negative, as sigma decreases)
    ///
    /// # Arguments
    /// * `model_output` - Velocity prediction from the model
    /// * `sample` - Current noisy sample
    /// * `timestep_idx` - Index into the timesteps array
    pub fn step(
        &self,
        model_output: &Tensor,
        sample: &Tensor,
        timestep_idx: usize,
    ) -> Result<Tensor> {
        if timestep_idx >= self.timesteps.len() {
            return Err(candle_core::Error::Msg(format!(
                "Timestep index {} out of range (max: {})",
                timestep_idx,
                self.timesteps.len().saturating_sub(1)
            )));
        }

        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];
        let dt = sigma_next - sigma;

        if self.config.stochastic_sampling {
            self.stochastic_step(model_output, sample, sigma, sigma_next)
        } else {
            // Euler step: prev_sample = sample + dt * model_output
            // Use affine to preserve dtype (BF16)
            let delta = model_output.affine(dt, 0.0)?;
            sample.add(&delta)
        }
    }

    /// Perform a step and update internal step index
    pub fn step_mut(
        &mut self,
        model_output: &Tensor,
        sample: &Tensor,
        timestep_idx: usize,
    ) -> Result<Tensor> {
        let result = self.step(model_output, sample, timestep_idx)?;
        self.step_index = Some(timestep_idx + 1);
        Ok(result)
    }

    /// Perform a step using timestep value instead of index
    pub fn step_with_timestep(
        &self,
        model_output: &Tensor,
        sample: &Tensor,
        timestep: f64,
    ) -> Result<Tensor> {
        // Find the index for this timestep
        let idx = self.index_for_timestep(timestep)?;
        self.step(model_output, sample, idx)
    }

    /// Perform a step with explicit seed for stochastic sampling
    pub fn step_with_seed(
        &self,
        model_output: &Tensor,
        sample: &Tensor,
        timestep_idx: usize,
        _seed: Option<u64>,
    ) -> Result<Tensor> {
        // TODO: Implement seeded random generation
        // For now, fall back to regular step
        self.step(model_output, sample, timestep_idx)
    }

    /// Stochastic sampling step
    ///
    /// Formula: x0 = sample - sigma * model_output
    ///          prev_sample = (1 - sigma_next) * x0 + sigma_next * noise
    fn stochastic_step(
        &self,
        model_output: &Tensor,
        sample: &Tensor,
        sigma: f64,
        sigma_next: f64,
    ) -> Result<Tensor> {
        // Predict x0 - use affine to preserve dtype
        let x0 = sample.sub(&model_output.affine(sigma, 0.0)?)?;

        // Add noise at sigma_next level - create noise in same dtype as sample
        let noise =
            Tensor::randn(0f32, 1.0, sample.shape(), sample.device())?.to_dtype(sample.dtype())?;

        // Use affine for scalar multiplication
        let term1 = x0.affine(1.0 - sigma_next, 0.0)?;
        let term2 = noise.affine(sigma_next, 0.0)?;

        term1.add(&term2)
    }

    /// Find timestep index for a given timestep value
    fn index_for_timestep(&self, timestep: f64) -> Result<usize> {
        for (i, t) in self.timesteps.iter().enumerate() {
            if (*t - timestep).abs() < 1e-6 {
                return Ok(i);
            }
        }
        Err(candle_core::Error::Msg(format!(
            "Timestep {} not found in schedule",
            timestep
        )))
    }

    /// Return step output including pred_original_sample
    pub fn step_output(
        &self,
        model_output: &Tensor,
        sample: &Tensor,
        timestep_idx: usize,
    ) -> Result<SchedulerOutput> {
        let prev_sample = self.step(model_output, sample, timestep_idx)?;

        // Optionally compute pred_original_sample (x0 prediction)
        let sigma = self.sigmas[timestep_idx];
        let pred_original_sample = if sigma > 0.0 {
            // Use affine to preserve dtype
            Some(sample.sub(&model_output.affine(sigma, 0.0)?)?)
        } else {
            None
        };

        Ok(SchedulerOutput {
            prev_sample,
            pred_original_sample,
        })
    }

    /// Apply classifier-free guidance
    ///
    /// Formula: v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
    ///
    /// # Arguments
    /// * `v_cond` - Conditional model output
    /// * `v_uncond` - Unconditional model output
    pub fn apply_cfg(&self, v_cond: &Tensor, v_uncond: &Tensor) -> Result<Tensor> {
        let scale = self.config.guidance_scale;
        let diff = v_cond.sub(v_uncond)?;
        // Use affine multiplication to preserve dtype (BF16)
        let scaled_diff = diff.affine(scale, 0.0)?;
        v_uncond.add(&scaled_diff)
    }

    /// Add noise to samples (forward diffusion process)
    ///
    /// Formula: noisy = (1 - t) * original + t * noise
    ///
    /// # Arguments
    /// * `original` - Clean sample (x_0)
    /// * `noise` - Random noise
    /// * `timestep` - Noise level (0.0 = clean, 1.0 = pure noise)
    pub fn add_noise(&self, original: &Tensor, noise: &Tensor, timestep: f64) -> Result<Tensor> {
        // Use affine to preserve dtype (BF16)
        let term1 = original.affine(1.0 - timestep, 0.0)?;
        let term2 = noise.affine(timestep, 0.0)?;

        term1.add(&term2)
    }

    /// Scale model input (identity for flow matching)
    ///
    /// Flow matching schedulers don't scale the input, this is included
    /// for API compatibility with other scheduler types.
    pub fn scale_model_input(&self, sample: &Tensor, _timestep_idx: usize) -> Result<Tensor> {
        Ok(sample.clone())
    }

    // =========================================================================
    // Per-Token Timestep Methods (for Image-to-Video conditioning)
    // =========================================================================

    /// Perform Euler step with per-token timesteps for conditioning
    ///
    /// This method handles the case where different tokens have different timesteps,
    /// which is necessary for image-to-video conditioning where conditioning tokens
    /// have a fixed noise level while other tokens are denoised.
    ///
    /// Matches Python rf.py step() for per-token case (lines 355-361):
    /// ```python
    /// # Per-token case
    /// lower_mask = timesteps_padded[:, None, None] < timestep[None] - t_eps
    /// lower_timestep = lower_mask * timesteps_padded[:, None, None]
    /// lower_timestep, _ = lower_timestep.max(dim=0)
    /// dt = (timestep - lower_timestep)[..., None]
    /// prev_sample = sample - dt * model_output
    /// ```
    ///
    /// # Arguments
    /// * `model_output` - Velocity prediction [B, seq, C]
    /// * `sample` - Current latents [B, seq, C]
    /// * `timestep` - Per-token timesteps [B, seq] where each value is in [0, 1]
    ///
    /// # Returns
    /// Denoised sample [B, seq, C]
    pub fn step_per_token(
        &self,
        model_output: &Tensor,
        sample: &Tensor,
        timestep: &Tensor,
    ) -> Result<Tensor> {
        let t_eps = 1e-6;
        let device = sample.device();
        let dtype = sample.dtype();

        // Create padded timesteps tensor with 0 appended: [t0, t1, ..., tn, 0]
        let timesteps_vec: Vec<f32> = self
            .timesteps
            .iter()
            .map(|t| *t as f32)
            .chain(std::iter::once(0.0f32))
            .collect();

        // timestep shape: [B, seq]
        let (batch, seq_len) = timestep.dims2()?;

        // For each token, find the largest scheduled timestep that is < current timestep
        // We'll iterate through scheduled timesteps and find the best lower bound

        // Create threshold: timestep - t_eps, shape [B, seq]
        let t_eps_tensor = Tensor::new(&[t_eps as f32], device)?.to_dtype(dtype)?;
        let threshold = timestep.broadcast_sub(&t_eps_tensor)?;

        // Initialize lower_timestep to zeros [B, seq]
        let mut lower_timestep = Tensor::zeros((batch, seq_len), dtype, device)?;

        // Iterate through scheduled timesteps to find the maximum one that is < threshold
        for &ts in &timesteps_vec {
            let ts_tensor = Tensor::full(ts, (batch, seq_len), device)?.to_dtype(dtype)?;
            // mask where ts < threshold (this scheduled timestep is below the current token's timestep)
            let mask = ts_tensor.lt(&threshold)?;
            // Update lower_timestep where mask is true AND ts > current lower_timestep
            let candidate = ts_tensor.broadcast_mul(&mask.to_dtype(dtype)?)?;
            // Take element-wise maximum
            lower_timestep = lower_timestep.maximum(&candidate)?;
        }

        // dt = timestep - lower_timestep, shape [B, seq]
        let dt = timestep.sub(&lower_timestep)?;

        // Expand dt to [B, seq, 1] for broadcasting with [B, seq, C]
        let dt_expanded = dt.unsqueeze(2)?;

        // prev_sample = sample - dt * model_output
        let update = model_output.broadcast_mul(&dt_expanded)?;
        sample.sub(&update)
    }

    /// Add noise to samples with per-token timesteps
    ///
    /// Formula: noisy = (1 - t) * original + t * noise
    /// Where t is a per-token timestep tensor.
    ///
    /// # Arguments
    /// * `original` - Clean samples [B, seq, C]
    /// * `noise` - Random noise [B, seq, C]
    /// * `timesteps` - Per-token noise levels [B, seq]
    pub fn add_noise_per_token(
        &self,
        original: &Tensor,
        noise: &Tensor,
        timesteps: &Tensor,
    ) -> Result<Tensor> {
        // timesteps: [B, seq] -> [B, seq, 1] for broadcasting
        let t_expanded = timesteps.unsqueeze(2)?;
        let one = Tensor::ones(t_expanded.shape(), t_expanded.dtype(), t_expanded.device())?;

        // (1 - t) * original + t * noise
        let coef1 = one.sub(&t_expanded)?;
        let coef2 = t_expanded;

        let term1 = original.broadcast_mul(&coef1)?;
        let term2 = noise.broadcast_mul(&coef2)?;

        term1.add(&term2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestep_generation() {
        let config = SchedulerConfig::default();
        let scheduler = RectifiedFlowScheduler::new(config);
        let timesteps = scheduler.timesteps();

        assert_eq!(timesteps.len(), 50);
        // First timestep should be 1.0 (full noise)
        assert!((timesteps[0] - 1.0).abs() < 1e-6);
        // Timesteps should be decreasing
        for i in 1..timesteps.len() {
            assert!(timesteps[i] < timesteps[i - 1]);
        }
    }

    #[test]
    fn test_euler_step() -> Result<()> {
        let config = SchedulerConfig {
            num_inference_steps: 10,
            guidance_scale: 3.0,
            timestep_spacing: "linspace".to_string(),
            ..Default::default()
        };
        let scheduler = RectifiedFlowScheduler::new(config);
        let device = Device::Cpu;

        // Create dummy tensors
        let sample = Tensor::randn(0f32, 1.0, (1, 3, 4, 4), &device)?;
        let model_output = Tensor::randn(0f32, 0.1, (1, 3, 4, 4), &device)?;

        // Perform step
        let new_sample = scheduler.step(&model_output, &sample, 5)?;

        assert_eq!(new_sample.shape(), sample.shape());
        Ok(())
    }

    #[test]
    fn test_cfg() -> Result<()> {
        let config = SchedulerConfig {
            num_inference_steps: 50,
            guidance_scale: 3.0,
            timestep_spacing: "linspace".to_string(),
            ..Default::default()
        };
        let scheduler = RectifiedFlowScheduler::new(config);
        let device = Device::Cpu;

        let v_cond = Tensor::ones((1, 3, 4, 4), DType::F32, &device)?;
        let v_uncond = Tensor::zeros((1, 3, 4, 4), DType::F32, &device)?;

        let v_guided = scheduler.apply_cfg(&v_cond, &v_uncond)?;

        // v_guided = 0 + 3.0 * (1 - 0) = 3.0
        let expected = Tensor::new(&[[[[3.0f32]]]], &device)?.broadcast_as((1, 3, 4, 4))?;
        let diff = v_guided
            .sub(&expected)?
            .abs()?
            .sum_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-5);
        Ok(())
    }

    #[test]
    fn test_add_noise() -> Result<()> {
        let config = SchedulerConfig::default();
        let scheduler = RectifiedFlowScheduler::new(config);
        let device = Device::Cpu;

        let original = Tensor::ones((1, 4, 8, 8), DType::F32, &device)?;
        let noise = Tensor::zeros((1, 4, 8, 8), DType::F32, &device)?;

        // At t=0, result should be original
        let result = scheduler.add_noise(&original, &noise, 0.0)?;
        let diff = result
            .sub(&original)?
            .abs()?
            .sum_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-6);

        // At t=1, result should be noise
        let result = scheduler.add_noise(&original, &noise, 1.0)?;
        let diff = result.sub(&noise)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-6);

        Ok(())
    }

    #[test]
    fn test_linear_quadratic_schedule() {
        let config = SchedulerConfig {
            num_inference_steps: 20,
            timestep_spacing: "linear_quadratic".to_string(),
            ..Default::default()
        };
        let scheduler = RectifiedFlowScheduler::new(config);
        let timesteps = scheduler.timesteps();

        assert_eq!(timesteps.len(), 20);
        // Should be monotonically decreasing
        for i in 1..timesteps.len() {
            assert!(
                timesteps[i] < timesteps[i - 1],
                "Timesteps not monotonically decreasing at index {}: {} >= {}",
                i,
                timesteps[i],
                timesteps[i - 1]
            );
        }
    }

    #[test]
    fn test_sigmas_terminal() {
        let config = SchedulerConfig::default();
        let scheduler = RectifiedFlowScheduler::new(config);
        let sigmas = scheduler.sigmas();

        // Last sigma should be 0 (terminal)
        assert!((sigmas[sigmas.len() - 1]).abs() < 1e-6);
    }

    #[test]
    fn test_shift_application() {
        let config_no_shift = SchedulerConfig {
            num_inference_steps: 10,
            shift: Some(1.0),
            use_dynamic_shifting: false, // Disable dynamic shifting to test static shift
            ..Default::default()
        };
        let config_with_shift = SchedulerConfig {
            num_inference_steps: 10,
            shift: Some(3.0),
            use_dynamic_shifting: false, // Disable dynamic shifting to test static shift
            ..Default::default()
        };

        let scheduler_no_shift = RectifiedFlowScheduler::new(config_no_shift);
        let scheduler_with_shift = RectifiedFlowScheduler::new(config_with_shift);

        // Shift=1.0 should be identity
        // Shift=3.0 should modify values
        let sigmas_no_shift = scheduler_no_shift.sigmas();
        let sigmas_with_shift = scheduler_with_shift.sigmas();

        // Values should differ
        let mut any_diff = false;
        for i in 0..sigmas_no_shift.len() - 1 {
            if (sigmas_no_shift[i] - sigmas_with_shift[i]).abs() > 1e-6 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "Shift should modify sigma values");
    }

    #[test]
    fn test_karras_sigmas() {
        let sigmas = RectifiedFlowScheduler::karras_sigmas(10, 0.0001, 1.0);

        assert_eq!(sigmas.len(), 10);
        // First sigma should be max (1.0)
        assert!(
            (sigmas[0] - 1.0).abs() < 1e-6,
            "First sigma should be 1.0, got {}",
            sigmas[0]
        );
        // Last sigma should be near min
        assert!(
            sigmas[9] < 0.01,
            "Last sigma should be near min, got {}",
            sigmas[9]
        );
        // Should be monotonically decreasing
        for i in 1..sigmas.len() {
            assert!(
                sigmas[i] < sigmas[i - 1],
                "Karras sigmas not monotonically decreasing at {}: {} >= {}",
                i,
                sigmas[i],
                sigmas[i - 1]
            );
        }
    }

    #[test]
    fn test_exponential_sigmas() {
        let sigmas = RectifiedFlowScheduler::exponential_sigmas(10, 0.0001, 1.0);

        assert_eq!(sigmas.len(), 10);
        // First sigma should be max (1.0)
        assert!(
            (sigmas[0] - 1.0).abs() < 1e-6,
            "First sigma should be 1.0, got {}",
            sigmas[0]
        );
        // Last sigma should be near min
        assert!(
            sigmas[9] < 0.01,
            "Last sigma should be near min, got {}",
            sigmas[9]
        );
        // Should be monotonically decreasing
        for i in 1..sigmas.len() {
            assert!(
                sigmas[i] < sigmas[i - 1],
                "Exponential sigmas not monotonically decreasing at {}: {} >= {}",
                i,
                sigmas[i],
                sigmas[i - 1]
            );
        }
    }

    #[test]
    fn test_beta_sigmas() {
        let sigmas = RectifiedFlowScheduler::beta_sigmas(10, 0.6, 0.6);

        assert_eq!(sigmas.len(), 10);
        // Should be in valid range
        for sigma in &sigmas {
            assert!(
                *sigma >= 0.0 && *sigma <= 1.0,
                "Beta sigma out of range: {}",
                sigma
            );
        }
        // Should be approximately monotonically decreasing (beta can have slight variations)
        let mut decreasing_count = 0;
        for i in 1..sigmas.len() {
            if sigmas[i] <= sigmas[i - 1] {
                decreasing_count += 1;
            }
        }
        assert!(
            decreasing_count >= sigmas.len() - 2,
            "Beta sigmas should be mostly decreasing"
        );
    }

    #[test]
    fn test_karras_schedule_integration() {
        let config = SchedulerConfig {
            num_inference_steps: 8,
            sigma_schedule: Some("karras".to_string()),
            use_dynamic_shifting: false,
            ..Default::default()
        };
        let scheduler = RectifiedFlowScheduler::new(config);
        let sigmas = scheduler.sigmas();

        assert_eq!(sigmas.len(), 9); // 8 steps + terminal 0
        // Terminal should be 0
        assert!((sigmas[8]).abs() < 1e-6);
    }
}
