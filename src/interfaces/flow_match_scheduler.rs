use candle_core::{DType, Device, Result, Tensor, bail};
use serde::{Deserialize, Serialize};
use statrs::distribution::{Beta, ContinuousCDF};

use crate::interfaces::scheduler_mixin::{SchedulerMixin, SchedulerStepOutput};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeShiftType {
    Exponential,
    Linear,
}

#[derive(Debug, Clone)]
pub enum TimestepsSpec {
    Steps(usize),
    Timesteps(Vec<i64>),
    Sigmas(Vec<f32>),
}

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub base_image_seq_len: usize,
    pub max_image_seq_len: usize,
    pub base_shift: f32,
    pub max_shift: f32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            base_image_seq_len: 256,
            max_image_seq_len: 4096,
            base_shift: 0.5,
            max_shift: 1.15,
        }
    }
}

pub trait Scheduler: SchedulerMixin {
    fn config(&self) -> &SchedulerConfig;
    fn order(&self) -> usize;
    fn set_timesteps(&mut self, spec: TimestepsSpec, device: &Device, mu: f32) -> Result<Vec<i64>>;
    fn step(&mut self, noise_pred: &Tensor, timestep: i64, latents: &Tensor) -> Result<Tensor>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowMatchEulerDiscreteSchedulerConfig {
    pub num_train_timesteps: usize,
    pub shift: f32,
    pub use_dynamic_shifting: bool,

    pub base_shift: Option<f32>,
    pub max_shift: Option<f32>,
    pub base_image_seq_len: Option<usize>,
    pub max_image_seq_len: Option<usize>,

    pub invert_sigmas: bool,
    pub shift_terminal: Option<f32>,

    pub use_karras_sigmas: bool,
    pub use_exponential_sigmas: bool,
    pub use_beta_sigmas: bool,

    pub time_shift_type: TimeShiftType,
    pub stochastic_sampling: bool,
}

impl Default for FlowMatchEulerDiscreteSchedulerConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            shift: 1.0,
            use_dynamic_shifting: false,
            base_shift: Some(0.5),
            max_shift: Some(1.15),
            base_image_seq_len: Some(256),
            max_image_seq_len: Some(4096),
            invert_sigmas: false,
            shift_terminal: None,
            use_karras_sigmas: false,
            use_exponential_sigmas: false,
            use_beta_sigmas: false,
            time_shift_type: TimeShiftType::Exponential,
            stochastic_sampling: false,
        }
    }
}

impl FlowMatchEulerDiscreteSchedulerConfig {
    pub fn ltx_video() -> Self {
        Self::default()
    }

    pub fn wan_720p() -> Self {
        Self {
            shift: 5.0,
            use_dynamic_shifting: false,
            ..Default::default()
        }
    }

    pub fn wan_480p() -> Self {
        Self {
            shift: 3.0,
            use_dynamic_shifting: false,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone)]
pub struct FlowMatchEulerDiscreteSchedulerOutput {
    pub prev_sample: Tensor,
}

#[derive(Debug)]
pub struct FlowMatchEulerDiscreteScheduler {
    pub config: FlowMatchEulerDiscreteSchedulerConfig,

    timesteps: Tensor,
    sigmas: Tensor,
    timesteps_cpu: Vec<f32>,
    timesteps_f64: Vec<f64>,
    sigmas_cpu: Vec<f32>,

    sigma_min: f32,
    sigma_max: f32,

    step_index: Option<usize>,
    begin_index: Option<usize>,
    num_inference_steps: Option<usize>,

    scheduler_config: SchedulerConfig,
}

impl FlowMatchEulerDiscreteScheduler {
    pub fn new(config: FlowMatchEulerDiscreteSchedulerConfig) -> Result<Self> {
        if config.use_beta_sigmas as u32
            + config.use_exponential_sigmas as u32
            + config.use_karras_sigmas as u32
            > 1
        {
            bail!(
                "Only one of use_beta_sigmas/use_exponential_sigmas/use_karras_sigmas can be enabled."
            );
        }

        let n = config.num_train_timesteps;
        let mut ts: Vec<f32> = (1..=n).map(|v| v as f32).collect();
        ts.reverse();

        let mut sigmas: Vec<f32> = ts.iter().map(|t| t / n as f32).collect();

        if !config.use_dynamic_shifting {
            sigmas = sigmas
                .into_iter()
                .map(|s| {
                    let shift = config.shift;
                    shift * s / (1.0 + (shift - 1.0) * s)
                })
                .collect();
            ts = sigmas.iter().map(|s| s * n as f32).collect();
        } else {
            ts = sigmas.iter().map(|s| s * n as f32).collect();
        }

        let device = Device::Cpu;
        let timesteps_t = Tensor::from_vec(ts.clone(), (ts.len(),), &device)?;
        let sigmas_t = Tensor::from_vec(sigmas.clone(), (sigmas.len(),), &device)?;

        let sigma_min = *sigmas.last().unwrap_or(&0.0);
        let sigma_max = *sigmas.first().unwrap_or(&1.0);

        let mut sigmas_cpu = sigmas.clone();
        sigmas_cpu.push(0.0);
        let sigmas_with_terminal =
            Tensor::cat(&[sigmas_t, Tensor::zeros((1,), DType::F32, &device)?], 0)?;
        let timesteps_f64 = ts.iter().map(|v| *v as f64).collect();

        let scheduler_config = SchedulerConfig {
            base_image_seq_len: config.base_image_seq_len.unwrap_or(256),
            max_image_seq_len: config.max_image_seq_len.unwrap_or(4096),
            base_shift: config.base_shift.unwrap_or(0.5),
            max_shift: config.max_shift.unwrap_or(1.15),
        };

        Ok(Self {
            config,
            timesteps: timesteps_t,
            sigmas: sigmas_with_terminal,
            timesteps_cpu: ts,
            timesteps_f64,
            sigmas_cpu,
            sigma_min,
            sigma_max,
            step_index: None,
            begin_index: None,
            num_inference_steps: None,
            scheduler_config,
        })
    }

    pub fn shift(&self) -> f32 {
        self.config.shift
    }

    pub fn step_index(&self) -> Option<usize> {
        self.step_index
    }

    pub fn begin_index(&self) -> Option<usize> {
        self.begin_index
    }

    pub fn set_begin_index(&mut self, begin_index: usize) {
        self.begin_index = Some(begin_index);
    }

    pub fn set_shift(&mut self, shift: f32) {
        self.config.shift = shift;
    }

    fn sigma_to_t(&self, sigma: f32) -> f32 {
        sigma * self.config.num_train_timesteps as f32
    }

    fn time_shift_scalar(&self, mu: f32, sigma: f32, t: f32) -> f32 {
        match self.config.time_shift_type {
            TimeShiftType::Exponential => {
                let emu = mu.exp();
                let base = (1.0 / t - 1.0).powf(sigma);
                emu / (emu + base)
            }
            TimeShiftType::Linear => {
                let base = (1.0 / t - 1.0).powf(sigma);
                mu / (mu + base)
            }
        }
    }

    fn stretch_shift_to_terminal_vec(&self, t: &mut [f32]) -> Result<()> {
        let shift_terminal = match self.config.shift_terminal {
            Some(v) => v,
            None => return Ok(()),
        };
        if t.is_empty() {
            return Ok(());
        }
        let one_minus_last = 1.0 - t[t.len() - 1];
        let denom = 1.0 - shift_terminal;
        if denom.abs() < 1e-12 {
            bail!("shift_terminal too close to 1.0, would divide by zero.");
        }
        let scale_factor = one_minus_last / denom;
        for v in t.iter_mut() {
            let one_minus_z = 1.0 - *v;
            *v = 1.0 - (one_minus_z / scale_factor);
        }
        Ok(())
    }

    fn linspace(start: f32, end: f32, steps: usize) -> Vec<f32> {
        if steps == 0 {
            return vec![];
        }
        if steps == 1 {
            return vec![start];
        }
        let denom = (steps - 1) as f32;
        (0..steps)
            .map(|i| start + (end - start) * (i as f32) / denom)
            .collect()
    }

    fn convert_to_karras(&self, in_sigmas: &[f32], num_inference_steps: usize) -> Vec<f32> {
        let sigma_min = in_sigmas.last().copied().unwrap_or(self.sigma_min);
        let sigma_max = in_sigmas.first().copied().unwrap_or(self.sigma_max);

        let rho: f32 = 7.0;
        let ramp = Self::linspace(0.0, 1.0, num_inference_steps);

        let min_inv_rho = sigma_min.powf(1.0 / rho);
        let max_inv_rho = sigma_max.powf(1.0 / rho);

        ramp.into_iter()
            .map(|r| (max_inv_rho + r * (min_inv_rho - max_inv_rho)).powf(rho))
            .collect()
    }

    fn convert_to_exponential(&self, in_sigmas: &[f32], num_inference_steps: usize) -> Vec<f32> {
        let sigma_min = in_sigmas.last().copied().unwrap_or(self.sigma_min);
        let sigma_max = in_sigmas.first().copied().unwrap_or(self.sigma_max);

        let start = sigma_max.ln();
        let end = sigma_min.ln();
        let logs = Self::linspace(start, end, num_inference_steps);
        logs.into_iter().map(|v| v.exp()).collect()
    }

    fn convert_to_beta(
        &self,
        in_sigmas: &[f32],
        num_inference_steps: usize,
        alpha: f64,
        beta: f64,
    ) -> Result<Vec<f32>> {
        let sigma_min = in_sigmas.last().copied().unwrap_or(self.sigma_min);
        let sigma_max = in_sigmas.first().copied().unwrap_or(self.sigma_max);

        let ts = Self::linspace(0.0, 1.0, num_inference_steps)
            .into_iter()
            .map(|v| 1.0 - v as f64)
            .collect::<Vec<_>>();

        let dist = Beta::new(alpha, beta).map_err(|e| candle_core::Error::msg(format!("{e:?}")))?;

        let mut out = Vec::with_capacity(num_inference_steps);
        for t in ts {
            let ppf = dist.inverse_cdf(t);
            let s = sigma_min as f64 + ppf * ((sigma_max - sigma_min) as f64);
            out.push(s as f32);
        }
        Ok(out)
    }

    pub fn set_timesteps_inner(
        &mut self,
        num_inference_steps: Option<usize>,
        device: &Device,
        sigmas: Option<&[f32]>,
        mu: Option<f32>,
        timesteps: Option<&[f32]>,
    ) -> Result<()> {
        if self.config.use_dynamic_shifting && mu.is_none() {
            bail!("mu must be provided when use_dynamic_shifting = true.");
        }

        if sigmas
            .zip(timesteps)
            .is_some_and(|(s, t)| s.len() != t.len())
        {
            bail!("sigmas and timesteps must have the same length.");
        }

        let mut num_inference_steps = num_inference_steps;
        if let Some(n) = num_inference_steps {
            if sigmas.is_some_and(|s| s.len() != n) {
                bail!("sigmas length must match num_inference_steps.");
            }
            if timesteps.is_some_and(|t| t.len() != n) {
                bail!("timesteps length must match num_inference_steps.");
            }
        } else if let Some(s) = sigmas {
            num_inference_steps = Some(s.len());
        } else if let Some(t) = timesteps {
            num_inference_steps = Some(t.len());
        } else {
            bail!(
                "num_inference_steps must be provided if neither sigmas nor timesteps are provided."
            );
        }
        let num_inference_steps = num_inference_steps.unwrap();
        self.num_inference_steps = Some(num_inference_steps);

        let is_timesteps_provided = timesteps.is_some();
        let mut ts_vec: Option<Vec<f32>> = timesteps.map(|t| t.to_vec());

        let mut sigmas_vec: Vec<f32> = if let Some(s) = sigmas {
            s.to_vec()
        } else {
            let timesteps_vec = match ts_vec.take() {
                Some(v) => v,
                None => {
                    let start = self.sigma_to_t(self.sigma_max);
                    let end = self.sigma_to_t(self.sigma_min);
                    Self::linspace(start, end, num_inference_steps)
                }
            };
            let s = timesteps_vec
                .iter()
                .map(|t| *t / self.config.num_train_timesteps as f32)
                .collect::<Vec<_>>();
            ts_vec = Some(timesteps_vec);
            s
        };

        if let Some(mu) = mu {
            sigmas_vec = sigmas_vec
                .into_iter()
                .map(|t| self.time_shift_scalar(mu, 1.0, t))
                .collect();
        } else if self.config.use_dynamic_shifting {
            bail!("mu must be provided when use_dynamic_shifting = true.");
        } else {
            let shift = self.config.shift;
            sigmas_vec = sigmas_vec
                .into_iter()
                .map(|s| shift * s / (1.0 + (shift - 1.0) * s))
                .collect();
        }

        if self.config.shift_terminal.is_some() {
            self.stretch_shift_to_terminal_vec(&mut sigmas_vec)?;
        }

        if self.config.use_karras_sigmas {
            sigmas_vec = self.convert_to_karras(&sigmas_vec, num_inference_steps);
        } else if self.config.use_exponential_sigmas {
            sigmas_vec = self.convert_to_exponential(&sigmas_vec, num_inference_steps);
        } else if self.config.use_beta_sigmas {
            sigmas_vec = self.convert_to_beta(&sigmas_vec, num_inference_steps, 0.6, 0.6)?;
        }

        let mut timesteps_vec: Vec<f32> = if is_timesteps_provided {
            ts_vec.unwrap_or_else(|| {
                sigmas_vec
                    .iter()
                    .map(|s| s * self.config.num_train_timesteps as f32)
                    .collect()
            })
        } else {
            sigmas_vec
                .iter()
                .map(|s| s * self.config.num_train_timesteps as f32)
                .collect()
        };

        if self.config.invert_sigmas {
            for v in sigmas_vec.iter_mut() {
                *v = 1.0 - *v;
            }
            timesteps_vec = sigmas_vec
                .iter()
                .map(|s| s * self.config.num_train_timesteps as f32)
                .collect();
            sigmas_vec.push(1.0);
        } else {
            sigmas_vec.push(0.0);
        }

        self.sigmas_cpu = sigmas_vec.clone();
        self.timesteps_cpu = timesteps_vec.clone();
        self.timesteps_f64 = self.timesteps_cpu.iter().map(|v| *v as f64).collect();

        self.sigmas = Tensor::from_vec(sigmas_vec, (self.sigmas_cpu.len(),), device)?;
        self.timesteps = Tensor::from_vec(timesteps_vec, (self.timesteps_cpu.len(),), device)?;

        self.step_index = None;
        self.begin_index = None;

        Ok(())
    }

    pub fn index_for_timestep(
        &self,
        timestep: f32,
        schedule_timesteps: Option<&[f32]>,
    ) -> Result<usize> {
        let st = schedule_timesteps.unwrap_or(&self.timesteps_cpu);
        let mut indices = Vec::new();
        for (i, &v) in st.iter().enumerate() {
            if (v - timestep).abs() < 1e-6 {
                indices.push(i);
            }
        }
        if indices.is_empty() {
            bail!("timestep not found in schedule_timesteps.");
        }
        let pos = if indices.len() > 1 { 1 } else { 0 };
        Ok(indices[pos])
    }

    fn init_step_index(&mut self, timestep: f32) -> Result<()> {
        if self.begin_index.is_none() {
            self.step_index = Some(self.index_for_timestep(timestep, None)?);
        } else {
            self.step_index = self.begin_index;
        }
        Ok(())
    }

    pub fn scale_noise(
        &self,
        sample: &Tensor,
        timestep: &Tensor,
        noise: Option<&Tensor>,
    ) -> Result<Tensor> {
        let device = sample.device();

        let ts: Vec<f32> = match timestep.rank() {
            0 => vec![timestep.to_scalar::<f32>()?],
            1 => timestep.to_vec1::<f32>()?,
            r => bail!("timestep must be rank 0 or 1, got rank={r}"),
        };

        let mut step_indices = Vec::with_capacity(ts.len());
        if self.begin_index.is_none() {
            for &t in ts.iter() {
                step_indices.push(self.index_for_timestep(t, Some(&self.timesteps_cpu))?);
            }
        } else if let Some(si) = self.step_index {
            step_indices.extend(std::iter::repeat_n(si, ts.len()));
        } else {
            let bi = self.begin_index.unwrap_or(0);
            step_indices.extend(std::iter::repeat_n(bi, ts.len()));
        }

        let gathered = step_indices
            .into_iter()
            .map(|idx| self.sigmas_cpu[idx])
            .collect::<Vec<f32>>();

        let mut sigma =
            Tensor::from_vec(gathered, (ts.len(),), device)?.to_dtype(sample.dtype())?;
        while sigma.rank() < sample.rank() {
            sigma = sigma.unsqueeze(sigma.rank())?;
        }

        let noise = match noise {
            Some(n) => n.clone(),
            None => Tensor::randn(0f32, 1f32, sample.shape(), device)?.to_dtype(sample.dtype())?,
        };

        let one_minus_sigma = sigma.affine(-1.0, 1.0)?;
        let a = sigma.broadcast_mul(&noise)?;
        let b = one_minus_sigma.broadcast_mul(sample)?;
        a.broadcast_add(&b)
    }

    pub fn step_inner(
        &mut self,
        model_output: &Tensor,
        timestep: f32,
        sample: &Tensor,
        per_token_timesteps: Option<&Tensor>,
    ) -> Result<FlowMatchEulerDiscreteSchedulerOutput> {
        if self.step_index.is_none() {
            self.init_step_index(timestep)?;
        }

        let mut sample_f = sample.to_dtype(DType::F32)?;
        let device = sample_f.device();

        let (current_sigma, next_sigma, dt) = if let Some(per_token_ts) = per_token_timesteps {
            let per_token_sigmas =
                per_token_ts.affine(1.0 / self.config.num_train_timesteps as f64, 0.0)?;

            let sigmas_t = self
                .sigmas
                .to_device(device)?
                .to_dtype(per_token_sigmas.dtype())?
                .unsqueeze(1)?
                .unsqueeze(2)?;

            let threshold = per_token_sigmas.unsqueeze(0)?.affine(1.0, -1e-6)?;
            let lower_mask = sigmas_t.broadcast_lt(&threshold)?;
            let lower_mask_f = lower_mask.to_dtype(per_token_sigmas.dtype())?;

            let lower_sigmas = lower_mask_f.broadcast_mul(&sigmas_t)?;
            let lower_sigmas = lower_sigmas.max(0)?;

            let current_sigma = per_token_sigmas.unsqueeze(per_token_sigmas.rank())?;
            let next_sigma = lower_sigmas.unsqueeze(lower_sigmas.rank())?;

            let dt = current_sigma.broadcast_sub(&next_sigma)?;
            (current_sigma, next_sigma, dt)
        } else {
            let idx = self.step_index.expect("step_index must be initialized");
            let sigma = self.sigmas_cpu[idx];
            let sigma_next = self.sigmas_cpu[idx + 1];

            let dt = sigma_next - sigma;

            let current_sigma = Tensor::new(sigma, device)?.to_dtype(DType::F32)?;
            let next_sigma = Tensor::new(sigma_next, device)?.to_dtype(DType::F32)?;
            let dt = Tensor::new(dt, device)?.to_dtype(DType::F32)?;
            (current_sigma, next_sigma, dt)
        };

        let prev_sample = if self.config.stochastic_sampling {
            let cs = current_sigma
                .broadcast_as(sample_f.shape())?
                .to_dtype(DType::F32)?;
            let x0 =
                sample_f.broadcast_sub(&cs.broadcast_mul(&model_output.to_dtype(DType::F32)?)?)?;

            let noise = Tensor::randn(0f32, 1f32, sample_f.shape(), device)?;

            let ns = next_sigma
                .broadcast_as(sample_f.shape())?
                .to_dtype(DType::F32)?;
            let one_minus_ns = ns.affine(-1.0, 1.0)?;
            let a = one_minus_ns.broadcast_mul(&x0)?;
            let b = ns.broadcast_mul(&noise)?;
            a.broadcast_add(&b)?
        } else {
            let dt = dt.broadcast_as(sample_f.shape())?.to_dtype(DType::F32)?;
            let scaled = model_output.to_dtype(DType::F32)?.broadcast_mul(&dt)?;
            sample_f = sample_f.broadcast_add(&scaled)?;
            sample_f
        };

        if let Some(si) = self.step_index.as_mut() {
            *si += 1;
        }

        Ok(FlowMatchEulerDiscreteSchedulerOutput { prev_sample })
    }

    pub fn timesteps(&self) -> &Tensor {
        &self.timesteps
    }

    pub fn sigmas(&self) -> &Tensor {
        &self.sigmas
    }

    pub fn len(&self) -> usize {
        self.config.num_train_timesteps
    }

    pub fn is_empty(&self) -> bool {
        self.config.num_train_timesteps == 0
    }
}

impl Scheduler for FlowMatchEulerDiscreteScheduler {
    fn config(&self) -> &SchedulerConfig {
        &self.scheduler_config
    }

    fn order(&self) -> usize {
        1
    }

    fn set_timesteps(&mut self, spec: TimestepsSpec, device: &Device, mu: f32) -> Result<Vec<i64>> {
        let (num, ts, sig) = match spec {
            TimestepsSpec::Steps(n) => (Some(n), None, None),
            TimestepsSpec::Timesteps(t) => (
                None,
                Some(t.iter().map(|&x| x as f32).collect::<Vec<f32>>()),
                None,
            ),
            TimestepsSpec::Sigmas(s) => (None, None, Some(s)),
        };

        self.set_timesteps_inner(num, device, sig.as_deref(), Some(mu), ts.as_deref())?;
        let t = self.timesteps.to_vec1::<f32>()?;
        Ok(t.into_iter().map(|x| x as i64).collect())
    }

    fn step(&mut self, noise_pred: &Tensor, timestep: i64, latents: &Tensor) -> Result<Tensor> {
        let ts = timestep as f32;
        let out = self.step_inner(noise_pred, ts, latents, None)?;
        Ok(out.prev_sample)
    }
}

impl SchedulerMixin for FlowMatchEulerDiscreteScheduler {
    fn set_timesteps(&mut self, num_steps: usize, device: &Device) -> Result<()> {
        let mu = if self.config.use_dynamic_shifting {
            Some(0.0)
        } else {
            None
        };
        self.set_timesteps_inner(Some(num_steps), device, None, mu, None)
    }

    fn timesteps(&self) -> &[f64] {
        &self.timesteps_f64
    }

    fn init_noise_sigma(&self) -> f64 {
        self.sigmas_cpu.first().copied().unwrap_or(1.0) as f64
    }

    fn scale_model_input(&self, latents: &Tensor, _t: f64) -> Result<Tensor> {
        Ok(latents.clone())
    }

    fn add_noise(&self, original: &Tensor, noise: &Tensor, t: f64) -> Result<Tensor> {
        let sigma = t as f32 / self.config.num_train_timesteps as f32;
        let scaled = noise.affine(sigma as f64, 0.0)?;
        original.broadcast_add(&scaled)
    }

    fn step(
        &mut self,
        model_output: &Tensor,
        t: f64,
        latents: &Tensor,
    ) -> Result<SchedulerStepOutput> {
        let out = self.step_inner(model_output, t as f32, latents, None)?;
        Ok(SchedulerStepOutput {
            prev_sample: out.prev_sample,
            pred_original_sample: None,
        })
    }
}

pub fn calculate_shift(
    image_seq_len: usize,
    base_seq_len: usize,
    max_seq_len: usize,
    base_shift: f32,
    max_shift: f32,
) -> f32 {
    let m = (max_shift - base_shift) / ((max_seq_len - base_seq_len) as f32);
    let b = base_shift - m * (base_seq_len as f32);
    (image_seq_len as f32) * m + b
}

pub fn retrieve_timesteps(
    scheduler: &mut dyn Scheduler,
    num_inference_steps: Option<usize>,
    device: &Device,
    timesteps: Option<Vec<i64>>,
    sigmas: Option<Vec<f32>>,
    mu: f32,
) -> Result<(Vec<i64>, usize)> {
    if timesteps.is_some() && sigmas.is_some() {
        bail!("Only one of `timesteps` or `sigmas` can be passed.");
    }

    let schedule = if let Some(ts) = timesteps {
        Scheduler::set_timesteps(scheduler, TimestepsSpec::Timesteps(ts), device, mu)?
    } else if let Some(s) = sigmas {
        Scheduler::set_timesteps(scheduler, TimestepsSpec::Sigmas(s), device, mu)?
    } else {
        let steps = num_inference_steps.unwrap_or(50);
        Scheduler::set_timesteps(scheduler, TimestepsSpec::Steps(steps), device, mu)?
    };

    let n = schedule.len();
    Ok((schedule, n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_creation() {
        let scheduler = FlowMatchEulerDiscreteScheduler::new(Default::default()).unwrap();
        assert!(scheduler.init_noise_sigma() > 0.0);
    }

    #[test]
    fn test_wan_preset() {
        let config = FlowMatchEulerDiscreteSchedulerConfig::wan_720p();
        assert_eq!(config.shift, 5.0);
    }

    #[test]
    fn test_ltx_preset() {
        let config = FlowMatchEulerDiscreteSchedulerConfig::ltx_video();
        assert_eq!(config.shift, 1.0);
    }
}
