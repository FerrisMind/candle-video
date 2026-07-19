//! Wan 2.1 schedulers and compatibility profiles.

use std::path::Path;
use std::str::FromStr;

use candle_core::{Result, Tensor};

use super::configs::WanSchedulerConfig;
use crate::models::ltx_video::loader::{LoaderError, load_model_config};

/// Scheduler compatibility profiles exposed by the Wan API.
///
/// `Diffusers` is the default for the local checkpoint because its
/// `scheduler_config.json` is a `UniPCMultistepScheduler`. `OfficialWan`
/// reproduces the Alibaba Wan sampling defaults (UniPC with a configurable
/// sample shift), while `FlowMatchEuler` is the first-order flow-matching
/// scheduler used by newer Diffusers pipelines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WanSchedulerProfile {
    #[default]
    Diffusers,
    OfficialWan,
    FlowMatchEuler,
}

impl FromStr for WanSchedulerProfile {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "diffusers" | "diffuser" | "unipc" => Ok(Self::Diffusers),
            "official-wan" | "official_wan" | "official" | "wan" => Ok(Self::OfficialWan),
            "flow-match-euler" | "flow_match_euler" | "flow-euler" | "euler" => {
                Ok(Self::FlowMatchEuler)
            }
            other => Err(format!(
                "unknown Wan scheduler profile `{other}`; expected diffusers, official-wan, or flow-match-euler"
            )),
        }
    }
}

impl std::fmt::Display for WanSchedulerProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Diffusers => "diffusers",
            Self::OfficialWan => "official-wan",
            Self::FlowMatchEuler => "flow-match-euler",
        })
    }
}

#[derive(Debug, Clone)]
pub struct UniPcStepOutput {
    pub prev_sample: Tensor,
}

/// Configuration for the deterministic flow-matching Euler scheduler.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlowMatchEulerDiscreteSchedulerConfig {
    pub num_train_timesteps: usize,
    pub shift: f32,
}

impl Default for FlowMatchEulerDiscreteSchedulerConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            shift: 5.0,
        }
    }
}

/// First-order Euler solver for flow-matching Wan models.
///
/// The schedule follows Diffusers' `FlowMatchEulerDiscreteScheduler` used by
/// the pinned 0.33.x reference: its constructor shifts the training extrema,
/// `set_timesteps` interpolates those extrema and applies the requested shift,
/// then appends a terminal zero sigma. The rational transform is
/// `s' = shift*s / (1 + (shift-1)*s)`.
#[derive(Debug)]
pub struct FlowMatchEulerDiscreteScheduler {
    config: FlowMatchEulerDiscreteSchedulerConfig,
    sigmas: Vec<f32>,
    timesteps: Vec<i64>,
    /// Floating-point timesteps used by Diffusers and by the transformer.
    /// The integer view is retained for backwards compatibility with the
    /// original Wan scheduler API.
    timesteps_f32: Vec<f32>,
    sigma_min: f32,
    sigma_max: f32,
    step_index: Option<usize>,
    begin_index: Option<usize>,
}

impl FlowMatchEulerDiscreteScheduler {
    pub fn new(config: FlowMatchEulerDiscreteSchedulerConfig) -> Result<Self> {
        if config.num_train_timesteps == 0 {
            candle_core::bail!("FlowMatchEuler scheduler requires num_train_timesteps > 0");
        }
        if !config.shift.is_finite() || config.shift <= 0.0 {
            candle_core::bail!("FlowMatchEuler scheduler shift must be finite and positive");
        }
        let sigma_min = Self::shift_sigma(config.shift, 1.0 / config.num_train_timesteps as f64);
        let sigma_max = Self::shift_sigma(config.shift, 1.0);
        Ok(Self {
            config,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            timesteps_f32: Vec::new(),
            sigma_min,
            sigma_max,
            step_index: None,
            begin_index: None,
        })
    }

    fn shift_sigma(shift: f32, sigma: f64) -> f32 {
        (shift as f64 * sigma / (1.0 + (shift as f64 - 1.0) * sigma)) as f32
    }

    pub fn config(&self) -> &FlowMatchEulerDiscreteSchedulerConfig {
        &self.config
    }

    pub fn shift(&self) -> f32 {
        self.config.shift
    }

    pub fn set_shift(&mut self, shift: f32) -> Result<()> {
        if !shift.is_finite() || shift <= 0.0 {
            candle_core::bail!("FlowMatchEuler scheduler shift must be finite and positive");
        }
        self.config.shift = shift;
        self.sigma_min = Self::shift_sigma(shift, 1.0 / self.config.num_train_timesteps as f64);
        self.sigma_max = Self::shift_sigma(shift, 1.0);
        Ok(())
    }

    pub fn timesteps(&self) -> &[i64] {
        &self.timesteps
    }

    /// Return the unquantized Diffusers timestep schedule.
    pub fn timesteps_f32(&self) -> &[f32] {
        &self.timesteps_f32
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    pub fn set_begin_index(&mut self, begin_index: usize) {
        self.begin_index = Some(begin_index);
        self.step_index = None;
    }

    pub fn set_timesteps(&mut self, num_inference_steps: usize) -> Result<()> {
        if num_inference_steps == 0 {
            candle_core::bail!("FlowMatchEuler scheduler requires at least one inference step");
        }
        // Diffusers first stores a shifted training grid in the constructor,
        // then linearly interpolates between its shifted extrema and applies
        // the fixed shift once more in `set_timesteps`. Preserve that order
        // (including the float32 boundary values) for checkpoint parity.
        let train = self.config.num_train_timesteps as f64;
        let start_t = self.sigma_max as f64 * train;
        let end_t = self.sigma_min as f64 * train;
        let base_sigmas = if num_inference_steps == 1 {
            vec![self.sigma_max as f64]
        } else {
            let denominator = (num_inference_steps - 1) as f64;
            (0..num_inference_steps)
                .map(|index| {
                    let t = start_t + (end_t - start_t) * index as f64 / denominator;
                    t / train
                })
                .collect::<Vec<_>>()
        };
        let mut sigmas = base_sigmas
            .into_iter()
            .map(|sigma| Self::shift_sigma(self.config.shift, sigma))
            .collect::<Vec<_>>();
        let timesteps_f32 = sigmas
            .iter()
            .map(|sigma| *sigma * self.config.num_train_timesteps as f32)
            .collect::<Vec<_>>();
        let timesteps = timesteps_f32
            .iter()
            .map(|timestep| *timestep as i64)
            .collect::<Vec<_>>();
        sigmas.push(0.0);
        self.sigmas = sigmas;
        self.timesteps = timesteps;
        self.timesteps_f32 = timesteps_f32;
        self.step_index = None;
        self.begin_index = None;
        Ok(())
    }

    fn index_for_timestep(&self, timestep: i64) -> Option<usize> {
        self.timesteps.iter().position(|value| *value == timestep)
    }

    fn index_for_timestep_f32(&self, timestep: f32) -> Option<usize> {
        self.timesteps_f32
            .iter()
            .position(|value| (*value - timestep).abs() <= 1e-5)
    }

    fn resolve_step_index(&self, timestep: Option<usize>) -> Result<usize> {
        let index = match self.step_index {
            Some(index) => index,
            None => self.begin_index.or(timestep).ok_or_else(|| {
                candle_core::Error::Msg("FlowMatchEuler timestep is not in the schedule".into())
            })?,
        };
        if index + 1 >= self.sigmas.len() {
            candle_core::bail!("FlowMatchEuler step index {index} is past the schedule");
        }
        Ok(index)
    }

    fn step_at_index(
        &mut self,
        model_output: &Tensor,
        index: usize,
        sample: &Tensor,
    ) -> Result<UniPcStepOutput> {
        let dt = (self.sigmas[index + 1] - self.sigmas[index]) as f64;
        let sample_f32 = sample.to_dtype(candle_core::DType::F32)?;
        let prediction = model_output.to_dtype(candle_core::DType::F32)?;
        let prev_sample = sample_f32
            .add(&prediction.affine(dt, 0.0)?)?
            .to_dtype(model_output.dtype())?;
        self.step_index = Some(index + 1);
        Ok(UniPcStepOutput { prev_sample })
    }

    pub fn step(
        &mut self,
        model_output: &Tensor,
        timestep: i64,
        sample: &Tensor,
    ) -> Result<UniPcStepOutput> {
        if self.sigmas.len() < 2 || self.timesteps.is_empty() {
            candle_core::bail!("call set_timesteps before FlowMatchEuler::step");
        }
        let timestep_index = self.index_for_timestep(timestep).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "timestep {timestep} is not in FlowMatchEuler schedule"
            ))
        })?;
        let index = self.resolve_step_index(Some(timestep_index))?;
        self.step_at_index(model_output, index, sample)
    }

    /// Diffusers-compatible step accepting the unquantized floating timestep.
    pub fn step_f32(
        &mut self,
        model_output: &Tensor,
        timestep: f32,
        sample: &Tensor,
    ) -> Result<UniPcStepOutput> {
        if self.sigmas.len() < 2 || self.timesteps_f32.is_empty() {
            candle_core::bail!("call set_timesteps before FlowMatchEuler::step_f32");
        }
        let timestep_index = self.index_for_timestep_f32(timestep).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "timestep {timestep} is not in FlowMatchEuler schedule"
            ))
        })?;
        let index = self.resolve_step_index(Some(timestep_index))?;
        self.step_at_index(model_output, index, sample)
    }
}

/// Runtime scheduler selected by [`WanSchedulerProfile`].
#[derive(Debug)]
pub enum WanScheduler {
    UniPc {
        scheduler: Box<UniPcMultistepScheduler>,
        profile: WanSchedulerProfile,
    },
    FlowMatchEuler(FlowMatchEulerDiscreteScheduler),
}

impl WanScheduler {
    pub fn from_profile(
        profile: WanSchedulerProfile,
        config: WanSchedulerConfig,
        shift: Option<f32>,
    ) -> Result<Self> {
        match profile {
            WanSchedulerProfile::Diffusers => {
                let mut scheduler = UniPcMultistepScheduler::new(config)?;
                if let Some(shift) = shift {
                    scheduler.set_flow_shift(shift)?;
                }
                Ok(Self::UniPc {
                    scheduler: Box::new(scheduler),
                    profile: WanSchedulerProfile::Diffusers,
                })
            }
            WanSchedulerProfile::OfficialWan => {
                let mut scheduler = UniPcMultistepScheduler::new(config)?;
                scheduler.set_flow_shift(shift.unwrap_or(5.0))?;
                Ok(Self::UniPc {
                    scheduler: Box::new(scheduler),
                    profile: WanSchedulerProfile::OfficialWan,
                })
            }
            WanSchedulerProfile::FlowMatchEuler => Ok(Self::FlowMatchEuler(
                FlowMatchEulerDiscreteScheduler::new(FlowMatchEulerDiscreteSchedulerConfig {
                    num_train_timesteps: config.num_train_timesteps,
                    shift: shift.unwrap_or(5.0),
                })?,
            )),
        }
    }

    pub fn profile(&self) -> WanSchedulerProfile {
        match self {
            Self::UniPc { profile, .. } => *profile,
            Self::FlowMatchEuler(_) => WanSchedulerProfile::FlowMatchEuler,
        }
    }

    pub fn timesteps(&self) -> &[i64] {
        match self {
            Self::UniPc { scheduler, .. } => scheduler.timesteps(),
            Self::FlowMatchEuler(scheduler) => scheduler.timesteps(),
        }
    }

    /// Return scheduler timesteps in the floating-point representation used by
    /// Diffusers. UniPC's schedule is integer-valued, so its compatibility
    /// view is lossless.
    pub fn timesteps_f32(&self) -> Vec<f32> {
        match self {
            Self::UniPc { scheduler, .. } => scheduler
                .timesteps()
                .iter()
                .map(|timestep| *timestep as f32)
                .collect(),
            Self::FlowMatchEuler(scheduler) => scheduler.timesteps_f32().to_vec(),
        }
    }

    pub fn set_begin_index(&mut self, begin_index: usize) {
        match self {
            Self::UniPc { scheduler, .. } => scheduler.set_begin_index(begin_index),
            Self::FlowMatchEuler(scheduler) => scheduler.set_begin_index(begin_index),
        }
    }

    pub fn set_timesteps(&mut self, num_inference_steps: usize) -> Result<()> {
        match self {
            Self::UniPc { scheduler, .. } => scheduler.set_timesteps(num_inference_steps),
            Self::FlowMatchEuler(scheduler) => scheduler.set_timesteps(num_inference_steps),
        }
    }

    pub fn step(
        &mut self,
        model_output: &Tensor,
        timestep: i64,
        sample: &Tensor,
    ) -> Result<UniPcStepOutput> {
        match self {
            Self::UniPc { scheduler, .. } => scheduler.step(model_output, timestep, sample),
            Self::FlowMatchEuler(scheduler) => scheduler.step(model_output, timestep, sample),
        }
    }

    /// Step using a Diffusers-compatible floating-point timestep.
    pub fn step_f32(
        &mut self,
        model_output: &Tensor,
        timestep: f32,
        sample: &Tensor,
    ) -> Result<UniPcStepOutput> {
        match self {
            Self::UniPc { scheduler, .. } => scheduler.step(model_output, timestep as i64, sample),
            Self::FlowMatchEuler(scheduler) => scheduler.step_f32(model_output, timestep, sample),
        }
    }
}

#[derive(Debug)]
pub struct UniPcMultistepScheduler {
    config: WanSchedulerConfig,
    sigmas: Vec<f32>,
    timesteps: Vec<i64>,
    num_inference_steps: Option<usize>,
    model_outputs: Vec<Option<Tensor>>,
    timestep_list: Vec<Option<i64>>,
    lower_order_nums: usize,
    last_sample: Option<Tensor>,
    step_index: Option<usize>,
    begin_index: Option<usize>,
    this_order: usize,
}

impl UniPcMultistepScheduler {
    pub fn new(config: WanSchedulerConfig) -> Result<Self> {
        if config.solver_order == 0 {
            candle_core::bail!("UniPC solver_order must be greater than zero");
        }
        if !config.flow_shift.is_finite() || config.flow_shift <= 0.0 {
            candle_core::bail!("UniPC flow_shift must be finite and positive");
        }
        if !config.use_flow_sigmas {
            candle_core::bail!("UniPcMultistepScheduler MVP requires use_flow_sigmas=true");
        }
        if config.prediction_type != "flow_prediction" {
            candle_core::bail!(
                "UniPcMultistepScheduler MVP requires prediction_type=flow_prediction"
            );
        }
        let order = config.solver_order;
        Ok(Self {
            config,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            num_inference_steps: None,
            model_outputs: vec![None; order],
            timestep_list: vec![None; order],
            lower_order_nums: 0,
            last_sample: None,
            step_index: None,
            begin_index: None,
            this_order: 1,
        })
    }

    pub fn from_config_path(path: impl AsRef<Path>) -> std::result::Result<Self, LoaderError> {
        let config: WanSchedulerConfig = load_model_config(path.as_ref())?;
        Self::new(config).map_err(LoaderError::Candle)
    }

    pub fn config(&self) -> &WanSchedulerConfig {
        &self.config
    }

    /// Override the flow-sigma shift for an explicit compatibility profile.
    pub fn set_flow_shift(&mut self, shift: f32) -> Result<()> {
        if !shift.is_finite() || shift <= 0.0 {
            candle_core::bail!("UniPC flow_shift must be finite and positive");
        }
        self.config.flow_shift = shift;
        Ok(())
    }

    pub fn timesteps(&self) -> &[i64] {
        &self.timesteps
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    pub fn set_begin_index(&mut self, begin_index: usize) {
        self.begin_index = Some(begin_index);
    }

    pub fn set_timesteps(&mut self, num_inference_steps: usize) -> Result<()> {
        let n = num_inference_steps;
        if n == 0 {
            candle_core::bail!("UniPC scheduler requires at least one inference step");
        }
        let train = self.config.num_train_timesteps as f32;
        let end = 1.0 / train;
        let mut sigmas: Vec<f32> = (0..n)
            .map(|i| 1.0 + (i as f32) * (end - 1.0) / n as f32)
            .collect();

        let shift = self.config.flow_shift;
        for s in &mut sigmas {
            *s = shift * *s / (1.0 + (shift - 1.0) * *s);
        }
        if (sigmas[0] - 1.0).abs() < 1e-6 {
            sigmas[0] -= 1e-6;
        }

        let timesteps: Vec<i64> = sigmas.iter().map(|s| (s * train) as i64).collect();

        let sigma_last = match self.config.final_sigmas_type.as_str() {
            "zero" => 0.0,
            "sigma_min" => *sigmas.last().unwrap_or(&0.0),
            other => candle_core::bail!("unsupported final_sigmas_type {other}"),
        };
        sigmas.push(sigma_last);

        self.sigmas = sigmas;
        self.timesteps = timesteps;
        self.num_inference_steps = Some(n);
        self.model_outputs = vec![None; self.config.solver_order];
        self.timestep_list = vec![None; self.config.solver_order];
        self.lower_order_nums = 0;
        self.last_sample = None;
        self.step_index = None;
        Ok(())
    }

    fn sigma_to_alpha_sigma_t(&self, sigma: f32) -> (f32, f32) {
        (1.0 - sigma, sigma)
    }

    fn init_step_index(&mut self, timestep: i64) -> Result<()> {
        let index = if let Some(begin) = self.begin_index {
            begin
        } else {
            self.index_for_timestep(timestep)?
        };
        if index >= self.sigmas.len().saturating_sub(1) {
            candle_core::bail!(
                "scheduler timestep index {} is outside the {}-step schedule",
                index,
                self.timesteps.len()
            );
        }
        self.step_index = Some(index);
        Ok(())
    }

    fn index_for_timestep(&self, timestep: i64) -> Result<usize> {
        let matches: Vec<usize> = self
            .timesteps
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == timestep)
            .map(|(i, _)| i)
            .collect();
        match matches.len() {
            0 => candle_core::bail!("timestep {} is not present in the UniPC schedule", timestep),
            2.. => Ok(matches[1]),
            _ => Ok(matches[0]),
        }
    }

    /// ponytail: rewrite to scalar `affine` to avoid per-step Tensor::new +
    /// to_dtype + reshape churn (was visible in `perf` heatmap).
    fn convert_model_output(&self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let idx = self.step_index.ok_or_else(|| {
            candle_core::Error::Msg("scheduler step index is not initialized".into())
        })?;
        let sigma = *self.sigmas.get(idx).ok_or_else(|| {
            candle_core::Error::Msg(format!("scheduler sigma index {} is out of bounds", idx))
        })?;
        if self.config.predict_x0 {
            // x0 = sample - sigma * model_output
            let sample_f32 = sample.to_dtype(candle_core::DType::F32)?;
            let model_output_f32 = model_output.to_dtype(candle_core::DType::F32)?;
            Ok(sample_f32.sub(&model_output_f32.affine(sigma as f64, 0.0)?)?)
        } else {
            model_output.to_dtype(candle_core::DType::F32)
        }
    }

    fn bh_update_coeffs(&self, h: f32) -> (f32, f32) {
        let hh = if self.config.predict_x0 { -h } else { h };
        let h_phi_1 = hh.exp() - 1.0;
        let b_h = if self.config.solver_type == "bh1" {
            hh
        } else {
            hh.exp_m1()
        };
        (h_phi_1, b_h)
    }

    /// ponytail: collapsed to scalar affine per term (no Tensor::new/reshape).
    /// Formula:
    ///   x_t_ = x * (sigma_t/sigma_s0)
    ///        - m0 * alpha_t * h_phi_1
    ///        - pred_res * alpha_t * b_h
    fn multistep_uni_p_bh_update(
        &self,
        _model_output: &Tensor,
        sample: &Tensor,
        order: usize,
    ) -> Result<Tensor> {
        let idx = self.step_index.ok_or_else(|| {
            candle_core::Error::Msg("scheduler step index is not initialized".into())
        })?;
        let m0 = self
            .model_outputs
            .last()
            .and_then(|t| t.as_ref())
            .ok_or_else(|| candle_core::Error::Msg("UniPC model history is empty".into()))?;
        let x = sample;

        let sigma_t = *self.sigmas.get(idx + 1).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "scheduler sigma index {} is out of bounds",
                idx + 1
            ))
        })?;
        let sigma_s0 = *self.sigmas.get(idx).ok_or_else(|| {
            candle_core::Error::Msg(format!("scheduler sigma index {} is out of bounds", idx))
        })?;
        let (alpha_t, sigma_t_a) = self.sigma_to_alpha_sigma_t(sigma_t);
        let (alpha_s0, sigma_s0_a) = self.sigma_to_alpha_sigma_t(sigma_s0);

        let lambda_t = (alpha_t / sigma_t_a).ln();
        let lambda_s0 = (alpha_s0 / sigma_s0_a).ln();
        let h = lambda_t - lambda_s0;

        let pred_res = if order > 1 && idx >= 1 {
            let i = 1;
            let si = idx - i;
            if si < self.sigmas.len() {
                let mi = self
                    .model_outputs
                    .get(self.config.solver_order - 1 - i)
                    .and_then(|t| t.as_ref())
                    .ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "UniPC model history is missing order-{} output",
                            i + 1
                        ))
                    })?;
                let (alpha_si, sigma_si) =
                    self.sigma_to_alpha_sigma_t(*self.sigmas.get(si).ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "scheduler sigma index {} is out of bounds",
                            si
                        ))
                    })?);
                let lambda_si = (alpha_si / sigma_si).ln();
                let rk = (lambda_si - lambda_s0) / h;
                // Order-2 predictor uses d1s[0] * 0.5 (matches diffusers).
                Some(mi.sub(m0)?.affine(0.5 / rk as f64, 0.0)?)
            } else {
                None
            }
        } else {
            None
        };

        let (h_phi_1, b_h) = self.bh_update_coeffs(h);
        let c1 = (sigma_t_a / sigma_s0_a) as f64;
        let c2 = (alpha_t * h_phi_1) as f64;
        let c3 = (alpha_t * b_h) as f64;

        let mut acc = x.affine(c1, 0.0)?;
        acc = acc.sub(&m0.affine(c2, 0.0)?)?;
        if let Some(pred) = pred_res {
            acc = acc.sub(&pred.affine(c3, 0.0)?)?;
        }
        Ok(acc)
    }

    fn multistep_uni_c_bh_update(
        &self,
        this_model_output: &Tensor,
        last_sample: &Tensor,
        _this_sample: &Tensor,
        order: usize,
    ) -> Result<Tensor> {
        let idx = self.step_index.ok_or_else(|| {
            candle_core::Error::Msg("scheduler step index is not initialized".into())
        })?;
        let m0 = self
            .model_outputs
            .last()
            .and_then(|t| t.as_ref())
            .ok_or_else(|| candle_core::Error::Msg("UniPC model history is empty".into()))?;
        let x = last_sample;
        let model_t = this_model_output;

        let sigma_t = *self.sigmas.get(idx).ok_or_else(|| {
            candle_core::Error::Msg(format!("scheduler sigma index {} is out of bounds", idx))
        })?;
        let previous_idx = idx.checked_sub(1).ok_or_else(|| {
            candle_core::Error::Msg("UniPC corrector requires a previous timestep".into())
        })?;
        let sigma_s0 = *self.sigmas.get(previous_idx).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "scheduler sigma index {} is out of bounds",
                previous_idx
            ))
        })?;
        let (alpha_t, sigma_t_a) = self.sigma_to_alpha_sigma_t(sigma_t);
        let (alpha_s0, sigma_s0_a) = self.sigma_to_alpha_sigma_t(sigma_s0);

        let lambda_t = (alpha_t / sigma_t_a).ln();
        let lambda_s0 = (alpha_s0 / sigma_s0_a).ln();
        let h = lambda_t - lambda_s0;

        let mut d1s = Vec::new();
        if order > 1 {
            for i in 1..order {
                let Some(si) = idx.checked_sub(i + 1) else {
                    break;
                };
                if si >= self.sigmas.len() {
                    break;
                }
                let mi = self
                    .model_outputs
                    .get(self.config.solver_order - 1 - i)
                    .and_then(|t| t.as_ref())
                    .ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "UniPC model history is missing order-{} output",
                            i + 1
                        ))
                    })?;
                let (alpha_si, sigma_si) =
                    self.sigma_to_alpha_sigma_t(*self.sigmas.get(si).ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "scheduler sigma index {} is out of bounds",
                            si
                        ))
                    })?);
                let lambda_si = (alpha_si / sigma_si).ln();
                let rk = (lambda_si - lambda_s0) / h;
                d1s.push(mi.sub(m0)?.affine(1.0 / rk as f64, 0.0)?);
            }
        }

        let (h_phi_1, b_h) = self.bh_update_coeffs(h);
        let rho_last = 0.5; // MVP: order 1 uses 0.5

        let d1_t = model_t.sub(m0)?;

        let c1 = (sigma_t_a / sigma_s0_a) as f64;
        let c2 = (alpha_t * h_phi_1) as f64;
        let b_h_rho = (alpha_t * b_h * rho_last) as f64;
        let c3 = (alpha_t * b_h) as f64;

        let mut acc = x.affine(c1, 0.0)?;
        acc = acc.sub(&m0.affine(c2, 0.0)?)?;
        acc = acc.sub(&d1_t.affine(b_h_rho, 0.0)?)?;
        let corr_res = if d1s.is_empty() {
            // no-op, just bookkeeping
            None
        } else {
            Some(d1s[0].add(&d1_t)?.affine(0.5, 0.0)?)
        };
        if let Some(c) = corr_res {
            acc = acc.sub(&c.affine(c3, 0.0)?)?;
        }
        Ok(acc)
    }

    pub fn step(
        &mut self,
        model_output: &Tensor,
        timestep: i64,
        sample: &Tensor,
    ) -> Result<UniPcStepOutput> {
        if self.num_inference_steps.is_none() {
            candle_core::bail!("call set_timesteps before step");
        }
        if self.step_index.is_none() {
            self.init_step_index(timestep)?;
        }
        let idx = self.step_index.ok_or_else(|| {
            candle_core::Error::Msg("scheduler step index is not initialized".into())
        })?;

        let use_corrector = idx > 0
            && !self.config.disable_corrector.contains(&(idx - 1))
            && self.last_sample.is_some();

        // UniPC's reference implementation performs solver arithmetic in
        // float32 even when the transformer emits F16/BF16 tensors. Keeping
        // the sample and history in one dtype also avoids Candle's strict
        // binary-op dtype checks on CUDA.
        let mut sample = sample.to_dtype(candle_core::DType::F32)?;
        let converted = self.convert_model_output(model_output, &sample)?;
        if use_corrector {
            sample = self.multistep_uni_c_bh_update(
                &converted,
                self.last_sample.as_ref().ok_or_else(|| {
                    candle_core::Error::Msg("UniPC corrector sample history is empty".into())
                })?,
                &sample,
                self.this_order,
            )?;
        }

        for i in 0..self.config.solver_order - 1 {
            self.model_outputs[i] = self.model_outputs[i + 1].take();
            self.timestep_list[i] = self.timestep_list[i + 1];
        }
        self.model_outputs[self.config.solver_order - 1] = Some(converted);
        self.timestep_list[self.config.solver_order - 1] = Some(timestep);

        let this_order = if self.config.lower_order_final {
            self.config
                .solver_order
                .min(self.timesteps.len().saturating_sub(idx))
        } else {
            self.config.solver_order
        };
        self.this_order = this_order.min(self.lower_order_nums + 1).max(1);

        self.last_sample = Some(sample.clone());
        let prev_sample = self.multistep_uni_p_bh_update(model_output, &sample, self.this_order)?;

        if self.lower_order_nums < self.config.solver_order {
            self.lower_order_nums += 1;
        }
        self.step_index = Some(idx + 1);

        Ok(UniPcStepOutput { prev_sample })
    }

    pub fn scale_model_input(&self, sample: &Tensor) -> Result<Tensor> {
        Ok(sample.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn wan_scheduler_config() -> WanSchedulerConfig {
        WanSchedulerConfig {
            num_train_timesteps: 1000,
            flow_shift: 3.0,
            prediction_type: "flow_prediction".to_string(),
            use_flow_sigmas: true,
            solver_order: 2,
            solver_type: "bh2".to_string(),
            beta_start: 0.0001,
            beta_end: 0.02,
            beta_schedule: "linear".to_string(),
            timestep_spacing: "linspace".to_string(),
            predict_x0: true,
            lower_order_final: true,
            final_sigmas_type: "zero".to_string(),
            disable_corrector: vec![],
        }
    }

    #[test]
    fn flow_sigmas_50_steps_shape() {
        let mut s = UniPcMultistepScheduler::new(wan_scheduler_config()).expect("new");
        s.set_timesteps(50).expect("set");
        assert_eq!(s.timesteps().len(), 50);
        assert_eq!(s.sigmas().len(), 51);
        assert!((s.sigmas()[0] - 0.999999).abs() < 1e-4);
        assert_eq!(s.sigmas()[50], 0.0);
    }

    #[test]
    fn step_runs_on_tiny_tensor() {
        let device = Device::Cpu;
        let mut s = UniPcMultistepScheduler::new(wan_scheduler_config()).expect("new");
        s.set_timesteps(4).expect("set");
        let mut sample = Tensor::randn(0f32, 1f32, (1, 8), &device).expect("sample");
        let timesteps: Vec<i64> = s.timesteps().to_vec();
        for t in timesteps {
            let model_output = sample.clone();
            sample = s.step(&model_output, t, &sample).expect("step").prev_sample;
        }
    }

    #[test]
    fn unipc_step_accepts_mixed_model_and_sample_dtypes() {
        let device = Device::Cpu;
        let mut scheduler = UniPcMultistepScheduler::new(wan_scheduler_config()).expect("new");
        scheduler.set_timesteps(1).expect("set");
        let sample = Tensor::zeros((1, 8), candle_core::DType::F32, &device).expect("sample");
        let model_output = sample
            .to_dtype(candle_core::DType::F16)
            .expect("model output");
        let output = scheduler
            .step(&model_output, scheduler.timesteps()[0], &sample)
            .expect("mixed dtype step")
            .prev_sample;
        assert_eq!(output.dtype(), candle_core::DType::F32);
    }

    #[test]
    fn flow_match_euler_applies_rational_shift_and_terminal_zero() {
        let mut scheduler =
            FlowMatchEulerDiscreteScheduler::new(FlowMatchEulerDiscreteSchedulerConfig {
                num_train_timesteps: 1000,
                shift: 5.0,
            })
            .expect("new");
        scheduler.set_timesteps(4).expect("timesteps");
        assert_eq!(scheduler.timesteps().len(), 4);
        assert_eq!(scheduler.sigmas().len(), 5);
        assert_eq!(scheduler.sigmas().last().copied(), Some(0.0));
        let expected = 5.0 / (1.0 + 4.0 * 1.0);
        assert_eq!(scheduler.timesteps(), &[1000, 909, 717, 24]);
        assert!((scheduler.sigmas()[0] - 1.0).abs() < 1e-7);
        let expected_last = 0.024414066;
        assert!((scheduler.sigmas()[3] - expected_last).abs() < 1e-6);
        let expected_timesteps = [1000.0, 909.70715, 717.31744, 24.414066];
        for (actual, expected) in scheduler.timesteps_f32().iter().zip(expected_timesteps) {
            assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
        }
        assert!(expected > 0.0);
    }

    #[test]
    fn scheduler_profiles_parse_without_aliasing_official_wan() {
        assert_eq!(
            "official-wan"
                .parse::<WanSchedulerProfile>()
                .expect("profile"),
            WanSchedulerProfile::OfficialWan
        );
        assert_eq!(
            "flow-match-euler"
                .parse::<WanSchedulerProfile>()
                .expect("profile"),
            WanSchedulerProfile::FlowMatchEuler
        );
        assert!("unknown".parse::<WanSchedulerProfile>().is_err());
    }
}
