//! Wan 2.1 `UniPCMultistepScheduler` (flow-sigma path only).

use std::path::Path;

use candle_core::{Result, Tensor};

use super::configs::WanSchedulerConfig;
use crate::models::ltx_video::loader::{LoaderError, load_model_config};

#[derive(Debug, Clone)]
pub struct UniPcStepOutput {
    pub prev_sample: Tensor,
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

    fn init_step_index(&mut self, timestep: i64) {
        self.step_index = Some(if let Some(begin) = self.begin_index {
            begin
        } else {
            self.index_for_timestep(timestep)
        });
    }

    fn index_for_timestep(&self, timestep: i64) -> usize {
        let matches: Vec<usize> = self
            .timesteps
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == timestep)
            .map(|(i, _)| i)
            .collect();
        match matches.len() {
            0 => self.timesteps.len().saturating_sub(1),
            2.. => matches[1],
            _ => matches[0],
        }
    }

    fn convert_model_output(&self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let idx = self.step_index.expect("step_index");
        let sigma = self.sigmas[idx];
        if self.config.predict_x0 {
            let sigma_t = Tensor::new(&[sigma], sample.device())?.to_dtype(sample.dtype())?;
            let mut shape = vec![1usize; sample.rank()];
            shape[0] = sample.dim(0)?;
            let sigma_b = sigma_t.reshape(shape.as_slice())?;
            sample.sub(&model_output.broadcast_mul(&sigma_b)?)
        } else {
            Ok(model_output.clone())
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

    fn multistep_uni_p_bh_update(
        &self,
        _model_output: &Tensor,
        sample: &Tensor,
        order: usize,
    ) -> Result<Tensor> {
        let idx = self.step_index.expect("step_index");
        let m0 = self
            .model_outputs
            .last()
            .and_then(|t| t.as_ref())
            .expect("m0");
        let x = sample;

        let sigma_t = self.sigmas[idx + 1];
        let sigma_s0 = self.sigmas[idx];
        let (alpha_t, sigma_t_a) = self.sigma_to_alpha_sigma_t(sigma_t);
        let (alpha_s0, sigma_s0_a) = self.sigma_to_alpha_sigma_t(sigma_s0);

        let lambda_t = (alpha_t / sigma_t_a).ln();
        let lambda_s0 = (alpha_s0 / sigma_s0_a).ln();
        let h = lambda_t - lambda_s0;

        let mut d1s = Vec::new();
        if order > 1 {
            for i in 1..order {
                let si = idx - i;
                let mi = self.model_outputs[self.config.solver_order - 1 - i]
                    .as_ref()
                    .expect("mi");
                let (alpha_si, sigma_si) = self.sigma_to_alpha_sigma_t(self.sigmas[si]);
                let lambda_si = (alpha_si / sigma_si).ln();
                let rk = (lambda_si - lambda_s0) / h;
                let diff = mi.sub(m0)?;
                d1s.push(diff.affine(1.0 / rk as f64, 0.0)?);
            }
        }

        let (h_phi_1, b_h) = self.bh_update_coeffs(h);

        let pred_res = if d1s.is_empty() {
            Tensor::zeros_like(x)?
        } else if order == 2 {
            d1s[0].affine(0.5, 0.0)?
        } else {
            candle_core::bail!("solver order > 2 not implemented in MVP");
        };

        let dtype = x.dtype();
        let device = x.device();
        let alpha_t_t = Tensor::new(&[alpha_t], device)?.to_dtype(dtype)?;
        let sigma_t_t = Tensor::new(&[sigma_t_a], device)?.to_dtype(dtype)?;
        let sigma_s0_t = Tensor::new(&[sigma_s0_a], device)?.to_dtype(dtype)?;
        let h_phi_1_t = Tensor::new(&[h_phi_1], device)?.to_dtype(dtype)?;
        let b_h_t = Tensor::new(&[b_h], device)?.to_dtype(dtype)?;

        let mut shape = vec![1usize; x.rank()];
        shape[0] = x.dim(0)?;

        let x_t_ = x
            .broadcast_mul(&sigma_t_t.reshape(shape.as_slice())?)?
            .broadcast_div(&sigma_s0_t.reshape(shape.as_slice())?)?
            .sub(
                &m0.broadcast_mul(&alpha_t_t.reshape(shape.as_slice())?)?
                    .broadcast_mul(&h_phi_1_t.reshape(shape.as_slice())?)?,
            )?;

        x_t_.sub(
            &pred_res
                .broadcast_mul(&alpha_t_t.reshape(shape.as_slice())?)?
                .broadcast_mul(&b_h_t.reshape(shape.as_slice())?)?,
        )
    }

    fn multistep_uni_c_bh_update(
        &self,
        this_model_output: &Tensor,
        last_sample: &Tensor,
        _this_sample: &Tensor,
        order: usize,
    ) -> Result<Tensor> {
        let idx = self.step_index.expect("step_index");
        let m0 = self
            .model_outputs
            .last()
            .and_then(|t| t.as_ref())
            .expect("m0");
        let x = last_sample;
        let model_t = this_model_output;

        let sigma_t = self.sigmas[idx];
        let sigma_s0 = self.sigmas[idx - 1];
        let (alpha_t, sigma_t_a) = self.sigma_to_alpha_sigma_t(sigma_t);
        let (alpha_s0, sigma_s0_a) = self.sigma_to_alpha_sigma_t(sigma_s0);

        let lambda_t = (alpha_t / sigma_t_a).ln();
        let lambda_s0 = (alpha_s0 / sigma_s0_a).ln();
        let h = lambda_t - lambda_s0;

        let mut d1s = Vec::new();
        if order > 1 {
            for i in 1..order {
                let si = idx - (i + 1);
                let mi = self.model_outputs[self.config.solver_order - 1 - i]
                    .as_ref()
                    .expect("mi");
                let (alpha_si, sigma_si) = self.sigma_to_alpha_sigma_t(self.sigmas[si]);
                let lambda_si = (alpha_si / sigma_si).ln();
                let rk = (lambda_si - lambda_s0) / h;
                d1s.push(mi.sub(m0)?.affine(1.0 / rk as f64, 0.0)?);
            }
        }

        let (h_phi_1, b_h) = self.bh_update_coeffs(h);
        let rho_last = if order == 1 { 0.5 } else { 0.5 }; // MVP: order 1 uses 0.5

        let corr_res = if d1s.is_empty() {
            Tensor::zeros_like(x)?
        } else {
            d1s[0].affine(0.5, 0.0)?
        };

        let d1_t = model_t.sub(m0)?;

        let dtype = x.dtype();
        let device = x.device();
        let alpha_t_t = Tensor::new(&[alpha_t], device)?.to_dtype(dtype)?;
        let sigma_t_t = Tensor::new(&[sigma_t_a], device)?.to_dtype(dtype)?;
        let sigma_s0_t = Tensor::new(&[sigma_s0_a], device)?.to_dtype(dtype)?;
        let h_phi_1_t = Tensor::new(&[h_phi_1], device)?.to_dtype(dtype)?;
        let b_h_t = Tensor::new(&[b_h], device)?.to_dtype(dtype)?;
        let rho_last_t = Tensor::new(&[rho_last], device)?.to_dtype(dtype)?;

        let mut shape = vec![1usize; x.rank()];
        shape[0] = x.dim(0)?;

        let x_t_ = x
            .broadcast_mul(&sigma_t_t.reshape(shape.as_slice())?)?
            .broadcast_div(&sigma_s0_t.reshape(shape.as_slice())?)?
            .sub(
                &m0.broadcast_mul(&alpha_t_t.reshape(shape.as_slice())?)?
                    .broadcast_mul(&h_phi_1_t.reshape(shape.as_slice())?)?,
            )?;

        let correction =
            corr_res.add(&d1_t.broadcast_mul(&rho_last_t.reshape(shape.as_slice())?)?)?;
        x_t_.sub(
            &correction
                .broadcast_mul(&alpha_t_t.reshape(shape.as_slice())?)?
                .broadcast_mul(&b_h_t.reshape(shape.as_slice())?)?,
        )
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
            self.init_step_index(timestep);
        }
        let idx = self.step_index.expect("step_index");

        let use_corrector = idx > 0
            && !self.config.disable_corrector.contains(&(idx - 1))
            && self.last_sample.is_some();

        let converted = self.convert_model_output(model_output, sample)?;
        let mut sample = sample.clone();
        if use_corrector {
            sample = self.multistep_uni_c_bh_update(
                &converted,
                self.last_sample.as_ref().expect("last_sample"),
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
}
