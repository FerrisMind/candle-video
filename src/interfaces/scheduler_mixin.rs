use candle_core::{Device, Result, Tensor};

#[derive(Debug)]
pub struct SchedulerStepOutput {
    pub prev_sample: Tensor,
    pub pred_original_sample: Option<Tensor>,
}

pub trait SchedulerMixin {
    fn set_timesteps(&mut self, num_steps: usize, device: &Device) -> Result<()>;
    fn timesteps(&self) -> &[f64];
    fn init_noise_sigma(&self) -> f64;
    fn scale_model_input(&self, latents: &Tensor, t: f64) -> Result<Tensor>;
    fn add_noise(&self, original: &Tensor, noise: &Tensor, t: f64) -> Result<Tensor>;
    fn step(
        &mut self,
        model_output: &Tensor,
        t: f64,
        latents: &Tensor,
    ) -> Result<SchedulerStepOutput>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[derive(Default)]
    struct DummyScheduler {
        timesteps: Vec<f64>,
    }

    impl SchedulerMixin for DummyScheduler {
        fn set_timesteps(&mut self, num_steps: usize, _device: &Device) -> Result<()> {
            self.timesteps = (0..num_steps).rev().map(|v| v as f64).collect();
            Ok(())
        }

        fn timesteps(&self) -> &[f64] {
            &self.timesteps
        }

        fn init_noise_sigma(&self) -> f64 {
            1.0
        }

        fn scale_model_input(&self, latents: &Tensor, _t: f64) -> Result<Tensor> {
            Ok(latents.clone())
        }

        fn add_noise(&self, original: &Tensor, noise: &Tensor, _t: f64) -> Result<Tensor> {
            original.broadcast_add(noise)
        }

        fn step(
            &mut self,
            model_output: &Tensor,
            _t: f64,
            latents: &Tensor,
        ) -> Result<SchedulerStepOutput> {
            let prev_sample = latents.broadcast_sub(model_output)?;
            Ok(SchedulerStepOutput {
                prev_sample,
                pred_original_sample: None,
            })
        }
    }

    #[test]
    fn uses_scheduler_mixin_contract() {
        let device = Device::Cpu;
        let mut scheduler = DummyScheduler::default();
        scheduler.set_timesteps(3, &device).unwrap();
        assert_eq!(scheduler.timesteps(), &[2.0, 1.0, 0.0]);
        assert_eq!(scheduler.init_noise_sigma(), 1.0);

        let latents = Tensor::zeros(1usize, DType::F32, &device).unwrap();
        let noise = Tensor::ones(1usize, DType::F32, &device).unwrap();
        let _ = scheduler.add_noise(&latents, &noise, 0.0).unwrap();
        let _ = scheduler.scale_model_input(&latents, 0.0).unwrap();
        let step = scheduler.step(&noise, 0.0, &latents).unwrap();
        let value = step.prev_sample.to_vec1::<f32>().unwrap()[0];
        assert_eq!(value, -1.0);
    }
}
