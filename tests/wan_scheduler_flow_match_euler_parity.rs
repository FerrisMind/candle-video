//! Wan FlowMatch Euler parity against Diffusers 0.33.0.

use std::path::PathBuf;

use candle_core::{Device, Tensor};
use candle_video::models::wan::{
    FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerConfig,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct SchedulerFixture {
    diffusers_version: String,
    num_train_timesteps: usize,
    shift: f32,
    num_inference_steps: usize,
    timesteps: Vec<f32>,
    sigmas: Vec<f32>,
    initial_sample: Vec<f32>,
    model_outputs: Vec<Vec<f32>>,
    trajectory: Vec<Vec<f32>>,
}

fn fixture() -> anyhow::Result<SchedulerFixture> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/wan/scheduler_flow_match_euler.json");
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

fn max_abs_diff(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f32::max)
}

#[test]
fn flow_match_euler_schedule_and_trajectory_match_diffusers() -> anyhow::Result<()> {
    let fixture = fixture()?;
    assert_eq!(fixture.diffusers_version, "0.33.0");

    let mut scheduler =
        FlowMatchEulerDiscreteScheduler::new(FlowMatchEulerDiscreteSchedulerConfig {
            num_train_timesteps: fixture.num_train_timesteps,
            shift: fixture.shift,
        })?;
    scheduler.set_timesteps(fixture.num_inference_steps)?;

    let timestep_diff = max_abs_diff(scheduler.timesteps_f32(), &fixture.timesteps);
    let sigma_diff = max_abs_diff(scheduler.sigmas(), &fixture.sigmas);
    assert!(
        timestep_diff < 1e-4,
        "FlowMatch timestep max diff {timestep_diff}"
    );
    assert!(sigma_diff < 1e-6, "FlowMatch sigma max diff {sigma_diff}");

    assert_eq!(fixture.model_outputs.len(), fixture.trajectory.len());
    let device = Device::Cpu;
    let width = fixture.initial_sample.len();
    let mut sample = Tensor::from_vec(fixture.initial_sample, (1, width), &device)?;
    for ((timestep, model_output), expected) in scheduler
        .timesteps_f32()
        .to_vec()
        .into_iter()
        .zip(fixture.model_outputs)
        .zip(fixture.trajectory)
    {
        let model_output = Tensor::from_vec(model_output, (1, width), &device)?;
        sample = scheduler
            .step_f32(&model_output, timestep, &sample)?
            .prev_sample;
        let actual = sample.flatten_all()?.to_vec1::<f32>()?;
        let step_diff = max_abs_diff(&actual, &expected);
        assert!(step_diff < 2e-6, "FlowMatch step max diff {step_diff}");
    }

    Ok(())
}
