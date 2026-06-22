//! Wan UniPC flow-sigma scheduler parity vs Diffusers.

mod wan_fixtures;

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_video::models::wan::UniPcMultistepScheduler;

use wan_fixtures::wan_scheduler_config;

fn scheduler_config_path() -> Option<PathBuf> {
    wan_scheduler_config()
}

fn fixture_path() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/wan/scheduler_unipc_flow.json");
    p.exists().then_some(p)
}

#[test]
fn load_scheduler_from_wan_fixture_config() {
    let path = match scheduler_config_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping load_scheduler: config not found");
            return;
        }
    };
    let sched = UniPcMultistepScheduler::from_config_path(&path).expect("load");
    assert!(sched.config().use_flow_sigmas);
    assert_eq!(sched.config().flow_shift, 3.0);
}

#[test]
fn scheduler_sigmas_timesteps_match_reference_if_fixture_present() {
    let fixture_path = match fixture_path() {
        Some(p) => p,
        None => {
            eprintln!(
                "Skipping scheduler parity: run scripts/wan/reference/dump_scheduler_unipc.py"
            );
            return;
        }
    };
    let config_path = match scheduler_config_path() {
        Some(p) => p,
        None => return,
    };

    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&fixture_path).expect("read")).expect("json");

    let mut sched = UniPcMultistepScheduler::from_config_path(&config_path).expect("load");
    let steps = fixture["num_inference_steps"].as_u64().unwrap() as usize;
    sched.set_timesteps(steps).expect("set_timesteps");

    let ref_sigmas: Vec<f32> = fixture["sigmas"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let ref_timesteps: Vec<i64> = fixture["timesteps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();

    assert_eq!(sched.sigmas().len(), ref_sigmas.len());
    assert_eq!(sched.timesteps().len(), ref_timesteps.len());

    for (i, (&a, &b)) in sched.sigmas().iter().zip(ref_sigmas.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "sigma[{i}] rust={a} ref={b}");
    }
    for (i, (&a, &b)) in sched
        .timesteps()
        .iter()
        .zip(ref_timesteps.iter())
        .enumerate()
    {
        assert_eq!(a, b, "timestep[{i}] mismatch");
    }
}

#[test]
fn scheduler_step_trajectory_matches_reference_if_fixture_present() {
    let fixture_path = match fixture_path() {
        Some(p) => p,
        None => return,
    };
    let config_path = match scheduler_config_path() {
        Some(p) => p,
        None => return,
    };

    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&fixture_path).expect("read")).expect("json");

    let device = Device::Cpu;
    let mut sched = UniPcMultistepScheduler::from_config_path(&config_path).expect("load");
    sched.set_timesteps(50).expect("set");

    let traj = fixture["trajectory"].as_array().expect("trajectory");
    let init: Vec<f32> = traj[0]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let mut sample = Tensor::from_vec(init, (1, 8), &device).expect("sample");

    let timesteps: Vec<i64> = sched.timesteps().iter().copied().take(5).collect();
    for (step_i, t) in timesteps.into_iter().enumerate() {
        let model_output = sample.affine(0.05, 0.0).expect("scale");
        sample = sched
            .step(&model_output, t, &sample)
            .expect("step")
            .prev_sample;

        let expected: Vec<f32> = traj[step_i + 1]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let got = sample
            .flatten_all()
            .expect("flat")
            .to_dtype(DType::F32)
            .expect("f32")
            .to_vec1::<f32>()
            .expect("vec");
        let diff = got
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(diff < 1e-4, "step {step_i} max diff {diff}");
    }
}
