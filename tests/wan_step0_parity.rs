//! Step-0 denoise parity vs Diffusers dump (`scripts/wan/compare_step0.py`).

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_video::models::wan::{
    UniPcMultistepScheduler, WanPipeline, WanTransformer3DModel,
};

fn fixture_path() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wan/step0_reference.json");
    p.exists().then_some(p)
}

fn transformer_dir() -> Option<PathBuf> {
    let p = PathBuf::from("/home/mod479711/Downloads/Wan2.1-T2V-1.3B-Diffusers/transformer");
    p.join("config.json").exists().then_some(p)
}

fn load_f32(data: &[serde_json::Value], shape: &[usize], device: &Device) -> Tensor {
    let v: Vec<f32> = data.iter().map(|x| x.as_f64().unwrap() as f32).collect();
    Tensor::from_vec(v, shape, device).expect("tensor")
}

#[test]
fn step0_noise_pred_matches_diffusers_if_fixture_present() {
    let fixture_path = match fixture_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: run scripts/wan/compare_step0.py first");
            return;
        }
    };
    let transformer_dir = match transformer_dir() {
        Some(p) => p,
        None => return,
    };

    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&fixture_path).expect("read")).expect("json");

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let model = WanTransformer3DModel::load(&transformer_dir, &device, DType::F32).expect("load");

    let lat_shape: Vec<usize> = fixture["latents_init_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let latents = load_f32(fixture["latents_init"].as_array().unwrap(), &lat_shape, &device);

    let p_shape: Vec<usize> = fixture["prompt_embeds_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let prompt_embeds =
        load_f32(fixture["prompt_embeds"].as_array().unwrap(), &p_shape, &device);
    let neg_shape: Vec<usize> = fixture["neg_embeds_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let neg_embeds = load_f32(fixture["neg_embeds"].as_array().unwrap(), &neg_shape, &device);

    let t = fixture["timestep"].as_i64().unwrap();
    let timestep = Tensor::full(t, 1, &device).expect("timestep");

    let noise_cond = model
        .forward(&latents, &timestep, &prompt_embeds)
        .expect("cond");
    let noise_uncond = model
        .forward(&latents, &timestep, &neg_embeds)
        .expect("uncond");
    let noise_pred =
        WanPipeline::apply_classifier_free_guidance(&noise_cond, &noise_uncond, 5.0).expect("cfg");

    let max_diff = {
        let ref_all: Vec<f32> = fixture["noise_pred"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let got_all = noise_pred
            .flatten_all()
            .expect("flat")
            .to_dtype(DType::F32)
            .expect("f32")
            .to_vec1::<f32>()
            .expect("vec");
        ref_all
            .iter()
            .zip(got_all.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
    };

    assert!(
        max_diff < 2e-3,
        "noise_pred diverges from Diffusers at step 0: max diff {max_diff}"
    );

    let sched_path = PathBuf::from(
        "/home/mod479711/Downloads/Wan2.1-T2V-1.3B-Diffusers/scheduler/scheduler_config.json",
    );
    let mut sched = UniPcMultistepScheduler::from_config_path(&sched_path).expect("sched");
    sched.set_timesteps(10).expect("set");
    sched.set_begin_index(0);
    let latents_after = sched
        .step(&noise_pred, t, &latents)
        .expect("step")
        .prev_sample;

    let ref_after: Vec<f32> = fixture["latents_after_first16"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let got_after: Vec<f32> = latents_after
        .flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("vec")
        .into_iter()
        .take(16)
        .collect();
    let max_after = ref_after
        .iter()
        .zip(got_after.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_after < 2e-3,
        "latents after step 0 diverge: max diff {max_after}"
    );
}
