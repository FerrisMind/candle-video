//! Wan 2.1 T2V performance benchmarks (CUDA + flash-attn).
//!
//! Run: `cargo bench --features flash-attn,cuda --bench wan_benchmarks`

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_video::{WanDenoiseStack, wan_patch_token_count};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use serde_json::Value;
use std::hint::black_box;

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wan")
}

fn load_step0(device: &Device) -> (Tensor, Tensor, Tensor) {
    let path = fixture_root().join("step0_reference.json");
    let raw = std::fs::read_to_string(&path).expect("step0 fixture");
    let fixture: Value = serde_json::from_str(&raw).expect("json");

    let load = |key: &str, shape_key: &str| -> Tensor {
        let shape: Vec<usize> = fixture[shape_key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let data: Vec<f32> = fixture[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        Tensor::from_vec(data, shape.as_slice(), device).unwrap()
    };

    let latents = load("latents_init", "latents_init_shape");
    let prompt_embeds = load("prompt_embeds", "prompt_embeds_shape");
    let timestep = Tensor::full(fixture["timestep"].as_i64().unwrap(), 1, device).unwrap();
    (latents, prompt_embeds, timestep)
}

fn bench_wan_transformer_forward(c: &mut Criterion) {
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("Skipping wan transformer bench: CUDA unavailable");
            return;
        }
    };

    let weights = std::env::var("WAN_WEIGHTS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/home/mod479711/Downloads/Wan2.1-T2V-1.3B-Diffusers"));

    if !weights.join("model_index.json").exists() {
        eprintln!("Skipping wan transformer bench: WAN_WEIGHTS missing");
        return;
    }

    let stack = WanDenoiseStack::load_with_vae_device(
        &weights,
        &device,
        &Device::Cpu,
        &device,
        DType::F16,
        DType::F32,
        false,
        false,
    )
    .expect("load stack");

    let (latents, prompt_embeds, timestep) = load_step0(&device);
    let latents = latents.to_dtype(DType::F16).unwrap();
    let prompt_embeds = prompt_embeds.to_dtype(DType::F16).unwrap();

    let mut group = c.benchmark_group("wan_transformer");
    group.bench_function("forward_step0_fixture", |b| {
        b.iter(|| {
            stack
                .transformer()
                .unwrap()
                .forward(
                    black_box(&latents),
                    black_box(&timestep),
                    black_box(&prompt_embeds),
                )
                .unwrap()
        })
    });
    group.finish();
}

fn bench_wan_denoise_steps(c: &mut Criterion) {
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => return,
    };

    let weights = std::env::var("WAN_WEIGHTS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/home/mod479711/Downloads/Wan2.1-T2V-1.3B-Diffusers"));

    if !weights.join("model_index.json").exists() {
        return;
    }

    let height = 256usize;
    let width = 448usize;
    let frames = 9usize;
    let tokens = wan_patch_token_count(height, width, frames);

    let mut group = c.benchmark_group("wan_denoise");
    for steps in [1usize, 3, 5] {
        group.bench_with_input(BenchmarkId::new("steps", steps), &steps, |b, &steps| {
            b.iter(|| {
                let mut stack = WanDenoiseStack::load_with_vae_device(
                    &weights,
                    &device,
                    &Device::Cpu,
                    &device,
                    DType::F16,
                    DType::F32,
                    false,
                    false,
                )
                .unwrap();
                let (latents, prompt_embeds, _) = load_step0(&device);
                let neg = prompt_embeds.zeros_like().unwrap();
                stack
                    .denoise_and_decode(
                        height,
                        width,
                        frames,
                        black_box(steps),
                        5.0,
                        Some(42),
                        Some(latents),
                        prompt_embeds.to_dtype(DType::F16).unwrap(),
                        Some(neg.to_dtype(DType::F16).unwrap()),
                    )
                    .unwrap();
            })
        });
    }
    group.finish();
    let _ = tokens;
}

criterion_group!(
    benches,
    bench_wan_transformer_forward,
    bench_wan_denoise_steps
);
criterion_main!(benches);
