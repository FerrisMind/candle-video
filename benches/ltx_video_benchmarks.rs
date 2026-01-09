//! Performance benchmarks for LTX-Video components.
//!
//! Run with: cargo bench --features flash-attn,cudnn
//!
//! These benchmarks measure the performance of:
//! - Scheduler step operations
//! - Transformer forward pass
//! - VAE decode operations
//!
//! For full pipeline benchmarks, use the example runner:
//!   cargo run --example ltx-video --release --features flash-attn,cudnn -- \
//!     --prompt "..." --benchmark
//!
//! Component benchmarks are useful for:
//! - Identifying performance bottlenecks
//! - Comparing before/after optimization changes
//! - Regression testing

use candle_core::{DType, Device, Tensor};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

// ============================================================================
// Scheduler Benchmarks
// ============================================================================

fn bench_scheduler_step(c: &mut Criterion) {
    use candle_video::models::ltx_video::scheduler::{
        FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerConfig,
    };

    let device = Device::Cpu;
    let mut group = c.benchmark_group("scheduler");

    // Test different latent sizes
    let latent_sizes = [(1, 128, 1024), (1, 128, 4096), (2, 128, 4096)];

    for (batch, channels, seq_len) in latent_sizes {
        let config = FlowMatchEulerDiscreteSchedulerConfig {
            use_dynamic_shifting: true,
            ..Default::default()
        };

        // Create test tensors
        let model_output = Tensor::randn(0f32, 1f32, (batch, seq_len, channels), &device).unwrap();
        let sample = Tensor::randn(0f32, 1f32, (batch, seq_len, channels), &device).unwrap();

        let id = BenchmarkId::new(
            "step",
            format!("batch={}_seq={}_ch={}", batch, seq_len, channels),
        );

        group.bench_with_input(
            id,
            &(model_output, sample, config),
            |b, (model_output, sample, config)| {
                b.iter(|| {
                    let mut scheduler =
                        FlowMatchEulerDiscreteScheduler::new(config.clone()).unwrap();
                    scheduler
                        .set_timesteps(Some(20), &device, None, Some(1.0), None)
                        .unwrap();
                    scheduler.step(black_box(model_output), 900.0, black_box(sample), None)
                })
            },
        );
    }

    group.finish();
}

fn bench_scheduler_set_timesteps(c: &mut Criterion) {
    use candle_video::models::ltx_video::scheduler::{
        FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerConfig,
    };

    let device = Device::Cpu;
    let mut group = c.benchmark_group("scheduler");

    let step_counts = [10, 20, 40, 50];

    for num_steps in step_counts {
        let config = FlowMatchEulerDiscreteSchedulerConfig {
            use_dynamic_shifting: true,
            ..Default::default()
        };

        let id = BenchmarkId::new("set_timesteps", format!("steps={}", num_steps));

        group.bench_with_input(id, &num_steps, |b, &num_steps| {
            b.iter(|| {
                let mut scheduler = FlowMatchEulerDiscreteScheduler::new(config.clone()).unwrap();
                scheduler.set_timesteps(Some(black_box(num_steps)), &device, None, Some(1.0), None)
            })
        });
    }

    group.finish();
}

// ============================================================================
// Transformer Benchmarks (requires model weights)
// ============================================================================

#[cfg(feature = "flash-attn")]
fn bench_transformer_forward(c: &mut Criterion) {
    use candle_nn::VarBuilder;
    use candle_video::models::ltx_video::ltx_transformer::{
        LtxVideoTransformer3DModel, LtxVideoTransformer3DModelConfig,
    };

    // Skip if no CUDA device available
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("Skipping transformer benchmark: CUDA not available");
            return;
        }
    };

    let mut group = c.benchmark_group("transformer");
    group.sample_size(10); // Reduce sample size for expensive operations

    // Test configurations matching typical inference scenarios
    let configs = [
        // (batch, seq_len, num_frames, height, width, encoder_seq_len)
        (1, 1024, 9, 32, 32, 128),  // Small: 256x256, 9 frames
        (1, 4096, 9, 64, 64, 128),  // Medium: 512x512, 9 frames
        (1, 8192, 17, 64, 64, 128), // Large: 512x512, 17 frames
    ];

    for (batch, seq_len, num_frames, height, width, enc_seq_len) in configs {
        let model_config = LtxVideoTransformer3DModelConfig {
            num_layers: 4, // Reduced for benchmarking
            ..Default::default()
        };

        // Use zeros for weights (we're measuring compute, not accuracy)
        let vb = VarBuilder::zeros(DType::BF16, &device);
        let model = match LtxVideoTransformer3DModel::new(&model_config, vb.pp("transformer")) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Failed to create transformer: {}", e);
                continue;
            }
        };

        // Create test tensors
        let hidden_states = Tensor::randn(
            0f32,
            1f32,
            (batch, seq_len, model_config.in_channels),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let encoder_hidden_states = Tensor::randn(
            0f32,
            1f32,
            (batch, enc_seq_len, model_config.caption_channels),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let timestep = Tensor::new(&[500.0f32], &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let encoder_mask = Tensor::ones((batch, enc_seq_len), DType::F32, &device).unwrap();

        let id = BenchmarkId::new(
            "forward",
            format!(
                "batch={}_seq={}_frames={}_h={}_w={}",
                batch, seq_len, num_frames, height, width
            ),
        );

        group.bench_with_input(
            id,
            &(hidden_states, encoder_hidden_states, timestep, encoder_mask),
            |b, (hidden_states, encoder_hidden_states, timestep, encoder_mask)| {
                b.iter(|| {
                    model.forward(
                        black_box(hidden_states),
                        black_box(encoder_hidden_states),
                        black_box(timestep),
                        Some(black_box(encoder_mask)),
                        num_frames,
                        height,
                        width,
                        None,
                        None,
                        None,
                    )
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// VAE Benchmarks (requires model weights)
// ============================================================================

#[cfg(feature = "flash-attn")]
fn bench_vae_decode(c: &mut Criterion) {
    use candle_nn::VarBuilder;
    use candle_video::models::ltx_video::vae::{
        AutoencoderKLLtxVideo, AutoencoderKLLtxVideoConfig,
    };

    // Skip if no CUDA device available
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("Skipping VAE benchmark: CUDA not available");
            return;
        }
    };

    let mut group = c.benchmark_group("vae");
    group.sample_size(10); // Reduce sample size for expensive operations
    group.measurement_time(std::time::Duration::from_secs(30)); // Limit measurement time

    // Test configurations - smaller sizes for faster benchmarking
    // Latent shape: (batch, channels, frames, height, width)
    let configs = [
        // (batch, latent_frames, latent_height, latent_width)
        (1, 5, 16, 24), // Small: 128x192, 33 frames
        (1, 9, 24, 32), // Medium: 192x256, 65 frames
    ];

    for (batch, latent_frames, latent_height, latent_width) in configs {
        let vae_config = AutoencoderKLLtxVideoConfig::default();

        // Use zeros for weights
        let vb = VarBuilder::zeros(DType::BF16, &device);
        let vae = match AutoencoderKLLtxVideo::new(vae_config.clone(), vb.pp("vae")) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Failed to create VAE: {}", e);
                continue;
            }
        };

        // Create test latents
        let latents = Tensor::randn(
            0f32,
            1f32,
            (
                batch,
                vae_config.latent_channels,
                latent_frames,
                latent_height,
                latent_width,
            ),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

        let id = BenchmarkId::new(
            "decode",
            format!(
                "batch={}_frames={}_h={}_w={}",
                batch, latent_frames, latent_height, latent_width
            ),
        );

        group.bench_with_input(id, &latents, |b, latents| {
            b.iter(|| vae.decode(black_box(latents), None, false, false))
        });
    }

    group.finish();
}

// ============================================================================
// CPU-only benchmarks (always run)
// ============================================================================

fn bench_tensor_operations(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("tensor_ops");
    group.sample_size(20); // Reduce samples for slow operations

    // Benchmark common tensor operations used in the pipeline
    // Using smaller sizes for faster benchmarking
    let sizes = [(1, 512, 1024), (1, 1024, 1024)];

    for (batch, seq, dim) in sizes {
        let a = Tensor::randn(0f32, 1f32, (batch, seq, dim), &device).unwrap();
        let b = Tensor::randn(0f32, 1f32, (batch, seq, dim), &device).unwrap();

        // Broadcast add
        let id = BenchmarkId::new("broadcast_add", format!("{}x{}x{}", batch, seq, dim));
        group.bench_with_input(id, &(&a, &b), |bench, (a, b)| {
            bench.iter(|| a.broadcast_add(black_box(b)))
        });

        // Broadcast mul
        let id = BenchmarkId::new("broadcast_mul", format!("{}x{}x{}", batch, seq, dim));
        group.bench_with_input(id, &(&a, &b), |bench, (a, b)| {
            bench.iter(|| a.broadcast_mul(black_box(b)))
        });

        // Matmul (for attention-like operations) - smaller size
        let q = Tensor::randn(0f32, 1f32, (batch, 8, seq, 64), &device).unwrap();
        let k = Tensor::randn(0f32, 1f32, (batch, 8, seq, 64), &device).unwrap();
        let id = BenchmarkId::new("matmul_attention", format!("{}x8x{}x64", batch, seq));
        group.bench_with_input(id, &(&q, &k), |bench, (q, k)| {
            bench.iter(|| {
                let kt = k.transpose(2, 3).unwrap();
                q.matmul(black_box(&kt))
            })
        });
    }

    group.finish();
}

fn bench_dtype_conversions(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("dtype_conversion");
    group.sample_size(50);

    let sizes = [(1, 1024, 1024)];

    for (batch, seq, dim) in sizes {
        let f32_tensor = Tensor::randn(0f32, 1f32, (batch, seq, dim), &device).unwrap();

        // F32 to BF16
        let id = BenchmarkId::new("f32_to_bf16", format!("{}x{}x{}", batch, seq, dim));
        group.bench_with_input(id, &f32_tensor, |bench, tensor| {
            bench.iter(|| tensor.to_dtype(black_box(DType::BF16)))
        });

        // BF16 to F32
        let bf16_tensor = f32_tensor.to_dtype(DType::BF16).unwrap();
        let id = BenchmarkId::new("bf16_to_f32", format!("{}x{}x{}", batch, seq, dim));
        group.bench_with_input(id, &bf16_tensor, |bench, tensor| {
            bench.iter(|| tensor.to_dtype(black_box(DType::F32)))
        });
    }

    group.finish();
}

// ============================================================================
// Criterion configuration
// ============================================================================

// CPU-only benchmarks (always available)
criterion_group!(
    cpu_benches,
    bench_scheduler_step,
    bench_scheduler_set_timesteps,
    bench_tensor_operations,
    bench_dtype_conversions,
);

// GPU benchmarks (require flash-attn feature)
#[cfg(feature = "flash-attn")]
criterion_group!(gpu_benches, bench_transformer_forward, bench_vae_decode,);

#[cfg(feature = "flash-attn")]
criterion_main!(cpu_benches, gpu_benches);

#[cfg(not(feature = "flash-attn"))]
criterion_main!(cpu_benches);
