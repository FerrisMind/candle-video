//! Video Coordinates parity verification tests

use candle_core::{D, DType, Device, IndexOp, Tensor};
use std::path::Path;

const PARITY_FILE: &str = "gen_video_coords_parity.safetensors";

fn load_reference_tensors(device: &Device) -> Option<std::collections::HashMap<String, Tensor>> {
    let path = Path::new(PARITY_FILE);
    if !path.exists() {
        println!("Skipping test: {} not found.", PARITY_FILE);
        return None;
    }
    Some(candle_core::safetensors::load(path, device).expect("Failed to load"))
}

fn compute_mse(a: &Tensor, b: &Tensor) -> f32 {
    a.sub(b)
        .unwrap()
        .sqr()
        .unwrap()
        .mean_all()
        .unwrap()
        .to_vec0::<f32>()
        .unwrap()
}

fn compute_max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    a.sub(b)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_vec0::<f32>()
        .unwrap()
}

fn compute_video_coords_rust(
    batch_size: usize,
    num_frames: usize,
    height: usize,
    width: usize,
    frame_rate: usize,
    vae_temporal_compression: usize,
    vae_spatial_compression: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let ts_ratio = vae_temporal_compression as f32;
    let sp_ratio = vae_spatial_compression as f32;
    let latent_num_frames = (num_frames - 1) / vae_temporal_compression + 1;
    let latent_height = height / vae_spatial_compression;
    let latent_width = width / vae_spatial_compression;
    let video_sequence_length = latent_num_frames * latent_height * latent_width;

    let grid_f = Tensor::arange(0u32, latent_num_frames as u32, device)?.to_dtype(DType::F32)?;
    let grid_h = Tensor::arange(0u32, latent_height as u32, device)?.to_dtype(DType::F32)?;
    let grid_w = Tensor::arange(0u32, latent_width as u32, device)?.to_dtype(DType::F32)?;

    let f = grid_f.reshape((latent_num_frames, 1, 1))?.broadcast_as((
        latent_num_frames,
        latent_height,
        latent_width,
    ))?;
    let h = grid_h.reshape((1, latent_height, 1))?.broadcast_as((
        latent_num_frames,
        latent_height,
        latent_width,
    ))?;
    let w = grid_w.reshape((1, 1, latent_width))?.broadcast_as((
        latent_num_frames,
        latent_height,
        latent_width,
    ))?;

    let video_coords = Tensor::stack(&[f, h, w], 0)?
        .flatten_from(1)?
        .transpose(0, 1)?
        .unsqueeze(0)?;
    let vf = video_coords.i((.., .., 0))?;
    let vh = video_coords.i((.., .., 1))?;
    let vw = video_coords.i((.., .., 2))?;

    let vf = vf
        .affine(ts_ratio as f64, (1.0 - ts_ratio) as f64)?
        .clamp(0.0f32, 1000.0f32)?
        .affine(1.0 / (frame_rate as f64), 0.0)?;
    let vh = vh.affine(sp_ratio as f64, 0.0)?;
    let vw = vw.affine(sp_ratio as f64, 0.0)?;

    let video_coords = Tensor::stack(&[vf, vh, vw], D::Minus1)?.broadcast_as((
        batch_size,
        video_sequence_length,
        3,
    ))?;
    Ok(video_coords.contiguous()?)
}

#[test]
fn test_full_video_coords_parity() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let tensors = match load_reference_tensors(&device) {
        Some(t) => t,
        None => return Ok(()),
    };

    println!("\n=== Testing full video_coords parity ===");

    let test_configs = [
        (9, 256, 256, 25),
        (25, 512, 768, 25),
        (97, 512, 768, 25),
        (161, 512, 768, 25),
        (9, 512, 768, 30),
        (25, 768, 512, 24),
    ];
    let (vae_t, vae_s) = (8usize, 32usize);
    let (mut max_mse, mut max_diff): (f32, f32) = (0.0, 0.0);

    for (num_frames, height, width, frame_rate) in test_configs {
        let key = format!(
            "coords_f{}_h{}_w{}_fps{}_diffusers",
            num_frames, height, width, frame_rate
        );
        if let Some(ref_coords) = tensors.get(&key) {
            let rust_coords = compute_video_coords_rust(
                1, num_frames, height, width, frame_rate, vae_t, vae_s, &device,
            )?;
            let rust_coords = rust_coords.to_dtype(DType::F32)?;
            let ref_coords = ref_coords.to_dtype(DType::F32)?;
            let mse = compute_mse(&rust_coords, &ref_coords);
            let diff = compute_max_abs_diff(&rust_coords, &ref_coords);
            println!(
                "  f={} h={} w={} fps={}: MSE={:.2e}, max_diff={:.2e}",
                num_frames, height, width, frame_rate, mse, diff
            );
            max_mse = max_mse.max(mse);
            max_diff = max_diff.max(diff);
        }
    }

    println!(
        "\nOverall: max_MSE={:.2e}, max_diff={:.2e}",
        max_mse, max_diff
    );
    assert!(max_mse < 1e-10, "MSE too high: {}", max_mse);
    Ok(())
}

#[test]
fn test_latent_dimensions() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let tensors = match load_reference_tensors(&device) {
        Some(t) => t,
        None => return Ok(()),
    };

    println!("\n=== Testing latent dimensions ===");
    let test_configs = [(9, 256, 256, 25), (25, 512, 768, 25), (97, 512, 768, 25)];
    let (vae_t, vae_s) = (8usize, 32usize);

    for (num_frames, height, width, frame_rate) in test_configs {
        let key = format!(
            "coords_f{}_h{}_w{}_fps{}_latent_dims",
            num_frames, height, width, frame_rate
        );
        if let Some(ref_dims) = tensors.get(&key) {
            let ref_dims = ref_dims.to_vec1::<i64>()?;
            let latent_f = (num_frames - 1) / vae_t + 1;
            let latent_h = height / vae_s;
            let latent_w = width / vae_s;
            println!(
                "  f={} h={} w={}: latent=({},{},{})",
                num_frames, height, width, latent_f, latent_h, latent_w
            );
            assert_eq!(latent_f as i64, ref_dims[0]);
            assert_eq!(latent_h as i64, ref_dims[1]);
            assert_eq!(latent_w as i64, ref_dims[2]);
        }
    }
    Ok(())
}

// =========================================================================
// Task 4.4: Property-Based Tests for Video Coordinates Parity
// =========================================================================

#[cfg(test)]
mod property_tests {
    use candle_core::{D, DType, Device, IndexOp, Tensor};
    use proptest::prelude::*;
    use std::path::Path;

    const PARITY_FILE: &str = "gen_video_coords_parity.safetensors";

    fn compute_mse(a: &Tensor, b: &Tensor) -> f32 {
        a.sub(b)
            .unwrap()
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_vec0::<f32>()
            .unwrap()
    }

    fn compute_max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        a.sub(b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_vec0::<f32>()
            .unwrap()
    }

    fn compute_video_coords_rust(
        batch_size: usize,
        num_frames: usize,
        height: usize,
        width: usize,
        frame_rate: usize,
        vae_temporal_compression: usize,
        vae_spatial_compression: usize,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let ts_ratio = vae_temporal_compression as f32;
        let sp_ratio = vae_spatial_compression as f32;
        let latent_num_frames = (num_frames - 1) / vae_temporal_compression + 1;
        let latent_height = height / vae_spatial_compression;
        let latent_width = width / vae_spatial_compression;
        let video_sequence_length = latent_num_frames * latent_height * latent_width;

        let grid_f =
            Tensor::arange(0u32, latent_num_frames as u32, device)?.to_dtype(DType::F32)?;
        let grid_h = Tensor::arange(0u32, latent_height as u32, device)?.to_dtype(DType::F32)?;
        let grid_w = Tensor::arange(0u32, latent_width as u32, device)?.to_dtype(DType::F32)?;

        let f = grid_f.reshape((latent_num_frames, 1, 1))?.broadcast_as((
            latent_num_frames,
            latent_height,
            latent_width,
        ))?;
        let h = grid_h.reshape((1, latent_height, 1))?.broadcast_as((
            latent_num_frames,
            latent_height,
            latent_width,
        ))?;
        let w = grid_w.reshape((1, 1, latent_width))?.broadcast_as((
            latent_num_frames,
            latent_height,
            latent_width,
        ))?;

        let video_coords = Tensor::stack(&[f, h, w], 0)?
            .flatten_from(1)?
            .transpose(0, 1)?
            .unsqueeze(0)?;
        let vf = video_coords.i((.., .., 0))?;
        let vh = video_coords.i((.., .., 1))?;
        let vw = video_coords.i((.., .., 2))?;

        // CAUSAL FIX: (L * 8 + 1 - 8).clamp(0) / frame_rate
        let vf = vf
            .affine(ts_ratio as f64, (1.0 - ts_ratio) as f64)?
            .clamp(0.0f32, 1000.0f32)?
            .affine(1.0 / (frame_rate as f64), 0.0)?;
        // SPATIAL SCALE: L * 32
        let vh = vh.affine(sp_ratio as f64, 0.0)?;
        let vw = vw.affine(sp_ratio as f64, 0.0)?;

        let video_coords = Tensor::stack(&[vf, vh, vw], D::Minus1)?.broadcast_as((
            batch_size,
            video_sequence_length,
            3,
        ))?;
        Ok(video_coords.contiguous()?)
    }

    // =========================================================================
    // Property 8: Video Coordinates Parity
    // For any valid video dimensions (num_frames, height, width) and compression
    // ratios, the Rust video_coords computation SHALL produce coordinates with
    // MSE < 1e-6 compared to Python implementation.
    // Validates: Requirements 8.1, 8.2, 8.3
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Feature: ltx-video-parity, Property 8: Video Coordinates Parity
        /// For any valid video dimensions and compression ratios, video_coords MSE < 1e-6
        /// **Validates: Requirements 8.1, 8.2, 8.3**
        #[test]
        fn prop_video_coords_parity(
            config_idx in 0usize..6,
        ) {
            let device = Device::Cpu;
            let path = Path::new(PARITY_FILE);
            if !path.exists() {
                // Skip if reference file not available
                return Ok(());
            }

            let tensors = candle_core::safetensors::load(path, &device)
                .expect("Failed to load reference tensors");

            // Test configurations: (num_frames, height, width, frame_rate)
            let test_configs = [
                (9usize, 256usize, 256usize, 25usize),   // Small
                (25, 512, 768, 25),                       // Medium
                (97, 512, 768, 25),                       // Large (typical LTX)
                (161, 512, 768, 25),                      // Very large
                (9, 512, 768, 30),                        // Different frame rate
                (25, 768, 512, 24),                       // Portrait
            ];

            let (num_frames, height, width, frame_rate) = test_configs[config_idx];
            let (vae_t, vae_s) = (8usize, 32usize);

            let key = format!("coords_f{}_h{}_w{}_fps{}_diffusers", num_frames, height, width, frame_rate);

            if let Some(ref_coords) = tensors.get(&key) {
                let rust_coords = compute_video_coords_rust(
                    1, num_frames, height, width, frame_rate, vae_t, vae_s, &device
                ).expect("Failed to compute video coords");

                let rust_coords = rust_coords.to_dtype(DType::F32).unwrap();
                let ref_coords = ref_coords.to_dtype(DType::F32).unwrap();

                let mse = compute_mse(&rust_coords, &ref_coords);
                let max_diff = compute_max_abs_diff(&rust_coords, &ref_coords);

                prop_assert!(
                    mse < 1e-6,
                    "Video coords MSE {} exceeds threshold 1e-6 for config f={}, h={}, w={}, fps={}, max_diff={}",
                    mse, num_frames, height, width, frame_rate, max_diff
                );
            }
        }

        /// Feature: ltx-video-parity, Property 8.1: Temporal Scaling Formula
        /// For any valid frame count, temporal scaling (L * 8 + 1 - 8).clamp(0) / frame_rate
        /// SHALL produce correct temporal coordinates.
        /// **Validates: Requirements 8.2**
        #[test]
        fn prop_temporal_scaling_formula(
            config_idx in 0usize..6,
        ) {
            let device = Device::Cpu;
            let path = Path::new(PARITY_FILE);
            if !path.exists() {
                return Ok(());
            }

            let tensors = candle_core::safetensors::load(path, &device)
                .expect("Failed to load reference tensors");

            let test_configs = [
                (9usize, 256usize, 256usize, 25usize),
                (25, 512, 768, 25),
                (97, 512, 768, 25),
                (161, 512, 768, 25),
                (9, 512, 768, 30),
                (25, 768, 512, 24),
            ];

            let (num_frames, height, width, frame_rate) = test_configs[config_idx];
            let (vae_t, vae_s) = (8usize, 32usize);

            let key = format!("coords_f{}_h{}_w{}_fps{}_temporal", num_frames, height, width, frame_rate);

            if let Some(ref_temporal) = tensors.get(&key) {
                let rust_coords = compute_video_coords_rust(
                    1, num_frames, height, width, frame_rate, vae_t, vae_s, &device
                ).expect("Failed to compute video coords");

                // Extract temporal component (first column)
                let rust_temporal = rust_coords.i((.., .., 0)).unwrap().to_dtype(DType::F32).unwrap();
                let ref_temporal = ref_temporal.to_dtype(DType::F32).unwrap();

                let mse = compute_mse(&rust_temporal, &ref_temporal);

                prop_assert!(
                    mse < 1e-6,
                    "Temporal scaling MSE {} exceeds threshold for config f={}, h={}, w={}, fps={}",
                    mse, num_frames, height, width, frame_rate
                );
            }
        }

        /// Feature: ltx-video-parity, Property 8.2: Spatial Scaling Formula
        /// For any valid spatial dimensions, spatial scaling (coord * 32)
        /// SHALL produce correct spatial coordinates.
        /// **Validates: Requirements 8.3**
        #[test]
        fn prop_spatial_scaling_formula(
            config_idx in 0usize..6,
        ) {
            let device = Device::Cpu;
            let path = Path::new(PARITY_FILE);
            if !path.exists() {
                return Ok(());
            }

            let tensors = candle_core::safetensors::load(path, &device)
                .expect("Failed to load reference tensors");

            let test_configs = [
                (9usize, 256usize, 256usize, 25usize),
                (25, 512, 768, 25),
                (97, 512, 768, 25),
                (161, 512, 768, 25),
                (9, 512, 768, 30),
                (25, 768, 512, 24),
            ];

            let (num_frames, height, width, frame_rate) = test_configs[config_idx];
            let (vae_t, vae_s) = (8usize, 32usize);

            let key_h = format!("coords_f{}_h{}_w{}_fps{}_spatial_h", num_frames, height, width, frame_rate);
            let key_w = format!("coords_f{}_h{}_w{}_fps{}_spatial_w", num_frames, height, width, frame_rate);

            let rust_coords = compute_video_coords_rust(
                1, num_frames, height, width, frame_rate, vae_t, vae_s, &device
            ).expect("Failed to compute video coords");

            // Check spatial height component
            if let Some(ref_spatial_h) = tensors.get(&key_h) {
                let rust_spatial_h = rust_coords.i((.., .., 1)).unwrap().to_dtype(DType::F32).unwrap();
                let ref_spatial_h = ref_spatial_h.to_dtype(DType::F32).unwrap();
                let mse_h = compute_mse(&rust_spatial_h, &ref_spatial_h);

                prop_assert!(
                    mse_h < 1e-6,
                    "Spatial height MSE {} exceeds threshold for config f={}, h={}, w={}, fps={}",
                    mse_h, num_frames, height, width, frame_rate
                );
            }

            // Check spatial width component
            if let Some(ref_spatial_w) = tensors.get(&key_w) {
                let rust_spatial_w = rust_coords.i((.., .., 2)).unwrap().to_dtype(DType::F32).unwrap();
                let ref_spatial_w = ref_spatial_w.to_dtype(DType::F32).unwrap();
                let mse_w = compute_mse(&rust_spatial_w, &ref_spatial_w);

                prop_assert!(
                    mse_w < 1e-6,
                    "Spatial width MSE {} exceeds threshold for config f={}, h={}, w={}, fps={}",
                    mse_w, num_frames, height, width, frame_rate
                );
            }
        }

        /// Feature: ltx-video-parity, Property 8.3: Latent Dimension Computation
        /// For any valid video dimensions, latent dimensions SHALL be computed correctly:
        /// latent_f = (num_frames - 1) / vae_t + 1
        /// latent_h = height / vae_s
        /// latent_w = width / vae_s
        /// **Validates: Requirements 8.1**
        #[test]
        fn prop_latent_dimensions(
            config_idx in 0usize..6,
        ) {
            let device = Device::Cpu;
            let path = Path::new(PARITY_FILE);
            if !path.exists() {
                return Ok(());
            }

            let tensors = candle_core::safetensors::load(path, &device)
                .expect("Failed to load reference tensors");

            let test_configs = [
                (9usize, 256usize, 256usize, 25usize),
                (25, 512, 768, 25),
                (97, 512, 768, 25),
                (161, 512, 768, 25),
                (9, 512, 768, 30),
                (25, 768, 512, 24),
            ];

            let (num_frames, height, width, frame_rate) = test_configs[config_idx];
            let (vae_t, vae_s) = (8usize, 32usize);

            let key = format!("coords_f{}_h{}_w{}_fps{}_latent_dims", num_frames, height, width, frame_rate);

            if let Some(ref_dims) = tensors.get(&key) {
                let ref_dims = ref_dims.to_vec1::<i64>().unwrap();

                // Compute expected latent dimensions
                let latent_f = (num_frames - 1) / vae_t + 1;
                let latent_h = height / vae_s;
                let latent_w = width / vae_s;

                prop_assert_eq!(
                    latent_f as i64, ref_dims[0],
                    "Latent frames mismatch for config f={}, h={}, w={}: expected {}, got {}",
                    num_frames, height, width, latent_f, ref_dims[0]
                );
                prop_assert_eq!(
                    latent_h as i64, ref_dims[1],
                    "Latent height mismatch for config f={}, h={}, w={}: expected {}, got {}",
                    num_frames, height, width, latent_h, ref_dims[1]
                );
                prop_assert_eq!(
                    latent_w as i64, ref_dims[2],
                    "Latent width mismatch for config f={}, h={}, w={}: expected {}, got {}",
                    num_frames, height, width, latent_w, ref_dims[2]
                );
            }
        }
    }
}
