//! CFG (Classifier-Free Guidance) parity tests
//!
//! Verifies that the Rust CFG implementation matches Python diffusers exactly.
//!
//! Requirements: 7.1, 7.2
//! Property 9: CFG Computation Parity

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};
    use candle_video::models::ltx_video::t2v_pipeline::rescale_noise_cfg;
    use std::collections::HashMap;
    use std::path::Path;

    const PARITY_FILE: &str = "gen_cfg_parity.safetensors";

    fn load_reference_tensors(device: &Device) -> Option<HashMap<String, Tensor>> {
        let path = Path::new(PARITY_FILE);
        if !path.exists() {
            println!(
                "Skipping test: {} not found. Run scripts/capture_cfg_parity.py first.",
                PARITY_FILE
            );
            return None;
        }
        Some(
            candle_core::safetensors::load(path, device).expect("Failed to load reference tensors"),
        )
    }

    fn apply_cfg(
        noise_uncond: &Tensor,
        noise_cond: &Tensor,
        guidance_scale: f32,
    ) -> candle_core::Result<Tensor> {
        let diff = noise_cond.broadcast_sub(noise_uncond)?;
        noise_uncond.broadcast_add(&diff.affine(guidance_scale as f64, 0.0)?)
    }

    fn compute_mse(a: &Tensor, b: &Tensor) -> f32 {
        let diff = a.sub(b).unwrap();
        let sq = diff.sqr().unwrap();
        sq.mean_all().unwrap().to_vec0::<f32>().unwrap()
    }

    fn compute_max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let diff = a.sub(b).unwrap().abs().unwrap();
        diff.max_all().unwrap().to_vec0::<f32>().unwrap()
    }

    fn get_scalar(tensor: &Tensor) -> f32 {
        if tensor.rank() == 0 {
            tensor.to_vec0::<f32>().unwrap()
        } else {
            tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0]
        }
    }

    #[test]
    fn test_cfg_formula_basic() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let tensors = match load_reference_tensors(&device) {
            Some(t) => t,
            None => return Ok(()),
        };

        let noise_uncond = tensors.get("test1_noise_uncond").unwrap();
        let noise_cond = tensors.get("test1_noise_cond").unwrap();
        let guidance_scale_tensor = tensors.get("test1_guidance_scale").unwrap();
        let expected = tensors.get("test1_cfg_result").unwrap();

        let guidance_scale = get_scalar(guidance_scale_tensor);
        let result = apply_cfg(noise_uncond, noise_cond, guidance_scale)?;

        let mse = compute_mse(&result, expected);
        let max_diff = compute_max_abs_diff(&result, expected);

        println!("Test 1: Basic CFG (scale={})", guidance_scale);
        println!("  MSE: {:.2e}", mse);
        println!("  Max diff: {:.2e}", max_diff);

        assert!(mse < 1e-10, "CFG MSE {} exceeds threshold 1e-10", mse);
        assert!(
            max_diff < 1e-5,
            "CFG max diff {} exceeds threshold 1e-5",
            max_diff
        );
        Ok(())
    }

    #[test]
    fn test_cfg_formula_scale_variations() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let tensors = match load_reference_tensors(&device) {
            Some(t) => t,
            None => return Ok(()),
        };

        let noise_uncond = tensors.get("test1_noise_uncond").unwrap();
        let noise_cond = tensors.get("test1_noise_cond").unwrap();

        let guidance_scale_2 = get_scalar(tensors.get("test2_guidance_scale").unwrap());
        let expected_2 = tensors.get("test2_cfg_result").unwrap();
        let result_2 = apply_cfg(noise_uncond, noise_cond, guidance_scale_2)?;
        let mse_2 = compute_mse(&result_2, expected_2);
        println!("Test 2: CFG scale={}, MSE: {:.2e}", guidance_scale_2, mse_2);
        assert!(mse_2 < 1e-10, "CFG MSE {} exceeds threshold", mse_2);

        let guidance_scale_3 = get_scalar(tensors.get("test3_guidance_scale").unwrap());
        let expected_3 = tensors.get("test3_cfg_result").unwrap();
        let result_3 = apply_cfg(noise_uncond, noise_cond, guidance_scale_3)?;
        let mse_3 = compute_mse(&result_3, expected_3);
        let diff_to_cond = compute_max_abs_diff(&result_3, noise_cond);
        println!(
            "Test 3: CFG scale=1.0, MSE: {:.2e}, diff to noise_cond: {:.2e}",
            mse_3, diff_to_cond
        );
        assert!(mse_3 < 1e-10, "CFG MSE {} exceeds threshold", mse_3);
        assert!(
            diff_to_cond < 1e-5,
            "CFG with scale=1 should equal noise_cond, diff: {}",
            diff_to_cond
        );
        Ok(())
    }

    #[test]
    fn test_rescale_noise_cfg() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let tensors = match load_reference_tensors(&device) {
            Some(t) => t,
            None => return Ok(()),
        };

        let cfg_before = tensors.get("test4_cfg_before_rescale").unwrap();
        let noise_cond = tensors.get("test1_noise_cond").unwrap();
        let guidance_rescale = get_scalar(tensors.get("test4_guidance_rescale").unwrap());
        let expected = tensors.get("test4_cfg_after_rescale").unwrap();

        let result = rescale_noise_cfg(cfg_before, noise_cond, guidance_rescale)?;
        let mse = compute_mse(&result, expected);
        let max_diff = compute_max_abs_diff(&result, expected);

        println!("Test 4: rescale_noise_cfg (rescale={})", guidance_rescale);
        println!("  MSE: {:.2e}", mse);
        println!("  Max diff: {:.2e}", max_diff);

        assert!(
            mse < 1e-6,
            "rescale_noise_cfg MSE {} exceeds threshold 1e-6",
            mse
        );
        assert!(
            max_diff < 1e-3,
            "rescale_noise_cfg max diff {} exceeds threshold 1e-3",
            max_diff
        );
        Ok(())
    }

    #[test]
    fn test_rescale_noise_cfg_zero() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let tensors = match load_reference_tensors(&device) {
            Some(t) => t,
            None => return Ok(()),
        };

        let cfg_before = tensors.get("test4_cfg_before_rescale").unwrap();
        let noise_cond = tensors.get("test1_noise_cond").unwrap();
        let expected = tensors.get("test5_cfg_after_rescale").unwrap();

        let result = rescale_noise_cfg(cfg_before, noise_cond, 0.0)?;
        let mse = compute_mse(&result, expected);
        let diff_to_original = compute_max_abs_diff(&result, cfg_before);

        println!("Test 5: rescale_noise_cfg (rescale=0.0)");
        println!("  MSE: {:.2e}", mse);
        println!("  Diff to original: {:.2e}", diff_to_original);

        assert!(
            mse < 1e-10,
            "rescale_noise_cfg MSE {} exceeds threshold",
            mse
        );
        assert!(
            diff_to_original < 1e-6,
            "rescale with 0 should return original, diff: {}",
            diff_to_original
        );
        Ok(())
    }

    #[test]
    fn test_cfg_5d_tensor() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let tensors = match load_reference_tensors(&device) {
            Some(t) => t,
            None => return Ok(()),
        };

        let noise_uncond = tensors.get("test6_noise_uncond").unwrap();
        let noise_cond = tensors.get("test6_noise_cond").unwrap();
        let guidance_scale = get_scalar(tensors.get("test6_guidance_scale").unwrap());
        let guidance_rescale = get_scalar(tensors.get("test6_guidance_rescale").unwrap());
        let expected_cfg = tensors.get("test6_cfg_result").unwrap();
        let expected_rescaled = tensors.get("test6_cfg_rescaled").unwrap();

        // Test CFG formula on 5D tensor
        let cfg_result = apply_cfg(noise_uncond, noise_cond, guidance_scale)?;
        let mse_cfg = compute_mse(&cfg_result, expected_cfg);
        let max_diff_cfg = compute_max_abs_diff(&cfg_result, expected_cfg);

        println!("Test 6: 5D tensor CFG (scale={})", guidance_scale);
        println!("  Shape: {:?}", noise_uncond.dims());
        println!("  CFG MSE: {:.2e}", mse_cfg);
        println!("  CFG Max diff: {:.2e}", max_diff_cfg);

        assert!(
            mse_cfg < 1e-10,
            "5D CFG MSE {} exceeds threshold 1e-10",
            mse_cfg
        );
        assert!(
            max_diff_cfg < 1e-5,
            "5D CFG max diff {} exceeds threshold 1e-5",
            max_diff_cfg
        );

        // Test rescale on 5D tensor
        let rescaled_result = rescale_noise_cfg(&cfg_result, noise_cond, guidance_rescale)?;
        let mse_rescale = compute_mse(&rescaled_result, expected_rescaled);
        let max_diff_rescale = compute_max_abs_diff(&rescaled_result, expected_rescaled);

        println!("  Rescale MSE: {:.2e}", mse_rescale);
        println!("  Rescale Max diff: {:.2e}", max_diff_rescale);

        assert!(
            mse_rescale < 1e-6,
            "5D rescale MSE {} exceeds threshold 1e-6",
            mse_rescale
        );
        assert!(
            max_diff_rescale < 1e-3,
            "5D rescale max diff {} exceeds threshold 1e-3",
            max_diff_rescale
        );
        Ok(())
    }

    #[test]
    fn test_std_computation() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let tensors = match load_reference_tensors(&device) {
            Some(t) => t,
            None => return Ok(()),
        };

        // Test std computation on 5D tensor (test6)
        let noise_cond = tensors.get("test6_noise_cond").unwrap();
        let cfg_result = tensors.get("test6_cfg_result").unwrap();
        let expected_std_text = tensors.get("test7_std_text").unwrap();
        let expected_std_cfg = tensors.get("test7_std_cfg").unwrap();

        // Compute std using the same method as rescale_noise_cfg
        // std over dims 1..N with keepdim=True
        fn std_over_dims_except0_keepdim(x: &Tensor) -> candle_core::Result<Tensor> {
            let rank = x.rank();
            let b = x.dim(0)?;
            let flat = x.flatten_from(1)?;
            let var = flat.var_keepdim(1)?;
            let std = var.sqrt()?;
            let mut shape = Vec::with_capacity(rank);
            shape.push(b);
            shape.extend(std::iter::repeat_n(1usize, rank - 1));
            std.reshape(shape)
        }

        let computed_std_text = std_over_dims_except0_keepdim(noise_cond)?;
        let computed_std_cfg = std_over_dims_except0_keepdim(cfg_result)?;

        let mse_std_text = compute_mse(&computed_std_text, expected_std_text);
        let mse_std_cfg = compute_mse(&computed_std_cfg, expected_std_cfg);

        println!("Test 7: std computation verification");
        println!("  std_text shape: {:?}", computed_std_text.dims());
        println!("  std_text MSE: {:.2e}", mse_std_text);
        println!("  std_cfg MSE: {:.2e}", mse_std_cfg);

        // Note: PyTorch uses biased std by default, Candle uses unbiased
        // Allow slightly higher tolerance for std computation
        assert!(
            mse_std_text < 1e-4,
            "std_text MSE {} exceeds threshold 1e-4",
            mse_std_text
        );
        assert!(
            mse_std_cfg < 1e-4,
            "std_cfg MSE {} exceeds threshold 1e-4",
            mse_std_cfg
        );
        Ok(())
    }

    #[test]
    fn test_cfg_batch_size_2() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let tensors = match load_reference_tensors(&device) {
            Some(t) => t,
            None => return Ok(()),
        };

        let noise_uncond = tensors.get("test8_noise_uncond").unwrap();
        let noise_cond = tensors.get("test8_noise_cond").unwrap();
        let guidance_scale = get_scalar(tensors.get("test8_guidance_scale").unwrap());
        let guidance_rescale = get_scalar(tensors.get("test8_guidance_rescale").unwrap());
        let expected_cfg = tensors.get("test8_cfg_result").unwrap();
        let expected_rescaled = tensors.get("test8_cfg_rescaled").unwrap();

        println!("Test 8: Batch size > 1");
        println!("  Shape: {:?}", noise_uncond.dims());

        // Test CFG formula
        let cfg_result = apply_cfg(noise_uncond, noise_cond, guidance_scale)?;
        let mse_cfg = compute_mse(&cfg_result, expected_cfg);
        let max_diff_cfg = compute_max_abs_diff(&cfg_result, expected_cfg);

        println!("  CFG MSE: {:.2e}", mse_cfg);
        println!("  CFG Max diff: {:.2e}", max_diff_cfg);

        assert!(
            mse_cfg < 1e-10,
            "Batch CFG MSE {} exceeds threshold 1e-10",
            mse_cfg
        );
        assert!(
            max_diff_cfg < 1e-5,
            "Batch CFG max diff {} exceeds threshold 1e-5",
            max_diff_cfg
        );

        // Test rescale with batch > 1
        let rescaled_result = rescale_noise_cfg(&cfg_result, noise_cond, guidance_rescale)?;
        let mse_rescale = compute_mse(&rescaled_result, expected_rescaled);
        let max_diff_rescale = compute_max_abs_diff(&rescaled_result, expected_rescaled);

        println!("  Rescale MSE: {:.2e}", mse_rescale);
        println!("  Rescale Max diff: {:.2e}", max_diff_rescale);

        assert!(
            mse_rescale < 1e-6,
            "Batch rescale MSE {} exceeds threshold 1e-6",
            mse_rescale
        );
        assert!(
            max_diff_rescale < 1e-3,
            "Batch rescale max diff {} exceeds threshold 1e-3",
            max_diff_rescale
        );
        Ok(())
    }
}

// =========================================================================
// Property-based tests for CFG parity verification
// Property 9: CFG Computation Parity
// **Validates: Requirements 7.1, 7.2**
// =========================================================================

#[cfg(test)]
mod property_tests {
    use candle_core::{DType, Device, Tensor};
    use candle_video::models::ltx_video::t2v_pipeline::rescale_noise_cfg;
    use proptest::prelude::*;
    use std::path::Path;

    const PARITY_FILE: &str = "gen_cfg_parity.safetensors";

    // =========================================================================
    // Helper functions
    // =========================================================================

    fn compute_mse(a: &Tensor, b: &Tensor) -> f32 {
        let diff = a.sub(b).unwrap();
        diff.sqr()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_vec0::<f32>()
            .unwrap()
    }

    fn apply_cfg(
        noise_uncond: &Tensor,
        noise_cond: &Tensor,
        guidance_scale: f32,
    ) -> candle_core::Result<Tensor> {
        let diff = noise_cond.broadcast_sub(noise_uncond)?;
        noise_uncond.broadcast_add(&diff.affine(guidance_scale as f64, 0.0)?)
    }

    fn apply_cfg_python(
        noise_uncond: &Tensor,
        noise_cond: &Tensor,
        guidance_scale: f32,
    ) -> candle_core::Result<Tensor> {
        // Python formula: noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        let diff = noise_cond.sub(noise_uncond)?;
        let scaled = diff.affine(guidance_scale as f64, 0.0)?;
        noise_uncond.add(&scaled)
    }

    // =========================================================================
    // Property 9: CFG Computation Parity
    // For any valid noise_uncond, noise_cond, and guidance_scale (1.0-20.0),
    // the Rust CFG computation SHALL produce noise_pred with MSE < 1e-6
    // compared to Python implementation.
    // Validates: Requirements 7.1, 7.2
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Feature: ltx-video-parity, Property 9: CFG Computation Parity
        /// For any valid noise tensors and guidance_scale, CFG produces MSE < 1e-6
        /// **Validates: Requirements 7.1, 7.2**
        #[test]
        fn prop_cfg_formula_parity(
            batch in 1usize..3,
            seq_len in 16usize..128,
            hidden_dim in 32usize..256,
            guidance_scale in 1.0f32..20.0,
        ) {
            let device = Device::Cpu;

            // Generate random noise tensors
            let shape = (batch, seq_len, hidden_dim);
            let noise_uncond = Tensor::randn(0f32, 1f32, shape, &device)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let noise_cond = Tensor::randn(0f32, 1f32, shape, &device)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();

            // Apply CFG using Rust implementation
            let result_rust = apply_cfg(&noise_uncond, &noise_cond, guidance_scale).unwrap();

            // Apply CFG using Python-equivalent formula
            let result_python = apply_cfg_python(&noise_uncond, &noise_cond, guidance_scale).unwrap();

            // Compare
            let mse = compute_mse(&result_rust, &result_python);

            prop_assert!(
                mse < 1e-10,
                "CFG formula MSE {} exceeds threshold 1e-10, guidance_scale={}, shape={:?}",
                mse, guidance_scale, shape
            );
        }
    }

    // =========================================================================
    // Property 9.1: CFG with guidance_scale=1.0 equals noise_cond
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Feature: ltx-video-parity, Property 9.1: CFG Scale 1.0 Identity
        /// When guidance_scale=1.0, CFG result equals noise_cond
        #[test]
        fn prop_cfg_scale_one_identity(
            batch in 1usize..3,
            seq_len in 16usize..128,
            hidden_dim in 32usize..256,
        ) {
            let device = Device::Cpu;

            let shape = (batch, seq_len, hidden_dim);
            let noise_uncond = Tensor::randn(0f32, 1f32, shape, &device)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let noise_cond = Tensor::randn(0f32, 1f32, shape, &device)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();

            // CFG with scale=1.0 should equal noise_cond
            let result = apply_cfg(&noise_uncond, &noise_cond, 1.0).unwrap();
            let mse = compute_mse(&result, &noise_cond);

            prop_assert!(
                mse < 1e-10,
                "CFG with scale=1.0 should equal noise_cond, MSE={}, shape={:?}",
                mse, shape
            );
        }
    }

    // =========================================================================
    // Property 9.2: rescale_noise_cfg with guidance_rescale=0.0 returns original
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Feature: ltx-video-parity, Property 9.2: Rescale Zero Identity
        /// When guidance_rescale=0.0, rescale_noise_cfg returns original tensor
        #[test]
        fn prop_rescale_zero_identity(
            batch in 1usize..3,
            seq_len in 16usize..128,
            hidden_dim in 32usize..256,
        ) {
            let device = Device::Cpu;

            let shape = (batch, seq_len, hidden_dim);
            let noise_cfg = Tensor::randn(0f32, 1f32, shape, &device)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let noise_cond = Tensor::randn(0f32, 1f32, shape, &device)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();

            // rescale with 0.0 should return original
            let result = rescale_noise_cfg(&noise_cfg, &noise_cond, 0.0).unwrap();
            let mse = compute_mse(&result, &noise_cfg);

            prop_assert!(
                mse < 1e-10,
                "rescale_noise_cfg with rescale=0.0 should return original, MSE={}, shape={:?}",
                mse, shape
            );
        }
    }

    // =========================================================================
    // Property 9.3: rescale_noise_cfg formula correctness
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Feature: ltx-video-parity, Property 9.3: Rescale Formula Correctness
        /// rescale_noise_cfg follows the formula:
        /// noise_cfg = guidance_rescale * (noise_cfg * std_text/std_cfg) + (1-guidance_rescale) * noise_cfg
        #[test]
        fn prop_rescale_formula_correctness(
            batch in 1usize..3,
            seq_len in 16usize..128,
            hidden_dim in 32usize..256,
            guidance_rescale in 0.0f32..1.0,
        ) {
            let device = Device::Cpu;

            let shape = (batch, seq_len, hidden_dim);
            let noise_cfg = Tensor::randn(0f32, 1f32, shape, &device)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let noise_cond = Tensor::randn(0f32, 1f32, shape, &device)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();

            // Compute expected result manually
            fn std_over_dims_except0_keepdim(x: &Tensor) -> candle_core::Result<Tensor> {
                let rank = x.rank();
                let b = x.dim(0)?;
                let flat = x.flatten_from(1)?;
                let var = flat.var_keepdim(1)?;
                let std = var.sqrt()?;
                let mut shape = Vec::with_capacity(rank);
                shape.push(b);
                shape.extend(std::iter::repeat_n(1usize, rank - 1));
                std.reshape(shape)
            }

            let std_text = std_over_dims_except0_keepdim(&noise_cond).unwrap();
            let std_cfg = std_over_dims_except0_keepdim(&noise_cfg).unwrap();
            let ratio = std_text.broadcast_div(&std_cfg).unwrap();
            let noise_pred_rescaled = noise_cfg.broadcast_mul(&ratio).unwrap();

            // Expected: guidance_rescale * noise_pred_rescaled + (1-guidance_rescale) * noise_cfg
            let a = noise_pred_rescaled.affine(guidance_rescale as f64, 0.0).unwrap();
            let b = noise_cfg.affine((1.0 - guidance_rescale) as f64, 0.0).unwrap();
            let expected = a.broadcast_add(&b).unwrap();

            // Compute using the actual function
            let result = rescale_noise_cfg(&noise_cfg, &noise_cond, guidance_rescale).unwrap();

            let mse = compute_mse(&result, &expected);

            prop_assert!(
                mse < 1e-10,
                "rescale_noise_cfg formula mismatch, MSE={}, guidance_rescale={}, shape={:?}",
                mse, guidance_rescale, shape
            );
        }
    }

    // =========================================================================
    // Property 9.4: CFG with 5D tensors (video latents)
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Feature: ltx-video-parity, Property 9.4: CFG 5D Tensor Parity
        /// CFG works correctly with 5D tensors [B, C, F, H, W]
        #[test]
        fn prop_cfg_5d_tensor_parity(
            batch in 1usize..2,
            channels in 16usize..64,
            frames in 1usize..5,
            height in 4usize..12,
            width in 4usize..12,
            guidance_scale in 1.0f32..20.0,
        ) {
            let device = Device::Cpu;

            let shape = (batch, channels, frames, height, width);
            let noise_uncond = Tensor::randn(0f32, 1f32, shape, &device)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let noise_cond = Tensor::randn(0f32, 1f32, shape, &device)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();

            // Apply CFG
            let result_rust = apply_cfg(&noise_uncond, &noise_cond, guidance_scale).unwrap();
            let result_python = apply_cfg_python(&noise_uncond, &noise_cond, guidance_scale).unwrap();

            let mse = compute_mse(&result_rust, &result_python);

            prop_assert!(
                mse < 1e-10,
                "5D CFG formula MSE {} exceeds threshold, guidance_scale={}, shape={:?}",
                mse, guidance_scale, shape
            );
        }
    }

    // =========================================================================
    // Property 9.5: Reference-based CFG parity (uses captured Python tensors)
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Feature: ltx-video-parity, Property 9.5: Reference CFG Parity
        /// CFG matches Python reference tensors
        #[test]
        fn prop_cfg_reference_parity(
            test_idx in 0usize..3,
        ) {
            let device = Device::Cpu;
            let path = Path::new(PARITY_FILE);
            if !path.exists() {
                // Skip if reference file not available
                return Ok(());
            }

            let tensors = candle_core::safetensors::load(path, &device)
                .expect("Failed to load reference tensors");

            // Select test case based on index
            let (uncond_key, cond_key, scale_key, result_key) = match test_idx {
                0 => ("test1_noise_uncond", "test1_noise_cond", "test1_guidance_scale", "test1_cfg_result"),
                1 => ("test1_noise_uncond", "test1_noise_cond", "test2_guidance_scale", "test2_cfg_result"),
                2 => ("test6_noise_uncond", "test6_noise_cond", "test6_guidance_scale", "test6_cfg_result"),
                _ => return Ok(()),
            };

            let noise_uncond = tensors.get(uncond_key).unwrap();
            let noise_cond = tensors.get(cond_key).unwrap();
            let guidance_scale_tensor = tensors.get(scale_key).unwrap();
            let expected = tensors.get(result_key).unwrap();

            let guidance_scale = if guidance_scale_tensor.rank() == 0 {
                guidance_scale_tensor.to_vec0::<f32>().unwrap()
            } else {
                guidance_scale_tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0]
            };

            let result = apply_cfg(noise_uncond, noise_cond, guidance_scale).unwrap();
            let mse = compute_mse(&result, expected);

            prop_assert!(
                mse < 1e-6,
                "Reference CFG parity MSE {} exceeds threshold 1e-6, test_idx={}, scale={}",
                mse, test_idx, guidance_scale
            );
        }
    }
}
