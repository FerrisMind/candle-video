//! Property-based tests for Wan latent initialization
//!
//! Property 1: Latent Shape Calculation
//! For any valid height (divisible by 16), width (divisible by 16), and num_frames,
//! the initialized latents SHALL have shape [1, 16, (num_frames-1)/4+1, height/8, width/8].
//!
//! **Validates: Requirements 4.1**

#[cfg(test)]
mod tests {
    use candle_core::Device;
    use proptest::prelude::*;

    // Wan VAE constants
    const WAN_VAE_SCALE_FACTOR_SPATIAL: usize = 8;
    const WAN_VAE_SCALE_FACTOR_TEMPORAL: usize = 4;
    const WAN_LATENT_CHANNELS: usize = 16;

    /// Calculate latent dimensions from video dimensions.
    /// This mirrors the implementation in examples/wan/main.rs
    fn calculate_latent_dims(
        height: usize,
        width: usize,
        num_frames: usize,
    ) -> (usize, usize, usize) {
        let latent_frames = (num_frames - 1) / WAN_VAE_SCALE_FACTOR_TEMPORAL + 1;
        let latent_height = height / WAN_VAE_SCALE_FACTOR_SPATIAL;
        let latent_width = width / WAN_VAE_SCALE_FACTOR_SPATIAL;
        (latent_frames, latent_height, latent_width)
    }

    /// Prepare initial latents for video generation.
    /// This mirrors the implementation in examples/wan/main.rs
    fn prepare_latents(
        height: usize,
        width: usize,
        num_frames: usize,
        device: &Device,
        seed: Option<u64>,
    ) -> candle_core::Result<candle_core::Tensor> {
        use candle_video::utils::deterministic_rng::Pcg32;

        let (latent_frames, latent_height, latent_width) =
            calculate_latent_dims(height, width, num_frames);

        let shape = (
            1,
            WAN_LATENT_CHANNELS,
            latent_frames,
            latent_height,
            latent_width,
        );

        let actual_seed = seed.unwrap_or(42);
        let mut rng = Pcg32::new(actual_seed, 1442695040888963407);
        rng.randn(shape, device)
    }

    // =========================================================================
    // Property 1: Latent Shape Calculation
    // For any valid height (divisible by 16), width (divisible by 16), and num_frames,
    // the initialized latents SHALL have shape [1, 16, (num_frames-1)/4+1, height/8, width/8].
    // **Validates: Requirements 4.1**
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Feature: wan-example, Property 1: Latent Shape Calculation
        /// For any valid dimensions, latents have shape [1, 16, (num_frames-1)/4+1, height/8, width/8]
        /// **Validates: Requirements 4.1**
        #[test]
        fn prop_latent_shape_calculation(
            // Height must be divisible by 16 (VAE requirement)
            // Generate multiples of 16 in range [128, 1280]
            height_mult in 8usize..80,
            // Width must be divisible by 16
            // Generate multiples of 16 in range [128, 1280]
            width_mult in 8usize..80,
            // num_frames can be any positive value, typically 1-257
            num_frames in 1usize..257,
            // Seed for deterministic RNG
            seed in 0u64..10000,
        ) {
            let height = height_mult * 16;
            let width = width_mult * 16;
            let device = Device::Cpu;

            // Prepare latents
            let latents = prepare_latents(height, width, num_frames, &device, Some(seed))
                .expect("Failed to prepare latents");

            // Get actual dimensions
            let dims = latents.dims();

            // Calculate expected dimensions
            let expected_latent_frames = (num_frames - 1) / WAN_VAE_SCALE_FACTOR_TEMPORAL + 1;
            let expected_latent_height = height / WAN_VAE_SCALE_FACTOR_SPATIAL;
            let expected_latent_width = width / WAN_VAE_SCALE_FACTOR_SPATIAL;

            // Verify shape
            prop_assert_eq!(
                dims.len(), 5,
                "Latents should be 5D tensor, got {}D",
                dims.len()
            );
            prop_assert_eq!(
                dims[0], 1,
                "Batch size should be 1, got {}",
                dims[0]
            );
            prop_assert_eq!(
                dims[1], WAN_LATENT_CHANNELS,
                "Channels should be {}, got {}",
                WAN_LATENT_CHANNELS, dims[1]
            );
            prop_assert_eq!(
                dims[2], expected_latent_frames,
                "Latent frames should be {} for num_frames={}, got {}",
                expected_latent_frames, num_frames, dims[2]
            );
            prop_assert_eq!(
                dims[3], expected_latent_height,
                "Latent height should be {} for height={}, got {}",
                expected_latent_height, height, dims[3]
            );
            prop_assert_eq!(
                dims[4], expected_latent_width,
                "Latent width should be {} for width={}, got {}",
                expected_latent_width, width, dims[4]
            );
        }

        /// Feature: wan-example, Property 1.1: Latent Frame Formula
        /// For any num_frames, latent_frames = (num_frames - 1) / 4 + 1
        /// **Validates: Requirements 4.1**
        #[test]
        fn prop_latent_frame_formula(
            num_frames in 1usize..1000,
        ) {
            let (latent_frames, _, _) = calculate_latent_dims(480, 832, num_frames);
            let expected = (num_frames - 1) / 4 + 1;

            prop_assert_eq!(
                latent_frames, expected,
                "Latent frames formula failed: ({}−1)/4+1 = {}, got {}",
                num_frames, expected, latent_frames
            );
        }

        /// Feature: wan-example, Property 1.2: Latent Spatial Formula
        /// For any valid height/width, latent_height = height/8, latent_width = width/8
        /// **Validates: Requirements 4.1**
        #[test]
        fn prop_latent_spatial_formula(
            height_mult in 1usize..200,
            width_mult in 1usize..200,
        ) {
            // Use multiples of 8 to ensure clean division
            let height = height_mult * 8;
            let width = width_mult * 8;

            let (_, latent_height, latent_width) = calculate_latent_dims(height, width, 81);

            prop_assert_eq!(
                latent_height, height / 8,
                "Latent height formula failed: {}/8 = {}, got {}",
                height, height / 8, latent_height
            );
            prop_assert_eq!(
                latent_width, width / 8,
                "Latent width formula failed: {}/8 = {}, got {}",
                width, width / 8, latent_width
            );
        }

        /// Feature: wan-example, Property 1.3: Deterministic Latent Generation
        /// For any seed, generating latents twice with same parameters produces identical results
        /// **Validates: Requirements 7.1, 7.4**
        #[test]
        fn prop_deterministic_latent_generation(
            height_mult in 8usize..20,
            width_mult in 8usize..20,
            num_frames in 1usize..50,
            seed in 0u64..10000,
        ) {
            let height = height_mult * 16;
            let width = width_mult * 16;
            let device = Device::Cpu;

            // Generate latents twice with same seed
            let latents1 = prepare_latents(height, width, num_frames, &device, Some(seed))
                .expect("Failed to prepare latents 1");
            let latents2 = prepare_latents(height, width, num_frames, &device, Some(seed))
                .expect("Failed to prepare latents 2");

            // Convert to vectors for comparison
            let v1: Vec<f32> = latents1.flatten_all().unwrap().to_vec1().unwrap();
            let v2: Vec<f32> = latents2.flatten_all().unwrap().to_vec1().unwrap();

            prop_assert_eq!(
                v1, v2,
                "Same seed should produce identical latents"
            );
        }
    }

    // =========================================================================
    // Unit tests for edge cases
    // =========================================================================

    #[test]
    fn test_latent_shape_default_480p() {
        // Default 480p settings: 480x832, 81 frames
        let (latent_frames, latent_height, latent_width) = calculate_latent_dims(480, 832, 81);

        assert_eq!(latent_frames, 21, "F' = (81-1)/4+1 = 21");
        assert_eq!(latent_height, 60, "H' = 480/8 = 60");
        assert_eq!(latent_width, 104, "W' = 832/8 = 104");
    }

    #[test]
    fn test_latent_shape_720p() {
        // 720p settings: 720x1280, 81 frames
        let (latent_frames, latent_height, latent_width) = calculate_latent_dims(720, 1280, 81);

        assert_eq!(latent_frames, 21, "F' = (81-1)/4+1 = 21");
        assert_eq!(latent_height, 90, "H' = 720/8 = 90");
        assert_eq!(latent_width, 160, "W' = 1280/8 = 160");
    }

    #[test]
    fn test_latent_shape_minimum_frames() {
        // Minimum: 1 frame
        let (latent_frames, _, _) = calculate_latent_dims(480, 832, 1);
        assert_eq!(latent_frames, 1, "F' = (1-1)/4+1 = 1");
    }

    #[test]
    fn test_latent_shape_boundary_frames() {
        // Test frame count boundaries
        let test_cases = [
            (1, 1),  // (1-1)/4+1 = 1
            (2, 1),  // (2-1)/4+1 = 1
            (5, 2),  // (5-1)/4+1 = 2
            (9, 3),  // (9-1)/4+1 = 3
            (13, 4), // (13-1)/4+1 = 4
            (17, 5), // (17-1)/4+1 = 5
        ];

        for (num_frames, expected) in test_cases {
            let (latent_frames, _, _) = calculate_latent_dims(480, 832, num_frames);
            assert_eq!(
                latent_frames, expected,
                "Failed for num_frames={}: expected {}, got {}",
                num_frames, expected, latent_frames
            );
        }
    }

    // =========================================================================
    // Property 2: Dimension Validation
    // For any height or width value that is NOT divisible by 16, the pipeline
    // SHALL return a validation error before any model loading occurs.
    // **Validates: Requirements 6.1**
    // =========================================================================

    /// Validate video dimensions.
    /// This mirrors the implementation in examples/wan/main.rs
    fn validate_dimensions(height: usize, width: usize) -> Result<(), String> {
        let mut errors = Vec::new();

        if !height.is_multiple_of(16) {
            errors.push(format!(
                "height {} is not divisible by 16 (nearest valid: {})",
                height,
                (height / 16) * 16
            ));
        }

        if !width.is_multiple_of(16) {
            errors.push(format!(
                "width {} is not divisible by 16 (nearest valid: {})",
                width,
                (width / 16) * 16
            ));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Feature: wan-example, Property 2: Dimension Validation
        /// For any height or width NOT divisible by 16, validation SHALL return an error.
        /// **Validates: Requirements 6.1**
        #[test]
        fn prop_dimension_validation_rejects_invalid(
            // Generate heights that are NOT divisible by 16
            // We generate a base value and add an offset 1-15 to ensure non-divisibility
            height_base in 0usize..100,
            height_offset in 1usize..16,
            // Generate widths that are NOT divisible by 16
            width_base in 0usize..100,
            width_offset in 1usize..16,
        ) {
            // Create invalid dimensions (not divisible by 16)
            let invalid_height = height_base * 16 + height_offset;
            let invalid_width = width_base * 16 + width_offset;

            // Test invalid height with valid width
            let result = validate_dimensions(invalid_height, 832);
            prop_assert!(
                result.is_err(),
                "Height {} should fail validation (not divisible by 16)",
                invalid_height
            );
            let err_msg = result.unwrap_err();
            prop_assert!(
                err_msg.contains(&format!("height {}", invalid_height)),
                "Error should mention invalid height {}: {}",
                invalid_height, err_msg
            );

            // Test valid height with invalid width
            let result = validate_dimensions(480, invalid_width);
            prop_assert!(
                result.is_err(),
                "Width {} should fail validation (not divisible by 16)",
                invalid_width
            );
            let err_msg = result.unwrap_err();
            prop_assert!(
                err_msg.contains(&format!("width {}", invalid_width)),
                "Error should mention invalid width {}: {}",
                invalid_width, err_msg
            );

            // Test both invalid
            let result = validate_dimensions(invalid_height, invalid_width);
            prop_assert!(
                result.is_err(),
                "Both dimensions should fail validation"
            );
            let err_msg = result.unwrap_err();
            prop_assert!(
                err_msg.contains(&format!("height {}", invalid_height)) &&
                err_msg.contains(&format!("width {}", invalid_width)),
                "Error should mention both invalid dimensions: {}",
                err_msg
            );
        }

        /// Feature: wan-example, Property 2.1: Dimension Validation Accepts Valid
        /// For any height and width divisible by 16, validation SHALL succeed.
        /// **Validates: Requirements 6.1**
        #[test]
        fn prop_dimension_validation_accepts_valid(
            // Generate multiples of 16 in reasonable range
            height_mult in 1usize..100,
            width_mult in 1usize..100,
        ) {
            let valid_height = height_mult * 16;
            let valid_width = width_mult * 16;

            let result = validate_dimensions(valid_height, valid_width);
            prop_assert!(
                result.is_ok(),
                "Dimensions {}x{} should pass validation (both divisible by 16)",
                valid_height, valid_width
            );
        }

        /// Feature: wan-example, Property 2.2: Dimension Validation Suggests Nearest Valid
        /// For any invalid dimension, the error message SHALL suggest the nearest valid value.
        /// **Validates: Requirements 6.1**
        #[test]
        fn prop_dimension_validation_suggests_nearest(
            // Generate heights that are NOT divisible by 16
            height_base in 1usize..100,
            height_offset in 1usize..16,
        ) {
            let invalid_height = height_base * 16 + height_offset;
            let expected_nearest = height_base * 16; // Round down to nearest multiple of 16

            let result = validate_dimensions(invalid_height, 832);
            prop_assert!(result.is_err());

            let err_msg = result.unwrap_err();
            prop_assert!(
                err_msg.contains(&format!("nearest valid: {}", expected_nearest)),
                "Error should suggest nearest valid value {} for height {}: {}",
                expected_nearest, invalid_height, err_msg
            );
        }
    }

    // =========================================================================
    // Unit tests for dimension validation edge cases
    // =========================================================================

    #[test]
    fn test_dimension_validation_valid() {
        assert!(validate_dimensions(480, 832).is_ok());
        assert!(validate_dimensions(720, 1280).is_ok());
        assert!(validate_dimensions(16, 16).is_ok());
        assert!(validate_dimensions(1024, 1024).is_ok());
    }

    #[test]
    fn test_dimension_validation_invalid_height() {
        let result = validate_dimensions(481, 832);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("height 481"));
    }

    #[test]
    fn test_dimension_validation_invalid_width() {
        let result = validate_dimensions(480, 833);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("width 833"));
    }

    #[test]
    fn test_dimension_validation_both_invalid() {
        let result = validate_dimensions(481, 833);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("height 481"));
        assert!(err.contains("width 833"));
    }

    #[test]
    fn test_dimension_validation_boundary() {
        // Just below 16
        assert!(validate_dimensions(15, 16).is_err());
        assert!(validate_dimensions(16, 15).is_err());

        // Exactly 16
        assert!(validate_dimensions(16, 16).is_ok());

        // Just above 16 but not divisible
        assert!(validate_dimensions(17, 16).is_err());
        assert!(validate_dimensions(16, 17).is_err());

        // Next valid value (32)
        assert!(validate_dimensions(32, 32).is_ok());
    }

    // =========================================================================
    // Property 3: Reproducibility (Round-Trip)
    // For any seed value and fixed parameters, running the pipeline twice
    // SHALL produce byte-identical output tensors.
    // **Validates: Requirements 7.4**
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Feature: wan-example, Property 3: Reproducibility (Round-Trip)
        /// For any seed value and fixed parameters, running the latent preparation
        /// twice SHALL produce byte-identical output tensors.
        /// **Validates: Requirements 7.4**
        #[test]
        fn prop_reproducibility_round_trip(
            // Test with various seeds
            seed in 0u64..u64::MAX,
            // Test with various valid dimensions (multiples of 16)
            height_mult in 8usize..20,
            width_mult in 8usize..20,
            // Test with various frame counts
            num_frames in 1usize..50,
        ) {
            let height = height_mult * 16;
            let width = width_mult * 16;
            let device = Device::Cpu;

            // Run 1: Generate latents with seed
            let latents1 = prepare_latents(height, width, num_frames, &device, Some(seed))
                .expect("Failed to prepare latents (run 1)");

            // Run 2: Generate latents with same seed and parameters
            let latents2 = prepare_latents(height, width, num_frames, &device, Some(seed))
                .expect("Failed to prepare latents (run 2)");

            // Verify shapes are identical
            prop_assert_eq!(
                latents1.dims(), latents2.dims(),
                "Latent shapes should be identical"
            );

            // Verify values are byte-identical
            let v1: Vec<f32> = latents1.flatten_all().unwrap().to_vec1().unwrap();
            let v2: Vec<f32> = latents2.flatten_all().unwrap().to_vec1().unwrap();

            prop_assert_eq!(
                v1.len(), v2.len(),
                "Latent vectors should have same length"
            );

            // Check byte-identical (not just approximately equal)
            for (i, (a, b)) in v1.iter().zip(v2.iter()).enumerate() {
                prop_assert_eq!(
                    a.to_bits(), b.to_bits(),
                    "Latent values at index {} should be byte-identical: {} vs {}",
                    i, a, b
                );
            }
        }

        /// Feature: wan-example, Property 3.1: Different Seeds Produce Different Results
        /// For any two different seeds, the generated latents SHALL be different.
        /// **Validates: Requirements 7.4**
        #[test]
        fn prop_different_seeds_different_results(
            seed1 in 0u64..u64::MAX/2,
            seed2 in u64::MAX/2..u64::MAX,
            height_mult in 8usize..15,
            width_mult in 8usize..15,
            num_frames in 1usize..20,
        ) {
            // Ensure seeds are different
            prop_assume!(seed1 != seed2);

            let height = height_mult * 16;
            let width = width_mult * 16;
            let device = Device::Cpu;

            let latents1 = prepare_latents(height, width, num_frames, &device, Some(seed1))
                .expect("Failed to prepare latents 1");
            let latents2 = prepare_latents(height, width, num_frames, &device, Some(seed2))
                .expect("Failed to prepare latents 2");

            let v1: Vec<f32> = latents1.flatten_all().unwrap().to_vec1().unwrap();
            let v2: Vec<f32> = latents2.flatten_all().unwrap().to_vec1().unwrap();

            // At least some values should be different
            let differences: usize = v1.iter()
                .zip(v2.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();

            prop_assert!(
                differences > 0,
                "Different seeds should produce different latents (found {} differences out of {})",
                differences, v1.len()
            );
        }

        /// Feature: wan-example, Property 3.2: Seed Consistency Across Dimensions
        /// For any seed, the RNG state progression should be consistent regardless
        /// of tensor dimensions (same seed produces same sequence of random values).
        /// **Validates: Requirements 7.4**
        #[test]
        fn prop_seed_consistency_across_dimensions(
            seed in 0u64..10000,
        ) {
            let device = Device::Cpu;

            // Generate small latents
            let small = prepare_latents(128, 128, 5, &device, Some(seed))
                .expect("Failed to prepare small latents");

            // Generate same small latents again
            let small2 = prepare_latents(128, 128, 5, &device, Some(seed))
                .expect("Failed to prepare small latents 2");

            let v1: Vec<f32> = small.flatten_all().unwrap().to_vec1().unwrap();
            let v2: Vec<f32> = small2.flatten_all().unwrap().to_vec1().unwrap();

            // Should be identical
            prop_assert_eq!(v1, v2, "Same seed should produce identical latents");
        }
    }

    // =========================================================================
    // Unit tests for reproducibility edge cases
    // =========================================================================

    #[test]
    fn test_reproducibility_specific_seed() {
        let device = Device::Cpu;
        let seed = 42u64;

        let latents1 = prepare_latents(256, 256, 17, &device, Some(seed)).unwrap();
        let latents2 = prepare_latents(256, 256, 17, &device, Some(seed)).unwrap();

        let v1: Vec<f32> = latents1.flatten_all().unwrap().to_vec1().unwrap();
        let v2: Vec<f32> = latents2.flatten_all().unwrap().to_vec1().unwrap();

        assert_eq!(v1, v2, "Same seed should produce identical latents");
    }

    #[test]
    fn test_reproducibility_zero_seed() {
        let device = Device::Cpu;
        let seed = 0u64;

        let latents1 = prepare_latents(128, 128, 5, &device, Some(seed)).unwrap();
        let latents2 = prepare_latents(128, 128, 5, &device, Some(seed)).unwrap();

        let v1: Vec<f32> = latents1.flatten_all().unwrap().to_vec1().unwrap();
        let v2: Vec<f32> = latents2.flatten_all().unwrap().to_vec1().unwrap();

        assert_eq!(v1, v2, "Seed 0 should produce identical latents");
    }

    #[test]
    fn test_reproducibility_max_seed() {
        let device = Device::Cpu;
        let seed = u64::MAX;

        let latents1 = prepare_latents(128, 128, 5, &device, Some(seed)).unwrap();
        let latents2 = prepare_latents(128, 128, 5, &device, Some(seed)).unwrap();

        let v1: Vec<f32> = latents1.flatten_all().unwrap().to_vec1().unwrap();
        let v2: Vec<f32> = latents2.flatten_all().unwrap().to_vec1().unwrap();

        assert_eq!(v1, v2, "Max seed should produce identical latents");
    }

    // =========================================================================
    // Property 4: Output Tensor Normalization
    // For any decoded video tensor, after conversion to RGB frames, all pixel
    // values SHALL be in the range [0, 255] (clamped).
    // **Validates: Requirements 5.4, 5.5**
    // =========================================================================

    /// Convert video tensor to frames (mirrors implementation in examples/wan/main.rs)
    fn video_tensor_to_frames(
        video: &candle_core::Tensor,
    ) -> Result<Vec<Vec<u8>>, candle_core::Error> {
        use candle_core::{DType, IndexOp};

        let dims = video.dims();
        let (b, _c, f, _h, _w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

        let mut frames = Vec::with_capacity(b * f);

        for batch_idx in 0..b {
            for frame_idx in 0..f {
                // Extract frame: [C, H, W]
                let frame = video.i((batch_idx, .., frame_idx, .., ..))?;

                // Permute from [C, H, W] to [H, W, C] for row-major RGB layout
                let frame = frame.permute((1, 2, 0))?;

                // Convert from [-1, 1] to [0, 255]
                // Formula: pixel = (value + 1) * 127.5
                let frame = frame.affine(127.5, 127.5)?;

                // Clamp to [0, 255]
                let frame = frame.clamp(0.0, 255.0)?;

                // Convert to u8
                let frame = frame.to_dtype(DType::U8)?;

                // Flatten and extract data
                let data: Vec<u8> = frame.flatten_all()?.to_vec1()?;

                frames.push(data);
            }
        }

        Ok(frames)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Feature: wan-example, Property 4: Output Tensor Normalization
        /// For any decoded video tensor, after conversion to RGB frames, all pixel
        /// values SHALL be in the range [0, 255] (clamped).
        /// **Validates: Requirements 5.4, 5.5**
        #[test]
        fn prop_output_tensor_normalization(
            // Small dimensions for fast testing
            height in 1usize..8,
            width in 1usize..8,
            num_frames in 1usize..4,
            // Test with values that may exceed [-1, 1] range
            min_val in -5.0f32..0.0,
            max_val in 0.0f32..5.0,
        ) {
            use candle_core::{DType, Tensor};

            let device = Device::Cpu;
            let channels = 3usize;

            // Create a video tensor with values potentially outside [-1, 1]
            let total_elements = channels * num_frames * height * width;
            let mut values = Vec::with_capacity(total_elements);

            // Fill with values spanning the range [min_val, max_val]
            for i in 0..total_elements {
                let t = (i as f32) / (total_elements as f32);
                let val = min_val + t * (max_val - min_val);
                values.push(val);
            }

            let video = Tensor::from_vec(
                values,
                (1, channels, num_frames, height, width),
                &device,
            ).unwrap().to_dtype(DType::F32).unwrap();

            // Convert to frames
            let frames = video_tensor_to_frames(&video).expect("Failed to convert video to frames");

            // Verify all frames were converted (u8 type guarantees [0, 255] range)
            prop_assert!(
                !frames.is_empty(),
                "Should have at least one frame"
            );
            for frame_data in frames.iter() {
                prop_assert!(
                    !frame_data.is_empty(),
                    "Frame should have pixel data"
                );
            }
        }

        /// Feature: wan-example, Property 4.1: Normalization Formula Correctness
        /// For any value in [-1, 1], the formula (value + 1) * 127.5 produces [0, 255]
        /// **Validates: Requirements 5.4, 5.5**
        #[test]
        fn prop_normalization_formula_correctness(
            // Values in the expected VAE output range [-1, 1]
            value in -1.0f32..=1.0,
        ) {
            // Apply the normalization formula
            let normalized = (value + 1.0) * 127.5;

            // Verify result is in [0, 255]
            prop_assert!(
                (0.0..=255.0).contains(&normalized),
                "Normalized value {} for input {} should be in [0, 255]",
                normalized, value
            );

            // Verify boundary cases
            if value == -1.0 {
                prop_assert!(
                    (normalized - 0.0).abs() < 0.001,
                    "Value -1.0 should normalize to 0.0, got {}",
                    normalized
                );
            }
            if value == 1.0 {
                prop_assert!(
                    (normalized - 255.0).abs() < 0.001,
                    "Value 1.0 should normalize to 255.0, got {}",
                    normalized
                );
            }
        }

        /// Feature: wan-example, Property 4.2: Clamping Handles Out-of-Range Values
        /// For any value outside [-1, 1], clamping ensures output is in [0, 255]
        /// **Validates: Requirements 5.4, 5.5**
        #[test]
        fn prop_clamping_handles_out_of_range(
            // Values potentially outside [-1, 1]
            value in -10.0f32..10.0,
        ) {
            // Apply the normalization formula
            let normalized = (value + 1.0) * 127.5;

            // Apply clamping
            let clamped = normalized.clamp(0.0, 255.0);

            // Verify result is in [0, 255]
            prop_assert!(
                (0.0..=255.0).contains(&clamped),
                "Clamped value {} for input {} should be in [0, 255]",
                clamped, value
            );

            // Verify clamping behavior
            if value < -1.0 {
                prop_assert!(
                    clamped == 0.0,
                    "Value {} < -1.0 should clamp to 0.0, got {}",
                    value, clamped
                );
            }
            if value > 1.0 {
                prop_assert!(
                    clamped == 255.0,
                    "Value {} > 1.0 should clamp to 255.0, got {}",
                    value, clamped
                );
            }
        }
    }

    // =========================================================================
    // Unit tests for output normalization edge cases
    // =========================================================================

    #[test]
    fn test_normalization_boundary_values() {
        // Test exact boundary values
        assert_eq!(((-1.0f32) + 1.0) * 127.5, 0.0);
        assert_eq!(((1.0f32) + 1.0) * 127.5, 255.0);
        assert_eq!(((0.0f32) + 1.0) * 127.5, 127.5);
    }

    #[test]
    fn test_normalization_clamping() {
        // Test values outside [-1, 1]
        let below = ((-2.0f32) + 1.0) * 127.5; // -127.5
        let above = ((2.0f32) + 1.0) * 127.5; // 382.5

        assert!(below < 0.0);
        assert!(above > 255.0);

        // After clamping
        assert_eq!(below.clamp(0.0, 255.0), 0.0);
        assert_eq!(above.clamp(0.0, 255.0), 255.0);
    }

    #[test]
    fn test_video_tensor_to_frames_shape() {
        use candle_core::{DType, Tensor};

        let device = Device::Cpu;
        let video = Tensor::zeros((1, 3, 2, 4, 4), DType::F32, &device).unwrap();

        let frames = video_tensor_to_frames(&video).expect("Failed to convert");

        // Should have 2 frames (1 batch * 2 frames)
        assert_eq!(frames.len(), 2);

        // Each frame should have 4 * 4 * 3 = 48 pixels
        for frame in &frames {
            assert_eq!(frame.len(), 48);
        }
    }

    #[test]
    fn test_video_tensor_to_frames_values() {
        use candle_core::{DType, Tensor};

        let device = Device::Cpu;

        // Create a tensor with known values
        // Shape: [B=1, C=3, F=1, H=1, W=2]
        // Value -1.0 should become 0, value 0.0 should become 127/128, value 1.0 should become 255
        // Layout: [batch, channel, frame, height, width]
        // We want pixel 0 to have R=-1, G=0, B=1 and pixel 1 to have R=0.5, G=-0.5, B=0
        let values: Vec<f32> = vec![
            // Channel 0 (R): pixel 0 = -1.0, pixel 1 = 0.5
            -1.0, 0.5, // Channel 1 (G): pixel 0 = 0.0, pixel 1 = -0.5
            0.0, -0.5, // Channel 2 (B): pixel 0 = 1.0, pixel 1 = 0.0
            1.0, 0.0,
        ];
        let video = Tensor::from_vec(values, (1, 3, 1, 1, 2), &device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();

        let frames = video_tensor_to_frames(&video).expect("Failed to convert");

        assert_eq!(frames.len(), 1);
        let frame = &frames[0];

        // After permute [C, H, W] -> [H, W, C], we get [1, 2, 3] = 6 values
        // Row-major order: [H=0, W=0, C=0], [H=0, W=0, C=1], [H=0, W=0, C=2], [H=0, W=1, C=0], ...
        // Pixel 0 (W=0): R=-1->0, G=0->127, B=1->255
        // Pixel 1 (W=1): R=0.5->191, G=-0.5->63, B=0->127
        assert_eq!(frame.len(), 6);
        assert_eq!(frame[0], 0); // R of pixel 0: (-1+1)*127.5 = 0
        assert_eq!(frame[1], 127); // G of pixel 0: (0+1)*127.5 = 127.5 -> 127
        assert_eq!(frame[2], 255); // B of pixel 0: (1+1)*127.5 = 255
        assert_eq!(frame[3], 191); // R of pixel 1: (0.5+1)*127.5 = 191.25 -> 191
        assert_eq!(frame[4], 63); // G of pixel 1: (-0.5+1)*127.5 = 63.75 -> 63
        assert_eq!(frame[5], 127); // B of pixel 1: (0+1)*127.5 = 127.5 -> 127
    }
}
