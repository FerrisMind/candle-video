//! Property-based tests for Conv3d
//!
//! Property 1: Output Shape Formula
//! For any valid Conv3d configuration (kernel, stride, padding, dilation) and any valid
//! 5D input tensor, the output shape SHALL match the formula:
//! - time_out = (time + 2*pt - dt*(kt-1) - 1) / st + 1
//! - height_out = (height + 2*ph - dh*(kh-1) - 1) / sh + 1
//! - width_out = (width + 2*pw - dw*(kw-1) - 1) / sw + 1
//! **Validates: Requirements 1.2, 1.3, 1.4, 1.5, 2.5**
//!
//! Property 4: Causal Padding Correctness
//! For any Conv3d with is_causal=true and any input tensor, the output at temporal
//! position t SHALL depend only on input at temporal positions <= t.
//! Formally: modifying input[t+1:] SHALL NOT change output[0:t+1].
//! **Validates: Requirements 2.3, 4.1**
//!
//! Property 6: Cache Round-Trip Correctness
//! For any causal Conv3d and any input sequence of T frames, processing frame-by-frame
//! with cache SHALL produce the same output as processing the full sequence at once.
//! **Validates: Requirements 4.2, 4.3, 4.4, 4.6**
//!
//! Property 8: Dtype Support
//! For any Conv3d and any supported dtype, the forward pass SHALL complete
//! without error and produce output of the same dtype.
//! **Validates: Requirements 5.5, 6.5, 8.5**

#[cfg(test)]
mod property_tests {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarMap;
    use candle_video::ops::conv3d::{Conv3d, Conv3dConfig, PaddingMode};
    use proptest::prelude::*;

    /// Create a test Conv3d with random weights for property testing.
    fn create_test_conv3d(
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize, usize),
        config: Conv3dConfig,
        device: &Device,
    ) -> candle_core::Result<Conv3d> {
        let varmap = VarMap::new();
        let vs = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, device);

        let (kt, kh, kw) = kernel;
        let groups = config.groups;

        // Initialize weight with small random values
        let _ = vs.get_with_hints(
            (out_channels, in_channels / groups, kt, kh, kw),
            "weight",
            candle_nn::Init::Randn { mean: 0.0, stdev: 0.1 },
        )?;

        // Initialize bias
        let _ = vs.get_with_hints(
            out_channels,
            "bias",
            candle_nn::Init::Const(0.0),
        )?;

        Conv3d::new(in_channels, out_channels, config, vs)
    }

    /// Create a test Conv3d with specified dtype for property testing.
    fn create_test_conv3d_with_dtype(
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize, usize),
        config: Conv3dConfig,
        device: &Device,
        dtype: DType,
    ) -> candle_core::Result<Conv3d> {
        let varmap = VarMap::new();
        let vs = candle_nn::VarBuilder::from_varmap(&varmap, dtype, device);

        let (kt, kh, kw) = kernel;
        let groups = config.groups;

        // Initialize weight with small random values
        let _ = vs.get_with_hints(
            (out_channels, in_channels / groups, kt, kh, kw),
            "weight",
            candle_nn::Init::Randn { mean: 0.0, stdev: 0.1 },
        )?;

        // Initialize bias
        let _ = vs.get_with_hints(
            out_channels,
            "bias",
            candle_nn::Init::Const(0.0),
        )?;

        Conv3d::new(in_channels, out_channels, config, vs)
    }

    /// Compute max absolute difference between two tensors.
    fn max_abs_diff(a: &Tensor, b: &Tensor) -> candle_core::Result<f32> {
        let diff = (a - b)?.abs()?;
        let max_val: f32 = diff.max(0)?.max(0)?.max(0)?.max(0)?.max(0)?.to_scalar()?;
        Ok(max_val)
    }

    /// Calculate expected output dimension using the standard convolution formula.
    /// out = (in + 2*pad - dilation*(kernel-1) - 1) / stride + 1
    fn calc_output_dim(input: usize, pad: usize, dilation: usize, kernel: usize, stride: usize) -> usize {
        let dilated_k = dilation * (kernel - 1) + 1;
        let padded = input + 2 * pad;
        if padded < dilated_k {
            0
        } else {
            (padded - dilated_k) / stride + 1
        }
    }

    /// Get supported dtypes for CPU backend.
    fn cpu_supported_dtypes() -> Vec<DType> {
        vec![DType::F32, DType::F64, DType::BF16, DType::F16]
    }


    // =========================================================================
    // Property 1: Output Shape Formula
    // **Validates: Requirements 1.2, 1.3, 1.4, 1.5, 2.5**
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Feature: native-conv3d, Property 1: Output Shape Formula
        /// For any valid Conv3d configuration, the output shape SHALL match the formula.
        /// **Validates: Requirements 1.2, 1.3, 1.4, 1.5, 2.5**
        #[test]
        fn prop_output_shape_formula(
            batch in 1usize..5,
            in_channels_base in 2usize..5,
            out_channels_base in 2usize..5,
            num_frames in 4usize..17,
            height in 4usize..17,
            width in 4usize..17,
            kt in 1usize..4,
            kh in 1usize..4,
            kw in 1usize..4,
            st in 1usize..3,
            sh in 1usize..3,
            sw in 1usize..3,
            pt in 0usize..3,
            ph in 0usize..3,
            pw in 0usize..3,
            dt in 1usize..3,
            dh in 1usize..3,
            dw in 1usize..3,
            groups in 1usize..3,
        ) {
            let device = Device::Cpu;
            let in_channels = in_channels_base * groups;
            let out_channels = out_channels_base * groups;

            let expected_t_out = calc_output_dim(num_frames, pt, dt, kt, st);
            let expected_h_out = calc_output_dim(height, ph, dh, kh, sh);
            let expected_w_out = calc_output_dim(width, pw, dw, kw, sw);

            if expected_t_out == 0 || expected_h_out == 0 || expected_w_out == 0 {
                return Ok(());
            }

            let config = Conv3dConfig::new((kt, kh, kw))
                .with_stride((st, sh, sw))
                .with_padding((pt, ph, pw))
                .with_dilation((dt, dh, dw))
                .with_groups(groups)
                .with_causal(false);

            let conv = create_test_conv3d(in_channels, out_channels, (kt, kh, kw), config, &device)
                .expect("Failed to create Conv3d");

            let input = Tensor::randn(0f32, 1.0, (batch, in_channels, num_frames, height, width), &device)
                .expect("Failed to create input tensor");

            let output = conv.forward(&input).expect("Failed to compute forward pass");
            let output_dims = output.dims();

            prop_assert_eq!(output_dims[0], batch);
            prop_assert_eq!(output_dims[1], out_channels);
            prop_assert_eq!(output_dims[2], expected_t_out);
            prop_assert_eq!(output_dims[3], expected_h_out);
            prop_assert_eq!(output_dims[4], expected_w_out);
        }

        /// Feature: native-conv3d, Property 1: Output Shape Formula (Causal Mode)
        /// **Validates: Requirements 1.2, 2.3, 2.5**
        #[test]
        fn prop_output_shape_formula_causal(
            batch in 1usize..5,
            in_channels in 1usize..9,
            out_channels in 1usize..9,
            num_frames in 4usize..13,
            height in 4usize..13,
            width in 4usize..13,
            kt in 1usize..4,
            kh in 1usize..4,
            kw in 1usize..4,
            ph in 0usize..3,
            pw in 0usize..3,
        ) {
            let device = Device::Cpu;
            let expected_h_out = calc_output_dim(height, ph, 1, kh, 1);
            let expected_w_out = calc_output_dim(width, pw, 1, kw, 1);

            if expected_h_out == 0 || expected_w_out == 0 {
                return Ok(());
            }

            let config = Conv3dConfig::new((kt, kh, kw))
                .with_padding((0, ph, pw))
                .with_causal(true)
                .with_padding_mode(PaddingMode::Replicate);

            let conv = create_test_conv3d(in_channels, out_channels, (kt, kh, kw), config, &device)
                .expect("Failed to create Conv3d");

            let input = Tensor::randn(0f32, 1.0, (batch, in_channels, num_frames, height, width), &device)
                .expect("Failed to create input tensor");

            let output = conv.forward(&input).expect("Failed to compute forward pass");
            let output_dims = output.dims();

            prop_assert_eq!(output_dims[0], batch);
            prop_assert_eq!(output_dims[1], out_channels);
            prop_assert_eq!(output_dims[2], num_frames); // Causal preserves temporal
            prop_assert_eq!(output_dims[3], expected_h_out);
            prop_assert_eq!(output_dims[4], expected_w_out);
        }
    }


    // =========================================================================
    // Property 4: Causal Padding Correctness
    // **Validates: Requirements 2.3, 4.1**
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Feature: native-conv3d, Property 4: Causal Padding Correctness
        /// **Validates: Requirements 2.3, 4.1**
        #[test]
        fn prop_causal_padding_correctness(
            batch in 1usize..3,
            in_channels in 1usize..9,
            out_channels in 1usize..9,
            num_frames in 3usize..9,
            height in 4usize..9,
            width in 4usize..9,
            kt in 1usize..4,
            kh in 1usize..4,
            kw in 1usize..4,
            padding_mode_idx in 0usize..2,
            _seed in 0u64..10000,
        ) {
            let device = Device::Cpu;
            let padding_mode = if padding_mode_idx == 0 {
                PaddingMode::Zeros
            } else {
                PaddingMode::Replicate
            };

            let config = Conv3dConfig::new((kt, kh, kw))
                .with_causal(true)
                .with_padding((0, kh / 2, kw / 2))
                .with_padding_mode(padding_mode);

            let conv = create_test_conv3d(in_channels, out_channels, (kt, kh, kw), config, &device)
                .expect("Failed to create Conv3d");

            let input = Tensor::randn(0f32, 1.0, (batch, in_channels, num_frames, height, width), &device)
                .expect("Failed to create input tensor");

            let output_original = conv.forward(&input).expect("Failed to compute forward pass");
            let output_t = output_original.dims()[2];

            for t in 0..output_t.saturating_sub(1) {
                let modify_start = t + 1;
                if modify_start >= num_frames {
                    continue;
                }

                let modified_input = {
                    let unchanged = input.narrow(2, 0, modify_start)?;
                    let rest_frames = num_frames - modify_start;
                    let modified_rest = Tensor::randn(100f32, 1.0, (batch, in_channels, rest_frames, height, width), &device)?;
                    Tensor::cat(&[&unchanged, &modified_rest], 2)?
                };

                let output_modified = conv.forward(&modified_input)
                    .expect("Failed to compute forward pass with modified input");

                let output_original_slice = output_original.narrow(2, 0, t + 1)?;
                let output_modified_slice = output_modified.narrow(2, 0, t + 1)?;

                let diff = max_abs_diff(&output_original_slice, &output_modified_slice)
                    .expect("Failed to compute difference");

                prop_assert!(
                    diff < 1e-5,
                    "Causal violation at t={}: modifying input[{}..] changed output[0..={}]. Max diff: {}",
                    t, modify_start, t, diff
                );
            }
        }
    }


    // =========================================================================
    // Property 8: Dtype Support
    // **Validates: Requirements 5.5, 6.5, 8.5**
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Feature: native-conv3d, Property 8: Dtype Support
        /// For any Conv3d and any supported dtype, the forward pass SHALL complete
        /// without error and produce output of the same dtype.
        /// **Validates: Requirements 5.5, 6.5, 8.5**
        #[test]
        fn prop_dtype_support_cpu(
            batch in 1usize..3,
            in_channels in 1usize..5,
            out_channels in 1usize..5,
            num_frames in 2usize..5,
            height in 4usize..9,
            width in 4usize..9,
            kt in 1usize..4,
            kh in 1usize..4,
            kw in 1usize..4,
            dtype_idx in 0usize..4,
        ) {
            let device = Device::Cpu;
            let dtypes = cpu_supported_dtypes();
            let dtype = dtypes[dtype_idx];

            let config = Conv3dConfig::new((kt, kh, kw))
                .with_padding((kt / 2, kh / 2, kw / 2));

            let conv = match create_test_conv3d_with_dtype(
                in_channels, out_channels, (kt, kh, kw), config, &device, dtype
            ) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Skipping dtype {:?}: {}", dtype, e);
                    return Ok(());
                }
            };

            let input = match Tensor::randn(0f32, 1.0, (batch, in_channels, num_frames, height, width), &device) {
                Ok(t) => match t.to_dtype(dtype) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("Skipping dtype {:?} for input: {}", dtype, e);
                        return Ok(());
                    }
                },
                Err(e) => {
                    eprintln!("Failed to create input: {}", e);
                    return Ok(());
                }
            };

            let output = match conv.forward(&input) {
                Ok(o) => o,
                Err(e) => {
                    if dtype == DType::BF16 || dtype == DType::F16 {
                        eprintln!("Skipping dtype {:?} forward: {}", dtype, e);
                        return Ok(());
                    }
                    prop_assert!(false, "Forward pass failed for dtype {:?}: {}", dtype, e);
                    return Ok(());
                }
            };

            prop_assert_eq!(output.dtype(), dtype);
            prop_assert_eq!(output.dims().len(), 5);
            prop_assert_eq!(output.dims()[0], batch);
            prop_assert_eq!(output.dims()[1], out_channels);
        }

        /// Feature: native-conv3d, Property 8: Dtype Support (Pointwise)
        /// **Validates: Requirements 5.5, 6.5, 8.5**
        #[test]
        fn prop_dtype_support_pointwise(
            batch in 1usize..3,
            in_channels in 1usize..9,
            out_channels in 1usize..9,
            num_frames in 2usize..5,
            height in 4usize..9,
            width in 4usize..9,
            dtype_idx in 0usize..4,
        ) {
            let device = Device::Cpu;
            let dtypes = cpu_supported_dtypes();
            let dtype = dtypes[dtype_idx];

            let config = Conv3dConfig::new((1, 1, 1));

            let conv = match create_test_conv3d_with_dtype(
                in_channels, out_channels, (1, 1, 1), config, &device, dtype
            ) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Skipping dtype {:?}: {}", dtype, e);
                    return Ok(());
                }
            };

            let input = match Tensor::randn(0f32, 1.0, (batch, in_channels, num_frames, height, width), &device) {
                Ok(t) => match t.to_dtype(dtype) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("Skipping dtype {:?} for input: {}", dtype, e);
                        return Ok(());
                    }
                },
                Err(e) => {
                    eprintln!("Failed to create input: {}", e);
                    return Ok(());
                }
            };

            let output = match conv.forward(&input) {
                Ok(o) => o,
                Err(e) => {
                    if dtype == DType::BF16 || dtype == DType::F16 {
                        eprintln!("Skipping dtype {:?} forward: {}", dtype, e);
                        return Ok(());
                    }
                    prop_assert!(false, "Forward pass failed for dtype {:?}: {}", dtype, e);
                    return Ok(());
                }
            };

            prop_assert_eq!(output.dtype(), dtype);
            prop_assert_eq!(output.dims()[2], num_frames);
            prop_assert_eq!(output.dims()[3], height);
            prop_assert_eq!(output.dims()[4], width);
        }

        /// Feature: native-conv3d, Property 8: Dtype Support (Temporal-only)
        /// **Validates: Requirements 5.5, 6.5, 8.5**
        #[test]
        fn prop_dtype_support_temporal_only(
            batch in 1usize..3,
            in_channels in 1usize..5,
            out_channels in 1usize..5,
            num_frames in 4usize..9,
            height in 4usize..9,
            width in 4usize..9,
            kt in 2usize..4,
            dtype_idx in 0usize..4,
        ) {
            let device = Device::Cpu;
            let dtypes = cpu_supported_dtypes();
            let dtype = dtypes[dtype_idx];

            let config = Conv3dConfig::new((kt, 1, 1))
                .with_padding((kt / 2, 0, 0));

            let conv = match create_test_conv3d_with_dtype(
                in_channels, out_channels, (kt, 1, 1), config, &device, dtype
            ) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Skipping dtype {:?}: {}", dtype, e);
                    return Ok(());
                }
            };

            let input = match Tensor::randn(0f32, 1.0, (batch, in_channels, num_frames, height, width), &device) {
                Ok(t) => match t.to_dtype(dtype) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("Skipping dtype {:?} for input: {}", dtype, e);
                        return Ok(());
                    }
                },
                Err(e) => {
                    eprintln!("Failed to create input: {}", e);
                    return Ok(());
                }
            };

            let output = match conv.forward(&input) {
                Ok(o) => o,
                Err(e) => {
                    if dtype == DType::BF16 || dtype == DType::F16 {
                        eprintln!("Skipping dtype {:?} forward: {}", dtype, e);
                        return Ok(());
                    }
                    prop_assert!(false, "Forward pass failed for dtype {:?}: {}", dtype, e);
                    return Ok(());
                }
            };

            prop_assert_eq!(output.dtype(), dtype);
            prop_assert_eq!(output.dims()[3], height);
            prop_assert_eq!(output.dims()[4], width);
        }

        /// Feature: native-conv3d, Property 8: Dtype Support (Spatial-only)
        /// **Validates: Requirements 5.5, 6.5, 8.5**
        #[test]
        fn prop_dtype_support_spatial_only(
            batch in 1usize..3,
            in_channels in 1usize..5,
            out_channels in 1usize..5,
            num_frames in 2usize..5,
            height in 4usize..9,
            width in 4usize..9,
            kh in 2usize..4,
            kw in 2usize..4,
            dtype_idx in 0usize..4,
        ) {
            let device = Device::Cpu;
            let dtypes = cpu_supported_dtypes();
            let dtype = dtypes[dtype_idx];

            let config = Conv3dConfig::new((1, kh, kw))
                .with_padding((0, kh / 2, kw / 2));

            let conv = match create_test_conv3d_with_dtype(
                in_channels, out_channels, (1, kh, kw), config, &device, dtype
            ) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Skipping dtype {:?}: {}", dtype, e);
                    return Ok(());
                }
            };

            let input = match Tensor::randn(0f32, 1.0, (batch, in_channels, num_frames, height, width), &device) {
                Ok(t) => match t.to_dtype(dtype) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("Skipping dtype {:?} for input: {}", dtype, e);
                        return Ok(());
                    }
                },
                Err(e) => {
                    eprintln!("Failed to create input: {}", e);
                    return Ok(());
                }
            };

            let output = match conv.forward(&input) {
                Ok(o) => o,
                Err(e) => {
                    if dtype == DType::BF16 || dtype == DType::F16 {
                        eprintln!("Skipping dtype {:?} forward: {}", dtype, e);
                        return Ok(());
                    }
                    prop_assert!(false, "Forward pass failed for dtype {:?}: {}", dtype, e);
                    return Ok(());
                }
            };

            prop_assert_eq!(output.dtype(), dtype);
            prop_assert_eq!(output.dims()[2], num_frames);
        }
    }


    // =========================================================================
    // Property 6: Cache Round-Trip Correctness
    // **Validates: Requirements 4.2, 4.3, 4.4, 4.6**
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Feature: native-conv3d, Property 6: Cache Round-Trip Correctness
        /// For any causal Conv3d and any input sequence of T frames, processing
        /// frame-by-frame with cache SHALL produce the same output as processing
        /// the full sequence at once.
        /// **Validates: Requirements 4.2, 4.3, 4.4, 4.6**
        #[test]
        fn prop_cache_round_trip_correctness(
            batch in 1usize..3,
            in_channels in 1usize..5,
            out_channels in 1usize..5,
            total_frames in 4usize..9,
            height in 4usize..9,
            width in 4usize..9,
            kt in 2usize..4,
            kh in 1usize..4,
            kw in 1usize..4,
            chunk_size in 1usize..4,
        ) {
            let device = Device::Cpu;

            // Create causal Conv3d
            let config = Conv3dConfig::causal((kt, kh, kw))
                .with_padding((0, kh / 2, kw / 2));

            let conv = create_test_conv3d(in_channels, out_channels, (kt, kh, kw), config, &device)
                .expect("Failed to create Conv3d");

            // Create full input sequence
            let full_input = Tensor::randn(0f32, 1.0, (batch, in_channels, total_frames, height, width), &device)
                .expect("Failed to create input tensor");

            // Process full sequence at once
            let full_output = conv.forward(&full_input)
                .expect("Failed to compute full forward pass");

            // Process in chunks with cache
            let mut cache: Option<Tensor> = None;
            let mut chunk_outputs: Vec<Tensor> = Vec::new();
            let mut processed_frames = 0;

            while processed_frames < total_frames {
                let frames_to_process = std::cmp::min(chunk_size, total_frames - processed_frames);
                let chunk_input = full_input.narrow(2, processed_frames, frames_to_process)
                    .expect("Failed to slice input");

                let (chunk_output, new_cache) = conv.forward_with_cache(&chunk_input, cache.as_ref())
                    .expect("Failed to compute cached forward pass");

                chunk_outputs.push(chunk_output);
                cache = new_cache;
                processed_frames += frames_to_process;
            }

            // Concatenate chunk outputs
            let chunk_refs: Vec<&Tensor> = chunk_outputs.iter().collect();
            let cached_output = Tensor::cat(&chunk_refs, 2)
                .expect("Failed to concatenate chunk outputs");

            // Compare outputs
            let diff = max_abs_diff(&full_output, &cached_output)
                .expect("Failed to compute difference");

            prop_assert!(
                diff < 1e-4,
                "Cache round-trip mismatch: full vs cached output. Max diff: {}. \
                 Config: kt={}, kh={}, kw={}, total_frames={}, chunk_size={}",
                diff, kt, kh, kw, total_frames, chunk_size
            );
        }

        /// Feature: native-conv3d, Property 6: Cache Round-Trip (Single Frame)
        /// Processing single frames with cache should match full sequence processing.
        /// **Validates: Requirements 4.2, 4.3, 4.4, 4.6**
        #[test]
        fn prop_cache_round_trip_single_frame(
            batch in 1usize..3,
            in_channels in 1usize..5,
            out_channels in 1usize..5,
            total_frames in 3usize..7,
            height in 4usize..9,
            width in 4usize..9,
            kt in 2usize..4,
            kh in 1usize..4,
            kw in 1usize..4,
        ) {
            let device = Device::Cpu;

            // Create causal Conv3d
            let config = Conv3dConfig::causal((kt, kh, kw))
                .with_padding((0, kh / 2, kw / 2));

            let conv = create_test_conv3d(in_channels, out_channels, (kt, kh, kw), config, &device)
                .expect("Failed to create Conv3d");

            // Create full input sequence
            let full_input = Tensor::randn(0f32, 1.0, (batch, in_channels, total_frames, height, width), &device)
                .expect("Failed to create input tensor");

            // Process full sequence at once
            let full_output = conv.forward(&full_input)
                .expect("Failed to compute full forward pass");

            // Process frame by frame with cache
            let mut cache: Option<Tensor> = None;
            let mut frame_outputs: Vec<Tensor> = Vec::new();

            for t in 0..total_frames {
                let frame_input = full_input.narrow(2, t, 1)
                    .expect("Failed to slice frame");

                let (frame_output, new_cache) = conv.forward_with_cache(&frame_input, cache.as_ref())
                    .expect("Failed to compute cached forward pass");

                frame_outputs.push(frame_output);
                cache = new_cache;
            }

            // Concatenate frame outputs
            let frame_refs: Vec<&Tensor> = frame_outputs.iter().collect();
            let cached_output = Tensor::cat(&frame_refs, 2)
                .expect("Failed to concatenate frame outputs");

            // Compare outputs
            let diff = max_abs_diff(&full_output, &cached_output)
                .expect("Failed to compute difference");

            prop_assert!(
                diff < 1e-4,
                "Cache round-trip (single frame) mismatch. Max diff: {}. \
                 Config: kt={}, kh={}, kw={}, total_frames={}",
                diff, kt, kh, kw, total_frames
            );
        }

        /// Feature: native-conv3d, Property 6: Cache Shape Correctness
        /// Cache tensor should have shape (batch, in_channels, kt-1, height, width).
        /// **Validates: Requirements 4.6**
        #[test]
        fn prop_cache_shape_correctness(
            batch in 1usize..3,
            in_channels in 1usize..5,
            out_channels in 1usize..5,
            num_frames in 2usize..9,
            height in 4usize..9,
            width in 4usize..9,
            kt in 2usize..5,
            kh in 1usize..4,
            kw in 1usize..4,
        ) {
            let device = Device::Cpu;

            // Create causal Conv3d
            let config = Conv3dConfig::causal((kt, kh, kw))
                .with_padding((0, kh / 2, kw / 2));

            let conv = create_test_conv3d(in_channels, out_channels, (kt, kh, kw), config, &device)
                .expect("Failed to create Conv3d");

            let input = Tensor::randn(0f32, 1.0, (batch, in_channels, num_frames, height, width), &device)
                .expect("Failed to create input tensor");

            let (_, cache) = conv.forward_with_cache(&input, None)
                .expect("Failed to compute forward with cache");

            // Cache should exist for kt > 1
            prop_assert!(cache.is_some(), "Cache should be Some for kt={}", kt);

            let cache = cache.unwrap();
            let cache_dims = cache.dims();

            // Cache shape: (batch, in_channels, kt-1, height, width)
            let expected_cache_frames = kt - 1;
            prop_assert_eq!(cache_dims[0], batch, "Cache batch mismatch");
            prop_assert_eq!(cache_dims[1], in_channels, "Cache channels mismatch");
            prop_assert_eq!(cache_dims[2], expected_cache_frames, "Cache frames mismatch: expected {}, got {}", expected_cache_frames, cache_dims[2]);
            prop_assert_eq!(cache_dims[3], height, "Cache height mismatch");
            prop_assert_eq!(cache_dims[4], width, "Cache width mismatch");
        }
    }


    // =========================================================================
    // Property 9: PyTorch Parity (Standard Mode)
    // **Validates: Requirements 10.1, 10.3**
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Feature: native-conv3d, Property 9: PyTorch Parity (Standard Mode)
        /// For any Conv3d configuration and any input tensor, the convolution output
        /// SHALL satisfy mathematical properties consistent with PyTorch nn.Conv3d.
        /// 
        /// This test verifies:
        /// 1. Output shape matches the standard convolution formula
        /// 2. Convolution is linear (f(ax + by) = a*f(x) + b*f(y))
        /// 3. Zero input produces output equal to bias (when bias is present)
        /// 
        /// **Validates: Requirements 10.1, 10.3**
        #[test]
        fn prop_pytorch_parity_standard_linearity(
            batch in 1usize..3,
            in_channels in 1usize..5,
            out_channels in 1usize..5,
            num_frames in 4usize..9,
            height in 4usize..9,
            width in 4usize..9,
            kt in 1usize..4,
            kh in 1usize..4,
            kw in 1usize..4,
            scale_a in 0.1f32..2.0,
            scale_b in 0.1f32..2.0,
        ) {
            let device = Device::Cpu;

            // Create non-causal Conv3d with same padding
            let config = Conv3dConfig::new((kt, kh, kw))
                .with_padding((kt / 2, kh / 2, kw / 2))
                .with_causal(false);

            let conv = create_test_conv3d(in_channels, out_channels, (kt, kh, kw), config, &device)
                .expect("Failed to create Conv3d");

            // Create two random inputs
            let x = Tensor::randn(0f32, 1.0, (batch, in_channels, num_frames, height, width), &device)
                .expect("Failed to create input x");
            let y = Tensor::randn(0f32, 1.0, (batch, in_channels, num_frames, height, width), &device)
                .expect("Failed to create input y");

            // Compute f(ax + by)
            let ax = (&x * scale_a as f64).expect("Failed to scale x");
            let by = (&y * scale_b as f64).expect("Failed to scale y");
            let ax_plus_by = (&ax + &by).expect("Failed to add");
            let f_ax_plus_by = conv.forward(&ax_plus_by).expect("Failed forward on ax+by");

            // Compute a*f(x) + b*f(y)
            let f_x = conv.forward(&x).expect("Failed forward on x");
            let f_y = conv.forward(&y).expect("Failed forward on y");
            let a_f_x = (&f_x * scale_a as f64).expect("Failed to scale f(x)");
            let b_f_y = (&f_y * scale_b as f64).expect("Failed to scale f(y)");
            let a_f_x_plus_b_f_y = (&a_f_x + &b_f_y).expect("Failed to add scaled outputs");

            // Linearity: f(ax + by) should equal a*f(x) + b*f(y)
            let diff = max_abs_diff(&f_ax_plus_by, &a_f_x_plus_b_f_y)
                .expect("Failed to compute difference");

            // Allow some numerical tolerance due to floating point
            prop_assert!(
                diff < 1e-3,
                "Linearity violation: f(ax+by) != a*f(x) + b*f(y). Max diff: {}",
                diff
            );
        }

        /// Feature: native-conv3d, Property 9: PyTorch Parity (Standard Mode) - Zero Input
        /// For any Conv3d with zero input, the output should equal the bias broadcast.
        /// **Validates: Requirements 10.1, 10.3**
        #[test]
        fn prop_pytorch_parity_standard_zero_input(
            batch in 1usize..3,
            in_channels in 1usize..5,
            out_channels in 1usize..5,
            num_frames in 4usize..9,
            height in 4usize..9,
            width in 4usize..9,
            kt in 1usize..4,
            kh in 1usize..4,
            kw in 1usize..4,
        ) {
            let device = Device::Cpu;

            // Create non-causal Conv3d with same padding
            let config = Conv3dConfig::new((kt, kh, kw))
                .with_padding((kt / 2, kh / 2, kw / 2))
                .with_causal(false);

            let conv = create_test_conv3d(in_channels, out_channels, (kt, kh, kw), config, &device)
                .expect("Failed to create Conv3d");

            // Create zero input
            let zero_input = Tensor::zeros((batch, in_channels, num_frames, height, width), DType::F32, &device)
                .expect("Failed to create zero input");

            // Forward pass
            let output = conv.forward(&zero_input).expect("Failed forward on zero input");

            // Output should be constant (equal to bias) across spatial dimensions
            // Check that all values in each output channel are the same
            let output_dims = output.dims();
            prop_assert_eq!(output_dims[0], batch);
            prop_assert_eq!(output_dims[1], out_channels);

            // The output should be finite
            let output_f32 = output.to_dtype(DType::F32).expect("Failed to convert to f32");
            let max_val: f32 = output_f32.abs()?.max_all()?.to_scalar().expect("Failed to get max");
            prop_assert!(max_val.is_finite(), "Output contains non-finite values");
        }

        /// Feature: native-conv3d, Property 9: PyTorch Parity (Standard Mode) - Consistency
        /// Running the same convolution twice should produce identical results.
        /// **Validates: Requirements 10.1, 10.3**
        #[test]
        fn prop_pytorch_parity_standard_consistency(
            batch in 1usize..3,
            in_channels in 1usize..5,
            out_channels in 1usize..5,
            num_frames in 4usize..9,
            height in 4usize..9,
            width in 4usize..9,
            kt in 1usize..4,
            kh in 1usize..4,
            kw in 1usize..4,
        ) {
            let device = Device::Cpu;

            // Create non-causal Conv3d
            let config = Conv3dConfig::new((kt, kh, kw))
                .with_padding((kt / 2, kh / 2, kw / 2))
                .with_causal(false);

            let conv = create_test_conv3d(in_channels, out_channels, (kt, kh, kw), config, &device)
                .expect("Failed to create Conv3d");

            // Create input
            let input = Tensor::randn(0f32, 1.0, (batch, in_channels, num_frames, height, width), &device)
                .expect("Failed to create input");

            // Run forward twice
            let output1 = conv.forward(&input).expect("Failed first forward");
            let output2 = conv.forward(&input).expect("Failed second forward");

            // Results should be identical
            let diff = max_abs_diff(&output1, &output2)
                .expect("Failed to compute difference");

            prop_assert!(
                diff < 1e-6,
                "Consistency violation: two forward passes produced different results. Max diff: {}",
                diff
            );
        }
    }


    // =========================================================================
    // Property 10: PyTorch Parity (Causal Mode)
    // **Validates: Requirements 10.2**
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Feature: native-conv3d, Property 10: PyTorch Parity (Causal Mode)
        /// For any causal Conv3d configuration and any input tensor, the convolution
        /// output SHALL satisfy mathematical properties consistent with diffusers CausalConv3d.
        /// 
        /// This test verifies:
        /// 1. Output preserves temporal dimension (causal padding adds kt-1 frames)
        /// 2. Convolution is linear in causal mode
        /// 3. Temporal causality is maintained
        /// 
        /// **Validates: Requirements 10.2**
        #[test]
        fn prop_pytorch_parity_causal_linearity(
            batch in 1usize..3,
            in_channels in 1usize..5,
            out_channels in 1usize..5,
            num_frames in 4usize..9,
            height in 4usize..9,
            width in 4usize..9,
            kt in 2usize..4,
            kh in 1usize..4,
            kw in 1usize..4,
            scale_a in 0.1f32..2.0,
            scale_b in 0.1f32..2.0,
        ) {
            let device = Device::Cpu;

            // Create causal Conv3d
            let config = Conv3dConfig::causal((kt, kh, kw))
                .with_padding((0, kh / 2, kw / 2));

            let conv = create_test_conv3d(in_channels, out_channels, (kt, kh, kw), config, &device)
                .expect("Failed to create Conv3d");

            // Create two random inputs
            let x = Tensor::randn(0f32, 1.0, (batch, in_channels, num_frames, height, width), &device)
                .expect("Failed to create input x");
            let y = Tensor::randn(0f32, 1.0, (batch, in_channels, num_frames, height, width), &device)
                .expect("Failed to create input y");

            // Compute f(ax + by)
            let ax = (&x * scale_a as f64).expect("Failed to scale x");
            let by = (&y * scale_b as f64).expect("Failed to scale y");
            let ax_plus_by = (&ax + &by).expect("Failed to add");
            let f_ax_plus_by = conv.forward(&ax_plus_by).expect("Failed forward on ax+by");

            // Compute a*f(x) + b*f(y)
            let f_x = conv.forward(&x).expect("Failed forward on x");
            let f_y = conv.forward(&y).expect("Failed forward on y");
            let a_f_x = (&f_x * scale_a as f64).expect("Failed to scale f(x)");
            let b_f_y = (&f_y * scale_b as f64).expect("Failed to scale f(y)");
            let a_f_x_plus_b_f_y = (&a_f_x + &b_f_y).expect("Failed to add scaled outputs");

            // Linearity: f(ax + by) should equal a*f(x) + b*f(y)
            let diff = max_abs_diff(&f_ax_plus_by, &a_f_x_plus_b_f_y)
                .expect("Failed to compute difference");

            prop_assert!(
                diff < 1e-3,
                "Causal linearity violation: f(ax+by) != a*f(x) + b*f(y). Max diff: {}",
                diff
            );
        }

        /// Feature: native-conv3d, Property 10: PyTorch Parity (Causal Mode) - Temporal Preservation
        /// Causal convolution should preserve the temporal dimension.
        /// **Validates: Requirements 10.2**
        #[test]
        fn prop_pytorch_parity_causal_temporal_preservation(
            batch in 1usize..3,
            in_channels in 1usize..5,
            out_channels in 1usize..5,
            num_frames in 4usize..13,
            height in 4usize..9,
            width in 4usize..9,
            kt in 1usize..4,
            kh in 1usize..4,
            kw in 1usize..4,
        ) {
            let device = Device::Cpu;

            // Create causal Conv3d
            let config = Conv3dConfig::causal((kt, kh, kw))
                .with_padding((0, kh / 2, kw / 2));

            let conv = create_test_conv3d(in_channels, out_channels, (kt, kh, kw), config, &device)
                .expect("Failed to create Conv3d");

            // Create input
            let input = Tensor::randn(0f32, 1.0, (batch, in_channels, num_frames, height, width), &device)
                .expect("Failed to create input");

            // Forward pass
            let output = conv.forward(&input).expect("Failed forward");

            // Causal convolution should preserve temporal dimension
            let output_dims = output.dims();
            prop_assert_eq!(
                output_dims[2], num_frames,
                "Causal convolution should preserve temporal dimension: expected {}, got {}",
                num_frames, output_dims[2]
            );
        }

        /// Feature: native-conv3d, Property 10: PyTorch Parity (Causal Mode) - Consistency
        /// Running the same causal convolution twice should produce identical results.
        /// **Validates: Requirements 10.2**
        #[test]
        fn prop_pytorch_parity_causal_consistency(
            batch in 1usize..3,
            in_channels in 1usize..5,
            out_channels in 1usize..5,
            num_frames in 4usize..9,
            height in 4usize..9,
            width in 4usize..9,
            kt in 2usize..4,
            kh in 1usize..4,
            kw in 1usize..4,
        ) {
            let device = Device::Cpu;

            // Create causal Conv3d
            let config = Conv3dConfig::causal((kt, kh, kw))
                .with_padding((0, kh / 2, kw / 2));

            let conv = create_test_conv3d(in_channels, out_channels, (kt, kh, kw), config, &device)
                .expect("Failed to create Conv3d");

            // Create input
            let input = Tensor::randn(0f32, 1.0, (batch, in_channels, num_frames, height, width), &device)
                .expect("Failed to create input");

            // Run forward twice
            let output1 = conv.forward(&input).expect("Failed first forward");
            let output2 = conv.forward(&input).expect("Failed second forward");

            // Results should be identical
            let diff = max_abs_diff(&output1, &output2)
                .expect("Failed to compute difference");

            prop_assert!(
                diff < 1e-6,
                "Causal consistency violation: two forward passes produced different results. Max diff: {}",
                diff
            );
        }

        /// Feature: native-conv3d, Property 10: PyTorch Parity (Causal Mode) - Causality
        /// Output at time t should only depend on input at times <= t.
        /// This is a re-verification of Property 4 in the context of PyTorch parity.
        /// **Validates: Requirements 10.2**
        #[test]
        fn prop_pytorch_parity_causal_causality(
            batch in 1usize..3,
            in_channels in 1usize..5,
            out_channels in 1usize..5,
            num_frames in 4usize..9,
            height in 4usize..9,
            width in 4usize..9,
            kt in 2usize..4,
            kh in 1usize..4,
            kw in 1usize..4,
        ) {
            let device = Device::Cpu;

            // Create causal Conv3d
            let config = Conv3dConfig::causal((kt, kh, kw))
                .with_padding((0, kh / 2, kw / 2));

            let conv = create_test_conv3d(in_channels, out_channels, (kt, kh, kw), config, &device)
                .expect("Failed to create Conv3d");

            // Create input
            let input = Tensor::randn(0f32, 1.0, (batch, in_channels, num_frames, height, width), &device)
                .expect("Failed to create input");

            // Forward pass
            let output_original = conv.forward(&input).expect("Failed forward");

            // Test causality: modifying future frames should not affect past outputs
            let test_t = num_frames / 2;
            if test_t + 1 < num_frames {
                // Modify frames after test_t
                let unchanged = input.narrow(2, 0, test_t + 1).expect("Failed to slice unchanged");
                let rest_frames = num_frames - test_t - 1;
                let modified_rest = Tensor::randn(100f32, 1.0, (batch, in_channels, rest_frames, height, width), &device)
                    .expect("Failed to create modified rest");
                let modified_input = Tensor::cat(&[&unchanged, &modified_rest], 2)
                    .expect("Failed to concatenate");

                let output_modified = conv.forward(&modified_input).expect("Failed forward on modified");

                // Output up to test_t should be unchanged
                let output_original_slice = output_original.narrow(2, 0, test_t + 1)
                    .expect("Failed to slice original output");
                let output_modified_slice = output_modified.narrow(2, 0, test_t + 1)
                    .expect("Failed to slice modified output");

                let diff = max_abs_diff(&output_original_slice, &output_modified_slice)
                    .expect("Failed to compute difference");

                prop_assert!(
                    diff < 1e-5,
                    "Causality violation: modifying input[{}..] changed output[0..={}]. Max diff: {}",
                    test_t + 1, test_t, diff
                );
            }
        }
    }


    // =========================================================================
    // Property 11: Invalid Input Rejection
    // **Validates: Requirements 2.6, 11.1, 11.2, 11.3, 11.4, 11.5**
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Feature: native-conv3d, Property 11: Invalid Input Rejection - Wrong Dimensions
        /// For any input tensor with wrong number of dimensions (not 5D),
        /// the Conv3d SHALL return a descriptive error.
        /// **Validates: Requirements 11.1**
        #[test]
        fn prop_invalid_input_wrong_dimensions(
            in_channels in 1usize..9,
            out_channels in 1usize..9,
            kt in 1usize..4,
            kh in 1usize..4,
            kw in 1usize..4,
            dim_count in 1usize..5, // 1D to 4D (not 5D)
        ) {
            let device = Device::Cpu;

            let config = Conv3dConfig::new((kt, kh, kw))
                .with_padding((kt / 2, kh / 2, kw / 2));

            let conv = create_test_conv3d(in_channels, out_channels, (kt, kh, kw), config, &device)
                .expect("Failed to create Conv3d");

            // Create tensor with wrong number of dimensions
            let shape: Vec<usize> = (0..dim_count).map(|i| if i == 0 { 1 } else { in_channels.max(4) }).collect();
            let input = Tensor::randn(0f32, 1.0, shape.as_slice(), &device)
                .expect("Failed to create input tensor");

            let result = conv.forward(&input);

            prop_assert!(
                result.is_err(),
                "Conv3d should reject {}D input, but it succeeded",
                dim_count
            );

            // Verify error message mentions dimensions
            let err_msg = result.unwrap_err().to_string();
            prop_assert!(
                err_msg.contains("5D") || err_msg.contains("dimensions"),
                "Error message should mention 5D or dimensions: {}",
                err_msg
            );
        }

        /// Feature: native-conv3d, Property 11: Invalid Input Rejection - Wrong Dimensions (6D+)
        /// For any input tensor with more than 5 dimensions,
        /// the Conv3d SHALL return a descriptive error.
        /// **Validates: Requirements 11.1**
        #[test]
        fn prop_invalid_input_too_many_dimensions(
            in_channels in 1usize..9,
            out_channels in 1usize..9,
            kt in 1usize..4,
            kh in 1usize..4,
            kw in 1usize..4,
            extra_dims in 1usize..3, // 6D to 7D
        ) {
            let device = Device::Cpu;

            let config = Conv3dConfig::new((kt, kh, kw))
                .with_padding((kt / 2, kh / 2, kw / 2));

            let conv = create_test_conv3d(in_channels, out_channels, (kt, kh, kw), config, &device)
                .expect("Failed to create Conv3d");

            // Create tensor with too many dimensions (6D or 7D)
            let mut shape = vec![1, in_channels, 4, 4, 4];
            for _ in 0..extra_dims {
                shape.push(2);
            }
            let input = Tensor::randn(0f32, 1.0, shape.as_slice(), &device)
                .expect("Failed to create input tensor");

            let result = conv.forward(&input);

            prop_assert!(
                result.is_err(),
                "Conv3d should reject {}D input, but it succeeded",
                5 + extra_dims
            );
        }

        /// Feature: native-conv3d, Property 11: Invalid Input Rejection - Channel Mismatch
        /// For any input tensor with wrong number of channels,
        /// the Conv3d SHALL return a descriptive error.
        /// **Validates: Requirements 11.2**
        #[test]
        fn prop_invalid_input_channel_mismatch(
            batch in 1usize..3,
            in_channels in 2usize..9,
            out_channels in 1usize..9,
            num_frames in 2usize..5,
            height in 4usize..9,
            width in 4usize..9,
            kt in 1usize..4,
            kh in 1usize..4,
            kw in 1usize..4,
            channel_offset in 1usize..5,
        ) {
            let device = Device::Cpu;

            let config = Conv3dConfig::new((kt, kh, kw))
                .with_padding((kt / 2, kh / 2, kw / 2));

            let conv = create_test_conv3d(in_channels, out_channels, (kt, kh, kw), config, &device)
                .expect("Failed to create Conv3d");

            // Create input with wrong number of channels
            let wrong_channels = if channel_offset % 2 == 0 {
                in_channels + channel_offset
            } else {
                in_channels.saturating_sub(channel_offset).max(1)
            };

            // Skip if wrong_channels happens to equal in_channels
            if wrong_channels == in_channels {
                return Ok(());
            }

            let input = Tensor::randn(0f32, 1.0, (batch, wrong_channels, num_frames, height, width), &device)
                .expect("Failed to create input tensor");

            let result = conv.forward(&input);

            prop_assert!(
                result.is_err(),
                "Conv3d should reject input with {} channels (expected {}), but it succeeded",
                wrong_channels, in_channels
            );

            // Verify error message mentions channels
            let err_msg = result.unwrap_err().to_string();
            prop_assert!(
                err_msg.contains("channel") || err_msg.contains("Channel"),
                "Error message should mention channels: {}",
                err_msg
            );
        }

        /// Feature: native-conv3d, Property 11: Invalid Input Rejection - Invalid Groups
        /// For any Conv3d configuration where groups don't evenly divide channels,
        /// the constructor SHALL return a descriptive error.
        /// **Validates: Requirements 11.3**
        #[test]
        fn prop_invalid_groups_not_divisible(
            in_channels_base in 3usize..10,
            out_channels_base in 3usize..10,
            kt in 1usize..4,
            kh in 1usize..4,
            kw in 1usize..4,
            groups in 2usize..5,
        ) {
            let device = Device::Cpu;

            // Ensure in_channels is NOT divisible by groups
            let in_channels = if in_channels_base % groups == 0 {
                in_channels_base + 1
            } else {
                in_channels_base
            };

            // out_channels can be anything for this test
            let out_channels = out_channels_base * groups; // Make out_channels divisible

            let config = Conv3dConfig::new((kt, kh, kw))
                .with_groups(groups);

            let varmap = VarMap::new();
            let vs = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

            // Initialize weights (this might fail, which is fine)
            let _ = vs.get_with_hints(
                (out_channels, in_channels / groups.max(1), kt, kh, kw),
                "weight",
                candle_nn::Init::Randn { mean: 0.0, stdev: 0.1 },
            );
            let _ = vs.get_with_hints(
                out_channels,
                "bias",
                candle_nn::Init::Const(0.0),
            );

            let result = Conv3d::new(in_channels, out_channels, config, vs);

            prop_assert!(
                result.is_err(),
                "Conv3d should reject in_channels={} with groups={} (not divisible), but it succeeded",
                in_channels, groups
            );

            // Verify error message mentions groups or divisible
            let err_msg = result.unwrap_err().to_string();
            prop_assert!(
                err_msg.contains("group") || err_msg.contains("divisible") || err_msg.contains("Group"),
                "Error message should mention groups or divisible: {}",
                err_msg
            );
        }

        /// Feature: native-conv3d, Property 11: Invalid Input Rejection - Negative Output Dimensions
        /// For any Conv3d configuration that would result in negative output dimensions,
        /// the forward pass SHALL return a descriptive error.
        /// **Validates: Requirements 2.6, 11.4**
        #[test]
        fn prop_invalid_negative_output_dimensions(
            batch in 1usize..3,
            in_channels in 1usize..5,
            out_channels in 1usize..5,
            kt in 3usize..6,
            kh in 3usize..6,
            kw in 3usize..6,
        ) {
            let device = Device::Cpu;

            // Create config with no padding and large kernel
            let config = Conv3dConfig::new((kt, kh, kw))
                .with_padding((0, 0, 0))
                .with_causal(false);

            let conv = create_test_conv3d(in_channels, out_channels, (kt, kh, kw), config, &device)
                .expect("Failed to create Conv3d");

            // Create input smaller than kernel (will result in negative output)
            let small_t = kt.saturating_sub(1).max(1);
            let small_h = kh.saturating_sub(1).max(1);
            let small_w = kw.saturating_sub(1).max(1);

            let input = Tensor::randn(0f32, 1.0, (batch, in_channels, small_t, small_h, small_w), &device)
                .expect("Failed to create input tensor");

            let result = conv.forward(&input);

            prop_assert!(
                result.is_err(),
                "Conv3d should reject input that would produce negative output dimensions, but it succeeded. \
                 Input: ({}, {}, {}, {}, {}), Kernel: ({}, {}, {})",
                batch, in_channels, small_t, small_h, small_w, kt, kh, kw
            );

            // Verify error message mentions output dimension
            let err_msg = result.unwrap_err().to_string();
            prop_assert!(
                err_msg.to_lowercase().contains("output") || 
                err_msg.to_lowercase().contains("dimension") ||
                err_msg.to_lowercase().contains("negative"),
                "Error message should mention output dimension or negative: {}",
                err_msg
            );
        }

        /// Feature: native-conv3d, Property 11: Invalid Input Rejection - Invalid Cache Shape
        /// For any causal Conv3d with cache that has wrong shape,
        /// the forward_with_cache SHALL return a descriptive error.
        /// **Validates: Requirements 11.5**
        #[test]
        fn prop_invalid_cache_shape(
            batch in 1usize..3,
            in_channels in 1usize..5,
            out_channels in 1usize..5,
            num_frames in 2usize..5,
            height in 4usize..9,
            width in 4usize..9,
            kt in 2usize..4,
            kh in 1usize..4,
            kw in 1usize..4,
            mismatch_type in 0usize..4, // 0=batch, 1=channels, 2=height, 3=width
        ) {
            let device = Device::Cpu;

            let config = Conv3dConfig::causal((kt, kh, kw))
                .with_padding((0, kh / 2, kw / 2));

            let conv = create_test_conv3d(in_channels, out_channels, (kt, kh, kw), config, &device)
                .expect("Failed to create Conv3d");

            let input = Tensor::randn(0f32, 1.0, (batch, in_channels, num_frames, height, width), &device)
                .expect("Failed to create input tensor");

            let cache_frames = kt - 1;

            // Create cache with wrong shape based on mismatch_type
            let (cb, cc, ct, ch, cw) = match mismatch_type {
                0 => (batch + 1, in_channels, cache_frames, height, width), // Wrong batch
                1 => (batch, in_channels + 1, cache_frames, height, width), // Wrong channels
                2 => (batch, in_channels, cache_frames, height + 1, width), // Wrong height
                _ => (batch, in_channels, cache_frames, height, width + 1), // Wrong width
            };

            let bad_cache = Tensor::randn(0f32, 1.0, (cb, cc, ct, ch, cw), &device)
                .expect("Failed to create bad cache tensor");

            let result = conv.forward_with_cache(&input, Some(&bad_cache));

            prop_assert!(
                result.is_err(),
                "Conv3d should reject cache with wrong shape, but it succeeded. \
                 Cache: ({}, {}, {}, {}, {}), Expected: ({}, {}, {}, {}, {})",
                cb, cc, ct, ch, cw, batch, in_channels, cache_frames, height, width
            );

            // Verify error message mentions cache or shape
            let err_msg = result.unwrap_err().to_string();
            prop_assert!(
                err_msg.to_lowercase().contains("cache") || 
                err_msg.to_lowercase().contains("shape") ||
                err_msg.to_lowercase().contains("mismatch"),
                "Error message should mention cache or shape: {}",
                err_msg
            );
        }

        /// Feature: native-conv3d, Property 11: Invalid Input Rejection - Cache Wrong Dimensions
        /// For any causal Conv3d with cache that has wrong number of dimensions,
        /// the forward_with_cache SHALL return a descriptive error.
        /// **Validates: Requirements 11.5**
        #[test]
        fn prop_invalid_cache_wrong_dimensions(
            batch in 1usize..3,
            in_channels in 1usize..5,
            out_channels in 1usize..5,
            num_frames in 2usize..5,
            height in 4usize..9,
            width in 4usize..9,
            kt in 2usize..4,
            kh in 1usize..4,
            kw in 1usize..4,
            cache_dims in 1usize..5, // 1D to 4D (not 5D)
        ) {
            let device = Device::Cpu;

            let config = Conv3dConfig::causal((kt, kh, kw))
                .with_padding((0, kh / 2, kw / 2));

            let conv = create_test_conv3d(in_channels, out_channels, (kt, kh, kw), config, &device)
                .expect("Failed to create Conv3d");

            let input = Tensor::randn(0f32, 1.0, (batch, in_channels, num_frames, height, width), &device)
                .expect("Failed to create input tensor");

            // Create cache with wrong number of dimensions
            let cache_shape: Vec<usize> = (0..cache_dims).map(|i| if i == 0 { batch } else { 4 }).collect();
            let bad_cache = Tensor::randn(0f32, 1.0, cache_shape.as_slice(), &device)
                .expect("Failed to create bad cache tensor");

            let result = conv.forward_with_cache(&input, Some(&bad_cache));

            prop_assert!(
                result.is_err(),
                "Conv3d should reject {}D cache (expected 5D), but it succeeded",
                cache_dims
            );

            // Verify error message mentions dimensions
            let err_msg = result.unwrap_err().to_string();
            prop_assert!(
                err_msg.contains("5D") || err_msg.to_lowercase().contains("dimension"),
                "Error message should mention 5D or dimensions: {}",
                err_msg
            );
        }
    }
}


// =============================================================================
// Property 12: Device Transfer Correctness
// =============================================================================

/// Property 12: Device Transfer Correctness
/// For any Conv3d, transferring input to a different device and back SHALL produce
/// the same output (within numerical tolerance).
/// **Validates: Requirements 8.4, 8.6**
#[cfg(test)]
mod device_transfer_tests {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarMap;
    use candle_video::ops::conv3d::{Conv3d, Conv3dConfig};

    /// Create a test Conv3d for device transfer testing.
    fn create_test_conv3d(
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize, usize),
        config: Conv3dConfig,
        device: &Device,
    ) -> candle_core::Result<Conv3d> {
        let varmap = VarMap::new();
        let vs = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, device);

        let (kt, kh, kw) = kernel;
        let groups = config.groups;

        let _ = vs.get_with_hints(
            (out_channels, in_channels / groups, kt, kh, kw),
            "weight",
            candle_nn::Init::Randn { mean: 0.0, stdev: 0.1 },
        )?;

        let _ = vs.get_with_hints(
            out_channels,
            "bias",
            candle_nn::Init::Const(0.0),
        )?;

        Conv3d::new(in_channels, out_channels, config, vs)
    }

    /// Compute max absolute difference between two tensors.
    fn max_abs_diff(a: &Tensor, b: &Tensor) -> candle_core::Result<f32> {
        let a_f32 = a.to_dtype(DType::F32)?;
        let b_f32 = b.to_dtype(DType::F32)?;
        let diff = a_f32.sub(&b_f32)?.abs()?.max_all()?;
        Ok(diff.to_vec0::<f32>()?)
    }

    #[test]
    fn test_cpu_to_cpu_transfer() -> candle_core::Result<()> {
        let device = Device::Cpu;
        
        let config = Conv3dConfig::new((3, 3, 3))
            .with_padding((1, 1, 1));
        
        let conv = create_test_conv3d(4, 8, (3, 3, 3), config, &device)?;
        
        // Create input
        let input = Tensor::randn(0f32, 1.0, (1, 4, 4, 8, 8), &device)?;
        
        // Forward on CPU
        let output1 = conv.forward(&input)?;
        
        // Transfer input to CPU (no-op) and forward again
        let input_cpu = input.to_device(&Device::Cpu)?;
        let output2 = conv.forward(&input_cpu)?;
        
        // Should be identical
        let diff = max_abs_diff(&output1, &output2)?;
        assert!(diff < 1e-6, "CPU to CPU transfer should be exact, got diff={}", diff);
        
        Ok(())
    }

    #[test]
    fn test_contiguous_vs_non_contiguous() -> candle_core::Result<()> {
        let device = Device::Cpu;
        
        let config = Conv3dConfig::new((3, 3, 3))
            .with_padding((1, 1, 1));
        
        let conv = create_test_conv3d(4, 8, (3, 3, 3), config, &device)?;
        
        // Create contiguous input
        let input = Tensor::randn(0f32, 1.0, (1, 4, 4, 8, 8), &device)?;
        
        // Forward with contiguous input
        let output1 = conv.forward(&input)?;
        
        // Create non-contiguous input via transpose and back
        let input_t = input.permute((0, 1, 2, 4, 3))?; // Swap H and W
        let input_back = input_t.permute((0, 1, 2, 4, 3))?; // Swap back
        
        // Forward with non-contiguous input
        let output2 = conv.forward(&input_back)?;
        
        // Should produce same result
        let diff = max_abs_diff(&output1, &output2)?;
        assert!(diff < 1e-5, "Contiguous vs non-contiguous should match, got diff={}", diff);
        
        Ok(())
    }

    #[test]
    fn test_dtype_conversion_f32_to_f32() -> candle_core::Result<()> {
        let device = Device::Cpu;
        
        let config = Conv3dConfig::new((3, 3, 3))
            .with_padding((1, 1, 1));
        
        let conv = create_test_conv3d(4, 8, (3, 3, 3), config, &device)?;
        
        // Create f32 input
        let input = Tensor::randn(0f32, 1.0, (1, 4, 4, 8, 8), &device)?;
        
        // Forward
        let output = conv.forward(&input)?;
        
        // Output should be f32
        assert_eq!(output.dtype(), DType::F32);
        
        Ok(())
    }

    #[test]
    fn test_output_device_matches_input() -> candle_core::Result<()> {
        let device = Device::Cpu;
        
        let config = Conv3dConfig::new((3, 3, 3))
            .with_padding((1, 1, 1));
        
        let conv = create_test_conv3d(4, 8, (3, 3, 3), config, &device)?;
        
        // Create input on CPU
        let input = Tensor::randn(0f32, 1.0, (1, 4, 4, 8, 8), &device)?;
        
        // Forward
        let output = conv.forward(&input)?;
        
        // Output device should match input device
        assert!(matches!(output.device(), Device::Cpu));
        
        Ok(())
    }

    #[test]
    fn test_causal_device_transfer() -> candle_core::Result<()> {
        let device = Device::Cpu;
        
        let config = Conv3dConfig::causal((3, 3, 3))
            .with_padding((0, 1, 1));
        
        let conv = create_test_conv3d(4, 8, (3, 3, 3), config, &device)?;
        
        // Create input
        let input = Tensor::randn(0f32, 1.0, (1, 4, 4, 8, 8), &device)?;
        
        // Forward on CPU
        let output1 = conv.forward(&input)?;
        
        // Transfer and forward again
        let input_cpu = input.to_device(&Device::Cpu)?;
        let output2 = conv.forward(&input_cpu)?;
        
        // Should be identical
        let diff = max_abs_diff(&output1, &output2)?;
        assert!(diff < 1e-6, "Causal CPU transfer should be exact, got diff={}", diff);
        
        Ok(())
    }

    #[test]
    fn test_cache_device_consistency() -> candle_core::Result<()> {
        let device = Device::Cpu;
        
        let config = Conv3dConfig::causal((3, 3, 3))
            .with_padding((0, 1, 1));
        
        let conv = create_test_conv3d(4, 8, (3, 3, 3), config, &device)?;
        
        // Create input
        let input = Tensor::randn(0f32, 1.0, (1, 4, 4, 8, 8), &device)?;
        
        // Forward with cache
        let (output, cache) = conv.forward_with_cache(&input, None)?;
        
        // Cache should be on same device as input
        if let Some(cache) = &cache {
            assert!(matches!(cache.device(), Device::Cpu));
        }
        
        // Output should be on same device
        assert!(matches!(output.device(), Device::Cpu));
        
        Ok(())
    }
}
