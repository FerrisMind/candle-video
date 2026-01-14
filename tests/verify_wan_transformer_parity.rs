//! Parity test for Wan transformer against diffusers reference.
//!
//! This test verifies that the Rust Wan transformer produces the same
//! output as the Python diffusers implementation.
//!
//! To generate reference data:
//!     python scripts/gen_wan_transformer_ref.py
//!
//! To run this test:
//!     cargo test verify_wan_transformer_parity --release --features flash-attn,cudnn

use candle_core::{DType, Device, Tensor};
use candle_video::models::wan::{WanTransformer3DConfig, load_transformer};
use std::path::Path;

/// Load reference tensors from safetensors file.
fn load_reference() -> Option<safetensors::SafeTensors<'static>> {
    let path = Path::new("gen_wan_transformer_ref.safetensors");
    if !path.exists() {
        println!("Reference file not found: {:?}", path);
        println!("Run: python scripts/gen_wan_transformer_ref.py");
        return None;
    }

    let data = std::fs::read(path).ok()?;
    let data = Box::leak(data.into_boxed_slice());
    safetensors::SafeTensors::deserialize(data).ok()
}

/// Convert safetensors tensor to candle tensor.
fn st_to_candle(st: &safetensors::SafeTensors, name: &str, device: &Device) -> Option<Tensor> {
    let view = st.tensor(name).ok()?;
    let shape: Vec<usize> = view.shape().to_vec();
    let data = view.data();

    // Assume F32 for reference data
    let f32_data: Vec<f32> = data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    Tensor::from_vec(f32_data, shape, device).ok()
}

/// Calculate relative error between two tensors.
fn relative_error(a: &Tensor, b: &Tensor) -> f64 {
    let a_f32 = a.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
    let b_f32 = b.to_dtype(DType::F32).unwrap().flatten_all().unwrap();

    let diff = a_f32.sub(&b_f32).unwrap().abs().unwrap();
    let max_val = a_f32
        .abs()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .max(1e-6);

    let max_diff = diff.max(0).unwrap().to_scalar::<f32>().unwrap();
    (max_diff / max_val) as f64
}

#[test]
fn verify_wan_transformer_small() {
    let reference = match load_reference() {
        Some(r) => r,
        None => {
            println!("Skipping test: reference data not available");
            return;
        }
    };

    // Check for model weights
    let model_path = Path::new("models/Wan2.1-T2V-1.3B/wan2.1_t2v_1.3B_fp16.safetensors");
    if !model_path.exists() {
        println!("Skipping test: model weights not found at {:?}", model_path);
        return;
    }

    // Setup device
    let device = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(_) => {
            println!("Skipping test: CUDA not available");
            return;
        }
    };

    let dtype = DType::BF16;

    // Load transformer
    let config = WanTransformer3DConfig::wan_t2v_1_3b();
    let transformer = load_transformer(model_path, config, &device, dtype).unwrap();

    // Load reference inputs
    let hidden_states = st_to_candle(&reference, "small_input_hidden_states", &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let timestep = st_to_candle(&reference, "small_input_timestep", &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let encoder_hidden_states =
        st_to_candle(&reference, "small_input_encoder_hidden_states", &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

    println!("Input shapes:");
    println!("  hidden_states: {:?}", hidden_states.dims());
    println!("  timestep: {:?}", timestep.dims());
    println!(
        "  encoder_hidden_states: {:?}",
        encoder_hidden_states.dims()
    );

    // Run forward pass
    let output = transformer
        .forward(
            &hidden_states,
            &timestep,
            &encoder_hidden_states,
            None,
            false,
        )
        .unwrap();

    let output_tensor = match output {
        Ok(out) => out.sample,
        Err(tensor) => tensor,
    };

    println!("Output shape: {:?}", output_tensor.dims());

    // Load reference output
    let ref_output = st_to_candle(&reference, "small_output", &device).unwrap();

    // Compare
    let rel_err = relative_error(&output_tensor, &ref_output);
    println!("Relative error: {:.6}", rel_err);

    // BF16 has limited precision, so we allow up to 5% relative error
    assert!(
        rel_err < 0.05,
        "Relative error too high: {} (expected < 0.05)",
        rel_err
    );

    println!("✓ Wan transformer small test passed");
}

#[test]
fn verify_wan_transformer_medium() {
    let reference = match load_reference() {
        Some(r) => r,
        None => {
            println!("Skipping test: reference data not available");
            return;
        }
    };

    // Check for model weights
    let model_path = Path::new("models/Wan2.1-T2V-1.3B/wan2.1_t2v_1.3B_fp16.safetensors");
    if !model_path.exists() {
        println!("Skipping test: model weights not found at {:?}", model_path);
        return;
    }

    // Setup device
    let device = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(_) => {
            println!("Skipping test: CUDA not available");
            return;
        }
    };

    let dtype = DType::BF16;

    // Load transformer
    let config = WanTransformer3DConfig::wan_t2v_1_3b();
    let transformer = load_transformer(model_path, config, &device, dtype).unwrap();

    // Load reference inputs
    let hidden_states = st_to_candle(&reference, "medium_input_hidden_states", &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let timestep = st_to_candle(&reference, "medium_input_timestep", &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let encoder_hidden_states =
        st_to_candle(&reference, "medium_input_encoder_hidden_states", &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

    println!("Input shapes:");
    println!("  hidden_states: {:?}", hidden_states.dims());
    println!("  timestep: {:?}", timestep.dims());
    println!(
        "  encoder_hidden_states: {:?}",
        encoder_hidden_states.dims()
    );

    // Run forward pass
    let output = transformer
        .forward(
            &hidden_states,
            &timestep,
            &encoder_hidden_states,
            None,
            false,
        )
        .unwrap();

    let output_tensor = match output {
        Ok(out) => out.sample,
        Err(tensor) => tensor,
    };

    println!("Output shape: {:?}", output_tensor.dims());

    // Load reference output
    let ref_output = st_to_candle(&reference, "medium_output", &device).unwrap();

    // Compare
    let rel_err = relative_error(&output_tensor, &ref_output);
    println!("Relative error: {:.6}", rel_err);

    // BF16 has limited precision, so we allow up to 5% relative error
    assert!(
        rel_err < 0.05,
        "Relative error too high: {} (expected < 0.05)",
        rel_err
    );

    println!("✓ Wan transformer medium test passed");
}
