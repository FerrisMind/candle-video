use candle_core::{Device, Tensor};
use candle_video::utils::deterministic_rng::Pcg32;

fn main() -> anyhow::Result<()> {
    let seed = 42;
    let mut rng = Pcg32::new(seed, 1442695040888963407); // Default PCG stream

    println!("Generating 10 random samples on CPU...");
    let device = Device::Cpu;

    // Generate 10 samples
    let tensor = rng.randn((10,), &device)?;

    println!("Random values:");
    let vals = tensor.to_vec1::<f32>()?;
    for v in &vals {
        println!("{:.8}", v);
    }

    // Also save to file for Python verification
    let save_path = "rng_rust.safetensors";
    candle_core::safetensors::save(
        &std::collections::HashMap::from([("noise".to_string(), tensor)]),
        save_path,
    )?;
    println!("Saved to {}", save_path);

    println!("\nGenerating 10 random samples with Candle Standard RNG (Tensor::randn)...");
    let candle_tensor = Tensor::randn(0f32, 1f32, (10,), &device)?;
    let candle_vals = candle_tensor.to_vec1::<f32>()?;
    for v in &candle_vals {
        println!("{:.8}", v);
    }

    println!(
        "\nConclusion: Standard Candle RNG produces different values (and relies on thread-local RNG state/seeding)."
    );

    Ok(())
}
