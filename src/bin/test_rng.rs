use candle_video::utils::deterministic_rng::Pcg32;
use candle_core::Device;

fn main() -> anyhow::Result<()> {
    let mut rng = Pcg32::new(42, 1442695040888963407);
    
    println!("Rust Pcg32 first 10 Gaussian values:");
    let mut values = Vec::new();
    for _ in 0..5 {
        let (z0, z1) = rng.next_gaussian();
        values.push(z0);
        values.push(z1);
    }
    println!("{:?}", &values[..10]);
    let mean: f32 = values[..10].iter().sum::<f32>() / 10.0;
    println!("mean: {:.6}", mean);
    
    // Compare with Python output
    println!("\nPython Pcg32 first 10 Gaussian values (from test):");
    println!("[-1.9665, 0.6807, 0.3309, 0.5145, 1.3200, 0.9138, 1.9544, 1.4492, -0.0328, 0.5270]");
    
    Ok(())
}
