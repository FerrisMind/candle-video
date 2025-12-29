//! Debug binary for comparing Rust RoPE output with diffusers
//! Run with: cargo run --release --bin debug-rope

use candle_core::{DType, Device};
use candle_video::rope::{FractionalRoPE, generate_indices_grid_for_diffusers};
use std::fs::File;
use std::io::Write;

fn main() -> anyhow::Result<()> {
    println!("{}", "=".repeat(60));
    println!("Rust RoPE Debug");
    println!("{}", "=".repeat(60));

    let device = Device::Cpu; // Use CPU for comparison
    let dtype = DType::F32;

    // Parameters matching our test run
    let batch_size = 1;
    let num_frames = 4; // latent frames
    let height = 10; // latent height
    let width = 16; // latent width
    let dim = 2048; // hidden_size (head_dim for RoPE)
    let seq_len = num_frames * height * width;

    // RoPE parameters from diffusers
    let frame_rate = 25.0_f32;
    let vae_temporal_compression = 8.0_f32;
    let vae_spatial_compression = 32.0_f32;

    let rope_scale_t = vae_temporal_compression / frame_rate; // 8/25 = 0.32
    let rope_scale_h = vae_spatial_compression; // 32
    let rope_scale_w = vae_spatial_compression; // 32

    println!(
        "num_frames={}, height={}, width={}",
        num_frames, height, width
    );
    println!(
        "rope_interpolation_scale=({:.2}, {:.0}, {:.0})",
        rope_scale_t, rope_scale_h, rope_scale_w
    );
    println!();

    // Generate indices grid
    let indices_grid = generate_indices_grid_for_diffusers(
        batch_size,
        num_frames,
        height,
        width,
        20,   // base_num_frames
        2048, // base_height
        2048, // base_width
        1,    // patch_size
        1,    // patch_size_t
        rope_scale_t,
        rope_scale_h,
        rope_scale_w,
        &device,
    )?;

    println!("indices_grid shape: {:?}", indices_grid.dims());

    // Print first few positions (indices_grid is (B, 3, L))
    let grid_f32 = indices_grid.to_dtype(DType::F32)?;
    let grid_vec: Vec<f32> = grid_f32.flatten_all()?.to_vec1()?;

    println!("\nFirst 5 positions (t,h,w coords at dim 1):");
    for i in 0..5.min(seq_len) {
        let t_idx = i;
        let h_idx = seq_len + i;
        let w_idx = 2 * seq_len + i;
        println!(
            "  pos {}: [{:.4}, {:.4}, {:.4}]",
            i, grid_vec[t_idx], grid_vec[h_idx], grid_vec[w_idx]
        );
    }

    // Compute RoPE frequencies
    let rope = FractionalRoPE::new(dim, 10000.0, seq_len);
    let (cos_freqs, sin_freqs) = rope.compute_freqs_cis(&indices_grid.to_dtype(dtype)?, &device)?;

    println!("\ncos_freqs shape: {:?}", cos_freqs.dims());
    println!("sin_freqs shape: {:?}", sin_freqs.dims());

    // Get first 10 values at position 0
    let cos_vec: Vec<f32> = cos_freqs.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let sin_vec: Vec<f32> = sin_freqs.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

    println!(
        "\nFirst 10 cos values at pos 0: {:?}",
        &cos_vec[0..10.min(cos_vec.len())]
    );
    println!(
        "First 10 sin values at pos 0: {:?}",
        &sin_vec[0..10.min(sin_vec.len())]
    );

    // Compute ranges
    let cos_min = cos_vec.iter().cloned().fold(f32::INFINITY, f32::min);
    let cos_max = cos_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sin_min = sin_vec.iter().cloned().fold(f32::INFINITY, f32::min);
    let sin_max = sin_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("\n{}", "=".repeat(60));
    println!("Summary");
    println!("{}", "=".repeat(60));
    println!("Final cos_freqs range: [{:.4}, {:.4}]", cos_min, cos_max);
    println!("Final sin_freqs range: [{:.4}, {:.4}]", sin_min, sin_max);

    // Save for comparison
    println!("\nSaving to output/rust_rope_*.bin for comparison...");
    std::fs::create_dir_all("output")?;

    let mut f = File::create("output/rust_cos_freqs.bin")?;
    for v in &cos_vec {
        f.write_all(&v.to_le_bytes())?;
    }

    let mut f = File::create("output/rust_sin_freqs.bin")?;
    for v in &sin_vec {
        f.write_all(&v.to_le_bytes())?;
    }

    println!("Done! Compare with Python:");
    println!("  python scripts/compare_rope.py");

    Ok(())
}
