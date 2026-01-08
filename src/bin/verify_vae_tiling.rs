use anyhow::Result;
use candle_core::{DType, Device, IndexOp};
use candle_video::models::ltx_video::{
    AutoencoderKLLtxVideo, loader::WeightLoader, vae::AutoencoderKLLtxVideoConfig,
};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Verify VAE Tiling: Rust vs Python")]
struct Args {
    #[arg(long, default_value = "models/models--Lightricks--LTX-Video-0.9.5")]
    local_weights: String,

    #[arg(
        long,
        default_value = "output/vae_tiling_debug/python_vs_rust_tiling.safetensors"
    )]
    verification_data: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::new_cuda(0)?;
    let dtype = DType::BF16;

    println!("{}", "=".repeat(60));
    println!("VAE Tiling Verification: Rust vs Python");
    println!("{}", "=".repeat(60));

    // Load VAE
    println!("\n[1/4] Loading VAE from: {}", args.local_weights);
    let vae_path =
        PathBuf::from(&args.local_weights).join("vae/diffusion_pytorch_model.safetensors");

    let vae_loader = WeightLoader::new(device.clone(), dtype);
    let vb = vae_loader.load_single(&vae_path)?;

    let vae_config = AutoencoderKLLtxVideoConfig {
        timestep_conditioning: true,
        ..Default::default()
    };

    let mut vae = AutoencoderKLLtxVideo::new(vae_config, vb)?;

    // Enable tiling with Python's default settings (448 stride, not 384)
    vae.use_tiling = true;
    vae.use_framewise_decoding = true;
    vae.tile_sample_min_height = 512;
    vae.tile_sample_min_width = 512;
    vae.tile_sample_stride_height = 448;
    vae.tile_sample_stride_width = 448;
    vae.tile_sample_min_num_frames = 16;
    vae.tile_sample_stride_num_frames = 8;

    println!("  VAE loaded with tiling enabled");
    println!(
        "  tile_sample_min: {}x{}",
        vae.tile_sample_min_height, vae.tile_sample_min_width
    );
    println!(
        "  tile_sample_stride: {}x{}",
        vae.tile_sample_stride_height, vae.tile_sample_stride_width
    );

    // Load verification data
    println!(
        "\n[2/4] Loading verification data: {}",
        args.verification_data
    );
    let verification_data = candle_core::safetensors::load(&args.verification_data, &device)?;

    let latents = verification_data
        .get("latents")
        .expect("Missing latents")
        .to_dtype(dtype)?;
    let temb = verification_data
        .get("temb")
        .expect("Missing temb")
        .to_dtype(dtype)?;
    let py_output = verification_data
        .get("output_python_tiled")
        .expect("Missing output_python_tiled")
        .to_dtype(dtype)?;

    println!("  Latents shape: {:?}", latents.dims());
    println!("  temb: {:?}", temb.dims());
    println!("  Python output shape: {:?}", py_output.dims());

    // Run Rust VAE tiled decode
    println!("\n[3/4] Running Rust VAE tiled decode...");
    let (_, rust_output) = vae.decode(&latents, Some(&temb), false, false)?;

    println!("  Rust output shape: {:?}", rust_output.dims());

    // Compare outputs
    println!("\n[4/4] Comparing outputs...");

    let rust_f32 = rust_output.to_dtype(DType::F32)?;
    let py_f32 = py_output.to_dtype(DType::F32)?;

    let diff = rust_f32.sub(&py_f32)?.abs()?;
    let max_diff = diff.max_all()?.to_scalar::<f32>()?;
    let mean_diff = diff.mean_all()?.to_scalar::<f32>()?;

    let rust_mean = rust_f32.mean_all()?.to_scalar::<f32>()?;
    let py_mean = py_f32.mean_all()?.to_scalar::<f32>()?;

    println!("\n  RESULTS:");
    println!("    Rust output mean: {:.6}", rust_mean);
    println!("    Python output mean: {:.6}", py_mean);
    println!("    Max absolute difference: {:.6}", max_diff);
    println!("    Mean absolute difference: {:.6}", mean_diff);

    if max_diff < 0.5 {
        println!("\n  ✅ PASS - Rust tiling matches Python within tolerance");
    } else if max_diff < 2.0 {
        println!("\n  ⚠️ WARNING - Some differences detected, may need investigation");
    } else {
        println!("\n  ❌ FAIL - Significant mismatch between Rust and Python");
    }

    // Per-frame comparison - analyze ALL frames to find spike points
    let num_frames = rust_output.dim(2)?;
    println!("\n  Per-frame analysis (all {} frames):", num_frames);
    println!("  NOTE: Temporal tile stride = 8 sample frames, boundaries at ~9, ~17, ~25...");
    let mut worst_frame = 0;
    let mut worst_diff = 0.0f32;
    for f in 0..num_frames {
        let rust_frame = rust_f32.i((.., .., f, .., ..))?;
        let py_frame = py_f32.i((.., .., f, .., ..))?;
        let frame_diff = rust_frame.sub(&py_frame)?.abs()?;
        let frame_max = frame_diff.max_all()?.to_scalar::<f32>()?;
        let frame_mean = frame_diff.mean_all()?.to_scalar::<f32>()?;

        // Only print frames with notable differences or at boundaries
        let is_boundary = f == 8 || f == 9 || f == 16 || f == 17 || f == 24 || f == 25;
        if frame_max > 0.2 || is_boundary || f < 3 || f == num_frames - 1 {
            let marker = if frame_max > 1.0 {
                " ⚠️  SPIKE"
            } else if is_boundary {
                " [boundary]"
            } else {
                ""
            };
            println!(
                "    Frame {:2}: max={:.4}, mean={:.4}{}",
                f, frame_max, frame_mean, marker
            );
        }

        if frame_max > worst_diff {
            worst_diff = frame_max;
            worst_frame = f;
        }
    }
    println!(
        "\n  WORST FRAME: {} with max diff = {:.4}",
        worst_frame, worst_diff
    );

    println!("\n{}", "=".repeat(60));
    println!("Verification complete!");
    println!("{}", "=".repeat(60));

    Ok(())
}
