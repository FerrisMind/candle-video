//! Generate a lightweight machine-readable manifest for a local Wan model.

use std::path::PathBuf;

use candle_video::write_wan_manifest;
use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "wan-manifest")]
struct Args {
    /// Diffusers or consolidated Wan model directory.
    #[arg(long)]
    model_path: PathBuf,
    /// Optional manifest path (defaults to <model_path>/candle-video-manifest.json).
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let output = write_wan_manifest(&args.model_path, args.output.as_deref())?;
    println!("wrote Wan manifest to {}", output.display());
    Ok(())
}
