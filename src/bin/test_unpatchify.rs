//! Test unpatchify with per-frame processing to avoid B*T folding issues

use candle_core::{Device, IndexOp, Result, Tensor};

/// Unpatchify a single 4D frame: [1, C, H, W] -> [1, c, H*p, W*p]
fn unpatchify_frame(x: &Tensor, patch_h: usize, patch_w: usize) -> Result<Tensor> {
    let (one, c_packed, h, w) = x.dims4()?;
    assert_eq!(one, 1);

    let c = c_packed / (patch_h * patch_w);

    // Reshape: [1, c, p_h, p_w, H, W]
    let x = x.reshape(&[1, c, patch_h, patch_w, h, w][..])?;

    // Permute to [1, c, H, p_w, W, p_h] - note swapped p_h/p_w for correct Python mapping
    // This gives output(y,x) = channel[x*patch_h + y] which matches Python
    let x = x.permute(&[0, 1, 4, 3, 5, 2][..])?;

    // Make contiguous and flatten
    let x = x.contiguous()?;
    let h_new = h * patch_h;
    let w_new = w * patch_w;
    x.reshape((1, c, h_new, w_new))
}

/// Unpatchify video by processing each frame separately
fn unpatchify_video(x: &Tensor, patch_h: usize, patch_w: usize) -> Result<Tensor> {
    let (b, c_packed, t, h, w) = x.dims5()?;
    let c = c_packed / (patch_h * patch_w);
    let h_new = h * patch_h;
    let w_new = w * patch_w;

    // Process each frame separately
    let mut frames = Vec::with_capacity(b * t);

    for bi in 0..b {
        for ti in 0..t {
            // Extract frame: [1, C, 1, H, W] -> [1, C, H, W]
            let frame = x.i((bi..bi + 1, .., ti..ti + 1, .., ..))?;
            let frame = frame.squeeze(2)?; // Remove T dimension

            // Unpatchify the frame
            let unpatchified = unpatchify_frame(&frame, patch_h, patch_w)?;
            frames.push(unpatchified);
        }
    }

    // Concatenate all frames along batch dimension: [B*T, c, H_new, W_new]
    let stacked = Tensor::cat(&frames, 0)?;

    // Reshape to [B, T, c, H_new, W_new] then permute to [B, c, T, H_new, W_new]
    let x = stacked.reshape(&[b, t, c, h_new, w_new][..])?;
    x.permute(&[0, 2, 1, 3, 4][..])
}

fn main() -> Result<()> {
    println!("=== Testing per-frame unpatchify ===\n");

    let device = Device::Cpu;

    // Test tensor: [B=1, C=48, T=2, H=4, W=6]
    let b = 1usize;
    let c_packed = 48usize;
    let t = 2usize;
    let h = 4usize;
    let w = 6usize;
    let p = 4usize;
    let _c = c_packed / (p * p);

    // Create values: channel * 1000 + t * 100 + h * 10 + w
    let mut data = Vec::with_capacity(b * c_packed * t * h * w);
    for _bi in 0..b {
        for ci in 0..c_packed {
            for ti in 0..t {
                for hi in 0..h {
                    for wi in 0..w {
                        let val = (ci as f32) * 1000.0
                            + (ti as f32) * 100.0
                            + (hi as f32) * 10.0
                            + (wi as f32);
                        data.push(val);
                    }
                }
            }
        }
    }

    let x = Tensor::from_vec(data, (b, c_packed, t, h, w), &device)?;
    println!("Input shape: {:?}", x.dims());

    // Apply per-frame unpatchify
    let out = unpatchify_video(&x, p, p)?;
    println!("Output shape: {:?}", out.dims());

    // Check values
    println!("\nOutput values at (0,0,0,y,x):");
    for y in 0..4 {
        for x_idx in 0..4 {
            let val = out.i((0, 0, 0, y, x_idx))?.to_scalar::<f32>()?;
            println!("  y={}, x={}: {:.0}", y, x_idx, val);
        }
    }

    // Expected from Python:
    // (0,0)=0, (0,1)=4000, (0,2)=8000, (0,3)=12000
    // (1,0)=1000, (1,1)=5000, ...
    let v00 = out.i((0, 0, 0, 0, 0))?.to_scalar::<f32>()?;
    let v01 = out.i((0, 0, 0, 0, 1))?.to_scalar::<f32>()?;
    let v10 = out.i((0, 0, 0, 1, 0))?.to_scalar::<f32>()?;
    let v11 = out.i((0, 0, 0, 1, 1))?.to_scalar::<f32>()?;

    println!("\nExpected: (0,0)=0, (0,1)=4000, (1,0)=1000, (1,1)=5000");
    println!(
        "Got: (0,0)={:.0}, (0,1)={:.0}, (1,0)={:.0}, (1,1)={:.0}",
        v00, v01, v10, v11
    );

    // Verify
    let tests = vec![(0, 0, 0.0), (0, 1, 4000.0), (1, 0, 1000.0), (1, 1, 5000.0)];
    let mut all_match = true;
    for (y, x_idx, expected) in tests {
        let actual = out.i((0, 0, 0, y, x_idx))?.to_scalar::<f32>()?;
        if (actual - expected).abs() > 0.1 {
            println!(
                "MISMATCH at ({},{}): expected {}, got {}",
                y, x_idx, expected, actual
            );
            all_match = false;
        }
    }

    if all_match {
        println!("\n✓ All values match Python output!");
    }

    Ok(())
}
