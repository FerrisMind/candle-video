//! Debug script to check GGUF tensor shapes

use candle_core::Device;
use candle_transformers::quantized_var_builder::VarBuilder;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let path = "ltxv-2b-0.9.8-distilled/t5-v1_1-xxl-encoder-Q5_K_M.gguf";

    println!("Loading GGUF from: {}", path);
    let vb = VarBuilder::from_gguf(path, &device)?;

    // Test 2D tensors
    println!("\n=== 2D Tensors ===");

    let test_2d = vec![
        ("token_embd.weight", vec![(32128, 4096), (4096, 32128)]),
        ("enc.blk.0.attn_q.weight", vec![(4096, 4096)]),
        ("enc.blk.0.attn_rel_b.weight", vec![(64, 32), (32, 64)]),
        (
            "enc.blk.0.ffn_up.weight",
            vec![(4096, 10240), (10240, 4096)],
        ),
        (
            "enc.blk.0.ffn_gate.weight",
            vec![(4096, 10240), (10240, 4096)],
        ),
        (
            "enc.blk.0.ffn_down.weight",
            vec![(10240, 4096), (4096, 10240)],
        ),
    ];

    for (name, shapes) in test_2d {
        println!("\nTensor: {}", name);
        for (a, b) in shapes {
            match vb.get((a, b), name) {
                Ok(_) => println!("  ✓ Shape ({}, {}) works!", a, b),
                Err(e) => {
                    let err_str = e.to_string();
                    if err_str.contains("shape mismatch") {
                        println!("  ✗ Shape ({}, {}) - {}", a, b, err_str);
                    } else {
                        println!("  ✗ Shape ({}, {}) - other error", a, b);
                    }
                }
            }
        }
    }

    // Test 1D tensors
    println!("\n=== 1D Tensors ===");

    let test_1d = vec![
        ("enc.blk.0.attn_norm.weight", 4096),
        ("enc.blk.0.ffn_norm.weight", 4096),
        ("enc.output_norm.weight", 4096),
    ];

    for (name, size) in test_1d {
        println!("\nTensor: {}", name);
        match vb.get((size,), name) {
            Ok(_) => println!("  ✓ Shape ({},) works!", size),
            Err(e) => println!("  ✗ Shape ({},) failed: {}", size, e),
        }
    }

    Ok(())
}
