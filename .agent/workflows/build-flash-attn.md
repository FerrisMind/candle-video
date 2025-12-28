---
description: Build candle-video with Flash Attention on Windows
---

# Build with Flash Attention on Windows

This workflow describes how to build candle-video with CUDA Flash Attention support on Windows.

## Prerequisites

1. **CUDA Toolkit 12.x** - Install from NVIDIA website
2. **Visual Studio 2022** - Install with "Desktop development with C++" workload
3. **Rust** - Install via rustup

## Quick Build

// turbo
1. Open "Developer Command Prompt for VS 2022" (not PowerShell!)

2. Navigate to project directory:
```cmd
cd C:\candle-video
```

// turbo
3. Set CUDA environment:
```cmd
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set PATH=%CUDA_PATH%\bin;%PATH%
```

4. Build with flash-attn:
```cmd
cargo build --release --features flash-attn
```

## Alternative: Use Build Script

Run the provided build script:
```cmd
build_flash_attn.cmd
```

## Troubleshooting

### nvcc not found
Ensure CUDA bin directory is in PATH:
```cmd
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;%PATH%
```

### cl.exe not found
Use Developer Command Prompt, not regular CMD/PowerShell.

### CUDA version mismatch
Check your CUDA version:
```cmd
nvcc --version
```
Update `build_env.cmd` and `build_env.ps1` if needed.

## Build Without Flash Attention

If you don't need Flash Attention (uses standard attention):
```cmd
cargo build --release
```

## Available Features

- `flash-attn` - Flash Attention v2 (requires CUDA)
- `cudnn` - cuDNN acceleration (requires cuDNN)
- `mkl` - Intel MKL for CPU (x86_64 only)
- `nccl` - Multi-GPU support
- `all-gpu` - All GPU features combined
