---
description: Build candle-video with Flash Attention on Windows
---

# Build with Flash Attention on Windows

This workflow describes how to build candle-video with CUDA Flash Attention support on Windows.

## Prerequisites

1. **CUDA Toolkit 12.x** - Install from NVIDIA website
2. **Visual Studio 2022** - Install with "Desktop development with C++" workload
3. **Rust** - Install via rustup

## Quick Build (with prebuilt kernels)

If you have prebuilt kernels in `prebuilt/libflashattention.a`:

// turbo
1. Run the build script:
```cmd
build_flash_attn.cmd
```

This takes ~3 seconds instead of 15+ minutes!

## First-Time Build (compiling CUDA kernels)

// turbo
1. Open "Developer Command Prompt for VS 2022"

2. Navigate to project:
```cmd
cd C:\candle-video
```

3. Set CUDA environment:
```cmd
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set PATH=%CUDA_PATH%\bin;%PATH%
```

4. Start initial build (CUDA kernels will compile):
```cmd
cargo check --lib --features flash-attn
```
This will fail at linking step - that's expected!

5. Create static library manually:
```cmd
for /d %D in (target\debug\build\candle-flash-attn-*) do lib /OUT:"%D\out\libflashattention.a" "%D\out\*.o"
```

6. Complete the build:
```cmd
cargo build --lib --features flash-attn
```

7. Save prebuilt kernels for future use:
```cmd
mkdir prebuilt
for /d %D in (target\debug\build\candle-flash-attn-*) do copy "%D\out\libflashattention.a" prebuilt\
```

## Troubleshooting

### nvcc not found
- Use Developer Command Prompt for VS 2022, not regular CMD
- Ensure CUDA bin is in PATH

### nvcc linking error
- This is expected on Windows
- Use the `lib.exe` workaround shown above

### Type mismatch errors
- Ensure all candle dependencies use git version with same tag
- Check Cargo.toml uses `git = "..."` not `version = "..."`

## Notes

- CUDA kernel compilation takes 15-20 minutes on first build
- After first build, `libflashattention.a` (~240MB) is cached
- Keep `prebuilt/` directory to skip recompilation
