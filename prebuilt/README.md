# Prebuilt Libraries

[![RU English](README.RU.md)](README.RU.md)
[![EN English](README.md)](README.md)

This directory contains precompiled CUDA kernels for Flash Attention to significantly speed up the build process.

## 📦 Contents

- **`libflashattention.a`** (~230 MB) - Precompiled static library containing CUDA kernels for Flash Attention 2

## 🎯 Purpose

Compiling CUDA kernels for Flash Attention from source takes **15-20 minutes** on the first build. By using precompiled kernels from this directory, subsequent builds complete in **seconds**.

## 🚀 Usage

The build script `build_flash_attn.cmd` automatically uses the prebuilt library if it exists:

```cmd
build_flash_attn.cmd
```

The script will:
1. Check for `prebuilt/libflashattention.a`
2. Copy it to the build directory
3. Skip CUDA kernel compilation
4. Complete the build in seconds

## 🔨 Creating Prebuilt Library (First Time)

If you don't have the prebuilt library yet, follow these steps:

### 1. Initial Build (Compile CUDA Kernels)

```cmd
cargo check --lib --features flash-attn
```

This will compile the CUDA kernels (takes 15-20 minutes). The build may fail at the linking step - this is expected on Windows.

### 2. Create Static Library

After compilation, create the static library manually:

```cmd
for /d %D in (target\debug\build\candle-flash-attn-*) do lib /OUT:"%D\out\libflashattention.a" "%D\out\*.o"
```

### 3. Save Prebuilt Library

Copy the created library to the `prebuilt/` directory:

```cmd
mkdir prebuilt
for /d %D in (target\debug\build\candle-flash-attn-*) do copy "%D\out\libflashattention.a" prebuilt\
```

### 4. Future Builds

Now you can use the fast build script:

```cmd
build_flash_attn.cmd
```

## ⚠️ Important Notes

### Git LFS

The `libflashattention.a` file is tracked via **Git LFS** because it exceeds GitHub's 100 MB file size limit. 

**Before cloning the repository:**
```bash
git lfs install
```

**If you already cloned without LFS:**
```bash
git lfs pull
```

### Platform Compatibility

⚠️ **Warning**: The prebuilt library is compiled for a specific:
- CUDA version
- GPU architecture (compute capability)
- Visual Studio/MSVC version

If you change any of these, you may need to rebuild the library.

### When to Rebuild

Rebuild the prebuilt library if:
- CUDA Toolkit version changes
- GPU architecture changes
- Visual Studio version changes
- You encounter linking errors

## 📝 File Size

The `libflashattention.a` file is approximately **230 MB**. This is normal for a static library containing compiled CUDA kernels.

## 🔗 Related Documentation

- [Build Script](../build_flash_attn.cmd) - Automated build script using prebuilt kernels
- [Build Workflow](../.agent/workflows/build-flash-attn.md) - Detailed build instructions
- [Main README](../README.md) - Project overview

---

**Note**: Keep this directory in version control (via Git LFS) to share the prebuilt library with other developers and CI/CD systems.

