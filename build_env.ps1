# Build Environment Setup for candle-video with Flash Attention
# ============================================================================
# This script configures the environment for building candle-video with
# CUDA Flash Attention support on Windows PowerShell.
#
# Usage: 
#   . .\build_env.ps1
#   cargo build --release --features flash-attn
# ============================================================================

Write-Host "[1/4] Setting up CUDA environment..." -ForegroundColor Cyan

# CUDA Toolkit path
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:CUDA_HOME = $env:CUDA_PATH

# Add CUDA binaries to PATH
$env:PATH = "$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:PATH"

# CUDA library paths
$env:CUDA_LIB_PATH = "$env:CUDA_PATH\lib\x64"

Write-Host "   CUDA_PATH = $env:CUDA_PATH" -ForegroundColor Gray

Write-Host "[2/4] Setting up Visual Studio environment..." -ForegroundColor Cyan

# Visual Studio 2022 paths
$VSINSTALLDIR = "C:\Program Files\Microsoft Visual Studio\2022\Community"
$VCINSTALLDIR = "$VSINSTALLDIR\VC"
$MSVC_VERSION = "14.44.35207"
$MSVC_PATH = "$VCINSTALLDIR\Tools\MSVC\$MSVC_VERSION"

# Add MSVC binaries to PATH (x64 native)
$env:PATH = "$MSVC_PATH\bin\Hostx64\x64;$env:PATH"

# Windows SDK (adjust version if needed)
$WindowsSdkDir = "C:\Program Files (x86)\Windows Kits\10"
$WindowsSDKVersion = "10.0.22621.0"

# Include paths for compilation
$env:INCLUDE = "$MSVC_PATH\include;$WindowsSdkDir\Include\$WindowsSDKVersion\ucrt;$WindowsSdkDir\Include\$WindowsSDKVersion\shared;$WindowsSdkDir\Include\$WindowsSDKVersion\um;$env:CUDA_PATH\include"

# Library paths
$env:LIB = "$MSVC_PATH\lib\x64;$WindowsSdkDir\Lib\$WindowsSDKVersion\ucrt\x64;$WindowsSdkDir\Lib\$WindowsSDKVersion\um\x64;$env:CUDA_PATH\lib\x64"

Write-Host "   MSVC_PATH = $MSVC_PATH" -ForegroundColor Gray

Write-Host "[3/4] Setting Rust/Cargo environment variables..." -ForegroundColor Cyan

# Tell nvcc to use cl.exe from MSVC
$env:NVCC_PREPEND_FLAGS = "-ccbin `"$MSVC_PATH\bin\Hostx64\x64`""

# CC for build scripts
$env:CC = "$MSVC_PATH\bin\Hostx64\x64\cl.exe"
$env:CXX = "$MSVC_PATH\bin\Hostx64\x64\cl.exe"

# Cargo build configuration
$env:CARGO_BUILD_JOBS = "4"
$env:RUSTFLAGS = "-C target-cpu=native"

Write-Host "   NVCC_PREPEND_FLAGS set" -ForegroundColor Gray

Write-Host "[4/4] Verifying tools..." -ForegroundColor Cyan

# Check nvcc
$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if ($nvcc) {
    Write-Host "   nvcc: OK" -ForegroundColor Green
    & nvcc --version | Select-String "release"
} else {
    Write-Host "   nvcc: NOT FOUND - Check CUDA installation" -ForegroundColor Red
}

# Check cl.exe
$cl = Get-Command cl.exe -ErrorAction SilentlyContinue
if ($cl) {
    Write-Host "   cl.exe: OK" -ForegroundColor Green
} else {
    Write-Host "   cl.exe: NOT FOUND - Check Visual Studio installation" -ForegroundColor Red
}

# Check cargo
$cargo = Get-Command cargo -ErrorAction SilentlyContinue
if ($cargo) {
    Write-Host "   cargo: OK" -ForegroundColor Green
} else {
    Write-Host "   cargo: NOT FOUND - Check Rust installation" -ForegroundColor Red
}

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Yellow
Write-Host "Environment ready! You can now build with:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  cargo build --release --features flash-attn" -ForegroundColor White
Write-Host "  cargo build --release --features `"flash-attn,cudnn`"" -ForegroundColor White
Write-Host "  cargo build --release --features all-gpu" -ForegroundColor White
Write-Host ""
Write-Host "Or run the binary:" -ForegroundColor Yellow
Write-Host "  cargo run --release --features flash-attn --bin ltx-video -- --help" -ForegroundColor White
Write-Host "============================================================================" -ForegroundColor Yellow
