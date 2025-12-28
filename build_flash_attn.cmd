@echo off
REM ============================================================================
REM Full Build Script for candle-video with Flash Attention
REM ============================================================================
REM This script initializes the full Visual Studio + CUDA environment and builds.
REM Run this directly (not sourced) from any command prompt.
REM ============================================================================

echo [1/5] Initializing Visual Studio 2022 environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo [2/5] Setting up CUDA environment...
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set CUDA_HOME=%CUDA_PATH%
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%

echo [3/5] Configuring nvcc...
set NVCC_PREPEND_FLAGS=-ccbin "%VCToolsInstallDir%\bin\Hostx64\x64"

echo [4/5] Verifying tools...
where nvcc
nvcc --version | findstr /C:"release"
where cl.exe
where cargo

echo [5/5] Building with Flash Attention...
cd /d %~dp0
cargo build --release --features flash-attn

echo.
if %ERRORLEVEL% == 0 (
    echo ============================================================================
    echo BUILD SUCCEEDED!
    echo Binary location: target\release\ltx-video.exe
    echo ============================================================================
) else (
    echo ============================================================================
    echo BUILD FAILED with error code %ERRORLEVEL%
    echo ============================================================================
)
pause
