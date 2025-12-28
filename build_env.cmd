@echo off
REM ============================================================================
REM Build Environment Setup for candle-video with Flash Attention
REM ============================================================================
REM This script configures the environment for building candle-video with
REM CUDA Flash Attention support on Windows.
REM
REM Usage: Run this script before building:
REM   build_env.cmd
REM   cargo build --release --features flash-attn
REM ============================================================================

echo [1/4] Setting up CUDA environment...

REM CUDA Toolkit path
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set CUDA_HOME=%CUDA_PATH%

REM Add CUDA binaries to PATH
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%

REM CUDA library paths
set CUDA_LIB_PATH=%CUDA_PATH%\lib\x64
set LD_LIBRARY_PATH=%CUDA_LIB_PATH%;%LD_LIBRARY_PATH%

echo    CUDA_PATH = %CUDA_PATH%

echo [2/4] Setting up Visual Studio environment...

REM Visual Studio 2022 paths
set VSINSTALLDIR=C:\Program Files\Microsoft Visual Studio\2022\Community
set VCINSTALLDIR=%VSINSTALLDIR%\VC
set MSVC_VERSION=14.44.35207
set MSVC_PATH=%VCINSTALLDIR%\Tools\MSVC\%MSVC_VERSION%

REM Add MSVC binaries to PATH (x64 native)
set PATH=%MSVC_PATH%\bin\Hostx64\x64;%PATH%

REM Windows SDK (adjust version if needed)
set WindowsSdkDir=C:\Program Files (x86)\Windows Kits\10
set WindowsSDKVersion=10.0.22621.0

REM Include paths for compilation
set INCLUDE=%MSVC_PATH%\include;%WindowsSdkDir%\Include\%WindowsSDKVersion%\ucrt;%WindowsSdkDir%\Include\%WindowsSDKVersion%\shared;%WindowsSdkDir%\Include\%WindowsSDKVersion%\um;%CUDA_PATH%\include;%INCLUDE%

REM Library paths
set LIB=%MSVC_PATH%\lib\x64;%WindowsSdkDir%\Lib\%WindowsSDKVersion%\ucrt\x64;%WindowsSdkDir%\Lib\%WindowsSDKVersion%\um\x64;%CUDA_PATH%\lib\x64;%LIB%

echo    MSVC_PATH = %MSVC_PATH%

echo [3/4] Setting Rust/Cargo environment variables...

REM Tell nvcc to use cl.exe from MSVC
set NVCC_PREPEND_FLAGS=-ccbin "%MSVC_PATH%\bin\Hostx64\x64"

REM Cargo build configuration
set CARGO_BUILD_JOBS=4
set RUSTFLAGS=-C target-cpu=native

echo    NVCC_PREPEND_FLAGS set

echo [4/4] Verifying tools...

where nvcc >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo    nvcc: OK
    nvcc --version | findstr /C:"release"
) else (
    echo    nvcc: NOT FOUND - Check CUDA installation
)

where cl.exe >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo    cl.exe: OK
) else (
    echo    cl.exe: NOT FOUND - Check Visual Studio installation
)

where cargo >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo    cargo: OK
) else (
    echo    cargo: NOT FOUND - Check Rust installation
)

echo.
echo ============================================================================
echo Environment ready! You can now build with:
echo.
echo   cargo build --release --features flash-attn
echo   cargo build --release --features "flash-attn,cudnn"
echo   cargo build --release --features all-gpu
echo.
echo Or run the binary:
echo   cargo run --release --features flash-attn --bin ltx-video -- --help
echo ============================================================================
