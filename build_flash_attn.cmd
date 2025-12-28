@echo off
REM ============================================================================
REM Build candle-video with Flash Attention using prebuilt CUDA kernels
REM ============================================================================
REM This script uses precompiled CUDA kernels from the prebuilt/ directory,
REM making builds much faster (seconds instead of 15+ minutes).
REM ============================================================================

echo [1/4] Initializing Visual Studio 2022 environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo [2/4] Checking for prebuilt CUDA kernels...
if not exist "%~dp0prebuilt\libflashattention.a" (
    echo ERROR: Prebuilt kernels not found!
    echo Run initial build first: cargo check --lib --features flash-attn
    echo Then create library manually with lib.exe and copy to prebuilt/
    exit /b 1
)
echo    Found: prebuilt\libflashattention.a

echo [3/4] Finding flash-attn build directory...
cd /d %~dp0
for /f "delims=" %%D in ('dir /b /ad target\debug\build\candle-flash-attn-* 2^>nul') do (
    set BUILD_DIR=target\debug\build\%%D\out
)

if not defined BUILD_DIR (
    echo First build not started yet. Running initial check...
    cargo check --lib --features flash-attn 2>&1 | findstr /V "nvcc error"
    for /f "delims=" %%D in ('dir /b /ad target\debug\build\candle-flash-attn-* 2^>nul') do (
        set BUILD_DIR=target\debug\build\%%D\out
    )
)

echo    Target directory: %BUILD_DIR%

REM Create output directory if it doesn't exist
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

REM Copy prebuilt library
copy /Y "prebuilt\libflashattention.a" "%BUILD_DIR%\"

echo [4/4] Building candle-video...
cargo build --lib --features flash-attn

if %ERRORLEVEL% == 0 (
    echo.
    echo ============================================================================
    echo BUILD SUCCEEDED in seconds!
    echo.
    echo For release build:
    echo   cargo build --release --features flash-attn
    echo ============================================================================
) else (
    echo.
    echo ============================================================================
    echo BUILD FAILED with error code %ERRORLEVEL%
    echo ============================================================================
)
