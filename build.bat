@echo off
REM Build .NET ONNX benchmarks
REM Usage: build.bat [cuda|directml|all]

setlocal
cd /d "%~dp0"

set TARGET=%1
if "%TARGET%"=="" set TARGET=all

echo ========================================
echo Building .NET ONNX Benchmarks
echo ========================================

if /i "%TARGET%"=="all" (
    call :build CUDA dotnet-benchmark\Benchmark.CUDA.csproj
    call :build DirectML dotnet-benchmark\Benchmark.DirectML.csproj
    call :build "GPU test" dotnet-gpu-test\OnnxGpuTest.csproj
) else if /i "%TARGET%"=="cuda" (
    call :build CUDA dotnet-benchmark\Benchmark.CUDA.csproj
) else if /i "%TARGET%"=="directml" (
    call :build DirectML dotnet-benchmark\Benchmark.DirectML.csproj
) else (
    echo ERROR: Unknown target: %TARGET%
    echo Usage: build.bat [cuda^|directml^|all]
    exit /b 1
)

echo.
echo Build complete!
exit /b 0

:build
echo Building %~1...
dotnet build %~2 -c Release -v q
if errorlevel 1 (
    echo ERROR: %~1 build failed
    exit /b 1
)
exit /b 0
