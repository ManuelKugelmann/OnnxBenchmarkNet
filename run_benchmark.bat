@echo off
REM Run .NET ONNX benchmark (mirrors Python run_benchmark.sh)
REM Usage: run_benchmark.bat [cpu|cuda|tensorrt|directml|gpu|all] [--size 512] [--model esrgan2x]
REM Note: Run build.bat first

setlocal enabledelayedexpansion
cd /d "%~dp0"

set VARIANT=%1
if "%VARIANT%"=="" set VARIANT=all
if /i "%VARIANT:~0,2%"=="--" set VARIANT=all& goto :skipshift
if /i "%VARIANT:~0,1%"=="-" set VARIANT=all& goto :skipshift
shift
:skipshift

REM Collect all remaining arguments
set ARGS=
:argloop
if "%~1"=="" goto :argsdone
set ARGS=!ARGS! %1
shift
goto :argloop
:argsdone

echo ========================================
echo .NET ONNX Runtime Benchmarks
echo ========================================
echo.

REM Map variants to builds and providers
REM cuda build: cpu, cuda, tensorrt
REM directml build: cpu, directml
set RUN_CUDA=0
set RUN_DIRECTML=0
set CUDA_PROVIDER=all
set DIRECTML_PROVIDER=all

if /i "%VARIANT%"=="all" set RUN_CUDA=1& set RUN_DIRECTML=1
if /i "%VARIANT%"=="gpu" set RUN_CUDA=1& set RUN_DIRECTML=1& set CUDA_PROVIDER=gpu& set DIRECTML_PROVIDER=directml
if /i "%VARIANT%"=="cuda" set RUN_CUDA=1& set CUDA_PROVIDER=cuda
if /i "%VARIANT%"=="tensorrt" set RUN_CUDA=1& set CUDA_PROVIDER=tensorrt
if /i "%VARIANT%"=="cpu" set RUN_CUDA=1& set CUDA_PROVIDER=cpu
if /i "%VARIANT%"=="directml" set RUN_DIRECTML=1& set DIRECTML_PROVIDER=directml

if "%RUN_CUDA%%RUN_DIRECTML%"=="00" (
    echo ERROR: Unknown variant: %VARIANT%
    echo Usage: run_benchmark.bat [cpu^|cuda^|tensorrt^|directml^|gpu^|all] [benchmark args...]
    exit /b 1
)

if "%RUN_CUDA%"=="1" call :run dotnet-benchmark\bin\CUDA\Benchmark.CUDA.exe %CUDA_PROVIDER%
if "%RUN_DIRECTML%"=="1" call :run dotnet-benchmark\bin\DirectML\Benchmark.DirectML.exe %DIRECTML_PROVIDER%

echo.
echo ========================================
echo Benchmarks complete!
echo Results saved to: benchmark_results.txt
echo ========================================
exit /b 0

:run
if not exist "%~1" (
    echo ERROR: %~1 not built. Run build.bat first.
    exit /b 1
)

%~1 --provider %~2 %ARGS%

echo.
exit /b 0
