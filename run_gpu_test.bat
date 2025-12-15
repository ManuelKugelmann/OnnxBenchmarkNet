@echo off
REM Run .NET ONNX GPU test
REM Usage: run_gpu_test.bat
REM Note: Run build.bat first

setlocal
cd /d "%~dp0"

echo ========================================
echo .NET ONNX GPU Test
echo ========================================
echo Date: %date% %time%
echo.

if not exist "dotnet-gpu-test\bin\Release\OnnxGpuTest.exe" (
    echo ERROR: GPU test not built. Run build.bat first.
    exit /b 1
)

dotnet-gpu-test\bin\Release\OnnxGpuTest.exe

echo.
echo ========================================
echo Test complete!
echo ========================================
