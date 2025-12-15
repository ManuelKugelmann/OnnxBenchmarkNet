# OnnxBenchmarkNet

A simple .NET benchmarking tool for ONNX Runtime execution providers. Tests inference performance across CPU, DirectML, CUDA, and TensorRT providers using image super-resolution (ESRGAN) and face restoration (GFPGAN, GPEN) models.

**Note:** This is a quick and simple benchmark for basic comparisons, not a sophisticated or rigorous benchmarking suite.

## Features

- Benchmarks ONNX Runtime inference across multiple execution providers:
  - **CPU** - Default provider
  - **DirectML** - Windows GPU acceleration via DirectX 12 (sustained engineering, see note below)
  - **CUDA** - NVIDIA GPU acceleration
  - **TensorRT** - NVIDIA optimized inference engine

> **Note:** DirectML is in sustained engineering mode. For new Windows projects, Microsoft recommends [WinML](https://learn.microsoft.com/en-us/windows/ai/windows-ml/) instead:
> ```
> dotnet add package Microsoft.AI.MachineLearning
> ```
- Tests multiple ESRGAN super-resolution models
- Configurable optimization levels (none, basic, extended, full)
- Save/load optimized models for faster subsequent runs
- Profiling support (view with chrome://tracing or https://ui.perfetto.dev/)
- Automatic hardware detection (CPU, GPU, VRAM)
- Results saved to file for comparison

## Requirements

- .NET 8.0 SDK
- Windows with DirectX 12 (for DirectML support)

**Optional (for NVIDIA GPU acceleration):**
- [CUDA toolkit](https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_576.57_windows.exe) - required for CUDA and TensorRT providers
- [cuDNN](https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.17.0.29_cuda12-archive.zip) - required for CUDA provider; extract and copy `bin/*.dll` to CUDA bin folder (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin`)
- [TensorRT](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.14.1/zip/TensorRT-10.14.1.48.Windows.win10.cuda-12.9.zip) - required for TensorRT provider; extract and copy `lib/*.dll` to CUDA bin folder (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin`)

> **Note:** TensorRT compiles optimized inference engines on first run, which can take several minutes per model. Use `--include-tensorrt` to explicitly enable it.

## Project Structure

```
OnnxBenchmarkNet/
├── dotnet-benchmark/
│   ├── Benchmark.cs              # Core benchmark logic
│   ├── Benchmark.DirectML.csproj # DirectML build
│   └── Benchmark.CUDA.csproj     # CUDA/TensorRT build
├── dotnet-gpu-test/
│   └── OnnxGpuTest.cs            # GPU detection test
├── runtimes/                      # Local ONNX Runtime (optional override)
├── models/                        # ONNX models (not included)
├── build.bat                      # Build script
├── run_benchmark.bat              # Benchmark runner
└── run_gpu_test.bat               # GPU test runner
```

## Building

```cmd
build.bat              # Build all variants
build.bat cuda         # Build CUDA only
build.bat directml     # Build DirectML only
```

## Usage

### Basic Usage

```cmd
run_benchmark.bat                           # Run all providers (TensorRT excluded by default)
run_benchmark.bat --include-tensorrt        # Run all providers including TensorRT
run_benchmark.bat cuda                      # CUDA provider only
run_benchmark.bat directml                  # DirectML provider only
run_benchmark.bat gpu                       # All GPU providers (no CPU, TensorRT excluded)
run_benchmark.bat tensorrt                  # TensorRT only
```

### With Options

```cmd
run_benchmark.bat cuda --size 256 --model esrgan2x --warmup 2 --runs 5
run_benchmark.bat cuda -c -s                # Compare all opt levels and save models
run_benchmark.bat cuda -l                   # Load pre-saved optimized model
run_benchmark.bat cuda -o basic             # Use basic optimization level
run_benchmark.bat cuda -g 1                 # Use GPU 1
run_benchmark.bat cuda -v                   # Verbose ONNX Runtime logging
run_benchmark.bat cuda -p                   # Enable profiling
```

### Options

| Flag | Long Form | Description |
|------|-----------|-------------|
| `--size` | | Input size (default: 512, use 0 for 256,512,1024) |
| `--model` | | Model: esrgan2x, esrgan2x_fp16, compact2x, gfpgan, gfpgan_fp16, gpen256, gpen256_fp16, gpen512_fp16, all |
| `--warmup` | | Number of warmup iterations (default: 5) |
| `--runs` | | Number of timed runs (default: 5) |
| `--provider` | | Provider: cpu, cuda, tensorrt, directml, gpu, all (default: cpu) |
| `--include-tensorrt` | Include TensorRT in all/gpu mode (excluded by default due to long compilation) |
| `-o` | `--optimization` | Optimization level: none, basic, extended, full (default: none) |
| `-c` | `--compare-optimizations` | Benchmark all optimization levels |
| `-s` | `--save-optimized` | Export optimized model to file |
| `-l` | `--load-optimized` | Load pre-saved optimized model |
| `-v` | `--verbose` | Enable verbose ONNX Runtime logging |
| `-p` | `--profile` | Enable profiling output |
| `-g` | `--gpu <id>` | Select GPU by device ID (default: 0) |


## Optimization Workflow

1. **Compare optimization levels** to find the best for your model:
   ```cmd
   run_benchmark.bat cuda -c --size 256 --model esrgan2x
   ```

2. **Save optimized models** (saves basic, extended, full - skips none and tensorrt):
   ```cmd
   run_benchmark.bat cuda -c -s --size 256 --model esrgan2x
   ```

3. **Load pre-optimized model** for faster startup:
   ```cmd
   run_benchmark.bat cuda -l -o full    # Load model optimized with -o full
   run_benchmark.bat cuda -l -o basic   # Load model optimized with -o basic
   ```
   Note: When loading pre-optimized models, runtime optimization is automatically disabled.

## Models

Baseline models are included in `models/`:

**Super-resolution (any input size):**
| Model | Alias | Size | Description |
|-------|-------|------|-------------|
| `2xNomosUni_esrgan_multijpg_fp32_opset17.onnx` | esrgan2x | 67 MB | 2x upscaling, FP32 |
| `2xNomosUni_esrgan_multijpg_fp16_opset17.onnx` | esrgan2x_fp16 | 34 MB | 2x upscaling, FP16 |
| `2xNomosUni_compact_multijpg_ldl_fp32_opset17.onnx` | compact2x | 2 MB | Compact 2x upscaling |

**Face restoration (fixed input size):**
| Model | Alias | Size | Fixed Size |
|-------|-------|------|------------|
| `GFPGANv1.4.onnx` | gfpgan | 340 MB | 512x512 |
| `GFPGANv1.4.fp16.onnx` | gfpgan_fp16 | 170 MB | 512x512 |
| `GPEN-BFR-256.onnx` | gpen256 | 76 MB | 256x256 |
| `GPEN-BFR-256.fp16.onnx` | gpen256_fp16 | 38 MB | 256x256 |
| `GPEN-BFR-512.fp16.onnx` | gpen512_fp16 | 142 MB | 512x512 |

Optimized models (`*_optimized_*.onnx`) are generated via `-s` flag and excluded from git.

**Model Licenses:**
| Model | Source | License |
|-------|--------|---------|
| 2xNomosUni (ESRGAN/Compact) | [OpenModelDB](https://openmodeldb.info/) | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) |
| GFPGAN | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) | [Apache-2.0](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE) |
| GPEN | [yangxy/GPEN](https://github.com/yangxy/GPEN) | [Non-commercial](https://github.com/yangxy/GPEN/blob/main/LICENSE) |

## Output

Results are printed to console and appended to `benchmark_results.txt`:

```
NVIDIA GeForce RTX 3080 (16GB)  | DOTNET | model_name | cuda | none | 512x512 | load: 0.7s | 1st: 0.5s | avg: 0.27s | fps: 3.70 | min: 0.26s | max: 0.28s
```

- `load` - Session creation time
- `1st` - First inference (includes graph optimization + TensorRT engine build)

## Local ONNX Runtime

To use a newer ONNX Runtime than NuGet provides:

1. Download from https://github.com/microsoft/onnxruntime/releases
2. Extract to `runtimes/onnxruntime-win-x64-gpu-X.X.X/`
3. Update path in `.csproj` if needed

## License

MIT License
