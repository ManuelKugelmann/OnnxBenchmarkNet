using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Management;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxBenchmark;

/// <summary>
/// ONNX Runtime Benchmark for image super-resolution and face restoration models.
/// Tests CPU, DirectML, CUDA, and TensorRT providers.
/// </summary>
class Benchmark
{
    // Paths relative to current working directory (batch file sets this to project root)
    static readonly string ProjectDir = Environment.CurrentDirectory;
    static readonly string ModelsDir = Path.Combine(ProjectDir, "models");
    static readonly string ResultsFile = Path.Combine(ProjectDir, "benchmark_results.txt");

    // GPU info (cached)
    static string? _gpuName;
    static string? _gpuRam;
    static List<GpuInfo>? _gpuList;

    // CPU info (cached)
    static string? _cpuName;
    static string? _systemRam;

    class GpuInfo
    {
        public int Id { get; set; }
        public string Name { get; set; } = "";
        public string Ram { get; set; } = "?GB";
    }

    // Available models (local Models folder preferred)
    static readonly Dictionary<string, string> Models = new()
    {
        // Super-resolution models (any size)
        ["esrgan2x"] = Path.Combine(ModelsDir, "2xNomosUni_esrgan_multijpg_fp32_opset17.onnx"),
        ["esrgan2x_fp16"] = Path.Combine(ModelsDir, "2xNomosUni_esrgan_multijpg_fp16_opset17.onnx"),
        ["compact2x"] = Path.Combine(ModelsDir, "2xNomosUni_compact_multijpg_ldl_fp32_opset17.onnx"),
        // Face restoration models (fixed sizes)
        ["gfpgan"] = Path.Combine(ModelsDir, "GFPGANv1.4.onnx"),
        ["gfpgan_fp16"] = Path.Combine(ModelsDir, "GFPGANv1.4.fp16.onnx"),
        ["gpen256"] = Path.Combine(ModelsDir, "GPEN-BFR-256.onnx"),
        ["gpen256_fp16"] = Path.Combine(ModelsDir, "GPEN-BFR-256.fp16.onnx"),
        ["gpen512_fp16"] = Path.Combine(ModelsDir, "GPEN-BFR-512.fp16.onnx"),
    };

    // Models with fixed input sizes (null = any size)
    static readonly Dictionary<string, int?> ModelFixedSizes = new()
    {
        ["compact2x"] = null,
        ["esrgan2x"] = null,
        ["esrgan2x_fp16"] = null,
        ["gpen256"] = 256,
        ["gpen256_fp16"] = 256,
        ["gpen512_fp16"] = 512,
        ["gfpgan"] = 512,
        ["gfpgan_fp16"] = 512,
    };

    static int Main(string[] args)
    {
        Console.WriteLine(new string('=', 100));
        Console.WriteLine("ONNX Runtime Benchmark (.NET)");
        Console.WriteLine(new string('=', 100));
        ListGpus();
        Console.WriteLine(new string('=', 100));

        // Parse args
        int size = 512;
        string provider = "cpu";
        string model = "esrgan2x";
        int warmup = 5;
        int runs = 5;
        bool verbose = false;
        bool profile = false;
        string optimizationLevel = "none";  // none, basic, extended, full, compare
        bool saveOptimized = false;
        bool loadOptimized = false;
        int gpuId = 0;
        bool includeTensorrt = false;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--size" when i + 1 < args.Length:
                    size = int.Parse(args[++i]);
                    break;
                case "--provider" when i + 1 < args.Length:
                    provider = args[++i].ToLower();
                    break;
                case "--model" when i + 1 < args.Length:
                    model = args[++i].ToLower();
                    break;
                case "--warmup" when i + 1 < args.Length:
                    warmup = int.Parse(args[++i]);
                    break;
                case "--runs" when i + 1 < args.Length:
                    runs = int.Parse(args[++i]);
                    break;
                case "--verbose":
                case "-v":
                    verbose = true;
                    break;
                case "--profile":
                case "-p":
                    profile = true;
                    break;
                case "--optimization":
                case "-o" when i + 1 < args.Length:
                    optimizationLevel = args[++i].ToLower();
                    break;
                case "--save-optimized":
                case "-s":
                    saveOptimized = true;
                    break;
                case "--load-optimized":
                case "-l":
                    loadOptimized = true;
                    break;
                case "--compare-optimizations":
                case "-c":
                    optimizationLevel = "compare";
                    break;
                case "--gpu":
                case "-g" when i + 1 < args.Length:
                    gpuId = int.Parse(args[++i]);
                    break;
                case "--include-tensorrt":
                    includeTensorrt = true;
                    break;
            }
        }

        // Detect hardware info
        DetectGpuInfo(gpuId);
        DetectCpuInfo();
        Console.WriteLine($"CPU: {_cpuName} ({_systemRam})");
        Console.WriteLine($"GPU [{gpuId}]: {_gpuName} ({_gpuRam})");

        // Check available providers
        string[] availableProviders;
        try
        {
            availableProviders = OrtEnv.Instance().GetAvailableProviders();
            Console.WriteLine($"Available providers: [{string.Join(", ", availableProviders)}]");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"ERROR: Failed to get providers: {ex.Message}");
            return 1;
        }

        Console.WriteLine(new string('=', 100));

        // Determine which optimization levels to benchmark
        string[] optimizationLevelsToTest = optimizationLevel == "compare"
            ? new[] { "none", "basic", "extended", "full" }
            : new[] { optimizationLevel };

        Console.WriteLine($"Optimizations to test: {string.Join(", ", optimizationLevelsToTest)}");

        // Determine which models to test
        var modelsToTest = new List<(string Name, string Path)>();
        if (model == "all")
        {
            foreach (var kvp in Models)
                modelsToTest.Add((kvp.Key, kvp.Value));
        }
        else if (Models.TryGetValue(model, out var modelPath))
        {
            modelsToTest.Add((model, modelPath));
        }
        else
        {
            Console.WriteLine($"ERROR: Unknown model: {model}");
            Console.WriteLine($"Available: {string.Join(", ", Models.Keys)}");
            return 1;
        }

        // Check models exist
        foreach (var (name, path) in modelsToTest)
        {
            if (!File.Exists(path))
            {
                Console.WriteLine($"ERROR: Model '{name}' not found: {path}");
                return 1;
            }
        }

        Console.WriteLine($"Models to test: {string.Join(", ", modelsToTest.Select(m => m.Name))}");


        // Determine which providers to test (filter by what's available)
        var providerMap = new Dictionary<string, string>
        {
            ["cpu"] = "CPUExecutionProvider",
            ["directml"] = "DmlExecutionProvider",
            ["cuda"] = "CUDAExecutionProvider",
            ["tensorrt"] = "TensorrtExecutionProvider"
        };
        string[] candidateProviders = provider switch
        {
            "all" => new[] { "cpu", "directml", "cuda", "tensorrt" },
            "gpu" => new[] { "directml", "cuda", "tensorrt" },
            _ => new[] { provider }
        };
        string[] providersToTest = candidateProviders
            .Where(p => providerMap.ContainsKey(p) && availableProviders.Contains(providerMap[p]))
            .Where(p => includeTensorrt || p != "tensorrt")
            .ToArray();

        Console.WriteLine($"Providers to test: {string.Join(", ", providersToTest)}");

        Console.WriteLine(new string('=', 100));


        var results = new System.Collections.Generic.List<string>();

        // Write header to results file
        try
        {
            using var writer = new StreamWriter(ResultsFile, append: true);
            writer.WriteLine();
            writer.WriteLine(new string('=', 100));
            writer.WriteLine($"Benchmark run: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            writer.WriteLine(new string('=', 100));
        }
        catch { }

        foreach (var (modelName, modelPath) in modelsToTest)
        {
            var modelSizeMb = new FileInfo(modelPath).Length / (1024.0 * 1024.0);
            Console.WriteLine($"\n{"="[0],100}");
            Console.WriteLine($"Testing Model: {modelName}");
            Console.WriteLine($"Path: {modelPath}");
            Console.WriteLine($"Size: {modelSizeMb:F1} MB");
            Console.WriteLine(new string('-', 100));

            // Determine sizes to test (respect model's fixed size constraint)
            var fixedSize = ModelFixedSizes.GetValueOrDefault(modelName, null);
            var sizesToTest = fixedSize.HasValue
                ? new[] { fixedSize.Value }
                : (size == 0 ? new[] { 256, 512, 1024 } : new[] { size });

            foreach (var testSize in sizesToTest)
            {
                string sizeStr = $"{testSize}x{testSize}";

                // Only run CPU for 256, skip CPU for larger sizes (mirrors Python)
                var sizeProviders = testSize == 256
                    ? providersToTest
                    : providersToTest.Where(p => p != "cpu").ToArray();

                Console.WriteLine($"\n--- {sizeStr} | providers: {string.Join(", ", sizeProviders)} | optimization: {string.Join(", ", optimizationLevelsToTest)} ---");

                foreach (var prov in sizeProviders)
                {
                    foreach (var opt in optimizationLevelsToTest)
                    {
                        // TensorRT warning
                        if (prov == "tensorrt")
                        {
                            Console.WriteLine($"  [TensorRT] First run and first warmup compiles the engine - this can take several minutes...");
                        }

                        Console.Write($"  {prov} | {opt} | ");
                        Console.Out.Flush();

                        var result = RunBenchmark(modelPath, testSize, prov, warmup, runs, availableProviders, verbose, profile, opt, saveOptimized, loadOptimized, gpuId);
                        result.ModelName = Path.GetFileNameWithoutExtension(modelPath);
                        // Only set if not already set (pre-optimized models set it with * suffix)
                        if (string.IsNullOrEmpty(result.OptimizationLevel))
                            result.OptimizationLevel = opt;

                        var resultLine = FormatResult(result, sizeStr);
                        results.Add(resultLine);

                        if (result.Success)
                        {
                            Console.WriteLine($"avg={result.AvgTimeS:F3}s");
                        }
                        else
                        {
                            Console.WriteLine($"FAILED: {result.Error}");
                        }
                        Console.WriteLine($" {resultLine}");

                        // Write to file immediately
                        try
                        {
                            using var writer = new StreamWriter(ResultsFile, append: true);
                            writer.WriteLine(resultLine);
                        }
                        catch { }
                    }
                }
            }
        }

        // Print results
        Console.WriteLine("\n" + new string('=', 100));
        Console.WriteLine("RESULTS:");
        Console.WriteLine(new string('=', 100));

        foreach (var line in results)
        {
            Console.WriteLine(line);
        }

        Console.WriteLine($"\nResults saved to: {ResultsFile}");
        if (profile)
        {
            Console.WriteLine($"Profile files saved to: {ProjectDir}");
            Console.WriteLine($"  View with: chrome://tracing or https://ui.perfetto.dev/");
        }

        return 0;
    }

    static void DetectGpuInfo(int gpuId = 0)
    {
        if (_gpuName != null) return;

        // Use the full GPU list to get info for specific GPU
        DetectAllGpus();

        if (_gpuList != null && gpuId < _gpuList.Count)
        {
            var gpu = _gpuList[gpuId];
            _gpuName = gpu.Name;
            _gpuRam = gpu.Ram;
        }
        else
        {
            _gpuName = "Unknown GPU";
            _gpuRam = "?GB";
        }
    }

    static void DetectCpuInfo()
    {
        if (_cpuName != null) return;

        try
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                // Get CPU name
                using var cpuSearcher = new ManagementObjectSearcher("SELECT Name FROM Win32_Processor");
                foreach (ManagementObject obj in cpuSearcher.Get())
                {
                    _cpuName = obj["Name"]?.ToString()?.Trim();
                    break;
                }

                // Get total system RAM
                using var memSearcher = new ManagementObjectSearcher("SELECT TotalPhysicalMemory FROM Win32_ComputerSystem");
                foreach (ManagementObject obj in memSearcher.Get())
                {
                    var totalMem = obj["TotalPhysicalMemory"];
                    if (totalMem != null)
                    {
                        var memBytes = Convert.ToInt64(totalMem);
                        _systemRam = $"{Math.Round(memBytes / (1024.0 * 1024 * 1024))}GB";
                    }
                    break;
                }
            }
        }
        catch
        {
        }

        _cpuName ??= "Unknown CPU";
        _systemRam ??= "?GB";
    }

    static void ListGpus()
    {
        DetectAllGpus();

        Console.WriteLine("Available GPUs:");
        Console.WriteLine(new string('-', 60));

        if (_gpuList == null || _gpuList.Count == 0)
        {
            Console.WriteLine("  No GPUs detected");
            return;
        }

        foreach (var gpu in _gpuList)
        {
            Console.WriteLine($"  [{gpu.Id}] {gpu.Name} ({gpu.Ram})");
        }

        Console.WriteLine(new string('-', 60));
        Console.WriteLine($"Use --gpu <id> or -g <id> to select a specific GPU");
    }

    static void DetectAllGpus()
    {
        if (_gpuList != null) return;

        _gpuList = new List<GpuInfo>();

        try
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                using var searcher = new ManagementObjectSearcher("SELECT Name, AdapterRAM FROM Win32_VideoController");
                int id = 0;
                foreach (ManagementObject obj in searcher.Get())
                {
                    var name = obj["Name"]?.ToString();
                    if (name != null && !name.Contains("Basic Display"))
                    {
                        var gpu = new GpuInfo { Id = id, Name = name };

                        // Try to get VRAM from registry for this adapter
                        var regPath = $@"SYSTEM\ControlSet001\Control\Class\{{4d36e968-e325-11ce-bfc1-08002be10318}}\{id:D4}";
                        try
                        {
                            using var key = Microsoft.Win32.Registry.LocalMachine.OpenSubKey(regPath);
                            if (key != null)
                            {
                                var qwMemorySize = key.GetValue("HardwareInformation.qwMemorySize");
                                if (qwMemorySize is long memBytes)
                                {
                                    gpu.Ram = $"{Math.Round(memBytes / (1024.0 * 1024 * 1024))}GB";
                                }
                                else if (qwMemorySize is byte[] memBytesArray && memBytesArray.Length >= 8)
                                {
                                    var mem = BitConverter.ToInt64(memBytesArray, 0);
                                    gpu.Ram = $"{Math.Round(mem / (1024.0 * 1024 * 1024))}GB";
                                }
                            }
                        }
                        catch { }

                        _gpuList.Add(gpu);
                        id++;
                    }
                }
            }
        }
        catch { }
    }

    static float[] CreateNoiseImageFloat32(int width, int height)
    {
        var random = new Random(42); // Reproducible
        var data = new float[1 * 3 * height * width];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (float)random.NextDouble();
        }
        return data;
    }

    static Float16[] CreateNoiseImageFloat16(int width, int height)
    {
        var random = new Random(42); // Reproducible
        var data = new Float16[1 * 3 * height * width];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (Float16)(float)random.NextDouble();
        }
        return data;
    }

    static BenchmarkResult RunBenchmark(string modelPath, int size,
        string provider, int warmup, int runs, string[] availableProviders, bool verbose, bool profile, string optimizationLevel, bool exportModel, bool loadOptimized, int gpuId)
    {
        var result = new BenchmarkResult { Provider = provider };

        try
        {
            var sessionOptions = new SessionOptions();
            var modelBaseName = Path.GetFileNameWithoutExtension(modelPath);
            var actualModelPath = modelPath;

            // Handle --load option - use pre-optimized model for the specified level
            if (loadOptimized)
            {
                var optimizedPath = Path.Combine(ModelsDir, $"{modelBaseName}_optimized_{provider}_{optimizationLevel}.onnx");
                if (File.Exists(optimizedPath))
                {
                    actualModelPath = optimizedPath;
                    // Disable optimization when loading pre-optimized model
                    sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL;
                    // Mark as pre-optimized in result
                    result.OptimizationLevel = $"{optimizationLevel}*";
                    Console.Write($"(loading {Path.GetFileName(optimizedPath)}) ");
                    Console.Out.Flush();
                }
                else
                {
                    result.Error = $"Optimized model not found: {Path.GetFileName(optimizedPath)}. Run with -s -o {optimizationLevel} first.";
                    return result;
                }
            }
            else
            {
                // Set optimization level
                sessionOptions.GraphOptimizationLevel = optimizationLevel switch
                {
                    "none" => GraphOptimizationLevel.ORT_DISABLE_ALL,
                    "basic" => GraphOptimizationLevel.ORT_ENABLE_BASIC,
                    "extended" => GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                    "full" => GraphOptimizationLevel.ORT_ENABLE_ALL,
                    _ => GraphOptimizationLevel.ORT_DISABLE_ALL
                };
            }

            // Set logging level
            if (verbose)
            {
                sessionOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;
                sessionOptions.LogVerbosityLevel = 1;
            }
            else
            {
                // Hide warnings unless verbose
                sessionOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
            }

            // Enable profiling
            if (profile)
            {
                sessionOptions.EnableProfiling = true;
            }

            // Export optimized model to file (skip "none" level and tensorrt - can't serialize compiled nodes)
            if (exportModel && !loadOptimized && optimizationLevel != "none" && provider != "tensorrt")
            {
                var optimizedPath = Path.Combine(ModelsDir, $"{modelBaseName}_optimized_{provider}_{optimizationLevel}.onnx");
                sessionOptions.OptimizedModelFilePath = optimizedPath;
                Console.Write($"(saving to {Path.GetFileName(optimizedPath)}) ");
                Console.Out.Flush();
            }

            // Configure provider
            switch (provider)
            {
                case "cpu":
                    // CPU is always available as fallback
                    break;

                case "directml":
                    if (!availableProviders.Contains("DmlExecutionProvider"))
                    {
                        result.Error = "DmlExecutionProvider not available";
                        return result;
                    }
                    sessionOptions.AppendExecutionProvider_DML(gpuId);
                    break;

                case "cuda":
                    if (!availableProviders.Contains("CUDAExecutionProvider"))
                    {
                        result.Error = "CUDAExecutionProvider not available";
                        return result;
                    }
                    sessionOptions.AppendExecutionProvider_CUDA(gpuId);
                    break;

                case "tensorrt":
                    if (!availableProviders.Contains("TensorrtExecutionProvider"))
                    {
                        result.Error = "TensorrtExecutionProvider not available";
                        return result;
                    }
                    sessionOptions.AppendExecutionProvider_Tensorrt(gpuId);
                    break;

                default:
                    result.Error = $"Unknown provider: {provider}";
                    return result;
            }

            // Load session
            Console.Write("load");
            var loadSw = Stopwatch.StartNew();
            using var session = new InferenceSession(actualModelPath, sessionOptions);
            loadSw.Stop();
            result.LoadTimeS = loadSw.Elapsed.TotalSeconds;
            Console.Write($" ({result.LoadTimeS:F3}s) | ");

            // Get input/output metadata
            var inputMeta = session.InputMetadata.First();
            var outputMeta = session.OutputMetadata.First();
            var inputName = inputMeta.Key;
            var outputName = outputMeta.Key;
            var inputElementType = inputMeta.Value.ElementDataType;
            var outputElementType = outputMeta.Value.ElementDataType;

            // Calculate shapes
            var inputShape = new long[] { 1, 3, size, size };
    
            // Get actual output shape from first inference
            long[] outputShape;

            bool isFp16Input = inputElementType == TensorElementType.Float16;
            bool isFp16Output = outputElementType == TensorElementType.Float16;

            // Create pre-allocated input OrtValue (pins the managed buffer for reuse)
            OrtValue inputOrtValue;
            if (isFp16Input)
            {
                var inputData = CreateNoiseImageFloat16(size, size);
                inputOrtValue = OrtValue.CreateTensorValueFromMemory<Float16>(
                    OrtMemoryInfo.DefaultInstance, inputData.AsMemory(), inputShape);
            }
            else
            {
                var inputData = CreateNoiseImageFloat32(size, size);
                inputOrtValue = OrtValue.CreateTensorValueFromMemory<float>(
                    OrtMemoryInfo.DefaultInstance, inputData.AsMemory(), inputShape);
            }

            using (inputOrtValue)
            {
                var inputNames = new[] { inputName };
                var outputNames = new[] { outputName };

                Console.Write($"1st ");
                // First run to determine output shape (includes optimization + TensorRT build)
                var firstSw = Stopwatch.StartNew();
                using (var firstOutput = session.Run(new RunOptions(), inputNames, new[] { inputOrtValue }, outputNames))
                {
                    var outputTensorInfo = firstOutput[0].GetTensorTypeAndShape();
                    outputShape = outputTensorInfo.Shape;
                    result.OutputShape = outputShape.Select(x => (int)x).ToArray();
                }
                firstSw.Stop();
                result.FirstRunTimeS = firstSw.Elapsed.TotalSeconds;
                Console.Write($" ({result.FirstRunTimeS:F3}s) | ");

                // Pre-allocate output buffer for reuse across all runs
                var outputSize = outputShape.Aggregate(1L, (a, b) => a * b);
                OrtValue outputOrtValue;
                if (isFp16Output)
                {
                    var outputData = new Float16[outputSize];
                    outputOrtValue = OrtValue.CreateTensorValueFromMemory<Float16>(
                        OrtMemoryInfo.DefaultInstance, outputData.AsMemory(), outputShape);
                }
                else
                {
                    var outputData = new float[outputSize];
                    outputOrtValue = OrtValue.CreateTensorValueFromMemory<float>(
                        OrtMemoryInfo.DefaultInstance, outputData.AsMemory(), outputShape);
                }

                using (outputOrtValue)
                {
                    var inputValues = new[] { inputOrtValue };
                    var outputValues = new[] { outputOrtValue };

                    // Warmup runs (reusing buffers)
                    using var runOptions = new RunOptions();
                    for (int i = 0; i < warmup; i++)
                    {
                        session.Run(runOptions, inputNames, inputValues, outputNames, outputValues);
                        Console.Write("w");
                        Console.Out.Flush();
                    }

                    // Timed runs (reusing pre-allocated buffers - no allocation overhead)
                    result.TimesS = new double[runs];
                    for (int i = 0; i < runs; i++)
                    {
                        var sw = Stopwatch.StartNew();
                        session.Run(runOptions, inputNames, inputValues, outputNames, outputValues);
                        sw.Stop();
                        result.TimesS[i] = sw.Elapsed.TotalSeconds;
                        Console.Write(".");
                        Console.Out.Flush();
                    }
                    Console.Write(" ");
                }
            }

            result.Success = true;
        }
        catch (Exception ex)
        {
            result.Error = ex.Message;
        }

        return result;
    }

    static string FormatResult(BenchmarkResult result, string sizeStr)
    {
        var hwInfo = result.Provider == "cpu"
            ? $"{_cpuName} ({_systemRam})"
            : $"{_gpuName} ({_gpuRam})";
        var platform = $"DOTNET | {result.ModelName}";

        if (result.Success)
        {
            var fps = result.AvgTimeS > 0 ? 1.0 / result.AvgTimeS : 0;
            return $"{hwInfo,-48} | {platform,-55} | {result.Provider,-10} | {result.OptimizationLevel,-8} | " +
                   $"{sizeStr,-9} | load: {result.LoadTimeS,5:F1}s | 1st: {result.FirstRunTimeS,5:F1}s | " +
                   $"avg: {result.AvgTimeS,6:F3}s | fps: {fps,6:F2} | min: {result.MinTimeS,6:F3}s | max: {result.MaxTimeS,6:F3}s";
        }
        else
        {
            return $"{hwInfo,-48} | {platform,-55} | {result.Provider,-10} | {result.OptimizationLevel,-8} | {sizeStr,-9} | ERROR: {result.Error}";
        }
    }

    class BenchmarkResult
    {
        public bool Success { get; set; }
        public string Provider { get; set; } = "";
        public string ModelName { get; set; } = "";
        public string OptimizationLevel { get; set; } = "";
        public string Error { get; set; } = "";
        public double LoadTimeS { get; set; }  // Session creation
        public double FirstRunTimeS { get; set; }  // First inference (includes optimization + TensorRT build)
        public double[] TimesS { get; set; } = Array.Empty<double>();
        public int[] OutputShape { get; set; } = Array.Empty<int>();

        public double AvgTimeS => TimesS.Length > 0 ? TimesS.Average() : 0;
        public double MinTimeS => TimesS.Length > 0 ? TimesS.Min() : 0;
        public double MaxTimeS => TimesS.Length > 0 ? TimesS.Max() : 0;
    }
}
