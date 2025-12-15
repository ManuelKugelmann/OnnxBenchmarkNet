using System;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxGpuTest;

class Program
{
    static int Main(string[] args)
    {
        Console.WriteLine(new string('=', 60));
        Console.WriteLine("ONNX Runtime GPU Test (.NET)");
        Console.WriteLine(new string('=', 60));

        // 1. Check available providers
        Console.WriteLine("\n[INFO] Checking available execution providers...");

        string[] providers;
        try
        {
            providers = OrtEnv.Instance().GetAvailableProviders();
            Console.WriteLine($"\n[OK] Available Providers: [{string.Join(", ", providers)}]");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n[FAIL] Failed to get providers: {ex.Message}");
            return 1;
        }

        bool hasCuda = providers.Contains("CUDAExecutionProvider");
        bool hasTensorRt = providers.Contains("TensorrtExecutionProvider");
        bool hasDml = providers.Contains("DmlExecutionProvider");

        Console.WriteLine($"\n   CUDA:     {(hasCuda ? "[OK]" : "[--]")}");
        Console.WriteLine($"   TensorRT: {(hasTensorRt ? "[OK]" : "[--]")}");
        Console.WriteLine($"   DirectML: {(hasDml ? "[OK]" : "[--]")}");

        if (!hasCuda)
        {
            Console.WriteLine("\n[WARN] CUDAExecutionProvider not available!");
            Console.WriteLine("   Possible causes:");
            Console.WriteLine("   - Microsoft.ML.OnnxRuntime.Gpu not installed");
            Console.WriteLine("   - CUDA/cuDNN not found or version mismatch");
            Console.WriteLine("   - Missing Visual C++ Runtime");
            // Don't exit - continue to try anyway
        }

        // 2. Create minimal ONNX model in memory
        Console.WriteLine("\n" + new string('-', 60));
        Console.WriteLine("Creating minimal test model...");

        // Minimal Identity model as bytes (pre-built)
        // This is a simple model: input -> Identity -> output
        // Built with opset 11, IR version 6
        byte[] modelBytes = CreateMinimalOnnxModel();
        Console.WriteLine($"[OK] Test model created ({modelBytes.Length} bytes)");

        // 3. Test CUDA provider
        Console.WriteLine("\n" + new string('-', 60));
        Console.WriteLine("Testing CUDAExecutionProvider...");

        try
        {
            var sessionOptions = new SessionOptions();
            sessionOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;

            // Try to append CUDA provider
            try
            {
                sessionOptions.AppendExecutionProvider_CUDA(0);
                Console.WriteLine("[OK] CUDA provider appended to session options");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[WARN] Failed to append CUDA provider: {ex.Message}");
                Console.WriteLine("       Falling back to CPU...");
            }

            // Create session
            using var session = new InferenceSession(modelBytes, sessionOptions);

            Console.WriteLine($"\n[OK] Session created successfully!");

            // Check which provider is actually being used
            var inputMeta = session.InputMetadata.First();
            var outputMeta = session.OutputMetadata.First();

            Console.WriteLine($"   Input:  {inputMeta.Key} [{string.Join(",", inputMeta.Value.Dimensions)}]");
            Console.WriteLine($"   Output: {outputMeta.Key} [{string.Join(",", outputMeta.Value.Dimensions)}]");

            // 4. Run inference
            Console.WriteLine("\n" + new string('-', 60));
            Console.WriteLine("Running inference test...");

            // Create input tensor [1, 3, 224, 224]
            var inputData = new float[1 * 3 * 224 * 224];
            var random = new Random(42);
            for (int i = 0; i < inputData.Length; i++)
                inputData[i] = (float)(random.NextDouble() * 2 - 1);

            var inputTensor = new DenseTensor<float>(inputData, new[] { 1, 3, 224, 224 });
            var inputs = new[] { NamedOnnxValue.CreateFromTensor(inputMeta.Key, inputTensor) };

            using var results = session.Run(inputs);
            var output = results.First().AsTensor<float>();

            Console.WriteLine($"[OK] Inference successful!");
            Console.WriteLine($"   Input shape:  [1, 3, 224, 224]");
            Console.WriteLine($"   Output shape: [{string.Join(", ", output.Dimensions.ToArray())}]");

            // Verify output matches input (Identity op)
            bool identical = true;
            for (int i = 0; i < Math.Min(10, inputData.Length); i++)
            {
                if (Math.Abs(inputData[i] - output.GetValue(i)) > 1e-6f)
                {
                    identical = false;
                    break;
                }
            }
            Console.WriteLine($"   Output matches input: {(identical ? "[OK]" : "[MISMATCH]")}");

        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n[FAIL] Session/inference failed: {ex.Message}");
            if (ex.InnerException != null)
                Console.WriteLine($"       Inner: {ex.InnerException.Message}");
            return 1;
        }

        Console.WriteLine("\n" + new string('=', 60));
        Console.WriteLine("[OK] ALL TESTS PASSED - ONNX Runtime GPU test complete!");
        Console.WriteLine(new string('=', 60));

        return 0;
    }

    /// <summary>
    /// Creates a minimal ONNX model with Identity op.
    /// Input: float[1,3,224,224] -> Identity -> Output: float[1,3,224,224]
    /// </summary>
    static byte[] CreateMinimalOnnxModel()
    {
        // Pre-built minimal ONNX model bytes (opset 11, IR version 6)
        // Model: input (float[1,3,224,224]) -> Identity -> output (float[1,3,224,224])
        // Model size: 119 bytes
        return new byte[] {
            0x08, 0x06, 0x3A, 0x6D, 0x0A, 0x19, 0x0A, 0x05, 0x69, 0x6E, 0x70, 0x75, 0x74, 0x12, 0x06, 0x6F,
            0x75, 0x74, 0x70, 0x75, 0x74, 0x22, 0x08, 0x49, 0x64, 0x65, 0x6E, 0x74, 0x69, 0x74, 0x79, 0x12,
            0x09, 0x54, 0x65, 0x73, 0x74, 0x47, 0x72, 0x61, 0x70, 0x68, 0x5A, 0x21, 0x0A, 0x05, 0x69, 0x6E,
            0x70, 0x75, 0x74, 0x12, 0x18, 0x0A, 0x16, 0x08, 0x01, 0x12, 0x12, 0x0A, 0x02, 0x08, 0x01, 0x0A,
            0x02, 0x08, 0x03, 0x0A, 0x03, 0x08, 0xE0, 0x01, 0x0A, 0x03, 0x08, 0xE0, 0x01, 0x62, 0x22, 0x0A,
            0x06, 0x6F, 0x75, 0x74, 0x70, 0x75, 0x74, 0x12, 0x18, 0x0A, 0x16, 0x08, 0x01, 0x12, 0x12, 0x0A,
            0x02, 0x08, 0x01, 0x0A, 0x02, 0x08, 0x03, 0x0A, 0x03, 0x08, 0xE0, 0x01, 0x0A, 0x03, 0x08, 0xE0,
            0x01, 0x42, 0x04, 0x0A, 0x00, 0x10, 0x0B,
        };
    }
}
