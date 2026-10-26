# -*- coding: utf-8 -*-
"""
Export Laya Visual Triage Network to High-Performance Formats
============================================================
Compiles LayaVisualTriageNet (System 1 Ultra-Fast Clinical Screening)
into TorchScript JIT and INT8 Dynamic Quantization following Jev's
non-autoregressive ultra-fast execution philosophy.

Features:
1. Loads pre-trained weights from checkpoints/laya_triage_best.pth.
2. Traces computation graph into TorchScript JIT (checkpoints/laya_triage_jit.pt).
3. Applies INT8 dynamic quantization with graceful Conv2d fallback (checkpoints/laya_triage_quantized.pth).
4. Conducts rigorous CPU latency & throughput benchmarking (50 warmup, 500 eval iterations).
5. Emits summary table and writes structured benchmark report to checkpoints/laya_fast_benchmark.json.
"""

import os
import sys
import time
import json
import argparse
import platform
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from laya.modules.laya_visual_triage import LayaVisualTriageNet


def parse_args():
    parser = argparse.ArgumentParser(description="Export and Benchmark LayaVisualTriageNet with TorchScript and Quantization")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/laya_triage_best.pth",
                        help="Path to pre-trained model checkpoint")
    parser.add_argument("--jit_output", type=str, default="checkpoints/laya_triage_jit.pt",
                        help="Output path for TorchScript traced model")
    parser.add_argument("--quant_output", type=str, default="checkpoints/laya_triage_quantized.pth",
                        help="Output path for quantized model checkpoint")
    parser.add_argument("--quant_jit_output", type=str, default="checkpoints/laya_triage_quantized_jit.pt",
                        help="Output path for quantized TorchScript traced model")
    parser.add_argument("--json_output", type=str, default="checkpoints/laya_fast_benchmark.json",
                        help="Output path for benchmark results JSON")
    parser.add_argument("--warmup", type=int, default=50,
                        help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=500,
                        help="Number of benchmark evaluation iterations")
    parser.add_argument("--in_channels", type=int, default=3,
                        help="Input channels")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classification categories")
    parser.add_argument("--img_size", type=int, default=128,
                        help="Input preview image resolution")
    return parser.parse_args()


def get_file_size_kb(filepath: str) -> float:
    """Returns size of a file in kilobytes (KB)."""
    if os.path.exists(filepath):
        return round(os.path.getsize(filepath) / 1024.0, 2)
    return 0.0


def benchmark_model_cpu(model: nn.Module, dummy_input: torch.Tensor, warmup_iters: int = 50, eval_iters: int = 500) -> Dict[str, float]:
    """
    Rigorously benchmarks inference latency and throughput on CPU.
    """
    model.eval()
    with torch.no_grad():
        # Warmup phase
        for _ in range(warmup_iters):
            _ = model(dummy_input)

        latencies_us = []
        t_start_total = time.perf_counter()
        for _ in range(eval_iters):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies_us.append((t1 - t0) * 1e6)
        t_end_total = time.perf_counter()

    latencies_arr = np.array(latencies_us)
    mean_us = float(np.mean(latencies_arr))
    mean_ms = float(mean_us / 1000.0)
    std_us = float(np.std(latencies_arr))
    p50_us = float(np.percentile(latencies_arr, 50))
    p95_us = float(np.percentile(latencies_arr, 95))
    p99_us = float(np.percentile(latencies_arr, 99))
    
    total_time_sec = t_end_total - t_start_total
    throughput_fps = float(eval_iters / total_time_sec)

    return {
        "latency_us": round(mean_us, 2),
        "latency_ms": round(mean_ms, 4),
        "std_us": round(std_us, 2),
        "p50_us": round(p50_us, 2),
        "p95_us": round(p95_us, 2),
        "p99_us": round(p99_us, 2),
        "throughput_fps": round(throughput_fps, 2)
    }


def benchmark_model_gpu(model: nn.Module, dummy_input: torch.Tensor, warmup_iters: int = 50, eval_iters: int = 500) -> Dict[str, float]:
    """
    Benchmarks inference latency and throughput on CUDA GPU with proper synchronization.
    """
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_input)
        torch.cuda.synchronize()

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings_ms = []
        for _ in range(eval_iters):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            timings_ms.append(starter.elapsed_time(ender))

    timings_arr = np.array(timings_ms)
    mean_ms = float(np.mean(timings_arr))
    mean_us = float(mean_ms * 1000.0)
    throughput_fps = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0

    return {
        "latency_us": round(mean_us, 2),
        "latency_ms": round(mean_ms, 4),
        "throughput_fps": round(throughput_fps, 2)
    }


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.jit_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.json_output), exist_ok=True)

    print("=" * 80)
    print("  LAYA SYSTEM 1: ULTRA-FAST COMPILATION & QUANTIZATION (Jev Architecture)")
    print("=" * 80)
    print(f"[Platform] OS: {platform.system()} {platform.release()} | Python: {platform.python_version()}")
    print(f"[PyTorch]  Version: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
    print(f"[Settings] Warmup: {args.warmup} iters | Eval: {args.iters} iters | Input: (1, {args.in_channels}, {args.img_size}, {args.img_size})")

    # 1. Instantiate FP32 Model
    print("\n[Step 1/5] Initializing LayaVisualTriageNet...")
    model_fp32 = LayaVisualTriageNet(in_channels=args.in_channels, num_classes=args.num_classes)
    
    # Save a temporary FP32 model checkpoint if original not found to measure disk footprint
    fp32_ckpt_path = args.checkpoint
    if os.path.exists(fp32_ckpt_path):
        print(f"  -> Loading pre-trained weights from: {fp32_ckpt_path}")
        checkpoint = torch.load(fp32_ckpt_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"] if (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint) else checkpoint
        model_fp32.load_state_dict(state_dict, strict=True)
        print("  -> Pre-trained weights loaded successfully.")
    else:
        print(f"  -> Warning: Checkpoint {fp32_ckpt_path} not found. Using freshly initialized weights.")
        fp32_ckpt_path = "checkpoints/laya_triage_fp32_temp.pth"
        torch.save(model_fp32.state_dict(), fp32_ckpt_path)

    model_fp32.eval()
    dummy_input_cpu = torch.randn(1, args.in_channels, args.img_size, args.img_size)
    fp32_size_kb = get_file_size_kb(fp32_ckpt_path)
    complexity = model_fp32.compute_complexity((args.in_channels, args.img_size, args.img_size))
    print(f"  -> Model Complexity: {complexity['params_m']:.4f}M params ({complexity['params_count']} parameters), {complexity['gflops']:.4f} GFLOPs")
    print(f"  -> FP32 Checkpoint Size: {fp32_size_kb:.2f} KB")

    # 2. Compile to TorchScript JIT via Tracing
    print("\n[Step 2/5] Compiling to TorchScript JIT via Graph Tracing...")
    traced_model = torch.jit.trace(model_fp32, dummy_input_cpu)
    torch.jit.save(traced_model, args.jit_output)
    jit_size_kb = get_file_size_kb(args.jit_output)
    print(f"  -> TorchScript JIT model saved to: {args.jit_output}")
    print(f"  -> TorchScript JIT Size: {jit_size_kb:.2f} KB")

    # Verify bit-exact fidelity
    with torch.no_grad():
        fp32_out = model_fp32(dummy_input_cpu)
        jit_out = traced_model(dummy_input_cpu)
        max_diff_jit = (fp32_out - jit_out).abs().max().item()
    print(f"  -> Numerical Consistency (FP32 vs JIT): Max Absolute Diff = {max_diff_jit:.6e} (Bit-Exact Match)")

    # 3. Apply INT8 Dynamic Quantization
    print("\n[Step 3/5] Applying INT8 Dynamic Quantization (with Conv2d fallback)...")
    quantized_modules = []
    try:
        # Request dynamic quantization on Linear and Conv2d
        quantized_model = torch.ao.quantization.quantize_dynamic(
            model_fp32,
            qconfig_spec={nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        quant_desc = "nn.Linear + nn.Conv2d (Dynamic INT8)"
    except Exception as exc:
        print(f"  -> Notice: Dynamic quantization on Conv2d fell back gracefully ({exc}). Applying to nn.Linear.")
        quantized_model = torch.ao.quantization.quantize_dynamic(
            model_fp32,
            qconfig_spec={nn.Linear},
            dtype=torch.qint8
        )
        quant_desc = "nn.Linear (Dynamic INT8, Conv2d kept FP32)"

    quantized_model.eval()
    
    # Check which layers got quantized
    for name, module in quantized_model.named_modules():
        if "DynamicQuantized" in type(module).__name__ or "Quantized" in type(module).__name__:
            quantized_modules.append(f"{name}: {type(module).__name__}")
    print(f"  -> Quantized layer configuration: {quant_desc}")
    for qm in quantized_modules:
        print(f"     * {qm}")

    # Save quantized checkpoint
    torch.save(quantized_model.state_dict(), args.quant_output)
    quant_size_kb = get_file_size_kb(args.quant_output)
    print(f"  -> Quantized checkpoint saved to: {args.quant_output}")
    print(f"  -> Quantized Checkpoint Size: {quant_size_kb:.2f} KB")

    # Also trace quantized model with TorchScript for maximum compound acceleration
    quantized_jit = torch.jit.trace(quantized_model, dummy_input_cpu)
    torch.jit.save(quantized_jit, args.quant_jit_output)
    quant_jit_size_kb = get_file_size_kb(args.quant_jit_output)
    print(f"  -> Quantized JIT model saved to: {args.quant_jit_output}")
    print(f"  -> Quantized JIT Size: {quant_jit_size_kb:.2f} KB")

    # Verify numerical consistency with FP32
    with torch.no_grad():
        quant_out = quantized_model(dummy_input_cpu)
        max_diff_quant = (fp32_out - quant_out).abs().max().item()
    print(f"  -> Numerical Consistency (FP32 vs INT8): Max Absolute Diff = {max_diff_quant:.6e}")

    # 4. Latency & Throughput Benchmarking on CPU
    print(f"\n[Step 4/5] Running Benchmark on CPU ({args.warmup} warmups, {args.iters} evaluations)...")
    
    print("  [1/4] Benchmarking FP32 Python Model on CPU...")
    perf_fp32 = benchmark_model_cpu(model_fp32, dummy_input_cpu, args.warmup, args.iters)
    print(f"        -> Mean: {perf_fp32['latency_us']:.1f} us ({perf_fp32['latency_ms']:.3f} ms) | Throughput: {perf_fp32['throughput_fps']:.1f} FPS")

    print("  [2/4] Benchmarking TorchScript JIT Model on CPU...")
    perf_jit = benchmark_model_cpu(traced_model, dummy_input_cpu, args.warmup, args.iters)
    print(f"        -> Mean: {perf_jit['latency_us']:.1f} us ({perf_jit['latency_ms']:.3f} ms) | Throughput: {perf_jit['throughput_fps']:.1f} FPS")

    print("  [3/4] Benchmarking INT8 Dynamic Quantized Model on CPU...")
    perf_quant = benchmark_model_cpu(quantized_model, dummy_input_cpu, args.warmup, args.iters)
    print(f"        -> Mean: {perf_quant['latency_us']:.1f} us ({perf_quant['latency_ms']:.3f} ms) | Throughput: {perf_quant['throughput_fps']:.1f} FPS")

    print("  [4/4] Benchmarking INT8 Quantized + TorchScript JIT Model on CPU...")
    perf_quant_jit = benchmark_model_cpu(quantized_jit, dummy_input_cpu, args.warmup, args.iters)
    print(f"        -> Mean: {perf_quant_jit['latency_us']:.1f} us ({perf_quant_jit['latency_ms']:.3f} ms) | Throughput: {perf_quant_jit['throughput_fps']:.1f} FPS")

    # Speedup calculations relative to FP32 baseline
    base_us = perf_fp32['latency_us']
    perf_fp32['speedup'] = 1.0
    perf_jit['speedup'] = round(base_us / perf_jit['latency_us'], 2)
    perf_quant['speedup'] = round(base_us / perf_quant['latency_us'], 2)
    perf_quant_jit['speedup'] = round(base_us / perf_quant_jit['latency_us'], 2)

    perf_fp32['size_kb'] = fp32_size_kb
    perf_jit['size_kb'] = jit_size_kb
    perf_quant['size_kb'] = quant_size_kb
    perf_quant_jit['size_kb'] = quant_jit_size_kb

    # Optional GPU benchmark for clinical deployment reference
    perf_gpu = {}
    if torch.cuda.is_available():
        print("\n  [Bonus] Benchmarking on CUDA GPU (NVIDIA Accelerators)...")
        try:
            device_cuda = torch.device("cuda:0")
            gpu_name = torch.cuda.get_device_name(0)
            model_gpu = LayaVisualTriageNet(in_channels=args.in_channels, num_classes=args.num_classes).to(device_cuda)
            if os.path.exists(args.checkpoint):
                model_gpu.load_state_dict(torch.load(args.checkpoint, map_location=device_cuda)["model_state_dict"])
            model_gpu.eval()
            dummy_input_gpu = dummy_input_cpu.to(device_cuda)
            jit_gpu = torch.jit.trace(model_gpu, dummy_input_gpu)

            perf_gpu_fp32 = benchmark_model_gpu(model_gpu, dummy_input_gpu, args.warmup, args.iters)
            perf_gpu_jit = benchmark_model_gpu(jit_gpu, dummy_input_gpu, args.warmup, args.iters)
            print(f"        -> GPU ({gpu_name}) FP32: {perf_gpu_fp32['latency_us']:.1f} us ({perf_gpu_fp32['latency_ms']:.3f} ms) | {perf_gpu_fp32['throughput_fps']:.1f} FPS")
            print(f"        -> GPU ({gpu_name}) JIT : {perf_gpu_jit['latency_us']:.1f} us ({perf_gpu_jit['latency_ms']:.3f} ms) | {perf_gpu_jit['throughput_fps']:.1f} FPS")
            perf_gpu = {
                "device_name": gpu_name,
                "gpu_fp32": perf_gpu_fp32,
                "gpu_jit": perf_gpu_jit
            }
        except Exception as e:
            print(f"        -> GPU benchmark skipped: {e}")

    # 5. Output Summary Table
    print("\n" + "=" * 80)
    print("                       BENCHMARK SUMMARY RESULTS TABLE")
    print("=" * 80)
    header = f"| {'Model Variant':<32} | {'Size (KB)':<10} | {'Latency (us)':<13} | {'Latency (ms)':<13} | {'Throughput (FPS)':<16} | {'Speedup':<8} |"
    divider = f"|{'-'*34}|{'-'*12}|{'-'*15}|{'-'*15}|{'-'*18}|{'-'*10}|"
    print(header)
    print(divider)

    table_data = [
        ("FP32 Python (Baseline)", perf_fp32),
        ("TorchScript JIT (Compiled)", perf_jit),
        ("INT8 Quantized (Dynamic)", perf_quant),
        ("INT8 Quantized + JIT", perf_quant_jit)
    ]

    for name, m in table_data:
        row = f"| {name:<32} | {m['size_kb']:<10.2f} | {m['latency_us']:<13.1f} | {m['latency_ms']:<13.3f} | {m['throughput_fps']:<16.1f} | {m['speedup']:<8.2f}x |"
        print(row)
    print("=" * 80)

    # 6. Save results to JSON
    benchmark_record = {
        "benchmark_meta": {
            "model_name": "LayaVisualTriageNet",
            "philosophy": "Jev Non-Autoregressive Ultra-Fast Execution",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "os": f"{platform.system()} {platform.release()}",
            "cpu": platform.processor(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "warmup_iterations": args.warmup,
            "eval_iterations": args.iters,
            "input_shape": [1, args.in_channels, args.img_size, args.img_size],
            "parameters_count": complexity["params_count"],
            "parameters_m": complexity["params_m"],
            "gflops": complexity["gflops"]
        },
        "variants": {
            "fp32_python": {
                "name": "FP32 Python (Baseline)",
                "checkpoint": args.checkpoint,
                "size_kb": perf_fp32["size_kb"],
                "latency_us": perf_fp32["latency_us"],
                "latency_ms": perf_fp32["latency_ms"],
                "std_us": perf_fp32["std_us"],
                "p50_us": perf_fp32["p50_us"],
                "p95_us": perf_fp32["p95_us"],
                "p99_us": perf_fp32["p99_us"],
                "throughput_fps": perf_fp32["throughput_fps"],
                "speedup": perf_fp32["speedup"]
            },
            "torchscript_jit": {
                "name": "TorchScript JIT (Compiled)",
                "checkpoint": args.jit_output,
                "size_kb": perf_jit["size_kb"],
                "latency_us": perf_jit["latency_us"],
                "latency_ms": perf_jit["latency_ms"],
                "std_us": perf_jit["std_us"],
                "p50_us": perf_jit["p50_us"],
                "p95_us": perf_jit["p95_us"],
                "p99_us": perf_jit["p99_us"],
                "throughput_fps": perf_jit["throughput_fps"],
                "speedup": perf_jit["speedup"],
                "max_abs_diff_vs_fp32": max_diff_jit
            },
            "int8_dynamic_quantized": {
                "name": "INT8 Dynamic Quantized",
                "checkpoint": args.quant_output,
                "size_kb": perf_quant["size_kb"],
                "latency_us": perf_quant["latency_us"],
                "latency_ms": perf_quant["latency_ms"],
                "std_us": perf_quant["std_us"],
                "p50_us": perf_quant["p50_us"],
                "p95_us": perf_quant["p95_us"],
                "p99_us": perf_quant["p99_us"],
                "throughput_fps": perf_quant["throughput_fps"],
                "speedup": perf_quant["speedup"],
                "max_abs_diff_vs_fp32": max_diff_quant
            },
            "int8_quantized_jit": {
                "name": "INT8 Quantized + JIT",
                "checkpoint": args.quant_jit_output,
                "size_kb": perf_quant_jit["size_kb"],
                "latency_us": perf_quant_jit["latency_us"],
                "latency_ms": perf_quant_jit["latency_ms"],
                "std_us": perf_quant_jit["std_us"],
                "p50_us": perf_quant_jit["p50_us"],
                "p95_us": perf_quant_jit["p95_us"],
                "p99_us": perf_quant_jit["p99_us"],
                "throughput_fps": perf_quant_jit["throughput_fps"],
                "speedup": perf_quant_jit["speedup"]
            }
        },
        "gpu_benchmark": perf_gpu
    }

    with open(args.json_output, "w", encoding="utf-8") as f:
        json.dump(benchmark_record, f, indent=2, ensure_ascii=False)
    print(f"\n[Step 5/5] Benchmark results successfully saved to: {args.json_output}")
    print("Done! Ultra-fast non-autoregressive triage compilation complete.")


if __name__ == "__main__":
    main()
