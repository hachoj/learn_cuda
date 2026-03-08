import torch
import triton
from torch.utils.cpp_extension import load

from src.add.add_triton import DEVICE  # pyrefly: ignore

cuda_matmul = load(
    name="cuda_matmul",
    sources=[
        "/home/chojnowski.h/weishao/chojnowski.h/CUDA/src/matmul/matmul_ext.cpp",
        "/home/chojnowski.h/weishao/chojnowski.h/CUDA/src/matmul/naive_matmul.cu",
    ],
    verbose=False,
)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(4, 12)],
        x_log=True,
        line_arg="provider",
        line_vals=["torch", "cuda_naive", "cuda_tiled"],
        line_names=["PyTorch", "Naive", "Tiled"],
        styles=[("green", "-"), ("blue", "-"), ("red", "-")],
        ylabel="TFLOP/s",
        plot_name="matrix-multiplication-performance-flops",
        args={},
    )
)
def benchmark(size, provider):
    x = torch.rand((size, size), device=DEVICE, dtype=torch.float32)
    y = torch.rand((size, size), device=DEVICE, dtype=torch.float32)
    # 2 * size^3 FLOPs: size^2 output elements, each doing size muls + size adds
    tflops = lambda ms: 2 * size**3 * 1e-12 / (ms * 1e-3)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: x @ y, quantiles=quantiles
        )
    elif provider == "cuda_naive":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: cuda_matmul.matmul(x, y),
            quantiles=quantiles,
        )
    elif provider == "cuda_tiled":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: cuda_matmul.tiled_matmul(x, y), quantiles=quantiles
        )

    return tflops(ms), tflops(max_ms), tflops(min_ms)

benchmark.run(print_data=True, show_plots=False, save_path="bench/out/")