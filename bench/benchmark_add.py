import torch
import triton
from torch.utils.cpp_extension import load

from src.add.add_triton import DEVICE  # pyrefly: ignore
from src.add.add_triton import add as triton_add  # pyrefly: ignore

cuda_add = load(
    name="cuda_add",
    sources=[
        "/home/chojnowski.h/weishao/chojnowski.h/CUDA/src/add/add_ext.cpp",
        "/home/chojnowski.h/weishao/chojnowski.h/CUDA/src/add/add.cu",
    ],
    verbose=False,
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(12, 26)],
        x_log=True,
        # x_log=False,
        line_arg="provider",
        line_vals=["torch", "triton", "cuda_ext"],
        line_names=["PyTorch", "Triton", "CUDA Extension"],
        styles=[("green", "-"), ("blue", "-"), ("red", "-")],
        ylabel="GB/s",
        plot_name="vector-add-performance",
        args={},
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)

    # Vector add touches 3 arrays (read x, read y, write out).
    # GB/s = bytes moved / time in seconds / 1e9
    # element_size() returns bytes per element — 4 for float32.
    gbps = lambda ms: 3 * size * x.element_size() * 1e-9 / (ms * 1e-3)

    quantiles = [0.5, 0.2, 0.8]  # median, lower band, upper band

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(  # pyrefly:ignore
            lambda: x + y, quantiles=quantiles
        )
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(  # pyrefly:ignore
            lambda: triton_add(x, y), quantiles=quantiles
        )
    elif provider == "cuda_ext":
        ms, min_ms, max_ms = triton.testing.do_bench(  # pyrefly:ignore
            lambda: cuda_add.add(x, y), quantiles=quantiles  # pyrefly:ignore
        )

    # Return (median, max, min) — perf_report uses max/min for the shaded band.
    # Note the order: max_ms gives the SLOWEST time, which is the LOWEST GB/s.
    return gbps(ms), gbps(max_ms), gbps(min_ms)  # pyrefly:ignore


benchmark.run(print_data=True, show_plots=False, save_path="bench/out/")
