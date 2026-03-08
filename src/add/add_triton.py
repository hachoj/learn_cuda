import torch
import triton
import triton.language as tl

# triton.runtime.driver.active.get_active_torch_device() asks the Triton runtime
# which GPU device PyTorch is currently using and returns it as a torch.device.
# This is more robust than hardcoding device='cuda' because it respects
# multi-GPU setups and whatever device PyTorch has selected.
DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _add_kernel(
    x_ptr,                      # pointer to first input vector
    y_ptr,                      # pointer to second input vector
    out_ptr,                    # pointer to output vector
    n_elements,                 # total number of elements
    BLOCK_SIZE: tl.constexpr,  # how many elements this program instance handles
):
    # this is basically just blockIdx.x
    pid = tl.program_id(axis=0)
    # this is basically just blockIdx.x * blockDim.x
    block_start = pid * BLOCK_SIZE
    # in triton you don't have thread access
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # this is equivalent to the if (i < n) in the CUDA kernel.
    # It just makes it so you don't try to load or store out-of-bounds memory
    mask = offsets < n_elements

    # tells tl to load the memory for this specific kernel
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = out.numel()

    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)

    # grid here is equivalent to the <<<gridDim,_>>> part of launching a CUDA kernel
    # however, you don't pass in the equivalent numThreads, that is instead handled
    # arbitraryily by the tl kernel it seems
    _add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)  # pyrefly: ignore
    return out


if __name__ == '__main__':
    size = 98432
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)

    ref = x + y
    out = add(x, y)

    max_diff = torch.max(torch.abs(ref - out)).item()
    print(f'Max difference vs torch: {max_diff:.2e}')
    assert max_diff < 1e-5, 'Results do not match!'
    print('Correctness check passed.')
