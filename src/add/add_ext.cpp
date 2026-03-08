// this includes both the torch extension as well as the extension called
// pybind, which let's C++ and Python work interoperably
#include <torch/extension.h>

// tell the C++ compiler that this function exists, but you don't want/need to 
// actually give the function yet.
// Also, because CUDA kernels are invalid cpp syntax, that's why you need this launcher
// function
void launchAdd(float* x, float* y, float* out, int n);

// same as the triton function that basically enables the use of the 
// triton kernel in PyTorch.
torch::Tensor add(torch::Tensor x, torch::Tensor y) {

    // since x is on device, out is implcitely on device
    torch::Tensor out = torch::empty_like(x);

    // you know what this does
    int n = x.numel();

    // already exists on the GPU.
    // this is how you extract the actual data pointer from a torch tensor
    // since that's what are adder kernel actually uses, a device memory pointer
    // as input.
    float* x_ptr   = x.data_ptr<float>();
    float* y_ptr   = y.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    // calls our launcher which in tern calls are cuda function.
    launchAdd(x_ptr, y_ptr, out_ptr, n);

    return out;
}

// Register our C++ function with Python using pybind11.
//
// PYBIND11_MODULE declares a Python module. When Python does `import cuda_add`,
// this block runs and populates the module.
//
// TORCH_EXTENSION_NAME is a macro that gets filled in at compile time with
// whatever name you pass to torch.utils.cpp_extension.load(name=...) in Python.
//
// m.def(python_name, &cpp_function, docstring) registers one function.
// You can call m.def() multiple times to expose multiple functions.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "Element-wise vector add (custom CUDA kernel)");
}
