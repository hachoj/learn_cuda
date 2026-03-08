#include <torch/extension.h>

void launchMatmul(float* M, float* N, float* P, int m, int n, int k);
void launchTiledMatmul(float* M, float* N, float* P, int m, int n, int k);


torch::Tensor matmul(torch::Tensor M, torch::Tensor N) {

    int m = M.size(0);
    int k = M.size(1);
    int n = N.size(1);

    torch::Tensor P = torch::empty({m, n}, M.options());

    float* M_ptr = M.data_ptr<float>();
    float* N_ptr = N.data_ptr<float>();
    float* P_ptr = P.data_ptr<float>();

    launchMatmul(M_ptr, N_ptr, P_ptr, m, n, k);

    return P;
}

torch::Tensor tiled_matmul(torch::Tensor M, torch::Tensor N) {

    int m = M.size(0);
    int k = M.size(1);
    int n = N.size(1);

    torch::Tensor P = torch::empty({m, n}, M.options());

    float* M_ptr = M.data_ptr<float>();
    float* N_ptr = N.data_ptr<float>();
    float* P_ptr = P.data_ptr<float>();

    launchTiledMatmul(M_ptr, N_ptr, P_ptr, m, n, k);

    return P;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul, "Matrix multiplication (custom CUDA kernel)");
    m.def("tiled_matmul", &tiled_matmul, "Tiled matrix multiplication (custom CUDA kernel)");
}