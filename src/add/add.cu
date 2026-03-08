#include <iostream>
#include <math.h>
 
// Scalar version (v1): one float loaded per thread per instruction.
__global__
void addKernel(float* x, float* y, float* out, int N) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i + 3 < N) {
    // the * before reinterpret_cast is because reinterpret cast
    // returns a pointer so I must dereference it.
    float4 x_vals = *reinterpret_cast<float4*>(&x[i]);
    float4 y_vals = *reinterpret_cast<float4*>(&y[i]);
    float4 out_vals;
    out_vals.x = x_vals.x + y_vals.x;
    out_vals.y = x_vals.y + y_vals.y;
    out_vals.z = x_vals.z + y_vals.z;
    out_vals.w = x_vals.w + y_vals.w;
    *reinterpret_cast<float4*>(&out[i]) = out_vals;
  }
  if (i < N && i + 3 >= N) {
    for (int j = i - 3; j < N; ++j) {
      out[j] = x[j] + y[j];
    }
  }
}

// this function assumes that the
void launchAdd(float* x, float* y, float* out, int n) {
  int blockSize = 1024;
  int numBlocks = ceil((n/4.0) / 1024.0);

  addKernel<<<numBlocks, blockSize>>>(x, y, out, n);
}

// correctness check
int main() {
  int N = 1<<20;

  float *A_h, *B_h, *C_h;
 
  A_h = (float*)malloc(N * sizeof(float));
  B_h = (float*)malloc(N * sizeof(float));
  C_h = (float*)malloc(N * sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    A_h[i] = 1.0f;
    B_h[i] = 2.0f;
  }

  float *A_d, *B_d, *C_d;
  cudaMalloc((void**)&A_d, N*sizeof(float));
  cudaMalloc((void**)&B_d, N*sizeof(float));
  cudaMalloc((void**)&C_d, N*sizeof(float));

  cudaMemcpy(A_d, A_h, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, N*sizeof(float), cudaMemcpyHostToDevice);

  launchAdd(A_d, B_d, C_d, N);

  cudaMemcpy(C_h, C_d, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = fmax(maxError, fabs(C_h[i]-3.0f));
  }
  std::cout << "Max error: " << maxError << std::endl;
  return 0;
}