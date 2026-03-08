#include <iostream>
#include <math.h>
 
__global__
void addKernel(float *A, float *B, float *C, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

void add(float *A_h, float *B_h, float *C_h, int N)
{
  int size = N * sizeof(float);
  float *A_d, *B_d, *C_d;

  // because these pointers actually point to locations on 
  // the device, they should not be dereferenced on the host
  // as that will cause undefined behavior and causes errors11

  // this moves the pointer of x and y to point to the 
  // place in memory on the device
  cudaMalloc((void**)&A_d, N*sizeof(float));
  cudaMalloc((void**)&B_d, N*sizeof(float));
  cudaMalloc((void**)&C_d, N*sizeof(float));

  cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

  dim3 numThreads(256, 1, 1);
  dim3 numBlocks(ceil(N/256), 1, 1);
  addKernel<<<numBlocks, numThreads>>>(A_d, B_d, C_d, N);

  cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

  // Free memory
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}
 
int main(void)
{
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

  add(A_h, B_h, C_h, N);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = fmax(maxError, fabs(C_h[i]-3.0f));
  }
  std::cout << "Max error: " << maxError << std::endl;
  return 0;
}