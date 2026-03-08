#include <iostream>
#include <math.h>
 
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *a, float *b, float *c)
{
  // int index = blockIdx.x * blockDim.x + threadIdx.x;
  // int stride = blockDim.x * gridDim.x;
  // if (index == 314322) {
  //   printf("blockIdx: %i\n", blockIdx.x);
  //   printf("blockDim: %i\n", blockDim.x);
  //   printf("threadIdx: %i\n", threadIdx.x);
  //   printf("gridDim: %i\n", gridDim.x);
  // }
  // for (int i = index; i < n; i += stride)
  //   y[i] = x[i] + y[i];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = a[i] + b[i];
}
 
int main(void)
{
  int N = 1<<20;

  float *a, *b, *c;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&a, N*sizeof(float));
  cudaMallocManaged(&b, N*sizeof(float));
  cudaMallocManaged(&c, N*sizeof(float));

  // initialize a and b arrays on the host
  for (int i = 0; i < N; i++) {
    a[i] = 1.0f;
    b[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  // the first number is the number of thread blocks
  // the second number is the number of threads in a thread block

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;  // this is ceil div

  // Prefetch the a and b arrays to the GPU
  cudaMemPrefetchAsync(a, N*sizeof(float), 0, 0);
  cudaMemPrefetchAsync(b, N*sizeof(float), 0, 0);
  cudaMemPrefetchAsync(c, N*sizeof(float), 0, 0);

  add<<<numBlocks, blockSize>>>(N, a, b, c);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = fmax(maxError, fabs(c[i]-3.0f));
  }
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  return 0;
}