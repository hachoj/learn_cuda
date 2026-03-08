#include <iostream>
#include <math.h>

#define TILE_SIZE 32


__global__ 
void matmulKernel(float* M, float* N, float* P, int m, int n, int k) {
  // these indices are for the output
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  // check condition within image bounds
  if ((row < m) && (col < n)) {
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
      sum += M[row*k + i] * N[n*i + col];
    }
    P[row*n + col] = sum;
  }
}

void launchMatmul(float* M, float* N, float* P, int m, int n, int k) {
  dim3 numThreads(TILE_SIZE, TILE_SIZE, 1);
  dim3 numBlocks(ceil(m/(float)TILE_SIZE), ceil(n/(float)TILE_SIZE), 1);
  
  matmulKernel<<<numBlocks, numThreads>>>(M, N, P, m, n, k);
}

__global__
void tiledMatmulKernel(float* M, float* N, float* P, int m, int n, int k) {
  // define the shared memory on SM which holds the current tile values
  __shared__ float Mds[TILE_SIZE][TILE_SIZE];
  __shared__ float Nds[TILE_SIZE][TILE_SIZE];

  // these indices are for the output
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = TILE_SIZE * blockIdx.y + ty;
  int col = TILE_SIZE * blockIdx.x + tx;

  float Pvalue = 0.0;
  for (int ph = 0; ph < (k + TILE_SIZE - 1) / TILE_SIZE; ++ph) {
    if ((row < m) && ((ph*TILE_SIZE + threadIdx.x) < k))
      Mds[ty][tx] = M[row*k + ph*TILE_SIZE + threadIdx.x];
    else Mds[ty][tx] = 0.0f;
    if (((ph*TILE_SIZE + threadIdx.y) < k) && (col < n))
      Nds[ty][tx] = N[(ph*TILE_SIZE + threadIdx.y)*n + col];
    else Nds[ty][tx] = 0.0f;
    __syncthreads();

    for (int i = 0; i < TILE_SIZE; ++i) {
      Pvalue += Mds[ty][i] * Nds[i][tx];
    }
    __syncthreads();
  }
  if ((row < m) && (col < n))
    P[row * n + col] = Pvalue;
}

void launchTiledMatmul(float* M, float* N, float* P, int m, int n, int k) {
  dim3 numThreads(TILE_SIZE, TILE_SIZE, 1);
  dim3 numBlocks(ceil(m/(float)TILE_SIZE), ceil(n/(float)TILE_SIZE), 1);
  
  tiledMatmulKernel<<<numBlocks, numThreads>>>(M, N, P, m, n, k);
}

int main(void) {
  int m = 1<<10;
  int k = 1<<10;
  int n = 1<<10;
  m += 4;
  k += 27;
  n += 15;
  // int m = 8;
  // int k = 9;
  // int n = 11;

  float *M_h, *N_h, *P_h;
  
  M_h = (float*)malloc((m * k)*sizeof(float));
  N_h = (float*)malloc((k * n)*sizeof(float));
  P_h = (float*)malloc((m * n)*sizeof(float));

  // populate the inputs arrays
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      M_h[i*k + j] = 1.0f;
    }
  }
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      N_h[i*n + j] = 2.0f;
    }
  }

  float *M_d, *N_d, *P_d;

  cudaMalloc((void**)&M_d, (m * k)*sizeof(float));
  cudaMalloc((void**)&N_d, (k * n)*sizeof(float));
  cudaMalloc((void**)&P_d, (m * n)*sizeof(float));

  cudaMemcpy(M_d, M_h, (m * k)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(N_d, N_h, (k * n)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(P_d, P_h, (m * n)*sizeof(float), cudaMemcpyHostToDevice);

  // launchMatmul(M_d, N_d, P_d, m, n, k);
  launchTiledMatmul(M_d, N_d, P_d, m, n, k);

  cudaMemcpy(P_h, P_d, (m * n)*sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(M_d);
  cudaFree(N_d);
  cudaFree(P_d);

  float maxError = 0.0f;
  for (int i = 0; i < m*n; ++i) {
    maxError = fmax(maxError, fabs(P_h[i] - 2.0f*k));
  }
  // if the output is small enough print M, N, and P
  if (n <= 16 && m <= 16) {
    std::cout << "Matrix M:" << std::endl;
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < k; ++j) {
        std::cout << M_h[i*k + j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "Matrix N:" << std::endl;
    for (int i = 0; i < k; ++i) {
      for (int j = 0; j < n; ++j) {
        std::cout << N_h[i*n + j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "Matrix P:" << std::endl;
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        std::cout << P_h[i*n + j] << " ";
      }
      std::cout << std::endl;
    }
  }
  printf("Maximum error: %f", maxError);
}