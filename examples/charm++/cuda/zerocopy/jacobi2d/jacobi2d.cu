#include "hapi.h"

#define TILE_SIZE 16
#define DIVIDEBY5 0.2

__global__ void initKernel(double* temperature, int block_size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < block_size + 2 && j < block_size + 2) {
    temperature[(block_size + 2) * j + i] = 0.0;
  }
}

__global__ void leftBoundaryKernel(double* temperature, int block_size) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < block_size) {
    temperature[(block_size + 2) * (1 + j)] = 1.0;
  }
}

__global__ void rightBoundaryKernel(double* temperature, int block_size) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < block_size) {
    temperature[(block_size + 2) * (1 + j) + (block_size + 1)] = 1.0;
  }
}

__global__ void topBoundaryKernel(double* temperature, int block_size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < block_size) {
    temperature[1 + i] = 1.0;
  }
}

__global__ void bottomBoundaryKernel(double* temperature, int block_size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < block_size) {
    temperature[(block_size + 2) * (block_size + 1) + (1 + i)] = 1.0;
  }
}

__global__ void leftPackingKernel(double* temperature, double* ghost, int block_size) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < block_size) {
    ghost[j] = temperature[(block_size + 2) * (1 + j) + 1];
  }
}

__global__ void rightPackingKernel(double* temperature, double* ghost, int block_size) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < block_size) {
    ghost[j] = temperature[(block_size + 2) * (1 + j) + (block_size)];
  }
}

__global__ void leftUnpackingKernel(double* temperature, double* ghost, int block_size) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < block_size) {
    temperature[(block_size + 2) * (1 + j) + 1] = ghost[j];
  }
}

__global__ void rightUnpackingKernel(double* temperature, double* ghost, int block_size) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < block_size) {
    temperature[(block_size + 2) * (1 + j) + (block_size)] = ghost[j];
  }
}

__global__ void jacobiKernel(double* temperature, double* new_temperature, int block_size) {
  int i = (blockDim.x * blockIdx.x + threadIdx.x) + 1;
  int j = (blockDim.y * blockIdx.y + threadIdx.y) + 1;

  if (i <= block_size && j <= block_size) {
    new_temperature[j * (block_size + 2) + i] =
        (temperature[j * (block_size + 2) + (i - 1)] +
         temperature[j * (block_size + 2) + (i + 1)] +
         temperature[(j - 1) * (block_size + 2) + i] +
         temperature[(j + 1) * (block_size + 2) + i] +
         temperature[j * (block_size + 2) + i]) *
        DIVIDEBY5;
  }
}

void invokeInitKernel(double* d_temperature, int block_size, cudaStream_t stream) {
  dim3 block_dim(TILE_SIZE, TILE_SIZE);
  dim3 grid_dim(((block_size + 2) + (block_dim.x - 1)) / block_dim.x,
      ((block_size + 2) + (block_dim.y - 1)) / block_dim.y);

  initKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature, block_size);
  hapiCheck(cudaPeekAtLastError());
}

void invokeBoundaryKernels(double* d_temperature, int block_size, bool left_bound,
    bool right_bound, bool top_bound, bool bottom_bound, cudaStream_t stream) {
  dim3 block_dim(TILE_SIZE * TILE_SIZE);
  dim3 grid_dim((block_size + (block_dim.x - 1)) / block_dim.x);

  if (left_bound) {
    leftBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature, block_size);
  }
  if (right_bound) {
    rightBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature, block_size);
  }
  if (top_bound) {
    topBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature, block_size);
  }
  if (bottom_bound) {
    bottomBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature, block_size);
  }
  hapiCheck(cudaPeekAtLastError());
}

void invokePackingKernels(double* d_temperature, double* d_left_ghost,
    double* d_right_ghost, bool left_bound, bool right_bound, int block_size,
    cudaStream_t stream) {
  dim3 block_dim(TILE_SIZE * TILE_SIZE);
  dim3 grid_dim((block_size + (block_dim.x - 1)) / block_dim.x);
  if (!left_bound) {
    leftPackingKernel<<<block_dim, grid_dim, 0, stream>>>(d_temperature, d_left_ghost, block_size);
  }
  if (!right_bound) {
    rightPackingKernel<<<block_dim, grid_dim, 0, stream>>>(d_temperature, d_right_ghost, block_size);
  }
  hapiCheck(cudaPeekAtLastError());
}

void invokeUnpackingKernel(double* d_temperature, double* d_ghost, bool is_left,
    int block_size, cudaStream_t stream) {
  dim3 block_dim(TILE_SIZE * TILE_SIZE);
  dim3 grid_dim((block_size + (block_dim.x - 1)) / block_dim.x);
  if (is_left) {
    leftUnpackingKernel<<<block_dim, grid_dim, 0, stream>>>(d_temperature, d_ghost, block_size);
  }
  else {
    rightUnpackingKernel<<<block_dim, grid_dim, 0, stream>>>(d_temperature, d_ghost, block_size);
  }
  hapiCheck(cudaPeekAtLastError());
}

void invokeJacobiKernel(double* d_temperature, double* d_new_temperature, int block_size,
    cudaStream_t stream) {
  dim3 block_dim(TILE_SIZE, TILE_SIZE);
  dim3 grid_dim((block_size + (block_dim.x - 1)) / block_dim.x,
      (block_size + (block_dim.y - 1)) / block_dim.y);

  jacobiKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature, d_new_temperature, block_size);
  hapiCheck(cudaPeekAtLastError());
}
