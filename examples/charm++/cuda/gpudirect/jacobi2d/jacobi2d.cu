#include "hapi.h"
#include "jacobi2d.h"

#define TILE_SIZE 16
#define DIVIDEBY5 0.2

__global__ void initKernel(DataType* temperature, int block_size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < block_size + 2 && j < block_size + 2) {
    temperature[(block_size + 2) * j + i] = 0;
  }
}

__global__ void leftBoundaryKernel(DataType* temperature, int block_size) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < block_size) {
    temperature[(block_size + 2) * (1 + j)] = 1;
  }
}

__global__ void rightBoundaryKernel(DataType* temperature, int block_size) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < block_size) {
    temperature[(block_size + 2) * (1 + j) + (block_size + 1)] = 1;
  }
}

__global__ void topBoundaryKernel(DataType* temperature, int block_size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < block_size) {
    temperature[1 + i] = 1;
  }
}

__global__ void bottomBoundaryKernel(DataType* temperature, int block_size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < block_size) {
    temperature[(block_size + 2) * (block_size + 1) + (1 + i)] = 1;
  }
}

__global__ void leftPackingKernel(DataType* temperature, DataType* ghost, int block_size) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < block_size) {
    ghost[j] = temperature[(block_size + 2) * (1 + j) + 1];
  }
}

__global__ void rightPackingKernel(DataType* temperature, DataType* ghost, int block_size) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < block_size) {
    ghost[j] = temperature[(block_size + 2) * (1 + j) + (block_size)];
  }
}

__global__ void leftUnpackingKernel(DataType* temperature, DataType* ghost, int block_size) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < block_size) {
    temperature[(block_size + 2) * (1 + j) + 1] = ghost[j];
  }
}

__global__ void rightUnpackingKernel(DataType* temperature, DataType* ghost, int block_size) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < block_size) {
    temperature[(block_size + 2) * (1 + j) + (block_size)] = ghost[j];
  }
}

__global__ void jacobiKernel(DataType* temperature, DataType* new_temperature, int block_size) {
  int i = (blockDim.x * blockIdx.x + threadIdx.x) + 1;
  int j = (blockDim.y * blockIdx.y + threadIdx.y) + 1;

  if (i <= block_size && j <= block_size) {
#ifdef TEST_CORRECTNESS
    new_temperature[j * (block_size + 2) + i] =
        (temperature[j * (block_size + 2) + (i - 1)] +
         temperature[j * (block_size + 2) + (i + 1)] +
         temperature[(j - 1) * (block_size + 2) + i] +
         temperature[(j + 1) * (block_size + 2) + i] +
         temperature[j * (block_size + 2) + i]) % 100000;
#else
    new_temperature[j * (block_size + 2) + i] =
        (temperature[j * (block_size + 2) + (i - 1)] +
         temperature[j * (block_size + 2) + (i + 1)] +
         temperature[(j - 1) * (block_size + 2) + i] +
         temperature[(j + 1) * (block_size + 2) + i] +
         temperature[j * (block_size + 2) + i]) * DIVIDEBY5;
#endif
  }
}

void invokeInitKernel(DataType* d_temperature, int block_size, cudaStream_t stream) {
  dim3 block_dim(TILE_SIZE, TILE_SIZE);
  dim3 grid_dim(((block_size + 2) + (block_dim.x - 1)) / block_dim.x,
      ((block_size + 2) + (block_dim.y - 1)) / block_dim.y);

  initKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature, block_size);
  hapiCheck(cudaPeekAtLastError());
}

void invokeBoundaryKernels(DataType* d_temperature, int block_size, bool left_bound,
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

void invokePackingKernels(DataType* d_temperature, DataType* d_left_ghost,
    DataType* d_right_ghost, bool left_bound, bool right_bound, int block_size,
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

void invokeUnpackingKernel(DataType* d_temperature, DataType* d_ghost, bool is_left,
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

void invokeJacobiKernel(DataType* d_temperature, DataType* d_new_temperature, int block_size,
    cudaStream_t stream) {
  dim3 block_dim(TILE_SIZE, TILE_SIZE);
  dim3 grid_dim((block_size + (block_dim.x - 1)) / block_dim.x,
      (block_size + (block_dim.y - 1)) / block_dim.y);

  jacobiKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature, d_new_temperature, block_size);
  hapiCheck(cudaPeekAtLastError());
}
