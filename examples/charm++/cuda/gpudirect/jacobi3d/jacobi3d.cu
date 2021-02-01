#include "hapi.h"
#include "jacobi3d.h"

#define TILE_SIZE_3D 8
#define TILE_SIZE_2D 16
#define DIVIDEBY7 0.142857

__global__ void initKernel(DataType* temperature, int block_width,
    int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int j = blockDim.y*blockIdx.y+threadIdx.y;
  int k = blockDim.z*blockIdx.z+threadIdx.z;
  if (i < block_width+2 && j < block_height+2 && k < block_depth+2) {
    temperature[IDX(i,j,k)] = 0;
  }
}

__global__ void ghostInitKernel(DataType* ghost, int ghost_count) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  if (i < ghost_count) {
    ghost[i] = 0;
  }
}

__global__ void leftBoundaryKernel(DataType* temperature, int block_width,
    int block_height, int block_depth) {
  int j = blockDim.x*blockIdx.x+threadIdx.x;
  int k = blockDim.y*blockIdx.y+threadIdx.y;
  if (j < block_height && k < block_depth) {
    temperature[IDX(0,1+j,1+k)] = 1;
  }
}

__global__ void rightBoundaryKernel(DataType* temperature, int block_width,
    int block_height, int block_depth) {
  int j = blockDim.x*blockIdx.x+threadIdx.x;
  int k = blockDim.y*blockIdx.y+threadIdx.y;
  if (j < block_height && k < block_depth) {
    temperature[IDX(block_width+1,1+j,1+k)] = 1;
  }
}

__global__ void topBoundaryKernel(DataType* temperature, int block_width,
    int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int k = blockDim.y*blockIdx.y+threadIdx.y;
  if (i < block_width && k < block_depth) {
    temperature[IDX(1+i,0,1+k)] = 1;
  }
}

__global__ void bottomBoundaryKernel(DataType* temperature, int block_width,
    int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int k = blockDim.y*blockIdx.y+threadIdx.y;
  if (i < block_width && k < block_depth) {
    temperature[IDX(1+i,block_height+1,1+k)] = 1;
  }
}

__global__ void frontBoundaryKernel(DataType* temperature, int block_width,
    int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int j = blockDim.y*blockIdx.y+threadIdx.y;
  if (i < block_width && j < block_height) {
    temperature[IDX(1+i,1+j,0)] = 1;
  }
}

__global__ void backBoundaryKernel(DataType* temperature, int block_width,
    int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int j = blockDim.y*blockIdx.y+threadIdx.y;
  if (i < block_width && j < block_height) {
    temperature[IDX(1+i,1+j,block_depth+1)] = 1;
  }
}

__global__ void jacobiKernel(DataType* temperature, DataType* new_temperature,
    int block_width, int block_height, int block_depth) {
  int i = (blockDim.x*blockIdx.x+threadIdx.x)+1;
  int j = (blockDim.y*blockIdx.y+threadIdx.y)+1;
  int k = (blockDim.z*blockIdx.z+threadIdx.z)+1;

  if (i <= block_width && j <= block_height && k <= block_depth) {
#ifdef TEST_CORRECTNESS
    new_temperature[IDX(i,j,k)] = (temperature[IDX(i,j,k)] +
      temperature[IDX(i-1,j,k)] + temperature[IDX(i+1,j,k)] +
      temperature[IDX(i,j-1,k)] + temperature[IDX(i,j+1,k)] +
      temperature[IDX(i,j,k-1)] + temperature[IDX(i,j,k+1)]) % 10000;
#else
    new_temperature[IDX(i,j,k)] = (temperature[IDX(i,j,k)] +
      temperature[IDX(i-1,j,k)] + temperature[IDX(i+1,j,k)] +
      temperature[IDX(i,j-1,k)] + temperature[IDX(i,j+1,k)] +
      temperature[IDX(i,j,k-1)] + temperature[IDX(i,j,k+1)]) * DIVIDEBY7;
#endif
  }
}

__global__ void leftPackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height, int block_depth) {
  int j = blockDim.x*blockIdx.x+threadIdx.x;
  int k = blockDim.y*blockIdx.y+threadIdx.y;
  if (j < block_height && k < block_depth) {
    ghost[block_height*k+j] = temperature[IDX(1,1+j,1+k)];
  }
}

__global__ void rightPackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height, int block_depth) {
  int j = blockDim.x*blockIdx.x+threadIdx.x;
  int k = blockDim.y*blockIdx.y+threadIdx.y;
  if (j < block_height && k < block_depth) {
    ghost[block_height*k+j] = temperature[IDX(block_width,1+j,1+k)];
  }
}

__global__ void topPackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int k = blockDim.y*blockIdx.y+threadIdx.y;
  if (i < block_width && k < block_depth) {
    ghost[block_width*k+i] = temperature[IDX(1+i,1,1+k)];
  }
}

__global__ void bottomPackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int k = blockDim.y*blockIdx.y+threadIdx.y;
  if (i < block_width && k < block_depth) {
    ghost[block_width*k+i] = temperature[IDX(1+i,block_height,1+k)];
  }
}

__global__ void frontPackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int j = blockDim.y*blockIdx.y+threadIdx.y;
  if (i < block_width && j < block_height) {
    ghost[block_width*j+i] = temperature[IDX(1+i,1+j,1)];
  }
}

__global__ void backPackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int j = blockDim.y*blockIdx.y+threadIdx.y;
  if (i < block_width && j < block_height) {
    ghost[block_width*j+i] = temperature[IDX(1+i,1+j,block_depth)];
  }
}

__global__ void leftUnpackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height, int block_depth) {
  int j = blockDim.x*blockIdx.x+threadIdx.x;
  int k = blockDim.y*blockIdx.y+threadIdx.y;
  if (j < block_height && k < block_depth) {
    temperature[IDX(0,1+j,1+k)] = ghost[block_height*k+j];
  }
}

__global__ void rightUnpackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height, int block_depth) {
  int j = blockDim.x*blockIdx.x+threadIdx.x;
  int k = blockDim.y*blockIdx.y+threadIdx.y;
  if (j < block_height && k < block_depth) {
    temperature[IDX(block_width+1,1+j,1+k)] = ghost[block_height*k+j];
  }
}

__global__ void topUnpackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int k = blockDim.y*blockIdx.y+threadIdx.y;
  if (i < block_width && k < block_depth) {
    temperature[IDX(1+i,0,1+k)] = ghost[block_width*k+i];
  }
}

__global__ void bottomUnpackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int k = blockDim.y*blockIdx.y+threadIdx.y;
  if (i < block_width && k < block_depth) {
    temperature[IDX(1+i,block_height+1,1+k)] = ghost[block_width*k+i];
  }
}

__global__ void frontUnpackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int j = blockDim.y*blockIdx.y+threadIdx.y;
  if (i < block_width && j < block_height) {
    temperature[IDX(1+i,1+j,0)] = ghost[block_width*j+i];
  }
}

__global__ void backUnpackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int j = blockDim.y*blockIdx.y+threadIdx.y;
  if (i < block_width && j < block_height) {
    temperature[IDX(1+i,1+j,block_depth+1)] = ghost[block_width*j+i];
  }
}

void invokeInitKernel(DataType* d_temperature, int block_width, int block_height,
    int block_depth, cudaStream_t stream) {
  dim3 block_dim(TILE_SIZE_3D, TILE_SIZE_3D, TILE_SIZE_3D);
  dim3 grid_dim(((block_width+2)+(block_dim.x-1))/block_dim.x,
      ((block_height+2)+(block_dim.y-1))/block_dim.y,
      ((block_depth+2)+(block_dim.z-1))/block_dim.z);

  initKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature, block_width,
      block_height, block_depth);
  hapiCheck(cudaPeekAtLastError());
}

void invokeGhostInitKernels(const std::vector<DataType*>& ghosts,
    const std::vector<int>& ghost_counts, cudaStream_t stream) {
  dim3 block_dim(256);
  for (int i = 0; i < ghosts.size(); i++) {
    DataType* ghost = ghosts[i];
    int ghost_count = ghost_counts[i];

    dim3 grid_dim((ghost_count+block_dim.x-1)/block_dim.x);

    ghostInitKernel<<<grid_dim, block_dim, 0, stream>>>(ghost,
        ghost_count);
    hapiCheck(cudaPeekAtLastError());
  }
}

void invokeBoundaryKernels(DataType* d_temperature, int block_width,
    int block_height, int block_depth, bool bounds[], cudaStream_t stream) {
  dim3 block_dim(TILE_SIZE_2D, TILE_SIZE_2D);

  if (bounds[LEFT]) {
    dim3 grid_dim((block_height+(block_dim.x-1))/block_dim.x,
        (block_depth+(block_dim.y-1))/block_dim.y);
    leftBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        block_width, block_height, block_depth);
  }
  if (bounds[RIGHT]) {
    dim3 grid_dim((block_height+(block_dim.x-1))/block_dim.x,
        (block_depth+(block_dim.y-1))/block_dim.y);
    rightBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        block_width, block_height, block_depth);
  }
  if (bounds[TOP]) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
      (block_depth+(block_dim.y-1))/block_dim.y);
    topBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        block_width, block_height, block_depth);
  }
  if (bounds[BOTTOM]) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
      (block_depth+(block_dim.y-1))/block_dim.y);
    bottomBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        block_width, block_height, block_depth);
  }
  if (bounds[FRONT]) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        (block_height+(block_dim.y-1))/block_dim.y);
    frontBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        block_width, block_height, block_depth);
  }
  if (bounds[BACK]) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        (block_height+(block_dim.y-1))/block_dim.y);
    backBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        block_width, block_height, block_depth);
  }
  hapiCheck(cudaPeekAtLastError());
}

void invokeJacobiKernel(DataType* d_temperature, DataType* d_new_temperature,
    int block_width, int block_height, int block_depth, cudaStream_t stream) {
  dim3 block_dim(TILE_SIZE_3D, TILE_SIZE_3D, TILE_SIZE_3D);
  dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
      (block_height+(block_dim.y-1))/block_dim.y,
      (block_depth+(block_dim.z-1))/block_dim.z);

  jacobiKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature, d_new_temperature,
      block_width, block_height, block_depth);
  hapiCheck(cudaPeekAtLastError());
}

void invokePackingKernels(DataType* d_temperature, DataType* d_ghosts[],
    bool bounds[], int block_width, int block_height, int block_depth,
    cudaStream_t stream) {
  dim3 block_dim(TILE_SIZE_2D, TILE_SIZE_2D);

  if (!bounds[LEFT]) {
    dim3 grid_dim((block_height+(block_dim.x-1))/block_dim.x,
        (block_depth+(block_dim.y-1))/block_dim.y);
    leftPackingKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        d_ghosts[LEFT], block_width, block_height, block_depth);
  }
  if (!bounds[RIGHT]) {
    dim3 grid_dim((block_height+(block_dim.x-1))/block_dim.x,
        (block_depth+(block_dim.y-1))/block_dim.y);
    rightPackingKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        d_ghosts[RIGHT], block_width, block_height, block_depth);
  }
  if (!bounds[TOP]) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        (block_depth+(block_dim.y-1))/block_dim.y);
    topPackingKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        d_ghosts[TOP], block_width, block_height, block_depth);
  }
  if (!bounds[BOTTOM]) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        (block_depth+(block_dim.y-1))/block_dim.y);
    bottomPackingKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        d_ghosts[BOTTOM], block_width, block_height, block_depth);
  }
  if (!bounds[FRONT]) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        (block_height+(block_dim.y-1))/block_dim.y);
    frontPackingKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        d_ghosts[FRONT], block_width, block_height, block_depth);
  }
  if (!bounds[BACK]) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        (block_height+(block_dim.y-1))/block_dim.y);
    backPackingKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        d_ghosts[BACK], block_width, block_height, block_depth);
  }
  hapiCheck(cudaPeekAtLastError());
}

void invokeUnpackingKernel(DataType* d_temperature, DataType* d_ghost, int dir,
    int block_width, int block_height, int block_depth, cudaStream_t stream) {
  dim3 block_dim(TILE_SIZE_2D, TILE_SIZE_2D);

  if (dir == LEFT) {
    dim3 grid_dim((block_height+(block_dim.x-1))/block_dim.x,
        (block_depth+(block_dim.y-1))/block_dim.y);
    leftUnpackingKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        d_ghost, block_width, block_height, block_depth);
  } else if (dir == RIGHT) {
    dim3 grid_dim((block_height+(block_dim.x-1))/block_dim.x,
        (block_depth+(block_dim.y-1))/block_dim.y);
    rightUnpackingKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        d_ghost, block_width, block_height, block_depth);
  } else if (dir == TOP) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        (block_depth+(block_dim.y-1))/block_dim.y);
    topUnpackingKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        d_ghost, block_width, block_height, block_depth);
  } else if (dir == BOTTOM) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        (block_depth+(block_dim.y-1))/block_dim.y);
    bottomUnpackingKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        d_ghost, block_width, block_height, block_depth);
  } else if (dir == FRONT) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        (block_height+(block_dim.y-1))/block_dim.y);
    frontUnpackingKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        d_ghost, block_width, block_height, block_depth);
  } else if (dir == BACK) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        (block_height+(block_dim.y-1))/block_dim.y);
    backUnpackingKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        d_ghost, block_width, block_height, block_depth);
  }
  hapiCheck(cudaPeekAtLastError());
}
