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
      temperature[IDX(i,j,k-1)] + temperature[IDX(i,j,k+1)]) % 1e5;
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

void invokeBoundaryKernels(DataType* d_temperature, int block_width,
    int block_height, int block_depth, bool left_bound, bool right_bound,
    bool top_bound, bool bottom_bound, bool front_bound, bool back_bound,
    cudaStream_t stream) {
  dim3 block_dim(TILE_SIZE_2D, TILE_SIZE_2D);

  if (left_bound) {
    dim3 grid_dim((block_height+(block_dim.x-1))/block_dim.x,
        (block_depth+(block_dim.y-1))/block_dim.y);
    leftBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        block_width, block_height, block_depth);
  }
  if (right_bound) {
    dim3 grid_dim((block_height+(block_dim.x-1))/block_dim.x,
        (block_depth+(block_dim.y-1))/block_dim.y);
    rightBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        block_width, block_height, block_depth);
  }
  if (top_bound) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
      (block_depth+(block_dim.y-1))/block_dim.y);
    topBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        block_width, block_height, block_depth);
  }
  if (bottom_bound) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
      (block_depth+(block_dim.y-1))/block_dim.y);
    bottomBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        block_width, block_height, block_depth);
  }
  if (front_bound) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        (block_height+(block_dim.y-1))/block_dim.y);
    frontBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        block_width, block_height, block_depth);
  }
  if (back_bound) {
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

void invokePackingKernels(DataType* d_temperature, DataType* d_left_ghost,
    DataType* d_right_ghost, DataType* d_top_ghost, DataType* d_bottom_ghost,
    DataType* d_front_ghost, DataType* d_back_ghost, bool left_bound,
    bool right_bound, bool top_bound, bool bottom_bound, bool front_bound,
    bool back_bound, int block_width, int block_height, int block_depth,
    cudaStream_t stream) {
  dim3 block_dim(TILE_SIZE_2D, TILE_SIZE_2D);

  if (!left_bound) {
    dim3 grid_dim((block_height+(block_dim.x-1))/block_dim.x,
        block_depth+(block_dim.y-1)/block_dim.y);
    leftPackingKernel<<<block_dim, grid_dim, 0, stream>>>(d_temperature,
        d_left_ghost, block_width, block_height, block_depth);
  }
  if (!right_bound) {
    dim3 grid_dim((block_height+(block_dim.x-1))/block_dim.x,
        block_depth+(block_dim.y-1)/block_dim.y);
    rightPackingKernel<<<block_dim, grid_dim, 0, stream>>>(d_temperature,
        d_right_ghost, block_width, block_height, block_depth);
  }
  if (!top_bound) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        block_depth+(block_dim.y-1)/block_dim.y);
    topPackingKernel<<<block_dim, grid_dim, 0, stream>>>(d_temperature,
        d_top_ghost, block_width, block_height, block_depth);
  }
  if (!bottom_bound) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        block_depth+(block_dim.y-1)/block_dim.y);
    bottomPackingKernel<<<block_dim, grid_dim, 0, stream>>>(d_temperature,
        d_bottom_ghost, block_width, block_height, block_depth);
  }
  if (!front_bound) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        block_height+(block_dim.y-1)/block_dim.y);
    frontPackingKernel<<<block_dim, grid_dim, 0, stream>>>(d_temperature,
        d_front_ghost, block_width, block_height, block_depth);
  }
  if (!back_bound) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        block_height+(block_dim.y-1)/block_dim.y);
    backPackingKernel<<<block_dim, grid_dim, 0, stream>>>(d_temperature,
        d_back_ghost, block_width, block_height, block_depth);
  }
  hapiCheck(cudaPeekAtLastError());
}

void invokeUnpackingKernel(DataType* d_temperature, DataType* d_ghost, int dir,
    int block_width, int block_height, int block_depth, cudaStream_t stream) {
  dim3 block_dim(TILE_SIZE_2D, TILE_SIZE_2D);

  if (dir == LEFT) {
    dim3 grid_dim((block_height+(block_dim.x-1))/block_dim.x,
        block_depth+(block_dim.y-1)/block_dim.y);
    leftUnpackingKernel<<<block_dim, grid_dim, 0, stream>>>(d_temperature,
        d_ghost, block_width, block_height, block_depth);
  } else if (dir == RIGHT) {
    dim3 grid_dim((block_height+(block_dim.x-1))/block_dim.x,
        block_depth+(block_dim.y-1)/block_dim.y);
    rightUnpackingKernel<<<block_dim, grid_dim, 0, stream>>>(d_temperature,
        d_ghost, block_width, block_height, block_depth);
  } else if (dir == TOP) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        block_depth+(block_dim.y-1)/block_dim.y);
    topUnpackingKernel<<<block_dim, grid_dim, 0, stream>>>(d_temperature,
        d_ghost, block_width, block_height, block_depth);
  } else if (dir == BOTTOM) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        block_depth+(block_dim.y-1)/block_dim.y);
    bottomUnpackingKernel<<<block_dim, grid_dim, 0, stream>>>(d_temperature,
        d_ghost, block_width, block_height, block_depth);
  } else if (dir == FRONT) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        block_height+(block_dim.y-1)/block_dim.y);
    frontUnpackingKernel<<<block_dim, grid_dim, 0, stream>>>(d_temperature,
        d_ghost, block_width, block_height, block_depth);
  } else if (dir == BACK) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        block_height+(block_dim.y-1)/block_dim.y);
    backUnpackingKernel<<<block_dim, grid_dim, 0, stream>>>(d_temperature,
        d_ghost, block_width, block_height, block_depth);
  }
  hapiCheck(cudaPeekAtLastError());
}
