/* Device kernels for jacobi_od -- a portable (CUDA and HIP) copy of the kernels
 * in examples/charm++/cuda/gpudirect/jacobi2d/jacobi2d.cu. The kernels
 * themselves are unchanged; the host-side entry points use the portable
 * spellings from hapi_portable.h and every launch is error-checked.
 *
 * hapi_portable.h is included instead of <cuda_runtime.h>/<hip/hip_runtime.h>
 * directly; it picks the backend from conv-mach-opt.h (CMK_CUDA / CMK_HIP), so
 * this file must be compiled with the Charm++ include directory on the path.
 */
#include "hapi_portable.h"
#include "jacobi_od.h"

#include <cstdio>
#include <cstdlib>

#define TILE_SIZE 16
#define DIVIDEBY5 0.2

/* Check the launch that just happened. hapiGetLastError() reports (and clears)
 * errors from the launch itself -- a bad grid/block shape, or a kernel that was
 * never compiled for this device's architecture. Without it such a launch is a
 * silent no-op and the run merely produces wrong numbers. Asynchronous
 * execution errors are NOT covered here; they surface at the next
 * synchronization point. */
#define CHECK_LAUNCH(kernel_name)                                             \
  do {                                                                        \
    hapiError_t err__ = hapiGetLastError();                                   \
    if (err__ != hapiSuccess) {                                               \
      fprintf(stderr, "%s:%d: launch of %s failed: %s\n", __FILE__, __LINE__, \
              (kernel_name), hapiGetErrorString(err__));                      \
      abort();                                                                \
    }                                                                         \
  } while (0)

__global__ void initKernel(DataType* temperature, int block_width,
    int block_height) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i < block_width + 2 && j < block_height + 2) {
    temperature[IDX(i,j)] = 0;
  }
}

__global__ void leftBoundaryKernel(DataType* temperature, int block_width,
    int block_height) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < block_height) {
    temperature[IDX(0,1+j)] = 1;
  }
}

__global__ void rightBoundaryKernel(DataType* temperature, int block_width,
    int block_height) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < block_height) {
    temperature[IDX(block_width+1,1+j)] = 1;
  }
}

__global__ void topBoundaryKernel(DataType* temperature, int block_width,
    int block_height) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < block_width) {
    temperature[IDX(1+i,0)] = 1;
  }
}

__global__ void bottomBoundaryKernel(DataType* temperature, int block_width,
    int block_height) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < block_width) {
    temperature[IDX(1+i,block_height+1)] = 1;
  }
}

__global__ void jacobiKernel(DataType* temperature, DataType* new_temperature,
    int block_width, int block_height) {
  int i = (blockDim.x * blockIdx.x + threadIdx.x) + 1;
  int j = (blockDim.y * blockIdx.y + threadIdx.y) + 1;

  if (i <= block_width && j <= block_height) {
#ifdef TEST_CORRECTNESS
    /* Copied verbatim from the jacobi2d example, INCLUDING its defect: with
     * TEST_CORRECTNESS the sum is int and `% 1e5` is int % double, which does
     * not compile ("invalid operands to binary expression"). jacobi3d's
     * equivalent line spells it `% 10000`, an integer literal. Left unchanged
     * here so this stays a port; fix it in the example first. */
    new_temperature[IDX(i,j)] = (temperature[IDX(i-1,j)] + temperature[IDX(i+1,j)] +
      temperature[IDX(i,j-1)] + temperature[IDX(i,j+1)] + temperature[IDX(i,j)]) %
      1e5;
#else
    new_temperature[IDX(i,j)] = (temperature[IDX(i-1,j)] + temperature[IDX(i+1,j)] +
      temperature[IDX(i,j-1)] + temperature[IDX(i,j+1)] + temperature[IDX(i,j)]) *
      DIVIDEBY5;
#endif
  }
}

__global__ void leftPackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < block_height) {
    ghost[j] = temperature[IDX(1,1+j)];
  }
}

__global__ void rightPackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < block_height) {
    ghost[j] = temperature[IDX(block_width,1+j)];
  }
}

__global__ void leftUnpackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < block_height) {
    temperature[IDX(0,1+j)] = ghost[j];
  }
}

__global__ void rightUnpackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < block_height) {
    temperature[IDX(block_width+1,1+j)] = ghost[j];
  }
}

void invokeInitKernel(DataType* d_temperature, int block_width, int block_height,
    hapiStream_t stream) {
  dim3 block_dim(TILE_SIZE, TILE_SIZE);
  dim3 grid_dim(((block_width + 2) + (block_dim.x - 1)) / block_dim.x,
      ((block_height + 2) + (block_dim.y - 1)) / block_dim.y);

  initKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature, block_width, block_height);
  CHECK_LAUNCH("initKernel");
}

void invokeBoundaryKernels(DataType* d_temperature, int block_width,
    int block_height, bool left_bound, bool right_bound, bool top_bound,
    bool bottom_bound, hapiStream_t stream) {
  dim3 block_dim(TILE_SIZE * TILE_SIZE);

  if (left_bound) {
    dim3 grid_dim((block_height + (block_dim.x - 1)) / block_dim.x);
    leftBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        block_width, block_height);
    CHECK_LAUNCH("leftBoundaryKernel");
  }
  if (right_bound) {
    dim3 grid_dim((block_height + (block_dim.x - 1)) / block_dim.x);
    rightBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        block_width, block_height);
    CHECK_LAUNCH("rightBoundaryKernel");
  }
  if (top_bound) {
    dim3 grid_dim((block_width + (block_dim.x - 1)) / block_dim.x);
    topBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        block_width, block_height);
    CHECK_LAUNCH("topBoundaryKernel");
  }
  if (bottom_bound) {
    dim3 grid_dim((block_width + (block_dim.x - 1)) / block_dim.x);
    bottomBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        block_width, block_height);
    CHECK_LAUNCH("bottomBoundaryKernel");
  }
}

void invokeJacobiKernel(DataType* d_temperature, DataType* d_new_temperature,
    int block_width, int block_height, hapiStream_t stream) {
  dim3 block_dim(TILE_SIZE, TILE_SIZE);
  dim3 grid_dim((block_width + (block_dim.x - 1)) / block_dim.x,
      (block_height + (block_dim.y - 1)) / block_dim.y);

  jacobiKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature, d_new_temperature, block_width, block_height);
  CHECK_LAUNCH("jacobiKernel");
}

void invokePackingKernels(DataType* d_temperature, DataType* d_left_ghost,
    DataType* d_right_ghost, bool left_bound, bool right_bound, int block_width,
    int block_height, hapiStream_t stream) {
  dim3 block_dim(TILE_SIZE * TILE_SIZE);
  dim3 grid_dim((block_height + (block_dim.x - 1)) / block_dim.x);
  if (!left_bound) {
    leftPackingKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature, d_left_ghost, block_width, block_height);
    CHECK_LAUNCH("leftPackingKernel");
  }
  if (!right_bound) {
    rightPackingKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature, d_right_ghost, block_width, block_height);
    CHECK_LAUNCH("rightPackingKernel");
  }
}

void invokeUnpackingKernel(DataType* d_temperature, DataType* d_ghost, bool is_left,
    int block_width, int block_height, hapiStream_t stream) {
  dim3 block_dim(TILE_SIZE * TILE_SIZE);
  dim3 grid_dim((block_height + (block_dim.x - 1)) / block_dim.x);
  if (is_left) {
    leftUnpackingKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature, d_ghost, block_width, block_height);
    CHECK_LAUNCH("leftUnpackingKernel");
  } else {
    rightUnpackingKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature, d_ghost, block_width, block_height);
    CHECK_LAUNCH("rightUnpackingKernel");
  }
}
