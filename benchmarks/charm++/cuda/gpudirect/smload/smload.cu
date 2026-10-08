#include "hapi.h"
#include "smload.h"

// 32 x 32 = 1024 threads. On an A40 an SM holds 1536 threads, so one such
// block is all an SM takes at a time, and the number of SMs a launch occupies
// is the number of blocks launched (hapi's computeKernelSMs: min(SMs,
// ceil(blocks / max active blocks per SM)), the same formula the occupancy
// API answers, which smloadBlocksPerSM asks so the launch adapts to a device
// with room for two).
#define TILE 32
#define SMALL_BLOCK 256
#define DIVIDEBY5 0.2f

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
  if (j < block_height) temperature[IDX(0,1+j)] = 1;
}

__global__ void rightBoundaryKernel(DataType* temperature, int block_width,
    int block_height) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < block_height) temperature[IDX(block_width+1,1+j)] = 1;
}

__global__ void topBoundaryKernel(DataType* temperature, int block_width,
    int block_height) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < block_width) temperature[IDX(1+i,0)] = 1;
}

__global__ void bottomBoundaryKernel(DataType* temperature, int block_width,
    int block_height) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < block_width) temperature[IDX(1+i,block_height+1)] = 1;
}

// The stencil, over the whole tile, from however many blocks were launched:
// each block walks the 32x32 tiles of the grid with a stride of the launch
// size. The work is the tile whatever the launch, so the kernel on s SMs runs
// about SMs/s times longer than on the full device, and the SM-seconds it
// consumes stay the same. `iter` repeats the stencil per cell to set the
// duration.
__global__ void __launch_bounds__(TILE * TILE)
jacobiSMKernel(const DataType* __restrict__ temperature,
    DataType* __restrict__ new_temperature, int block_width, int block_height,
    int iter) {
  const int tiles_x = (block_width + TILE - 1) / TILE;
  const int tiles_y = (block_height + TILE - 1) / TILE;
  for (int t = blockIdx.x; t < tiles_x * tiles_y; t += gridDim.x) {
    const int i = (t % tiles_x) * TILE + threadIdx.x + 1;
    const int j = (t / tiles_x) * TILE + threadIdx.y + 1;
    if (i <= block_width && j <= block_height) {
      DataType temp = 0;
      for (int it = 0; it < iter; it++)
        temp += (temperature[IDX(i-1,j)] + temperature[IDX(i+1,j)] +
                 temperature[IDX(i,j-1)] + temperature[IDX(i,j+1)] +
                 temperature[IDX(i,j)]) * DIVIDEBY5;
      new_temperature[IDX(i,j)] = temp / iter;
    }
  }
}

__global__ void leftPackingKernel(const DataType* temperature, DataType* ghost,
    int block_width, int block_height) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < block_height) ghost[j] = temperature[IDX(1,1+j)];
}

__global__ void rightPackingKernel(const DataType* temperature, DataType* ghost,
    int block_width, int block_height) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < block_height) ghost[j] = temperature[IDX(block_width,1+j)];
}

__global__ void leftUnpackingKernel(DataType* temperature, const DataType* ghost,
    int block_width, int block_height) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < block_height) temperature[IDX(0,1+j)] = ghost[j];
}

__global__ void rightUnpackingKernel(DataType* temperature, const DataType* ghost,
    int block_width, int block_height) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < block_height) temperature[IDX(block_width+1,1+j)] = ghost[j];
}

void invokeInitKernel(DataType* d_temperature, int block_width, int block_height,
    cudaStream_t stream) {
  dim3 block_dim(16, 16);
  dim3 grid_dim(((block_width + 2) + (block_dim.x - 1)) / block_dim.x,
      ((block_height + 2) + (block_dim.y - 1)) / block_dim.y);
  hapiLaunchKernelWrapper(initKernel, grid_dim, block_dim, 0, stream,
      d_temperature, block_width, block_height);
  hapiCheck(cudaPeekAtLastError());
}

void invokeBoundaryKernels(DataType* d_temperature, int block_width,
    int block_height, bool left_bound, bool right_bound, bool top_bound,
    bool bottom_bound, cudaStream_t stream) {
  dim3 block_dim(SMALL_BLOCK);
  dim3 grid_h((block_height + (block_dim.x - 1)) / block_dim.x);
  dim3 grid_w((block_width + (block_dim.x - 1)) / block_dim.x);
  if (left_bound)
    hapiLaunchKernelWrapper(leftBoundaryKernel, grid_h, block_dim, 0, stream,
        d_temperature, block_width, block_height);
  if (right_bound)
    hapiLaunchKernelWrapper(rightBoundaryKernel, grid_h, block_dim, 0, stream,
        d_temperature, block_width, block_height);
  if (top_bound)
    hapiLaunchKernelWrapper(topBoundaryKernel, grid_w, block_dim, 0, stream,
        d_temperature, block_width, block_height);
  if (bottom_bound)
    hapiLaunchKernelWrapper(bottomBoundaryKernel, grid_w, block_dim, 0, stream,
        d_temperature, block_width, block_height);
  hapiCheck(cudaPeekAtLastError());
}

void invokeJacobiSM(const DataType* d_temperature, DataType* d_new_temperature,
    int block_width, int block_height, int iter, int nblocks, cudaStream_t stream) {
  hapiLaunchKernelWrapper(jacobiSMKernel, dim3(nblocks), dim3(TILE, TILE), 0,
      stream, d_temperature, d_new_temperature, block_width, block_height, iter);
  hapiCheck(cudaPeekAtLastError());
}

// How many stencil blocks an SM of the current device runs at once: the
// launch multiplies the SM count by this, so the occupancy model reads the
// SM count back.
int smloadBlocksPerSM() {
  int n = 0;
  hapiCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, jacobiSMKernel,
      TILE * TILE, 0));
  return n < 1 ? 1 : n;
}

void invokePackingKernels(const DataType* d_temperature, DataType* d_left_ghost,
    DataType* d_right_ghost, bool left_bound, bool right_bound, int block_width,
    int block_height, cudaStream_t stream) {
  dim3 block_dim(SMALL_BLOCK);
  dim3 grid_dim((block_height + (block_dim.x - 1)) / block_dim.x);
  if (!left_bound)
    hapiLaunchKernelWrapper(leftPackingKernel, grid_dim, block_dim, 0, stream,
        d_temperature, d_left_ghost, block_width, block_height);
  if (!right_bound)
    hapiLaunchKernelWrapper(rightPackingKernel, grid_dim, block_dim, 0, stream,
        d_temperature, d_right_ghost, block_width, block_height);
  hapiCheck(cudaPeekAtLastError());
}

void invokeUnpackingKernel(DataType* d_temperature, const DataType* d_ghost,
    bool is_left, int block_width, int block_height, cudaStream_t stream) {
  dim3 block_dim(SMALL_BLOCK);
  dim3 grid_dim((block_height + (block_dim.x - 1)) / block_dim.x);
  if (is_left)
    hapiLaunchKernelWrapper(leftUnpackingKernel, grid_dim, block_dim, 0, stream,
        d_temperature, d_ghost, block_width, block_height);
  else
    hapiLaunchKernelWrapper(rightUnpackingKernel, grid_dim, block_dim, 0, stream,
        d_temperature, d_ghost, block_width, block_height);
  hapiCheck(cudaPeekAtLastError());
}
