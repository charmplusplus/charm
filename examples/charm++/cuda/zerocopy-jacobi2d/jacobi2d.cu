#include "hapi.h"

#ifndef DIVIDEBY5
#define DIVIDEBY5 0.2
#endif

__global__ void jacobiKernel(double* temperature, double* new_temperature,
    int block_size, int thread_size) {
  int i_start = (blockDim.x * blockIdx.x + threadIdx.x) * thread_size + 1;
  int i_finish = (blockDim.x * blockIdx.x + threadIdx.x) * thread_size + thread_size;
  int j_start = (blockDim.y * blockIdx.y + threadIdx.y) * thread_size + 1;
  int j_finish = (blockDim.y * blockIdx.y + threadIdx.y) * thread_size + thread_size;

  for (int i = i_start; i <= i_finish; i++) {
    for (int j = j_start; j <= j_finish; j++) {
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
  }

  /* TODO Use shared memory
  int i = istart + threadIdx.x + blockDim.x*blockIdx.x;
  int j = jstart + threadIdx.y + blockDim.y*blockIdx.y;

  if (i < ifinish && j < jfinish) {
    __shared__ double shared_temperature[TILE_SIZE][TILE_SIZE];
    double center = temperature[j*(block_size+2)+i];

    shared_temperature[threadIdx.x][threadIdx.y] = center;
    __syncthreads();

    // update my value based on the surrounding values
    new_temperature[j*(block_size+2)+i] = (
      ((threadIdx.x > 1) ? shared_temperature[threadIdx.x-1][threadIdx.y] :
  temperature[j*(block_size+2)+(i-1)]) +
      ((threadIdx.x < blockDim.x-1) ?
  shared_temperature[threadIdx.x+1][threadIdx.y] :
  temperature[j*(block_size+2)+(i+1)]) +
      ((threadIdx.y > 1) ? shared_temperature[threadIdx.x][threadIdx.y-1] :
  temperature[(j-1)*(block_size+2)+i]) +
      ((threadIdx.y < blockDim.y-1) ?
  shared_temperature[threadIdx.x][threadIdx.y+1] :
  temperature[(j+1)*(block_size+2)+i]) +
      center) * DIVIDEBY5;
  }
  */
}

void invokeKernel(cudaStream_t stream, double* d_temperature,
    double* d_new_temperature, int block_size, int thread_size) {
  dim3 block_dim(TILE_SIZE, TILE_SIZE);
  dim3 grid_dim((block_size + (block_dim.x * thread_size - 1))
      / (block_dim.x * thread_size), (block_size + (block_dim.y * thread_size - 1))
      / (block_dim.y * thread_size));

  jacobiKernel<<<grid_dim, block_dim, 0, stream>>>(
      d_temperature, d_new_temperature, block_size, thread_size);
  hapiCheck(cudaPeekAtLastError());
}
