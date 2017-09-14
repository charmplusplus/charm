#include "hapi.h"

#define BLOCK_SIZE 256

__global__ void clock_block(clock_t clock_count) {
  unsigned int start_clock = (unsigned int)clock();
  clock_t clock_offset = 0;

  while (clock_offset < clock_count) {
    unsigned int end_clock = (unsigned int)clock();

    clock_offset = (clock_t)(end_clock - start_clock);
  }
}

void blockingKernel(char* h_A, char* h_B, char* d_A, char* d_B, int data_size, int num_threads, int clock_count, cudaStream_t stream, void* cb, void* cb_msg) {
  dim3 grid_dim(ceil((float)num_threads / BLOCK_SIZE));
  dim3 block_dim((num_threads > BLOCK_SIZE) ? BLOCK_SIZE : num_threads);

  if (data_size > 0)
    hapiCheck(cudaMemcpyAsync(d_A, h_A, data_size, cudaMemcpyHostToDevice, stream));

  clock_block<<<grid_dim, block_dim, 0, stream>>>((clock_t)clock_count);
  hapiCheck(cudaPeekAtLastError());

  if (data_size > 0)
    hapiCheck(cudaMemcpyAsync(h_B, d_B, data_size, cudaMemcpyDeviceToHost, stream));

  if (cb == NULL)
    hapiCheck(cudaStreamSynchronize(stream));
  else
    hapiAddCallback(stream, cb, cb_msg);
}
