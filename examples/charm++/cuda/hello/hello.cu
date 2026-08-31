#include <stdio.h>
#include <stdlib.h>
#include "hapi.h"
#include "hello.h"

// The kernel writes a known value into every element so that the host can tell
// "the kernel ran" apart from "the kernel never launched". An empty kernel
// cannot make that distinction: a launch that silently fails -- for instance
// because the object carries no cubin or PTX for this device's architecture --
// looks exactly like a successful one.
__global__ void helloKernel(int* out, int n, int val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = val;
}

void runHello(struct hapiWorkRequest* wr, cudaStream_t kernel_stream,
              void** deviceBuffers) {
  helloKernel<<<wr->grid_dim, wr->block_dim, wr->shared_mem, kernel_stream>>>(
      (int*)deviceBuffers[wr->getBufferID(0)], HELLO_ELEMS,
      *((int*)wr->getUserData()));
  hapiCheck(cudaPeekAtLastError());
}

void kernelSetup(cudaStream_t stream, const CkCallback& cb, int* h_out,
                 int* d_out, int val) {
  int size = HELLO_ELEMS * sizeof(int);
  dim3 block_dim(HELLO_ELEMS, 1);
  dim3 grid_dim(1, 1);

#ifdef USE_WR
  // DEPRECATED
  hapiWorkRequest* wr = hapiCreateWorkRequest();
  wr->setExecParams(grid_dim, block_dim);
  wr->setStream(stream);
  wr->addBuffer(h_out, size, false, true, true);
  wr->setCallback(cb);
#ifdef HAPI_TRACE
  wr->setTraceName("hello");
#endif
  wr->setRunKernel(runHello);
  wr->copyUserData(&val, sizeof(int));

  hapiEnqueue(wr);
#else
  // Zero the destination first, so that a kernel which never runs leaves a
  // value the host check is guaranteed to reject rather than whatever the
  // allocation happened to contain.
  hapiCheck(cudaMemsetAsync(d_out, 0, size, stream));

  helloKernel<<<grid_dim, block_dim, 0, stream>>>(d_out, HELLO_ELEMS, val);
  // Without this an unlaunchable kernel is invisible: the stream drains, the
  // callback fires, and only the payload check in hello.C notices anything.
  hapiCheck(cudaPeekAtLastError());

  hapiCheck(cudaMemcpyAsync(h_out, d_out, size, cudaMemcpyDeviceToHost, stream));
  hapiAddCallback(stream, cb);
#endif
}
