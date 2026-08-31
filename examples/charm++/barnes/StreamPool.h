#ifndef __STREAM_POOL_H__
#define __STREAM_POOL_H__

#ifdef GPU_GRAVITY

#include "barnes.decl.h"
#include "hapi.h"

#include <vector>

// How many CUDA streams a PE hands out. Tree pieces on a PE are independent --
// they touch disjoint buckets -- so giving each its own stream is what lets
// their kernels overlap. More streams than tree pieces per PE buys nothing.
#define NUM_STREAMS 8

// Streams are created here rather than in each chare because creating one is a
// device call, and tree pieces are constructed during startup, before HAPI has
// selected this PE's device.
class StreamPool : public CBase_StreamPool {
  std::vector<cudaStream_t> streams;
  int next;

 public:
  StreamPool() : streams(NUM_STREAMS), next(0) {
    for (int i = 0; i < NUM_STREAMS; i++)
      hapiCheck(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
  }
  StreamPool(CkMigrateMessage *m) : streams(0), next(0) {}

  cudaStream_t acquire() { return streams[(next++) % NUM_STREAMS]; }
};

#endif // GPU_GRAVITY
#endif // __STREAM_POOL_H__
