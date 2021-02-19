#ifndef _CKRDMADEVICE_H_
#define _CKRDMADEVICE_H_

#include "ckcallback.h"
#include "conv-rdmadevice.h"

#if CMK_CUDA
#include <cuda_runtime.h>

#define CkNcpyModeDevice CmiNcpyModeDevice

struct CkDeviceBufferPost {
  // CUDA stream for device transfers
  //cudaStream_t cuda_stream;

  // Use per-thread stream by default
  //CkDeviceBufferPost() : cuda_stream(cudaStreamPerThread) {}
  //CkDeviceBufferPost() : cuda_stream(cudaStreamPerThread) {}
};

class CkDeviceBuffer : public CmiDeviceBuffer {
public:
  // Callback to be invoked on the sender/receiver
  CkCallback cb;

  CkDeviceBuffer() : CmiDeviceBuffer() {
    cb = CkCallback(CkCallback::ignore);
  }

  explicit CkDeviceBuffer(const void* ptr_) : CmiDeviceBuffer(ptr_, 0) {
    cb = CkCallback(CkCallback::ignore);
  }

  explicit CkDeviceBuffer(const void* ptr_, const CkCallback& cb_) : CmiDeviceBuffer(ptr_, 0) {
    cb = cb_;
  }

  /*
  explicit CkDeviceBuffer(const void* ptr_, cudaStream_t cuda_stream_) : CmiDeviceBuffer(ptr_, 0) {
    cb = CkCallback(CkCallback::ignore);
    cuda_stream = cuda_stream_;
  }

  explicit CkDeviceBuffer(const void* ptr_, const CkCallback& cb_, cudaStream_t cuda_stream_) : CmiDeviceBuffer(ptr_, 0) {
    cb = cb_;
    cuda_stream = cuda_stream_;
  }
  */

  explicit CkDeviceBuffer(const void* ptr_, size_t cnt_) : CmiDeviceBuffer(ptr_, cnt_) {
    cb = CkCallback(CkCallback::ignore);
  }

  explicit CkDeviceBuffer(const void* ptr_, size_t cnt_, const CkCallback& cb_) : CmiDeviceBuffer(ptr_, cnt_) {
    cb = cb_;
  }

  /*
  explicit CkDeviceBuffer(const void* ptr_, size_t cnt_, cudaStream_t cuda_stream_) : CmiDeviceBuffer(ptr_, cnt_) {
    cb = CkCallback(CkCallback::ignore);
    cuda_stream = cuda_stream_;
  }

  explicit CkDeviceBuffer(const void* ptr_, size_t cnt_, const CkCallback& cb_, cudaStream_t cuda_stream_) : CmiDeviceBuffer(ptr_, cnt_) {
    cb = cb_;
    cuda_stream = cuda_stream_;
  }
  */

  void pup(PUP::er &p) {
    CmiDeviceBuffer::pup(p);
    p|cb;
  }

  friend bool CkRdmaDeviceIssueRgets(envelope *env, int numops, void **arrPtrs, int *arrSizes, CkDeviceBufferPost *postStructs);
#if CMK_CHARM4PY
  friend bool CkRdmaDeviceIssueRgetsFromUnpackedMessage(int numops, CkDeviceBuffer **sourceStructs, void **arrPtrs, int *arrSizes, CkDeviceBufferPost *postStructs);
#endif
};

#define CKCALLBACK_POOL_SIZE 65536
#define CKCALLBACK_POOL_INC_FACTOR 2

struct CkCallbackPool {
  std::forward_list<CkCallback*> cbs;
  int max_size;
  int cur_size;

  CkCallbackPool(int initial_size = CKCALLBACK_POOL_SIZE) :
    max_size(initial_size), cur_size(0) {}

  ~CkCallbackPool() {
    for (CkCallback* cb : cbs) delete cb;
  }

  inline CkCallback* alloc() {
    if (cur_size == 0) {
      // No remaining slots, need to expand
      cbs.resize(max_size);
      cur_size = max_size;
      max_size *= 2;
      for (CkCallback*& cb : cbs) cb = new CkCallback();
    }

    CkCallback* ret = cbs.front();
    cbs.pop_front();
    cur_size--;

    return ret;
  }

  inline void free(CkCallback* cb) {
    // No sanity check
    cbs.push_front(cb);
    cur_size++;
  }
};

void CkRdmaDeviceRecvHandler(void* data);
void CkRdmaDeviceAmpiRecvHandler(void* data);
#if CMK_CHARM4PY
void CkRdmaDeviceExtRecvHandler(void* data);
bool CkRdmaDeviceIssueRgetsFromUnpackedMessage(int numops, CkDeviceBuffer **sourceStructs, void **arrPtrs, int *arrSizes, CkDeviceBufferPost *postStructs, CkCallback &destCb);
#endif
bool CkRdmaDeviceIssueRgets(envelope *env, int numops, void **arrPtrs, int *arrSizes, CkDeviceBufferPost *postStructs);
void CkRdmaDeviceOnSender(int dest_pe, int numops, CkDeviceBuffer** buffers);
#endif // CMK_CUDA

#endif // _CKRDMADEVICE_H_
