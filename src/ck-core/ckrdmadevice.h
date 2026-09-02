#ifndef _CKRDMADEVICE_H_
#define _CKRDMADEVICE_H_

/* Two-runtime header (same pattern as conv-rdmadevice.h / cmirdmautils.h). */
#include "charm-config.h"

#if CMK_RECONVERSE

#include "ckcallback.h"
#include "conv-rdmadevice.h"

#if CMK_CUDA || CMK_HIP
#include "hapi_portable.h"

#define CkNcpyModeDevice CmiNcpyModeDevice
#define CkDeviceStatus CmiDeviceStatus

struct CkDevicePersistent {
  const void* ptr;
  size_t cnt;
  CkCallback cb;
  void* cb_msg;
  hapiStream_t hapi_stream;
  int pe;
  hapiIpcMemHandle_t hapi_ipc_handle;
  void* ipc_ptr;
  bool ipc_open; // Used only by the remote chare

  CkDevicePersistent() : ptr(nullptr), cnt(0), cb(CkCallback(CkCallback::ignore)),
                         cb_msg(nullptr), pe(-1), ipc_ptr(nullptr), ipc_open(false) {}

  explicit CkDevicePersistent(const void* ptr_, size_t cnt_)
    : ptr(ptr_), cnt(cnt_), cb(CkCallback(CkCallback::ignore)) {
    init();
  }

  explicit CkDevicePersistent(const void* ptr_, size_t cnt_, const CkCallback& cb_)
    : ptr(ptr_), cnt(cnt_), cb(cb_) {
    init();
  }

  explicit CkDevicePersistent(const void* ptr_, size_t cnt_, hapiStream_t hapi_stream_)
    : ptr(ptr_), cnt(cnt_), cb(CkCallback(CkCallback::ignore)),
      hapi_stream(hapi_stream_) {
    init();
  }

  explicit CkDevicePersistent(const void* ptr_, size_t cnt_, const CkCallback& cb_,
      hapiStream_t hapi_stream_)
    : ptr(ptr_), cnt(cnt_), cb(cb_), hapi_stream(hapi_stream_) {
    init();
  }

  void init();
  void open();
  void close();
  void set_msg(void* msg);

  // Should only be used for exchanging between chares, not for migration.
  // After the owner chare migrates, CkDevicePersistent needs to be recreated
  // and exchanged again.
  void pup(PUP::er& p);

  CkDeviceStatus get(CkDevicePersistent& src);
  CkDeviceStatus put(CkDevicePersistent& dst);
};

#define CK_DEVICE_ACK_CAP 8

struct CkDeviceBufferPost {
  // CUDA stream for device transfers
  hapiStream_t hapi_stream;

  // Use per-thread stream by default
  CkDeviceBufferPost() : hapi_stream(hapiStreamPerThread) {}
};

class CkDeviceBuffer : public CmiDeviceBuffer {
public:
  // Callback to be invoked on the sender/receiver
  CkCallback cb;

  // Piggybacked acknowledgements (reconverse RDMA path).  ack_id is the id
  // the sender minted for THIS buffer (0: nothing to acknowledge); the
  // receiver returns it, and the sender resolves it locally to a release
  // of a per-message registration and/or the source callback.  acks[] is
  // the block of ids this MESSAGE carries back to its destination PE; only
  // the first buffer of a message carries any, the rest pup one zero byte.
  uint64_t ack_id = 0;
  unsigned char ack_count = 0;
  uint64_t acks[CK_DEVICE_ACK_CAP];

  CkDeviceBuffer() : CmiDeviceBuffer() {
    cb = CkCallback(CkCallback::ignore);
  }

  explicit CkDeviceBuffer(const void* ptr_) : CmiDeviceBuffer(ptr_, 0) {
    cb = CkCallback(CkCallback::ignore);
  }

  explicit CkDeviceBuffer(const void* ptr_, const CkCallback& cb_) : CmiDeviceBuffer(ptr_, 0) {
    cb = cb_;
  }

  explicit CkDeviceBuffer(const void* ptr_, hapiStream_t hapi_stream_) : CmiDeviceBuffer(ptr_, 0) {
    cb = CkCallback(CkCallback::ignore);
    hapi_stream = hapi_stream_;
  }

  explicit CkDeviceBuffer(const void* ptr_, const CkCallback& cb_, hapiStream_t hapi_stream_) : CmiDeviceBuffer(ptr_, 0) {
    cb = cb_;
    hapi_stream = hapi_stream_;
  }

  explicit CkDeviceBuffer(const void* ptr_, size_t cnt_) : CmiDeviceBuffer(ptr_, cnt_) {
    cb = CkCallback(CkCallback::ignore);
  }

  explicit CkDeviceBuffer(const void* ptr_, size_t cnt_, const CkCallback& cb_) : CmiDeviceBuffer(ptr_, cnt_) {
    cb = cb_;
  }

  explicit CkDeviceBuffer(const void* ptr_, size_t cnt_, hapiStream_t hapi_stream_) : CmiDeviceBuffer(ptr_, cnt_) {
    cb = CkCallback(CkCallback::ignore);
    hapi_stream = hapi_stream_;
  }

  explicit CkDeviceBuffer(const void* ptr_, size_t cnt_, const CkCallback& cb_, hapiStream_t hapi_stream_) : CmiDeviceBuffer(ptr_, cnt_) {
    cb = cb_;
    hapi_stream = hapi_stream_;
  }

  void pup(PUP::er &p) {
    CmiDeviceBuffer::pup(p);
    p|cb;
    p|ack_id;
    p|ack_count;
    PUParray(p, acks, ack_count);
  }

  friend void CkRdmaDeviceIssueRgets(envelope *env, int numops, void **arrPtrs, int *arrSizes, CkDeviceBufferPost *postStructs);
};

void CkRdmaDeviceRecvHandler(void* data);
void CkRdmaDeviceRecvHandler(void* data, void* msg);
void CkRdmaDeviceIssueRgets(envelope *env, int numops, void **arrPtrs, int *arrSizes, CkDeviceBufferPost *postStructs);
void CkRdmaDeviceOnSender(int dest_pe, int numops, CkDeviceBuffer** buffers);

extern "C" {
  /* implementations arrive with stage 9.2 (reconverse D2D in ckrdmadevice.C);
   * the init.C registration site is gated to match */
  void* loopback_bridge(void* arg);
  extern int loopback_handler;
  void* device_dereg_bridge(void* arg);
  extern int device_dereg_handler;
  void* device_ack_bridge(void* arg);
  extern int device_ack_handler;
}

// Persistent registration of a device buffer used across many messages
// (send source or posted receive destination): register once, deregister
// when the application frees the buffer.  Exact (pointer, length) match.
void CkDeviceBufferRegister(const void* ptr, size_t cnt);
void CkDeviceBufferDeregister(const void* ptr, size_t cnt);
// Per-PE init of the device zerocopy path (idle-time ack flush).
void CkRdmaDeviceInit();

#endif // CMK_CUDA

#else /* classic */

#include "ckcallback.h"
#include "conv-rdmadevice.h"

#if CMK_CUDA
#include <cuda_runtime.h>

#define CkNcpyModeDevice CmiNcpyModeDevice
#define CkDeviceStatus CmiDeviceStatus

struct CkDevicePersistent {
  const void* ptr;
  size_t cnt;
  CkCallback cb;
  void* cb_msg;
  cudaStream_t cuda_stream;
  int pe;
  cudaIpcMemHandle_t cuda_ipc_handle;
  void* ipc_ptr;
  bool ipc_open; // Used only by the remote chare

  CkDevicePersistent() : ptr(nullptr), cnt(0), cb(CkCallback(CkCallback::ignore)),
                         cb_msg(nullptr), pe(-1), ipc_ptr(nullptr), ipc_open(false) {}

  explicit CkDevicePersistent(const void* ptr_, size_t cnt_)
    : ptr(ptr_), cnt(cnt_), cb(CkCallback(CkCallback::ignore)) {
    init();
  }

  explicit CkDevicePersistent(const void* ptr_, size_t cnt_, const CkCallback& cb_)
    : ptr(ptr_), cnt(cnt_), cb(cb_) {
    init();
  }

  explicit CkDevicePersistent(const void* ptr_, size_t cnt_, cudaStream_t cuda_stream_)
    : ptr(ptr_), cnt(cnt_), cb(CkCallback(CkCallback::ignore)),
      cuda_stream(cuda_stream_) {
    init();
  }

  explicit CkDevicePersistent(const void* ptr_, size_t cnt_, const CkCallback& cb_,
      cudaStream_t cuda_stream_)
    : ptr(ptr_), cnt(cnt_), cb(cb_), cuda_stream(cuda_stream_) {
    init();
  }

  void init();
  void open();
  void close();
  void set_msg(void* msg);

  // Should only be used for exchanging between chares, not for migration.
  // After the owner chare migrates, CkDevicePersistent needs to be recreated
  // and exchanged again.
  void pup(PUP::er& p);

  CkDeviceStatus get(CkDevicePersistent& src);
  CkDeviceStatus put(CkDevicePersistent& dst);
};

struct CkDeviceBufferPost {
  // CUDA stream for device transfers
  cudaStream_t cuda_stream;

  // Use per-thread stream by default
  CkDeviceBufferPost() : cuda_stream(cudaStreamPerThread) {}
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

  explicit CkDeviceBuffer(const void* ptr_, cudaStream_t cuda_stream_) : CmiDeviceBuffer(ptr_, 0) {
    cb = CkCallback(CkCallback::ignore);
    cuda_stream = cuda_stream_;
  }

  explicit CkDeviceBuffer(const void* ptr_, const CkCallback& cb_, cudaStream_t cuda_stream_) : CmiDeviceBuffer(ptr_, 0) {
    cb = cb_;
    cuda_stream = cuda_stream_;
  }

  explicit CkDeviceBuffer(const void* ptr_, size_t cnt_) : CmiDeviceBuffer(ptr_, cnt_) {
    cb = CkCallback(CkCallback::ignore);
  }

  explicit CkDeviceBuffer(const void* ptr_, size_t cnt_, const CkCallback& cb_) : CmiDeviceBuffer(ptr_, cnt_) {
    cb = cb_;
  }

  explicit CkDeviceBuffer(const void* ptr_, size_t cnt_, cudaStream_t cuda_stream_) : CmiDeviceBuffer(ptr_, cnt_) {
    cb = CkCallback(CkCallback::ignore);
    cuda_stream = cuda_stream_;
  }

  explicit CkDeviceBuffer(const void* ptr_, size_t cnt_, const CkCallback& cb_, cudaStream_t cuda_stream_) : CmiDeviceBuffer(ptr_, cnt_) {
    cb = cb_;
    cuda_stream = cuda_stream_;
  }

  void pup(PUP::er &p) {
    CmiDeviceBuffer::pup(p);
    p|cb;
  }

  friend void CkRdmaDeviceIssueRgets(envelope *env, int numops, void **arrPtrs, int *arrSizes, CkDeviceBufferPost *postStructs);
};

#if !CMK_GPU_COMM
void CkRdmaDeviceRecvHandler(void* data, void* msg);
#else
void CkRdmaDeviceRecvHandler(void* data);
#endif
void CkRdmaDeviceIssueRgets(envelope *env, int numops, void **arrPtrs, int *arrSizes, CkDeviceBufferPost *postStructs);
void CkRdmaDeviceOnSender(int dest_pe, int numops, CkDeviceBuffer** buffers);

#endif // CMK_CUDA

#endif /* CMK_RECONVERSE */
#endif
