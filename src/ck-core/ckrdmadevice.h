#ifndef _CKRDMADEVICE_H_
#define _CKRDMADEVICE_H_

/* Two-runtime header (same pattern as conv-rdmadevice.h / cmirdmautils.h). */
#include "charm-config.h"

/* Stage 9.1/9.2 boundary, single source of truth.
 *
 * ckrdmadevice.C holds the classic D2D implementation, written against the
 * classic half of this header (cuda_* fields, 4-param CmiSendDevice); the
 * reconverse D2D port arrives with stage 9.2 (see
 * doc/reconverse-migration-ledger.README.md). Until then the file compiles
 * empty everywhere else.
 *
 * Every site that references a symbol *defined* in that file must test this
 * macro, not re-derive the condition. Two guards that drifted apart are what
 * broke the first reconverse+GPU link: the file was excised under
 * `CMK_CUDA && !CMK_RECONVERSE` while init.C still registered
 * loopback_bridge/loopback_handler under `(CMK_CUDA || CMK_HIP) &&
 * CMK_GPU_COMM`, so both reconverse+CUDA and any HIP build failed with those
 * two symbols undefined. Retire this macro with stage 9.2. */
#if CMK_CUDA && !CMK_RECONVERSE
#define CMK_CKRDMADEVICE_CLASSIC_D2D 1
#else
#define CMK_CKRDMADEVICE_CLASSIC_D2D 0
#endif

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
  }

  friend void CkRdmaDeviceIssueRgets(envelope *env, int numops, void **arrPtrs, int *arrSizes, CkDeviceBufferPost *postStructs);
};

void CkRdmaDeviceRecvHandler(void* data);
void CkRdmaDeviceRecvHandler(void* data, void* msg);
void CkRdmaDeviceIssueRgets(envelope *env, int numops, void **arrPtrs, int *arrSizes, CkDeviceBufferPost *postStructs);
void CkRdmaDeviceOnSender(int dest_pe, int numops, CkDeviceBuffer** buffers);

extern "C" {
  void* loopback_bridge(void* arg);
  extern int loopback_handler;
}

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
