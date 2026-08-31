#ifndef _CKRDMADEVICE_H_
#define _CKRDMADEVICE_H_

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

class CkLocRec;
CkpvExtern(CkLocRec*, _currentLocRec);

// Device registration cache, off unless CHARM_DEVICE_MR_CACHE is set; see the
// design note in ckrdmadevice.C. DropRegistrations must be called wherever an
// element's device buffers are about to be freed, and only where no transfer
// against them is in flight.
void CkRdmaDeviceRegistrationCacheInit();
// Stalled-receive watchdog, off unless CHARM_ZC_STALL_SECS is set.
void CkRdmaDeviceStallWatchInit();

// A correction defers a receive across a round trip; these count it against the
// receiving element so migration stands down until it lands. See the note at
// the definitions in cklocation.C.
void CkNoteDeviceRecvDeferred(CkGroupID aid, CmiUInt8 id);
void CkNoteDeviceRecvComplete(CkGroupID aid, CmiUInt8 id);
void CkRdmaDeviceDropRegistrations(CkLocRec* owner);

extern "C" {
  void* loopback_bridge(void* arg);
  extern int loopback_handler;
  // Migration-mismatch payload correction; see the protocol notes in
  // ckrdmadevice.C.
  void* device_restage_req_bridge(void* arg);
  void* device_restage_meta_bridge(void* arg);
  void* device_restage_put_done_bridge(void* arg);
  extern int device_restage_req_handler;
  extern int device_restage_meta_handler;
  extern int device_restage_put_done_handler;
}

#endif // CMK_CUDA

// Returns and clears the device payload size recorded by the most recent
// CkRdmaDeviceOnSender on this PE, so the LB communication graph can weight a
// device zerocopy edge by what actually crosses the wire rather than by the
// descriptor envelope. Returns 0 for a non-device send. Declared outside the
// CUDA guard so the send path can call it without an #ifdef.
#if CMK_CUDA || CMK_HIP
size_t CkRdmaDeviceTakePendingSendBytes();
// Claimed IPC event slots for this PE, or -1 when shm IPC is not in use.
int CkRdmaDeviceBusyIpcSlots();
// Allocate `size` bytes from the device load-balance region of `dm`'s buffer,
// reclaiming retired IPC slots and retrying once if the first attempt does not
// fit. Returns nullptr if it still does not. Handles dm->lock itself, so the
// caller must not hold it.
//
// The migration path allocates from this region but never sends through
// acquireIpcSendSlot, so it has no other way to reach the scan that hands these
// blocks back. `dm` is a DeviceManager*, passed opaquely so this declaration
// does not drag the HAPI headers into every includer.
void* CkRdmaDeviceAllocLbBuffer(void* dm, size_t size);
#else
inline size_t CkRdmaDeviceTakePendingSendBytes() { return 0; }
inline int CkRdmaDeviceBusyIpcSlots() { return -1; }
inline void* CkRdmaDeviceAllocLbBuffer(void* dm, size_t size) { return nullptr; }
#endif

#endif // _CKRDMADEVICE_H_
