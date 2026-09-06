#ifndef _CKRDMADEVICE_H_
#define _CKRDMADEVICE_H_

#include "ckcallback.h"
#include "conv-rdmadevice.h"

#if CMK_CUDA || CMK_HIP
#include "hapi_portable.h"
#include <functional>

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
// Generated asynchronous proxies use the four-argument prepare followed by
// SendWhenReady. The latter owns the marshalled message and continuation until
// all producer streams complete. The three-argument API remains synchronous.
struct CkDeviceDeferredSend;
void CkRdmaDeviceOnSender(int dest_pe, int numops, CkDeviceBuffer** buffers,
                          CkDeviceDeferredSend** pending);
void CkRdmaDeviceSendWhenReady(CkDeviceDeferredSend* pending, void* msg,
                              const std::function<void()>& send);
// A device-send message that is leaving this process for another one on the
// same physical node: re-prepare every memcpy-prepared payload in it as a
// direct IPC send, in place, so the new home reads it without a correction.
// Returns true when the message was redirected (consumed) rather than
// repaired in place; the caller must then not deliver it.
bool CkRdmaDeviceRepairForward(envelope* env, int newPe);
extern "C" void* device_forward_redirect_bridge(void* arg);
extern "C" int device_forward_redirect_handler;

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
// The element a device zerocopy receive is addressed to, or NULL. Lets the
// registration cache attribute a receive-side registration to the element
// whose migration must retire it (see acquireDeviceRegistration).
CkLocRec* CkFindDeviceRecvElement(CkGroupID aid, CmiUInt8 id);
// Deferred-migration kick, enqueued by CkLocRec::noteDeviceSendDone so a
// migration never starts inside the handler that released it.
extern int _deferredMigrateHandlerIdx;
extern "C" void _deferredMigrateHandler(void* arg);
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

// Device pool: CkDeviceMalloc / CkDeviceFree hand out device buffers from
// arenas -- one large device allocation each, managed by the buddy allocator
// that already backs the communication buffer, grown on demand (CK_GPU_ARENA_MB,
// default 256), one arena set per device. Both calls are host-side bookkeeping:
// no cudaMalloc and no cudaFree per buffer, so neither synchronizes the device.
//
// Ported from device-rdma-dereg-3960 (8580c18ba, a58bf3ce4) without that
// branch's lazy whole-arena registration, which hooks into an interval map of
// registered regions this branch does not have.
//
// The pool answers the allocation-cost question only. CkDeviceFree returns the
// block at once with no stream ordering, so a buffer with device work still in
// flight against it must not be freed -- the caller owns that, exactly as with
// cudaFreeAsync. A buffer that was rebound into a migration arena by
// pup_buffer_device is NOT from this pool and must go back through
// hapiFreeMigratable; callers that mix the two have to know which is which.
void* CkDeviceMalloc(size_t size);
void CkDeviceFree(void* ptr);
// +gpupool: the pool is the runtime's allocation policy (migration arenas and
// payloads from it, direct CUDA IPC only, no comm or load-balancing buffer).
// An application that follows the runtime's choice allocates through
// CkDeviceMalloc when this is true and through hapiMalloc otherwise.
bool CkDevicePoolOn();
// Set by the location manager around the send of a migration payload: the
// runtime's own buffer, released by the receiver's ack rather than by a
// completion callback, so the "direct send with no callback" warning does not
// apply to it.
void CkRdmaDeviceMarkMigrationPayload(bool sending);

// Load balance pool block recycling, without a host barrier.
//
// A block handed out by CkRdmaDeviceAllocLbBuffer had a previous life, and the
// device work of that life -- the landing write into it, and the unpack copies
// out of it -- can still be in flight when it is handed out again. Until now
// that was covered by synchronizing the host, which waits for far more than
// this block and stops the scheduler to do it.
//
// Instead: when a block goes back to the pool, record an event on the stream
// whose work touched it (Freed). When one comes out, make the stream that is
// about to write into it wait on those events (Gate). The ordering is then
// exact, expressed on the device, and costs no host time. Events are reused
// per (device, stream), so this allocates nothing per migration.
void CkRdmaDeviceNoteLbBufferFreed(void* dm, cudaStream_t usedBy);
void CkRdmaDeviceGateLbBuffer(void* dm, cudaStream_t consumer);
#else
inline size_t CkRdmaDeviceTakePendingSendBytes() { return 0; }
inline int CkRdmaDeviceBusyIpcSlots() { return -1; }
inline void* CkRdmaDeviceAllocLbBuffer(void* dm, size_t size) { return nullptr; }
inline void* CkDeviceMalloc(size_t size) { return nullptr; }
inline void CkDeviceFree(void* ptr) {}
inline bool CkDevicePoolOn() { return false; }
inline void CkRdmaDeviceMarkMigrationPayload(bool sending) {}
inline void CkRdmaDeviceNoteLbBufferFreed(void* dm, cudaStream_t usedBy) {}
inline void CkRdmaDeviceGateLbBuffer(void* dm, cudaStream_t consumer) {}
#endif

#endif // _CKRDMADEVICE_H_
