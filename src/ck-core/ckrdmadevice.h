#ifndef _CKRDMADEVICE_H_
#define _CKRDMADEVICE_H_

/* Charm++-level device zerocopy over conv-rdmadevice.h. */
#include "charm-config.h"


#include "ckcallback.h"
#include "conv-rdmadevice.h"

#if CMK_CUDA || CMK_HIP
#include "hapi_portable.h"

#define CkNcpyModeDevice CmiNcpyModeDevice
#define CkDeviceStatus CmiDeviceStatus

// Persistent device buffers (manual: "Persistent GPU Communication"; mechanism:
// doc/internals/ckdyn.md, "Persistent device buffers"). get/put choose a mode
// per pair (mode()): a device copy within a process, CUDA/HIP IPC between
// processes on one host with +gpushm, RDMA on the memory registered by open()
// otherwise. Both buffers' callbacks are sent from the issuing PE once the
// bytes are in the destination.
struct CkDevicePersistent {
  const void* ptr;
  size_t cnt;
  CkCallback cb;
  void* cb_msg;
  hapiStream_t hapi_stream = 0;   // the default stream unless a constructor names one
  int pe;
  hapiIpcMemHandle_t hapi_ipc_handle;
  void* ipc_ptr;      // unused; the mapping of a remote buffer is kept per process
  bool ipc_open;      // unused
  // set by open() and carried by pup
  size_t ipc_offset = 0;    // ptr minus the base of its allocation (what the IPC handle maps)
  bool ipc_valid = false;   // hapi_ipc_handle names ptr's allocation
  int node = -1;            // the process (logical node) of pe
  int phys_node = -1;       // the host of pe
  bool registered = false;  // ncpy holds a registration of [ptr, ptr + cnt)
  CmiNcpyBuffer ncpy;
  bool owner = false;       // this object opened the buffer (not pupped)

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
  // Takes the IPC handle of the allocation and, in a program of more than one
  // process, registers the memory for RDMA. Call before sending the object.
  void open();
  // On the object that opened the buffer: releases the registration. On a
  // copy received from a peer: nothing (IPC mappings are kept per process).
  void close();
  void set_msg(void* msg);

  // Should only be used for exchanging between chares, not for migration.
  // After the owner chare migrates, CkDevicePersistent needs to be recreated
  // and exchanged again.
  void pup(PUP::er& p);

  // 0 copy within a process, 1 IPC, 2 RDMA: how get/put with other moves the bytes
  int mode(const CkDevicePersistent& other) const;

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
  // of a per-message registration and/or the source callback.  Its top bit
  // says the id carries a source callback, which the receiver returns at
  // once rather than holding for the piggyback.  acks[] is the block of ids
  // this MESSAGE carries back to its destination PE; only the first buffer
  // of a message carries any, the rest pup one zero byte.
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
  void* persist_put_bridge(void* arg);
  extern int persist_put_handler;
  void* persist_ack_bridge(void* arg);
  extern int persist_ack_handler;
}

// Device pool: arenas registered lazily, whole, on first network use.
// Every other device buffer is registered per message and released on
// completion; holding registrations across messages is the network layer's
// registration cache to do.
void* CkDeviceMalloc(size_t size);
void CkDeviceFree(void* ptr);
// Per-PE init of the device zerocopy path (idle-time ack flush).
void CkRdmaDeviceInit();

#endif // CMK_CUDA

#endif
