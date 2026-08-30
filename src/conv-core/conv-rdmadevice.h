#ifndef _CONV_RDMADEVICE_H_
#define _CONV_RDMADEVICE_H_

#include "charm-config.h"
#include "converse.h"
#include "cmirdmautils.h"
#include "pup.h"
#include "conv-rdma.h"

#define CMK_GPU_COMM 1

#if CMK_CUDA || CMK_HIP
#include "hapi_portable.h"

// Represents the mode of device-side zerocopy transfer
// MEMCPY indicates that the PEs are on the same logical node and hapiMemcpyDeviceToDevice can be used
// IPC indicates that the PEs are on different logical nodes within the same physical node and CUDA IPC can be used
// RDMA indicates that the PEs are on different physical nodes and requires GPUDirect RDMA
enum class CmiNcpyModeDevice : char { MEMCPY, IPC, RDMA };

// Status of Direct API (persistent) transfer
enum class CmiDeviceStatus : char { incomplete, complete };

// Which inter-process transport the sender prepared for a device buffer.
//
// STAGED: the bytes were copied into the sender's device communication buffer,
//   whose IPC handle every peer opened once at startup. {device_idx,
//   comm_offset} locate them there. Two device copies, no per-transfer handle
//   work, and the sender's own buffer is free for reuse as soon as its staging
//   copy retires -- which is ordered on the sender's stream, so an application
//   gets that for free without observing completion.
//
// DIRECT: nothing was copied. ipc_handle exports the allocation holding ptr and
//   the receiver copies straight out of it at ipc_offset, saving one device
//   copy and consuming no communication buffer. In exchange the sender's buffer
//   stays live until the receiver signals completion, so an application that
//   reuses or frees it MUST wait for the CkDeviceBuffer's completion callback.
enum class CmiIpcProtocol : char { NONE = 0, STAGED = 1, DIRECT = 2 };

class CmiDeviceBuffer {
public:
  // Pointer to and size of the buffer
  const void* ptr;
  size_t cnt;
  hapiStream_t hapi_stream;

  // Source and destination PEs
  int src_pe;
  int src_mpi_rank;
  int dest_pe;
  int dest_mpi_rank;

  // Used for CUDA IPC
  int device_idx;
  size_t comm_offset;
  int event_idx;

  // Which of the two inter-process transports the sender prepared, and the
  // export that DIRECT needs: a handle for the whole allocation containing ptr
  // (cudaIpcGetMemHandle only ever names allocations, never interior
  // addresses) plus the distance from that allocation's base to ptr.
  CmiIpcProtocol ipc_protocol;
  hapiIpcMemHandle_t ipc_handle;
  size_t ipc_offset;

  // Same-process (MEMCPY) ordering. The receiver reads ptr directly on its own
  // stream, which has no ordering against the stream still producing the data,
  // so the two have to be tied together somehow. This carries an event recorded
  // on the sender's stream for the receiver to wait on -- a raw handle, which is
  // meaningful precisely because MEMCPY means one process. Null when the sender
  // did not record one, in which case the receiver must not assume ordering.
  void* memcpy_event;

  // Store the actual data for host-staged inter-node messaging (no GPUDirect RDMA)
  // Whether the sender prepared a source this buffer's receiver can actually
  // read from another process -- staged/exported for IPC, or registered for
  // RDMA. False means the sender resolved the destination to its own process
  // and chose a plain memcpy, which is only readable in that address space.
  // The receiver compares this against the mode it resolves for itself; the
  // two disagree exactly when the target migrated across a process boundary
  // after the send was posted.
  bool sender_prepared;

  bool data_stored;
  void* data;

  CmiNcpyBuffer lci_ncpy_buffer;

  CmiDeviceBuffer() : ptr(NULL), cnt(0), src_pe(-1), dest_pe(-1) { init(); }

  explicit CmiDeviceBuffer(const void* ptr_, size_t cnt_) : ptr(ptr_), cnt(cnt_),
    src_pe(CmiMyPe()), src_mpi_rank(CmiNodeOf(CmiMyPe())), dest_pe(-1), dest_mpi_rank(-1) { init(); }

  void init() {
    device_idx = -1;
    comm_offset = 0;
    event_idx = -1;
    ipc_protocol = CmiIpcProtocol::NONE;
    ipc_offset = 0;
    memcpy_event = NULL;
    hapi_stream = hapiStreamPerThread;

    sender_prepared = false;
    data_stored = false;
    data = NULL;
  }

  uint64_t tag;

  void pup(PUP::er &p) {
    p((char *)&ptr, sizeof(ptr));
    p|cnt;
    p|src_pe;
    p|dest_pe;
    p|device_idx;
    p|comm_offset;
    p|event_idx;
    p|sender_prepared;
    // The 64-byte handle is only meaningful to a DIRECT receive, and every
    // device send carries one of these descriptors, so pup it conditionally.
    // The width stays a function of ipc_protocol alone, which nothing rewrites
    // -- what CkRdmaDeviceIssueRgets patches in place is ptr -- so the
    // read-then-write retarget there still round-trips to the same size.
    p((char *)&ipc_protocol, sizeof(ipc_protocol));
    if (ipc_protocol == CmiIpcProtocol::DIRECT) {
      p((char *)&ipc_handle, sizeof(ipc_handle));
      p|ipc_offset;
    }
    p((char *)&memcpy_event, sizeof(memcpy_event));
    p|data_stored;
    if (data_stored) {
      if (p.isUnpacking()) {
        hapiMallocHost(&data, cnt);
      }
      PUParray(p, (char*)data, cnt);
    }
    p|tag;
    p|src_pe;
    p|src_mpi_rank;
    p|dest_pe;
    p|dest_mpi_rank;
    p|lci_ncpy_buffer;
  }

  ~CmiDeviceBuffer() {
#if !CMK_GPU_COMM
    if (data) hapiFreeHost(data);
#endif
  }
};

CmiNcpyModeDevice findTransferModeDevice(int srcPe, int destPe);

#if CMK_GPU_COMM
typedef void (*RdmaAckCallerFn)(void *token);

void CmiSendDevice(int dest_rank, int src_rank, const void*& ptr, size_t size, uint64_t& tag);
void CmiRecvDevice(DeviceRdmaOp* op, DeviceRecvType type);
void CmiRdmaDeviceRecvInit(RdmaAckCallerFn fn);
void CmiInvokeRecvHandler(void* data);
#endif // CMK_GPU_COMM
#endif // CMK_CUDA

#endif // _CONV_RDMADEVICE_H_
