/*
 * Direct GPU Messaging
 *
 * Uses host-bypass mechanisms to directly transfer data between GPU devices.
 *
 * 1) Intra-process (intra-node): The sender sends a metadata message to the
 *    receiver containing the pointer to the source GPU buffer. There is
 *    no setup needed on the sender side. The receiver invokes device-to-device
 *    transfer from the source GPU buffer to the destination GPU buffer.
 *
 * 2) Inter-process (intra-node): The pointer to the source GPU buffer will be
 *    invalid on the receiver as it is a different process. Thus CUDA IPC is
 *    used to create a handle to the source GPU buffer, which can be opened on
 *    the receiver side to initiate the data transfer. To mitigate the overheads
 *    of creating and destroying IPC handles, the runtime first allocates a
 *    'device communication buffer' on each GPU device, creates IPC handles
 *    only for these buffers, and then exchanges the handles between processes
 *    on the same physical node. This means each process will have IPC handles
 *    for all device communication buffers on the same host (that are potentially
 *    managed by other processes) and can perform data transfers using these
 *    handles. Each GPU-GPU data transfer invovles requesting a block from the
 *    device communication buffer on the sender, copying the source GPU buffer
 *    to the allocated block, sending a metadata message to the receiver
 *    (that contains the offset of the allocated block), and performing a
 *    transfer from the block on the sender's device communication buffer to
 *    the destination GPU buffer. CUDA Events are used to enforce the correct
 *    ordering between these data transfers. Because multiple PEs can be mapped
 *    to the same GPU and hence concurrently request allocations from the same
 *    device communication buffer, a thread-safe allocator using the buddy
 *    allocation algorithm was implemented. The allocator first calls hapiMalloc
 *    to obtain a relatively large chunk of memory and then services allocation
 *    and deallocation requests from PEs that are mapped to its GPU device.
 *    The buddy algorithm was used to minimize the external fragmentation that
 *    could occur from concurrent manipulations of the device communication
 *    buffer.
 *
 * TODO
 * 3) Inter-node: This currently uses a simple host-staged mechanism to perform
 *    a device-to-host copy of the source GPU buffer to a message, which is sent
 *    to the receiver. The receiver then performs a host-to-device copy to the
 *    destination GPU buffer. This will be updated to use GPUDirect RDMA to
 *    directly performa true device-to-device transfer.
 */

#ifndef _WIN32
#include <pthread.h>
#endif
#include "envelope.h"
#include "charm++.h"
#include "ck.h"
#include "ckrdmadevice.h"

#define CMK_GPU_COMM 1

#if CMK_CUDA || CMK_HIP

CmiNcpyModeDevice findTransferModeDevice(int srcPe, int dstPe) {
  CmiEnforce((srcPe >= 0) && (srcPe <= CmiNumPes()));
  CmiEnforce((dstPe >= 0) && (dstPe <= CmiNumPes()));

  if (CmiNodeOf(srcPe) == CmiNodeOf(dstPe)) {
    // Same logical node
    return CmiNcpyModeDevice::MEMCPY;
  } else if (CmiPeOnSamePhysicalNode(srcPe, dstPe)) {
    // Different logical nodes, same physical node
    return CmiNcpyModeDevice::IPC;
  } else {
    // Different physical nodes, requires GPUDirect RDMA
    return CmiNcpyModeDevice::RDMA;
  }
}

#include <atomic>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "hapi.h"
#include "gpumanager.h"

CsvExtern(GPUManager, gpu_manager);
CpvExtern(int, my_device_id);

// void CkRdmaDeviceRecvHandler(void* data)
// {
//   DeviceRdmaOp* op = (DeviceRdmaOp*)data;
//   DeviceRdmaInfo* info = op->info;

//   // Invoke source callbacks
//   if (op->src_cb) {
//     int rank;
//     CkCallback* cb = (CkCallback*)op->src_cb;
//     cb->send();
//     delete cb;
//   }

//   // Update counter (there may be multiple buffers in transit)
//   info->counter++;

//   // Check if all buffers have been received
//   // If so, invoke regular entry method
//   if (info->counter == info->n_ops) {
//     QdCreate(1);

//     enqueueNcpyMessage(op->dest_pe, info->msg);

//     // Free RDMA metadata
//     CmiFree(info);
//   }
// }

struct LoopBackMsg {
  char header[CmiMsgHeaderSizeBytes];
  void* msg;
};

extern "C" {
  void* loopback_bridge(void* arg) {
    QdProcess(1);
    LoopBackMsg* recv_msg = (LoopBackMsg*)arg;
    CkRdmaDeviceRecvHandler(recv_msg->msg);
    CmiFree(recv_msg);
    return NULL;
  }
  
  int loopback_handler;
}

void CkRdmaDeviceRecvHandler(void* data)
{
  NcpyOperationInfo *ncpy_op_info = (NcpyOperationInfo *)data;
  DeviceRdmaOp* op = (DeviceRdmaOp*)(ncpy_op_info->deviceRdmaOpInfo);

  if(op->dest_pe != CmiMyPe()) {
        int infoSize = ncpy_op_info->ncpyOpInfoSize;
        NcpyOperationInfo* copy = (NcpyOperationInfo*)CmiAlloc(infoSize);
        memcpy(copy, ncpy_op_info, infoSize);

        LoopBackMsg* conv_msg = (LoopBackMsg*)CmiAlloc(sizeof(LoopBackMsg));
        conv_msg->msg = copy;

        QdCreate(1);
        CmiSetHandler(conv_msg, loopback_handler);
        CmiPushPE(CmiRankOf(op->dest_pe), conv_msg);
        return;
  }

  QdProcess(1);
  DeviceRdmaInfo* info = op->info;

  // Invoke source callbacks
  if (op->src_cb) {
    CkCallback* cb = (CkCallback*)op->src_cb;
    cb->send();
    delete cb;
  }

  // Update counter (there may be multiple buffers in transit)
  info->counter++;

  // Check if all buffers have been received
  // If so, invoke regular entry method
  if (info->counter == info->n_ops) {
    QdCreate(1);

    enqueueNcpyMessage(op->dest_pe, info->msg);

    // Free RDMA metadata
    // CmiFree(info);
  }
}
// Invoked when a GPU buffer arrives on the receiver
void CkRdmaDeviceRecvHandler(void* data, void* msg)
{
  DeviceRdmaOp* op = (DeviceRdmaOp*)data;
  DeviceRdmaInfo* info = op->info;

  // Invoke source callbacks
  if (op->src_cb) {
    CkCallback* cb = (CkCallback*)op->src_cb;
    cb->send();
    delete cb;
  }

  // Update counter (there may be multiple buffers in transit)
  info->counter++;

  // Check if all buffers have been received
  // If so, invoke regular entry method
  if (info->counter == info->n_ops) {
    QdCreate(1);

    enqueueNcpyMessage(op->dest_pe, info->msg);

    // Free RDMA metadata
    CmiFree(info);
  }
}

/****************************** Direct (Persistent) API ******************************/

void CkDevicePersistent::init() {
  pe = CkMyPe();
  cb_msg = nullptr;
  ipc_ptr = nullptr;
  ipc_open = false;
}

void CkDevicePersistent::open() {
  // Create a CUDA IPC handle for inter-process communication
  hapiCheck(hapiIpcGetMemHandle(&hapi_ipc_handle, (void*)ptr));
}

void CkDevicePersistent::close() {
  // Close the CUDA IPC handle if it was opened
  hapiCheck(hapiIpcCloseMemHandle(ipc_ptr));
}

void CkDevicePersistent::set_msg(void* msg) {
  cb_msg = msg;
}

void CkDevicePersistent::pup(PUP::er& p) {
  p((char*)&ptr, sizeof(ptr));
  p|cnt;
  p|pe;
  p|cb;
  p((char*)&hapi_ipc_handle, sizeof(hapi_ipc_handle));
}

CkDeviceStatus CkDevicePersistent::get(CkDevicePersistent& src) {
  // Check that the source buffer fits into the destination buffer
  if (cnt < src.cnt) {
    CkAbort("CkDevicePersistent::get: Destination buffer is smaller than source buffer\n");
  }

  CkNcpyModeDevice mode = findTransferModeDevice(src.pe, CkMyPe());

  // Perform get
  if (mode == CkNcpyModeDevice::MEMCPY) {
    hapiMemcpyAsync((void*)ptr, src.ptr, cnt, hapiMemcpyDeviceToDevice, hapi_stream);
  } else if (mode == CkNcpyModeDevice::IPC) {
    if (!src.ipc_open) {
      hapiCheck(hapiIpcOpenMemHandle(&src.ipc_ptr, src.hapi_ipc_handle,
            hapiIpcMemLazyEnablePeerAccess));
      src.ipc_open = true;
    }
    hapiMemcpyAsync((void*)ptr, src.ipc_ptr, cnt, hapiMemcpyDeviceToDevice, hapi_stream);
  } else {
    CkAbort("Persistant GPU messaging is currently not supported for inter-node messages");
  }

  // Set callbacks to be invoked once get is complete
  if (src.cb.type != CkCallback::ignore) {
    hapiAddCallback(hapi_stream, src.cb, src.cb_msg);
  }
  if (cb.type != CkCallback::ignore) {
    hapiAddCallback(hapi_stream, cb, cb_msg);
  }

  return CkDeviceStatus::incomplete;
}

CkDeviceStatus CkDevicePersistent::put(CkDevicePersistent& dst) {
  // Check that the source buffer fits into the destination buffer
  if (dst.cnt < cnt) {
    CkAbort("CkDevicePersistent::put: Destination buffer is smaller than source buffer\n");
  }

  CkNcpyModeDevice mode = findTransferModeDevice(CkMyPe(), dst.pe);

  // Perform put
  if (mode == CkNcpyModeDevice::MEMCPY) {
    hapiMemcpyAsync((void*)dst.ptr, ptr, cnt, hapiMemcpyDeviceToDevice, hapi_stream);
  } else if (mode == CkNcpyModeDevice::IPC) {
    if (!dst.ipc_open) {
      hapiCheck(hapiIpcOpenMemHandle(&dst.ipc_ptr, dst.hapi_ipc_handle,
            hapiIpcMemLazyEnablePeerAccess));
      dst.ipc_open = true;
    }
    hapiMemcpyAsync(dst.ipc_ptr, ptr, cnt, hapiMemcpyDeviceToDevice, hapi_stream);
  } else {
    CkAbort("Persistant GPU messaging is not yet supported for inter-node messages");
  }

  // Set callbacks to be invoked once get is complete
  if (cb.type != CkCallback::ignore) {
    hapiAddCallback(hapi_stream, cb, cb_msg);
  }
  if (dst.cb.type != CkCallback::ignore) {
    hapiAddCallback(hapi_stream, dst.cb, dst.cb_msg);
  }

  return CkDeviceStatus::incomplete;
}

/****************************** Recv Entry Method API ******************************/

// Returns the local rank of the logical node (process) that the given PE belongs to
static inline int CmiNodeRankLocal(int pe) {
  // Logical node index % Number of logical nodes per physical node
  return CmiNodeOf(pe) % (CmiNumNodes() / CmiNumPhysicalNodes());
}

// Returns the local rank of the logical node that I belong to
static inline int CmiMyNodeRankLocal() {
  return CmiNodeRankLocal(CmiMyPe());
}

// TEMPORARY (CHARM_DEBUG_IPC_RECV): synchronize after each individual CUDA
// operation on the shm/IPC path and abort naming the exact step that failed.
// Illegal-access errors are sticky and asynchronous, so without this they
// surface at whatever call happens to be checked next -- which is how the same
// fault has been reported from three unrelated lines. With it, the first
// failing operation identifies itself.
static inline void ipcDebugSync(const char* step, hapiStream_t stream) {
  if (!getenv("CHARM_DEBUG_IPC_RECV")) return;
  hapiError_t err = hapiStreamSynchronize(stream);
  if (err != hapiSuccess) {
    CmiPrintf("[%d] IPC step '%s' FAILED: %s\n", CmiMyPe(), step,
              cudaGetErrorString(err));
    fflush(stdout);
    CmiAbort("IPC debug: step '%s' failed", step);
  }
}

// Per-process tally of how device zerocopy receives actually resolved, enabled
// with CHARM_ZC_STATS=1. Says whether a placement decision (e.g. a
// communication-aware load balancer) turned cross-process IPC transfers into
// same-process device-to-device copies, which is the locality question that
// raw timings alone cannot answer. Counting is a relaxed atomic increment on a
// path that already issues a CUDA call, and the env lookup is cached, so an
// instrumented run stays representative.
namespace {
struct ZcModeStats {
  std::atomic<long> memcpy_n{0};
  std::atomic<long> ipc_n{0};
  std::atomic<long> other_n{0};
  ~ZcModeStats() {
    const long m = memcpy_n.load(), i = ipc_n.load(), o = other_n.load();
    const long total = m + i + o;
    if (total == 0) return;
    fprintf(stderr, "[zc-stats] pid=%d MEMCPY=%ld IPC=%ld OTHER=%ld "
                    "(same-process %.1f%% of %ld)\n",
            (int)getpid(), m, i, o, 100.0 * m / total, total);
    fflush(stderr);
  }
};
ZcModeStats zc_mode_stats;

// Sender-side companion: how often the destination could not be resolved at
// send time. An unresolved destination cannot take the cheap same-process path,
// so a burst of these right after a migration is what turns the first
// post-migration step into a slow one.
struct ZcDestStats {
  std::atomic<long> confirmed{0};
  std::atomic<long> unconfirmed{0};
  ~ZcDestStats() {
    const long c = confirmed.load(), u = unconfirmed.load();
    if (c + u == 0) return;
    fprintf(stderr, "[zc-dest] pid=%d confirmed=%ld unconfirmed=%ld (%.2f%% unresolved)\n",
            (int)getpid(), c, u, 100.0 * u / (c + u));
    fflush(stderr);
  }
};
ZcDestStats zc_dest_stats;

inline void zcDestCount(bool confirmed) {
  static const bool on = (getenv("CHARM_ZC_STATS") != nullptr);
  if (!on) return;
  if (confirmed) zc_dest_stats.confirmed.fetch_add(1, std::memory_order_relaxed);
  else zc_dest_stats.unconfirmed.fetch_add(1, std::memory_order_relaxed);
}

inline void zcStatsCount(CkNcpyModeDevice mode) {
  static const bool on = (getenv("CHARM_ZC_STATS") != nullptr);
  if (!on) return;
  switch (mode) {
    case CkNcpyModeDevice::MEMCPY: zc_mode_stats.memcpy_n.fetch_add(1, std::memory_order_relaxed); break;
    case CkNcpyModeDevice::IPC:    zc_mode_stats.ipc_n.fetch_add(1, std::memory_order_relaxed); break;
    default:                       zc_mode_stats.other_n.fetch_add(1, std::memory_order_relaxed); break;
  }
}
}  // namespace

// Per-PE ring of events used only to order same-process transfers.
//
// A wait enqueued on an event captures the most recent record at enqueue time,
// and the sender always records before the message is sent, so the receiver's
// wait cannot run ahead of the record. Re-recording an event that some earlier
// receiver is still waiting on is therefore harmless: that wait already
// captured the earlier record. The ring only needs to be large enough that
// re-use does not add false dependencies, not for correctness.
static hapiEvent_t* ckDeviceMemcpyEventRing() {
  static const int ring_size = []() {
    const char* s = getenv("CHARM_ZC_MEMCPY_EVENTS");
    const int n = s ? atoi(s) : 1024;
    return n > 0 ? n : 1024;
  }();
  static thread_local hapiEvent_t* ring = nullptr;
  static thread_local int next = 0;
  if (ring == nullptr) {
    ring = new hapiEvent_t[ring_size];
    for (int i = 0; i < ring_size; i++) {
      if (hapiEventCreateWithFlags(&ring[i], hapiEventDisableTiming) != hapiSuccess) {
        delete[] ring;
        ring = nullptr;
        return (hapiEvent_t*)nullptr;
      }
    }
  }
  return ring;
}

void* ckDeviceRecordMemcpyEvent(hapiStream_t stream) {
  static const int ring_size = []() {
    const char* s = getenv("CHARM_ZC_MEMCPY_EVENTS");
    const int n = s ? atoi(s) : 1024;
    return n > 0 ? n : 1024;
  }();
  static thread_local int next = 0;
  hapiEvent_t* ring = ckDeviceMemcpyEventRing();
  if (ring == nullptr) return NULL;
  hapiEvent_t ev = ring[next];
  next = (next + 1) % ring_size;
  if (hapiEventRecord(ev, stream) != hapiSuccess) return NULL;
  return (void*)ev;
}

// Invoked after post entry method
void CkRdmaDeviceIssueRgets(envelope *env, int numops, void **arrPtrs, int *arrSizes, CkDeviceBufferPost *postStructs) {
  // Change message header to invoke regular entry method
  CMI_ZC_MSGTYPE(env) = CMK_REG_NO_ZC_MSG;

  // Create a copy of this message for regular entry method invocation
  // FIXME: Reuse the old message instead of creating a new one
  void* old_msg = EnvToUsr(env);
  envelope* new_env = UsrToEnv(CkCopyMsg(&old_msg));

  // Retarget the copied message's device buffers to the buffers this receiver
  // posted. Set CHARM_NO_ZC_RETARGET to skip this and restore the previous
  // behaviour (entry method sees the SENDER's pointer) for A/B testing.
  // The transfers below land in arrPtrs[], but the copy still carries
  // the SENDER's CkDeviceBuffer::ptr, and the entry method delivered from it
  // reads that pointer as its data. Within one process the sender's pointer is
  // a valid local address holding the same bytes, so this went unnoticed;
  // across processes it names memory in another address space and the first
  // kernel touching it faults. Rewriting in place is safe because pupping a
  // CkDeviceBuffer is fixed-width -- only ptr changes value.
  static const bool zc_retarget_off = (getenv("CHARM_NO_ZC_RETARGET") != nullptr);
  if (!zc_retarget_off) {
    char* new_buf = ((CkMarshallMsg*)EnvToUsr(new_env))->msgBuf;
    PUP::fromMem walk(new_buf);
    int copy_numops;
    walk | copy_numops;
    for (int i = 0; i < numops && i < copy_numops; i++) {
      const size_t field_off = walk.size();
      CkDeviceBuffer db;
      walk | db;
      const size_t field_len = walk.size() - field_off;
      db.ptr = arrPtrs[i];
      PUP::toMem patch(new_buf + field_off);
      patch | db;
      // Rewriting in place is only sound if packing a CkDeviceBuffer produces
      // exactly as many bytes as unpacking consumed. If that ever stops being
      // true, this would silently overwrite the marshalled parameters that
      // follow (ref/dir/n) instead of just the pointer.
      if (patch.size() != field_len)
        CkAbort("CkRdmaDeviceIssueRgets: device buffer pup asymmetry "
                "(read %zu bytes, wrote %zu) -- in-place retarget is unsafe",
                field_len, patch.size());
    }
  }

  // Start unpacking marshalled message
  PUP::fromMem up((void *)((CkMarshallMsg *)EnvToUsr(env))->msgBuf);
  int received_numops;
  up|received_numops;
  CkAssert(numops == received_numops);

  CkDeviceBuffer source;

  // Machine layer does not support GPU-aware communication
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

  // Find which mode of transfer should be used
  // CmiPrintf("[%d] CkRdmaDeviceOnSender: src_pe=%d, dst_pe=%d\n", CkMyPe(), env->getSrcPe(), CkMyPe());
  CkNcpyModeDevice mode = findTransferModeDevice(env->getSrcPe(), CkMyPe());

  // Allocate and fill in metadata for this zerocopy operation
  void* rdma_data = CmiAlloc(sizeof(DeviceRdmaInfo) + sizeof(DeviceRdmaOp) * numops);
  CmiEnforce(rdma_data);
  DeviceRdmaInfo* rdma_info = (DeviceRdmaInfo*)rdma_data;
  rdma_info->n_ops = numops;
  rdma_info->counter = 0;
  rdma_info->msg = new_env;

  for (int i = 0; i < numops; i++) {
    // Unpack source buffer from sender
    up|source;

    if (arrSizes[i] > source.cnt) {
      CkAbort("CkRdmaDeviceIssueRgets: posted data size is larger than source data size!");
    }

    // Store information about this buffer
    DeviceRdmaOp& save_op = *(DeviceRdmaOp*)((char*)rdma_data
        + sizeof(DeviceRdmaInfo) + sizeof(DeviceRdmaOp) * i);
    // Use the PE we are actually running on, not the one the sender recorded.
    // If the target chare migrated after the sender posted this transfer, the
    // sender's dest_pe is stale; the message itself has already been routed
    // here by the location manager, so this PE is the one that hosts the chare
    // and that posted arrPtrs below.
    save_op.dest_pe  = CkMyPe();
    save_op.dest_ptr = arrPtrs[i];
    save_op.size = (size_t)arrSizes[i];
    save_op.info = rdma_info;
    save_op.src_cb = (source.cb.type != CkCallback::ignore) ? new CkCallback(source.cb) : nullptr;
    save_op.dst_cb = nullptr;

    // A mismatch here means the target chare migrated between the sender
    // posting this transfer and its arrival. That is expected and safe: the
    // transfer mode above is derived from CkMyPe(), the destination buffers
    // are the ones this PE posted, and the message reached this PE precisely
    // because the location manager knows the chare lives here now. Previously
    // this aborted, which made any migration concurrent with a GPU-direct send
    // fatal.
    if (source.dest_pe != CkMyPe() && _lb_args.debug() > 1) {
      CmiPrintf("[%d] CkRdmaDeviceIssueRgets: sender addressed PE %d (src PE %d); "
                "chare has since migrated here, retargeting.\n",
                CkMyPe(), source.dest_pe, env->getSrcPe());
    }

    // Destination buffer (on this receiver)
    CkDeviceBuffer dest((const void *)arrPtrs[i], arrSizes[i]);

    // Perform data transfers.
    //
    // `mode` (computed above from the true, always-accurate delivery PEs) is
    // authoritative -- it is never a guess, unlike the sender's transfer_mode
    // decision (see CkRdmaDeviceOnSender), which can be made without knowing
    // the destination. CkRdmaDeviceOnSender's job is to make sure whatever
    // `mode` picks here finds the metadata it needs already staged: it always
    // stages IPC info when it isn't certain MEMCPY suffices and RDMA can be
    // ruled out, so an unconfirmed-destination send still leaves
    // source.device_idx valid for the IPC branch below.
    zcStatsCount(mode);

    // CHARM_ZC_VALIDATE: check both pointers before handing them to CUDA. An
    // illegal access raised by the copies below is asynchronous and sticky, so
    // it otherwise surfaces at some unrelated later call and says nothing about
    // which side was bad. Migration is what makes this ambiguous: the source
    // buffer can be freed by an object that migrated away, and the destination
    // is whatever the (possibly just-migrated) receiver posted.
    if (getenv("CHARM_ZC_VALIDATE")) {
      cudaPointerAttributes sattr{}, dattr{};
      const cudaError_t serr = cudaPointerGetAttributes(&sattr, source.ptr);
      const cudaError_t derr = cudaPointerGetAttributes(&dattr, (void*)dest.ptr);
      const bool sbad = (serr != cudaSuccess || sattr.type == cudaMemoryTypeUnregistered);
      const bool dbad = (derr != cudaSuccess || dattr.type == cudaMemoryTypeUnregistered);
      // A staged IPC receive reads the sender's comm buffer through an imported
      // handle, never source.ptr, so source.ptr being unmapped here is the normal
      // cross-process case and says nothing. Only report it when this receive
      // would actually dereference it.
      const bool src_will_be_read =
          !(source.device_idx != -1 && csv_gpu_manager.use_shm);
      if ((sbad && src_will_be_read) || dbad) {
        CmiPrintf("[%d] ZC VALIDATE FAIL mode=%d srcPe=%d src=%p(%s type=%d) "
                  "dst=%p(%s type=%d) cnt=%zu dev_idx=%d\n",
                  CkMyPe(), (int)mode, env->getSrcPe(),
                  source.ptr, sbad ? "BAD" : "ok", (int)sattr.type,
                  (void*)dest.ptr, dbad ? "BAD" : "ok", (int)dattr.type,
                  (size_t)dest.cnt, source.device_idx);
        fflush(stdout);
        cudaGetLastError();  // clear so the report is not itself sticky
      }
    }

    // Prefer whatever the sender actually staged over the mode derived here.
    // A staged copy already holds the bytes, so it survives the sending chare
    // migrating away and freeing its source buffer -- which leanmd does on
    // every LB step: Compute::sendForces hands out CkDeviceBuffer(d_force) with
    // no completion callback and calls AtSync immediately after, so ~Compute
    // releases d_force while receivers are still pulling from it. Reading
    // source.ptr in that window is a use-after-free on the device.
    const bool sender_staged =
        (source.device_idx != -1 && csv_gpu_manager.use_shm);

    if (mode == CkNcpyModeDevice::MEMCPY && !sender_staged) {
      // Source and destination PEs are in the same process (logical node)
      // Directly invoke memcpy from source buffer to destination buffer.
      // Order against the sender's stream first: without this the copy could
      // run before the kernels that produced the source data. A null event
      // means the sender blocked instead, so the data is already there.
      if (source.memcpy_event != NULL) {
        hapiCheck(hapiStreamWaitEvent(postStructs[i].hapi_stream,
              (hapiEvent_t)source.memcpy_event, 0));
      }
      hapiCheck(hapiMemcpyAsync((void*)dest.ptr, source.ptr, dest.cnt,
            hapiMemcpyDeviceToDevice, postStructs[i].hapi_stream));

      // The sender may have staged IPC info for this transfer anyway: an
      // unconfirmed destination that turned out to be this same process
      // (see CkRdmaDeviceOnSender). That staging allocated a device
      // comm-buffer slot and claimed a CUDA IPC event that nothing will
      // ever free unless we tell the sender it's unused -- the free-up
      // logic in reclaimCompletedIpcEvents only runs once it sees the dst_flag
      // this receiver would have set had it taken the IPC branch below.
      // Signal that now (skip the actual comm-buffer copy in steps 1-2,
      // since the real data already moved via source.ptr above; just record
      // the completion event and flag so the sender can reclaim the slot).
      if (source.device_idx != -1 && csv_gpu_manager.use_shm) {
        hapi_ipc_device_info& device_info =
          csv_gpu_manager.hapi_ipc_device_infos[source.device_idx];
        hapiCheck(hapiEventRecord(device_info.dst_event_pool[source.event_idx],
              postStructs[i].hapi_stream));
        hapi_ipc_event_shared* shm_event_shared =
          (hapi_ipc_event_shared*)((char*)csv_gpu_manager.shm_ptr
              + csv_gpu_manager.shm_chunk_size * source.device_idx
              + sizeof(hapiIpcMemHandle_t)) + source.event_idx;
        pthread_mutex_lock(&shm_event_shared->lock);
        shm_event_shared->dst_flag = true;
        pthread_mutex_unlock(&shm_event_shared->lock);
      }
    } else if (sender_staged) {
      // sender_staged already guarantees device_idx is a real index; guard the
      // upper bound only, since a corrupted index would index the pool out of
      // bounds and surface later as an asynchronous illegal access, far from here.
      if ((size_t)source.device_idx >= csv_gpu_manager.hapi_ipc_device_infos.size()) {
        CkAbort("CkRdmaDeviceIssueRgets: receive on PE %d from PE %d carries an "
                "out-of-range IPC device index %d (pool size %zu).",
                CkMyPe(), env->getSrcPe(), source.device_idx,
                csv_gpu_manager.hapi_ipc_device_infos.size());
      }
      // Inter-process using shared memory optimizations
      // Use optimiziations with POSIX shared memory
      hapi_ipc_device_info& device_info =
        csv_gpu_manager.hapi_ipc_device_infos[source.device_idx];

      // TEMPORARY: locate the invalid pointer behind the illegal access seen in
      // the first cross-process ghost exchange. Prints the peer mapping and the
      // pool sizes actually backing the indices used just below, so a null peer
      // buffer, an out-of-range index, or a desynchronised event pool is
      // visible directly rather than inferred.
      if (getenv("CHARM_DEBUG_IPC_RECV")) {
        CmiPrintf("[%d] IPC recv: dev_idx=%d (infos=%zu) ev_idx=%d "
                  "(src_pool=%zu dst_pool=%zu flags=%zu) peer_buf=%p "
                  "off=%zu dest=%p cnt=%zu src_pe=%d\n",
                  CkMyPe(), source.device_idx,
                  csv_gpu_manager.hapi_ipc_device_infos.size(),
                  source.event_idx,
                  device_info.src_event_pool.size(),
                  device_info.dst_event_pool.size(),
                  device_info.event_pool_flags.size(),
                  device_info.buffer, (size_t)source.comm_offset,
                  (void*)dest.ptr, (size_t)dest.cnt, source.src_pe);
        fflush(stdout);
      }

      // 1. Make user-provided stream wait for IPC event using hapiStreamWaitEvent
      //    (source buffer to device comm buffer on source)
      hapiCheck(hapiStreamWaitEvent(postStructs[i].hapi_stream,
            device_info.src_event_pool[source.event_idx], 0));
      ipcDebugSync("recv 1: wait imported src_event", postStructs[i].hapi_stream);

      // 2. Invoke hapiMemcpyAsync (from source device comm buffer to destination buffer)
      hapiCheck(hapiMemcpyAsync((void*)dest.ptr,
            (void*)((char*)device_info.buffer + source.comm_offset),
            dest.cnt, hapiMemcpyDeviceToDevice, postStructs[i].hapi_stream));
      ipcDebugSync("recv 2: peer copy comm_buffer -> dest", postStructs[i].hapi_stream);

      // 3. Record IPC event so that the sender can query it for freeing
      //    device comm buffer and corresponding pair of CUDA IPC events
      hapiCheck(hapiEventRecord(device_info.dst_event_pool[source.event_idx],
            postStructs[i].hapi_stream));
      ipcDebugSync("recv 3: record imported dst_event", postStructs[i].hapi_stream);

      // 4. Set flag in shared memory so that the sender can start querying
      //    completion of the IPC event
      hapi_ipc_event_shared* shm_event_shared =
        (hapi_ipc_event_shared*)((char*)csv_gpu_manager.shm_ptr
            + csv_gpu_manager.shm_chunk_size * source.device_idx
            + sizeof(hapiIpcMemHandle_t)) + source.event_idx;
      pthread_mutex_lock(&shm_event_shared->lock);
      // The sender clears this when it reclaims the slot, so finding it already
      // set means a second receive is signalling the same (device, event) pair
      // before the first was retired. The sender would then free the block
      // belonging to whichever transfer claimed the slot next, while that
      // transfer is still reading it -- a use-after-free inside the comm buffer
      // that surfaces later as an illegal access on an unrelated stream.
      const bool already = shm_event_shared->dst_flag;
      shm_event_shared->dst_flag = true;
      pthread_mutex_unlock(&shm_event_shared->lock);
      if (already) {
        CmiPrintf("[%d] IPC DUPLICATE dst_flag dev_idx=%d ev_idx=%d srcPe=%d "
                  "off=%zu cnt=%zu\n",
                  CkMyPe(), source.device_idx, source.event_idx,
                  env->getSrcPe(), (size_t)source.comm_offset, (size_t)dest.cnt);
        fflush(stdout);
      }
    } else {
      // Nothing staged and not same-process. That is legitimate only for a
      // genuine cross-physical-node transfer, where the sender's else-branch
      // registered source.lci_ncpy_buffer for the rdmaGet below.
      //
      // IPC here means the sender resolved this destination to its own process
      // and staged nothing -- and then the target migrated to another process
      // before the message landed. The transfer mode is chosen at send time from
      // the location the sender knows; it only becomes true at delivery. The
      // source buffer lives in an address space this PE cannot read, and
      // source.lci_ncpy_buffer was never registered, so the rdmaGet below would
      // issue against an unregistered descriptor -- which surfaces as an LCI
      // "Message too long" assert or silent corruption, nowhere near the cause.
      //
      // Fail here, where the send is still identifiable, rather than there.
      if (mode == CkNcpyModeDevice::IPC) {
        CkAbort("CkRdmaDeviceIssueRgets: receive on PE %d from PE %d resolved to "
                "IPC, but the sender staged no IPC metadata (device_idx=%d, "
                "cnt=%zu). The target migrated across processes after the sender "
                "chose its transfer mode, so the source buffer is in another "
                "address space and cannot be read from here.",
                CkMyPe(), env->getSrcPe(), source.device_idx, (size_t)dest.cnt);
      }
      // CmiPrintf("it should never be called during intra node\n");
#if CMK_GPU_COMM
      // Machine layer supports GPU-aware communication
      QdCreate(1);
      CmiSetDirectNcpyAckHandler(CkRdmaDeviceRecvHandler);
      CmiNcpyBuffer lci_dest_ncpy_buffer(arrPtrs[i], (size_t)arrSizes[i], (void*)(&save_op));
      lci_dest_ncpy_buffer.rdmaGet(source.lci_ncpy_buffer, 0, nullptr, nullptr);
      continue;
#else
      // Handle all other cases (basic inter-process and inter-node)
      // Transfer the received/unpacked data on host to the destination device buffer
      // FIXME: Print warning that this is slow?
      CkAssert(source.data_stored);
      hapiCheck(hapiMemcpyAsync((void*)dest.ptr, source.data, dest.cnt,
            hapiMemcpyHostToDevice, postStructs[i].hapi_stream));
#endif
    }

    // Add source callback for polling, so that it can be invoked once the transfer is complete
    hapiAddCallback(postStructs[i].hapi_stream, CkCallback(CkRdmaDeviceRecvHandler, &save_op));
  }
}

// Unused, left for future reference
/*
int CkRdmaGetDestPEChare(int dest_pe, void* obj_ptr) {
  // Mechanism extracted from _prepareMsg() in ck.C
  if (dest_pe < 0) {
    int pe = -(dest_pe+1);
    if (pe == CkMyPe()) {
      VidBlock* vblk = CkpvAccess(vidblocks)[(CmiIntPtr)obj_ptr];
      void *objPtr = vblk->getLocalChare();
      dest_pe = objPtr ? pe : vblk->getActualID().onPE;
    } else {
      dest_pe = pe;
    }
  }

  return dest_pe;
}
*/

// Reclaim the IPC events in this PE's slice whose transfers have completed,
// releasing the device comm-buffer block each one was holding.
//
// Split out of the old findFreeIpcEvent so a sender that finds either resource
// exhausted can keep re-running just this scan while it waits (see
// acquireIpcSendSlot). Nothing here depends on this PE's scheduler: an event
// becomes reclaimable when the *peer* process's receiver records its event and
// sets dst_flag in shared memory, so repeating the scan is what lets a
// momentarily-exhausted pool recover.
//
// Caller must hold dm->lock (this frees comm-buffer blocks).
static void reclaimCompletedIpcEvents(DeviceManager* dm, int cpv_my_device_id) {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  int pool_size = csv_gpu_manager.hapi_ipc_event_pool_size_pe;
  int pool_start = CkMyRank() * pool_size;
  hapi_ipc_device_info& my_device_info = csv_gpu_manager.hapi_ipc_device_infos[csv_gpu_manager.device_count * CmiMyNodeRankLocal() + cpv_my_device_id];

  // Free IPC events that are complete
  for (int i = pool_start; i < pool_start + pool_size; i++) {
    int& event_flag = my_device_info.event_pool_flags[i];
    hapiEvent_t& ev = my_device_info.dst_event_pool[i];
    size_t& buff_offset = my_device_info.event_pool_buff_offsets[i];
    // For a used event, check if it's complete and mark as free if so
    if (event_flag != 0) {
      // Check in shared memory if receiver has invoked the memcpy from
      // the device comm buffer on sender to destination buffer
      hapi_ipc_event_shared* shm_event_shared =
        (hapi_ipc_event_shared*)((char*)csv_gpu_manager.shm_ptr
            + csv_gpu_manager.shm_chunk_size * (csv_gpu_manager.device_count * CmiMyNodeRankLocal() + cpv_my_device_id)
            + sizeof(hapiIpcMemHandle_t)) + i;
      bool can_query = false;
      pthread_mutex_lock(&shm_event_shared->lock);
      if (shm_event_shared->dst_flag == true) can_query = true;
      pthread_mutex_unlock(&shm_event_shared->lock);

      // If the receiver has invoked the memcpy,
      // the sender can query the event for completion
      if (can_query) {
        if (hapiEventQuery(ev) == hapiSuccess) {
          // Event completion means that the transfer from source device comm buffer
          // to dest buffer is complete, so free the allocated block
          if (event_flag == 1) {
            dm->free_comm_buffer(buff_offset);
          } else {
            CkAbort("Retrieved hapiSuccess for a free IPC event");
          }

          // Mark event as free
          event_flag = 0;
          pthread_mutex_lock(&shm_event_shared->lock);
          shm_event_shared->dst_flag = false;
          pthread_mutex_unlock(&shm_event_shared->lock);
        }
      }
    }
  }

}

// Claim a free IPC event from this PE's slice, or -1 if none is free.
//
// Two events are used per message:
// 1) Recorded by the sender after 'source buffer -> device comm buffer' hapiMemcpy.
//    Can be used by the sender to determine if the sender buffer is free for reuse.
//    It is also used by the receiver to create a dependency for the second hapiMemcpy
//    ('device comm buffer -> dest buffer')
// 2) Recorded by the receiver after 'device comm buffer -> dest buffer' hapiMemcpy.
//    It is used by the sender to determine when the allocated block on
//    device comm buffer and IPC events can be freed.
//
// Caller must hold dm->lock and should have reclaimed first.
static int claimFreeIpcEvent(int cpv_my_device_id, const size_t comm_offset) {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  int pool_size = csv_gpu_manager.hapi_ipc_event_pool_size_pe;
  int pool_start = CkMyRank() * pool_size;
  hapi_ipc_device_info& my_device_info = csv_gpu_manager.hapi_ipc_device_infos[csv_gpu_manager.device_count * CmiMyNodeRankLocal() + cpv_my_device_id];

  for (int i = pool_start; i < pool_start + pool_size; i++) {
    int& event_flag = my_device_info.event_pool_flags[i];
    size_t& buff_offset = my_device_info.event_pool_buff_offsets[i];
    if (event_flag == 0) {
      event_flag = 1;
      buff_offset = comm_offset;
      return i;
    }
  }

  return -1;
}

// Acquire the two resources a cross-process IPC send needs -- a block of the
// device comm buffer and an IPC event pair -- waiting for in-flight transfers
// to release them rather than aborting when the pools are momentarily empty.
//
// Both pools are sized for steady state, but a burst can transiently want more
// than they hold. leanmd's first step is the case that exposed this: every cell
// contacts all 26 neighbours at once, and on a 32-PE run that asks for more
// concurrent transfers per PE than a 256-event slice provides. Enlarging the
// pool is not the fix -- every slot costs two cudaEventInterprocess events per
// PE sharing the device, so merely doubling the default already fails device
// allocation at init (hapi_impl.cpp, ipcEventPoolInit). The slots are genuinely
// transient, so the sender waits for one to come back instead.
//
// The wait makes progress only while peers keep running: a slot is released
// once the receiving process handles the transfer and sets dst_flag in shared
// memory. That happens independently of this PE -- the sends holding those
// slots are already in flight -- so the common case drains quickly. It cannot
// happen if every participating PE is parked in this loop simultaneously, so
// the wait is bounded and says so, rather than hanging. Re-entering the
// scheduler (CsdSchedulePoll) would break such a cycle, but is not safe here:
// this runs mid-marshalling inside the caller's entry method, and re-entry
// could deliver another message to the very chare that is partway through a
// send.
static void acquireIpcSendSlot(DeviceManager* dm, int cpv_my_device_id,
                               bool is_lb_buffer, const void* src_ptr,
                               size_t cnt, void** out_buffer,
                               int* out_event_idx) {
  static const double timeout_s = []() {
    const char* s = getenv("CHARM_IPC_SLOT_TIMEOUT");
    return s ? atof(s) : 60.0;
  }();

  double wait_start = 0.0;
  bool waiting = false;

  for (;;) {
#if CMK_SMP
    CmiLock(dm->lock);
#endif
    // Reclaim before allocating, so a completed transfer's block is available
    // to this attempt. (Reclaiming only after the allocation, as the original
    // code did, meant comm-buffer exhaustion aborted without ever running the
    // scan that frees comm buffers.)
    reclaimCompletedIpcEvents(dm, cpv_my_device_id);

    void* buf = is_lb_buffer ? const_cast<void*>(src_ptr)
                             : dm->alloc_comm_buffer(cnt);
    if (buf != nullptr) {
      const size_t off = (char*)buf - (char*)dm->comm_buffer->base_ptr;
      const int ev = claimFreeIpcEvent(cpv_my_device_id, off);
      if (ev != -1) {
#if CMK_SMP
        CmiUnlock(dm->lock);
#endif
        *out_buffer = buf;
        *out_event_idx = ev;
        return;
      }
      // Got a block but no event. Hand the block back before waiting: holding
      // half the pair while blocked lets two senders pin each other's missing
      // half indefinitely.
      if (!is_lb_buffer) dm->free_comm_buffer(off);
    }
#if CMK_SMP
    CmiUnlock(dm->lock);
#endif

    if (!waiting) {
      wait_start = CkWallTimer();
      waiting = true;
    } else if (CkWallTimer() - wait_start > timeout_s) {
      CkAbort("PE %d, device %d: no free CUDA IPC event/comm-buffer slot after "
              "%.0fs (comm buffer: %zu bytes free). If every peer is blocked "
              "here too, the transfers that would release these slots cannot "
              "run; reduce concurrent device sends, or raise +gpucommbuffer "
              "(raising +gpuipceventpool costs device memory at init).",
              CkMyPe(), dm->global_index, timeout_s,
              dm->get_comm_buffer_free_size());
    }

    // Back off so the peer processes that release these slots get the core,
    // and so this does not spin on the lock that reclaiming needs.
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = 50000;  // 50us
    nanosleep(&ts, nullptr);
  }
}

// Device payload of the zerocopy send currently being marshalled, in bytes.
//
// The load balancer's communication graph weights each edge by
// UsrToEnv(msg)->getTotalsize(), but a device zerocopy message carries only the
// CkDeviceBuffer descriptors in its envelope -- a few hundred bytes standing in
// for a transfer that is routinely megabytes. Weighted that way, GPU-to-GPU
// edges are effectively invisible to any communication-aware strategy.
//
// The real sizes are known here and nowhere later on the send path, so they are
// parked per PE and picked up by CkArray::sendToPe. The generated code between
// this call and that one is straight-line marshalling with no intervening entry
// method, so the value cannot be interleaved with another send on this PE.
// CkArray::sendToPe takes and clears it unconditionally, so a stale value
// cannot outlive one array send.
//
// Limitation: group and nodegroup device sends never reach CkArray::sendToPe,
// so their value is left for the next array send on that PE to discard. The LB
// only builds object-to-object edges from array elements, so this costs
// nothing today, but it is why the take-and-clear must stay unconditional.
static thread_local size_t _ck_pending_device_send_bytes = 0;

// How many of this PE's IPC event slots are currently claimed. A slot is
// released only when the receiving process signals it took the staged bytes,
// so a count that climbs across load balancing rounds and never comes back
// down means transfers staged before a migration are never being acknowledged.
int CkRdmaDeviceBusyIpcSlots() {
  GPUManager& gm = CsvAccess(gpu_manager);
  if (!gm.use_shm) return -1;
  const int pool_size = gm.hapi_ipc_event_pool_size_pe;
  const int pool_start = CkMyRank() * pool_size;
  const int idx = gm.device_count * CmiMyNodeRankLocal() + CpvAccess(my_device_id);
  if (idx < 0 || (size_t)idx >= gm.hapi_ipc_device_infos.size()) return -1;
  hapi_ipc_device_info& info = gm.hapi_ipc_device_infos[idx];
  if ((size_t)(pool_start + pool_size) > info.event_pool_flags.size()) return -1;
  int busy = 0;
  for (int i = pool_start; i < pool_start + pool_size; i++)
    if (info.event_pool_flags[i] != 0) busy++;
  return busy;
}

size_t CkRdmaDeviceTakePendingSendBytes() {
  const size_t bytes = _ck_pending_device_send_bytes;
  _ck_pending_device_send_bytes = 0;
  return bytes;
}

// Performs sender-side operations necessary for device zerocopy
void CkRdmaDeviceOnSender(int dest_pe, int numops, CkDeviceBuffer** buffers) {
  // dest_pe == -1 means this PE has never confirmed where the target element
  // actually lives (xi-Parameter.C asks the location manager directly for
  // this, rather than substituting a homePe() guess). Don't decide
  // MEMCPY-vs-IPC from an unconfirmed location. But RDMA is not a safe
  // universal fallback here: findTransferModeDevice (above) only ever
  // returns RDMA for genuinely different physical nodes, so reconverse's
  // CmiIssueRget has never had to handle an RDMA-mode transfer that lands on
  // the sender's own physical node -- and its same-node loopback path does a
  // raw host memcpy between src/dst pointers, which corrupts memory (or
  // segfaults) when those are two different processes' device pointers, as
  // an unconfirmed-destination first contact on a single-node job always is.
  // When the whole job is on one physical node, an unconfirmed destination
  // can only ever truly resolve to MEMCPY or IPC (never RDMA), so stage IPC:
  // if the true destination turns out to be this same process after all,
  // the receiver's MEMCPY branch below uses source.ptr directly and ignores
  // this staging entirely -- the only cost is one unused, never-freed
  // comm-buffer slot and IPC event, a bounded one-time waste per
  // newly-confirmed pair, not a per-step cost. Only for a genuinely
  // multi-physical-node job with an unconfirmed destination do we fall back
  // to RDMA -- correct only if that destination doesn't land back on our own
  // physical node; that residual case isn't handled yet.
  const bool dest_confirmed = (dest_pe != -1);
  zcDestCount(dest_confirmed);
  CkNcpyModeDevice transfer_mode;
  if (dest_confirmed) {
    transfer_mode = findTransferModeDevice(CkMyPe(), dest_pe);
  } else if (CmiNumPhysicalNodes() == 1) {
    transfer_mode = CkNcpyModeDevice::IPC;
  } else {
    transfer_mode = CkNcpyModeDevice::RDMA;
  }

  // CHARM_ZC_ALWAYS_IPC makes migration-safe staging the rule rather than the
  // exception. The mode above is chosen from where the target is *now*, but only
  // becomes true when the message lands: if the element migrates to another
  // process in between, a MEMCPY decision leaves the receiver with a pointer
  // into an address space it cannot read (see the abort in
  // CkRdmaDeviceIssueRgets). Staging IPC for every send off this PE keeps the
  // transfer serviceable wherever it is delivered; the receiver's MEMCPY branch
  // ignores and releases the staging when the destination did stay local.
  // The cost is real -- a same-process transfer becomes two device copies plus
  // an event pair instead of one copy -- so this is opt-in while the proper
  // fix (re-staging on demand when the receiver detects the mismatch) does not
  // exist.
  static const bool always_ipc = (getenv("CHARM_ZC_ALWAYS_IPC") != nullptr);
  if (always_ipc && transfer_mode == CkNcpyModeDevice::MEMCPY &&
      dest_pe != CkMyPe() && CmiNumPhysicalNodes() == 1) {
    transfer_mode = CkNcpyModeDevice::IPC;
  }

  // Store destination PE in the metadata message
  // FIXME: Not necessary? save_op.dest_pe is set to CkMyPe() on the receiver
  size_t device_bytes = 0;
  for (int i = 0; i < numops; i++) {
    buffers[i]->dest_pe = dest_pe;
    buffers[i]->dest_mpi_rank = dest_confirmed ? CmiNodeOf(dest_pe) : -1;
    buffers[i]->src_pe = CmiMyPe();
    buffers[i]->src_mpi_rank = CmiNodeOf(CmiMyPe());
    device_bytes += buffers[i]->cnt;
  }
  _ck_pending_device_send_bytes = device_bytes;

  // CHARM_ZC_VALIDATE: check every outgoing source pointer, in every transfer
  // mode, at the moment the send is posted. A device buffer whose owning chare
  // has migrated is either unmapped (freed by the old PE) or resident on a
  // different device than the one now current, and both surface later as an
  // "illegal memory access" inside an async copy with no hint of which send
  // produced it. Report it here, where dest_pe/cnt/mode still identify the send.
  {
    static const bool validate = (getenv("CHARM_ZC_VALIDATE") != nullptr);
    if (validate) {
      int cur_dev = -1;
      cudaGetDevice(&cur_dev);
      for (int i = 0; i < numops; i++) {
        cudaPointerAttributes sattr{};
        const cudaError_t serr = cudaPointerGetAttributes(&sattr, buffers[i]->ptr);
        const bool unmapped = (serr != cudaSuccess || sattr.type == cudaMemoryTypeUnregistered);
        const bool wrong_dev = (!unmapped && sattr.type == cudaMemoryTypeDevice &&
                                sattr.device != cur_dev);
        if (unmapped || wrong_dev) {
          CmiPrintf("[%d] ZC SEND VALIDATE FAIL src=%p %s (type=%d ptr_dev=%d "
                    "cur_dev=%d err=%d) cnt=%zu dest_pe=%d mode=%s\n",
                    CkMyPe(), buffers[i]->ptr,
                    unmapped ? "UNMAPPED" : "WRONG-DEVICE",
                    (int)sattr.type, (int)sattr.device, cur_dev, (int)serr,
                    (size_t)buffers[i]->cnt, dest_pe,
                    transfer_mode == CkNcpyModeDevice::MEMCPY ? "MEMCPY"
                      : (transfer_mode == CkNcpyModeDevice::IPC ? "IPC" : "RDMA"));
          fflush(stdout);
          cudaGetLastError();
        }
      }
    }
  }

  if(transfer_mode == CkNcpyModeDevice::MEMCPY)
  {
    // Same process: the receiver can read buffers[i]->ptr directly, but only
    // once the kernels producing it have finished. Blocking the host here
    // (hapiStreamSynchronize) guaranteed that at the cost of stalling the
    // sender on every such send -- and same-process is the common case, so
    // that stall dominated. Record an event instead and let the receiver's
    // stream wait on it: identical ordering, no host block, no extra copy.
    // Set CHARM_ZC_MEMCPY_SYNC to restore the blocking behaviour.
    static const bool force_sync = (getenv("CHARM_ZC_MEMCPY_SYNC") != nullptr);
    for (int i = 0; i < numops; i++) {
      if (force_sync) {
        hapiStreamSynchronize(buffers[i]->hapi_stream);
      } else {
        buffers[i]->memcpy_event = ckDeviceRecordMemcpyEvent(buffers[i]->hapi_stream);
        if (buffers[i]->memcpy_event == NULL)  // no event available; fall back
          hapiStreamSynchronize(buffers[i]->hapi_stream);
      }
    }
    return;
  }

  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  //int cpv_my_device_id = CmiMyRank() % csv_gpu_manager.device_count;
  int cpv_my_device_id = CpvAccess(my_device_id);

  if(transfer_mode == CkNcpyModeDevice::IPC && csv_gpu_manager.use_shm) {
    // Use optimizations with POSIX shaerd memory
    // Allocate blocks on device comm buffer
    DeviceManager* dm = csv_gpu_manager.device_map[CkMyPe()];

    for (int i = 0; i < numops; i++) {
      bool is_lb_buffer = ( (size_t)((char*)(buffers[i]->ptr) - (char*)(dm->comm_buffer->base_ptr)) < dm->comm_buffer->total_size );
      // Waits for a slot if the pools are momentarily drained rather than
      // aborting; takes and releases dm->lock itself.
      void* alloc_comm_buffer;
      int acquired_event_idx;
      acquireIpcSendSlot(dm, cpv_my_device_id, is_lb_buffer, buffers[i]->ptr,
                         buffers[i]->cnt, &alloc_comm_buffer,
                         &acquired_event_idx);
      buffers[i]->comm_offset = (char*)alloc_comm_buffer - (char*)dm->comm_buffer->base_ptr;
      buffers[i]->device_idx = (csv_gpu_manager.device_count * CmiMyNodeRankLocal() + cpv_my_device_id);
      buffers[i]->event_idx = acquired_event_idx;

      // TEMPORARY: paired with the receive-side print, so the indices and
      // offsets the sender publishes can be compared against what the receiver
      // resolves them to.
      if (getenv("CHARM_DEBUG_IPC_RECV")) {
        CmiPrintf("[%d] IPC send: dev_idx=%d ev_idx=%d off=%zu cnt=%zu "
                  "src_ptr=%p comm_base=%p alloc=%p is_lb=%d dest_pe=%d\n",
                  CkMyPe(), buffers[i]->device_idx, buffers[i]->event_idx,
                  (size_t)buffers[i]->comm_offset, (size_t)buffers[i]->cnt,
                  buffers[i]->ptr, dm->comm_buffer->base_ptr,
                  alloc_comm_buffer, (int)is_lb_buffer, dest_pe);
        fflush(stdout);
      }

      // Initiate transfer from source buffer to device comm buffer
      if(!is_lb_buffer) {
        // CHARM_ZC_VALIDATE: the buffer being staged belongs to the sending
        // chare. If that chare has migrated, its device scratch may already be
        // freed (or not yet reallocated on the new PE) while a send referencing
        // it is still being marshalled -- report that here rather than letting
        // it surface asynchronously somewhere unrelated.
        if (getenv("CHARM_ZC_VALIDATE")) {
          cudaPointerAttributes sattr{};
          const cudaError_t serr = cudaPointerGetAttributes(&sattr, buffers[i]->ptr);
          if (serr != cudaSuccess || sattr.type == cudaMemoryTypeUnregistered) {
            CmiPrintf("[%d] ZC SEND VALIDATE FAIL src=%p (type=%d err=%d) cnt=%zu "
                      "dest_pe=%d mode=IPC\n",
                      CkMyPe(), buffers[i]->ptr, (int)sattr.type, (int)serr,
                      (size_t)buffers[i]->cnt, dest_pe);
            fflush(stdout);
            cudaGetLastError();
          }
        }
        hapiCheck(hapiMemcpyAsync(alloc_comm_buffer, buffers[i]->ptr, buffers[i]->cnt,
              hapiMemcpyDeviceToDevice, buffers[i]->hapi_stream));
        ipcDebugSync("send 1: stage src -> comm_buffer", buffers[i]->hapi_stream);
      }

      // Record event
      hapi_ipc_device_info& my_device_info = csv_gpu_manager.hapi_ipc_device_infos[(csv_gpu_manager.device_count * CmiMyNodeRankLocal() + cpv_my_device_id)];
      hapiCheck(hapiEventRecord(my_device_info.src_event_pool[buffers[i]->event_idx], buffers[i]->hapi_stream));
      ipcDebugSync("send 2: record own src_event", buffers[i]->hapi_stream);
    }
  } else {
#if !CMK_GPU_COMM
    // Use a naive host-staged mechanism
    // Allocate temporary host buffers and copy source buffers
    for (int i = 0; i < numops; i++) {
      buffers[i]->data_stored = true;
      hapiCheck(hapiMallocHost(&buffers[i]->data, buffers[i]->cnt));
      hapiCheck(hapiMemcpyAsync(buffers[i]->data, buffers[i]->ptr, buffers[i]->cnt,
            hapiMemcpyDeviceToHost, buffers[i]->hapi_stream));
    }

    // Wait for the copies to finish
    for (int i = 0; i < numops; i++) {
      hapiCheck(hapiStreamSynchronize(buffers[i]->hapi_stream));
    }
#else
  for (int i = 0; i < numops; i++) {
    cudaStreamSynchronize(buffers[i]->hapi_stream);
    buffers[i]->lci_ncpy_buffer = CmiNcpyBuffer(buffers[i]->ptr, buffers[i]->cnt);
  }
#endif
  }
}
#endif // CMK_CUDA
