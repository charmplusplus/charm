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

    // Perform data transfers
    if (mode == CkNcpyModeDevice::MEMCPY) {
      // Source and destination PEs are in the same process (logical node)
      // Directly invoke memcpy from source buffer to destination buffer
      hapiCheck(hapiMemcpyAsync((void*)dest.ptr, source.ptr, dest.cnt,
            hapiMemcpyDeviceToDevice, postStructs[i].hapi_stream));      
    } else if (mode == CkNcpyModeDevice::IPC && csv_gpu_manager.use_shm) {
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
      shm_event_shared->dst_flag = true;
      pthread_mutex_unlock(&shm_event_shared->lock);
    } else {
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

static int findFreeIpcEvent(DeviceManager* dm, const size_t comm_offset, int cpv_my_device_id) {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  int pool_size = csv_gpu_manager.hapi_ipc_event_pool_size_pe;
  int pool_start = CkMyRank() * pool_size;
  hapi_ipc_device_info& my_device_info = csv_gpu_manager.hapi_ipc_device_infos[csv_gpu_manager.device_count * CmiMyNodeRankLocal() + cpv_my_device_id];

  // Free IPC events that are complete
  // TODO: Don't do this every time but only when the event pool is somewhat empty
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

  // Allocate CUDA IPC events from the pool
  // Two events are used per message:
  // 1) Recorded by the sender after 'source buffer -> device comm buffer' hapiMemcpy.
  //    Can be used by the sender to determine if the sender buffer is free for reuse.
  //    It is also used by the receiver to create a dependency for the second hapiMemcpy
  //    ('device comm buffer -> dest buffer')
  // 2) Recorded by the receiver after 'device comm buffer -> dest buffer' hapiMemcpy.
  //    It is used by the sender to determine when the allocated block on
  //    device comm buffer and IPC events can be freed.
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

// Performs sender-side operations necessary for device zerocopy
void CkRdmaDeviceOnSender(int dest_pe, int numops, CkDeviceBuffer** buffers) {
  // TODO: Need to handle the case where the destination PE could be wrong
  //       (due to migration, etc.). Currently the code relies on a global
  //       location update after migration (with CMK_GLOBAL_LOCATION_UPDATE).
  // CmiPrintf("[%d] CkRdmaDeviceOnSender: src_pe=%d, dst_pe=%d\n", CkMyPe(), CkMyPe(), dest_pe);
  CkNcpyModeDevice transfer_mode = findTransferModeDevice(CkMyPe(), dest_pe);

  // Store destination PE in the metadata message
  // FIXME: Not necessary? save_op.dest_pe is set to CkMyPe() on the receiver
  for (int i = 0; i < numops; i++) {
    buffers[i]->dest_pe = dest_pe;
    buffers[i]->dest_mpi_rank = CmiNodeOf(dest_pe);
    buffers[i]->src_pe = CmiMyPe();
    buffers[i]->src_mpi_rank = CmiNodeOf(CmiMyPe());
  }
  if(transfer_mode == CkNcpyModeDevice::MEMCPY)
  {
    for (int i = 0; i < numops; i++)
      hapiStreamSynchronize(buffers[i]->hapi_stream);
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
#if CMK_SMP
      CmiLock(dm->lock);
#endif
      void* alloc_comm_buffer;
      if(is_lb_buffer) {
        alloc_comm_buffer = const_cast<void*>(buffers[i]->ptr);
      } else {
        alloc_comm_buffer = dm->alloc_comm_buffer(buffers[i]->cnt);
        if (alloc_comm_buffer == nullptr) {
          CkAbort("PE %d, device %d: Not enough memory on device communication buffer (%zu free)",
              CkMyPe(), dm->global_index, dm->get_comm_buffer_free_size());
        }
      }
      buffers[i]->comm_offset = (char*)alloc_comm_buffer - (char*)dm->comm_buffer->base_ptr;
      buffers[i]->device_idx = (csv_gpu_manager.device_count * CmiMyNodeRankLocal() + cpv_my_device_id);
      buffers[i]->event_idx = findFreeIpcEvent(dm, buffers[i]->comm_offset, cpv_my_device_id);
      // Abort if no free IPC event was found
      // FIXME: Instead of aborting, we can maybe create IPC events on demand
      // (although they probably cannot be shared through the shared memory
      // allocated and shared between processes at init time)
      if (buffers[i]->event_idx == -1) {
        CkAbort("CUDA IPC event pool empty");
      }
#if CMK_SMP
      CmiUnlock(dm->lock);
#endif

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
