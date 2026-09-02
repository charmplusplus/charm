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
 *    allocation algorithm was implemented. The allocator first calls cudaMalloc
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
#include <cstdlib>
#include <pthread.h>
#endif
#include "envelope.h"
#include "charm++.h"
#include "ck.h"
#include "ckrdmadevice.h"

/* Stage 9.1 boundary: this file is the classic D2D implementation, written
 * against the classic half of ckrdmadevice.h (cuda_* fields, 4-param
 * CmiSendDevice). The reconverse D2D implementation arrives with stage 9.2
 * (see doc/reconverse-migration-ledger.README.md); until then reconverse
 * CUDA builds compile this file empty, and device-RDMA entry parameters
 * fail at link with these symbols missing. */
#if CMK_CUDA && !CMK_RECONVERSE

#include "hapi.h"
#include "gpumanager.h"

CsvExtern(GPUManager, gpu_manager);

// Invoked when a GPU buffer arrives on the receiver
#if !CMK_GPU_COMM
void CkRdmaDeviceRecvHandler(void* data, void* msg)
#else
void CkRdmaDeviceRecvHandler(void* data)
#endif
{
#if CMK_GPU_COMM
  // Process QD to mark completion of buffer transfer
  QdProcess(1);
#endif

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
  hapiCheck(cudaIpcGetMemHandle(&cuda_ipc_handle, (void*)ptr));
}

void CkDevicePersistent::close() {
  // Close the CUDA IPC handle if it was opened
  hapiCheck(cudaIpcCloseMemHandle(ipc_ptr));
}

void CkDevicePersistent::set_msg(void* msg) {
  cb_msg = msg;
}

void CkDevicePersistent::pup(PUP::er& p) {
  p((char*)&ptr, sizeof(ptr));
  p|cnt;
  p|pe;
  p|cb;
  p((char*)&cuda_ipc_handle, sizeof(cuda_ipc_handle));
}

CkDeviceStatus CkDevicePersistent::get(CkDevicePersistent& src) {
  // Check that the source buffer fits into the destination buffer
  if (cnt < src.cnt) {
    CkAbort("CkDevicePersistent::get: Destination buffer is smaller than source buffer\n");
  }

  CkNcpyModeDevice mode = findTransferModeDevice(src.pe, CkMyPe());

  // Perform get
  if (mode == CkNcpyModeDevice::MEMCPY) {
    cudaMemcpyAsync((void*)ptr, src.ptr, cnt, cudaMemcpyDeviceToDevice, cuda_stream);
  } else if (mode == CkNcpyModeDevice::IPC) {
    if (!src.ipc_open) {
      hapiCheck(cudaIpcOpenMemHandle(&src.ipc_ptr, src.cuda_ipc_handle,
            cudaIpcMemLazyEnablePeerAccess));
      src.ipc_open = true;
    }
    cudaMemcpyAsync((void*)ptr, src.ipc_ptr, cnt, cudaMemcpyDeviceToDevice, cuda_stream);
  } else {
    CkAbort("Persistant GPU messaging is currently not supported for inter-node messages");
  }

  // Set callbacks to be invoked once get is complete
  if (src.cb.type != CkCallback::ignore) {
    hapiAddCallback(cuda_stream, src.cb, src.cb_msg);
  }
  if (cb.type != CkCallback::ignore) {
    hapiAddCallback(cuda_stream, cb, cb_msg);
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
    cudaMemcpyAsync((void*)dst.ptr, ptr, cnt, cudaMemcpyDeviceToDevice, cuda_stream);
  } else if (mode == CkNcpyModeDevice::IPC) {
    if (!dst.ipc_open) {
      hapiCheck(cudaIpcOpenMemHandle(&dst.ipc_ptr, dst.cuda_ipc_handle,
            cudaIpcMemLazyEnablePeerAccess));
      dst.ipc_open = true;
    }
    cudaMemcpyAsync(dst.ipc_ptr, ptr, cnt, cudaMemcpyDeviceToDevice, cuda_stream);
  } else {
    CkAbort("Persistant GPU messaging is not yet supported for inter-node messages");
  }

  // Set callbacks to be invoked once get is complete
  if (cb.type != CkCallback::ignore) {
    hapiAddCallback(cuda_stream, cb, cb_msg);
  }
  if (dst.cb.type != CkCallback::ignore) {
    hapiAddCallback(cuda_stream, dst.cb, dst.cb_msg);
  }

  return CkDeviceStatus::incomplete;
}

/****************************** Recv Entry Method API ******************************/

// Invoked after post entry method
void CkRdmaDeviceIssueRgets(envelope *env, int numops, void **arrPtrs, int *arrSizes, CkDeviceBufferPost *postStructs) {
  // Change message header to invoke regular entry method
  CMI_ZC_MSGTYPE(env) = CMK_REG_NO_ZC_MSG;

  // Create a copy of this message for regular entry method invocation
  // FIXME: Reuse the old message instead of creating a new one
  void* old_msg = EnvToUsr(env);
  envelope* new_env = UsrToEnv(CkCopyMsg(&old_msg));

  // Start unpacking marshalled message
  PUP::fromMem up((void *)((CkMarshallMsg *)EnvToUsr(env))->msgBuf);
  int received_numops;
  up|received_numops;
  CkAssert(numops == received_numops);

  CkDeviceBuffer source;

#if !CMK_GPU_COMM
  // Machine layer does not support GPU-aware communication
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

  // Find which mode of transfer should be used
  CkNcpyModeDevice mode = findTransferModeDevice(env->getSrcPe(), CkMyPe());
#endif

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
    save_op.dest_pe = CkMyPe();
    save_op.dest_ptr = arrPtrs[i];
    save_op.size = (size_t)arrSizes[i];
    save_op.info = rdma_info;
    save_op.src_cb = (source.cb.type != CkCallback::ignore) ? new CkCallback(source.cb) : nullptr;
    save_op.dst_cb = nullptr;

#if !CMK_GPU_COMM
    // Machine layer does not support GPU-aware communication
    // Check if destination PE is correct
    // TODO: Handle this case instead of aborting
    if (source.dest_pe != CkMyPe()) {
      CkAbort("Current PE does not match the destination PE determined by the sender. "
          "Please enable CMK_GLOBAL_LOCATION_UPDATE.");
    }

    // Destination buffer (on this receiver)
    CkDeviceBuffer dest((const void *)arrPtrs[i], arrSizes[i]);

    // Perform data transfers
    if (mode == CkNcpyModeDevice::MEMCPY) {
      // Source and destination PEs are in the same process (logical node)
      // Directly invoke memcpy from source buffer to destination buffer
      hapiCheck(cudaMemcpyAsync((void*)dest.ptr, source.ptr, dest.cnt,
            cudaMemcpyDeviceToDevice, postStructs[i].cuda_stream));
    } else if (mode == CkNcpyModeDevice::IPC && csv_gpu_manager.use_shm) {
      // Inter-process using shared memory optimizations
      // Use optimiziations with POSIX shared memory
      hapi_ipc_device_info& device_info =
        csv_gpu_manager.hapi_ipc_device_infos[source.device_idx];

      // 1. Make user-provided stream wait for IPC event using cudaStreamWaitEvent
      //    (source buffer to device comm buffer on source)
      hapiCheck(cudaStreamWaitEvent(postStructs[i].cuda_stream,
            device_info.src_event_pool[source.event_idx], 0));

      // 2. Invoke cudaMemcpyAsync (from source device comm buffer to destination buffer)
      hapiCheck(cudaMemcpyAsync((void*)dest.ptr,
            (void*)((char*)device_info.buffer + source.comm_offset),
            dest.cnt, cudaMemcpyDeviceToDevice, postStructs[i].cuda_stream));

      // 3. Record IPC event so that the sender can query it for freeing
      //    device comm buffer and corresponding pair of CUDA IPC events
      hapiCheck(cudaEventRecord(device_info.dst_event_pool[source.event_idx],
            postStructs[i].cuda_stream));

      // 4. Set flag in shared memory so that the sender can start querying
      //    completion of the IPC event
      hapi_ipc_event_shared* shm_event_shared =
        (hapi_ipc_event_shared*)((char*)csv_gpu_manager.shm_ptr
            + csv_gpu_manager.shm_chunk_size * source.device_idx
            + sizeof(cudaIpcMemHandle_t)) + source.event_idx;
      pthread_mutex_lock(&shm_event_shared->lock);
      shm_event_shared->dst_flag = true;
      pthread_mutex_unlock(&shm_event_shared->lock);
    } else {
      // Handle all other cases (basic inter-process and inter-node)
      // Transfer the received/unpacked data on host to the destination device buffer
      // FIXME: Print warning that this is slow?
      CkAssert(source.data_stored);
      hapiCheck(cudaMemcpyAsync((void*)dest.ptr, source.data, dest.cnt,
            cudaMemcpyHostToDevice, postStructs[i].cuda_stream));
    }

    // Add source callback for polling, so that it can be invoked once the transfer is complete
    hapiAddCallback(postStructs[i].cuda_stream, CkCallback(CkRdmaDeviceRecvHandler, &save_op));
#else
    // Machine layer supports GPU-aware communication
    save_op.tag = source.tag;
#endif // CMK_GPU_COMM
  }

#if CMK_GPU_COMM
  // Post ucp_tag_recv_nb's to receive GPU data
  for (int i = 0; i < numops; i++) {
    DeviceRdmaOp* save_op = (DeviceRdmaOp*)((char*)rdma_data
        + sizeof(DeviceRdmaInfo) + sizeof(DeviceRdmaOp) * i);
    QdCreate(1);
    CmiRecvDevice(save_op, DEVICE_RECV_TYPE_CHARM);
  }
#endif
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

static int findFreeIpcEvent(DeviceManager* dm, const size_t comm_offset) {
  int pool_size = CsvAccess(gpu_manager).hapi_ipc_event_pool_size_pe;
  int pool_start = CkMyRank() * pool_size;
  int device_index = dm->global_index;
  hapi_ipc_device_info& my_device_info = CsvAccess(gpu_manager).hapi_ipc_device_infos[device_index];

  // Free IPC events that are complete
  // TODO: Don't do this every time but only when the event pool is somewhat empty
  for (int i = pool_start; i < pool_start + pool_size; i++) {
    int& event_flag = my_device_info.event_pool_flags[i];
    cudaEvent_t& ev = my_device_info.dst_event_pool[i];
    size_t& buff_offset = my_device_info.event_pool_buff_offsets[i];
    // For a used event, check if it's complete and mark as free if so
    if (event_flag != 0) {
      // Check in shared memory if receiver has invoked the memcpy from
      // the device comm buffer on sender to destination buffer
      hapi_ipc_event_shared* shm_event_shared =
        (hapi_ipc_event_shared*)((char*)CsvAccess(gpu_manager).shm_ptr
            + CsvAccess(gpu_manager).shm_chunk_size * device_index
            + sizeof(cudaIpcMemHandle_t)) + i;
      bool can_query = false;
      pthread_mutex_lock(&shm_event_shared->lock);
      if (shm_event_shared->dst_flag == true) {
        shm_event_shared->dst_flag = false;
        can_query = true;
      }
      pthread_mutex_unlock(&shm_event_shared->lock);

      // If the receiver has invoked the memcpy,
      // the sender can query the event for completion
      if (can_query) {
        if (cudaEventQuery(ev) == cudaSuccess) {
          // Event completion means that the transfer from source device comm buffer
          // to dest buffer is complete, so free the allocated block
          if (event_flag == 1) {
            dm->free_comm_buffer(buff_offset);
          } else {
            CkAbort("Retrieved cudaSuccess for a free IPC event");
          }

          // Mark event as free
          event_flag = 0;
        }
      }
    }
  }

  // Allocate CUDA IPC events from the pool
  // Two events are used per message:
  // 1) Recorded by the sender after 'source buffer -> device comm buffer' cudaMemcpy.
  //    Can be used by the sender to determine if the sender buffer is free for reuse.
  //    It is also used by the receiver to create a dependency for the second cudaMemcpy
  //    ('device comm buffer -> dest buffer')
  // 2) Recorded by the receiver after 'device comm buffer -> dest buffer' cudaMemcpy.
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
#if !CMK_GPU_COMM
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

  // Determine transfer mode (intra-process, inter-process, inter-node)
  CkNcpyModeDevice transfer_mode = findTransferModeDevice(CkMyPe(), dest_pe);

  // Store destination PE in the metadata message
  // FIXME: Not necessary? save_op.dest_pe is set to CkMyPe() on the receiver
  for (int i = 0; i < numops; i++) {
    buffers[i]->dest_pe = dest_pe;
  }

  if (transfer_mode == CkNcpyModeDevice::MEMCPY) {
    // Don't need to do anything for intra-process
    return;
  } else if (transfer_mode == CkNcpyModeDevice::IPC && csv_gpu_manager.use_shm) {
    // Use optimizations with POSIX shaerd memory
    // Allocate blocks on device comm buffer
    DeviceManager* dm = csv_gpu_manager.device_map[CkMyPe()];

    for (int i = 0; i < numops; i++) {
#if CMK_SMP
      CmiLock(dm->lock);
#endif
      void* alloc_comm_buffer = dm->alloc_comm_buffer(buffers[i]->cnt);
      if (alloc_comm_buffer == nullptr) {
        CkAbort("PE %d, device %d: Not enough memory on device communication buffer (%zu free)",
            CkMyPe(), dm->global_index, dm->get_comm_buffer_free_size());
      }
      buffers[i]->comm_offset = (char*)alloc_comm_buffer - (char*)dm->comm_buffer->base_ptr;
      buffers[i]->device_idx = dm->global_index;
      buffers[i]->event_idx = findFreeIpcEvent(dm, buffers[i]->comm_offset);
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

      // Initiate transfer from source buffer to device comm buffer
      hapiCheck(cudaMemcpyAsync(alloc_comm_buffer, buffers[i]->ptr, buffers[i]->cnt,
            cudaMemcpyDeviceToDevice, buffers[i]->cuda_stream));

      // Record event
      hapi_ipc_device_info& my_device_info = csv_gpu_manager.hapi_ipc_device_infos[dm->global_index];
      hapiCheck(cudaEventRecord(my_device_info.src_event_pool[buffers[i]->event_idx], buffers[i]->cuda_stream));
    }
  } else {
    // Use a naive host-staged mechanism
    // Allocate temporary host buffers and copy source buffers
    for (int i = 0; i < numops; i++) {
      buffers[i]->data_stored = true;
      hapiCheck(cudaMallocHost(&buffers[i]->data, buffers[i]->cnt));
      hapiCheck(cudaMemcpyAsync(buffers[i]->data, buffers[i]->ptr, buffers[i]->cnt,
            cudaMemcpyDeviceToHost, buffers[i]->cuda_stream));
    }

    // Wait for the copies to finish
    for (int i = 0; i < numops; i++) {
      hapiCheck(cudaStreamSynchronize(buffers[i]->cuda_stream));
    }
  }
#else
  // Post ucp_tag_send_nb's to send GPU data. When receiver receives the metadata,
  // it should post ucp_tag_recv_nb's to receive the GPU data.
  for (int i = 0; i < numops; i++) {
    CmiSendDevice(dest_pe, buffers[i]->ptr, buffers[i]->cnt, buffers[i]->tag);
  }
#endif // CMK_GPU_COMM
}
#endif // CMK_CUDA

/* ==========================================================================
 * Reconverse D2D  --  GPU migration plan stage 9.2
 *
 * The classic half above is left byte-for-byte alone; this is the same design
 * expressed against the reconverse interfaces. Three differences are worth
 * knowing before reading:
 *
 *  - findTransferModeDevice() is defined here rather than in
 *    conv-core/conv-rdmadevice.C, because that file belongs to Converse and is
 *    not compiled at all in a reconverse build (reconverse replaces the
 *    machine layer wholesale). Everything the D2D path needs therefore has to
 *    live on the ck-core side.
 *
 *  - The inter-node leg is a real RDMA get through reconverse's ncpy Direct
 *    API (CmiNcpyBuffer::rdmaGet), not the host-staged device-to-host /
 *    host-to-device round trip the classic half falls back to. CMK_GPU_COMM is
 *    unconditionally 1 for reconverse (see conv-rdmadevice.h), so the
 *    host-staged branches are omitted here instead of being carried as dead
 *    #if arms; the classic half above still has them if a non-RDMA reconverse
 *    backend ever needs one.
 *
 *  - Completions arrive on whichever PE in the process the backend happened to
 *    progress, which need not be the destination PE. loopback_bridge bounces
 *    them to the right PE; see CkRdmaDeviceRecvHandler.
 * ========================================================================== */
#if (CMK_CUDA || CMK_HIP) && CMK_RECONVERSE

#include "hapi.h"
#include "gpumanager.h"

CsvExtern(GPUManager, gpu_manager);
CpvExtern(int, my_device_id);

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

// Carrier for bouncing a completion to the destination PE within this process.
struct LoopBackMsg {
  char header[CmiMsgHeaderSizeBytes];
  void* msg;
};

extern "C" {
  void* loopback_bridge(void* arg) {
    LoopBackMsg* recv_msg = (LoopBackMsg*)arg;
    void* op_info_copy = recv_msg->msg;
    QdProcess(1);  // matches the QdCreate in CkRdmaDeviceRecvHandler
    CkRdmaDeviceRecvHandler(op_info_copy);
    // This NcpyOperationInfo is our own copy -- reconverse frees the original
    // as soon as the ack handler returns -- so it is ours to release. (Not to
    // be confused with the DeviceRdmaInfo the handler may free; separate
    // allocation.)
    CmiFree(op_info_copy);
    CmiFree(recv_msg);
    return NULL;
  }

  int loopback_handler;
}

/* Registration release for the inter-node (RDMA) device path
 * (charmplusplus/charm#3960).  Both ends of every device get are registered
 * per message -- the sender in CkRdmaDeviceOnSender (CmiNcpyBuffer ctor), the
 * receiver in CkRdmaDeviceIssueRgets -- and, before this, neither was ever
 * released: the completion returns from CkRdmaDirectAckHandler before the
 * host path's deregistration, and CkRdmaDeviceRecvHandler had none of its
 * own.  With LCI's registration cache off that exhausts the NIC's
 * registration table (ENOSPC in fi_mr_regattr) after ~1e5 messages per
 * rank; with the cache on it pins every region at a refcount that never
 * returns to zero.
 *
 * The destination registration belongs to this process, so it is released
 * in place when the get completes.  The source registration belongs to the
 * sender's process; the sender is told with a DeviceDeregMsg carrying only
 * what CmiDeregisterMem needs.  PE 0 prints once that releases are in
 * force, so a run states its own configuration. */
struct DeviceDeregMsg {
  char header[CmiMsgHeaderSizeBytes];
  const void* ptr;
  int pe;
  unsigned short regMode;
  char mr[CMK_NOCOPY_DIRECT_BYTES];  // the layer's memory-region handle
};

extern "C" {
  void* device_dereg_bridge(void* arg) {
    DeviceDeregMsg* m = (DeviceDeregMsg*)arg;
    QdProcess(1);  // matches the QdCreate in releaseDeviceRegistrations
    CmiDeregisterMem(m->ptr, m->mr, m->pe, m->regMode);
    CmiFree(m);
    return NULL;
  }
  int device_dereg_handler;
}

/* Persistent device-buffer registration (fix (b) for #3960).
 *
 * Registered regions live in one interval map per process: [start, end)
 * with the layer's memory-region handle.  CkDeviceBufferRegister(ptr, cnt)
 * registers a buffer the application allocated itself, once (a whole array
 * may be registered and sub-ranges of it sent or posted);
 * CkDeviceBufferDeregister releases it before the application frees it.
 * The send path (CkRdmaDeviceOnSender) and the receive path
 * (CkRdmaDeviceIssueRgets) look a buffer up by range containment; a hit
 * reuses the region's handle and marks the buffer NODEREG so the
 * completion-time release above leaves it alone; a miss falls back to the
 * per-message registration that fix (a) releases.  The lifetime is the
 * application's, explicitly: no allocator hooks, no cache eviction, and an
 * unregistered buffer stays correct, only slower. */
#include <mutex>
#include <map>
#include <vector>

struct DeviceRegion { uintptr_t end; char layerInfo[CMK_NOCOPY_DIRECT_BYTES]; };
static std::map<uintptr_t, DeviceRegion> device_regions;   // start -> region
static std::mutex device_reg_mutex;

// Under device_reg_mutex.
static const DeviceRegion* deviceRegionFind(uintptr_t p, size_t cnt) {
  auto it = device_regions.upper_bound(p);
  if (it != device_regions.begin()) {
    --it;
    if (p >= it->first && p + cnt <= it->second.end) return &it->second;
  }
  return nullptr;
}

void CkDeviceBufferRegister(const void* ptr, size_t cnt) {
  uintptr_t p = (uintptr_t)ptr;
  std::lock_guard<std::mutex> g(device_reg_mutex);
  if (deviceRegionFind(p, cnt)) return;  // already covered
  if (device_regions.empty() && CkMyPe() == 0)
    CmiPrintf("CkRdmaDevice> persistent device-buffer registration in use (CkDeviceBufferRegister)\n");
  DeviceRegion r; r.end = p + cnt;
  CmiSetRdmaBufferInfo(r.layerInfo, ptr, cnt, CMK_BUFFER_REG);
  device_regions.emplace(p, r);
}

void CkDeviceBufferDeregister(const void* ptr, size_t cnt) {
  std::lock_guard<std::mutex> g(device_reg_mutex);
  auto it = device_regions.find((uintptr_t)ptr);
  if (it == device_regions.end()) return;
  CmiDeregisterMem(ptr, it->second.layerInfo, CkMyPe(), CMK_BUFFER_REG);
  device_regions.erase(it);
}

// Initialise 'b' for ptr/cnt: from a registered region if one covers it
// (kept handle, never released per message), else per message.
static void deviceNcpyBufferInit(CmiNcpyBuffer& b, const void* ptr, size_t cnt, void* opinfo) {
  b.deviceRdmaOpInfo = opinfo;
  {
    std::lock_guard<std::mutex> g(device_reg_mutex);
    const DeviceRegion* r = deviceRegionFind((uintptr_t)ptr, cnt);
    if (r) {
      b.init(ptr, cnt, CMK_BUFFER_UNREG, CMK_BUFFER_NODEREG);  // common info only, no registration
      memcpy(b.layerInfo + CmiGetRdmaCommonInfoSize(), r->layerInfo, CMK_NOCOPY_DIRECT_BYTES);
      b.regMode = CMK_BUFFER_REG;   // rdmaGet must not register it again
      b.isRegistered = true;
      return;
    }
  }
  b.init(ptr, cnt, CMK_BUFFER_REG, CMK_BUFFER_DEREG);  // per message; released on completion
}

/* Piggybacked acknowledgements for the RDMA path.
 *
 * Two things the sender needs to learn after a get completes on the far
 * side: that a per-message registration may be released (fix (a)), and,
 * if it asked, that the source buffer may be reused (the source callback).
 * Both used to cost one message per buffer.  Now the sender mints an 8-byte
 * id per buffer that needs either, keeps the work in a per-process table,
 * and the receiver hands the id back inside the next device message it
 * sends to that PE (CkDeviceBuffer::acks, capacity CK_DEVICE_ACK_CAP, on the
 * first buffer of the message).  Ids that find no such message go out in a
 * standalone DeviceAckMsg when the block fills, and at idle.  A stencil
 * therefore sends no standalone acks at all in steady state.
 * CK_GPU_ACK_NOPIGGY in the environment flushes every id at once, for
 * measurement of what the piggyback saves. */
#include <vector>

struct DevicePendingAck {
  const void* ptr; size_t cnt;
  unsigned char needs_dereg; unsigned char has_cb;
  char mr[CMK_NOCOPY_DIRECT_BYTES];
  CkCallback cb;
};
struct DeviceAckMsg {
  char header[CmiMsgHeaderSizeBytes];
  int count;
  uint64_t ids[CK_DEVICE_ACK_CAP];
};
static std::unordered_map<uint64_t, DevicePendingAck> device_pending;
static std::unordered_map<int, std::vector<uint64_t>> device_ack_acc;  // dest PE -> ids owed
static std::mutex device_ack_mutex;
static uint64_t device_ack_seq = 0;
static int device_ack_nopiggy = -1;

static void deviceResolveAck(uint64_t id) {
  DevicePendingAck rec;
  {
    std::lock_guard<std::mutex> g(device_ack_mutex);
    auto it = device_pending.find(id);
    if (it == device_pending.end()) {
      CmiPrintf("[%d] CkRdmaDevice> WARNING: ack for unknown id %llu\n", CmiMyPe(), (unsigned long long)id);
      return;
    }
    rec = it->second;
    device_pending.erase(it);
  }
  if (rec.needs_dereg) CmiDeregisterMem(rec.ptr, rec.mr, CkMyPe(), CMK_BUFFER_REG);
  if (rec.has_cb) rec.cb.send();
}

extern "C" {
  void* device_ack_bridge(void* arg) {
    DeviceAckMsg* m = (DeviceAckMsg*)arg;
    QdProcess(1);
    for (int i = 0; i < m->count; i++) deviceResolveAck(m->ids[i]);
    CmiFree(m);
    return NULL;
  }
  int device_ack_handler;
}

static void deviceSendAcks(int pe, const uint64_t* ids, int n) {
  for (int off = 0; off < n; off += CK_DEVICE_ACK_CAP) {
    int k = std::min(n - off, (int)CK_DEVICE_ACK_CAP);
    DeviceAckMsg* m = (DeviceAckMsg*)CmiAlloc(sizeof(DeviceAckMsg));
    m->count = k;
    memcpy(m->ids, ids + off, k * sizeof(uint64_t));
    CmiSetHandler(m, device_ack_handler);
    QdCreate(1);
    CmiSyncSendAndFree(pe, sizeof(DeviceAckMsg), m);
  }
}

static void deviceFlushAcks(int pe) {
  std::vector<uint64_t> ids;
  {
    std::lock_guard<std::mutex> g(device_ack_mutex);
    auto it = device_ack_acc.find(pe);
    if (it == device_ack_acc.end()) return;
    ids.swap(it->second);
    device_ack_acc.erase(it);
  }
  QdProcess((int)ids.size());
  deviceSendAcks(pe, ids.data(), (int)ids.size());
}

static void deviceFlushAllAcks(void*) {
  std::vector<int> pes;
  {
    std::lock_guard<std::mutex> g(device_ack_mutex);
    if (device_ack_acc.empty()) return;
    for (auto& kv : device_ack_acc) pes.push_back(kv.first);
  }
  for (int pe : pes) deviceFlushAcks(pe);
}

void CkRdmaDeviceInit() {
  CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE, (CcdCondFn)deviceFlushAllAcks, NULL);
}

// The receiver owes 'id' to 'pe'.  Held for the next device message to that
// PE; sent now if the block is full or piggybacking is disabled.
static void deviceQueueAck(int pe, uint64_t id) {
  if (device_ack_nopiggy < 0) {
    device_ack_nopiggy = (getenv("CK_GPU_ACK_NOPIGGY") != nullptr) ? 1 : 0;
    if (device_ack_nopiggy && CkMyPe() == 0)
      CmiPrintf("CkRdmaDevice> acks sent standalone, one per buffer (CK_GPU_ACK_NOPIGGY)\n");
  }
  size_t n;
  {
    std::lock_guard<std::mutex> g(device_ack_mutex);
    auto& v = device_ack_acc[pe];
    v.push_back(id);
    n = v.size();
  }
  QdCreate(1);
  if (device_ack_nopiggy || n >= CK_DEVICE_ACK_CAP) deviceFlushAcks(pe);
}

// Sender side: drain what this PE owes 'pe' into the first buffer of the
// message about to go there.
static void deviceDrainAcksInto(int pe, CkDeviceBuffer* b) {
  b->ack_count = 0;
  int n = 0;
  {
    std::lock_guard<std::mutex> g(device_ack_mutex);
    auto it = device_ack_acc.find(pe);
    if (it == device_ack_acc.end()) return;
    auto& v = it->second;
    n = std::min((int)v.size(), (int)CK_DEVICE_ACK_CAP);
    memcpy(b->acks, v.data(), n * sizeof(uint64_t));
    v.erase(v.begin(), v.begin() + n);
    if (v.empty()) device_ack_acc.erase(it);
  }
  b->ack_count = (unsigned char)n;
  QdProcess(n);
}

// Sender side: mint an id for a buffer that will need a release and/or a
// source callback when the far side's get completes.  0 if neither.
static uint64_t deviceMintAck(CmiNcpyBuffer& nb, CkCallback& cb) {
  DevicePendingAck rec;
  rec.ptr = nb.ptr; rec.cnt = nb.cnt;
  rec.needs_dereg = (nb.isRegistered && nb.deregMode == CMK_BUFFER_DEREG) ? 1 : 0;
  rec.has_cb = (cb.type != CkCallback::ignore) ? 1 : 0;
  if (!rec.needs_dereg && !rec.has_cb) return 0;
  if (rec.needs_dereg) memcpy(rec.mr, nb.layerInfo + CmiGetRdmaCommonInfoSize(), CMK_NOCOPY_DIRECT_BYTES);
  if (rec.has_cb) { rec.cb = cb; cb = CkCallback(CkCallback::ignore); }  // the sender fires it, not the receiver
  uint64_t id;
  {
    std::lock_guard<std::mutex> g(device_ack_mutex);
    id = ++device_ack_seq;
    device_pending.emplace(id, rec);
  }
  return id;
}

static void deviceDeregAnnounce() {
  static bool done = false;
  if (!done) {
    done = true;
    if (CkMyPe() == 0)
      CmiPrintf("CkRdmaDevice> device RDMA registrations released on completion\n");
  }
}

static void releaseDeviceRegistrations(NcpyOperationInfo* info, uint64_t ack_id) {
  deviceDeregAnnounce();
  if (info->isDestRegistered && info->destDeregMode == CMK_BUFFER_DEREG) {
    CmiDeregisterMem(info->destPtr,
                     info->destLayerInfo + CmiGetRdmaCommonInfoSize(),
                     info->destPe, info->destRegMode);
    info->isDestRegistered = 0;
  }
  if (ack_id != 0) {
    // The sender minted an id: it releases its own registration and fires
    // its own callback when the id comes back, piggybacked or flushed.
    deviceQueueAck(info->srcPe, ack_id);
    info->isSrcRegistered = 0;
    return;
  }
  if (info->isSrcRegistered && info->srcDeregMode == CMK_BUFFER_DEREG) {
    DeviceDeregMsg* m = (DeviceDeregMsg*)CmiAlloc(sizeof(DeviceDeregMsg));
    m->ptr = info->srcPtr;
    m->pe = info->srcPe;
    m->regMode = info->srcRegMode;
    memcpy(m->mr, info->srcLayerInfo + CmiGetRdmaCommonInfoSize(),
           CMK_NOCOPY_DIRECT_BYTES);
    CmiSetHandler(m, device_dereg_handler);
    QdCreate(1);  // matched in device_dereg_bridge
    CmiSyncSendAndFree(info->srcPe, sizeof(DeviceDeregMsg), m);
    info->isSrcRegistered = 0;
  }
}

// Completion of an inter-node (RDMA) device transfer. Reached through
// CkRdmaDirectAckHandler, which routes any NcpyOperationInfo carrying a
// deviceRdmaOpInfo here; 'data' is that NcpyOperationInfo.
void CkRdmaDeviceRecvHandler(void* data)
{
  NcpyOperationInfo* ncpy_op_info = (NcpyOperationInfo*)data;
  DeviceRdmaOp* op = (DeviceRdmaOp*)(ncpy_op_info->deviceRdmaOpInfo);

  // The get is complete: release both registrations now, before any bounce.
  // This runs on the PE the backend raised the completion on, and the
  // registrations are process-level state, so no hand-off is needed; the
  // flags it clears keep the loopback copy below from releasing them twice.
  releaseDeviceRegistrations(ncpy_op_info, op->tag);

  // The backend raises the completion on whichever PE of this process was
  // progressing the network, which is not necessarily the PE that posted the
  // get. rdma_info below is per-destination-PE bookkeeping, so hand the
  // completion to the destination PE before touching it.
  if (op->dest_pe != CmiMyPe()) {
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

  // Matches the QdCreate issued when the get was posted in
  // CkRdmaDeviceIssueRgets.
  QdProcess(1);

  DeviceRdmaInfo* info = op->info;

  // Invoke source callback
  if (op->src_cb) {
    CkCallback* cb = (CkCallback*)op->src_cb;
    cb->send();
    delete cb;
  }

  // Update counter (there may be multiple buffers in transit)
  info->counter++;

  // Once every buffer has landed, run the real entry method. The counter is
  // safe to touch unguarded because both completion routes -- the stream
  // callback and, via loopback_bridge, the network ack -- run on this PE.
  if (info->counter == info->n_ops) {
    QdCreate(1);
    enqueueNcpyMessage(op->dest_pe, info->msg);
    // 'info' is the whole rdma_data block CkRdmaDeviceIssueRgets allocated,
    // save_op included; the last completion is the last reader of it, so this
    // is where it goes. (The source branch leaves this free commented out and
    // leaks the block once per message.)
    CmiFree(info);
  }
}

// Completion of an intra-node device transfer (MEMCPY or IPC). Invoked from
// the HAPI stream-completion poll, so 'data' is the DeviceRdmaOp directly.
void CkRdmaDeviceRecvHandler(void* data, void* msg)
{
  DeviceRdmaOp* op = (DeviceRdmaOp*)data;
  DeviceRdmaInfo* info = op->info;

  // Invoke source callback
  if (op->src_cb) {
    CkCallback* cb = (CkCallback*)op->src_cb;
    cb->send();
    delete cb;
  }

  info->counter++;

  if (info->counter == info->n_ops) {
    QdCreate(1);
    enqueueNcpyMessage(op->dest_pe, info->msg);
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
  // Create an IPC handle for inter-process communication
  hapiCheck(hapiIpcGetMemHandle(&hapi_ipc_handle, (void*)ptr));
}

void CkDevicePersistent::close() {
  // Close the IPC handle if it was opened
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
    hapiCheck(hapiMemcpyAsync((void*)ptr, src.ptr, cnt, hapiMemcpyDeviceToDevice,
          hapi_stream));
  } else if (mode == CkNcpyModeDevice::IPC) {
    if (!src.ipc_open) {
      hapiCheck(hapiIpcOpenMemHandle(&src.ipc_ptr, src.hapi_ipc_handle,
            hapiIpcMemLazyEnablePeerAccess));
      src.ipc_open = true;
    }
    hapiCheck(hapiMemcpyAsync((void*)ptr, src.ipc_ptr, cnt,
          hapiMemcpyDeviceToDevice, hapi_stream));
  } else {
    CkAbort("Persistent GPU messaging is currently not supported for inter-node messages");
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
    hapiCheck(hapiMemcpyAsync((void*)dst.ptr, ptr, cnt, hapiMemcpyDeviceToDevice,
          hapi_stream));
  } else if (mode == CkNcpyModeDevice::IPC) {
    if (!dst.ipc_open) {
      hapiCheck(hapiIpcOpenMemHandle(&dst.ipc_ptr, dst.hapi_ipc_handle,
            hapiIpcMemLazyEnablePeerAccess));
      dst.ipc_open = true;
    }
    hapiCheck(hapiMemcpyAsync(dst.ipc_ptr, ptr, cnt, hapiMemcpyDeviceToDevice,
          hapi_stream));
  } else {
    CkAbort("Persistent GPU messaging is not yet supported for inter-node messages");
  }

  // Set callbacks to be invoked once put is complete
  if (cb.type != CkCallback::ignore) {
    hapiAddCallback(hapi_stream, cb, cb_msg);
  }
  if (dst.cb.type != CkCallback::ignore) {
    hapiAddCallback(hapi_stream, dst.cb, dst.cb_msg);
  }

  return CkDeviceStatus::incomplete;
}

/****************************** Recv Entry Method API ******************************/

// Invoked after post entry method
void CkRdmaDeviceIssueRgets(envelope *env, int numops, void **arrPtrs, int *arrSizes, CkDeviceBufferPost *postStructs) {
  // Change message header to invoke regular entry method
  CMI_ZC_MSGTYPE(env) = CMK_REG_NO_ZC_MSG;

  // Create a copy of this message for regular entry method invocation
  // FIXME: Reuse the old message instead of creating a new one
  void* old_msg = EnvToUsr(env);
  envelope* new_env = UsrToEnv(CkCopyMsg(&old_msg));

  // Start unpacking marshalled message
  PUP::fromMem up((void *)((CkMarshallMsg *)EnvToUsr(env))->msgBuf);
  int received_numops;
  up|received_numops;
  CkAssert(numops == received_numops);

  CkDeviceBuffer source;

  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

  // Find which mode of transfer should be used
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
    // Acks the sender owes THIS PE ride on its first buffer: resolve them.
    if (i == 0)
      for (int k = 0; k < source.ack_count; k++) deviceResolveAck(source.acks[k]);

    if (arrSizes[i] > source.cnt) {
      CkAbort("CkRdmaDeviceIssueRgets: posted data size is larger than source data size!");
    }

    // Store information about this buffer
    DeviceRdmaOp& save_op = *(DeviceRdmaOp*)((char*)rdma_data
        + sizeof(DeviceRdmaInfo) + sizeof(DeviceRdmaOp) * i);
    // The completion has to re-enqueue the message on the PE that posted these
    // gets -- this one -- which is not always the PE the sender addressed. A
    // nodegroup entry method runs on whichever PE of the destination process
    // picks the message up, so the sender can only ever name the process.
    save_op.dest_pe  = CkMyPe();
    save_op.dest_ptr = arrPtrs[i];
    save_op.size = (size_t)arrSizes[i];
    save_op.info = rdma_info;
    save_op.src_cb = (source.cb.type != CkCallback::ignore) ? new CkCallback(source.cb) : nullptr;
    save_op.dst_cb = nullptr;
    save_op.tag = 0;

    // What has to hold is that the sender picked the same transfer mode we are
    // about to use, and mode is a property of the process pair, not the PE
    // pair: MEMCPY means "sender's pointer is valid in this address space", IPC
    // means "reachable through this process's IPC handles". So compare
    // processes. A stricter PE-equality test would reject every nodegroup
    // message that lands on a PE other than the one the sender named.
    // TODO: Handle a genuinely migrated object instead of aborting.
    if (CmiNodeOf(source.dest_pe) != CmiMyNode()) {
      CmiPrintf("PE %d (process %d) received a device buffer the sender (PE %d) "
                "addressed to PE %d (process %d)\n",
          CkMyPe(), CmiMyNode(), env->getSrcPe(), source.dest_pe,
          CmiNodeOf(source.dest_pe));
      CkAbort("Destination process does not match the one the sender determined. "
          "Please enable CMK_GLOBAL_LOCATION_UPDATE.");
    }

    // Destination buffer (on this receiver)
    CkDeviceBuffer dest((const void *)arrPtrs[i], arrSizes[i]);

    // Perform data transfers
    if (mode == CkNcpyModeDevice::MEMCPY) {
      // Source and destination PEs are in the same process (logical node),
      // so the source pointer is valid here: copy straight across.
      hapiCheck(hapiMemcpyAsync((void*)dest.ptr, source.ptr, dest.cnt,
            hapiMemcpyDeviceToDevice, postStructs[i].hapi_stream));
    } else if (mode == CkNcpyModeDevice::IPC && csv_gpu_manager.use_shm) {
      // Inter-process, same physical node: read out of the sender's device
      // communication buffer through its IPC handle.
      hapi_ipc_device_info& device_info =
        csv_gpu_manager.hapi_ipc_device_infos[source.device_idx];

      // 1. Make the user-provided stream wait on the sender's IPC event
      //    (source buffer -> device comm buffer, recorded on the sender)
      hapiCheck(hapiStreamWaitEvent(postStructs[i].hapi_stream,
            device_info.src_event_pool[source.event_idx], 0));

      // 2. Copy from the sender's device comm buffer to the destination buffer
      hapiCheck(hapiMemcpyAsync((void*)dest.ptr,
            (void*)((char*)device_info.buffer + source.comm_offset),
            dest.cnt, hapiMemcpyDeviceToDevice, postStructs[i].hapi_stream));

      // 3. Record the IPC event the sender polls to decide when the block in
      //    its device comm buffer (and the event pair) can be released
      hapiCheck(hapiEventRecord(device_info.dst_event_pool[source.event_idx],
            postStructs[i].hapi_stream));

      // 4. Tell the sender, through shared memory, that step 3 has been issued
      //    and the event is now worth querying
      hapi_ipc_event_shared* shm_event_shared =
        (hapi_ipc_event_shared*)((char*)csv_gpu_manager.shm_ptr
            + csv_gpu_manager.shm_chunk_size * source.device_idx
            + sizeof(hapiIpcMemHandle_t)) + source.event_idx;
      __atomic_store_n(&shm_event_shared->dst_flag, 1, __ATOMIC_RELEASE);
    } else {
      // Inter-node (or intra-node without the shared-memory IPC path): pull the
      // source buffer over the network with a device-to-device RDMA get.
      // Completion arrives at CkRdmaDeviceRecvHandler(void*) by way of
      // CkRdmaDirectAckHandler, which recognises the operation as a device one
      // from the deviceRdmaOpInfo we attach to the destination buffer here.
      QdCreate(1);
      // With an ack id the SENDER fires its source callback (on ack), not us.
      save_op.tag = source.ack_id;
      if (save_op.tag != 0 && save_op.src_cb) { delete (CkCallback*)save_op.src_cb; save_op.src_cb = nullptr; }
      CmiNcpyBuffer lci_dest_ncpy_buffer;
      deviceNcpyBufferInit(lci_dest_ncpy_buffer, arrPtrs[i], (size_t)arrSizes[i], (void*)(&save_op));
      lci_dest_ncpy_buffer.rdmaGet(source.lci_ncpy_buffer, 0, nullptr, nullptr);
      continue;  // no stream callback: the network, not the GPU, completes this
    }

    // Intra-node transfers finish on the stream, so hang the completion off it.
    // CK_GPU_STREAMORDER prototype (2026-09-01): "stream-ordered delivery".
    // The copy into the destination buffer is already enqueued on the stream
    // the receiver posted, so any device work the entry method issues on that
    // same stream is ordered after it.  With the env var set, and when the
    // sender asked for no source callback (which would otherwise promise the
    // source buffer is reusable), complete the operation now instead of
    // spending an event record + pending polls per message.  The entry method
    // must not touch the buffer from the host or from another stream without
    // its own synchronisation -- that is the contract this mode trades on.
    static int stream_order = -1;
    if (stream_order < 0) {
      stream_order = (getenv("CK_GPU_STREAMORDER") != nullptr) ? 1 : 0;
      if (stream_order && CkMyPe() == 0)
        CmiPrintf("CkRdmaDevice> stream-ordered delivery for intra-node device receives (CK_GPU_STREAMORDER)\n");
    }
    if (stream_order && save_op.src_cb == nullptr) {
      CkRdmaDeviceRecvHandler(&save_op, nullptr);
      continue;
    }
    hapiAddCallback(postStructs[i].hapi_stream, CkCallback(CkRdmaDeviceRecvHandler, &save_op));
  }
}

// Reclaims IPC events whose transfers have completed, then hands out a free
// one. Two events are used per message:
//   1) recorded by the sender after 'source buffer -> device comm buffer'.
//      The receiver waits on it before starting its own copy.
//   2) recorded by the receiver after 'device comm buffer -> dest buffer'.
//      The sender polls it to know when the comm-buffer block is reusable.
// Returns -1 when the pool is exhausted.
static int findFreeIpcEvent(DeviceManager* dm, const size_t comm_offset, int cpv_my_device_id) {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  int pool_size = csv_gpu_manager.hapi_ipc_event_pool_size_pe;
  int pool_start = CkMyRank() * pool_size;
  const int my_device_index =
    csv_gpu_manager.device_count * CmiMyNodeRankLocal() + cpv_my_device_id;
  hapi_ipc_device_info& my_device_info =
    csv_gpu_manager.hapi_ipc_device_infos[my_device_index];

  // Free IPC events that are complete
  // TODO: Don't do this every time but only when the event pool is somewhat empty
  for (int i = pool_start; i < pool_start + pool_size; i++) {
    int& event_flag = my_device_info.event_pool_flags[i];
    hapiEvent_t& ev = my_device_info.dst_event_pool[i];
    size_t& buff_offset = my_device_info.event_pool_buff_offsets[i];
    if (event_flag == 0) continue;

    // Only worth querying once the receiver says it has issued its copy
    hapi_ipc_event_shared* shm_event_shared =
      (hapi_ipc_event_shared*)((char*)csv_gpu_manager.shm_ptr
          + csv_gpu_manager.shm_chunk_size * my_device_index
          + sizeof(hapiIpcMemHandle_t)) + i;
    if (!__atomic_load_n(&shm_event_shared->dst_flag, __ATOMIC_ACQUIRE)) continue;

    if (hapiEventQuery(ev) == hapiSuccess) {
      // The receiver is done reading, so the block can go back to the allocator
      if (event_flag != 1) {
        CkAbort("Retrieved hapiSuccess for a free IPC event");
      }
      dm->free_comm_buffer(buff_offset);

      event_flag = 0;
      __atomic_store_n(&shm_event_shared->dst_flag, 0, __ATOMIC_RELEASE);
    }
  }

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
  CkNcpyModeDevice transfer_mode = findTransferModeDevice(CkMyPe(), dest_pe);

  for (int i = 0; i < numops; i++) {
    buffers[i]->dest_pe = dest_pe;
    buffers[i]->dest_mpi_rank = CmiNodeOf(dest_pe);
    buffers[i]->src_pe = CmiMyPe();
    buffers[i]->src_mpi_rank = CmiNodeOf(CmiMyPe());
  }

  if (transfer_mode == CkNcpyModeDevice::MEMCPY) {
    // The receiver dereferences buffers[i]->ptr directly, so the producing
    // kernel/copy has to have retired before the metadata message goes out.
    for (int i = 0; i < numops; i++)
      hapiCheck(hapiStreamSynchronize(buffers[i]->hapi_stream));
    return;
  }

  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  int cpv_my_device_id = CpvAccess(my_device_id);

  if (transfer_mode == CkNcpyModeDevice::IPC && csv_gpu_manager.use_shm) {
    // Stage each buffer into this device's communication buffer, which the
    // receiving process can reach through the IPC handle exchanged at startup.
    DeviceManager* dm = csv_gpu_manager.device_map[CkMyPe()];

    for (int i = 0; i < numops; i++) {
      // A buffer that already lives inside the comm buffer (the LB region)
      // needs no staging copy -- it is already visible through the handle.
      bool is_lb_buffer = ((size_t)((char*)(buffers[i]->ptr)
            - (char*)(dm->comm_buffer->base_ptr)) < dm->comm_buffer->total_size);
#if CMK_SMP
      CmiLock(dm->lock);
#endif
      void* alloc_comm_buffer;
      if (is_lb_buffer) {
        alloc_comm_buffer = const_cast<void*>(buffers[i]->ptr);
      } else {
        alloc_comm_buffer = dm->alloc_comm_buffer(buffers[i]->cnt);
        if (alloc_comm_buffer == nullptr) {
          CkAbort("PE %d, device %d: Not enough memory on device communication buffer (%zu free)",
              CkMyPe(), dm->global_index, dm->get_comm_buffer_free_size());
        }
      }
      buffers[i]->comm_offset = (char*)alloc_comm_buffer - (char*)dm->comm_buffer->base_ptr;
      buffers[i]->device_idx = csv_gpu_manager.device_count * CmiMyNodeRankLocal() + cpv_my_device_id;
      buffers[i]->event_idx = findFreeIpcEvent(dm, buffers[i]->comm_offset, cpv_my_device_id);
      // FIXME: Instead of aborting, we could create IPC events on demand
      // (though they cannot then be shared through the shared memory region
      // allocated at init time)
      if (buffers[i]->event_idx == -1) {
        CkAbort("GPU IPC event pool empty");
      }
#if CMK_SMP
      CmiUnlock(dm->lock);
#endif

      if (!is_lb_buffer) {
        hapiCheck(hapiMemcpyAsync(alloc_comm_buffer, buffers[i]->ptr, buffers[i]->cnt,
              hapiMemcpyDeviceToDevice, buffers[i]->hapi_stream));
      }

      hapi_ipc_device_info& my_device_info =
        csv_gpu_manager.hapi_ipc_device_infos[buffers[i]->device_idx];
      hapiCheck(hapiEventRecord(my_device_info.src_event_pool[buffers[i]->event_idx],
            buffers[i]->hapi_stream));
    }
  } else {
    // Inter-node: expose the source buffer for the receiver's RDMA get. The
    // registration travels to the receiver inside lci_ncpy_buffer when the
    // CkDeviceBuffer is PUP'd into the metadata message.
    for (int i = 0; i < numops; i++) {
      hapiCheck(hapiStreamSynchronize(buffers[i]->hapi_stream));
      deviceNcpyBufferInit(buffers[i]->lci_ncpy_buffer, buffers[i]->ptr, buffers[i]->cnt, nullptr);
      buffers[i]->ack_id = deviceMintAck(buffers[i]->lci_ncpy_buffer, buffers[i]->cb);
    }
  }
  // Whatever this PE owes the destination rides on the first buffer.
  if (numops > 0) deviceDrainAcksInto(dest_pe, buffers[0]);
}

#endif // (CMK_CUDA || CMK_HIP) && CMK_RECONVERSE
