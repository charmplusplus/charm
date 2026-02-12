#include "sycl_impl.h"
#include "gpumanager.h"

CsvExtern(GPUManager, gpu_manager);

// Global device list that includes sub-devices (tiles) for Intel GPUs
static std::vector<sycl::device> all_devices;
static bool devices_enumerated = false;

// Helper function to enumerate devices including sub-devices (tiles)
static void enumerate_devices_with_tiles() {
  if (devices_enumerated) return;
  
  std::vector<sycl::device> root_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  
  for (auto& root_dev : root_devices) {
    // Try to partition device into tiles (sub-devices by affinity domain)
    try {
      auto sub_devices = root_dev.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(
        sycl::info::partition_affinity_domain::next_partitionable);
      
      if (!sub_devices.empty()) {
        // Device has tiles, add each tile as a separate device
        for (auto& sub_dev : sub_devices) {
          all_devices.push_back(sub_dev);
        }
      } else {
        // No sub-devices, use root device
        all_devices.push_back(root_dev);
      }
    } catch (...) {
      // Partitioning not supported or failed, use root device
      all_devices.push_back(root_dev);
    }
  }
  
  devices_enumerated = true;
}

int hapiSetDevice(int dev) {
  enumerate_devices_with_tiles();
  
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  CmiLock(csv_gpu_manager.context_lock_);
  if (csv_gpu_manager.ctx_initialized_) {
    CmiUnlock(csv_gpu_manager.context_lock_);
    return hapiSuccess;
  }
  csv_gpu_manager.ctx_initialized_ = true;
  csv_gpu_manager.ctx = sycl::context(all_devices[dev]);
  csv_gpu_manager.dev = all_devices[dev];
  CmiUnlock(csv_gpu_manager.context_lock_);
  return hapiSuccess;
}

int hapiStreamCreate(sycl::queue** stream) {
    *(stream) = new sycl::queue(CsvAccess(gpu_manager).ctx, CsvAccess(gpu_manager).dev, sycl::property::queue::in_order());
    return hapiSuccess;
}

int hapiDeviceCanAccessPeer(int* canAccess, int devIdx1, int devIdx2) {
    enumerate_devices_with_tiles();
    sycl::device dev0 = all_devices[devIdx1];
    sycl::device dev1 = all_devices[devIdx2];

    // Note: This requires Intel extensions for P2P capabilities
    // If not available, assume no P2P
    *canAccess = 0;
    return hapiSuccess;
}

int hapiIpcGetMemHandle(hapiIpcMemHandle_t* handle, void* ptr) {
    return zeMemGetIpcHandle(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(CsvAccess(gpu_manager).ctx),
                      ptr, &((handle)->ze_handle));
}

int hapiIpcGetEventHandle(hapiIpcEventHandle_t* handle, hapiEvent_t event) {
    ze_event_handle_t hEvent = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(event->ev);
    ze_event_pool_handle_t hPool;
    zeEventGetEventPool(hEvent, &hPool);
    return zeEventPoolGetIpcHandle(hPool, &((handle)->ze_handle));
}

int hapiIpcOpenMemHandle(void** ptr, hapiIpcMemHandle_t handle, int flags) {
    ze_context_handle_t hCtx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(CsvAccess(gpu_manager).ctx);
    ze_device_handle_t hDev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(CsvAccess(gpu_manager).dev);
    return zeMemOpenIpcHandle(hCtx, hDev, (handle).ze_handle, flags, ptr);
}

int hapiGetDevice(int* dev) {
    enumerate_devices_with_tiles();
    for (size_t i = 0; i < all_devices.size(); i++) {
        if (all_devices[i] == CsvAccess(gpu_manager).dev) {
            *dev = i;
            return hapiSuccess;
        }
    }
    return hapiErrorInitializationError;
}

int hapiMallocImpl(void** ptr, size_t size) {
    *(ptr) = sycl::malloc_device(size, sycl::queue(CsvAccess(gpu_manager).ctx, CsvAccess(gpu_manager).dev));
    if (*(ptr) == nullptr) {
        return hapiErrorMemoryAllocation;
    }
    return hapiSuccess;
}

int hapiFree(void* ptr) {
    sycl::free(ptr, sycl::queue(CsvAccess(gpu_manager).ctx, CsvAccess(gpu_manager).dev));
    return hapiSuccess;
}

int hapiMallocHostImpl(void** ptr, size_t size) {
    *(ptr) = sycl::malloc_host(size, sycl::queue(CsvAccess(gpu_manager).ctx, CsvAccess(gpu_manager).dev));
    if (*(ptr) == nullptr) {
        return hapiErrorMemoryAllocation;
    }
    return hapiSuccess;
}

int hapiFreeHost(void* ptr) {
    sycl::free(ptr, sycl::queue(CsvAccess(gpu_manager).ctx, CsvAccess(gpu_manager).dev));
    return hapiSuccess;
}

int hapiMemcpy(void* dst, const void* src, size_t size, hapiMemcpyKind kind) {
    sycl::queue q(CsvAccess(gpu_manager).ctx, CsvAccess(gpu_manager).dev);
    if (kind == hapiMemcpyHostToDevice || kind == hapiMemcpyHostToHost || 
        kind == hapiMemcpyDeviceToDevice || kind == hapiMemcpyDeviceToHost) {
        q.memcpy(dst, src, size).wait();
    } else {
        return hapiErrorNotReady;
    }
    return hapiSuccess;
}

int hapiMemcpyAsync(void* dst, const void* src, size_t size, hapiMemcpyKind kind, hapiStream_t stream) {
    if (kind == hapiMemcpyHostToDevice || kind == hapiMemcpyHostToHost || 
        kind == hapiMemcpyDeviceToDevice || kind == hapiMemcpyDeviceToHost) {
        stream->memcpy(dst, src, size);
    } else {
        return hapiErrorNotReady;
    }
    return hapiSuccess;
}

int hapiEventCreateWithFlags(hapiEvent_t* event, unsigned int flags) {
    *event = new hapiEventStruct();
    (*event)->flag = flags;
    
    if (flags & hapiEventInterprocess) {
        GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
        ze_context_handle_t hCtx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(csv_gpu_manager.ctx);
        ze_device_handle_t hDev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(csv_gpu_manager.dev);

        ze_event_pool_desc_t poolDesc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr, 
                                        ZE_EVENT_POOL_FLAG_IPC | ZE_EVENT_POOL_FLAG_HOST_VISIBLE, 1};
        ze_event_pool_handle_t hPool;
        zeEventPoolCreate(hCtx, &poolDesc, 1, &hDev, &hPool);

        ze_event_desc_t eventDesc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, 0, 0, 0};
        ze_event_handle_t hEvent;
        zeEventCreate(hPool, &eventDesc, &hEvent);
        sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::event> eventInteropInput = {
            hEvent,             // The native Level Zero event handle
            sycl::ext::oneapi::level_zero::ownership::keep // SYCL will not destroy the native handle
        };
        (*event)->ev = sycl::make_event<sycl::backend::ext_oneapi_level_zero>(eventInteropInput, csv_gpu_manager.ctx);
    }
    
    return hapiSuccess;
}

int hapiEventRecord(hapiEvent_t event, hapiStream_t stream) {
    event->ev = stream->ext_oneapi_submit_barrier();
    return hapiSuccess;
}

int hapiEventQuery(hapiEvent_t event) {
    auto status = event->ev.get_info<sycl::info::event::command_execution_status>();
    if (status == sycl::info::event_command_status::complete) {
        return hapiSuccess;
    } else {
        return hapiErrorNotReady;
    }
}

int hapiEventDestroy(hapiEvent_t event) {
    delete event;
    return hapiSuccess;
}

int hapiStreamSynchronize(hapiStream_t stream) {
    stream->wait_and_throw();
    return hapiSuccess;
}

int hapiStreamWaitEvent(hapiStream_t stream, hapiEvent_t event, unsigned int flags) {
    event->ev.wait_and_throw();
    return hapiSuccess;
}

int hapiStreamDestroy(hapiStream_t stream) {
    delete stream;
    return hapiSuccess;
}

int hapiLaunchHostFunc(hapiStream_t stream, void (*func)(void*), void* args) {
    stream->submit([=](sycl::handler& cgh) {
        cgh.host_task([=]() {
            func(args);
        });
    });
    return hapiSuccess;
}

int hapiGetDeviceProperties(hapiDeviceProp* prop, int dev) {
    enumerate_devices_with_tiles();
    if (dev < 0 || dev >= static_cast<int>(all_devices.size())) {
        return hapiErrorInitializationError;
    }
    *prop = all_devices[dev];
    return hapiSuccess;
}

int hapiIpcOpenEventHandle(hapiEvent_t* event, hapiIpcEventHandle_t handle) {
    GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
    ze_context_handle_t hCtx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(csv_gpu_manager.ctx);
    
    ze_event_pool_handle_t hPool;
    zeEventPoolOpenIpcHandle(hCtx, handle.ze_handle, &hPool);

    ze_event_desc_t eventDesc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, 0, 0, 0};
    ze_event_handle_t hEvent;
    zeEventCreate(hPool, &eventDesc, &hEvent);

    *event = new hapiEventStruct();
    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::event> eventInteropInput = {
        hEvent,             // The native Level Zero event handle
        sycl::ext::oneapi::level_zero::ownership::keep // SYCL will not destroy the native handle
    };
    (*event)->ev = sycl::make_event<sycl::backend::ext_oneapi_level_zero>(eventInteropInput, csv_gpu_manager.ctx);
    (*event)->flag = hapiEventInterprocess;
    
    return hapiSuccess;
}

int getNumStreams(hapiDeviceProp& device_prop) {
    // This is a heuristic based on the compute capability of the device.
    // For simplicity, we return a fixed number here, but this can be
    // enhanced to return different numbers based on the device properties.
    return 4; // Default to 4 streams for concurrency
}

int hapiGetDeviceCount(int* count) {
    enumerate_devices_with_tiles();
    *count = all_devices.size();
    return hapiSuccess;
}