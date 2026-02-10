#include "sycl_impl.h"
#include "gpumanager.h"

CsvExtern(GPUManager, gpu_manager);

int hapiSetDevice(int dev) {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  CmiLock(csv_gpu_manager.context_lock_);
  if (csv_gpu_manager.ctx_initialized_) {
    CmiUnlock(csv_gpu_manager.context_lock_);
    return hapiSuccess;
  }
  csv_gpu_manager.ctx_initialized_ = true;
  std::vector<sycl::device> devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  csv_gpu_manager.ctx = sycl::context(devices[dev]);
  csv_gpu_manager.dev = devices[dev];
  CmiUnlock(csv_gpu_manager.context_lock_);
  return hapiSuccess;
}

int hapiStreamCreate(sycl::queue** stream) {
    *(stream) = new sycl::queue(CsvAccess(gpu_manager).ctx, CsvAccess(gpu_manager).dev, sycl::property::queue::in_order());
    return hapiSuccess;
}

int hapiDeviceCanAccessPeer(int* canAccess, int devIdx1, int devIdx2) {
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    sycl::device dev0 = devices[devIdx1];
    sycl::device dev1 = devices[devIdx2];

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
    return zeEventPoolGetIpcHandle(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(event->ev),
                            &((handle)->ze_handle));
}

int hapiIpcOpenMemHandle(void** ptr, hapiIpcMemHandle_t handle, int flags) {
    ze_context_handle_t hCtx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(CsvAccess(gpu_manager).ctx);
    ze_device_handle_t hDev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(CsvAccess(gpu_manager).dev);
    return zeMemOpenIpcHandle(hCtx, hDev, (handle).ze_handle, flags, ptr);
}

int hapiGetDevice(int* dev) {
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i] == CsvAccess(gpu_manager).dev) {
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
        
        (*event)->ev = sycl::make_event<sycl::backend::ext_oneapi_level_zero>(
            hEvent, csv_gpu_manager.ctx, sycl::ext::oneapi::level_zero::ownership::keep);
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
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (dev < 0 || dev >= static_cast<int>(devices.size())) {
        return hapiErrorInitializationError;
    }
    *prop = devices[dev];
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
    (*event)->ev = sycl::make_event<sycl::backend::ext_oneapi_level_zero>(
        hEvent, csv_gpu_manager.ctx, sycl::ext::oneapi::level_zero::ownership::keep);
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
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    *count = devices.size();
    return hapiSuccess;
}