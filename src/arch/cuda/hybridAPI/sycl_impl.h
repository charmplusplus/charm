#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>

//#include "gpumanager.h"


// SYCL uses queues instead of streams
#define hapiStream_t sycl::queue*
// SYCL events
//#define hapiEvent_t sycl::event
#define hapiDeviceProp sycl::device

// Error codes
#define hapiErrorMemoryAllocation -1
#define hapiErrorInitializationError -2
#define hapiErrorNotReady -3
#define hapiSuccess 0
#define hapiError_t int

// Memory copy operations
#define hapiMemcpyKind int
#define hapiMemcpyHostToHost 0
#define hapiMemcpyHostToDevice 1
#define hapiMemcpyDeviceToHost 2
#define hapiMemcpyDeviceToDevice 3

// Event flags
#define hapiEventDisableTiming 0x01
#define hapiEventInterprocess 0x02
#define hapiIpcMemLazyEnablePeerAccess 0x01

#define hapiGetErrorString(err) \
    ((err) == hapiSuccess ? "Success" : "Error")


typedef struct {
    sycl::event ev;
    int flag;
}* hapiEvent_t;

// IPC Handles - Level Zero based
typedef struct {
    ze_ipc_mem_handle_t ze_handle;
} hapiIpcMemHandle_t;

typedef struct {
    ze_ipc_event_pool_handle_t ze_handle;
} hapiIpcEventHandle_t;


int hapiSetDevice(int dev) {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  CmiLock(csv_gpu_manager.context_lock_);
  if (csv_gpu_manager.ctx_initialized_) {
    CmiUnlock(csv_gpu_manager.context_lock_);
    return;
  }
  csv_gpu_manager.ctx_initialized_ = true;
  std::vector<sycl::device> devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  csv_gpu_manager.ctx = sycl::context(devices[dev]);
  csv_gpu_manager.dev = devices[dev];
  CmiUnlock(csv_gpu_manager.context_lock_);
  return 0;
}

int hapiStreamCreate(sycl::queue** stream) {
    *(stream) = new sycl::queue(CsvAccess(gpu_manager).ctx, CsvAccess(gpu_manager).dev, sycl::property::queue::in_order());
    return 0;
}

int hapiDeviceCanAccessPeer(int* canAccess, int devIdx1, int devIdx2) {
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    sycl::device dev0 = devices[devIdx1];
    sycl::device dev1 = devices[devIdx2];

    auto caps = dev0.get_info<sycl::ext::intel::info::device::p2p_capabilities>(dev1);
    *canAccess = caps & sycl::ext::intel::info::device::p2p_capability::access;
    return 0;
}

int hapiIpcGetMemHandle(hapiIpcMemHandle_t* handle, void* ptr) {
    return zeMemGetIpcHandle(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(CsvAccess(gpu_manager).ctx),
                      ptr, &((handle)->ze_handle));
}

int hapiIpcGetEventHandle(hapiIpcEventHandle_t* handle, hapiEvent_t event) {
    return zeEventPoolGetIpcHandle(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(*(event)),
                            &((handle)->ze_handle));
}

int hapiIpcOpenMemHandle(void** ptr, hapiIpcMemHandle_t handle, int flags) {
    return zeMemOpenIpcHandle(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(CsvAccess(gpu_manager).ctx),
                       CsvAccess(gpu_manager).dev, (handle).ze_handle, flags, ptr);
}

int hapiGetDevice(int* dev) {
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i] == CsvAccess(gpu_manager).dev) {
            *dev = i;
            return hapiSuccess;
        }
    }
}

int hapiMalloc(void** ptr, size_t size) {
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

int hapiMallocHost(void** ptr, size_t size) {
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
    if (kind == hapiMemcpyHostToDevice || kind == hapiMemcpyHostToHost || kind == hapiMemcpyDeviceToDevice || kind == hapiMemcpyDeviceToHost) {
        q.memcpy(dst, src, size).wait();
    } else {
        return hapiErrorNotReady;
    }
    return hapiSuccess;
}

int hapiMemcpyAsync(void* dst, const void* src, size_t size, hapiMemcpyKind kind, hapiStream_t stream) {
    if (kind == hapiMemcpyHostToDevice || kind == hapiMemcpyHostToHost || kind == hapiMemcpyDeviceToDevice || kind == hapiMemcpyDeviceToHost) {
        stream->memcpy(dst, src, size);
    } else {
        return hapiErrorNotReady;
    }
    return hapiSuccess;
}

int hapiCreateEventWithFlags(hapiEvent_t* event, unsigned int flags) {
    if (flags & hapiEventInterprocess) {
        GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
        ze_context_handle_t hCtx = sycl::get_native<backend::ext_oneapi_level_zero>(csv_gpu_manager.ctx);
        ze_device_handle_t hDev = sycl::get_native<backend::ext_oneapi_level_zero>(csv_gpu_manager.dev);

        ze_event_pool_desc_t poolDesc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr, 
                                        ZE_EVENT_POOL_FLAG_IPC | ZE_EVENT_POOL_FLAG_HOST_VISIBLE, 1};
        ze_event_pool_handle_t hPool;
        zeEventPoolCreate(hCtx, &poolDesc, 1, &hDev, &hPool);

        ze_event_desc_t eventDesc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, 0, 0, 0};
        ze_event_handle_t hEvent;
        zeEventCreate(hPool, &eventDesc, &hEvent);
        event->ev = sycl::make_event<backend::ext_oneapi_level_zero>(
            hEvent, csv_gpu_manager.ctx, keep_ownership);
    } else {
        *event = new struct { sycl::event ev; int flag; }();
        (*event)->flag = flags;
        return hapiSuccess;
    }
}

int hapiEventRecord(hapiEvent_t event, hapiStream_t stream) {
    if (event->flag & hapiEventInterprocess) {
        // Level Zero event pool IPC record - implementation needed
        q.submit([&](handler& h) {
            h.host_task([=](interop_handle ih) {
                auto hEvent = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(event->ev);
                auto hCommandList = ih.get_native_resource<sycl::backend::ext_oneapi_level_zero, 
                                        backend_return_t<sycl::backend::ext_oneapi_level_zero, command_list>>();
                zeCommandListAppendSignalEvent(hCommandList, hEvent);
            });
        });
    } else {
        event->ev = stream->ext_oneapi_submit_barrier();
    }
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
    // Level Zero event pool IPC open - implementation needed
    GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
    ze_context_handle_t hCtx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(csv_gpu_manager.ctx);
    zeEventPoolOpenIpcHandle(hCtx, event_handle, &(handle.ze_handle));

    ze_event_desc_t eventDesc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, 0, 0, 0};
    ze_event_handle_t hEvent;
    zeEventCreate(handle.ze_handle, &eventDesc, &hEvent);

    // 3. Wrap Level Zero Event into SYCL Event
    event->ev = sycl::make_event<sycl::backend::ext_oneapi_level_zero>(hEvent, csv_gpu_manager.ctx, keep_ownership);
    return hapiSuccess;
}