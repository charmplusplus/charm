#ifndef __DEVICEMANAGER_H_
#define __DEVICEMANAGER_H_

#include <cuda_runtime.h>
#include "buggy.h"

// Manages a GPU device - accessible through GPUManager
// With SMP, locks should be acquired before access and released after
struct DeviceManager {

  // Device ordinal
  int device;

  // Buddy allocator for eager communication buffer
  buggy::allocator* eager_comm_buffer;

  // CUDA IPC handle of eager communication buffer
  cudaIpcMemHandle_t eager_ipc_handle;

  DeviceManager(int device_ = 0) : device(device_), eager_comm_buffer(nullptr) {}

  ~DeviceManager() { destroy_eager_comm_buffer(); }

  void create_eager_comm_buffer(size_t size) {
    if (eager_comm_buffer == nullptr)
      eager_comm_buffer = new buggy::allocator(size);
  }

  void destroy_eager_comm_buffer() {
    if (eager_comm_buffer) {
      delete eager_comm_buffer;
      eager_comm_buffer = nullptr;
    }
  }

};

#endif // __DEVICEMANAGER_H_
