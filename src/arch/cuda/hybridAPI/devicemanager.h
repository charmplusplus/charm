#ifndef __DEVICEMANAGER_H_
#define __DEVICEMANAGER_H_

#include <cuda_runtime.h>
#include "converse.h"
#include "buggy.h"

// Manages a GPU device - accessible through GPUManager
struct DeviceManager {
#if CMK_SMP
  // Used in SMP mode, should be locked by the caller
  CmiNodeLock lock;
#endif

  // Device ordinal
  int device;

  // Buddy allocator for communication buffer
  buggy::allocator* comm_buffer;

  DeviceManager(int device_ = 0) : device(device_), comm_buffer(nullptr) {
#if CMK_SMP
    lock = CmiCreateLock();
#endif
  }

  ~DeviceManager() {
#if CMK_SMP
    CmiDestroyLock(lock);
#endif
    destroy_comm_buffer();
  }

  void create_comm_buffer(size_t size) {
    if (comm_buffer == nullptr)
      comm_buffer = new buggy::allocator(size);
  }

  void* alloc_comm_buffer(size_t size) {
    return comm_buffer->malloc(size);
  }

  void free_comm_buffer(void* ptr) {
    comm_buffer->free(ptr);
  }

  void destroy_comm_buffer() {
    if (comm_buffer) {
      delete comm_buffer;
      comm_buffer = nullptr;
    }
  }

};

#endif // __DEVICEMANAGER_H_
