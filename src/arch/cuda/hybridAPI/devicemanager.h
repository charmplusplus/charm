#ifndef __DEVICEMANAGER_H_
#define __DEVICEMANAGER_H_

#include <cuda_runtime.h>
#include "converse.h"
#include "buddy_allocator.h"

// Manages a GPU device, accessible through GPUManager
struct DeviceManager {
#if CMK_SMP
  // Used in SMP mode, should be locked by the caller
  CmiNodeLock lock;
#endif

  // Device ordinals
  int local_index; // Within process
  int global_index; // Within physical node

  // Buddy allocator for communication buffer
  buddy::allocator* comm_buffer;

  DeviceManager(int local_index_, int global_index_) :
    local_index(local_index_), global_index(global_index_), comm_buffer(nullptr) {
#if CMK_SMP
    lock = CmiCreateLock();
#endif
  }

  void destroy() {
#if CMK_SMP
    CmiDestroyLock(lock);
#endif
    destroy_comm_buffer();
  }

  void create_comm_buffer(size_t size) {
    if (comm_buffer == nullptr)
      comm_buffer = new buddy::allocator(size);
  }

  void* alloc_comm_buffer(size_t size) {
    return comm_buffer->malloc(size);
  }

  void free_comm_buffer(size_t offset) {
    comm_buffer->free((void*)(comm_buffer->base_ptr + offset));
  }

  size_t comm_buffer_free_size() {
    return comm_buffer->get_free_size();
  }

  void destroy_comm_buffer() {
    if (comm_buffer) {
      delete comm_buffer;
      comm_buffer = nullptr;
    }
  }
};

#endif // __DEVICEMANAGER_H_
