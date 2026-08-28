#ifndef __DEVICEMANAGER_H_
#define __DEVICEMANAGER_H_

#include <hapi_portable.h>
#include "converse.h"
#include "buddy_allocator.h"

#ifdef CMK_LBDB_ON
#include "GpuScalingModel.h"
#endif

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

  // Dedicated stream for migration pack copies (PUP::toMem's device branch).
  // Created lazily, with default (blocking) flags on purpose: a blocking
  // stream keeps legacy ordering against null-stream work, while moving the
  // pack copies off the null stream stops them barriering the application's
  // streams the way null-stream memcpys do. Callers serialize creation with
  // this manager's lock.
  hapiStream_t migration_stream;

#ifdef CMK_LBDB_ON
  // Device properties needed to estimate how many SMs a kernel occupies.
  // Filled lazily by hapiPopulateDeviceProps once device_managers is ready.
  int multi_processor_count;
  int max_threads_per_sm;
  int max_blocks_per_sm;
  int max_registers_per_sm;
  int max_shared_mem_per_sm;
  int warp_size;
  bool props_initialized;
  GpuDeviceDescriptor descriptor;
#endif

  DeviceManager(int local_index_, int global_index_) :
    local_index(local_index_), global_index(global_index_), comm_buffer(nullptr),
    migration_stream(NULL)
#ifdef CMK_LBDB_ON
    , multi_processor_count(0), max_threads_per_sm(0), max_blocks_per_sm(0),
      max_registers_per_sm(0), max_shared_mem_per_sm(0), warp_size(0),
      props_initialized(false)
#endif
  {
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

  buddy::allocator* get_comm_buffer() {
    return comm_buffer;
  }

  void create_comm_buffer(size_t total_size, size_t comm_size) {
    if (comm_buffer == nullptr)
      comm_buffer = new buddy::allocator(total_size, comm_size);
  }

  void* alloc_comm_buffer(size_t size, bool is_comm = true) {
    return comm_buffer->malloc(size, is_comm);
  }

  void free_comm_buffer(size_t offset) {
    comm_buffer->free((void*)(comm_buffer->base_ptr + offset));
  }

  size_t get_comm_buffer_free_size() {
    return comm_buffer->get_free_size();
  }

  size_t get_lb_buffer_free_size() {
    return comm_buffer->get_lb_free_size();
  }

  void destroy_comm_buffer() {
    if (comm_buffer) {
      delete comm_buffer;
      comm_buffer = nullptr;
    }
  }
};

#endif // __DEVICEMANAGER_H_
