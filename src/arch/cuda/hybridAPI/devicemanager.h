#ifndef __DEVICEMANAGER_H_
#define __DEVICEMANAGER_H_

#include "buggy.h"

// Manages a GPU device - accessible through GPUManager.
struct DeviceManager {
  int device;

  // Buddy allocator for eager communication buffer
  buggy::allocator* eager_comm_buffer;

  DeviceManager(int device_ = 0) : device(device_) {}

  void create_eager_comm_buffer(size_t size) {
    eager_comm_buffer = new buggy::allocator(size);
  }

  void destroy_eager_comm_buffer() {
    delete eager_comm_buffer;
  }
};

#endif // __DEVICEMANAGER_H_
