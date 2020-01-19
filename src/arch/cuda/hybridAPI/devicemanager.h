#ifndef __DEVICEMANAGER_H_
#define __DEVICEMANAGER_H_

// Manages a GPU device - accessible through GPUManager.
class DeviceManager {
  int device;

public:
  DeviceManager(int device_ = 0) : device(device_) {}
  ~DeviceManager() {}
};

#endif // __DEVICEMANAGER_H_
