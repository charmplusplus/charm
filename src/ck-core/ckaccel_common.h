#ifndef __CK_ACCEL_COMMON_H__
#define __CK_ACCEL_COMMON_H__


#define ACCEL_CUDA_TRIGGERED_BUFFER_HEADER_SIZE    (8)

#define ACCEL_CUDA_KERNEL_LEN_INDEX                (0)
#define ACCEL_CUDA_KERNEL_BIT_PATTERN_0_INDEX      (1)
#define ACCEL_CUDA_KERNEL_BIT_PATTERN_1_INDEX      (2)
#define ACCEL_CUDA_KERNEL_ERROR_INDEX              (3)
#define ACCEL_CUDA_KERNEL_NUM_SPLITS               (4)  // 1 for no splits, -1 for unequal per element, or >= 1 for equal for all elements
#define ACCEL_CUDA_KERNEL_SET_SIZE                 (5)
#define ACCEL_CUDA_KERNEL_RESERVED_0               (6)
#define ACCEL_CUDA_KERNEL_RESERVED_1               (7)

#define ACCEL_CUDA_KERNEL_BIT_PATTERN_0   (0xCAFEDEED)
#define ACCEL_CUDA_KERNEL_BIT_PATTERN_1   (0xDEADBEAF)

#define ACCEL_CUDA_ELEMENT_ALIGN                  (16)

extern void markKernelStart();
extern void markKernelEnd();

// These functions return -1 or NULL on failure, a valid pointer or 0 on success
#if CMK_CUDA
extern void* newDeviceBuffer(size_t size);
extern int deleteDeviceBuffer(void *devicePtr);
extern int pushToDevice(void *hostPtr, void* devicePtr, size_t size);
extern int pullFromDevice(void *hostPtr, void* devicePtr, size_t size);
#endif


#endif //__CK_ACCEL_COMMON_H__
