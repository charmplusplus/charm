#ifndef __BUDDY_ALLOCATOR_H_
#define __BUDDY_ALLOCATOR_H_

#include <cstdint>
#include <list>
#include <unordered_map>

// A cached memory allocator with GPU memory as the backing store.
// A fixed size allocation is initially made to the backing store,
// and the allocator hands out a block of memory from it when the user
// requests an allocation. This mechanism is currently used to support
// one-time creation and sharing of CUDA IPC handles for device-to-device
// data transfers. It also uses the buddy allocation algorithm to
// minimize external fragmentation that could occur from concurrent
// allocation/deallocation requests (hence its name).
namespace buddy {

#define BUDDY_DEBUG 0

#if BUDDY_DEBUG
#define DEBUG_PRINT(...) printf("Buddy> " __VA_ARGS__)
#else
#define DEBUG_PRINT(...) do {} while (0)
#endif

  struct allocator {
    // Free block
    struct FreeBlock {
      uint8_t* ptr;
      size_t size;

      FreeBlock() : ptr(nullptr), size(0) {}

      FreeBlock(uint8_t* ptr_, size_t size_) : ptr(ptr_), size(size_) {}
    };

    // Allocated block - separately stores requested and allocated memory
    // to track (internal) fragmentation. Does not contain the pointer as it
    // is stored as a key in the unordered map.
    struct AllocBlock {
      size_t size;
      size_t requested;

      AllocBlock() : size(0), requested(0) {}

      AllocBlock(size_t size_, size_t requested_) : size(size_), requested(requested_) {}
    };

    // Allocation size limits
    size_t total_size;
    const size_t min_size;

    // Base pointer of the initial allocation
    uint8_t* base_ptr;

    // Buckets each with a free list
    std::list<FreeBlock>* buckets;
    int bucket_count;

    // Map of allocated blocks
    std::unordered_map<uint8_t*, AllocBlock> alloc_map;

    // Utility functions
    void print_status();
    size_t get_free_size();
    int get_bucket(size_t size);
    int get_block_index(uint8_t* ptr, size_t size);

    // Allocation functions
    allocator(size_t size);
    ~allocator();
    void* malloc(size_t request);
    void free(void* ptr);
  };
}

#endif // __BUDDY_ALLOCATOR_H_
