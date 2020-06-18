#include "buddy_allocator.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

namespace buddy {
  void allocator::print_status() {
    printf("(buckets)\n");
    size_t free = 0;
    for (int i = 0; i < bucket_count; i++) {
      printf("bucket[%d]: ", i);
      for (const auto& block : buckets[i]) {
        free += block.size;
        printf("{%p, %lu} ", block.ptr, block.size);
      }
      printf("\n");
    }

    printf("(alloc_map)\n");
    size_t allocated = 0;
    size_t used = 0;
    for (const auto& elem : alloc_map) {
      const auto& block = elem.second;
      allocated += block.size;
      used += block.requested;
      printf("ptr: %p, size: %lu, req: %lu\n", elem.first, block.size, block.requested);
    }

    printf("(fragmentation) free: %lu, allocated: %lu, used: %lu\n", free, allocated, used);
  }

  size_t allocator::get_free_size() {
    size_t free = 0;
    for (int i = 0; i < bucket_count; i++) {
      for (const auto& block : buckets[i]) {
        free += block.size;
      }
    }

    return free;
  }

  int allocator::get_bucket(size_t size) {
    return (int)std::ceil(std::log2((double)size)) - 2;
  }

  int allocator::get_block_index(uint8_t* ptr, size_t size) {
    return (size_t)(ptr - base_ptr) / size;
  }

  allocator::allocator(size_t size) : min_size(4), base_ptr(NULL) {
    if (size == 0) {
      fprintf(stderr, "Allocator size has to be larger than 0 bytes\n");
      abort();
    }

    // Request GPU memory (closest power of 2)
    int total_size_log2 = std::ceil(std::log2((double)size));
    total_size = (size_t)std::pow(2, total_size_log2);
    cudaError_t status = cudaMalloc(&base_ptr, total_size);
    if (status != cudaSuccess) {
      fprintf(stderr, "Failed to allocate GPU memory\n");
      abort();
    }
    DEBUG_PRINT("Initialized base_ptr %p with %lu bytes\n", (void*)base_ptr, total_size);

    // Initialize buckets and set up last bucket (for size min_size)
    bucket_count = total_size_log2 - 1;
    buckets = new std::list<FreeBlock>[bucket_count];
    buckets[bucket_count-1].emplace_back(base_ptr, total_size);
  }

  allocator::~allocator() {
    // Free GPU memory
    cudaError_t status = cudaFree(base_ptr);
    if (status != cudaSuccess) {
      fprintf(stderr, "Failed to free GPU memory\n");
      abort();
    }
    delete[] buckets;
  }

  void* allocator::malloc(size_t request) {
    // Cannot satisfy request larger than total size
    if (request > total_size) return nullptr;

    // Has to be larger than minimum allocation size (4 bytes)
    // Size is rounded up to the nearest power of 2
    size_t alloc_size = (request > min_size) ? request : min_size;
    int bucket = get_bucket(alloc_size);
    int original_bucket = bucket;

    // Find an empty bucket
    while (buckets[bucket].empty() && bucket < bucket_count) {
      bucket++;
    }

    // No empty bucket found
    if (bucket == bucket_count) {
      DEBUG_PRINT("No free blocks, malloc request %lu\n", request);
      return nullptr;
    }

    // Found bucket with free block, take it and start splitting if needed
    FreeBlock& block = buckets[bucket].front();
    uint8_t* ptr = block.ptr;
    size_t size = block.size;
    buckets[bucket].pop_front();

    while (bucket-- > original_bucket) {
      buckets[bucket].emplace_back(ptr, size / 2);
      buckets[bucket].emplace_back(ptr + size / 2, size / 2);

      block = buckets[bucket].front();
      ptr = block.ptr;
      size = block.size;
      buckets[bucket].pop_front();
    }

    // Store allocation info
    alloc_map.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(ptr),
        std::forward_as_tuple(size, request));

    DEBUG_PRINT("Allocated ptr %p (base_ptr + %lu) with %lu bytes, requested was %lu bytes\n",
        (void*)ptr, (size_t)(ptr - base_ptr), size, request);

#if BUDDY_DEBUG
    print_status();
#endif

    return ptr;
  }

  void allocator::free(void* ptr) {
    // Find pointer in allocation map
    auto alloc_it = alloc_map.find((uint8_t*)ptr);
    if (alloc_it == alloc_map.end()) {
      fprintf(stderr, "Free invalid pointer: %p\n", ptr);
      std::abort();
    }

    const auto& alloc_block = alloc_it->second;
    size_t size = alloc_block.size;
    size_t requested = alloc_block.requested;
    int bucket = get_bucket(size);

    // Add to free list
    buckets[bucket].emplace_back((uint8_t*)ptr, size);

    // Remove entry from allocation map
    alloc_map.erase(alloc_it);

    // Recursively merge free blocks
    uint8_t* merge_ptr = (uint8_t*)ptr;
    size_t merge_size = size;
    for (int i = bucket; i < bucket_count; i++) {

      // Find buddy of current block
      // If block index is even, it is on the left side of its buddy and vice versa
      int block_index = get_block_index(merge_ptr, merge_size);
      bool block_index_even = (block_index % 2 == 0);
      uint8_t* buddy_ptr = block_index_even ? (merge_ptr + merge_size) : (merge_ptr - merge_size);

      // If buddy is also free, merge
      for (std::list<FreeBlock>::iterator it = buckets[i].begin(); it != buckets[i].end(); it++) {
        const auto& block = *it;
        if (block.ptr == buddy_ptr) {
          buckets[i+1].emplace_back(block_index_even ? merge_ptr : buddy_ptr, 2 * merge_size);
          buckets[i].erase(it); // Iterator is invalid after this erase
          buckets[i].pop_back();
          break;
        }
        else {
          // Did not find free buddy block, stop merging
          goto merge_done;
        }
      }

      if (!block_index_even) merge_ptr = buddy_ptr;
      merge_size *= 2;
    }

merge_done:
    DEBUG_PRINT("Freed ptr %p with %lu bytes, requested was %lu bytes\n", ptr, size, requested);

#if BUDDY_DEBUG
    print_status();
#endif
  }
}
