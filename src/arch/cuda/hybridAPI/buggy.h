#pragma once

#include <cstdio>
#include <cstdlib>
#include <list>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <mutex>

namespace buggy {

#ifdef DEBUG
#define DEBUG_PRINT(...) printf("[Buggy] " __VA_ARGS__)
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

    // Allocation limits
    size_t limit;
    const size_t min_alloc;

    // Base pointer of the initial allocation
    uint8_t* base_ptr;

    // Mutex for thread-safe access
    std::mutex mutex;

    // Buckets each with a free list
    std::list<FreeBlock>* buckets;
    int bucket_count;

    // Map of allocated blocks
    std::unordered_map<uint8_t*, AllocBlock> alloc_map;

    /* --------------------- */
    /* | Utility functions | */
    /* --------------------- */

    void print_status() {
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

    int get_bucket(size_t size) {
      return (int)std::ceil(std::log2((double)size)) - 2;
    }

    int get_block_index(uint8_t* ptr, size_t size) {
      return (size_t)(ptr - base_ptr) / size;
    }

    /* ------------------------ */
    /* | Allocation functions | */
    /* ------------------------ */

    allocator(size_t size = 1 << 26) : min_alloc(4), base_ptr(NULL) {
      // Request GPU memory (closest power of 2)
      int limit_log2 = std::ceil(std::log2((double)size));
      limit = (size_t)std::pow(2, limit_log2);
      cudaMalloc(&base_ptr, limit);
      DEBUG_PRINT("Initialized base_ptr %p with %lu bytes\n", (void*)base_ptr, limit);

      // Initialize buckets and set up last bucket (for size min_alloc)
      bucket_count = limit_log2 - 1;
      buckets = new std::list<FreeBlock>[bucket_count];
      buckets[bucket_count-1].emplace_back(base_ptr, limit);
    }

    ~allocator() {
      // Free GPU memory
      cudaFree(base_ptr);
      delete[] buckets;
    }

    void* malloc(size_t request) {
      const std::lock_guard<std::mutex> lock(mutex);

      // Cannot satisfy request larger than limit
      if (request > limit) return nullptr;

      // Has to be larger than minimum allocation size (4 bytes)
      // Size is rounded up to the nearest power of 2
      size_t alloc_size = (request > min_alloc) ? request : min_alloc;
      int bucket = get_bucket(alloc_size);
      int original_bucket = bucket;

      // Find an empty bucket
      while (buckets[bucket].empty() && bucket < bucket_count) {
        bucket++;
      }

      // All buckets were empty!
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

#ifdef DEBUG
      print_status();
#endif

      return ptr;
    }

    void free(void* ptr) {
      const std::lock_guard<std::mutex> lock(mutex);

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

#ifdef DEBUG
      print_status();
#endif
    }
  }; // struct allocator

} // namespace buggy
