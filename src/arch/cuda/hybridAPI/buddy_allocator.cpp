#include "buddy_allocator.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <hapi_portable.h>

namespace buddy {
  void allocator::print_status() {
    printf("(buckets)\n");
    size_t free = 0;
    for (int i = 0; i < bucket_count; i++) {
      printf("bucket[%d]: ", i);
      for (const auto& block : buckets[i]) {
        free += block.size;
        printf("{%p, %zu} ", block.ptr, block.size);
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
      printf("ptr: %p, size: %zu, req: %zu\n", elem.first, block.size, block.requested);
    }

    printf("(fragmentation) free: %zu, allocated: %zu, used: %zu\n", free, allocated, used);
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

  size_t allocator::get_lb_free_size() {
    // No LB region: head/tail are null, so there is no free list to walk.
    if (lb_size == 0) return 0;

    size_t free = 0;
    lb_free_list* tmp = head->next;
    while(tmp != tail) {
      free += tmp->size;
      tmp = tmp->next;
    }

    free += lb_size - (size_t)(lb_ptr - lb_base_ptr);
    return free;
  }

  int allocator::get_bucket(size_t size) {
    return (int)std::ceil(std::log2((double)size)) - 2;
  }

  int allocator::get_block_index(uint8_t* ptr, size_t size) {
    return (size_t)(ptr - base_ptr) / size;
  }

  allocator::allocator(size_t _comm_lb_size, size_t _comm_size) : min_size(4), base_ptr(NULL) {
    if (_comm_size == 0) {
      fprintf(stderr, "Allocator size has to be larger than 0 bytes\n");
      abort();
    }

    // Request GPU memory (closest power of 2)
    hapiError_t status = hapiMalloc(&base_ptr, _comm_lb_size);
    this->comm_size = _comm_size;
    if (status != hapiSuccess) {
      fprintf(stderr, "Failed to allocate GPU memory\n");
      abort();
    }
    DEBUG_PRINT("Initialized base_ptr %p with %zu bytes\n", (void*)base_ptr, total_size);

    // Initialize buckets and set up last bucket (for size min_size)
    int total_size_log2 = std::ceil(std::log2((double)_comm_size));
    bucket_count = total_size_log2 - 1;
    buckets = new std::list<FreeBlock>[bucket_count];
    buckets[bucket_count-1].emplace_back(base_ptr, _comm_size);

    // Initialize vars for load balancing.
    //
    // These must be set even when no LB region was requested: free() routes a
    // pointer to the LB path purely by comparing it against lb_base_ptr, and
    // the runtime's default configuration is exactly the no-LB one
    // (GPUManager::lb_buffer_size is 0 unless +gpulbbuffer is passed, so
    // create_comm_buffer() is called with total == comm). Leaving lb_base_ptr
    // uninitialized there made every comm-buffer free() compare against
    // garbage and, about half the time, abort in the LB branch with
    // "got a request to free buffer of size 0".
    //
    // Anchoring the empty LB region at the end of the comm region makes the
    // comparison false for every comm allocation, which is the correct
    // routing.
    lb_size = 0;
    lb_base_ptr = base_ptr + _comm_size;
    lb_ptr = lb_base_ptr;
    head = nullptr;
    tail = nullptr;

    if(_comm_lb_size != _comm_size) {
      lb_free_pool.resize(128);

      head = &(lb_free_pool[0]);
      lb_free_pool_taken[0] = true;

      tail = &(lb_free_pool[1]);
      lb_free_pool_taken[1] = true;

      head->next = tail;
      head->prev = nullptr;
      tail->next = nullptr;
      tail->prev = head;

      this->lb_size = _comm_lb_size - _comm_size;
      this->lb_base_ptr = base_ptr + _comm_size;
      this->lb_ptr = lb_base_ptr;
    }

    this->total_size = _comm_lb_size;
  }

  allocator::~allocator() {
    // Free GPU memory
    hapiError_t status = hapiFree(base_ptr);
    if (status != hapiSuccess) {
      fprintf(stderr, "Failed to free GPU memory\n");
      abort();
    }
    delete[] buckets;
  }

  void* allocator::malloc(size_t request, bool is_comm) {
    if(!is_comm) {
      // buffers for load balancing
      if(request > lb_size) return nullptr;
      lb_free_list* tmp = head->next;

      // see if existing free block can service request
      while (tmp != tail) {
        if(tmp->size == request) {
          tmp->prev->next = tmp->next;
          tmp->next->prev = tmp->prev;
          void* ptr = tmp->ptr;
          lb_ptr_size[ptr] = request;

          tmp->next = nullptr;
          tmp->prev = nullptr;
          tmp->size = 0;
          lb_free_pool_taken[tmp->indx] = false;
          tmp->ptr = nullptr;

          return ptr;
        } else if(tmp->size > request) {
          void* ptr = tmp->ptr;
          lb_ptr_size[ptr] = request;

          tmp->size = tmp->size - request;
          tmp->ptr = tmp->ptr + request;

          return ptr;
        }

        tmp = tmp->next;
      }

      // service the request from the tip of lb_ptr
      if(lb_ptr + request > lb_base_ptr + lb_size) return nullptr;
      void* ptr = lb_ptr;
      lb_ptr_size[ptr] = request;
      lb_ptr = lb_ptr + request;
      return ptr;
    }

    DEBUG_PRINT("REQUEST: %ld, TOTAL_SIZE: %ld", request, total_size);
    // Cannot satisfy request larger than total size
    if (request > comm_size) return nullptr;

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
      DEBUG_PRINT("No free blocks, malloc request %zu\n", request);
      return nullptr;
    }

    // Found bucket with free block, take it and start splitting if needed
    FreeBlock block = buckets[bucket].front();
    uint8_t* ptr = block.ptr;
    size_t size = block.size;
    buckets[bucket].pop_front();

    while (bucket-- > original_bucket) {
      size /= 2;
      buckets[bucket].emplace_back(ptr + size, size);
    }

    // Store allocation info
    alloc_map.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(ptr),
        std::forward_as_tuple(size, request));


    return ptr;
  }

  void allocator::free(void* ptr) {
    if((uint8_t*)ptr >= lb_base_ptr) {
      size_t alloc_size = lb_ptr_size[ptr];
      if(alloc_size == 0) {
        printf("Load balancing allocator got a request to free buffer of size 0\n");
        fflush(stdout);
        std::abort();
      }

      // see if the ptr is just before lb_ptr
      if((uint8_t*)ptr + alloc_size == lb_ptr) {
        lb_ptr -= alloc_size;
        lb_ptr_size[ptr] = 0;
        return;
      }

      // see if mergeable with any existing free block
      lb_free_list* tmp = head->next;
      bool merged = false;
      while(tmp != tail) {
        if (((uint8_t*)tmp->ptr + tmp->size) == (uint8_t*)ptr) {
          tmp->size = tmp->size + alloc_size;
          lb_free_list* tmp_next = tmp->next;

          if (tmp_next != tail && (((uint8_t*)tmp->ptr + tmp->size) == (uint8_t*)tmp_next->ptr)) {
            tmp->size = tmp->size + tmp_next->size;

            tmp_next->prev->next = tmp_next->next;
            tmp_next->next->prev = tmp_next->prev;
            
            tmp_next->next = nullptr;
            tmp_next->prev = nullptr;
            tmp_next->size = 0;
            lb_free_pool_taken[tmp_next->indx] = false;
            tmp_next->ptr = nullptr;
          }
          
          merged = true;
          break;
        } else if(((uint8_t*)ptr + alloc_size) == (uint8_t*)tmp->ptr) {
          tmp->size = tmp->size + alloc_size;
          tmp->ptr = ptr;

          merged = true;
          break;
        } else if (tmp->ptr > ptr) {
          break;
        }

        tmp = tmp->next;
      }

      // see if merging reached the lb_ptr
      if(merged) {
        if((uint8_t*)tmp->ptr + tmp->size == lb_ptr) {
            lb_ptr -= tmp->size;

            tmp->prev->next = tmp->next;
            tmp->next->prev = tmp->prev;
            
            tmp->next = nullptr;
            tmp->prev = nullptr;
            tmp->size = 0;
            lb_free_pool_taken[tmp->indx] = false;
            tmp->ptr = nullptr;

          }

          lb_ptr_size[ptr] = 0;
          return;
      }

      // add a free node just before tmp
      size_t free_space_indx = 2;
      while(lb_free_pool_taken[free_space_indx] && free_space_indx < lb_free_pool.size())
        free_space_indx++;
      lb_free_list* free_node;
      if(free_space_indx == lb_free_pool.size()) {
        // TODO : Implement this logic or just increase the default size of 
        // lb_free_pool
        printf("Load balancing allocator does not have any more free nodes\n");
        fflush(stdout);
        std::abort();
      } else {
        free_node = &(lb_free_pool[free_space_indx]);
        lb_free_pool_taken[free_space_indx] = true;
      }

      free_node->indx = free_space_indx;
      free_node->ptr = ptr;
      free_node->size = lb_ptr_size[ptr];
      free_node->prev = tmp->prev;
      free_node->next = tmp;

      tmp->prev->next = free_node;
      tmp->prev = free_node;

      lb_ptr_size[ptr] = 0;
      return;
    }
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
      bool merged = false;
      for (std::list<FreeBlock>::iterator it = buckets[i].begin(); it != buckets[i].end(); it++) {
        if (it->ptr == buddy_ptr) {
          buckets[i+1].emplace_back(block_index_even ? merge_ptr : buddy_ptr, 2 * merge_size);
          buckets[i].erase(it);
          buckets[i].pop_back();
          merged = true;
          break;
        }
      }

      if (!merged) break;

      if (!block_index_even) merge_ptr = buddy_ptr;
      merge_size *= 2;
    }
  }
}
