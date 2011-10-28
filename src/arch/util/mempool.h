
#ifndef MEMPOOL_H
#define MEMPOOL_H  

#include "conv-config.h"

#if CMK_CONVERSE_GEMINI_UGNI
#include "gni_pub.h"
#include "pmi.h"
typedef gni_mem_handle_t    mem_handle_t;
#else
  // in uGNI, it is memory handler, other versions, this is an integer
  // a unique integer to represent the memory block
typedef size_t    mem_handle_t;
#endif

// multiple mempool for different size allocation
typedef struct mempool_block_t
{
    void                *mempool_ptr;
    mem_handle_t    mem_hndl;
    int                 size;
    size_t              memblock_next;     // offset to next memblock
} mempool_block;


typedef struct mempool_header
{
  int size;
  mem_handle_t  mem_hndl;
  size_t            next_free;
#if CMK_SMP
  void*             pool_addr;
#endif
} mempool_header;

typedef void * (* mempool_newblockfn)(size_t *size, mem_handle_t *mem_hndl, int expand_flag);
typedef void (* mempool_freeblock)(void *ptr, mem_handle_t mem_hndl);

// only at beginning of first block of mempool
typedef struct mempool_type
{
  mempool_block      mempools_head;
  mempool_newblockfn     newblockfn;
  mempool_freeblock      freeblockfn;
  size_t          freelist_head;
  size_t          memblock_tail;
#if CMK_SMP
  CmiNodeLock     mempoolLock;
#endif
} mempool_type;

mempool_type *mempool_init(size_t pool_size, mempool_newblockfn newfn, mempool_freeblock freefn);
void  mempool_destroy(mempool_type *mptr);
void*  mempool_malloc(mempool_type *mptr, int size, int expand);
void mempool_free(mempool_type *mptr, void *ptr_free);
#if CMK_SMP
void mempool_free_thread(void *ptr_free);
#endif
#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__cplusplus)
}
#endif

#endif /* MEMPOOL.H */
