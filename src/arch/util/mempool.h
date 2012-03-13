
#ifndef MEMPOOL_H
#define MEMPOOL_H  

#include "conv-config.h"
#include "converse.h"

#if CMK_CONVERSE_GEMINI_UGNI
#include "gni_pub.h"
#include "pmi.h"
typedef gni_mem_handle_t    mem_handle_t;
#else
// in uGNI, it is memory handler, other versions, this is an integer
// a unique integer to represent the memory block
typedef CmiInt8   mem_handle_t;
#endif

typedef void * (* mempool_newblockfn)(size_t *size, mem_handle_t *mem_hndl, int expand_flag);
typedef void (* mempool_freeblock)(void *ptr, mem_handle_t mem_hndl);

#define cutOffNum 25 

struct block_header;
struct mempool_type;

//header of an free slot
typedef struct slot_header_
{
  struct block_header  *block_ptr;     // block_header
  int          		size,status;  //status is 1 for free, 0 for used
  size_t      		gprev,gnext;  //global slot list within a block
  size_t      		prev,next;    //link list for freelists slots
  size_t                padding;      // fix for 32 bit machines
} slot_header;

typedef struct used_header_
{
  struct block_header  *block_ptr;     // block_header
  int         		size,status;  //status is 1 for free, 0 for used
  size_t      		gprev,gnext;  //global slot list within a block
  size_t                padding;      // fix for 32 bit machines
} used_header;

typedef used_header mempool_header;

// multiple mempool for different size allocation
typedef struct block_header
{
  mem_handle_t        mem_hndl;
  size_t              size, used;
  size_t              block_prev,block_next;   // offset to next memblock
  size_t              freelists[cutOffNum];
  struct mempool_type  *mptr;               // mempool_type
  size_t              padding;              // fix for 32 bit machines
#if CMK_CONVERSE_GEMINI_UGNI
  int                 msgs_in_send;
  int                 msgs_in_recv;
#endif
} block_header;

// only at beginning of first block of mempool, representing the mempool
typedef struct mempool_type
{
  block_header           block_head;
  mempool_newblockfn     newblockfn;
  mempool_freeblock      freeblockfn;
  size_t                 block_tail;
  size_t                 limit;
  size_t                 size;
#if CMK_USE_MEMPOOL_ISOMALLOC || (CMK_SMP && CMK_CONVERSE_GEMINI_UGNI)
  CmiNodeLock		 mempoolLock;
#endif
} mempool_type;

mempool_type *mempool_init(size_t pool_size, mempool_newblockfn newfn, mempool_freeblock freefn, size_t limit);
void  mempool_destroy(mempool_type *mptr);
void*  mempool_malloc(mempool_type *mptr, int size, int expand);
void mempool_free(mempool_type *mptr, void *ptr_free);
#if CMK_USE_MEMPOOL_ISOMALLOC || (CMK_SMP && CMK_CONVERSE_GEMINI_UGNI)
void mempool_free_thread(void *ptr_free);
#endif

#if CMK_CONVERSE_GEMINI_UGNI
void* getNextRegisteredPool();
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__cplusplus)
}
#endif

#endif /* MEMPOOL.H */
