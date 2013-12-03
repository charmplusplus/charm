
#ifndef MEMPOOL_H
#define MEMPOOL_H  

#include "conv-config.h"
#include "converse.h"

#if CMK_CONVERSE_UGNI
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

//given x as mptr get
#define   MEMPOOL_GetBlockHead(x)   (block_header*)&(x->block_head)	
//given x as block header, get ...
#define   MEMPOOL_GetBlockSize(x)    (((block_header*)x)->size)
#define   MEMPOOL_GetBlockMemHndl(x) (((block_header*)x)->mem_hndl)
#define   MEMPOOL_GetBlockNext(x)    (((block_header*)x)->block_next)     
//given x as user pointer, get mempool_header/slot_header
#define   MEMPOOL_GetMempoolHeader(x,align) \
                                  ((mempool_header*)((char*)(x)-align))
//given x as mempool_header/slot_header, get ...
#define   MEMPOOL_GetBlockPtr(x)    ((block_header*)(x->block_ptr))
#define   MEMPOOL_GetMempoolPtr(x)  ((mempool_type*)(MEMPOOL_GetBlockPtr(x)->mptr))
#define   MEMPOOL_GetSize(x)      (MEMPOOL_GetBlockPtr(x)->size)
#define   MEMPOOL_GetMemHndl(x)   (MEMPOOL_GetBlockPtr(x)->mem_hndl)
#define   MEMPOOL_GetMsgInRecv(x) (MEMPOOL_GetBlockPtr(x)->msgs_in_recv)
#define   MEMPOOL_GetMsgInSend(x) (MEMPOOL_GetBlockPtr(x)->msgs_in_send)
#define   MEMPOOL_IncMsgInRecv(x) (MEMPOOL_GetBlockPtr(x)->msgs_in_recv)++
#define   MEMPOOL_DecMsgInRecv(x) (MEMPOOL_GetBlockPtr(x)->msgs_in_recv)--
#define   MEMPOOL_IncMsgInSend(x) (MEMPOOL_GetBlockPtr(x)->msgs_in_send)++
#define   MEMPOOL_DecMsgInSend(x) (MEMPOOL_GetBlockPtr(x)->msgs_in_send)--
#define   MEMPOOL_GetSlotGNext(x)     (x->gnext)
#define   MEMPOOL_GetSlotStatus(x)    (x->status)
#define	  MEMPOOL_GetSlotSize(x)      (cutOffPoints[x->size])
struct block_header;
struct mempool_type;

//header of an free slot
typedef struct slot_header_
{
  struct block_header  *block_ptr;     // block_header
  int          		size,status;  //status is 1 for free, 0 for used
  size_t      		gprev,gnext;  //global slot list within a block
  size_t      		prev,next;    //link list for freelists slots
#if ! CMK_64BIT
  size_t                padding;      // fix for 32 bit machines
#endif
} slot_header;

typedef struct used_header_
{
  struct block_header  *block_ptr;     // block_header
  int         		size,status;  //status is 1 for free, 0 for used
  size_t      		gprev,gnext;  //global slot list within a block
#if ! CMK_64BIT
  size_t                padding;      // fix for 32 bit machines
#endif
} used_header;

typedef used_header mempool_header;

// multiple mempool for different size allocation
// make sure this is 16 byte aligned
typedef struct block_header
{
  mem_handle_t        mem_hndl;
  size_t              size, used;
  size_t              block_prev,block_next;   // offset to next memblock
  size_t              freelists[cutOffNum];
  struct mempool_type  *mptr;               // mempool_type
#if CMK_CONVERSE_UGNI
  int                 msgs_in_send;
  int                 msgs_in_recv;
#endif
  size_t              padding;
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
#if CMK_USE_MEMPOOL_ISOMALLOC || (CMK_SMP && CMK_CONVERSE_UGNI)
  CmiNodeLock		 mempoolLock;
#endif
} mempool_type;

#ifdef __cplusplus
extern "C" {
#endif

mempool_type *mempool_init(size_t pool_size, mempool_newblockfn newfn, mempool_freeblock freefn, size_t limit);
void  mempool_destroy(mempool_type *mptr);
void*  mempool_malloc(mempool_type *mptr, int size, int expand);
void mempool_free(mempool_type *mptr, void *ptr_free);
#if CMK_USE_MEMPOOL_ISOMALLOC || (CMK_SMP && CMK_CONVERSE_UGNI)
void mempool_free_thread(void *ptr_free);
#endif

#if defined(__cplusplus)
}
#endif

#if CMK_CONVERSE_UGNI
void* getNextRegisteredPool();
#endif

#endif /* MEMPOOL.H */
