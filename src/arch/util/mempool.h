
#ifndef MEMPOOL_H
#define MEMPOOL_H  

#include "conv-config.h"

#if CMK_CONVERSE_GEMINI_UGNI

#include "gni_pub.h"
#include "pmi.h"
extern gni_nic_handle_t      nic_hndl;
#if CMK_ERROR_CHECKING
#define GNI_RC_CHECK(msg,rc) do { if(rc != GNI_RC_SUCCESS) {           CmiPrintf("[%d] %s; err=%s\n",CmiMyPe(),msg,gni_err_str[rc]); CmiAbort("GNI_RC_CHECK"); } } while(0)
#else
#define GNI_RC_CHECK(msg,rc)
#endif

#else
  // in uGNI, it is memory handler, other versions, this is an integer
  // a unique integer to represent the memory block
typedef int    gni_mem_handle_t;
#endif

// multiple mempool for different size allocation
typedef struct mempool_block_t
{
    void                *mempool_ptr;
    struct              mempool_block_t *next;
    gni_mem_handle_t    mem_hndl;
} mempool_block;


typedef struct mempool_header
{
  int size;
  gni_mem_handle_t  mem_hndl;
  size_t            next_free;
} mempool_header;


// only at beginning of first block of mempool
typedef struct mempool_type
{
  mempool_block   mempools_head;
  size_t          freelist_head;
} mempool_type;


mempool_type *init_mempool(void *pool, size_t pool_size, gni_mem_handle_t mem_hndl);
void kill_allmempool(mempool_type *mptr);
void*  mempool_malloc(mempool_type *mptr, int size, int expand);
void mempool_free(mempool_type *mptr, void *ptr_free);

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__cplusplus)
}
#endif

#endif /* MEMPOOL.H */
