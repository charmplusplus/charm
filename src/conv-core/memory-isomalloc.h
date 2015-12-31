/*Contains declarations used by memory-isomalloc.c to provide
migratable heap allocation to arbitrary clients.
*/
#ifndef CMK_MEMORY_ISOMALLOC_H
#define CMK_MEMORY_ISOMALLOC_H

/*Grab CmiIsomalloc* protoypes, and CmiIsomallocBlock*/
#include <stddef.h>
#include "conv-config.h"

#ifdef __cplusplus
extern "C" {
#endif

/****** Isomalloc: Migratable Memory Allocation ********/
/*Simple block-by-block interface:*/
void  CmiIsomallocPup(pup_er p,void **block);
void  CmiIsomallocFree(void *block);
int   CmiIsomallocEnabled();
void  CmiEnableIsomalloc();
void  CmiDisableIsomalloc();

CmiInt8   CmiIsomallocLength(void *block);
int   CmiIsomallocInRange(void *addr);

#if CMK_USE_MEMPOOL_ISOMALLOC
struct mempool_type;
#endif

/*List-of-blocks interface:*/
struct CmiIsomallocBlockList {/*Circular doubly-linked list of blocks:*/
  struct CmiIsomallocBlockList *prev, *next;
#if CMK_USE_MEMPOOL_ISOMALLOC
  struct mempool_type *pool;
#endif
  /*actual data of block follows here...*/
};
typedef struct CmiIsomallocBlockList CmiIsomallocBlockList;

/*Build/pup/destroy an entire blockList.*/
CmiIsomallocBlockList *CmiIsomallocBlockListNew();
void CmiIsomallocBlockListPup(pup_er p, CmiIsomallocBlockList **l);
void CmiIsomallocBlockListDelete(CmiIsomallocBlockList *l);

/*Allocate/free a block from this blockList*/
void *CmiIsomallocBlockListMalloc(CmiIsomallocBlockList *l,size_t nBytes);
void *CmiIsomallocBlockListMallocAlign(CmiIsomallocBlockList *l,size_t align,size_t nBytes);
void CmiIsomallocBlockListFree(void *doomedMallocedBlock);

/* Allocate a block from the blockList associated with a thread */
void *CmiIsomallocMallocForThread(CthThread th, size_t nBytes);
void *CmiIsomallocMallocAlignForThread(CthThread th, size_t align, size_t nBytes);

/*Allocate non-migratable memory*/
void *malloc_nomigrate(size_t size);
void free_nomigrate(void *mem);

/*Reentrant versions of memory routines, used inside isomalloc*/
void *malloc_reentrant(size_t size);
void free_reentrant(void *mem);

/*Make this blockList active (returns the old blocklist).*/
CmiIsomallocBlockList *CmiIsomallocBlockListActivate(CmiIsomallocBlockList *l);
CmiIsomallocBlockList *CmiIsomallocBlockListCurrent();

#ifdef __cplusplus
}
#endif

#endif

