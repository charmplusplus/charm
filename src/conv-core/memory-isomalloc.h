/*Contains declarations used by memory-isomalloc.c to provide
migratable heap allocation to arbitrary clients.
*/
#ifndef CMK_MEMORY_ISOMALLOC_H
#define CMK_MEMORY_ISOMALLOC_H

/*Grab CmiIsomalloc* protoypes, and CmiIsomallocBlock*/
#include <stdlib.h>
#include "converse.h"
#include "pup_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/*Allocate non-migratable memory*/
void *malloc_nomigrate(size_t size);
void free_nomigrate(void *mem);

/*Reentrant versions of memory routines, used inside isomalloc*/
void *malloc_reentrant(size_t size);
void free_reentrant(void *mem);

/*A linked list of allocated migratable memory blocks:*/
struct CmiIsomallocBlockList_tag;
typedef struct CmiIsomallocBlockList_tag CmiIsomallocBlockList;

/*Build a new blockList.*/
CmiIsomallocBlockList *CmiIsomallocBlockListNew(void);

/*Make this blockList active (returns the old blocklist).*/
CmiIsomallocBlockList *CmiIsomallocBlockListActivate(CmiIsomallocBlockList *l);

/*Pup all the blocks in this list.*/
void CmiIsomallocBlockListPup(pup_er p,CmiIsomallocBlockList **l);

/*Delete all the blocks in this list.*/
void CmiIsomallocBlockListFree(CmiIsomallocBlockList *l);

#ifdef __cplusplus
};
#endif

#endif

