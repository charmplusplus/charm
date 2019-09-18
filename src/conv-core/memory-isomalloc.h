/*Contains declarations used by memory-isomalloc.C to provide
migratable heap allocation to arbitrary clients.
*/
#ifndef CMK_MEMORY_ISOMALLOC_H
#define CMK_MEMORY_ISOMALLOC_H

#include <stddef.h>
#include "conv-config.h"

#ifdef __cplusplus
extern "C" {
#endif

/****** Isomalloc: Migratable Memory Allocation ********/
int CmiIsomallocEnabled(void);

int CmiIsomallocInRange(void * addr);

struct CmiIsomallocContext;
typedef struct CmiIsomallocContext CmiIsomallocContext;

/*Build/pup/destroy a context.*/
/* TODO: Some kind of registration scheme so multiple users can coexist.
 * No use case for this currently exists. */
CmiIsomallocContext * CmiIsomallocContextCreate(int myunit, int numunits);
void CmiIsomallocContextDelete(CmiIsomallocContext * pool);
void CmiIsomallocContextPup(pup_er p, CmiIsomallocContext ** pPool);

/*Allocate/free from this context*/
void * CmiIsomallocContextMalloc(CmiIsomallocContext * pool, size_t nBytes);
void * CmiIsomallocContextMallocAlign(CmiIsomallocContext * pool, size_t align, size_t nBytes);
void CmiIsomallocContextFree(CmiIsomallocContext * pool, void * ptr);
size_t CmiIsomallocContextGetLength(CmiIsomallocContext * pool, void * ptr);

CmiIsomallocContext * CmiIsomallocGetThreadContext(CthThread th);

/****** Converse Thread functionality that depends on Isomalloc ********/

int CthMigratable(void);
CthThread CthPup(pup_er, CthThread);
CthThread CthCreateMigratable(CthVoidFn fn, void * arg, int size, CmiIsomallocContext * ctx);

/****** Memory-Isomalloc: malloc wrappers for Isomalloc ********/

/*Allocate non-migratable memory*/
void * malloc_nomigrate(size_t size);
void free_nomigrate(void *mem);

/*Make this context active for malloc interception.*/
void CmiMemoryIsomallocContextActivate(CmiIsomallocContext * pool);

/* Only for internal runtime use, not for Isomalloc users. */
void CmiMemoryIsomallocDisablePush(void);
void CmiMemoryIsomallocDisablePop(void);

#ifdef __cplusplus
}
#endif

#endif

