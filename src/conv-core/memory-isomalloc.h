/*Contains declarations used by memory-isomalloc.C to provide
migratable heap allocation to arbitrary clients.
*/
#ifndef CMK_MEMORY_ISOMALLOC_H
#define CMK_MEMORY_ISOMALLOC_H

#include <stddef.h>
/* converse.h rather than conv-config.h: under Reconverse there is no
   conv-config.h, and converse.h is the header that exists in both worlds and
   supplies CthThread. */
/* Angle brackets, not quotes: this file sits next to Charm++'s own
   converse.h, and under Reconverse the converse.h that must win is the one on
   the include path, not the sibling. */
#include <converse.h>
#include "pup_c.h"

#ifdef __cplusplus
#include <vector>
#include <tuple>

extern "C" {
#endif

/****** Isomalloc: Migratable Memory Allocation ********/
int CmiIsomallocEnabled(void);

int CmiIsomallocInRange(void * addr);

#ifndef CMI_ISOMALLOC_CONTEXT_DEFINED
#define CMI_ISOMALLOC_CONTEXT_DEFINED 1
typedef struct CmiIsomallocContext {
  void * opaque;
} CmiIsomallocContext;
#endif

typedef struct CmiIsomallocRegion {
  void * start, * end;
} CmiIsomallocRegion;

/*Build/pup/destroy a context.*/
/* TODO: Some kind of registration scheme so multiple users can coexist.
 * No use case for this currently exists. */
CmiIsomallocContext CmiIsomallocContextCreate(int myunit, int numunits);
void CmiIsomallocContextDelete(CmiIsomallocContext ctx);
void CmiIsomallocContextPup(pup_er p, CmiIsomallocContext * ctxptr);
void CmiIsomallocContextEnableRandomAccess(CmiIsomallocContext ctx);
void CmiIsomallocContextJustMigrated(CmiIsomallocContext ctx);
void CmiIsomallocEnableRDMA(CmiIsomallocContext ctx, int enable); /* on by default */
CmiIsomallocRegion CmiIsomallocContextGetUsedExtent(CmiIsomallocContext ctx);

/*Allocate/free from this context*/
void * CmiIsomallocContextMalloc(CmiIsomallocContext ctx, size_t size);
void * CmiIsomallocContextMallocAlign(CmiIsomallocContext ctx, size_t align, size_t size);
void * CmiIsomallocContextCalloc(CmiIsomallocContext ctx, size_t nelem, size_t size);
void * CmiIsomallocContextRealloc(CmiIsomallocContext ctx, void * ptr, size_t size);
void CmiIsomallocContextFree(CmiIsomallocContext ctx, void * ptr);
size_t CmiIsomallocContextGetLength(CmiIsomallocContext ctx, void * ptr);
void CmiIsomallocContextProtect(CmiIsomallocContext ctx, void * addr, size_t len, int prot);

void * CmiIsomallocContextPermanentAlloc(CmiIsomallocContext ctx, size_t size);
void * CmiIsomallocContextPermanentAllocAlign(CmiIsomallocContext ctx, size_t align, size_t size);

CmiIsomallocContext CmiIsomallocGetThreadContext(CthThread th);

void CmiIsomallocContextEnableRecording(CmiIsomallocContext ctx, int enable); /* internal use only */

/* The job's agreed global address range. A process that joined after the range
   was agreed adopts it rather than using the one it probed for itself; see
   CmiIsomallocAdoptRegion in isomalloc.C. */
void CmiIsomallocGetRegion(CmiUInt8 * start, CmiUInt8 * end);
void CmiIsomallocAdoptRegion(CmiUInt8 start, CmiUInt8 end);
#ifdef __cplusplus
void CmiIsomallocGetRecordedHeap(CmiIsomallocContext ctx,
  std::vector<std::tuple<uintptr_t, size_t, size_t>> & heap_vector);
#endif

/****** Converse Thread functionality that depends on Isomalloc ********/

CthThread CthPup(pup_er, CthThread);

/* Under Reconverse the thread layer owns these two and declares them itself,
   with C++ linkage; redeclaring them here inside extern "C" would conflict. */
#ifndef CMI_MIGRATABLE_THREADS_DECLARED
int CthMigratable(void);
CthThread CthCreateMigratable(CthVoidFn fn, void * arg, int size, CmiIsomallocContext ctx);
#endif

/****** Memory-Isomalloc: malloc wrappers for Isomalloc ********/

/*Allocate non-migratable memory*/
void * malloc_nomigrate(size_t size);
void free_nomigrate(void *mem);

/*Make this context active for malloc interception.*/
void CmiMemoryIsomallocContextActivate(CmiIsomallocContext ctx);

/* Only for internal runtime use, not for Isomalloc users. */
void CmiMemoryIsomallocDisablePush(void);
void CmiMemoryIsomallocDisablePop(void);

#ifdef __cplusplus
}
#endif

#endif

