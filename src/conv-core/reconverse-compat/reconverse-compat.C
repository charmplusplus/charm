/*
  Definitions for the classic Converse thread-migration API on reconverse
  builds. See reconverse-compat.h for what each one does and why. Linked
  into libmoduletcharm, so every TCharm user (AMPI, collide, mblock, ...)
  gets them.
*/
#include "conv-config.h"

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#if CMK_USE_BACKTRACE
#include <execinfo.h>
#endif

#include "converse.h"
#include "pup_c.h"
#include "memory-isomalloc.h"
#include "reconverse-compat.h"

/* Threads */

int CmiThreadIs(int flag) { return 0; }

int CthMigratable(void) { return 0; }

CthThread CthCreateMigratable(CthVoidFn fn, void *arg, int size,
                              CmiIsomallocContext ctx)
{
  return CthCreate(fn, arg, size);
}

CthThread CthPup(pup_er p, CthThread t)
{
  CmiAbort("CthPup: thread migration and checkpointing of threads are not "
           "supported on reconverse.\n");
  return t;
}

size_t CthStackOffset(CthThread t, char *p) { return (size_t)p; }

char *CthPointer(CthThread t, size_t pos) { return (char *)pos; }

void CtgInit(void) {}

void CthInterceptionsDeactivatePush(CthThread th) {}
void CthInterceptionsDeactivatePop(CthThread th) {}
int CthInterceptionsTemporarilyActivateStart(CthThread th) { return 0; }
void CthInterceptionsTemporarilyActivateEnd(CthThread th, int old) {}

void CmiPrintStackTrace(int nSkip)
{
#if CMK_USE_BACKTRACE
  void *frames[64];
  int n = backtrace(frames, 64);
  int skip = 1 + nSkip; /* this function, then the caller's request */
  if (n > skip) {
    fflush(stdout);
    backtrace_symbols_fd(frames + skip, n - skip, STDERR_FILENO);
  }
#endif
}

/* Isomalloc: every context is the null context; memory comes from the
   ordinary heap and does not migrate. */

int CmiIsomallocEnabled(void) { return 0; }
int CmiIsomallocInRange(void *addr) { return 0; }

CmiIsomallocContext CmiIsomallocContextCreate(int myunit, int numunits)
{
  CmiIsomallocContext ctx = {nullptr};
  return ctx;
}

void CmiIsomallocContextDelete(CmiIsomallocContext ctx) {}

void CmiIsomallocContextPup(pup_er p, CmiIsomallocContext *ctxptr)
{
  CmiAbort("CmiIsomallocContextPup: isomalloc is not available on "
           "reconverse.\n");
}

void CmiIsomallocContextEnableRandomAccess(CmiIsomallocContext ctx) {}
void CmiIsomallocContextJustMigrated(CmiIsomallocContext ctx) {}
void CmiIsomallocEnableRDMA(CmiIsomallocContext ctx, int enable) {}
void CmiIsomallocContextEnableRecording(CmiIsomallocContext ctx, int enable) {}

CmiIsomallocRegion CmiIsomallocContextGetUsedExtent(CmiIsomallocContext ctx)
{
  CmiIsomallocRegion r = {nullptr, nullptr};
  return r;
}

static void *alignedAlloc(size_t align, size_t size)
{
  void *p = nullptr;
  if (align < sizeof(void *)) align = sizeof(void *);
  if (posix_memalign(&p, align, size) != 0) return nullptr;
  return p;
}

void *CmiIsomallocContextMalloc(CmiIsomallocContext ctx, size_t size)
{
  return malloc(size);
}
void *CmiIsomallocContextMallocAlign(CmiIsomallocContext ctx, size_t align,
                                     size_t size)
{
  return alignedAlloc(align, size);
}
void *CmiIsomallocContextCalloc(CmiIsomallocContext ctx, size_t nelem,
                                size_t size)
{
  return calloc(nelem, size);
}
void *CmiIsomallocContextRealloc(CmiIsomallocContext ctx, void *ptr,
                                 size_t size)
{
  return realloc(ptr, size);
}
void CmiIsomallocContextFree(CmiIsomallocContext ctx, void *ptr) { free(ptr); }

size_t CmiIsomallocContextGetLength(CmiIsomallocContext ctx, void *ptr)
{
  CmiAbort("CmiIsomallocContextGetLength: isomalloc is not available on "
           "reconverse.\n");
  return 0;
}

void CmiIsomallocContextProtect(CmiIsomallocContext ctx, void *addr,
                                size_t len, int prot) {}

void *CmiIsomallocContextPermanentAlloc(CmiIsomallocContext ctx, size_t size)
{
  return malloc(size);
}
void *CmiIsomallocContextPermanentAllocAlign(CmiIsomallocContext ctx,
                                             size_t align, size_t size)
{
  return alignedAlloc(align, size);
}

CmiIsomallocContext CmiIsomallocGetThreadContext(CthThread th)
{
  CmiIsomallocContext ctx = {nullptr};
  return ctx;
}

void CmiIsomallocGetRecordedHeap(
    CmiIsomallocContext ctx,
    std::vector<std::tuple<uintptr_t, size_t, size_t>> &heap_vector)
{
  heap_vector.clear();
}

/* memory-isomalloc: there is no malloc interception, so the non-migratable
   allocators are malloc and free, and context activation does nothing. */

void *malloc_nomigrate(size_t size) { return malloc(size); }
void free_nomigrate(void *mem) { free(mem); }
void CmiMemoryIsomallocContextActivate(CmiIsomallocContext ctx) {}
void CmiMemoryIsomallocDisablePush(void) {}
void CmiMemoryIsomallocDisablePop(void) {}
