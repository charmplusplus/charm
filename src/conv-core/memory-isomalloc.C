/******************************************************************************

A migratable memory allocator.

NOTE: isomalloc is threadsafe, so the isomallocs are not wrapped in CmiMemLock.

*****************************************************************************/

#define CMK_ISOMALLOC_EXCLUDE_FORTRAN_CALLS   0

#include "memory-isomalloc.h"
#include <errno.h>

struct CmiMemoryIsomallocState {
  CmiIsomallocContext context;
  unsigned char disabled;
};

/*The current allocation arena */
CpvStaticDeclare(struct CmiMemoryIsomallocState, isomalloc_state);

/* temporarily disable/enable isomalloc. Note the following two fucntions
 * must be used in pair, and no suspend of thread is allowed in between
 * */
void CmiMemoryIsomallocDisablePush()
{
  CpvAccess(isomalloc_state).disabled++;
}
void CmiMemoryIsomallocDisablePop()
{
  CmiAssert(CpvAccess(isomalloc_state).disabled > 0);
  CpvAccess(isomalloc_state).disabled--;
}

#if CMK_HAS_TLS_VARIABLES
/**
 * make sure isomalloc is only called in pthreads that is spawned by Charm++.
 * It is not safe to call isomalloc in system spawned pthreads for example
 * mpich pthreads, or aio pthreads.
 * Use the following TLS variable to distinguish those pthreads.
 * when set to 1, the current pthreads is allowed to call isomalloc.
 */
static CMK_THREADLOCAL int isomalloc_thread = 0;
#endif

static int meta_inited = 0;

static void meta_init(char **argv)
{
   if (CmiMyRank()==0) CmiMemoryIs_flag|=CMI_MEMORY_IS_ISOMALLOC;
   CpvInitialize(struct CmiMemoryIsomallocState, isomalloc_state);
   CpvAccess(isomalloc_state).context.opaque = nullptr;
   CpvAccess(isomalloc_state).disabled = 0;
#if CMK_HAS_TLS_VARIABLES
   isomalloc_thread = 1;         /* isomalloc is allowed in this pthread */
#endif
   if (CmiMyRank()==0) meta_inited = 1;
}

static bool meta_active()
{
  return meta_inited
    && CpvInitialized(isomalloc_state)
    && CpvAccess(isomalloc_state).context.opaque
    && !CpvAccess(isomalloc_state).disabled
#if CMK_HAS_TLS_VARIABLES
    && (isomalloc_thread || CmiThreadIs(CMI_THREAD_IS_TLS))
#endif
    ;
}

static void *meta_malloc(size_t size)
{
  if (!meta_active())
    return mm_malloc(size);
#if CMK_ISOMALLOC_EXCLUDE_FORTRAN_CALLS
  else if (CmiIsFortranLibraryCall() == 1)
    return mm_malloc(size);
#endif

  CmiMemoryIsomallocDisablePush();
  void * ret = CmiIsomallocContextMalloc(CpvAccess(isomalloc_state).context, size);
  CmiMemoryIsomallocDisablePop();
  return ret;
}

static void meta_free(void *mem)
{
  if (!CmiIsomallocInRange(mem))
  {
    mm_free(mem);
    return;
  }

  if (mem == nullptr || !CpvInitialized(isomalloc_state))
    return;

  auto ctx = CpvAccess(isomalloc_state).context;
  if (ctx.opaque == nullptr)
    return;

  CmiMemoryIsomallocDisablePush();

  auto region = CmiIsomallocContextGetUsedExtent(ctx);
  if (region.start <= mem && mem < region.end)
    CmiIsomallocContextFree(ctx, mem);

  CmiMemoryIsomallocDisablePop();
}

static void *meta_calloc(size_t nelem, size_t size)
{
  if (!meta_active())
    return mm_calloc(nelem, size);
#if CMK_ISOMALLOC_EXCLUDE_FORTRAN_CALLS
  else if (CmiIsFortranLibraryCall() == 1)
    return mm_calloc(nelem, size);
#endif

  CmiMemoryIsomallocDisablePush();
  void * ret = CmiIsomallocContextCalloc(CpvAccess(isomalloc_state).context, nelem, size);
  CmiMemoryIsomallocDisablePop();
  return ret;
}

static void meta_cfree(void *mem)
{
	meta_free(mem);
}

static void *meta_realloc(void *oldBuffer, size_t newSize)
{
  /*Just forget it for regular malloc's:*/
  if (!CmiIsomallocInRange(oldBuffer) || !meta_active())
    return mm_realloc(oldBuffer, newSize);
#if CMK_ISOMALLOC_EXCLUDE_FORTRAN_CALLS
  else if (CmiIsFortranLibraryCall() == 1)
    return mm_realloc(oldBuffer, newSize);
#endif

  CmiMemoryIsomallocDisablePush();
  void * ret = CmiIsomallocContextRealloc(CpvAccess(isomalloc_state).context, oldBuffer, newSize);
  CmiMemoryIsomallocDisablePop();
  return ret;
}

static void *meta_memalign(size_t align, size_t size)
{
  if (!meta_active())
    return mm_memalign(align, size);
#if CMK_ISOMALLOC_EXCLUDE_FORTRAN_CALLS
  else if (CmiIsFortranLibraryCall() == 1)
    return mm_memalign(align, size);
#endif

  CmiMemoryIsomallocDisablePush();
  void * ret = CmiIsomallocContextMallocAlign(CpvAccess(isomalloc_state).context, align, size);
  CmiMemoryIsomallocDisablePop();
  return ret;
}

static int meta_posix_memalign(void **outptr, size_t align, size_t size)
{
  if (!meta_active())
    return mm_posix_memalign(outptr, align, size);
#if CMK_ISOMALLOC_EXCLUDE_FORTRAN_CALLS
  else if (CmiIsFortranLibraryCall() == 1)
    return mm_posix_memalign(outptr, align, size);
#endif

  CmiMemoryIsomallocDisablePush();
  void * ret = CmiIsomallocContextMallocAlign(CpvAccess(isomalloc_state).context, align, size);
  CmiMemoryIsomallocDisablePop();
  if (ret == nullptr)
    return ENOMEM;
  *outptr = ret;
  return 0;
}

static void *meta_aligned_alloc(size_t align, size_t size)
{
  if (!meta_active())
    return mm_aligned_alloc(align, size);
#if CMK_ISOMALLOC_EXCLUDE_FORTRAN_CALLS
  else if (CmiIsFortranLibraryCall() == 1)
    return mm_aligned_alloc(align, size);
#endif

  CmiMemoryIsomallocDisablePush();
  void * ret = CmiIsomallocContextMallocAlign(CpvAccess(isomalloc_state).context, align, size);
  CmiMemoryIsomallocDisablePop();
  return ret;
}

static void *meta_valloc(size_t size)
{
	return meta_memalign(CmiGetPageSize(), size);
}

static void *meta_pvalloc(size_t size)
{
	const size_t pagesize = CmiGetPageSize();
	return meta_memalign(pagesize, (size + pagesize - 1) & ~(pagesize - 1));
}

#define CMK_MEMORY_HAS_NOMIGRATE
/*Allocate non-migratable memory:*/
void *malloc_nomigrate(size_t size) { 
  return mm_malloc(size);
}

void free_nomigrate(void *mem)
{
  mm_free(mem);
}

#define CMK_MEMORY_HAS_ISOMALLOC

/*Make this context "active"-- the recipient of incoming mallocs.*/
void CmiMemoryIsomallocContextActivate(CmiIsomallocContext l)
{
	CpvAccess(isomalloc_state).context = l;
}
