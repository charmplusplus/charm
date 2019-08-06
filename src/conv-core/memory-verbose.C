/******************************************************************************

A caching memory allocator-- #included whole by memory.C

Orion Sky Lawlor, olawlor@acm.org, 6/22/2001

*****************************************************************************/

#if ! CMK_MEMORY_BUILD_OS
/* Use Gnumalloc as meta-meta malloc fallbacks (mm_*) */
#include "memory-gnu.C"
#endif

static int memInit=0;
static int inMemVerbose=0;

static void meta_init(char **argv)
{
  if (CmiMyRank()==0) memInit=1;
  CmiNodeAllBarrier();
  if (memInit) CmiPrintf("CMI_MEMORY(%d)> Called meta_init\n", CmiMyPe());
}

static void *meta_malloc(size_t size)
{
  void *ret=mm_malloc(size);
  if (memInit) CmiPrintf("CMI_MEMORY(%d)> malloc(%zu) => %p\n",
			 CmiMyPe(),size,ret);
  if (memInit>1) {int memBack=memInit; memInit=0; CmiPrintStackTrace(0); memInit=memBack;}
  return ret;
}

static void meta_free(void *mem)
{
  if (memInit && !inMemVerbose) {
    inMemVerbose = 1;
    CmiPrintf("CMI_MEMORY(%d)> free(%p)\n", CmiMyPe(),mem);
    inMemVerbose = 0;
  }
  if (memInit>1) {int memBack=memInit; memInit=0; CmiPrintStackTrace(0); memInit=memBack;}
  mm_free(mem);
}

static void *meta_calloc(size_t nelem, size_t size)
{
  void *ret=mm_calloc(nelem,size);
  if (memInit) CmiPrintf("CMI_MEMORY(%d)> calloc(%zu,%zu) => %p\n",
			 CmiMyPe(),nelem,size,ret);
  return ret;
}

static void meta_cfree(void *mem)
{
  if (memInit) CmiPrintf("CMI_MEMORY(%d)> free(%p)\n",
			 CmiMyPe(),mem);
  mm_cfree(mem);
}

static void *meta_realloc(void *mem, size_t size)
{
  void *ret=mm_realloc(mem,size);
  if (memInit) CmiPrintf("CMI_MEMORY(%d)> realloc(%p,%zu) => %p\n",
			 CmiMyPe(),mem,size,ret);
  return ret;
}

static void *meta_memalign(size_t align, size_t size)
{
  void *ret=mm_memalign(align,size);
  if (memInit) CmiPrintf("CMI_MEMORY(%d)> memalign(%zu,%zu) => %p\n",
			 CmiMyPe(),align,size,ret);
  return ret;
}

static int meta_posix_memalign(void **outptr, size_t align, size_t size)
{
  void *origptr = *outptr;
  int ret=mm_posix_memalign(outptr,align,size);
  if (memInit) CmiPrintf("CMI_MEMORY(%d)> posix_memalign(%p,%zu,%zu), %p => %d, %p\n",
			 CmiMyPe(),outptr,align,size,origptr,ret,*outptr);
  return ret;
}

static void *meta_aligned_alloc(size_t align, size_t size)
{
  void *ret=mm_aligned_alloc(align,size);
  if (memInit) CmiPrintf("CMI_MEMORY(%d)> aligned_alloc(%zu,%zu) => %p\n",
			 CmiMyPe(),align,size,ret);
  return ret;
}

static void *meta_valloc(size_t size)
{
  void *ret=mm_valloc(size);
  if (memInit) CmiPrintf("CMI_MEMORY(%d)> valloc(%zu) => %p\n",
			 CmiMyPe(),size,ret);  
  return ret;
}

static void *meta_pvalloc(size_t size)
{
  void *ret=mm_pvalloc(size);
  if (memInit) CmiPrintf("CMI_MEMORY(%d)> pvalloc(%zu) => %p\n",
			 CmiMyPe(),size,ret);
  return ret;
}

