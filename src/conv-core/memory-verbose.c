/******************************************************************************

A caching memory allocator-- #included whole by memory.c

Orion Sky Lawlor, olawlor@acm.org, 6/22/2001

*****************************************************************************/

#if ! CMK_MEMORY_BUILD_OS
/* Use Gnumalloc as meta-meta malloc fallbacks (mm_*) */
#include "memory-gnu.c"
#endif

static int memInit=0;
static int inMemVerbose=0;

static void meta_init(char **argv)
{
  memInit=1;
  if (memInit) CmiPrintf("CMI_MEMORY(%d)> Called meta_init\n",
			 CmiMyPe());
}

static void *meta_malloc(size_t size)
{
  void *ret=mm_malloc(size);
  if (memInit) CmiPrintf("CMI_MEMORY(%d)> malloc(%d) => %p\n",
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
  if (memInit) CmiPrintf("CMI_MEMORY(%d)> calloc(%d,%d) => %p\n",
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
  if (memInit) CmiPrintf("CMI_MEMORY(%d)> realloc(%p,%d) => %p\n",
			 CmiMyPe(),mem,size,ret);
  return ret;
}

static void *meta_memalign(size_t align, size_t size)
{
  void *ret=mm_memalign(align,size);
  if (memInit) CmiPrintf("CMI_MEMORY(%d)> memalign(%p,%d) => %p\n",
			 CmiMyPe(),align,size,ret);
  return ret;
}

static void *meta_valloc(size_t size)
{
  void *ret=mm_valloc(size);
  if (memInit) CmiPrintf("CMI_MEMORY(%d)> valloc(%d) => %p\n",
			 CmiMyPe(),size,ret);  
  return ret;
}


