/******************************************************************************
 *
 * This module provides the functions malloc, free, calloc, cfree,
 * realloc, valloc, and memalign.
 *
 * There are several possible implementations provided here, each machine can
 * choose whichever one is best using one of the following flags.
 *
 * CMK_MALLOC_USE_OS_BUILTIN
 * CMK_MALLOC_USE_GNU
 * CMK_MALLOC_USE_GNU_WITH_CMIMEMLOCK
 *
 *****************************************************************************/

#include "converse.h"

/*****************************************************************************
 *
 * CMK_MALLOC_USE_OS_BUILTIN
 *
 * Just use the OS's built-in malloc.  All we provide is CmiMemoryInit.
 *
 *****************************************************************************/

#if CMK_MALLOC_USE_OS_BUILTIN

void CmiMemoryInit(argv)
  char **argv;
{
}

#endif

/*****************************************************************************
 *
 * CMK_MALLOC_USE_GNU_MALLOC
 *
 * The GNU memory allocator is a good all-round memory allocator for
 * distributed memory machines.  It has the advantage that you can define
 * CmiMemLock and CmiMemUnlock to provide locking around it's operations.
 *
 *****************************************************************************/

#if CMK_MALLOC_USE_GNU_MALLOC

#define malloc   CmiMemory_Gnu_malloc
#define free     CmiMemory_Gnu_free
#define calloc   CmiMemory_Gnu_calloc
#define cfree    CmiMemory_Gnu_cfree
#define realloc  CmiMemory_Gnu_realloc
#define memalign CmiMemory_Gnu_memalign
#define valloc   CmiMemory_Gnu_valloc

#undef sun /* I don't care if it's a sun, dangit.  No special treatment. */
#undef BSD /* I don't care if it's BSD.  Same thing. */
#if CMK_GETPAGESIZE_AVAILABLE
#define HAVE_GETPAGESIZE
#endif

#include "gnumalloc.c"

#undef malloc
#undef free
#undef calloc
#undef cfree
#undef realloc
#undef memalign
#undef valloc

void CmiMemoryInit(argv)
char **argv;
{
}

void *malloc(size)
    unsigned size;
{
  char *result;
  CmiMemLock();
  result = CmiMemory_Gnu_malloc(size);
  CmiMemUnlock();
  return (void *) result;
}

void free(mem)
    void *mem;
{
  CmiMemLock();
  CmiMemory_Gnu_free(mem);
  CmiMemUnlock();
}

void *calloc(nelem, size)
    unsigned nelem, size;
{
  char *result;
  CmiMemLock();
  result = CmiMemory_Gnu_calloc(nelem, size);
  CmiMemUnlock();
  return (void *) result;
}

void cfree(mem)
    char *mem;
{
  CmiMemLock();
  CmiMemory_Gnu_cfree(mem);
  CmiMemUnlock();
}

void *realloc(mem, size)
    void *mem;
    size_t size;
{
  char *result;
  CmiMemLock();
  result = CmiMemory_Gnu_realloc(mem, size);
  CmiMemUnlock();
  return (void *) result;
}

char *memalign(align, size)
    int align, size;
{
  char *result;
  CmiMemLock();
  result = CmiMemory_Gnu_memalign(align, size);
  CmiMemUnlock();
  return result;    
}

void *valloc(size)
    size_t size;
{
  char *result;
  CmiMemLock();
  result = CmiMemory_Gnu_valloc(size);
  CmiMemUnlock();
  return (void *) result;
}

#endif /* CMK_MALLOC_USE_GNU_MALLOC */
