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

#ifndef _PAGESZ
static int _PAGESZ;
#endif

#undef sun /* I don't care if it's a sun, dangit.  No special treatment. */
#undef BSD /* I don't care if it's BSD.  Same thing. */
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
#ifndef _PAGESZ
  _PAGESZ = getpagesize();
#endif
}

char *malloc(size)
    unsigned size;
{
  char *result;
  CmiMemLock();
  result = CmiMemory_Gnu_malloc(size);
  CmiMemUnlock();
  return result;
}

void free(mem)
    char *mem;
{
  CmiMemLock();
  CmiMemory_Gnu_free(mem);
  CmiMemUnlock();
}

char *calloc(nelem, size)
    unsigned nelem, size;
{
  char *result;
  CmiMemLock();
  result = CmiMemory_Gnu_calloc(nelem, size);
  CmiMemUnlock();
  return result;
}

void cfree(mem)
    char *mem;
{
  CmiMemLock();
  CmiMemory_Gnu_cfree(mem);
  CmiMemUnlock();
}

char *realloc(mem, size)
    char *mem;
    int size;
{
  char *result;
  CmiMemLock();
  result = CmiMemory_Gnu_realloc(mem, size);
  CmiMemUnlock();
  return result;
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

char *valloc(size)
    int size;
{
  char *result;
  CmiMemLock();
  result = CmiMemory_Gnu_valloc(size);
  CmiMemUnlock();
  return result;
}

#endif /* CMK_MALLOC_USE_GNU_MALLOC */
