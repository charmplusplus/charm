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
 * CMK_MALLOC_USE_GNU_WITH_INTERRUPT_SUPPORT
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
 * CMK_MALLOC_USE_GNU
 *
 * The GNU memory allocator is a good all-round memory allocator for
 * distributed memory machines.  It has no support for shared memory.
 *
 *****************************************************************************/

#if CMK_MALLOC_USE_GNU

void CmiMemoryInit(argv)
  char **argv;
{
}

#undef sun /* I don't care if it's a sun, dangit.  No special treatment. */
#undef BSD /* I don't care if it's BSD.  Same thing. */
#include "gnumalloc.c"
#endif

/*****************************************************************************
 *
 * CMK_MALLOC_USE_GNU_WITH_INTERRUPT_SUPPORT
 *
 * This setting uses the GNU memory allocator, however, it surrounds every
 * memory routines with CmiInterruptsBlock and CmiInterruptsRelease (to make
 * it possible to use the memory allocator in an interrupt handler).  For
 * this to work correctly, the interrupt handler must start like this:
 *
 * void InterruptHandlerName()
 * {
 *    CmiInterruptHeader(InterruptHandlerName);
 *    ... rest of handler can use malloc ...
 * }
 *
 * 
 *****************************************************************************/

#if CMK_MALLOC_USE_GNU_WITH_INTERRUPT_SUPPORT

#define malloc   CmiMemory_Gnu_malloc
#define free     CmiMemory_Gnu_free
#define calloc   CmiMemory_Gnu_calloc
#define cfree    CmiMemory_Gnu_cfree
#define realloc  CmiMemory_Gnu_realloc
#define memalign CmiMemory_Gnu_memalign
#define valloc   CmiMemory_Gnu_valloc

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
}

char *malloc(size)
    unsigned size;
{
    char *result;
    CmiInterruptsBlock();
    result = CmiMemory_Gnu_malloc(size);
    CmiInterruptsRelease();
    return result;
}

void free(mem)
    char *mem;
{
    CmiInterruptsBlock();
    CmiMemory_Gnu_free(mem);
    CmiInterruptsRelease();
}

char *calloc(nelem, size)
    unsigned nelem, size;
{
    char *result;
    CmiInterruptsBlock();
    result = CmiMemory_Gnu_calloc(nelem, size);
    CmiInterruptsRelease();
    return result;
}

void cfree(mem)
    char *mem;
{
    CmiInterruptsBlock();
    CmiMemory_Gnu_cfree(mem);
    CmiInterruptsRelease();
}

char *realloc(mem, size)
    char *mem;
    int size;
{
    char *result;
    CmiInterruptsBlock();
    result = CmiMemory_Gnu_realloc(mem, size);
    CmiInterruptsRelease();
    return result;
}

char *memalign(align, size)
    int align, size;
{
    char *result;
    CmiInterruptsBlock();
    result = CmiMemory_Gnu_memalign(align, size);
    CmiInterruptsRelease();
    return result;
}

char *valloc(size)
    int size;
{
    char *result;
    CmiInterruptsBlock();
    result = CmiMemory_Gnu_valloc(size);
    CmiInterruptsRelease();
    return result;
}

#endif /* CMK_MALLOC_USE_GNU_WITH_INTERRUPT_SUPPORT */
