/******************************************************************************
 *
 * This module provides the functions malloc, free, calloc, cfree,
 * realloc, valloc, and memalign.
 *
 * There are several possible implementations provided here, each machine can
 * choose whichever one is best using one of the following flags.
 *
 * CMK_USE_OS_MALLOC
 * CMK_USE_GNU_MALLOC
 * CMK_USE_GNU_MALLOC_WITH_INTERRUPT_SUPPORT
 *
 *****************************************************************************/

#include "converse.h"

/*****************************************************************************
 *
 * CMK_USE_GNU_MALLOC
 *
 * The GNU memory allocator is a good all-round memory allocator for
 * distributed memory machines.  It has no support for shared memory.
 *
 *****************************************************************************/

#ifdef CMK_USE_GNU_MALLOC
#undef sun /* I don't care if it's a sun, dangit.  No special treatment. */
#undef BSD /* I don't care if it's BSD.  Same thing. */
#include "gnumalloc.c"
#endif

/*****************************************************************************
 *
 * CMK_USE_GNU_MALLOC_WITH_INTERRUPT_SUPPORT
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

#ifdef CMK_USE_GNU_MALLOC_WITH_INTERRUPT_SUPPORT

#define malloc   Cmem_gnu_malloc
#define free     Cmem_gnu_free
#define calloc   Cmem_gnu_calloc
#define cfree    Cmem_gnu_cfree
#define realloc  Cmem_gnu_realloc
#define memalign Cmem_gnu_memalign
#define valloc   Cmem_gnu_valloc

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

char *malloc(size)
    unsigned size;
{
    char *result;
    CmiInterruptsBlock();
    result = Cmem_gnu_malloc(size);
    CmiInterruptsRelease();
    return result;
}

void free(mem)
    char *mem;
{
    CmiInterruptsBlock();
    Cmem_gnu_free(mem);
    CmiInterruptsRelease();
}

char *calloc(nelem, size)
    unsigned nelem, size;
{
    char *result;
    CmiInterruptsBlock();
    result = Cmem_gnu_calloc(nelem, size);
    CmiInterruptsRelease();
    return result;
}

void cfree(mem)
    char *mem;
{
    CmiInterruptsBlock();
    Cmem_gnu_cfree(mem);
    CmiInterruptsRelease();
}

char *realloc(mem, size)
    char *mem;
    int size;
{
    char *result;
    CmiInterruptsBlock();
    result = Cmem_gnu_realloc(mem, size);
    CmiInterruptsRelease();
    return result;
}

char *memalign(align, size)
    int align, size;
{
    char *result;
    CmiInterruptsBlock();
    result = Cmem_gnu_memalign(align, size);
    CmiInterruptsRelease();
    return result;
}

char *valloc(size)
    int size;
{
    char *result;
    CmiInterruptsBlock();
    result = Cmem_gnu_valloc(size);
    CmiInterruptsRelease();
    return result;
}

#endif /* CMK_USE_GNU_MALLOC_WITH_INTERRUPT_SUPPORT */
