/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/******************************************************************************
 *
 * This module provides the functions malloc, free, calloc, cfree,
 * realloc, valloc, and memalign.
 *
 * There are several possible implementations provided here-- the user
 * can link in whichever is best using the -malloc option with charmc.
 *
 * The major possibilities here are empty (use the system's malloc),
 * GNU (use malloc-gnu.c), and meta (use malloc-cache.c or malloc-paranoid.c).
 * On machines without sbrk(), only the system's malloc is available.
 * 
 * The CMK_MEMORY_BUILD_* symbols come in from the compiler command line.
 *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h> /*For memset, memcpy*/
#ifndef WIN32
#  include <unistd.h> /*For getpagesize*/
#endif
#include "converse.h"

/*Choose the proper default configuration*/
#if CMK_MEMORY_BUILD_DEFAULT

# if CMK_MALLOC_USE_OS_BUILTIN
/*Default to the system malloc-- perhaps the only possibility*/
#  define CMK_MEMORY_BUILD_OS 1
# else
/*Choose a good all-around default malloc*/
#  define CMK_MEMORY_BUILD_GNU 1
# endif

#endif

/*Rank of the processor that's currently holding the CmiMemLock,
or -1 if nobody has it.  Only set when malloc might be reentered.
*/
static int rank_holding_CmiMemLock=-1;


/**
 * memory_lifeRaft is a very small heap-allocated region.
 * The lifeRaft is supposed to be just big enough to provide 
 * enough memory to cleanly shut down if we run out of memory.
 */
static char *memory_lifeRaft=NULL;

void CmiOutOfMemoryInit(void) {
  memory_lifeRaft=(char *)malloc(65536/2);
}

void CmiOutOfMemory(int nBytes) 
{ /* We're out of memory: free up the liferaft memory and abort */
  char errMsg[200];
  if (memory_lifeRaft) free(memory_lifeRaft);
  if (nBytes>0) sprintf(errMsg,"Could not malloc() %d bytes--are we out of memory?",nBytes);
  else sprintf(errMsg,"Could not malloc()--are we out of memory?");
  CmiAbort(errMsg);
}


#if CMK_MEMORY_BUILD_OS
/* Just use the OS's built-in malloc.  All we provide is CmiMemoryInit.
*/
void CmiMemoryInit(argv)
  char **argv;
{
  CmiOutOfMemoryInit();
}
void *malloc_reentrant(size_t size) { return malloc(size); }
void free_reentrant(void *mem) { free(mem); }

#else 
/*************************************************************
*Not* using the system malloc-- first pick the implementation:
*/

#if CMK_MEMORY_BUILD_GNU
static void meta_init(char **argv) {}
#define meta_malloc   mm_malloc
#define meta_free     mm_free
#define meta_calloc   mm_calloc
#define meta_cfree    mm_cfree
#define meta_realloc  mm_realloc
#define meta_memalign mm_memalign
#define meta_valloc   mm_valloc

#include "memory-gnu.c"
#endif /* CMK_MEMORY_BUILD_GNU */

#if CMK_MEMORY_BUILD_VERBOSE
#include "memory-verbose.c"
#endif

#if CMK_MEMORY_BUILD_PARANOID
#include "memory-paranoid.c"
#endif

#if CMK_MEMORY_BUILD_CACHE
#include "memory-cache.c"
#endif 

#if CMK_MEMORY_BUILD_ISOMALLOC
#include "memory-isomalloc.c"
#endif 

/*A trivial sample implementation of the meta_* calls:*/
#if 0
/* Use Gnumalloc as meta-meta malloc fallbacks (mm_*) */
#include "memory-gnu.c"
static void meta_init(char **argv)
{
  
}
static void *meta_malloc(size_t size)
{
  return mm_malloc(size);
}
static void meta_free(void *mem)
{
  mm_free(mem);
}
static void *meta_calloc(size_t nelem, size_t size)
{
  return mm_calloc(nelem,size);
}
static void meta_cfree(void *mem)
{
  mm_cfree(m);
}
static void *meta_realloc(void *mem, size_t size)
{
  return mm_realloc(mem,size);
}
static void *meta_memalign(size_t align, size_t size)
{
  return mm_memalign(align,size);
}
static void *meta_valloc(size_t size)
{
  return mm_valloc(size);
}
#endif


/*******************************************************************
The locking code is common to all implementations except OS-builtin.
*/

void CmiMemoryInit(char **argv)
{
  meta_init(argv);
  CmiOutOfMemoryInit();
}

void *malloc(size_t size)
{
  void *result;
  CmiMemLock();
  result = meta_malloc(size);
  CmiMemUnlock();
  if (result==NULL) CmiOutOfMemory(size);
  return result;
}

void free(void *mem)
{
  CmiMemLock();
  meta_free(mem);
  CmiMemUnlock();
}

void *calloc(size_t nelem, size_t size)
{
  void *result;
  CmiMemLock();
  result = meta_calloc(nelem, size);
  CmiMemUnlock();
  if (result==NULL) CmiOutOfMemory(size);
  return result;
}

void cfree(void *mem)
{
  CmiMemLock();
  meta_cfree(mem);
  CmiMemUnlock();
}

void *realloc(void *mem, size_t size)
{
  void *result;
  CmiMemLock();
  result = meta_realloc(mem, size);
  CmiMemUnlock();
  return result;
}

void *memalign(size_t align, size_t size)
{
  void *result;
  CmiMemLock();
  result = meta_memalign(align, size);
  CmiMemUnlock();
  if (result==NULL) CmiOutOfMemory(align*size);
  return result;    
}

void *valloc(size_t size)
{
  void *result;
  CmiMemLock();
  result = meta_valloc(size);
  CmiMemUnlock();
  if (result==NULL) CmiOutOfMemory(size);
  return result;
}

/*These are special "reentrant" versions of malloc,
for use from code that may be called from within malloc.
The only difference is that these versions check a global
flag to see if they already hold the memory lock before
actually trying the lock, which prevents a deadlock where
you try to aquire one of your own locks.
*/

void *malloc_reentrant(size_t size) {
  void *result;
  if (CmiMyRank()!=rank_holding_CmiMemLock) CmiMemLock();
  result = meta_malloc(size);
  if (CmiMyRank()!=rank_holding_CmiMemLock) CmiMemUnlock();
  return result;
}

void free_reentrant(void *mem)
{
  if (CmiMyRank()!=rank_holding_CmiMemLock) CmiMemLock();
  meta_free(mem);
  if (CmiMyRank()!=rank_holding_CmiMemLock) CmiMemUnlock();
}

#endif /* ! CMK_MEMORY_BUILD_BUILTIN*/

#ifndef CMK_MEMORY_HAS_NOMIGRATE
/*Default implementations of the nomigrate routines:*/
void *malloc_nomigrate(size_t size) { return malloc(size); }
void free_nomigrate(void *mem) { free(mem); }
#endif

#ifndef CMK_MEMORY_HAS_ISOMALLOC
#include "memory-isomalloc.h"
/*Not using isomalloc heaps, so forget about activating block list:*/
CmiIsomallocBlockList *CmiIsomallocBlockListActivate(CmiIsomallocBlockList *l)
   {return l;}
#endif

