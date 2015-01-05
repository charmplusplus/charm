

/**  
 @defgroup MemoryModule Memory Allocation and Monitoring

 This module provides the functions malloc, free, calloc, cfree,
 realloc, valloc, memalign, and other functions for determining
 the current and past memory usage. 

 There are several possible implementations provided here-- the user
 can link in whichever is best using the -malloc option with charmc.

 The major possibilities here are empty (use the system's malloc),
 GNU (use malloc-gnu.c), and meta (use malloc-cache.c or malloc-paranoid.c).
 On machines without sbrk(), only the system's malloc is available.

 The CMK_MEMORY_BUILD_* symbols come in from the compiler command line.

 To determine how much memory your Charm++ program is using on a 
 processor, call CmiMemoryUsage(). This function will return the 
 number of bytes allocated, usually representing the heap size. 
 It is possible that this measurement will not exactly represent
 the heap size, but rather will reflect the total amount of 
 memory used by the program.

*/


/** 
    @addtogroup MemoryModule
    @{
*/


/******************************************************************************
 *
 * This module provides the functions malloc, free, calloc, cfree,
 * realloc, valloc, and memalign.
 *
 *
 *****************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <string.h> /*For memset, memcpy*/
#ifndef WIN32
#  include <unistd.h> /*For getpagesize*/
#endif
#include "converse.h"

void * memory_stack_top; /*The higher end of the stack (approximation)*/
int cpdInSystem=1; /*Start inside the system (until we start executing user code)*/

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

#if CMK_MEMORY_BUILD_OS_WRAPPED
#define CMK_MEMORY_BUILD_OS 1
#endif
#if CMK_MEMORY_BUILD_GNU_HOOKS
/*While in general on could build hooks on the GNU we have, for now the hooks
 * are setup to work only with OS memory libraries --Filippo*/
#define CMK_MEMORY_BUILD_OS 1
#endif

#if CMK_MEMORY_BUILD_OS
#if CMK_MEMORY_BUILD_OS_WRAPPED

void initialize_memory_wrapper();
void * initialize_memory_wrapper_calloc(size_t nelem, size_t size);
void * initialize_memory_wrapper_malloc(size_t size);
void * initialize_memory_wrapper_realloc(void *ptr, size_t size);
void * initialize_memory_wrapper_memalign(size_t align, size_t size);
void * initialize_memory_wrapper_valloc(size_t size);
void initialize_memory_wrapper_free(void *ptr);
void initialize_memory_wrapper_cfree(void *ptr);

void * (*mm_malloc)(size_t) = initialize_memory_wrapper_malloc;
void * (*mm_calloc)(size_t,size_t) = initialize_memory_wrapper_calloc;
void * (*mm_realloc)(void*,size_t) = initialize_memory_wrapper_realloc;
void * (*mm_memalign)(size_t,size_t) = initialize_memory_wrapper_memalign;
void * (*mm_valloc)(size_t) = initialize_memory_wrapper_valloc;
void (*mm_free)(void*) = initialize_memory_wrapper_free;
void (*mm_cfree)(void*) = initialize_memory_wrapper_cfree;

void * initialize_memory_wrapper_calloc(size_t nelem, size_t size) {
  static int calloc_wrapper = 0;
  if (calloc_wrapper) return NULL;
  calloc_wrapper = 1;
  initialize_memory_wrapper();
  return (*mm_calloc)(nelem,size);
}

void * initialize_memory_wrapper_malloc(size_t size) {
  static int malloc_wrapper = 0;
  if (malloc_wrapper) return NULL;
  malloc_wrapper = 1;
  initialize_memory_wrapper();
  return (*mm_malloc)(size);
}

void * initialize_memory_wrapper_realloc(void *ptr, size_t size) {
  initialize_memory_wrapper();
  return (*mm_realloc)(ptr,size);
}

void * initialize_memory_wrapper_memalign(size_t align, size_t size) {
  initialize_memory_wrapper();
  return (*mm_memalign)(align,size);
}

void * initialize_memory_wrapper_valloc(size_t size) {
  initialize_memory_wrapper();
  return (*mm_valloc)(size);
}

void initialize_memory_wrapper_free(void *ptr) {
  initialize_memory_wrapper();
  (*mm_free)(ptr);
}

void initialize_memory_wrapper_cfree(void *ptr) {
  initialize_memory_wrapper();
  (*mm_cfree)(ptr);
}

#define mm_malloc   (*mm_malloc)
#define mm_free     (*mm_free)
#define mm_calloc   (*mm_calloc)
#define mm_cfree    (*mm_cfree)
#define mm_realloc  (*mm_realloc)
#define mm_memalign (*mm_memalign)
#define mm_valloc   (*mm_valloc)

#else
#define mm_malloc   malloc
#define mm_calloc   calloc
#define mm_memalign memalign
#define mm_free     free
#endif
#endif

CMK_TYPEDEF_UINT8 _memory_allocated = 0;
CMK_TYPEDEF_UINT8 _memory_allocated_max = 0; /* High-Water Mark */
CMK_TYPEDEF_UINT8 _memory_allocated_min = 0; /* Low-Water Mark */

/*Rank of the processor that's currently holding the CmiMemLock,
or -1 if nobody has it.  Only set when malloc might be reentered.
*/
static int rank_holding_CmiMemLock=-1;

/* By default, there are no flags */
static int CmiMemoryIs_flag=0;

int CmiMemoryIs(int flag)
{
	return (CmiMemoryIs_flag&flag)==flag;
}

/**
 * memory_lifeRaft is a very small heap-allocated region.
 * The lifeRaft is supposed to be just big enough to provide
 * enough memory to cleanly shut down if we run out of memory.
 */
static char *memory_lifeRaft=NULL;

void CmiOutOfMemoryInit(void);

void CmiOutOfMemory(int nBytes)
{ /* We're out of memory: free up the liferaft memory and abort */
  char errMsg[200];
  if (memory_lifeRaft) free(memory_lifeRaft);
  if (nBytes>0) sprintf(errMsg,"Could not malloc() %d bytes--are we out of memory? (used :%.3fMB)",nBytes,CmiMemoryUsage()/1000000.0);
  else sprintf(errMsg,"Could not malloc()--are we out of memory? (used: %.3fMB)", CmiMemoryUsage()/1000000.0);
  CmiAbort(errMsg);
}

/* Global variables keeping track of the status of the system (mostly used by charmdebug) */
int memory_status_info=0;
int memory_chare_id=0;

#if CMK_MEMORY_BUILD_OS
/* Just use the OS's built-in malloc.  All we provide is CmiMemoryInit.
*/


#if CMK_MEMORY_BUILD_OS_WRAPPED || CMK_MEMORY_BUILD_GNU_HOOKS

#if CMK_MEMORY_BUILD_GNU_HOOKS

static void *meta_malloc(size_t);
static void *meta_realloc(void*,size_t);
static void *meta_memalign(size_t,size_t);
static void meta_free(void*);
static void *meta_malloc_hook(size_t s, const void* c) {return meta_malloc(s);}
static void *meta_realloc_hook(void* p,size_t s, const void* c) {return meta_realloc(p,s);}
static void *meta_memalign_hook(size_t s1,size_t s2, const void* c) {return meta_memalign(s1,s2);}
static void meta_free_hook(void* p, const void* c) {meta_free(p);}

#define BEFORE_MALLOC_CALL \
  __malloc_hook = old_malloc_hook; \
  __realloc_hook = old_realloc_hook; \
  __memalign_hook = old_memalign_hook; \
  __free_hook = old_free_hook;
#define AFTER_MALLOC_CALL \
  old_malloc_hook = __malloc_hook; \
  old_realloc_hook = __realloc_hook; \
  old_memalign_hook = __memalign_hook; \
  old_free_hook = __free_hook; \
  __malloc_hook = meta_malloc_hook; \
  __realloc_hook = meta_realloc_hook; \
  __memalign_hook = meta_memalign_hook; \
  __free_hook = meta_free_hook;

#if CMK_HAS_MALLOC_H
#include <malloc.h>
#endif
static void *(*old_malloc_hook) (size_t, const void*);
static void *(*old_realloc_hook) (void*,size_t, const void*);
static void *(*old_memalign_hook) (size_t,size_t, const void*);
static void (*old_free_hook) (void*, const void*);

#else /* CMK_MEMORY_BUILD_GNU_HOOKS */
#define BEFORE_MALLOC_CALL   /*empty*/
#define AFTER_MALLOC_CALL    /*empty*/
#endif

#if CMK_MEMORY_BUILD_VERBOSE
#include "memory-verbose.c"
#endif

#if CMK_MEMORY_BUILD_PARANOID
#include "memory-paranoid.c"
#endif

#if CMK_MEMORY_BUILD_LEAK
#include "memory-leak.c"
#endif

#if CMK_MEMORY_BUILD_CACHE
#include "memory-cache.c"
#endif

#if CMK_MEMORY_BUILD_ISOMALLOC
#include "memory-isomalloc.c"
#endif

#if CMK_MEMORY_BUILD_LOCK
#include "memory-lock.c"
#endif

#if CMK_MEMORY_BUILD_CHARMDEBUG
#include "memory-charmdebug.c"
#endif

#if CMK_MEMORY_BUILD_GNU_HOOKS

static void
my_init_hook (void)
{
  old_malloc_hook = __malloc_hook;
  old_realloc_hook = __realloc_hook;
  old_memalign_hook = __memalign_hook;
  old_free_hook = __free_hook;
  __malloc_hook = meta_malloc_hook;
  __realloc_hook = meta_realloc_hook;
  __memalign_hook = meta_memalign_hook;
  __free_hook = meta_free_hook;
}
/* Override initializing hook from the C library. */
#if defined(__MALLOC_HOOK_VOLATILE)
void (* __MALLOC_HOOK_VOLATILE __malloc_initialize_hook) (void) = my_init_hook;
#else
void (* __malloc_initialize_hook) (void) = my_init_hook;
#endif
#else
void *malloc(size_t size) { return meta_malloc(size); }
void free(void *ptr) { meta_free(ptr); }
void *calloc(size_t nelem, size_t size) { return meta_calloc(nelem,size); }
void cfree(void *ptr) { meta_cfree(ptr); }
void *realloc(void *ptr, size_t size) { return meta_realloc(ptr,size); }
void *memalign(size_t align, size_t size) { return meta_memalign(align,size); }
void *valloc(size_t size) { return meta_valloc(size); }
#endif

#endif

static int skip_mallinfo = 0;

void CmiMemoryInit(argv)
  char **argv;
{
  if(CmiMyRank() == 0)   CmiMemoryIs_flag |= CMI_MEMORY_IS_OS;
#if CMK_MEMORY_BUILD_OS_WRAPPED || CMK_MEMORY_BUILD_GNU_HOOKS
  CmiArgGroup("Converse","Memory module");
  if(CmiMyRank() == 0) meta_init(argv);
#endif
  CmiOutOfMemoryInit();
  if (getenv("MEMORYUSAGE_NO_MALLINFO"))  skip_mallinfo = 1;
}
void *malloc_reentrant(size_t size) { return malloc(size); }
void free_reentrant(void *mem) { free(mem); }

/******Start of a general way to get memory usage information*****/
/*CMK_TYPEDEF_UINT8 CmiMemoryUsage() { return 0; }*/

#if ! CMK_HAS_SBRK
int sbrk(int s) { return 0; }
#endif

#if CMK_HAS_MSTATS
#include <malloc/malloc.h>
#if CMK_C_INLINE
inline
#endif
static CMK_TYPEDEF_UINT8 MemusageMstats(){
	struct mstats ms = mstats();
	CMK_TYPEDEF_UINT8 memtotal = ms.bytes_used;
	return memtotal;
}
#else
#if CMK_C_INLINE
inline
#endif
static CMK_TYPEDEF_UINT8 MemusageMstats() { return 0; }
#endif

static int MemusageInited = 0;
static CMK_TYPEDEF_UINT8 MemusageInitSbrkval = 0;
#if CMK_C_INLINE
inline
#endif
static CMK_TYPEDEF_UINT8 MemusageSbrk(){
	CMK_TYPEDEF_UINT8 newval;
	if(MemusageInited==0){
		MemusageInitSbrkval = (CMK_TYPEDEF_UINT8)sbrk(0);
		MemusageInited = 1;
	}
	newval = (CMK_TYPEDEF_UINT8)sbrk(0);
	return (newval - MemusageInitSbrkval);
}

#if CMK_C_INLINE
inline
#endif
static CMK_TYPEDEF_UINT8 MemusageProcSelfStat(){
    FILE *f;
    int i, ret;
    static int failed_once = 0;
    CMK_TYPEDEF_UINT8 vsz = 0; /* should remain 0 on failure */

    if(failed_once) return 0; /* no point in retrying */
    
    f = fopen("/proc/self/stat", "r");
    if(!f) { failed_once = 1; return 0; }
    for(i=0; i<22; i++) ret = fscanf(f, "%*s");
    ret = fscanf(f, "%lu", &vsz);
    fclose(f);
    if(!vsz) failed_once=1;
    return vsz;
}

#if ! CMK_HAS_MALLINFO || defined(CMK_MALLINFO_IS_BROKEN)
#if CMK_C_INLINE
inline
#endif
static CMK_TYPEDEF_UINT8 MemusageMallinfo(){ return 0;}	
#else
#if CMK_HAS_MALLOC_H
#include <malloc.h>
#endif
#if CMK_C_INLINE
inline
#endif
static CMK_TYPEDEF_UINT8 MemusageMallinfo(){
    /* IA64 seems to ignore mi.uordblks, but updates mi.hblkhd correctly */
    if (skip_mallinfo) return 0;
    else {
    struct mallinfo mi = mallinfo();
    CMK_TYPEDEF_UINT8 memtotal = (CMK_TYPEDEF_UINT8) mi.uordblks;   /* malloc */
    CMK_TYPEDEF_UINT8 memtotal2 = (CMK_TYPEDEF_UINT8) mi.usmblks;   /* unused */
    memtotal2 += (CMK_TYPEDEF_UINT8) mi.hblkhd;               /* mmap */
    /* printf("%lld %lld %lld %lld %lld\n", mi.uordblks, mi.usmblks,mi.hblkhd,mi.arena,mi.keepcost); */
#if ! CMK_CRAYXT && ! CMK_CRAYXE && !CMK_CRAYXC
    if(memtotal2 > memtotal) memtotal = memtotal2;
#endif
    return memtotal;
    }
}
#endif

#if CMK_C_INLINE
inline
#endif
static CMK_TYPEDEF_UINT8 MemusagePS(){
#if ! CMK_HAS_POPEN
    return 0;
#else	
    char pscmd[100];
    CMK_TYPEDEF_UINT8 vsz=0;
    FILE *p;
    int ret;
    sprintf(pscmd, "/bin/ps -o vsz= -p %d", getpid());
    p = popen(pscmd, "r");
    if(p){
	ret = fscanf(p, "%ld", &vsz);
	pclose(p);
    }
    return (vsz * (CMK_TYPEDEF_UINT8)1024);
#endif	
}

#if defined(_WIN32) && ! defined(__CYGWIN__)
#include <windows.h>
#include <psapi.h>

#if CMK_C_INLINE
inline
#endif
static CMK_TYPEDEF_UINT8 MemusageWindows(){
    PROCESS_MEMORY_COUNTERS pmc;
    if ( GetProcessMemoryInfo( GetCurrentProcess(), &pmc, sizeof(pmc)) )
    {
      /* return pmc.WorkingSetSize; */ 
      return pmc.PagefileUsage;    /* total vm size, possibly not in memory */
    }
    return 0;
}
#else
static CMK_TYPEDEF_UINT8 MemusageWindows(){
    return 0;
}
#endif

#if CMK_BLUEGENEP
/* Report the memory usage according to the following wiki page i
* https://wiki.alcf.anl.gov/index.php/Debugging#How_do_I_get_information_on_used.2Favailable_memory_in_my_code.3F
*/
#include <malloc.h>
#if CMK_C_INLINE
inline
#endif
static CMK_TYPEDEF_UINT8 MemusageBGP(){
    struct mallinfo m = mallinfo();
    return m.hblkhd + m.uordblks;
}
#else
static CMK_TYPEDEF_UINT8 MemusageBGP(){
    return 0;
}
#endif

#if CMK_BLUEGENEQ
#include <spi/include/kernel/memory.h>
#if CMK_C_INLINE
inline
#endif
static CMK_TYPEDEF_UINT8 MemusageBGQ(){
  CMK_TYPEDEF_UINT8 heapUsed;
  Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAP, &heapUsed);
  return heapUsed;
}
#else
static CMK_TYPEDEF_UINT8 MemusageBGQ(){
    return 0;
}
#endif

typedef CMK_TYPEDEF_UINT8 (*CmiMemUsageFn)();

/* this structure defines the order of testing for memory usage functions */
struct CmiMemUsageStruct {
    CmiMemUsageFn  fn;
    char *name;
} memtest_order[] = {
    {MemusageBGQ, "BlueGene/Q"},
    {MemusageBGP, "BlueGene/P"},
    {MemusageWindows, "Windows"},
    {MemusageMstats, "Mstats"},
    {MemusageMallinfo, "Mallinfo"},
    {MemusageProcSelfStat, "/proc/self/stat"},
    {MemusageSbrk, "sbrk"},
    {MemusagePS, "ps"},
};

CMK_TYPEDEF_UINT8 CmiMemoryUsage(){
    int i;
    CMK_TYPEDEF_UINT8 memtotal = 0;
    for (i=0; i<sizeof(memtest_order)/sizeof(struct CmiMemUsageStruct); i++) {
        memtotal = memtest_order[i].fn();
        if (memtotal) break;
    }
    return memtotal;
}

char *CmiMemoryUsageReporter(){
    int i;
    CMK_TYPEDEF_UINT8 memtotal = 0;
    char *reporter = NULL;
    for (i=0; i<sizeof(memtest_order)/sizeof(struct CmiMemUsageStruct); i++) {
        memtotal = memtest_order[i].fn();
        reporter = memtest_order[i].name;
        if (memtotal) break;
    }
    return reporter;
}

/******End of a general way to get memory usage information*****/

CMK_TYPEDEF_UINT8 CmiMaxMemoryUsage() { return 0; }
void CmiResetMaxMemory() {}
CMK_TYPEDEF_UINT8 CmiMinMemoryUsage() { return 0; }
void CmiResetMinMemory() {}

#define MEM_LOCK_AROUND(code)   code

#else       /* of CMK_MEMORY_BUILD_OS */

/*************************************************************
*Not* using the system malloc-- first pick the implementation:
*/

#if CMK_MEMORY_BUILD_GNU 
#define meta_malloc   mm_malloc
#define meta_free     mm_free
#define meta_calloc   mm_calloc
#define meta_cfree    mm_cfree
#define meta_realloc  mm_realloc
#define meta_memalign mm_memalign
#define meta_valloc   mm_valloc

#include "memory-gnu.c"
static void meta_init(char **argv) {
  CmiMemoryIs_flag |= CMI_MEMORY_IS_GNU;
}

#endif /* CMK_MEMORY_BUILD_GNU */

#define BEFORE_MALLOC_CALL   /*empty*/
#define AFTER_MALLOC_CALL    /*empty*/

#if CMK_MEMORY_BUILD_VERBOSE
#include "memory-verbose.c"
#endif

#if CMK_MEMORY_BUILD_PARANOID
#include "memory-paranoid.c"
#endif

#if CMK_MEMORY_BUILD_LEAK
#include "memory-leak.c"
#endif

#if CMK_MEMORY_BUILD_CACHE
#include "memory-cache.c"
#endif

#if CMK_MEMORY_BUILD_ISOMALLOC
#include "memory-isomalloc.c"
#endif

#if CMK_MEMORY_BUILD_CHARMDEBUG
#include "memory-charmdebug.c"
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
  CmiArgGroup("Converse","Memory module");
  meta_init(argv);
  CmiOutOfMemoryInit();
}

/* Wrap a CmiMemLock around this code */
#define MEM_LOCK_AROUND(code) \
  CmiMemLock(); \
  code; \
  CmiMemUnlock();

/* Wrap a reentrant CmiMemLock around this code */
#define REENTRANT_MEM_LOCK_AROUND(code) \
  int myRank=CmiMyRank(); \
  if (myRank!=rank_holding_CmiMemLock) { \
  	CmiMemLock(); \
	rank_holding_CmiMemLock=myRank; \
	code; \
	rank_holding_CmiMemLock=-1; \
	CmiMemUnlock(); \
  } \
  else /* I'm already holding the memLock (reentrancy) */ { \
  	code; \
  }

void *malloc(size_t size)
{
  void *result;
  MEM_LOCK_AROUND( result = meta_malloc(size); )
  if (result==NULL) CmiOutOfMemory(size);
  return result;
}

void free(void *mem)
{
  MEM_LOCK_AROUND( meta_free(mem); )
}

void *calloc(size_t nelem, size_t size)
{
  void *result;
  MEM_LOCK_AROUND( result = meta_calloc(nelem, size); )
  if (result==NULL) CmiOutOfMemory(size);
  return result;
}

void cfree(void *mem)
{
  MEM_LOCK_AROUND( meta_cfree(mem); )
}

void *realloc(void *mem, size_t size)
{
  void *result;
  MEM_LOCK_AROUND( result = meta_realloc(mem, size); )
  return result;
}

void *memalign(size_t align, size_t size)
{
  void *result;
  MEM_LOCK_AROUND( result = meta_memalign(align, size); )
  if (result==NULL) CmiOutOfMemory(align*size);
  return result;
}

void *valloc(size_t size)
{
  void *result;
  MEM_LOCK_AROUND( result = meta_valloc(size); )
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
  REENTRANT_MEM_LOCK_AROUND( result = meta_malloc(size); )
  return result;
}

void free_reentrant(void *mem)
{
  REENTRANT_MEM_LOCK_AROUND( meta_free(mem); )
}

/** Return number of bytes currently allocated, if possible. */
CMK_TYPEDEF_UINT8 CmiMemoryUsage()
{
  return _memory_allocated;
}

/** Return number of maximum number of bytes allocated since the last call to CmiResetMaxMemory(), if possible. */
CMK_TYPEDEF_UINT8 CmiMaxMemoryUsage()
{
  return _memory_allocated_max;
}

/** Reset the mechanism that records the highest seen (high watermark) memory usage. */
void CmiResetMaxMemory() {
  _memory_allocated_max=_memory_allocated;
}

CMK_TYPEDEF_UINT8 CmiMinMemoryUsage()
{
  return _memory_allocated_min;
}

void CmiResetMinMemory() {
  _memory_allocated_min=_memory_allocated;
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
CmiIsomallocBlockList *CmiIsomallocBlockListCurrent(){
	return NULL;
}
void CmiEnableIsomalloc() {}
void CmiDisableIsomalloc() {}
#endif

#ifndef CMI_MEMORY_ROUTINES
void CmiMemoryMark(void) {}
void CmiMemoryMarkBlock(void *blk) {}
void CmiMemorySweep(const char *where) {}
void CmiMemoryCheck(void) {}
#endif

void memory_preallocate_hack()
{
#if CMK_MEMORY_PREALLOCATE_HACK
  /* Work around problems with brk() on some systems (e.g., Blue Gene/L)
     by grabbing a bunch of memory from the OS (which calls brk()),
     then releasing the memory back to malloc(), except for one block
     at the end, which is used to prevent malloc from moving brk() back down.
  */
#define MEMORY_PREALLOCATE_MAX 4096
  void *ptrs[MEMORY_PREALLOCATE_MAX];
  int i,len=0;
  for (i=0;i<MEMORY_PREALLOCATE_MAX;i++) {
    ptrs[i] = mm_malloc(1024*1024);
    if (ptrs[i]==NULL) break;
    else len=i+1; /* this allocation worked */
  }
  /* we now own all the memory-- release all but the last meg. */
  /* printf("CMK_MEMORY_PREALLOCATE_HACK claimed %d megs\n",len); */
  for (i=len-2;i>=0;i--) {
    mm_free(ptrs[i]);
  }
#endif
}

void CmiOutOfMemoryInit(void) {
  if (CmiMyRank() == 0) {
#if CMK_MEMORY_PREALLOCATE_HACK
  memory_preallocate_hack();
#endif
  MEM_LOCK_AROUND( memory_lifeRaft=(char *)mm_malloc(65536/2); )
  }
}

#ifndef CMK_MEMORY_BUILD_CHARMDEBUG
/* declare the cpd_memory routines */
void CpdSetInitializeMemory(int v) { }
size_t  cpd_memory_length(void *lenParam) { return 0; }
void cpd_memory_pup(void *itemParam,pup_er p,CpdListItemsRequest *req) { }
void cpd_memory_leak(void *itemParam,pup_er p,CpdListItemsRequest *req) { }
void check_memory_leaks(LeakSearchInfo* i) { }
size_t  cpd_memory_getLength(void *lenParam) { return 0; }
void cpd_memory_get(void *itemParam,pup_er p,CpdListItemsRequest *req) { }
void CpdMemoryMarkClean(char *msg) { }
/* routine used by CthMemory{Protect,Unprotect} to specify that some region of
   memory has been protected */
void setProtection(char *mem, char *ptr, int len, int flag) { }
/* Routines used to specify how the memory will the used */
#ifdef setMemoryTypeChare
#undef setMemoryTypeChare
#endif
void setMemoryTypeChare(void *ptr) { }
#ifdef setMemoryTypeMessage
#undef setMemoryTypeMessage
#endif
void setMemoryTypeMessage(void *ptr) { }
void CpdSystemEnter() { }
void CpdSystemExit() { }

void CpdResetMemory() { }
void CpdCheckMemory() { }

int get_memory_allocated_user_total() { return 0; }
void * MemoryToSlot(void *ptr) { return NULL; }
int Slot_ChareOwner(void *s) { return 0; }
int Slot_AllocatedSize(void *s) { return 0; }
int Slot_StackTrace(void *s, void ***stack) { return 0; }
#ifdef setMemoryChareIDFromPtr
#undef setMemoryChareIDFromPtr
#endif
int setMemoryChareIDFromPtr(void *ptr) { return 0; }
#ifdef setMemoryChareID
#undef setMemoryChareID
#endif
void setMemoryChareID(int id) { }
#ifdef setMemoryOwnedBy
#undef setMemoryOwnedBy
#endif
void setMemoryOwnedBy(void *ptr, int id) { }

#endif


/* Genearl functions for malloc'ing aligned buffers */

/* CmiMallocAligned: Allocates a memory buffer with the given byte alignment.
     Additionally, the length of the returned buffer is also a multiple of
     alignment and >= size.
   NOTE: Memory allocated with CmiMallocAligned MUST be free'd with CmiFreeAligned()
*/
void* CmiMallocAligned(const size_t size, const unsigned int alignment) {

  void* rtn = NULL;
  int tailPadding;
  unsigned short offset = 0;

  /* Verify the pararmeters */
  if (size <= 0 || alignment <= 0) return NULL;

  /* Malloc memory of size equal to size + alignment + (alignment - (size % alignment)).  The
   *   last term 'alignment - (size % alignment)' ensures that there is enough tail padding
   *   so a DMA can be performed based on the alignment.  (I.e. - Start and end the memory
   *   region retured on an alignment boundry specified.)
   * NOTE: Since we need a byte long header, even if we "get lucky" and the malloc
   *   returns a pointer with the given alignment, we need to put in a byte
   *   preamble anyway.
   */
  tailPadding = alignment - (size % alignment);
  if (tailPadding == alignment)
    tailPadding = 0;

  /* Allocate the memory */
  rtn = malloc(size + alignment + tailPadding);

  /* Calculate the offset into the returned memory chunk that has the required alignment */
  offset = (char)(((size_t)rtn) % alignment);
  offset = alignment - offset;
  if (offset == 0) offset = alignment;

  /* Write the offset into the byte before the address to be returned */
  *((char*)rtn + offset - 1) = offset;

  /* Return the address with offset */
  /* cppcheck-suppress memleak */
  return (void*)((char*)rtn + offset);
}

void CmiFreeAligned(void* ptr) {

  char offset;

  /* Verify the parameter */
  if (ptr == NULL) return;

  /* Read the offset (byte before ptr) */
  offset = *((char*)ptr - 1);

  /* Free the memory */
  free ((void*)((char*)ptr - offset));
}



/** @} */
