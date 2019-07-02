/******************************************************************************

A migratable memory allocator.

NOTE: isomalloc is threadsafe, so the isomallocs are not wrapped in CmiMemLock.

*****************************************************************************/

#define CMK_ISOMALLOC_EXCLUDE_FORTRAN_CALLS   0

#if ! CMK_MEMORY_BUILD_OS
/* Use Gnumalloc as meta-meta malloc fallbacks (mm_*) */
#include "memory-gnu.C"
#endif

#include "memory-isomalloc.h"

/*The current allocation arena */
CpvStaticDeclare(CmiIsomallocBlockList *,isomalloc_blocklist);
CpvStaticDeclare(CmiIsomallocBlockList *,pushed_blocklist);

#define ISOMALLOC_PUSH \
	CmiIsomallocBlockList *pushed_blocklist=CpvAccess(isomalloc_blocklist);\
	CpvAccess(isomalloc_blocklist)=NULL;\

#define ISOMALLOC_POP \
	CpvAccess(isomalloc_blocklist)=pushed_blocklist;\

/* temporarily disable/enable isomalloc. Note the following two fucntions
 * must be used in pair, and no suspend of thread is allowed in between
 * */
void CmiDisableIsomalloc()
{
	CpvAccess(pushed_blocklist)=CpvAccess(isomalloc_blocklist);
	CpvAccess(isomalloc_blocklist)=NULL;
}

void CmiEnableIsomalloc()
{
	CpvAccess(isomalloc_blocklist)=CpvAccess(pushed_blocklist);
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
#else
#if BIGSIM_OUT_OF_CORE && BIGSIM_OOC_PREFETCH
#error TLS support is required for bigsim out-of-core prefetch optimization
#endif
#endif

static int meta_inited = 0;
extern int _sync_iso;
extern int _sync_iso_warned;

static void meta_init(char **argv)
{
   if (CmiMyRank()==0) CmiMemoryIs_flag|=CMI_MEMORY_IS_ISOMALLOC;
   CpvInitialize(CmiIsomallocBlockList *,isomalloc_blocklist);
   CpvInitialize(CmiIsomallocBlockList *,pushed_blocklist);
   CpvAccess(isomalloc_blocklist) = NULL;
   CpvAccess(pushed_blocklist) = NULL;
#if CMK_HAS_TLS_VARIABLES
   isomalloc_thread = 1;         /* isomalloc is allowed in this pthread */
#endif
   if (CmiMyRank()==0) meta_inited = 1;
#if CMK_SMP
    if (CmiMyPe()==0 && _sync_iso == 0 && _sync_iso_warned == 0) {
        _sync_iso_warned = 1;
        printf("Warning> Using Isomalloc in SMP mode, you may need to run with '+isomalloc_sync'.\n");
    }
#endif
}

static void *meta_malloc(size_t size)
{
	void *ret=NULL;
#if CMK_HAS_TLS_VARIABLES
        int _isomalloc_thread = isomalloc_thread;
        if (CmiThreadIs(CMI_THREAD_IS_TLS)) _isomalloc_thread = 1;
#endif
	if (meta_inited && CpvInitialized(isomalloc_blocklist) && CpvAccess(isomalloc_blocklist)
#if CMK_HAS_TLS_VARIABLES
             && _isomalloc_thread
#endif
           )
	{ /*Isomalloc a new block and link it in*/
		ISOMALLOC_PUSH /*Disable isomalloc while inside isomalloc*/
#if CMK_ISOMALLOC_EXCLUDE_FORTRAN_CALLS
		if (CmiIsFortranLibraryCall()==1) {
		  ret=mm_malloc(size);
		}
		else
#endif
		ret=CmiIsomallocBlockListMalloc(pushed_blocklist,size);
		ISOMALLOC_POP
	}
	else /*Just use regular malloc*/
		ret=mm_malloc(size);
	return ret;
}

static void meta_free(void *mem)
{	
	if (mem != NULL && CmiIsomallocInRange(mem)) 
	{ /*Unlink this slot and isofree*/
		ISOMALLOC_PUSH
		CmiIsomallocBlockListFree(mem);
		ISOMALLOC_POP
	}
	else /*Just use regular malloc*/
		mm_free(mem);
}

static void *meta_calloc(size_t nelem, size_t size)
{
	void *ret=meta_malloc(nelem*size);
	if (ret != NULL) memset(ret,0,nelem*size);
	return ret;
}

static void meta_cfree(void *mem)
{
	meta_free(mem);
}

static void *meta_realloc(void *oldBuffer, size_t newSize)
{
	void *newBuffer;
	/*Just forget it for regular malloc's:*/
	if (!CmiIsomallocInRange(oldBuffer))
		return mm_realloc(oldBuffer,newSize);
	
	newBuffer = meta_malloc(newSize);
	if ( newBuffer && oldBuffer ) {
		/*Must preserve old buffer contents, so we need the size of the
		  buffer.  SILLY HACK: muck with internals of blocklist header.*/
		size_t size=CmiIsomallocLength(((CmiIsomallocBlockList *)oldBuffer)-1)-
			sizeof(CmiIsomallocBlockList);
		if (size>newSize) size=newSize;
		if (size > 0)
			memcpy(newBuffer, oldBuffer, size);
	}
	if (oldBuffer)
		meta_free(oldBuffer);
	return newBuffer;
}

static void *meta_memalign(size_t align, size_t size)
{
	void *ret=NULL;
	if (CpvInitialized(isomalloc_blocklist) && CpvAccess(isomalloc_blocklist)) 
	{ /*Isomalloc a new block and link it in*/
		ISOMALLOC_PUSH /*Disable isomalloc while inside isomalloc*/
#if CMK_ISOMALLOC_EXCLUDE_FORTRAN_CALLS
		if (CmiIsFortranLibraryCall()==1) {
		  ret=mm_memalign(align, size);
		}
		else
#endif
		  ret=CmiIsomallocBlockListMallocAlign(pushed_blocklist,align,size);
		ISOMALLOC_POP
	}
	else /*Just use regular memalign*/
		ret=mm_memalign(align, size);
	return ret;
}

static int meta_posix_memalign(void **outptr, size_t align, size_t size)
{
	int ret = 0;
	if (CpvInitialized(isomalloc_blocklist) && CpvAccess(isomalloc_blocklist))
	{ /*Isomalloc a new block and link it in*/
		ISOMALLOC_PUSH /*Disable isomalloc while inside isomalloc*/
#if CMK_ISOMALLOC_EXCLUDE_FORTRAN_CALLS
		if (CmiIsFortranLibraryCall()==1) {
		  ret=mm_posix_memalign(outptr, align, size);
		}
		else
#endif
		  *outptr = CmiIsomallocBlockListMallocAlign(pushed_blocklist,align,size);
		ISOMALLOC_POP
	}
	else /*Just use regular posix_memalign*/
		ret=mm_posix_memalign(outptr, align, size);
	return ret;
}

static void *meta_aligned_alloc(size_t align, size_t size)
{
	void *ret=NULL;
	if (CpvInitialized(isomalloc_blocklist) && CpvAccess(isomalloc_blocklist))
	{ /*Isomalloc a new block and link it in*/
		ISOMALLOC_PUSH /*Disable isomalloc while inside isomalloc*/
#if CMK_ISOMALLOC_EXCLUDE_FORTRAN_CALLS
		if (CmiIsFortranLibraryCall()==1) {
		  ret=mm_aligned_alloc(align, size);
		}
		else
#endif
		  ret=CmiIsomallocBlockListMallocAlign(pushed_blocklist,align,size);
		ISOMALLOC_POP
	}
	else /*Just use regular aligned_alloc*/
		ret=mm_aligned_alloc(align, size);
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
  void *result;
  CmiMemLock();
  result = mm_malloc(size);
  CmiMemUnlock();
  return result;
}

void free_nomigrate(void *mem)
{
  CmiMemLock();
  mm_free(mem);
  CmiMemUnlock();
}

#define CMK_MEMORY_HAS_ISOMALLOC

/*Make this blockList "active"-- the recipient of incoming
mallocs.  Returns the old blocklist.*/
CmiIsomallocBlockList *CmiIsomallocBlockListActivate(CmiIsomallocBlockList *l)
{
	CmiIsomallocBlockList **s=&CpvAccess(isomalloc_blocklist);
	CmiIsomallocBlockList *ret=*s;
	*s=l;
	return ret;
}

CmiIsomallocBlockList *CmiIsomallocBlockListCurrent(void){
	return CpvAccess(isomalloc_blocklist);
}




