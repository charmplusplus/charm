/******************************************************************************

A migratable memory allocator.

FIXME: isomalloc is threadsafe, so the isomallocs *don't* need to
be wrapped in CmiMemLock.  (Doesn't hurt, tho')

*****************************************************************************/

#define CMK_ISOMALLOC_EXCLUDE_FORTRAN_CALLS   0

#if ! CMK_MEMORY_BUILD_OS
/* Use Gnumalloc as meta-meta malloc fallbacks (mm_*) */
#include "memory-gnu.c"
#endif

#include "memory-isomalloc.h"

/*The current allocation arena */
CpvStaticDeclare(CmiIsomallocBlockList *,isomalloc_blocklist);
CpvStaticDeclare(CmiIsomallocBlockList *,pushed_blocklist);

#define ISOMALLOC_PUSH \
	CmiIsomallocBlockList *pushed_blocklist=CpvAccess(isomalloc_blocklist);\
	CpvAccess(isomalloc_blocklist)=NULL;\
	rank_holding_CmiMemLock=CmiMyRank();\

#define ISOMALLOC_POP \
	CpvAccess(isomalloc_blocklist)=pushed_blocklist;\
	rank_holding_CmiMemLock=-1;\

/* temporarily disable/enable isomalloc. Note the following two fucntions
 * must be used in pair, and no suspend of thread is allowed in between
 * */
void CmiDisableIsomalloc()
{
	CpvAccess(pushed_blocklist)=CpvAccess(isomalloc_blocklist);
	CpvAccess(isomalloc_blocklist)=NULL;
	rank_holding_CmiMemLock=CmiMyRank();
}

void CmiEnableIsomalloc()
{
	CpvAccess(isomalloc_blocklist)=CpvAccess(pushed_blocklist);
	rank_holding_CmiMemLock=-1;
}

#if CMK_HAS_TLS_VARIABLES
/**
 * make sure isomalloc is only called in pthreads that is spawned by Charm++.
 * It is not safe to call isomalloc in system spawned pthreads for example
 * mpich pthreads, or aio pthreads.
 * Use the following TLS variable to distinguish those pthreads.
 * when set to 1, the current pthreads is allowed to call isomalloc.
 */
static __thread int isomalloc_thread = 0;
#else
#if BIGSIM_OUT_OF_CORE && BIGSIM_OOC_PREFETCH
#error TLS support is required for bigsim out-of-core prefetch optimization
#endif
#endif

static int meta_inited = 0;

static void meta_init(char **argv)
{
   CmiMemoryIs_flag|=CMI_MEMORY_IS_ISOMALLOC;
   CpvInitialize(CmiIsomallocBlockList *,isomalloc_blocklist);
   CpvInitialize(CmiIsomallocBlockList *,pushed_blocklist);
#if CMK_HAS_TLS_VARIABLES
   isomalloc_thread = 1;         /* isomalloc is allowed in this pthread */
#endif
   CmiNodeAllBarrier();
   meta_inited = 1;
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

static void *meta_valloc(size_t size)
{
	return meta_malloc(size);
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
	register CmiIsomallocBlockList **s=&CpvAccess(isomalloc_blocklist);
	CmiIsomallocBlockList *ret=*s;
	*s=l;
	return ret;
}

CmiIsomallocBlockList *CmiIsomallocBlockListCurrent(){
	return CpvAccess(isomalloc_blocklist);
}




