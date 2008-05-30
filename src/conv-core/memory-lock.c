
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

static void meta_init(char **argv)
{
/*   CmiMemoryIs_flag|=CMI_MEMORY_IS_OSLOCK;   */
}

void *meta_malloc(size_t size)
{
  void *result;
  MEM_LOCK_AROUND( result = mm_malloc(size); )
  if (result==NULL) CmiOutOfMemory(size);
  return result;
}

void meta_free(void *mem)
{
  MEM_LOCK_AROUND( mm_free(mem); )
}

void *meta_calloc(size_t nelem, size_t size)
{
  void *result;
  MEM_LOCK_AROUND( result = mm_calloc(nelem, size); )
  if (result==NULL) CmiOutOfMemory(size);
  return result;
}

void meta_cfree(void *mem)
{
  MEM_LOCK_AROUND( mm_cfree(mem); )
}

void *meta_realloc(void *mem, size_t size)
{
  void *result;
  MEM_LOCK_AROUND( result = mm_realloc(mem, size); )
  return result;
}

void *meta_memalign(size_t align, size_t size)
{
  void *result;
  MEM_LOCK_AROUND( result = mm_memalign(align, size); )
  if (result==NULL) CmiOutOfMemory(align*size);
  return result;    
}

void *meta_valloc(size_t size)
{
  void *result;
  MEM_LOCK_AROUND( result = mm_valloc(size); )
  if (result==NULL) CmiOutOfMemory(size);
  return result;
}
