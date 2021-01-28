
static void meta_init(char **argv)
{
/*   if (CmiMyRank()==0) CmiMemoryIs_flag|=CMI_MEMORY_IS_OSLOCK;   */
}

void *meta_malloc(size_t size)
{
  void *result;
  MEM_LOCK_AROUND( result = mm_impl_malloc(size); )
  if (result==NULL) CmiOutOfMemory(size);
  return result;
}

void meta_free(void *mem)
{
  MEM_LOCK_AROUND( mm_impl_free(mem); )
}

void *meta_calloc(size_t nelem, size_t size)
{
  void *result;
  MEM_LOCK_AROUND( result = mm_impl_calloc(nelem, size); )
  if (result==NULL) CmiOutOfMemory(size);
  return result;
}

void meta_cfree(void *mem)
{
  MEM_LOCK_AROUND( mm_impl_cfree(mem); )
}

void *meta_realloc(void *mem, size_t size)
{
  void *result;
  MEM_LOCK_AROUND( result = mm_impl_realloc(mem, size); )
  return result;
}

void *meta_memalign(size_t align, size_t size)
{
  void *result;
  MEM_LOCK_AROUND( result = mm_impl_memalign(align, size); )
  if (result==NULL) CmiOutOfMemory(align*size);
  return result;    
}

int meta_posix_memalign(void **outptr, size_t align, size_t size)
{
  int result;
  MEM_LOCK_AROUND( result = mm_impl_posix_memalign(outptr, align, size); )
  if (result!=0) CmiOutOfMemory(align*size);
  return result;
}

void *meta_aligned_alloc(size_t align, size_t size)
{
  void *result;
  MEM_LOCK_AROUND( result = mm_impl_aligned_alloc(align, size); )
  if (result==NULL) CmiOutOfMemory(align*size);
  return result;
}

void *meta_valloc(size_t size)
{
  void *result;
  MEM_LOCK_AROUND( result = mm_impl_valloc(size); )
  if (result==NULL) CmiOutOfMemory(size);
  return result;
}

void *meta_pvalloc(size_t size)
{
  void *result;
  MEM_LOCK_AROUND( result = mm_impl_pvalloc(size); )
  if (result==NULL) CmiOutOfMemory(size);
  return result;
}
