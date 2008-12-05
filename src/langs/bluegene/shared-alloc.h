// Shared allocation for bulk data used in emulated processes
// Author: Phil Miller
//
// These allocation functions should only be uesd for plain old data (POD)
// types.
// Currently, they don't worry about alignment at all. This will probably 
// cause errors on some architectures (POWER, I'm looking at you)
#ifdef __cplusplus
extern "C" {
#endif

  // At program point i, allocate sz bytes, or get the address of a previous
  // allocation at i
  void *shalloc(size_t sz, int i);
  // Free p from the perspective of the current process. The memory will be 
  // released when all processes that allocated at i call shfree
  void shfree(void *p, int i);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
void* operator new(size_t sz, int i);
void operator delete(void *p, int i);
void* operator new[](size_t sz, int i);
void operator delete[](void *p, int i);
#endif
