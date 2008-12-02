// Shared allocation for bulk data used in emulated processes
// Author: Phil Miller

#ifdef __cplusplus
extern "C" {
#endif

  // At program point i, allocate sz bytes, or get the address of a previous
  // allocation at i
  void *shalloc(int i, size_t sz);
  // Free p from the perspective of the current process. The memory will be 
  // released when all processes that allocated at i call shfree
  void shfree(int i, void *p);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
// Define something like shnew and shdelete here?

#endif
