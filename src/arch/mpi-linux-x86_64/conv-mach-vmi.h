
#undef CMK_MALLOC_USE_GNU_MALLOC
#undef CMK_MALLOC_USE_OS_BUILTIN
#define CMK_MALLOC_USE_GNU_MALLOC                          0
#define CMK_MALLOC_USE_OS_BUILTIN                          1

#undef CMK_THREADS_USE_CONTEXT
#define CMK_THREADS_USE_CONTEXT                            1

#define CMK_DISABLE_SYNC				   1

#define CMK_MPI_VMI					   1
