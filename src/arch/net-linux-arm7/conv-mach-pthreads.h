
#undef CMK_MALLOC_USE_GNU_MALLOC
#undef CMK_MALLOC_USE_OS_BUILTIN
#undef CMK_MALLOC_USE_GNUOLD_MALLOC
#define CMK_MALLOC_USE_OS_BUILTIN                          1
                                                                                
#undef CMK_THREADS_USE_CONTEXT
#undef CMK_THREADS_USE_PTHREADS
#define CMK_THREADS_USE_PTHREADS                           1

