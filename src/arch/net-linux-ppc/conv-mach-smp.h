
#define CMK_SMP						   1


#undef CMK_SHARED_VARS_UNAVAILABLE
#undef CMK_SHARED_VARS_POSIX_THREADS_SMP
#define CMK_SHARED_VARS_UNAVAILABLE                        0
#define CMK_SHARED_VARS_POSIX_THREADS_SMP                  1

#undef CMK_MALLOC_USE_GNU_MALLOC
#undef CMK_MALLOC_USE_OS_BUILTIN
#define CMK_MALLOC_USE_GNU_MALLOC                          0
#define CMK_MALLOC_USE_OS_BUILTIN                          1

#define CMK_MMAP_PROBE                                     0

#define CMK_NOT_USE_TLS_THREAD                             1
#if ( defined(__xlc__) || defined(__xlC__) ) && CMK_POWER7
#warning "XLC compiler on Power7 does not support memory fence correctly, pcqueue lock is turned back on."
#define CMK_PCQUEUE_LOCK                                   1
#endif
