#define CMK_SMP                                            1


#undef CMK_SHARED_VARS_UNAVAILABLE
#undef CMK_SHARED_VARS_POSIX_THREADS_SMP
#define CMK_SHARED_VARS_UNAVAILABLE                        0
#define CMK_SHARED_VARS_POSIX_THREADS_SMP                  1

/* Right now only comm thread (no multicore) and tls thread version with gcc works on Blue Gene*/
#define CMK_MULTICORE                                      0

#ifdef __GNUC__
#define CMK_NOT_USE_TLS_THREAD                             0
#else
#define CMK_NOT_USE_TLS_THREAD                             1
#endif

#define CMK_PCQUEUE_LOCK                                   1
