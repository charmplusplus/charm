#define CMK_SMP						   1


#undef CMK_SHARED_VARS_UNAVAILABLE
#undef CMK_SHARED_VARS_POSIX_THREADS_SMP
#define CMK_SHARED_VARS_UNAVAILABLE                        0
#define CMK_SHARED_VARS_POSIX_THREADS_SMP                  1

#undef CMK_USE_POLL
#define CMK_USE_POLL                                       1
#undef CMK_USE_KQUEUE
#define CMK_USE_KQUEUE                                     0

#define CMK_NOT_USE_TLS_THREAD                             1
