
#define CMK_SMP                                            1

#undef CMK_NODE_QUEUE_AVAILABLE
#define CMK_NODE_QUEUE_AVAILABLE                           1

#undef CMK_SHARED_VARS_UNAVAILABLE
#undef CMK_SHARED_VARS_POSIX_THREADS_SMP
#define CMK_SHARED_VARS_UNAVAILABLE                        0
#define CMK_SHARED_VARS_POSIX_THREADS_SMP                  1

#define CMK_MULTICORE                                      0

#define CMK_NOT_USE_TLS_THREAD                             0

#define CMK_PCQUEUE_LOCK                                   1
/*#define PCQUEUE_MULTIQUEUE                                 1*/

#define CMK_SMP_NO_COMMTHD                                 1

#define CMK_FAKE_SCHED_YIELD                               1

#define CMK_PPC_ATOMIC_QUEUE                               1
#define CMK_PPC_ATOMIC_MUTEX                               1

#define  CMK_PPC_ATOMIC_DEFAULT_IMPL                       1
