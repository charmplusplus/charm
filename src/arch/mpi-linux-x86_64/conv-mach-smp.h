/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#define CMK_SMP						   1

#undef CMK_NODE_QUEUE_AVAILABLE
#define CMK_NODE_QUEUE_AVAILABLE                           1

#undef CMK_SHARED_VARS_UNAVAILABLE
#undef CMK_SHARED_VARS_POSIX_THREADS_SMP
#define CMK_SHARED_VARS_UNAVAILABLE                        0
#define CMK_SHARED_VARS_POSIX_THREADS_SMP                  1

#define CMK_LINUX_PTHREAD_HACK                             0

#undef CMK_SYNCHRONIZE_ON_TCP_CLOSE
#define CMK_SYNCHRONIZE_ON_TCP_CLOSE                       1

#undef CMK_TIMER_USE_GETRUSAGE
#undef CMK_TIMER_USE_SPECIAL
#define CMK_TIMER_USE_GETRUSAGE                            1
#define CMK_TIMER_USE_SPECIAL                              0

/*#define  CMK_USE_MFENCE                                    1 */
#define  CMK_PCQUEUE_LOCK                                  1 
#define CMK_USE_TLS_THREAD                                 1

#undef CMK_MALLOC_USE_GNU_MALLOC
#undef CMK_MALLOC_USE_OS_BUILTIN
#define CMK_MALLOC_USE_GNU_MALLOC                          1
#define CMK_MALLOC_USE_OS_BUILTIN                          0
