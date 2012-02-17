#define CMK_SMP                                            1

#undef CMK_NODE_QUEUE_AVAILABLE
#define CMK_NODE_QUEUE_AVAILABLE                           1

#undef CMK_SHARED_VARS_UNAVAILABLE
#undef CMK_SHARED_VARS_POSIX_THREADS_SMP
#define CMK_SHARED_VARS_UNAVAILABLE                        0
#define CMK_SHARED_VARS_POSIX_THREADS_SMP                  1

#undef CMK_THREADS_USE_CONTEXT
#define CMK_THREADS_USE_CONTEXT                            1


#define CMK_SMP_TRACE_COMMTHREAD                           0
