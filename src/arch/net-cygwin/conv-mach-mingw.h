
#undef CMK_MALLOC_USE_OS_BUILTIN
#undef CMK_MALLOC_USE_GNUOLD_MALLOC
#define CMK_MALLOC_USE_OS_BUILTIN                          1
#define CMK_MALLOC_USE_GNUOLD_MALLOC                       0

#undef CMK_SIGNAL_NOT_NEEDED
#undef CMK_SIGNAL_USE_SIGACTION
#define CMK_SIGNAL_NOT_NEEDED                              1
#define CMK_SIGNAL_USE_SIGACTION                           0

#undef CMK_THREADS_USE_JCONTEXT 
#undef CMK_THREADS_ARE_WIN32_FIBERS
#define CMK_THREADS_USE_JCONTEXT                           0
#define CMK_THREADS_ARE_WIN32_FIBERS                       1

#undef CMK_SHARED_VARS_UNAVAILABLE
#undef CMK_SHARED_VARS_NT_THREADS
#define CMK_SHARED_VARS_UNAVAILABLE                        0
#define CMK_SHARED_VARS_NT_THREADS                         1

#undef CMK_SSH_KILL
#undef CMK_SSH_NOT_NEEDED
#define CMK_SSH_KILL                                       0
#define CMK_SSH_NOT_NEEDED                                 1

