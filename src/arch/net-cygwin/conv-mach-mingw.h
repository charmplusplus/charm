
#undef CMK_MALLOC_USE_OS_BUILTIN
#undef CMK_MALLOC_USE_GNUOLD_MALLOC
#define CMK_MALLOC_USE_OS_BUILTIN                          1
#define CMK_MALLOC_USE_GNUOLD_MALLOC                       0

#undef CMK_MEMORY_PROTECTABLE
#define CMK_MEMORY_PROTECTABLE                             0

#undef CMK_RSH_KILL
#undef CMK_RSH_NOT_NEEDED
#define CMK_RSH_KILL                                       0
#define CMK_RSH_NOT_NEEDED                                 1

#undef CMK_SIGNAL_NOT_NEEDED
#undef CMK_SIGNAL_USE_SIGACTION
#define CMK_SIGNAL_NOT_NEEDED                              1
#define CMK_SIGNAL_USE_SIGACTION                           0
