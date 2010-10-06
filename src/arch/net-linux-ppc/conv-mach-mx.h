

#undef CMK_USE_MX
#define CMK_USE_MX                                         1

#undef CMK_NETPOLL
#define CMK_NETPOLL                                        1

//#undef CMK_MULTICAST_LIST_USE_COMMON_CODE
//#define CMK_MULTICAST_LIST_USE_COMMON_CODE		   0

#undef CMK_MALLOC_USE_GNU_MALLOC
#undef CMK_MALLOC_USE_OS_BUILTIN
#define CMK_MALLOC_USE_GNU_MALLOC                          0
#define CMK_MALLOC_USE_OS_BUILTIN                          1

#undef CMK_BARRIER_USE_COMMON_CODE
#define CMK_BARRIER_USE_COMMON_CODE                        0

