
#undef CMK_WEB_MODE
#undef CMK_CCS_AVAILABLE
#define CMK_WEB_MODE 					0
#define CMK_CCS_AVAILABLE				0

#undef CMK_TIMER_USE_RDTSC
#undef CMK_TIMER_USE_GETRUSAGE
#define CMK_TIMER_USE_RDTSC				0
#define CMK_TIMER_USE_GETRUSAGE				1

#undef CMK_MALLOC_USE_GNU_MALLOC
#undef CMK_MALLOC_USE_OS_BUILTIN
#define CMK_MALLOC_USE_GNU_MALLOC                          0
#define CMK_MALLOC_USE_OS_BUILTIN                          0
#define CMK_MALLOC_USE_GNUOLD_MALLOC		           1
