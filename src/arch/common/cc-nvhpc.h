#undef CMK_DLL_CC

/* nvc cannot compile RDTSC timer */
#if CMK_TIMER_USE_RDTSC
# undef CMK_TIMER_USE_GETRUSAGE
# undef CMK_TIMER_USE_RDTSC
# define CMK_TIMER_USE_GETRUSAGE                            1
# define CMK_TIMER_USE_RDTSC                                0
#endif
