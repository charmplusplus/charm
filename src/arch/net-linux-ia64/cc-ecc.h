#undef  CMK_COMPILEMODE_ORIG
#undef  CMK_COMPILEMODE_ANSI
#define CMK_COMPILEMODE_ORIG                               0
#define CMK_COMPILEMODE_ANSI                               1

#undef CMK_TIMER_USE_GETRUSAGE
#undef CMK_TIMER_USE_RDTSC
#define CMK_TIMER_USE_GETRUSAGE                            1
#define CMK_TIMER_USE_RDTSC                                0
