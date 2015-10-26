
#ifndef _CONV_MACH_H
#define _CONV_MACH_H

#define CMK_USE_IBVERBS				1
#define CMK_NETPOLL         1

#define CMK_AMD64					   1

#define CMK_ASYNC_NOT_NEEDED                               0
#define CMK_ASYNC_USE_FIOASYNC_AND_FIOSETOWN               0
#define CMK_ASYNC_USE_FIOASYNC_AND_SIOCSPGRP               0
#define CMK_ASYNC_USE_FIOSSAIOSTAT_AND_FIOSSAIOOWN         0
#define CMK_ASYNC_USE_F_SETFL_AND_F_SETOWN                 1

#define CMK_DLL_CC  "g++ -shared -O3 -o "

#define CMK_GETPAGESIZE_AVAILABLE                          1

#define CMK_MALLOC_USE_GNU_MALLOC                          0
#define CMK_MALLOC_USE_OS_BUILTIN                          1

#define CMK_MEMORY_PAGESIZE                                8192
#define CMK_MEMORY_PROTECTABLE                             0


#define CMK_SSH_IS_A_COMMAND                               1
#define CMK_SSH_NOT_NEEDED                                 0

#define CMK_SHARED_VARS_UNAVAILABLE                        1
#define CMK_SHARED_VARS_UNIPROCESSOR                       0

#define CMK_THREADS_USE_CONTEXT                            1
#define CMK_THREADS_USE_PTHREADS                           0
#define CMK_THREADS_ARE_WIN32_FIBERS                       0

#define CMK_SIGNAL_NOT_NEEDED                              0
#define CMK_SIGNAL_USE_SIGACTION                           0
#define CMK_SIGNAL_USE_SIGACTION_WITH_RESTART              1

#define CMK_THREADS_REQUIRE_NO_CPV                         0
#define CMK_THREADS_COPY_STACK                             0

#define CMK_TIMER_USE_RDTSC                                0
#define CMK_TIMER_USE_GETRUSAGE                            1
#define CMK_TIMER_USE_SPECIAL                              0
#define CMK_TIMER_USE_TIMES                                0


#define CMK_64BIT    1

#define CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT                   0
#define CMK_WHEN_PROCESSOR_IDLE_USLEEP                     1


#define CMK_DEBUG_MODE					   0 
#define CMK_WEB_MODE                                       1

#define CMK_LBDB_ON					   1

/*
#undef CMK_IMMEDIATE_MSG
#define CMK_IMMEDIATE_MSG       0
*/

#undef CMK_DIRECT                 
#define CMK_DIRECT                                        1

#undef CMK_DISABLE_SYNC
#define CMK_DISABLE_SYNC       1

#undef CMK_IBVERBS_FAST_START
#define CMK_IBVERBS_FAST_START                         0

#endif
