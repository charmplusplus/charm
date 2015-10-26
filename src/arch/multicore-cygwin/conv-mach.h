#ifndef _CONV_MACH_H
#define _CONV_MACH_H

#define CMK_MULTICORE                                      1
#define CMK_SMP                                            1

#undef CMK_IMMEDIATE_MSG
#define CMK_IMMEDIATE_MSG                                  0

#define CMK_ASYNC_NOT_NEEDED                               1
#define CMK_ASYNC_USE_FIOASYNC_AND_FIOSETOWN               0
#define CMK_ASYNC_USE_FIOASYNC_AND_SIOCSPGRP               0
#define CMK_ASYNC_USE_FIOSSAIOSTAT_AND_FIOSSAIOOWN         0
#define CMK_ASYNC_USE_F_SETFL_AND_F_SETOWN                 0

#define CMK_GETPAGESIZE_AVAILABLE                          0

#define CMK_MALLOC_USE_GNU_MALLOC                          0
#define CMK_MALLOC_USE_OS_BUILTIN                          1
#define CMK_MALLOC_USE_GNUOLD_MALLOC			   0

#define CMK_MEMORY_PAGESIZE                                4096
#define CMK_MEMORY_PROTECTABLE                             1


#define CMK_SSH_IS_A_COMMAND                               1
#define CMK_SSH_KILL                                       1
#define CMK_SSH_NOT_NEEDED                                 0

#define CMK_CONV_HOST_CSH_UNAVAILABLE			   1

#define CMK_SHARED_VARS_UNAVAILABLE                        0
#define CMK_SHARED_VARS_UNIPROCESSOR                       0
#define CMK_SHARED_VARS_NT_THREADS                         1
#define CMK_SHARED_VARS_POSIX_THREADS_SMP                  0

#define CMK_SIGNAL_NOT_NEEDED                              1
#define CMK_SIGNAL_USE_SIGACTION                           0
#define CMK_SIGNAL_USE_SIGACTION_WITH_RESTART              0

#define CMK_THREADS_REQUIRE_NO_CPV                         0
#define CMK_THREADS_COPY_STACK                             0
#define CMK_THREADS_ARE_WIN32_FIBERS                       0
#define CMK_THREADS_USE_CONTEXT                            0
#define CMK_THREADS_USE_JCONTEXT                           1

#define CMK_TIMER_USE_GETRUSAGE                            1
#define CMK_TIMER_USE_SPECIAL                              0
#define CMK_TIMER_USE_TIMES                                0
#define CMK_TIMER_USE_WIN32API                             0

#define CMK_TYPEDEF_INT2 short
#define CMK_TYPEDEF_INT4 int
#define CMK_TYPEDEF_INT8 long long
#define CMK_TYPEDEF_UINT2 unsigned short
#define CMK_TYPEDEF_UINT4 unsigned int
#define CMK_TYPEDEF_UINT8 unsigned long long

#define CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT                   0
#define CMK_WHEN_PROCESSOR_IDLE_USLEEP                     1


#define CMK_DEBUG_MODE					   0
#define CMK_WEB_MODE                                       1

#define CMK_NO_ISO_MALLOC				   1

#define CMK_LBDB_ON					   1

#endif

