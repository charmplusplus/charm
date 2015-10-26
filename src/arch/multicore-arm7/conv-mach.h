
#ifndef _CONV_MACH_H
#define _CONV_MACH_H

#define CMK_NETPOLL         1
#define CMK_MULTICORE                                      1

#define CMK_ARM					  1 

#undef CMK_IMMEDIATE_MSG
#define CMK_IMMEDIATE_MSG                                  0

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

#define CMK_SHARED_VARS_UNAVAILABLE                        0
#define CMK_SHARED_VARS_UNIPROCESSOR                       0
#define CMK_SHARED_VARS_POSIX_THREADS_SMP                  1

#define CMK_THREADS_USE_CONTEXT                            0 
#define CMK_THREADS_USE_JCONTEXT                           0
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

#if CMK_HAS_STDINT_H
#include <stdint.h>
#define CMK_TYPEDEF_INT2 int16_t
#define CMK_TYPEDEF_INT4 int32_t
#define CMK_TYPEDEF_INT8 int64_t
#define CMK_TYPEDEF_UINT2 uint16_t
#define CMK_TYPEDEF_UINT4 uint32_t
#define CMK_TYPEDEF_UINT8 uint64_t
#else
#define CMK_TYPEDEF_INT2 short
#define CMK_TYPEDEF_INT4 int
#define CMK_TYPEDEF_UINT2 unsigned short
#define CMK_TYPEDEF_UINT4 unsigned int
#if CMK_LONG_LONG_DEFINED
#define CMK_TYPEDEF_INT8 long long
#define CMK_TYPEDEF_UINT8 unsigned long long
#else
#error "No definition for a 64-bit integer"
#endif
#endif


#define CMK_64BIT   0 
#define CMK_32BIT   1
#define CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT                   0
#define CMK_WHEN_PROCESSOR_IDLE_USLEEP                     1


#define CMK_DEBUG_MODE					   0 
#define CMK_WEB_MODE                                       1

#define CMK_LBDB_ON					   1


#endif



#define CMK_SMP						   1
