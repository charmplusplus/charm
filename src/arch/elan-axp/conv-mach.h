/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _CONV_MACH_H
#define _CONV_MACH_H


#define CMK_DLL_CC   "g++ -shared -O3 -o "

#define CMK_FIX_HP_CONNECT_BUG                             0

#define CMK_GETPAGESIZE_AVAILABLE                          0

#define CMK_MALLOC_USE_GNU_MALLOC                          0
#define CMK_MALLOC_USE_OS_BUILTIN                          1

#define CMK_MEMORY_PAGESIZE                                8192
#define CMK_MEMORY_PROTECTABLE                             1

#define CMK_NODE_QUEUE_AVAILABLE                           0

#define CMK_SHARED_VARS_EXEMPLAR                           0
#define CMK_SHARED_VARS_UNAVAILABLE                        1
#define CMK_SHARED_VARS_SUN_THREADS                        0
#define CMK_SHARED_VARS_UNIPROCESSOR                       0

#define CMK_SIGNAL_NOT_NEEDED                              0
#define CMK_SIGNAL_USE_SIGACTION                           1
#define CMK_SIGNAL_USE_SIGACTION_WITH_RESTART              0

#define CMK_STRERROR_IS_A_BUILTIN                          1
#define CMK_STRERROR_USE_SYS_ERRLIST                       0

#define CMK_STRINGS_USE_OWN_DECLARATIONS                   0
#define CMK_STRINGS_USE_STRINGS_H                          0
#define CMK_STRINGS_USE_STRING_H                           1

#define CMK_THREADS_REQUIRE_NO_CPV                         0
#define CMK_THREADS_COPY_STACK                             0

#define CMK_TIMER_USE_GETRUSAGE                            0
#define CMK_TIMER_USE_SPECIAL                              0
#define CMK_TIMER_USE_TIMES                                0

#define CMK_TYPEDEF_INT2 short
#define CMK_TYPEDEF_INT4 int
#define CMK_TYPEDEF_INT8 long
#define CMK_TYPEDEF_UINT2 unsigned short
#define CMK_TYPEDEF_UINT4 unsigned int
#define CMK_TYPEDEF_UINT8 unsigned long long
#define CMK_TYPEDEF_FLOAT4 float
#define CMK_TYPEDEF_FLOAT8 double

#define CMK_WAIT_NOT_NEEDED                                0
#define CMK_WAIT_USES_SYS_WAIT_H                           1
#define CMK_WAIT_USES_WAITFLAGS_H                          0

#define CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT                   0
#define CMK_WHEN_PROCESSOR_IDLE_USLEEP                     1


#define CMK_COMPILEMODE_ORIG                               0
#define CMK_COMPILEMODE_ANSI                               1

#define CMK_WEB_MODE                                       1
#define CMK_DEBUG_MODE                                     0

#define CMK_LBDB_ON					   1


#endif

