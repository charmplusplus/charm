#ifndef _CONV_MACH_H
#define _CONV_MACH_H

#define CMK_CRAYXT                                         1

#define XT4_TOPOLOGY                                       0

#define XT5_TOPOLOGY                                       0

#define CMK_MALLOC_USE_GNU_MALLOC                          0
#define CMK_MALLOC_USE_OS_BUILTIN                          1

#define CMK_ARENA_MALLOC                                   1

#define CMK_GETPAGESIZE_AVAILABLE                          1
#define CMK_MEMORY_PAGESIZE                                8192
#define CMK_MEMORY_PROTECTABLE                             0

#define CMI_IO_BUFFER_EXPLICIT                             1
#define CMI_IO_FLUSH_USER                                  1

#define CMK_NODE_QUEUE_AVAILABLE                           0

#define CMK_RSH_IS_A_COMMAND                               0
#define CMK_RSH_NOT_NEEDED                                 1
#define CMK_RSH_USE_REMSH                                  0

#define CMK_SHARED_VARS_EXEMPLAR                           0
#define CMK_SHARED_VARS_UNAVAILABLE                        1
#define CMK_SHARED_VARS_UNIPROCESSOR                       0

#define CMK_SIGNAL_NOT_NEEDED                              0
#define CMK_SIGNAL_USE_SIGACTION                           0
#define CMK_SIGNAL_USE_SIGACTION_WITH_RESTART              1

#define CMK_THREADS_REQUIRE_NO_CPV                         0
#define CMK_THREADS_COPY_STACK                             0

#define CMK_TIMER_USE_GETRUSAGE                            1
#define CMK_TIMER_USE_SPECIAL                              0
#define CMK_TIMER_USE_TIMES                                0

#define CMK_TYPEDEF_INT2 short
#define CMK_TYPEDEF_INT4 int
#define CMK_TYPEDEF_INT8 long
#define CMK_TYPEDEF_UINT2 unsigned short
#define CMK_TYPEDEF_UINT4 unsigned int
#define CMK_TYPEDEF_UINT8 unsigned long
#define CMK_TYPEDEF_FLOAT4 float
#define CMK_TYPEDEF_FLOAT8 double

#define CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT                   1
#define CMK_WHEN_PROCESSOR_IDLE_USLEEP                     0

#define CMK_DEBUG_MODE                                     0
#define CMK_WEB_MODE                                       1

#define CMK_LBDB_ON					   1

#define CMK_SHMEM_H					   <mpp/shmem.h>
#define CMK_SHMEM_INIT					   shmem_init()
#define CMK_SHMEM_LOCK					   1

#endif

