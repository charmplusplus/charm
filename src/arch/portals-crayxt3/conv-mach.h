/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _CONV_MACH_H
#define _CONV_MACH_H

#define CMK_CMIDELIVERS_USE_COMMON_CODE                    1

#define CMK_CMIPRINTF_IS_A_BUILTIN                         0

#define CMK_HANDLE_SIGUSR                                  1

#define CMK_MSG_HEADER_EXT_    CmiUInt2 rank, root, hdl,xhdl,info, stratid; unsigned char cksum, magic; CmiUInt2 redID;
#define CMK_MSG_HEADER_BASIC  CMK_MSG_HEADER_EXT
#define CMK_MSG_HEADER_EXT    { CMK_MSG_HEADER_EXT_ }
#define CMK_MSG_HEADER_BIGSIM_    { CMK_MSG_HEADER_EXT_ CMK_BIGSIM_FIELDS }

#define CMK_MULTICAST_GROUP_TYPE                struct { unsigned pe, id; }
#define CMK_MULTICAST_DEF_USE_COMMON_CODE                  1
#define CMK_MULTICAST_LIST_USE_COMMON_CODE                 1
#define CMK_MULTICAST_GROUP_USE_COMMON_CODE                1

#define CMK_RSH_IS_A_COMMAND                               0
#define CMK_RSH_NOT_NEEDED                                 1
#define CMK_RSH_USE_REMSH                                  0

#define CMK_SPANTREE_MAXSPAN                               4
#define CMK_SPANTREE_USE_COMMON_CODE                       1

#define CMK_VECTOR_SEND_USES_COMMON_CODE                   1

#define NODE_0_IS_CONVHOST                                 1

#define CMK_IMMEDIATE_MSG                                  0
#define CMK_MACHINE_PROGRESS_DEFINED                       1

#define XT3_ONLY_PCQUEUE_WORKAROUND                        1

#define CMK_CONVERSE_MPI                                   1

#define CMK_MEMORY_PREALLOCATE_HACK			   0

#define CMK_DEFAULT_MAIN_USES_COMMON_CODE                  1

#define CMK_IS_HETERO                                      0

#define CMK_MALLOC_USE_GNU_MALLOC                          0
#define CMK_MALLOC_USE_OS_BUILTIN                          1

#define CMI_IO_BUFFER_EXPLICIT                             1
#define CMI_IO_FLUSH_USER                                  1

#define CMK_GETPAGESIZE_AVAILABLE                          1
#define CMK_MEMORY_PAGESIZE                                8192
#define CMK_MEMORY_PROTECTABLE                             0

#define CMK_NODE_QUEUE_AVAILABLE                           0

#define CMK_SHARED_VARS_EXEMPLAR                           0
#define CMK_SHARED_VARS_UNAVAILABLE                        1
#define CMK_SHARED_VARS_UNIPROCESSOR                       0

#define CMK_SIGNAL_NOT_NEEDED                              0
#define CMK_SIGNAL_USE_SIGACTION                           0
#define CMK_SIGNAL_USE_SIGACTION_WITH_RESTART              1

#define CMK_THREADS_USE_CONTEXT                            0
#define CMK_THREADS_USE_PTHREADS                           0

#define CMK_THREADS_REQUIRE_NO_CPV                         0

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

#define CMK_WEB_MODE                                       1
#define CMK_DEBUG_MODE                                     0

#define CMK_LBDB_ON					   1
#define CMK_LB_CPUTIMER					   0

#define CMK_NO_SOCKETS                                     1

#define CMK_CCS_AVAILABLE                                  0


#endif

