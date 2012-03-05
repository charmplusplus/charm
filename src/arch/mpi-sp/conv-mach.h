#ifndef _CONV_MACH_H
#define _CONV_MACH_H


/*MPI Machine option: clean out the send buffer each time through
  the scheduler's loop.  This ensures sends actually leave, which
  on the SP speeds things up more than the extra wait slows things down.
  See TMS entry for task "Communication Performance", 
  11/01/2002, Orion Lawlor.
*/
#define CMK_NO_OUTSTANDING_SENDS                        1

#define CMK_BAD_MMAP_ADDRESS                              0xd0000000u

#define CMK_GETPAGESIZE_AVAILABLE                          0

#define CMK_MALLOC_USE_GNU_MALLOC                          0
#define CMK_MALLOC_USE_OS_BUILTIN                          1

#define CMK_MEMORY_PAGESIZE                                8192
#define CMK_MEMORY_PROTECTABLE                             0

#define CMK_NODE_QUEUE_AVAILABLE                           0

#define CMK_SHARED_VARS_UNAVAILABLE                        1
#define CMK_SHARED_VARS_UNIPROCESSOR                       0

#define CMK_THREADS_USE_CONTEXT                            0
#define CMK_THREADS_USE_PTHREADS                           0
#define CMK_THREADS_ARE_WIN32_FIBERS                       0

#define CMK_SIGNAL_NOT_NEEDED                              0
#define CMK_SIGNAL_USE_SIGACTION                           0
#define CMK_SIGNAL_USE_SIGACTION_WITH_RESTART              1

#define CMK_SYNCHRONIZE_ON_TCP_CLOSE                       0

#define CMK_THREADS_REQUIRE_NO_CPV                         0
#define CMK_THREADS_COPY_STACK                             0

#define CMK_TIMER_USE_GETRUSAGE                            0
#define CMK_TIMER_USE_SPECIAL                              1
#define CMK_TIMER_USE_TIMES                                0
#define CMK_TIMER_USE_AIX_READ_TIME                        0

#define CMK_TYPEDEF_INT2 short
#define CMK_TYPEDEF_INT4 int
#define CMK_TYPEDEF_UINT2 unsigned short
#define CMK_TYPEDEF_UINT4 unsigned int
#if CMK_LONG_LONG_DEFINED
#define CMK_TYPEDEF_INT8 long long
#define CMK_TYPEDEF_UINT8 unsigned long long
#else
#define CMK_TYPEDEF_INT8 long
#define CMK_TYPEDEF_UINT8 unsigned long
#endif
#define CMK_TYPEDEF_FLOAT4 float
#define CMK_TYPEDEF_FLOAT8 double


#define CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT                   1
#define CMK_WHEN_PROCESSOR_IDLE_USLEEP                     0


#define CMK_WEB_MODE                                       1
#define CMK_DEBUG_MODE                                     0

#define CMK_LBDB_ON					   1


#endif

