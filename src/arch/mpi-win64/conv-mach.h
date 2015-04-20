#ifndef _CONV_MACH_H
#define _CONV_MACH_H

#define CMK_CONVERSE_MPI                                   1
#define CMK_64BIT                                          1

#define CMK_GETPAGESIZE_AVAILABLE                          0

#define CMK_MALLOC_USE_GNU_MALLOC                          0
#define CMK_MALLOC_USE_OS_BUILTIN                          1

#define CMK_MEMORY_PAGESIZE                                8192
#define CMK_MEMORY_PROTECTABLE                             0


#define CMK_SHARED_VARS_UNAVAILABLE                        1
#define CMK_SHARED_VARS_UNIPROCESSOR                       0
#define CMK_SHARED_VARS_NT_THREADS                         0

#define CMK_SIGNAL_NOT_NEEDED                              1
#define CMK_SIGNAL_USE_SIGACTION                           0
#define CMK_SIGNAL_USE_SIGACTION_WITH_RESTART              0

#define CMK_THREADS_USE_JCONTEXT                           0
#define CMK_THREADS_REQUIRE_NO_CPV                         0
#define CMK_THREADS_COPY_STACK                             0
#define CMK_THREADS_ARE_WIN32_FIBERS                       1

#define CMK_TIMER_USE_GETRUSAGE                            0
#define CMK_TIMER_USE_SPECIAL                              0
#define CMK_TIMER_USE_TIMES                                0
#define CMK_TIMER_USE_WIN32API                             1


#define CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT                   1
#define CMK_WHEN_PROCESSOR_IDLE_USLEEP                     0

#define CMK_WEB_MODE                                       0

#define CMK_LBDB_ON					   1

#undef  CMK_IMMEDIATE_MSG
#define CMK_IMMEDIATE_MSG				   0

/*
#define CMK_PCQUEUE_LOCK                                   1
*/

#endif
