
#ifndef _CONV_MACH_H
#define _CONV_MACH_H

#define CMK_CRAYEX                                         1

#define CMK_CONVERSE_MPI                                   1

#define CMK_MEMORY_PREALLOCATE_HACK			   0

#define CMK_DEFAULT_MAIN_USES_COMMON_CODE                  1

#define CMK_MALLOC_USE_GNU_MALLOC                          0
#define CMK_MALLOC_USE_OS_BUILTIN                          1

#define CMI_IO_BUFFER_EXPLICIT                             1
#define CMI_IO_FLUSH_USER                                  1

#define CMK_GETPAGESIZE_AVAILABLE			   1
#define CMK_MEMORY_PAGESIZE				   4096
#define CMK_MEMORY_PROTECTABLE				   0


#define CMK_SHARED_VARS_UNAVAILABLE                        1

#define CMK_SIGNAL_NOT_NEEDED                              0
#define CMK_SIGNAL_USE_SIGACTION                           0
#define CMK_SIGNAL_USE_SIGACTION_WITH_RESTART              1

#define CMK_THREADS_USE_CONTEXT                            0
#define CMK_THREADS_USE_FCONTEXT                           1
#define CMK_THREADS_USE_JCONTEXT                           0
#define CMK_THREADS_USE_PTHREADS                           0

#define CMK_TIMER_USE_GETRUSAGE                            0
#define CMK_TIMER_USE_SPECIAL                              1
#define CMK_TIMER_USE_TIMES                                0


#define CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT                   1
#define CMK_WHEN_PROCESSOR_IDLE_USLEEP                     0

#define CMK_64BIT					   1
#define CMK_AMD64					   1

#define CMK_WEB_MODE                                       1
#define CMK_DEBUG_MODE                                     0

#define CMK_LBDB_ON					   1

#define CMK_DISABLE_SYNC                                   1

#endif

