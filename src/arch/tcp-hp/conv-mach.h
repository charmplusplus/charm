/***************************************************************************
 * RCS INFORMATION:
 *
 *   $RCSfile$
 *   $Author$       $Locker$        $State$
 *   $Revision$     $Date$
 *
 ***************************************************************************
 *
 * $Log$
 * Revision 2.29  1997-01-15 19:23:47  milind
 * Fixed CMK_HP_MAIN_FIX bug in tcp-hp version.
 *
 * Revision 2.28  1996/11/23 02:25:44  milind
 * Fixed several subtle bugs in the converse runtime for convex
 * exemplar.
 *
 * Revision 2.27  1996/11/08 22:23:09  brunner
 * Put _main in for HP-UX CC compilation.  It is ignored according to the
 * CMK_USE_HP_MAIN_FIX flag.
 *
 * Revision 2.26  1996/10/24 19:40:32  milind
 * Added CMK_IS_HETERO to all the net-all versions.
 *
 * Revision 2.25  1996/08/08 20:16:53  jyelon
 * *** empty log message ***
 *
 * Revision 2.24  1996/07/19 17:07:37  jyelon
 * *** empty log message ***
 *
 * Revision 2.23  1996/07/16 17:23:37  jyelon
 * Renamed a flag.
 *
 * Revision 2.22  1996/07/16 05:20:41  milind
 * Added CMK_VECTOR_SEND
 *
 * Revision 2.21  1996/07/15  20:58:27  jyelon
 * Flags now use #if, not #ifdef.  Also cleaned up a lot.
 *
 *
 **************************************************************************/

#ifndef _CONV_MACH_H
#define _CONV_MACH_H

#define CMK_ASYNC_DOESNT_WORK_USE_TIMER_INSTEAD            1
#define CMK_ASYNC_NOT_NEEDED                               0
#define CMK_ASYNC_USE_FIOASYNC_AND_FIOSETOWN               0
#define CMK_ASYNC_USE_FIOASYNC_AND_SIOCSPGRP               0
#define CMK_ASYNC_USE_FIOSSAIOSTAT_AND_FIOSSAIOOWN         0
#define CMK_ASYNC_USE_F_SETFL_AND_F_SETOWN                 0

#define CMK_CMIDELIVERS_USE_COMMON_CODE                    1
#define CMK_CMIDELIVERS_USE_SPECIAL_CODE                   0

#define CMK_CMIMYPE_IS_A_BUILTIN                           0
#define CMK_CMIMYPE_IS_A_VARIABLE                          1
#define CMK_CMIMYPE_UNIPROCESSOR                           0

#define CMK_CMIPRINTF_IS_A_BUILTIN                         1
#define CMK_CMIPRINTF_IS_JUST_PRINTF                       0

#define CMK_COMMHANDLE_IS_AN_INTEGER                       0
#define CMK_COMMHANDLE_IS_A_POINTER                        1

#define CMK_CSDEXITSCHEDULER_IS_A_FUNCTION                 0
#define CMK_CSDEXITSCHEDULER_SET_CSDSTOPFLAG               1

#define CMK_CTHINIT_IS_IN_CONVERSEINIT                     1
#define CMK_CTHINIT_IS_IN_MAIN                             0

#define CMK_DEFAULT_MAIN_USES_COMMON_CODE                  1
#define CMK_DEFAULT_MAIN_USES_SIMULATOR_CODE               0

#define CMK_DGRAM_BUF_SIZE                                 0
#define CMK_DGRAM_MAX_SIZE                                 0
#define CMK_DGRAM_WINDOW_SIZE                              0

#define CMK_FIX_HP_CONNECT_BUG                             1

#define CMK_IS_HETERO                                      1

#define CMK_MACHINE_NAME                                   "tcp-hp"

#define CMK_MALLOC_USE_GNU                                 0
#define CMK_MALLOC_USE_GNU_WITH_INTERRUPT_SUPPORT          1
#define CMK_MALLOC_USE_OS_BUILTIN                          0

#define CMK_MSG_HEADER_SIZE_BYTES                          8

#define CMK_PREPROCESSOR_CANNOT_DO_CONCATENATION           0
#define CMK_PREPROCESSOR_USES_ANSI_STANDARD_CONCATENATION  1

#define CMK_PROTOTYPES_FAIL                                0
#define CMK_PROTOTYPES_WORK                                1

#define CMK_RSH_IS_A_COMMAND                               0
#define CMK_RSH_NOT_NEEDED                                 0
#define CMK_RSH_USE_REMSH                                  1

#define CMK_SHARED_VARS_EXEMPLAR                           0
#define CMK_SHARED_VARS_UNAVAILABLE                        1
#define CMK_SHARED_VARS_UNIPROCESSOR                       0

#define CMK_SIGHOLD_IS_A_BUILTIN                           0
#define CMK_SIGHOLD_NOT_NEEDED                             0
#define CMK_SIGHOLD_USE_SIGMASK                            1

#define CMK_SIGNAL_IS_A_BUILTIN                            0
#define CMK_SIGNAL_NOT_NEEDED                              0
#define CMK_SIGNAL_USE_SIGACTION                           1
#define CMK_SIGNAL_USE_SIGACTION_WITH_RESTART              0

#define CMK_SIZE_T                                         unsigned int

#define CMK_STATIC_PROTO_FAILS                             0
#define CMK_STATIC_PROTO_WORKS                             1

#define CMK_STRERROR_IS_A_BUILTIN                          0
#define CMK_STRERROR_USE_SYS_ERRLIST                       1

#define CMK_STRINGS_USE_OWN_DECLARATIONS                   0
#define CMK_STRINGS_USE_STRINGS_H                          0
#define CMK_STRINGS_USE_STRING_H                           1

#define CMK_THREADS_UNAVAILABLE                            0
#define CMK_THREADS_USE_ALLOCA                             1
#define CMK_THREADS_USE_ALLOCA_WITH_HEADER_FILE            0
#define CMK_THREADS_USE_ALLOCA_WITH_PRAGMA                 0
#define CMK_THREADS_USE_JB_TWEAKING                        0
#define CMK_THREADS_USE_JB_TWEAKING_EXEMPLAR               0

#define CMK_TIMER_USE_GETRUSAGE                            0
#define CMK_TIMER_USE_SPECIAL                              0
#define CMK_TIMER_USE_TIMES                                1

#define CMK_VECTOR_SEND_USES_COMMON_CODE                        1
#define CMK_VECTOR_SEND_USES_SPECIAL_CODE                        0

#define CMK_WAIT_NOT_NEEDED                                0
#define CMK_WAIT_USES_SYS_WAIT_H                           1
#define CMK_WAIT_USES_WAITFLAGS_H                          0

#define CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT                   0
#define CMK_WHEN_PROCESSOR_IDLE_USLEEP                     1

#define CMK_USE_HP_MAIN_FIX                                1
#define CMK_DONT_USE_HP_MAIN_FIX                           0

#endif

