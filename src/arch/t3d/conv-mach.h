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
 * Revision 1.16  1997-03-19 04:58:18  jyelon
 * Removed the CMK_DEFAULT_MAIN_USES_SIMULATOR_CODE flag.
 *
 * Revision 1.15  1997/02/13 09:31:56  jyelon
 * Updated for new main/ConverseInit structure.
 *
 * Revision 1.14  1997/02/08 14:10:29  jyelon
 * Correcting bugs in network version.
 *
 * Revision 1.13  1997/02/06 20:21:10  jyelon
 * Added BLANK_SPACE bla bla.
 *
 * Revision 1.12  1997/01/17 15:50:35  jyelon
 * Minor adjustments to deal with recent changes to Common code.
 *
 * Revision 1.11  1996/11/23 02:25:43  milind
 * Fixed several subtle bugs in the converse runtime for convex
 * exemplar.
 *
 * Revision 1.10  1996/11/08 22:23:09  brunner
 * Put _main in for HP-UX CC compilation.  It is ignored according to the
 * CMK_USE_HP_MAIN_FIX flag.
 *
 * Revision 1.9  1996/10/24 19:40:31  milind
 * Added CMK_IS_HETERO to all the net-all versions.
 *
 * Revision 1.8  1996/08/08 20:16:53  jyelon
 * *** empty log message ***
 *
 * Revision 1.7  1996/07/16 21:08:30  gursoy
 * selected the correct thread option
 *
 * Revision 1.6  1996/07/16  17:23:37  jyelon
 * Renamed a flag.
 *
 * Revision 1.5  1996/07/16 05:20:41  milind
 * Added CMK_VECTOR_SEND
 *
 * Revision 1.4  1996/07/15  20:58:27  jyelon
 * Flags now use #if, not #ifdef.  Also cleaned up a lot.
 *
 *
 **************************************************************************/

#ifndef _CONV_MACH_H
#define _CONV_MACH_H

#define CMK_ASYNC_NOT_NEEDED                               1
#define CMK_ASYNC_USE_FIOASYNC_AND_FIOSETOWN               0
#define CMK_ASYNC_USE_FIOASYNC_AND_SIOCSPGRP               0
#define CMK_ASYNC_USE_FIOSSAIOSTAT_AND_FIOSSAIOOWN         0
#define CMK_ASYNC_USE_F_SETFL_AND_F_SETOWN                 0

#define CMK_CMIDELIVERS_USE_COMMON_CODE                    0
#define CMK_CMIDELIVERS_USE_SPECIAL_CODE                   1

#define CMK_CMIPRINTF_IS_A_BUILTIN                         0
#define CMK_CMIPRINTF_IS_JUST_PRINTF                       1

#define CMK_COMMHANDLE_IS_AN_INTEGER                       1
#define CMK_COMMHANDLE_IS_A_POINTER                        0

#define CMK_CSDEXITSCHEDULER_IS_A_FUNCTION                 0
#define CMK_CSDEXITSCHEDULER_SET_CSDSTOPFLAG               1

#define CMK_FIX_HP_CONNECT_BUG                             0

#define CMK_IS_HETERO                                      0

#define CMK_MACHINE_NAME                                   "t3d"

#define CMK_MALLOC_USE_GNU_MALLOC                          0
#define CMK_MALLOC_USE_OS_BUILTIN                          1

#define CMK_MSG_HEADER_SIZE_BYTES                          8
#define CMK_MSG_HEADER_BLANK_SPACE                         0

#define CMK_PREPROCESSOR_CANNOT_DO_CONCATENATION           0
#define CMK_PREPROCESSOR_USES_ANSI_STANDARD_CONCATENATION  1

#define CMK_PROTOTYPES_FAIL                                0
#define CMK_PROTOTYPES_WORK                                1

#define CMK_RSH_IS_A_COMMAND                               0
#define CMK_RSH_NOT_NEEDED                                 1
#define CMK_RSH_USE_REMSH                                  0

#define CMK_SHARED_VARS_EXEMPLAR                           0
#define CMK_SHARED_VARS_UNAVAILABLE                        1
#define CMK_SHARED_VARS_SUN_THREADS                            0
#define CMK_SHARED_VARS_UNIPROCESSOR                       0

#define CMK_SIGHOLD_IS_A_BUILTIN                           0
#define CMK_SIGHOLD_NOT_NEEDED                             1
#define CMK_SIGHOLD_USE_SIGMASK                            0

#define CMK_SIGNAL_NOT_NEEDED                              1
#define CMK_SIGNAL_USE_SIGACTION                           0
#define CMK_SIGNAL_USE_SIGACTION_WITH_RESTART              0

#define CMK_SIZE_T                                         unsigned

#define CMK_STATIC_PROTO_FAILS                             0
#define CMK_STATIC_PROTO_WORKS                             1

#define CMK_STRERROR_IS_A_BUILTIN                          1
#define CMK_STRERROR_USE_SYS_ERRLIST                       0

#define CMK_STRINGS_USE_OWN_DECLARATIONS                   0
#define CMK_STRINGS_USE_STRINGS_H                          0
#define CMK_STRINGS_USE_STRING_H                           1

#define CMK_THREADS_REQUIRE_ALLOCA_H                       0
#define CMK_THREADS_REQUIRE_PRAGMA_ALLOCA                  0
#define CMK_THREADS_REQUIRE_NO_CPV                         0

#define CMK_THREADS_UNAVAILABLE                            0
#define CMK_THREADS_USE_ALLOCA                             0
#define CMK_THREADS_USE_JB_TWEAKING                        1
#define CMK_THREADS_USE_JB_TWEAKING_EXEMPLAR               0

#define CMK_TIMER_USE_GETRUSAGE                            0
#define CMK_TIMER_USE_SPECIAL                              1
#define CMK_TIMER_USE_TIMES                                0

#define CMK_VECTOR_SEND_USES_COMMON_CODE                        1
#define CMK_VECTOR_SEND_USES_SPECIAL_CODE                        0

#define CMK_WAIT_NOT_NEEDED                                1
#define CMK_WAIT_USES_SYS_WAIT_H                           0
#define CMK_WAIT_USES_WAITFLAGS_H                          0

#define CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT                   1
#define CMK_WHEN_PROCESSOR_IDLE_USLEEP                     0

#define CMK_USE_HP_MAIN_FIX                                0
#define CMK_DONT_USE_HP_MAIN_FIX                           1

#endif

