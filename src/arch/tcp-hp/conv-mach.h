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
 * Revision 2.42  1998-02-19 08:40:34  jyelon
 * Added multicast code.
 *
 * Revision 2.41  1997/12/22 21:58:33  jyelon
 * Changed LDB initialization scheme.
 *
 * Revision 2.40  1997/08/06 20:36:19  jyelon
 * Fixed bugs.
 *
 * Revision 2.39  1997/07/28 19:01:31  jyelon
 * *** empty log message ***
 *
 * Revision 2.38  1997/07/26 16:42:23  jyelon
 * *** empty log message ***
 *
 * Revision 2.37  1997/05/05 13:56:33  jyelon
 * Updated for quickthreads
 *
 * Revision 2.36  1997/04/01 08:10:41  jyelon
 * Added CMK_GETPAGESIZE_AVAILABLE
 *
 * Revision 2.35  1997/03/25 23:09:24  milind
 * Got threads to work on 64-bit irix. Had to add JB_TWEAKING_ORIGIN flag to
 * all the conv-mach.h files. Also, _PAGESZ was undefined on irix. Added
 * code to memory.c to make it a static variable.
 *
 * Revision 2.34  1997/03/19 04:58:19  jyelon
 * Removed the CMK_DEFAULT_MAIN_USES_SIMULATOR_CODE flag.
 *
 * Revision 2.33  1997/02/13 09:31:58  jyelon
 * Updated for new main/ConverseInit structure.
 *
 * Revision 2.32  1997/02/08 14:10:29  jyelon
 * Correcting bugs in network version.
 *
 * Revision 2.31  1997/02/06 20:21:11  jyelon
 * Added BLANK_SPACE bla bla.
 *
 * Revision 2.30  1997/01/17 15:50:36  jyelon
 * Minor adjustments to deal with recent changes to Common code.
 *
 * Revision 2.29  1997/01/15 19:23:47  milind
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

#define CMK_ASYNC_NOT_NEEDED                               1
#define CMK_ASYNC_USE_FIOASYNC_AND_FIOSETOWN               0
#define CMK_ASYNC_USE_FIOASYNC_AND_SIOCSPGRP               0
#define CMK_ASYNC_USE_FIOSSAIOSTAT_AND_FIOSSAIOOWN         0
#define CMK_ASYNC_USE_F_SETFL_AND_F_SETOWN                 0

#define CMK_CMIDELIVERS_USE_COMMON_CODE                    1
#define CMK_CMIDELIVERS_USE_SPECIAL_CODE                   0

#define CMK_CMIPRINTF_IS_A_BUILTIN                         1
#define CMK_CMIPRINTF_IS_JUST_PRINTF                       0

#define CMK_COMMHANDLE_IS_AN_INTEGER                       0
#define CMK_COMMHANDLE_IS_A_POINTER                        1

#define CMK_CSDEXITSCHEDULER_IS_A_FUNCTION                 0
#define CMK_CSDEXITSCHEDULER_SET_CSDSTOPFLAG               1

#define CMK_FIX_HP_CONNECT_BUG                             1

#define CMK_GETPAGESIZE_AVAILABLE                          0

#define CMK_IS_HETERO                                      1

#define CMK_MACHINE_NAME                                   "tcp-hp"

#define CMK_MALLOC_USE_GNU_MALLOC                          1
#define CMK_MALLOC_USE_OS_BUILTIN                          0

#define CMK_MEMORY_PAGESIZE                                8192
#define CMK_MEMORY_PROTECTABLE                             0

#define CMK_MSG_HEADER_SIZE_BYTES                          8
#define CMK_MSG_HEADER_BLANK_SPACE                         0

#define CMK_MULTICAST_GROUP_TYPE                struct { unsigned pe, id; }
#define CMK_MULTICAST_DEF_USE_COMMON_CODE                  1
#define CMK_MULTICAST_LIST_USE_COMMON_CODE                 1
#define CMK_MULTICAST_GROUP_USE_COMMON_CODE                1

#define CMK_PREPROCESSOR_CANNOT_DO_CONCATENATION           0
#define CMK_PREPROCESSOR_USES_ANSI_STANDARD_CONCATENATION  1

#define CMK_PROTOTYPES_FAIL                                0
#define CMK_PROTOTYPES_WORK                                1

#define CMK_RSH_IS_A_COMMAND                               0
#define CMK_RSH_NOT_NEEDED                                 0
#define CMK_RSH_USE_REMSH                                  1

#define CMK_SHARED_VARS_EXEMPLAR                           0
#define CMK_SHARED_VARS_UNAVAILABLE                        1
#define CMK_SHARED_VARS_SUN_THREADS                            0
#define CMK_SHARED_VARS_UNIPROCESSOR                       0

#define CMK_SIGHOLD_IS_A_BUILTIN                           0
#define CMK_SIGHOLD_NOT_NEEDED                             0
#define CMK_SIGHOLD_USE_SIGMASK                            1

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

#define CMK_SYNCHRONIZE_ON_TCP_CLOSE                       0

#define CMK_THREADS_REQUIRE_NO_CPV                         0

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

