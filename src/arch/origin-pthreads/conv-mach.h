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
 * Revision 1.2  1997-12-22 21:58:04  jyelon
 * Changed LDB initialization scheme.
 *
 * Revision 1.1  1997/11/26 19:13:59  milind
 * Origin2000 Posix Threads Version
 *
 * Revision 1.5  1997/08/06 20:35:58  jyelon
 * Fixed bugs.
 *
 * Revision 1.4  1997/07/28 19:01:08  jyelon
 * *** empty log message ***
 *
 * Revision 1.3  1997/07/26 16:42:03  jyelon
 * *** empty log message ***
 *
 * Revision 1.2  1997/07/07 22:13:50  milind
 * Made threads to work.
 *
 * Revision 1.1  1997/03/28 17:38:23  milind
 * Added Origin2000 version.
 *
 * Revision 1.2  1997/03/25 23:09:09  milind
 * Got threads to work on 64-bit irix. Had to add JB_TWEAKING_ORIGIN flag to
 * all the conv-mach.h files. Also, _PAGESZ was undefined on irix. Added
 * code to memory.c to make it a static variable.
 *
 * Revision 1.1  1997/03/19 21:45:27  milind
 * net-irix-64 bit bersion. Not tested yet.
 *
 * Revision 1.3  1997/02/13 09:31:45  jyelon
 * Updated for new main/ConverseInit structure.
 *
 * Revision 1.2  1997/02/08 14:10:19  jyelon
 * Correcting bugs in network version.
 *
 * Revision 1.1  1997/01/28 16:26:42  milind
 * Added net-irix version. Had to fix the charm++ translator for that.
 * Also, threads are currently not supported on net-irix.
 * Added a test program to measure scheduling overhead for both
 * normal as well as threaded entry methods.
 *
 * Revision 2.28  1997/01/17 15:50:26  jyelon
 * Minor adjustments to deal with recent changes to Common code.
 *
 * Revision 2.27  1996/11/23 02:25:39  milind
 * Fixed several subtle bugs in the converse runtime for convex
 * exemplar.
 *
 * Revision 2.26  1996/11/08 22:22:59  brunner
 * Put _main in for HP-UX CC compilation.  It is ignored according to the
 * CMK_USE_HP_MAIN_FIX flag.
 *
 * Revision 2.25  1996/10/24 19:40:26  milind
 * Added CMK_IS_HETERO to all the net-all versions.
 *
 * Revision 2.24  1996/08/08 20:16:53  jyelon
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

#define CMK_CMIPRINTF_IS_A_BUILTIN                         0
#define CMK_CMIPRINTF_IS_JUST_PRINTF                       1

#define CMK_COMMHANDLE_IS_AN_INTEGER                       0
#define CMK_COMMHANDLE_IS_A_POINTER                        1

#define CMK_CSDEXITSCHEDULER_IS_A_FUNCTION                 0
#define CMK_CSDEXITSCHEDULER_SET_CSDSTOPFLAG               1

#define CMK_DEFAULT_MAIN_USES_COMMON_CODE                  1

#define CMK_FIX_HP_CONNECT_BUG                             0

#define CMK_GETPAGESIZE_AVAILABLE                          1

#define CMK_IS_HETERO                                      0

#define CMK_MACHINE_NAME                                   "origin-pthreads"

#define CMK_MALLOC_USE_GNU_MALLOC                          0
#define CMK_MALLOC_USE_OS_BUILTIN                          1

#define CMK_MEMORY_PAGESIZE                                8192
#define CMK_MEMORY_PROTECTABLE                             0

#define CMK_MSG_HEADER_SIZE_BYTES                          4
#define CMK_MSG_HEADER_BLANK_SPACE                         0

#define CMK_PREPROCESSOR_CANNOT_DO_CONCATENATION           0
#define CMK_PREPROCESSOR_USES_ANSI_STANDARD_CONCATENATION  1

#define CMK_PROTOTYPES_FAIL                                0
#define CMK_PROTOTYPES_WORK                                1

#define CMK_RSH_IS_A_COMMAND                               0
#define CMK_RSH_NOT_NEEDED                                 1
#define CMK_RSH_USE_REMSH                                  0

#define CMK_SHARED_VARS_EXEMPLAR                           0
#define CMK_SHARED_VARS_UNAVAILABLE                        0
#define CMK_SHARED_VARS_UNIPROCESSOR                       0
#define CMK_SHARED_VARS_SUN_THREADS                        0
#define CMK_SHARED_VARS_PTHREADS                           1

#define CMK_SIGHOLD_IS_A_BUILTIN                           0
#define CMK_SIGHOLD_NOT_NEEDED                             1
#define CMK_SIGHOLD_USE_SIGMASK                            0

#define CMK_SIGNAL_NOT_NEEDED                              1
#define CMK_SIGNAL_USE_SIGACTION                           0
#define CMK_SIGNAL_USE_SIGACTION_WITH_RESTART              0

#define CMK_SIZE_T                                         unsigned long

#define CMK_STATIC_PROTO_FAILS                             0
#define CMK_STATIC_PROTO_WORKS                             1

#define CMK_STRERROR_IS_A_BUILTIN                          1
#define CMK_STRERROR_USE_SYS_ERRLIST                       0

#define CMK_STRINGS_USE_OWN_DECLARATIONS                   0
#define CMK_STRINGS_USE_STRINGS_H                          0
#define CMK_STRINGS_USE_STRING_H                           1

#define CMK_SYNCHRONIZE_ON_TCP_CLOSE                       0

#define CMK_THREADS_REQUIRE_NO_CPV                         0

#define CMK_THREADS_UNAVAILABLE                            0
#define CMK_THREADS_USE_ALLOCA                             0
#define CMK_THREADS_USE_JB_TWEAKING                        0
#define CMK_THREADS_USE_JB_TWEAKING_EXEMPLAR               0
#define CMK_THREADS_USE_JB_TWEAKING_ORIGIN                 1

#define CMK_TIMER_USE_GETRUSAGE                            0
#define CMK_TIMER_USE_SPECIAL                              0
#define CMK_TIMER_USE_TIMES                                1

#define CMK_VECTOR_SEND_USES_COMMON_CODE                   1
#define CMK_VECTOR_SEND_USES_SPECIAL_CODE                  0

#define CMK_WAIT_NOT_NEEDED                                0
#define CMK_WAIT_USES_SYS_WAIT_H                           1
#define CMK_WAIT_USES_WAITFLAGS_H                          0

#define CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT                   1
#define CMK_WHEN_PROCESSOR_IDLE_USLEEP                     0

#define CMK_USE_HP_MAIN_FIX                                0
#define CMK_DONT_USE_HP_MAIN_FIX                           1

#endif

