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
 * Revision 1.28  1998-03-04 17:17:19  milind
 * Fixed the size_t errors.
 *
 * Revision 1.27  1998/02/19 08:39:31  jyelon
 * Added multicast code.
 *
 * Revision 1.26  1997/12/22 21:57:48  jyelon
 * Changed LDB initialization scheme.
 *
 * Revision 1.25  1997/08/06 20:35:44  jyelon
 * Fixed bugs.
 *
 * Revision 1.24  1997/07/28 20:13:26  milind
 * Fixed bugs due to ckfutures declarations in c++interface.h
 * Also, wrote macros for node numbering in exemplar.
 *
 * Revision 1.23  1997/07/28 19:00:54  jyelon
 * *** empty log message ***
 *
 * Revision 1.22  1997/07/26 16:41:50  jyelon
 * *** empty log message ***
 *
 * Revision 1.21  1997/05/05 13:52:41  jyelon
 * Updated for quickthreads
 *
 * Revision 1.20  1997/04/01 08:10:22  jyelon
 * Added CMK_GETPAGESIZE_AVAILABLE
 *
 * Revision 1.19  1997/03/25 23:09:07  milind
 * Got threads to work on 64-bit irix. Had to add JB_TWEAKING_ORIGIN flag to
 * all the conv-mach.h files. Also, _PAGESZ was undefined on irix. Added
 * code to memory.c to make it a static variable.
 *
 * Revision 1.18  1997/03/19 04:58:02  jyelon
 * Removed the CMK_DEFAULT_MAIN_USES_SIMULATOR_CODE flag.
 *
 * Revision 1.17  1997/02/13 17:32:42  milind
 * Fixed a minor typo in CmiSignal in convcore.c.
 * Changed net-hp-cc/conv-mach.h to set ASYNC_NOT_NEEDED.
 *
 * Revision 1.16  1997/02/13 09:31:44  jyelon
 * Updated for new main/ConverseInit structure.
 *
 * Revision 1.15  1997/02/08 14:10:18  jyelon
 * Correcting bugs in network version.
 *
 * Revision 1.14  1997/02/06 20:20:38  jyelon
 * Added BLANK_SPACE bla bla.
 *
 * Revision 1.13  1997/01/17 15:50:25  jyelon
 * Minor adjustments to deal with recent changes to Common code.
 *
 * Revision 1.12  1997/01/15 16:17:48  milind
 * Changed CmiAlloc interrupt-safe for HP machines.
 *
 * Revision 1.11  1996/11/23 02:25:38  milind
 * Fixed several subtle bugs in the converse runtime for convex
 * exemplar.
 *
 * Revision 1.10  1996/11/08 22:22:58  brunner
 * Put _main in for HP-UX CC compilation.  It is ignored according to the
 * CMK_USE_HP_MAIN_FIX flag.
 *
 * Revision 1.9  1996/10/24 19:40:25  milind
 * Added CMK_IS_HETERO to all the net-all versions.
 *
 * Revision 1.8  1996/10/22 19:08:32  milind
 * Added +z option to produce position independent code.
 * Needed for parallel perl.
 *
 * Revision 1.7  1996/08/08 20:16:53  jyelon
 * *** empty log message ***
 *
 * Revision 1.6  1996/07/16 17:23:37  jyelon
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

#include <stdlib.h>

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

#define CMK_FIX_HP_CONNECT_BUG                             0

#define CMK_GETPAGESIZE_AVAILABLE                          0

#define CMK_IS_HETERO                                      1

#define CMK_MACHINE_NAME                                   "net-hp-cc"

#define CMK_MALLOC_USE_GNU_MALLOC                          1
#define CMK_MALLOC_USE_OS_BUILTIN                          0

#define CMK_MEMORY_PAGESIZE                                8192
#define CMK_MEMORY_PROTECTABLE                             0

#define CMK_MSG_HEADER_SIZE_BYTES                         16
#define CMK_MSG_HEADER_BLANK_SPACE                        12

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
#define CMK_SHARED_VARS_SUN_THREADS                        0
#define CMK_SHARED_VARS_UNIPROCESSOR                       0

#define CMK_SIGHOLD_IS_A_BUILTIN                           0
#define CMK_SIGHOLD_NOT_NEEDED                             0
#define CMK_SIGHOLD_USE_SIGMASK                            1

#define CMK_SIGNAL_NOT_NEEDED                              0
#define CMK_SIGNAL_USE_SIGACTION                           1
#define CMK_SIGNAL_USE_SIGACTION_WITH_RESTART              0

#define CMK_SIZE_T                                         unsigned

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

#define CMK_VECTOR_SEND_USES_COMMON_CODE                   1
#define CMK_VECTOR_SEND_USES_SPECIAL_CODE                  0

#define CMK_WAIT_NOT_NEEDED                                0
#define CMK_WAIT_USES_SYS_WAIT_H                           1
#define CMK_WAIT_USES_WAITFLAGS_H                          0

#define CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT                   0
#define CMK_WHEN_PROCESSOR_IDLE_USLEEP                     1

#define CMK_USE_HP_MAIN_FIX                                1
#define CMK_DONT_USE_HP_MAIN_FIX                           0

#endif

