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
 * Revision 1.36  1998-06-15 19:54:03  jyelon
 * Adding new typedef stuff.
 *
 * Revision 1.35  1998/05/22 22:07:34  milind
 * Added Charm-IDL
 *
 * Revision 1.34  1998/04/17 17:18:35  milind
 * Added CMK_CCS_AVAILABLE flag.
 *
 * Revision 1.33  1998/02/19 08:40:58  jyelon
 * Added multicast code.
 *
 * Revision 1.32  1997/12/22 21:58:45  jyelon
 * Changed LDB initialization scheme.
 *
 * Revision 1.31  1997/12/10 21:59:37  jyelon
 * Modified CmiDeliverSpecificMsg so that it works with uth version.
 *
 * Revision 1.30  1997/08/06 20:36:31  jyelon
 * Fixed bugs.
 *
 * Revision 1.29  1997/07/28 19:01:40  jyelon
 * *** empty log message ***
 *
 * Revision 1.28  1997/07/26 16:42:33  jyelon
 * *** empty log message ***
 *
 * Revision 1.27  1997/05/05 13:57:08  jyelon
 * Updated for quickthreads
 *
 * Revision 1.26  1997/04/01 08:10:47  jyelon
 * Added CMK_GETPAGESIZE_AVAILABLE
 *
 * Revision 1.25  1997/03/25 23:09:30  milind
 * Got threads to work on 64-bit irix. Had to add JB_TWEAKING_ORIGIN flag to
 * all the conv-mach.h files. Also, _PAGESZ was undefined on irix. Added
 * code to memory.c to make it a static variable.
 *
 * Revision 1.24  1997/03/19 04:58:25  jyelon
 * Removed the CMK_DEFAULT_MAIN_USES_SIMULATOR_CODE flag.
 *
 * Revision 1.23  1997/02/13 09:32:01  jyelon
 * Updated for new main/ConverseInit structure.
 *
 * Revision 1.22  1997/02/08 14:10:32  jyelon
 * Correcting bugs in network version.
 *
 * Revision 1.21  1997/02/06 20:21:14  jyelon
 * Added BLANK_SPACE bla bla.
 *
 * Revision 1.20  1997/01/17 15:50:41  jyelon
 * Minor adjustments to deal with recent changes to Common code.
 *
 * Revision 1.19  1996/11/23 02:25:45  milind
 * Fixed several subtle bugs in the converse runtime for convex
 * exemplar.
 *
 * Revision 1.18  1996/11/08 22:23:12  brunner
 * Put _main in for HP-UX CC compilation.  It is ignored according to the
 * CMK_USE_HP_MAIN_FIX flag.
 *
 * Revision 1.17  1996/10/24 19:40:33  milind
 * Added CMK_IS_HETERO to all the net-all versions.
 *
 * Revision 1.16  1996/08/08 20:16:53  jyelon
 * *** empty log message ***
 *
 * Revision 1.15  1996/07/19 17:07:37  jyelon
 * *** empty log message ***
 *
 * Revision 1.14  1996/07/16 17:23:37  jyelon
 * Renamed a flag.
 *
 * Revision 1.13  1996/07/16 05:20:41  milind
 * Added CMK_VECTOR_SEND
 *
 * Revision 1.12  1996/07/15  20:58:27  jyelon
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

#define CMK_CCS_AVAILABLE                                  0

#define CMK_CMIDELIVERS_USE_COMMON_CODE                    1
#define CMK_CMIDELIVERS_USE_SPECIAL_CODE                   0

#define CMK_CMIPRINTF_IS_A_BUILTIN                         0
#define CMK_CMIPRINTF_IS_JUST_PRINTF                       1

#define CMK_COMMHANDLE_IS_AN_INTEGER                       0
#define CMK_COMMHANDLE_IS_A_POINTER                        1

#define CMK_CSDEXITSCHEDULER_IS_A_FUNCTION                 0
#define CMK_CSDEXITSCHEDULER_SET_CSDSTOPFLAG               1

#define CMK_FIX_HP_CONNECT_BUG                             0

#define CMK_GETPAGESIZE_AVAILABLE                          0

#define CMK_IS_HETERO                                      0

#define CMK_MACHINE_NAME                                   "uth-rs6k"

#define CMK_MALLOC_USE_GNU_MALLOC                          0
#define CMK_MALLOC_USE_OS_BUILTIN                          1

#define CMK_MEMORY_PAGESIZE                                8192
#define CMK_MEMORY_PROTECTABLE                             0

#define CMK_MSG_HEADER_SIZE_BYTES                          4
#define CMK_MSG_HEADER_BLANK_SPACE                         0

#define CMK_MULTICAST_GROUP_TYPE                struct { unsigned pe, id; }
#define CMK_MULTICAST_DEF_USE_COMMON_CODE                  1
#define CMK_MULTICAST_LIST_USE_COMMON_CODE                 1
#define CMK_MULTICAST_GROUP_USE_COMMON_CODE                1

#define CMK_PREPROCESSOR_CANNOT_DO_CONCATENATION           0
#define CMK_PREPROCESSOR_USES_ANSI_STANDARD_CONCATENATION  1

#define CMK_PROTOTYPES_FAIL                                0
#define CMK_PROTOTYPES_WORK                                1

#define CMK_RSH_IS_A_COMMAND                               1
#define CMK_RSH_NOT_NEEDED                                 0
#define CMK_RSH_USE_REMSH                                  0

#define CMK_SHARED_VARS_EXEMPLAR                           0
#define CMK_SHARED_VARS_UNAVAILABLE                        0
#define CMK_SHARED_VARS_SUN_THREADS                            0
#define CMK_SHARED_VARS_UNIPROCESSOR                       1

#define CMK_SIGHOLD_IS_A_BUILTIN                           0
#define CMK_SIGHOLD_NOT_NEEDED                             0
#define CMK_SIGHOLD_USE_SIGMASK                            1

#define CMK_SIGNAL_NOT_NEEDED                              0
#define CMK_SIGNAL_USE_SIGACTION                           1
#define CMK_SIGNAL_USE_SIGACTION_WITH_RESTART              0

#define CMK_SIZE_T                                         long unsigned int

#define CMK_STATIC_PROTO_FAILS                             0
#define CMK_STATIC_PROTO_WORKS                             1

#define CMK_STRERROR_IS_A_BUILTIN                          0
#define CMK_STRERROR_USE_SYS_ERRLIST                       1

#define CMK_STRINGS_USE_OWN_DECLARATIONS                   1
#define CMK_STRINGS_USE_STRINGS_H                          0
#define CMK_STRINGS_USE_STRING_H                           0

#define CMK_SYNCHRONIZE_ON_TCP_CLOSE                       0

#define CMK_THREADS_REQUIRE_NO_CPV                         1

#define CMK_TIMER_USE_GETRUSAGE                            0
#define CMK_TIMER_USE_SPECIAL                              0
#define CMK_TIMER_USE_TIMES                                1

#define CMK_TYPEDEF_INT2    short
#define CMK_TYPEDEF_INT4    int
#define CMK_TYPEDEF_INT8    long long
#define CMK_TYPEDEF_FLOAT4  float
#define CMK_TYPEDEF_FLOAT8  double
#define CMK_TYPEDEF_FLOAT16 struct { char d[16]; }

#define CMK_VECTOR_SEND_USES_COMMON_CODE                   1
#define CMK_VECTOR_SEND_USES_SPECIAL_CODE                  0

#define CMK_WAIT_NOT_NEEDED                                0
#define CMK_WAIT_USES_SYS_WAIT_H                           1
#define CMK_WAIT_USES_WAITFLAGS_H                          0

#define CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT                   1
#define CMK_WHEN_PROCESSOR_IDLE_USLEEP                     0

#define CMK_USE_HP_MAIN_FIX                                0
#define CMK_DONT_USE_HP_MAIN_FIX                           1

#define CPP_LOCATION "/usr/lib/cpp"

#endif

