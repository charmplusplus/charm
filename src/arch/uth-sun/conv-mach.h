/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile$
 *	$Author$	$Locker$		$State$
 *	$Revision$	$Date$
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 * $Log$
 * Revision 1.2  1995-10-11 00:35:06  jyelon
 * Added MALLOC options.
 *
 * Revision 1.1  1995/09/30  14:58:49  jyelon
 * Initial revision
 *
 * Revision 2.7  1995/09/29  09:50:07  jyelon
 * CmiGet-->CmiDeliver, added protos, etc.
 *
 * Revision 2.6  1995/09/22  17:54:52  sanjeev
 * added CMK_MACHINE_NAME
 *
 * Revision 2.5  1995/09/20  13:22:57  jyelon
 * Added 'CMK_THREAD_' flags.
 *
 * Revision 2.4  1995/09/19  18:56:26  jyelon
 * added CMK_PREPROCESSOR_USES_ANSI_STANDARD_CONCATENATION and other goodies.
 *
 * Revision 2.3  1995/07/05  21:33:36  brunner
 * Took spaces out from before #defines
 *
 ***************************************************************************/

#ifndef _CONV_MACH_H
#define _CONV_MACH_H

/* #define CMK_USE_GNU_MALLOC */
/* #define CMK_USE_GNU_MALLOC_WITH_INTERRUPT_SUPPORT */
#define CMK_USE_OS_MALLOC
 
/* #define CMK_CTHINIT_IS_IN_CONVERSEINIT */
#define CMK_CTHINIT_IS_IN_MAIN

/* #define CMK_CSDEXITSCHEDULER_IS_A_FUNCTION */
#define CMK_CSDEXITSCHEDULER_SET_CSDSTOPFLAG

/* #define CMK_COMMHANDLE_IS_AN_INTEGER */
#define CMK_COMMHANDLE_IS_A_POINTER
 
/* #define CMK_USES_SPECIAL_CMIDELIVERS */ 
#define CMK_USES_COMMON_CMIDELIVERS

/* #define CMK_NO_SHARED_VARS_AT_ALL */
/* #define CMK_SHARED_VARS_EXEMPLAR */
#define CMK_SHARED_VARS_UNIPROCESSOR

/* #define CMK_PREPROCESSOR_USES_K_AND_R_STANDARD_CONCATENATION */
#define CMK_PREPROCESSOR_USES_ANSI_STANDARD_CONCATENATION

/* #define CMK_COMPILER_HATES_PROTOTYPES */
#define CMK_COMPILER_LIKES_PROTOTYPES

/* #define CMK_COMPILER_HATES_STATIC_PROTO */
#define CMK_COMPILER_LIKES_STATIC_PROTO

/* #define CMK_CMIMYPE_IS_A_BUILTIN */
/* #define CMK_CMIMYPE_IS_A_VARIABLE */
#define CMK_CMIMYPE_UNIPROCESSOR

/* #define CMK_CMIPRINTF_IS_A_BUILTIN */
#define CMK_CMIPRINTF_IS_JUST_PRINTF

/* #define CMK_THREADS_UNAVAILABLE */
#define CMK_THREADS_USE_ALLOCA




/* #define CMK_SIGHOLD_IS_A_BUILTIN */
#define CMK_SIGHOLD_USE_SIGMASK

/* #define CMK_RSH_USE_REMSH */
#define CMK_RSH_IS_A_COMMAND

/* #define CMK_TIMER_USE_TIMES */
#define CMK_TIMER_USE_GETRUSAGE

/* #define CMK_ASYNC_USE_SIOCGPGRP_AND_FIOASYNC */
#define CMK_ASYNC_USE_SETOWN_AND_SETFL

/* #define CMK_SIGNAL_USE_SIGACTION */
/* #define CMK_SIGNAL_USE_SIGACTION_AND_SIGEMPTYSET */
#define CMK_SIGNAL_IS_A_BUILTIN

#define CMK_MAX_DGRAM_SIZE 4096

/* #define CMK_STRERROR_IS_A_BUILTIN */
#define CMK_STRERROR_USE_SYS_ERRLIST

/* #define CMK_HAVE_STRINGS_H */
/* #define CMK_HAVE_STRING_H */
#define CMK_JUST_DECLARE_STRING_FNS

/* #define CMK_HAVE_WAITFLAGS_H */
#define CMK_HAVE_SYS_WAIT_H

#define CMK_MACHINE_NAME "UTH-SUN"


#endif
