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
 *
 ***************************************************************************/

#ifndef _CONV_MACH_H
#define _CONV_MACH_H

/* #define CMK_USE_OS_MALLOC */
/* #define CMK_USE_GNU_MALLOC */
#define CMK_USE_GNU_MALLOC_WITH_INTERRUPT_SUPPORT

/* #define CMK_CTHINIT_IS_IN_MAIN */
#define CMK_CTHINIT_IS_IN_CONVERSEINIT

/* #define CMK_CSDEXITSCHEDULER_IS_A_FUNCTION */
#define CMK_CSDEXITSCHEDULER_SET_CSDSTOPFLAG

/* #define CMK_COMMHANDLE_IS_AN_INTEGER */
#define CMK_COMMHANDLE_IS_A_POINTER
 
/* #define CMK_USES_SPECIAL_CMIDELIVERS */ 
#define CMK_USES_COMMON_CMIDELIVERS

/* #define CMK_SHARED_VARS_EXEMPLAR */
/* #define CMK_SHARED_VARS_UNIPROCESSOR */
#define CMK_NO_SHARED_VARS_AT_ALL

/* #define CMK_PREPROCESSOR_CANNOT_DO_CONCATENATION */
#define CMK_PREPROCESSOR_USES_ANSI_STANDARD_CONCATENATION

/* #define CMK_COMPILER_HATES_PROTOTYPES */
#define CMK_COMPILER_LIKES_PROTOTYPES

/* #define CMK_COMPILER_HATES_STATIC_PROTO */
#define CMK_COMPILER_LIKES_STATIC_PROTO

/* #define CMK_CMIMYPE_IS_A_BUILTIN */
#define CMK_CMIMYPE_IS_A_VARIABLE

/* #define CMK_CMIPRINTF_IS_JUST_PRINTF */
#define CMK_CMIPRINTF_IS_A_BUILTIN

/* #define CMK_THREADS_UNAVAILABLE */
#define CMK_THREADS_USE_ALLOCA





/* #define CMK_SIGHOLD_USE_SIGMASK */
#define CMK_SIGHOLD_IS_A_BUILTIN 

/* #define CMK_RSH_USE_REMSH */
#define CMK_RSH_IS_A_COMMAND

/* #define CMK_TIMER_USE_GETRUSAGE */
#define CMK_TIMER_USE_TIMES

/* #define CMK_ASYNC_USE_SIOCGPGRP_AND_FIOASYNC */
#define CMK_ASYNC_USE_SETOWN_AND_SETFL

/* #define CMK_SIGNAL_USE_SIGACTION */
/* #define CMK_SIGNAL_IS_A_BUILTIN */
#define CMK_SIGNAL_USE_SIGACTION_AND_SIGEMPTYSET

#define CMK_MAX_DGRAM_SIZE 4096

/* #define CMK_STRERROR_IS_A_BUILTIN */
#define CMK_STRERROR_USE_SYS_ERRLIST

/* #define CMK_HAVE_STRINGS_H */
/* #define CMK_JUST_DECLARE_STRING_FNS */
#define CMK_HAVE_STRING_H

/* #define CMK_HAVE_WAITFLAGS_H */
#define CMK_HAVE_SYS_WAIT_H

#define CMK_MACHINE_NAME "NETWORK"


#endif
