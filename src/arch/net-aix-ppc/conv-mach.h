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
 * Revision 1.1  1995-09-14 21:25:37  jyelon
 * Initial revision
 *
 * Revision 2.3  1995/07/05  21:33:36  brunner
 * Took spaces out from before #defines
 *
 ***************************************************************************/

/* #define CMK_CMIMYPE_IS_A_BUILTIN */
#define CMK_CMIMYPE_IS_A_VARIABLE

#define CMK_CMIPRINTF_IS_A_BUILTIN
/* #define CMK_CMIPRINTF_IS_JUST_PRINTF */

/***************************************************************************/
/* All flags after this line are currently used only by the Common.net     */
/***************************************************************************/

/* #define CMK_SIGHOLD_IS_A_BUILTIN */
#define CMK_SIGHOLD_USE_SIGMASK

#define CMK_RSH_IS_A_COMMAND
/* #define CMK_RSH_USE_REMSH */

#define CMK_TIMER_USE_GETRUSAGE
/* #define CMK_TIMER_USE_TIMES */

#define CMK_ASYNC_USE_SETOWN_AND_SETFL
/* #define CMK_ASYNC_USE_SIOCGPGRP_AND_FIOASYNC */

#define CMK_SIGNAL_IS_A_BUILTIN
/* #define CMK_SIGNAL_USE_SIGACTION */
/* #define CMK_SIGNAL_USE_SIGACTION_AND_SIGEMPTYSET */

#define CMK_MAX_DGRAM_SIZE 4096

/* #define CMK_STRERROR_IS_A_BUILTIN */
#define CMK_STRERROR_USE_SYS_ERRLIST

/* #define CMK_HAVE_STRINGS_H */
/* #define CMK_HAVE_STRING_H */
#define CMK_JUST_DECLARE_STRING_FNS

#define CMK_HAVE_SYS_WAIT_H
/* #define CMK_HAVE_WAITFLAGS_H */

#define CMK_NO_SHARED_VARS_AT_ALL
/* #define CMK_SHARED_VARS_EXEMPLAR */

#define CMK_COMPILER_LIKES_STATIC_PROTO
/* #define CMK_COMPILER_HATES_STATIC_PROTO */

