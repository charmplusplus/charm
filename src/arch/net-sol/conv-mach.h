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
 * Revision 2.0  1995-06-08 16:44:52  gursoy
 * Reorganized directory structure
 * ,
 *
 ***************************************************************************/

/* #define CMK_CMIMYPE_IS_A_BUILTIN */
   #define CMK_CMIMYPE_IS_A_VARIABLE

   #define CMK_CMIPRINTF_IS_A_BUILTIN
/* #define CMK_CMIPRINTF_IS_JUST_PRINTF */

/***************************************************************************/
/* All flags after this line are currently used only by the Common.net     */
/***************************************************************************/

   #define CMK_SIGHOLD_IS_A_BUILTIN
/* #define CMK_SIGHOLD_USE_SIGMASK */

   #define CMK_RSH_IS_A_COMMAND
/* #define CMK_RSH_USE_REMSH */

/* #define CMK_TIMER_USE_GETRUSAGE  */
   #define CMK_TIMER_USE_TIMES 

   #define CMK_ASYNC_USE_SETOWN_AND_SETFL
/* #define CMK_ASYNC_USE_SIOCGPGRP_AND_FIOASYNC */

/* #define CMK_SIGNAL_IS_A_BUILTIN */
   #define CMK_SIGNAL_USE_SIGACTION 
/* #define CMK_SIGNAL_USE_SIGACTION_AND_SIGEMPTYSET */

   #define CMK_MAX_DGRAM_SIZE 1024

   #define CMK_STRERROR_IS_A_BUILTIN
/* #define CMK_STRERROR_USE_SYS_ERRLIST */

/* #define CMK_NEED_DECLARATION_FOR_STRING_FNS */

   #define CMK_HAVE_SYS_WAIT_H 
/* #define CMK_HAVE_WAITFLAGS_H */

   #define CMK_HAVE_STRING_H
/* #define CMK_HAVE_STRINGS_H */

   #define CMK_NO_SHARED_VARS_AT_ALL

