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
 * Revision 2.0  1995-06-14 16:27:49  brunner
 * HP/gcc port for new directory structure
 *
 *
 ***************************************************************************/

#ifndef _CONV_MACH_H
#define _CONV_MACH_H

/* #define CMK_CMIMYPE_IS_A_BUILTIN */
#define CMK_CMIMYPE_IS_A_VARIABLE

#define CMK_CMIPRINTF_IS_A_BUILTIN
/* #define CMK_CMIPRINTF_IS_JUST_PRINTF */

/***************************************************************************/
/* All flags after this line are currently used only by the Common.net     */
/***************************************************************************/

/* #define CMK_SIGHOLD_IS_A_BUILTIN */
#define CMK_SIGHOLD_USE_SIGMASK

/* #define CMK_RSH_IS_A_COMMAND */
#define CMK_RSH_USE_REMSH

/* #define CMK_TIMER_USE_GETRUSAGE */
#define CMK_TIMER_USE_TIMES

/* #define CMK_ASYNC_USE_SETOWN_AND_SETFL */
#define CMK_ASYNC_USE_SIOCGPGRP_AND_FIOASYNC

/* #define CMK_SIGNAL_IS_A_BUILTIN */
#define CMK_SIGNAL_USE_SIGACTION
/* #define CMK_SIGNAL_USE_SIGACTION_AND_SIGEMPTYSET */

#define CMK_MAX_DGRAM_SIZE 4096

/* #define CMK_STRERROR_IS_A_BUILTIN */
#define CMK_STRERROR_USE_SYS_ERRLIST

#define CMK_HAVE_STRING_H
/* #define CMK_HAVE_STRINGS_H */
/* #define CMK_JUST_DECLARE_STRING_FNS */

#define CMK_HAVE_SYS_WAIT_H
/* #define CMK_HAVE_WAITFLAGS_H */

#define CMK_NO_SHARED_VARS_AT_ALL
/* #define CMK_SHARED_VARS_EXEMPLAR */

#if 0
   typedef unsigned char        u_char;    /* Try to avoid using these */
   typedef unsigned short       u_short;   /* Try to avoid using these */
   typedef unsigned int         u_int;     /* Try to avoid using these */
   typedef unsigned long        u_long;    /* Try to avoid using these */
   typedef unsigned int         uint;      /* Try to avoid using these */
   typedef unsigned short       ushort;    /* Try to avoid using these */
   typedef unsigned char  ubit8;
   typedef unsigned short ubit16;
   typedef unsigned long  ubit32;
   typedef char           sbit8;
   typedef short          sbit16;
   typedef long           sbit32;

#define enum_t int
typedef char *caddr_t; /* same as in types.h */
#define bool_t int
#endif /* 0 */

#endif
