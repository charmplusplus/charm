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
 * Revision 2.7  1995-10-13 18:17:44  jyelon
 * I can't even remember.
 *
 * Revision 2.6  1995/10/11  00:35:06  jyelon
 * Added MALLOC options.
 *
 * Revision 2.5  1995/09/30  15:04:25  jyelon
 * Added CMK_CTHINIT_IS_IN_MAIN, etc.
 *
 * Revision 2.4  1995/09/29  09:50:07  jyelon
 * CmiGet-->CmiDeliver, added protos, etc.
 *
 * Revision 2.3  1995/09/22  17:54:52  sanjeev
 * added CMK_MACHINE_NAME
 *
 * Revision 2.2  1995/09/20  13:22:58  jyelon
 * Added 'CMK_THREAD_' flags.
 *
 * Revision 2.1  1995/09/19  18:56:28  jyelon
 * added CMK_PREPROCESSOR_USES_ANSI_STANDARD_CONCATENATION and other goodies.
 *
 * Revision 2.0  1995/07/10  22:12:21  knauff
 * Initial revision
 *
 ***************************************************************************/

#ifndef _CONV_MACH_H
#define _CONV_MACH_H

/* #define CMK_USE_GNU_MALLOC */
/* #define CMK_USE_GNU_MALLOC_WITH_INTERRUPT_SUPPORT */
#define CMK_USE_OS_MALLOC
 
/* #define CMK_CTHINIT_IS_IN_MAIN */
#define CMK_CTHINIT_IS_IN_CONVERSEINIT

/* #define CMK_CSDEXITSCHEDULER_IS_A_FUNCTION */
#define CMK_CSDEXITSCHEDULER_SET_CSDSTOPFLAG

/* #define CMK_COMMHANDLE_IS_AN_INTEGER */
#define CMK_COMMHANDLE_IS_A_POINTER
 
/* #define CMK_USES_SPECIAL_CMIDELIVERS */ 
#define CMK_USES_COMMON_CMIDELIVERS

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

/* #define CMK_CMIPRINTF_IS_A_BUILTIN */
#define CMK_CMIPRINTF_IS_JUST_PRINTF

/* #define CMK_THREADS_USE_ALLOCA */
#define CMK_THREADS_UNAVAILABLE

#define CMK_MACHINE_NAME "SP1"

#endif
