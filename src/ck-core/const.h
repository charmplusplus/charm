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
 * Revision 2.13  1997-07-18 21:21:04  milind
 * all files of the form perf-*.c have been changed to trace-*.c, with
 * name expansions. For example, perf-proj.c has been changed to
 * trace-projections.c.
 * performance.h has been renamed as trace.h, and perfio.c has been
 * renamed as traceio.c.
 * Corresponding changes have been made in the Makefile too.
 * Earlier, there used to be three libck-core-*.a where * was projections,
 * summary or none. Now, there will be a single libck-core.a and
 * three libck-trace-*.a where *=projections, summary and none.
 * The execmode parameter to charmc script has been renamed as
 * tracemode.
 * Also, the perfModuleInit function has been renamed as traceModuleInit,
 * RecdPerfMsg => RecdTraceMsg
 * CollectPerfFromNodes => CollectTraceFromNodes
 *
 * Revision 2.12  1996/08/01 21:10:13  jyelon
 * *** empty log message ***
 *
 * Revision 2.11  1995/09/20 14:24:27  jyelon
 * *** empty log message ***
 *
 * Revision 2.10  1995/09/01  02:13:17  jyelon
 * VID_BLOCK, CHARE_BLOCK, BOC_BLOCK consolidated.
 *
 * Revision 2.9  1995/07/27  20:29:34  jyelon
 * Improvements to runtime system, general cleanup.
 *
 * Revision 2.8  1995/07/25  00:29:31  jyelon
 * *** empty log message ***
 *
 * Revision 2.7  1995/07/24  01:54:40  jyelon
 * *** empty log message ***
 *
 * Revision 2.6  1995/07/22  23:44:13  jyelon
 * *** empty log message ***
 *
 * Revision 2.5  1995/07/19  22:15:33  jyelon
 * *** empty log message ***
 *
 * Revision 2.4  1995/07/12  16:28:45  jyelon
 * *** empty log message ***
 *
 * Revision 2.3  1995/06/29  21:31:10  narain
 * Took out the LdbBocNum macro and made it a Extern in globals.h
 *
 * Revision 2.2  1995/06/20  14:50:24  gursoy
 * removed SHARED_DECL
 *
 * Revision 2.1  1995/06/08  17:07:12  gursoy
 * Cpv macro changes done
 *
 * Revision 1.3  1995/05/03  20:57:54  sanjeev
 * bug fixes for finding uninitialized modules
 *
 * Revision 1.2  1994/11/11  05:31:12  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:34  brunner
 * Initial revision
 *
 ***************************************************************************/
#ifndef CONST_H
#define CONST_H


/* constants for denoting illegal values (used as flags) */
#define NULL_VID       NULL
#define NULL_PE        NULL
#define NULL_PACK_ID    0  

/* Constant values for PE */
#define CK_PE_SPECIAL(x) ((x)>=0xFFF0)
#define CK_PE_ALL        (0xFFF0)
#define CK_PE_ALL_BUT_ME (0xFFF1)
#define CK_PE_ANY        (0xFFF2)
#define CK_PE_INVALID    (0xFFF3)

/* Constant values for queueing in envelope */
#define CK_QUEUEING_FIFO  CQS_QUEUEING_FIFO
#define CK_QUEUEING_LIFO  CQS_QUEUEING_LIFO
#define CK_QUEUEING_IFIFO CQS_QUEUEING_IFIFO
#define CK_QUEUEING_ILIFO CQS_QUEUEING_ILIFO
#define CK_QUEUEING_BFIFO CQS_QUEUEING_BFIFO
#define CK_QUEUEING_BLIFO CQS_QUEUEING_BLIFO

/* define the System BocNum's corresponding to system BOCs
   e.g LoadBalancing, Quiescence, Also Increment the NumSysBoc
   defined below */

/* load-balancing    - this is now an integer    */

#define QDBocNum  1    /* Quiescecence Detection */
#define WOVBocNum 2    /* write once variables   */
#define TblBocNum 3    /* dynamic table boc      */
#define DynamicBocNum 4 /* to manage dynamic boc  */
#define StatisticBocNum 5 /* to manage statistics */

/* define NumSysBoc as the number of system Bocs: Increment this
   as more system BOCs are added.
   1 boc is for load balancing 
   1 boc is for quiescence detection
   1 boc is for write once variables
   1 boc is for dynamic tables      */

#define MaxBocs		      100
#define PSEUDO_Max	      20
#define NumSysBoc             6

#define MainInitEp            0
#define NumHostSysEps         0
#define NumNodeSysEps         0

/* MsgCategories */
/* at the moment only vaguely defined. Use will be clearer later, if needed */
#define	IMMEDIATEcat	0
#define USERcat    	1


#define CHAREKIND_CHARE    0   /* Plain old chare */
#define CHAREKIND_BOCNODE  1   /* BOC node */
#define CHAREKIND_UVID     2   /* Unfilled-VID */
#define CHAREKIND_FVID     3   /* Filled-VID */

/* MsgTypes */
/* The trace tools use these also. */
/**********		USERcat			*******/
#define NewChareMsg  		0
#define NewChareNoBalanceMsg    1
#define ForChareMsg  		2
#define BocInitMsg   		3
#define BocMsg       		4
#define TerminateToZero 	5   /* never used??? */
#define TerminateSys		6   /* never used??? */
#define InitCountMsg 		7
#define ReadVarMsg   		8
#define ReadMsgMsg 		9
#define BroadcastBocMsg 	10
#define DynamicBocInitMsg 	11

/**********		IMMEDIATEcat		*******/
#define LdbMsg			12  /* never used??? */
#define VidSendOverMsg          13
#define QdBocMsg		14
#define QdBroadcastBocMsg	15
#define ImmBocMsg               16
#define ImmBroadcastBocMsg      17
#define InitBarrierPhase1       18
#define InitBarrierPhase2       19

#endif
