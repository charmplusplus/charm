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
 * Revision 2.5  1995-07-19 22:15:33  jyelon
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
#define NULL_PACK_ID    0  
#define NULL_PE        -2   /* -1 indicates all nodes, so use -2 */

/* Constant values for destPE in envelope */
#define ALL_NODES           (-1)
#define ALL_NODES_EXCEPT_ME (-2)

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
#define VidBocNum 2    /* virtual id's           */
#define WOVBocNum 3    /* write once variables   */
#define TblBocNum 4    /* dynamic table boc      */
#define DynamicBocNum 5 /* to manage dynamic boc  */
#define StatisticBocNum 6 /* to manage statistics */

/* define NumSysBoc as the number of system Bocs: Increment this
   as more system BOCs are added.
   1 boc is for load balancing 
   1 boc is for quiescence detection
   1 boc is for virtual ids  
   1 boc is for write once variables
   1 boc is for dynamic tables      */

#define MaxBocs		      100
#define PSEUDO_Max	      20
#define NumSysBoc             7

#define MainInitEp            0
#define NumHostSysEps         0
#define NumNodeSysEps         0

/* Note : Entry Point 0 is the _CkNullEP  --Sanjeev 5/3/95 */

/* Entry points for Quiescence detection BOC 	*/
#define QDInsertQuiescenceList_EP 					1
#define QDHost_EndPhI_EP      						2
#define QDHost_EndPhII_EP     						3
#define QDInit_EP             						4
#define QDPhaseIBroadcast_EP  						5
#define QDPhaseIMsg_EP        						6
#define QDPhaseIIBroadcast_EP 						7
#define QDPhaseIIMsg_EP       						8

/* Entry points for VID BOC			*/
#define VidQueueUpInVidBlock_EP 					9 
#define VidSendOverMessages_EP  					10

/* Entry points for Write Once Variables 	*/
#define NodeAddWOV_EP        						11 
#define NodeRcvAck_EP         						12
#define HostAddWOV_EP        						13 
#define HostRcvAck_EP         						14

/* Entry points for dynamic tables BOC    	*/
#define TblUnpack_EP								15

/* Entry points for accumulator BOC		*/
#define ACC_CollectFromNode_EP						16
#define ACC_LeafNodeCollect_EP 						17
#define ACC_InteriorNodeCollect_EP 					18
#define ACC_BranchInit_EP							19

/* Entry points for monotonic BOC		*/
#define MONO_BranchInit_EP							20
#define MONO_BranchUpdate_EP						21
#define MONO_ChildrenUpdate_EP						22

/* These are the entry points necessary for the dynamic BOC creation. */
#define RegisterDynamicBocInitMsg_EP 				23
#define OtherCreateBoc_EP							24
#define InitiateDynamicBocBroadcast_EP 				25

/* These are the entry points for the statistics BOC */
#define StatCollectNodes_EP      					26
#define StatData_EP	      							27
#define StatPerfCollectNodes_EP      				28
#define StatBroadcastExitMessage_EP 				29
#define StatExitMessage_EP 							30

/* Entry points for LoadBalancing BOC 		*/
#define LdbNbrStatus_EP       						31



/* Total Number of system BOC entry points (numbers 1 through 31) */
#define NumSysBocEps        						31

/* MsgCategories */
/* at the moment only vaguely defined. Use will be clearer later, if needed */
#define	IMMEDIATEcat	0
#define USERcat    	1


/* MsgTypes */
/* The performance tools use these also. */
/**********		USERcat			*******/
#define NewChareMsg  		0
#define ForChareMsg  		1
#define BocInitMsg   		2
#define BocMsg       		3
#define TerminateToZero 	4
#define TerminateSys		5
#define InitCountMsg 		6
#define ReadVarMsg   		7
#define ReadMsgMsg 		8
#define BroadcastBocMsg 	9
#define DynamicBocInitMsg 	10

/**********		IMMEDIATEcat		*******/
#define LdbMsg			12
#define VidMsg			13
#define QdBocMsg		14
#define QdBroadcastBocMsg	15


#endif
