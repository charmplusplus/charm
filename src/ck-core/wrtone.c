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
 * Revision 2.7  1997-10-29 23:52:55  milind
 * Fixed CthInitialize bug on uth machines.
 *
 * Revision 2.6  1995/09/07 21:21:38  jyelon
 * Added prefixes to Cpv and Csv macros, fixed bugs thereby revealed.
 *
 * Revision 2.5  1995/09/06  21:48:50  jyelon
 * Eliminated 'CkProcess_BocMsg', using 'CkProcess_ForChareMsg' instead.
 *
 * Revision 2.4  1995/09/01  02:13:17  jyelon
 * VID_BLOCK, CHARE_BLOCK, BOC_BLOCK consolidated.
 *
 * Revision 2.3  1995/07/27  20:29:34  jyelon
 * Improvements to runtime system, general cleanup.
 *
 * Revision 2.2  1995/07/24  01:54:40  jyelon
 * *** empty log message ***
 *
 * Revision 2.1  1995/06/08  17:07:12  gursoy
 * Cpv macro changes done
 *
 * Revision 1.3  1995/04/13  20:55:22  sanjeev
 * Changed Mc to Cmi
 *
 * Revision 1.2  1994/12/01  23:58:06  sanjeev
 * interop stuff
 *
 * Revision 1.1  1994/11/03  17:38:59  brunner
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";
/*****************************************************************************
 
                   Write once variable system boc (branch part)

*****************************************************************************/

#include "chare.h"
#include "globals.h"
#include "wrtone.h"

extern CHARE_BLOCK *CreateChareBlock();
void HostReceiveAcknowledge();


/*****************************************************************************
*****************************************************************************/
void WOVBocInit()
{
	CHARE_BLOCK *bocBlock;
	WOV_Boc_Data *bocData;
	int       i;

	TRACE(CmiPrintf("[%d]: WOVBocInit() called\n", CmiMyPe()));

	bocBlock = CreateChareBlock(sizeof(WOV_Boc_Data),CHAREKIND_BOCNODE,0);
        bocBlock->x.boc_num = WOVBocNum;
	SetBocBlockPtr(WOVBocNum, bocBlock);

	bocData  = (WOV_Boc_Data *) (bocBlock + 1);

	for(i = 0; i < MAXWRITEONCEVARS; i++)
		bocData->WOVArray[i].numAcks =
				 CmiNumSpanTreeChildren(CmiMyPe());
}


/*****************************************************************************
*****************************************************************************/
void NodeAddWriteOnceVar(msgptr_,localdataptr_)
void *msgptr_,*localdataptr_;
{
	Node_New_WOV_Msg  *newWovMsg = (Node_New_WOV_Msg *) msgptr_;
	WOV_Boc_Data *bocData   = (WOV_Boc_Data *) localdataptr_;
	Ack_To_Host_Msg   *ackMsg;


	/* instead of removing data from message just point to data IN msg */
	bocData->WOVArray[(int) newWovMsg->wovID].wovData = 
	    	(char *) ( (Node_New_WOV_Msg *) newWovMsg + 1);

	/* now we need to acknowledge the fact that we have created the WOV */
	if(isLeaf(CmiMyPe())) {
	        ackMsg = (Ack_To_Host_Msg *)CkAllocMsg(sizeof(Ack_To_Host_Msg));
		CkMemError(ackMsg);
		ackMsg->wovID = newWovMsg->wovID;

		if(CmiMyPe() == CmiSpanTreeRoot()) {
TRACE(CmiPrintf("[%d]:: NodeWrtOnceVar...Ack to Host for wovID %d\n",
				CmiMyPe() ,ackMsg->wovID);)
			HostReceiveAcknowledge(ackMsg, bocData);
		}
		else
		{
	       /* Note: we add TotalHostBocEps to deal with weird decrement */
TRACE(CmiPrintf("[%d]:: NodeWrtOnceVar...Ack to Parent %d for wovID %d\n",
			    CmiMyPe(), CmiSpanTreeParent(CmiMyPe()),
			    ackMsg->wovID);)
			GeneralSendMsgBranch(CsvAccess(CkEp_WOV_RcvAck), ackMsg, 
				 CmiSpanTreeParent(CmiMyPe()),
				 ImmBocMsg, WOVBocNum);
		}
	}
}


/*****************************************************************************
*****************************************************************************/
void NodeReceiveAcknowledge(msgptr_,localdataptr_)
void *msgptr_,*localdataptr_;
{
	Ack_To_Host_Msg   *theMsg  = (Ack_To_Host_Msg *) msgptr_;
	WOV_Boc_Data *bocData = (WOV_Boc_Data *) localdataptr_;
	Ack_To_Host_Msg   *ackMsg;  /* msg to pass up the tree */
	WriteOnceID wovID;

	wovID = theMsg->wovID;
	bocData->WOVArray[(int) wovID].numAcks--; 


TRACE(CmiPrintf("[%d]:: NodeRecvAck...wovID = %d, kids left = %d\n",
		CmiMyPe(), wovID, bocData->WOVArray[(int) wovID].numAcks));
   	if(!(bocData->WOVArray[(int) wovID].numAcks)) 
	{      
		/* if we've got all acks */
		ackMsg = (Ack_To_Host_Msg *)
				 CkAllocMsg(sizeof(Ack_To_Host_Msg));
		CkMemError(ackMsg);
		ackMsg->wovID = theMsg->wovID;

		if(CmiMyPe() == CmiSpanTreeRoot())
			    HostReceiveAcknowledge(ackMsg, bocData);
		else
			    GeneralSendMsgBranch(CsvAccess(CkEp_WOV_RcvAck), ackMsg, 
				CmiSpanTreeParent(CmiMyPe()),
				ImmBocMsg, WOVBocNum);
	}
	else 
		CkFreeMsg(msgptr_);
}

/*****************************************************************************
*****************************************************************************/
void *DerefWriteOnce(ID)
WriteOnceID ID;
{
	WOV_Boc_Data * localBocData;


	localBocData = (WOV_Boc_Data *) GetBocDataPtr(WOVBocNum);
	return(localBocData->WOVArray[(int) ID].wovData);
}


/***************************************************************************
***************************************************************************/
void HostAddWriteOnceVar(msgptr_,localdataptr_)
void *msgptr_,*localdataptr_;
{
	Host_New_WOV_Msg  *newWov   = (Host_New_WOV_Msg *)  msgptr_;
	WOV_Boc_Data *localBocData  = (WOV_Boc_Data *) localdataptr_;
	ENVELOPE *env;
	int i, nodeMsgSize;

	Node_New_WOV_Msg *msgForNodes; /* going to broadcast this to nodes */
	char *src,*dest;               /* just indices to copy bytes       */


	if (localBocData->numWOVs > MAXWRITEONCEVARS)
	{
		CmiPrintf("*** ERROR *** Exceeded permitted number of WriteOnce Variables.\n");
		CkExit();
	}
	localBocData->WOVArray[localBocData->numWOVs].ep      = newWov->ep;
	localBocData->WOVArray[localBocData->numWOVs].cid     = newWov->cid;
	localBocData->WOVArray[localBocData->numWOVs].wovSize = newWov->wovSize;
	localBocData->WOVArray[localBocData->numWOVs].wovData = 
	    (char *) (( Host_New_WOV_Msg *) newWov + 1);

	nodeMsgSize = sizeof(Node_New_WOV_Msg) + newWov->wovSize;
	msgForNodes = (Node_New_WOV_Msg *) CkAllocMsg(nodeMsgSize);
	CkMemError(msgForNodes);

	msgForNodes->wovID   = (WriteOnceID) localBocData->numWOVs;
	msgForNodes->wovSize = newWov->wovSize;

	src  = (char *) ((Host_New_WOV_Msg  *) newWov + 1);
	dest = (char *) ((Node_New_WOV_Msg  *) msgForNodes + 1); 
	for(i = 0; i < newWov->wovSize; i++)  
		*dest++ = *src++; /* copy data into the message struct */

TRACE(CmiPrintf("Host::  HostAddWriteOnceVar...wovID %d, size %d\n",
			msgForNodes->wovID, newWov->wovSize);)

	GeneralBroadcastMsgBranch(CsvAccess(CkEp_WOV_AddWOV), msgForNodes,
			ImmBroadcastBocMsg, WOVBocNum);

	localBocData->numWOVs++;            /* weve got one more wov */
}


/***************************************************************************
 Here we receive the acknowledgement from our child node (the root of the 
 spanning tree) that ALL the nodes have indeed created the write once variable
 in question. We can then send a message to the user program on the originating
 node, telling the user that it is now ok to access the WOV, or to broadcast
 the id to other nodes, or whatever.  ***************************************************************************/ 
void HostReceiveAcknowledge(msgptr_,localdataptr_)
void *msgptr_,*localdataptr_;
{
	Ack_To_Host_Msg *ackMessage    = (Ack_To_Host_Msg *) msgptr_;
	WOV_Boc_Data    *localBocData  = (WOV_Boc_Data *) localdataptr_;
	Return_To_Origin_Msg *msgForCreator;
	WriteOnceID          wovID;

TRACE(CmiPrintf("[%d]:: HostRecvAck...wovID %d, Notifying user at ep [%d]\n",
		CmiMyPe(),
	    	ackMessage->wovID,
		localBocData->WOVArray[(int)ackMessage->wovID].ep);)

	msgForCreator = (Return_To_Origin_Msg *)
				CkAllocMsg(sizeof(Return_To_Origin_Msg));
	CkMemError(msgForCreator);
	wovID = msgForCreator->wovID = ackMessage->wovID;

	SendMsg(localBocData->WOVArray[(int) wovID].ep,msgForCreator,
	    &localBocData->WOVArray[(int) wovID].cid);
}


/***************************************************************************
If we call WriteOnce from the host, we don't need to send a message, just
call the appropriate routine.
***************************************************************************/
void WriteOnce(dataPtr,dataSize,ep,cid)
char           *dataPtr;
int            dataSize;
EntryPointType ep;
ChareIDType    cid;
{
	Host_New_WOV_Msg *newWov;
	int msgSize,i;
	char *src,*dest;

	TRACE(CmiPrintf("Node %d : WriteOnce called.\n", CmiMyPe()));
	msgSize = dataSize + sizeof(Host_New_WOV_Msg);
	newWov = (Host_New_WOV_Msg *) CkAllocMsg(msgSize);
	CkMemError(newWov);

	newWov->cid = cid;
	newWov->ep = ep;            /* fill up the fields of the struct */
	newWov->wovSize = dataSize;
	src  = dataPtr; 
	dest = (char *) ((Host_New_WOV_Msg *) newWov + 1);
	for(i = 0; i < dataSize; i++)
		*dest++ = *src++;
	/* here we bypass the SendMsg function and just call directly */
	if (CmiMyPe() == 0)
		HostAddWriteOnceVar((void *)newWov,(void *)GetBocDataPtr(WOVBocNum));
	else
		GeneralSendMsgBranch(CsvAccess(CkEp_WOV_HostAddWOV), newWov,
		    0, ImmBocMsg, WOVBocNum);
}


void WOVAddSysBocEps(void)
{
  CsvAccess(CkEp_WOV_AddWOV) =
    registerBocEp("CkEp_WOV_AddWOV",
		  NodeAddWriteOnceVar,
		  CHARM, 0, 0);
  CsvAccess(CkEp_WOV_RcvAck) =
    registerBocEp("CkEp_WOV_RcvAck",
		  NodeReceiveAcknowledge,
		  CHARM, 0, 0);
  CsvAccess(CkEp_WOV_HostAddWOV) =
    registerBocEp("CkEp_WOV_HostAddWOV",
		  HostAddWriteOnceVar,
		  CHARM, 0, 0);
  CsvAccess(CkEp_WOV_HostRcvAck) =
    registerBocEp("CkEp_WOV_HostRcvAck",
		  HostReceiveAcknowledge,
		  CHARM, 0, 0);
}

