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
 * Revision 2.0  1995-06-02 17:27:40  brunner
 * Reorganized directory structure
 *
 * Revision 1.4  1995/04/13  20:55:50  sanjeev
 * Changed Mc to Cmi
 *
 * Revision 1.3  1995/04/02  00:48:02  sanjeev
 * changes for separating Converse
 *
 * Revision 1.2  1994/12/01  23:57:00  sanjeev
 * interop stuff
 *
 * Revision 1.1  1994/11/03  17:38:55  brunner
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";
#include "chare.h"
#include "globals.h"
#include "performance.h"
#include "vid.h"


VidBocInit()
{
    BOC_BLOCK *bocBlock;

    bocBlock = (BOC_BLOCK *) CreateBocBlock(sizeof(DATA_BR_VID));
    bocBlock->boc_num = VidBocNum;
    SetBocDataPtr(VidBocNum, (void *) (bocBlock + 1));
}



/************************************************************************/
/*			VidQueueUpInVidBlock				*/
/*	The message msgPtr has been sent to a chare created with a	*/
/*	virtual id. The address of the virtual id block is available	*/
/*	as part of the message, and is used to determine whether or	*/
/*	not the chare has been created. If the chare hasn't been	*/
/*	created, then msgPtr is queued up in the virtual block, else	*/
/*	the message is forwarded to the processor on which the chare	*/
/*	was created.							*/
/************************************************************************/

VidQueueUpInVidBlock(msgPtr, data_area)
void *msgPtr;
void *data_area;
{
    ENVELOPE * env = ENVELOPE_UPTR(msgPtr);
    VID_BLOCK * vidblock;
    void *vidqueue;

    vidblock = (VID_BLOCK *) GetEnv_vidBlockPtr(env);
    vidqueue = vidblock->info_block.vid_queue;
    if (vidblock->vidPenum == -1)
	FIFO_EnQueue(vidqueue, env);
    else
    {
	SetEnv_category(env,USERcat);
	SetEnv_msgType(env,ForChareMsg);
	SetEnv_destPE(env,vidblock->vidPenum);
	SetEnv_chareBlockPtr(env, (int) vidblock->info_block.chareBlockPtr);
	SetEnv_chare_magic_number(env, vidblock->chare_magic_number);
	trace_creation(GetEnv_msgType(env), GetEnv_EP(env), env);
	CkCheck_and_Send(env, GetEnv_EP(env));
	QDCountThisCreation(GetEnv_EP(env), USERcat, ForChareMsg, 1);
    }
}

/************************************************************************/
/*			VidSendOverMessage				*/
/*	Once the chare ha been created it needs to get the messages	*/
/*	that were sent to it while it hadn't been created. These	*/
/* 	messages are queued up in its virtual id block, whose address	*/
/* 	it has available. The chare on creation sends a messages to	*/
/* 	the branch office chare, on which the virtual block resides	*/
/* 	asking it to send over the messages to it. The messages are 	*/
/* 	then dequeued and sent over to the processor on which the	*/
/* 	was finally created. 						*/
/************************************************************************/

VidSendOverMessages(msgPtr, data_area)
CHARE_ID_MSG *msgPtr;
void *data_area;
{
    CHARE_BLOCK 	*cblock = msgPtr->dataPtr;
    ENVELOPE 		*env = ENVELOPE_UPTR(msgPtr);
    VID_BLOCK 		*vidblock = (VID_BLOCK *) GetEnv_vidBlockPtr(env);
    void 		*vidqueue;
    int 		magic_number = msgPtr->chare_magic_number;

    vidblock->vidPenum = GetEnv_onPE(env);
    vidblock->chare_magic_number = magic_number;
    vidqueue = vidblock->info_block.vid_queue;
    while (!FIFO_Empty(vidqueue))
    {
	FIFO_DeQueue(vidqueue, &env);
	SetEnv_category(env, USERcat);
	SetEnv_msgType(env, ForChareMsg);
	SetEnv_destPE(env, vidblock->vidPenum);
	SetEnv_chareBlockPtr(env, (int) cblock);
	SetEnv_chare_magic_number(env, magic_number);
	trace_creation(GetEnv_msgType(env), GetEnv_EP(env), env);
	CkCheck_and_Send(env, GetEnv_EP(env));
	QDCountThisCreation(GetEnv_EP(env), USERcat, ForChareMsg, 1);
   }
   vidblock->info_block.chareBlockPtr = cblock; 
}


VidAddSysBocEps()
{
   EpTable[VidQueueUpInVidBlock_EP] = VidQueueUpInVidBlock;
   EpTable[VidSendOverMessages_EP] = VidSendOverMessages;
}


/************************************************************************/
/*			VidSend						*/
/*	This call is replication of the SendMsgBranch call, but sends 	*/
/* 	the message with the category field as VIDcat.			*/
/************************************************************************/

VidSend(chareblockPtr,destPE,vidPtr)
CHARE_BLOCK * chareblockPtr;
PeNumType   destPE;
VID_BLOCK *vidPtr;
{
    CHARE_ID_MSG * msg;
    ENVELOPE * env;

    msg  = (CHARE_ID_MSG *) CkAllocMsg(sizeof(CHARE_ID_MSG));
    CkMemError(msg);
    env = ENVELOPE_UPTR(msg);
    SetEnv_destPE(env, destPE);
    SetEnv_category(env, IMMEDIATEcat);
    SetEnv_vidEP(env, VidSendOverMessages_EP);
    SetEnv_msgType(env, VidMsg);
    SetEnv_destPeFixed(env, 1);
    SetEnv_onPE(env, CmiMyPe());
    SetEnv_vidBlockPtr(env, (int)  vidPtr);

    msg->dataPtr = chareblockPtr;
    msg->chare_magic_number =
	GetID_chare_magic_number(currentChareBlock->selfID);

    QDCountThisCreation(GetEnv_vidEP(env), IMMEDIATEcat, VidMsg, 1);
    trace_creation(GetEnv_msgType(env), GetEnv_EP(env), env);
    CkCheck_and_Send(env, GetEnv_vidEP(env));
}
