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
 * Revision 2.5  1995-07-24 01:54:40  jyelon
 * *** empty log message ***
 *
 * Revision 2.4  1995/07/22  23:45:15  jyelon
 * *** empty log message ***
 *
 * Revision 2.3  1995/07/05  22:51:39  sanjeev
 * added comments at top
 *
 * Revision 2.2  1995/07/05  22:11:59  sanjeev
 * put SetEnv_EP(env, 0) in VidSend() to fix CM5 bug
 *
 * Revision 2.1  1995/06/08  17:07:12  gursoy
 * Cpv macro changes done
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


/************************************************************************
Comments : 7/5/95 (after debugging bug on CM5 with Megatest++).
Bug was intermittent crash in TTAB and ACC.
Caused due to EP field not being filled in VidSend, so causing
EpLanguageTable to be indexed by 65535 in CallProcessMsg.
************************************************************************/




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
	SetEnv_msgType(env,ForChareMsg);
	SetEnv_chareBlockPtr(env, (int) vidblock->info_block.chareBlockPtr);
	SetEnv_chare_magic_number(env, vidblock->chare_magic_number);
	trace_creation(GetEnv_msgType(env), GetEnv_EP(env), env);
	CkCheck_and_Send(vidblock->vidPenum, env);
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
    ChareIDType         ID=msgPtr->ID;
    ENVELOPE 		*env = ENVELOPE_UPTR(msgPtr);
    VID_BLOCK 		*vidblock = (VID_BLOCK *) GetEnv_vidBlockPtr(env);
    void 		*vidqueue;
    int 		chare_magic = GetID_chare_magic_number(ID);
    int                 chare_pe    = GetID_onPE(ID);
    CHARE_BLOCK        *chare_block = GetID_chareBlockPtr(ID);

    vidqueue = vidblock->info_block.vid_queue;
    while (!FIFO_Empty(vidqueue))
    {
	FIFO_DeQueue(vidqueue, &env);
	SetEnv_msgType(env, ForChareMsg);
	SetEnv_chareBlockPtr(env, chare_block);
	SetEnv_chare_magic_number(env, chare_magic);
	trace_creation(GetEnv_msgType(env), GetEnv_EP(env), env);
	CkCheck_and_Send(vidblock->vidPenum, env);
	QDCountThisCreation(GetEnv_EP(env), USERcat, ForChareMsg, 1);
   }
   vidblock->vidPenum = chare_pe;
   vidblock->chare_magic_number = chare_magic;
   vidblock->info_block.chareBlockPtr = chare_block;
}


VidAddSysBocEps()
{
   CsvAccess(EpTable)[VidQueueUpInVidBlock_EP] = VidQueueUpInVidBlock;
   CsvAccess(EpTable)[VidSendOverMessages_EP] = VidSendOverMessages;
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

    msg = (CHARE_ID_MSG *)CkAllocMsg(sizeof(CHARE_ID_MSG));
    msg->ID = CpvAccess(currentChareBlock)->selfID;
    CkMemError(msg);
    env = ENVELOPE_UPTR(msg);
    SetEnv_msgType(env, VidSendOverMsg);
    SetEnv_vidBlockPtr(env, (int)  vidPtr);
    SetEnv_EP(env, 0);


    QDCountThisCreation(VidSendOverMessages_EP, IMMEDIATEcat, VidSendOverMsg, 1);
    trace_creation(GetEnv_msgType(env), GetEnv_EP(env), env);
    CkCheck_and_Send(destPE, env);
}
