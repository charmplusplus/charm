#include "charm.h"
#include "trace.h"

typedef struct data_brnch_vid {
	int dummy;
} DATA_BR_VID;

typedef struct chare_id_msg{
        ChareIDType ID;
} CHARE_ID_MSG;

void VidEnqueueMsg(env)
ENVELOPE *env;
{
  CHARE_BLOCK *vid = GetEnv_chareBlockPtr(env);
  void *vidqueue = vid->x.vid_queue;
  FIFO_EnQueue(vidqueue, env);
}

void VidForwardMsg(env)
ENVELOPE *env;
{
  CHARE_BLOCK *vid = GetEnv_chareBlockPtr(env);
  SetEnv_chareBlockPtr(env, GetID_chareBlockPtr(vid->x.realID));
  SetEnv_chare_magic_number(env, GetID_chare_magic_number(vid->x.realID));
  if(CpvAccess(traceOn))
    trace_creation(GetEnv_msgType(env), GetEnv_EP(env), env);
  CldEnqueue(GetID_onPE(vid->x.realID), env, CpvAccess(CkInfo_Index));
  QDCountThisCreation(1);
}

/************************************************************************/
/*			VidSendOverMessage				*/
/*	Once the charm.ha been created it needs to get the messages	*/
/*	that were sent to it while it hadn't been created. These	*/
/* 	messages are queued up in its virtual id block, whose address	*/
/* 	it has available. The messages are then dequeued and sent over  */
/*      to the processor on which the chare was finally created.        */
/************************************************************************/

void VidSendOverMessages(msgPtr, data_area)
CHARE_ID_MSG *msgPtr;
void *data_area;
{
    ChareIDType         ID;
    ENVELOPE 		*env;
    CHARE_BLOCK		*vidblock;
    void 		*vidqueue;
    int 		chare_magic;
    int                 chare_pe;
    CHARE_BLOCK        *chare_block;

    ID = msgPtr->ID;
    env = ENVELOPE_UPTR(msgPtr);
    vidblock = GetEnv_vidBlockPtr(env);
    chare_magic = GetID_chare_magic_number(ID);
    chare_pe = GetID_onPE(ID);
    chare_block  = GetID_chareBlockPtr(ID);

    if (vidblock->charekind != CHAREKIND_UVID) {
      CmiPrintf("system error #12983781\n");
      exit(1);
    }
    vidqueue = vidblock->x.vid_queue;
    while (!FIFO_Empty(vidqueue))
    {
	FIFO_DeQueue(vidqueue, &env);
	SetEnv_chareBlockPtr(env, chare_block);
	SetEnv_chare_magic_number(env, chare_magic);
        if(CpvAccess(traceOn))
	  trace_creation(GetEnv_msgType(env), GetEnv_EP(env), env);
	CldEnqueue(chare_pe, env, CpvAccess(CkInfo_Index));
	QDCountThisCreation(1);
   }
   FIFO_Destroy(vidqueue);
   vidblock->charekind = CHAREKIND_FVID;
   vidblock->x.realID = ID;
}


/************************************************************************/
/*			VidRetrieveMessages                             */
/*	This is used by the newly-created chare to request the          */
/*      messages that were stored in its VID                            */
/************************************************************************/

void VidRetrieveMessages(chareblockPtr,vidPE,vidBlockPtr)
CHARE_BLOCK * chareblockPtr;
PeNumType  vidPE;
CHARE_BLOCK *vidBlockPtr;
{
    CHARE_ID_MSG * msg;
    ENVELOPE * env;

    msg = (CHARE_ID_MSG *)CkAllocMsg(sizeof(CHARE_ID_MSG));
    CkMemError(msg);
    msg->ID = CpvAccess(currentChareBlock)->selfID;
    env = ENVELOPE_UPTR(msg);
    SetEnv_msgType(env, VidSendOverMsg);
    SetEnv_vidBlockPtr(env, vidBlockPtr);
    SetEnv_EP(env, 0);

    QDCountThisCreation(1);
    if(CpvAccess(traceOn))
      trace_creation(GetEnv_msgType(env), GetEnv_EP(env), env);
    CmiSetHandler(env, CpvAccess(HANDLE_INCOMING_MSG_Index));
    CldEnqueue(vidPE, env, CpvAccess(CkInfo_Index));
}
