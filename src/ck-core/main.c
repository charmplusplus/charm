/*************************************************************************/
/** This file now contains only the Charm/Charm++ part of the run-time.
    The core (scheduler/Converse) part is in converse.c 		 */
/*************************************************************************/

#include "charm.h"

#include "trace.h"

CHARE_BLOCK *GetBocBlockPtr();

extern void CkLdbSend();
void HANDLE_INCOMING_MSG();

void mainModuleInit()
{
}

void CheckMagicNumber(chare, env)
    CHARE_BLOCK *chare; ENVELOPE *env;
{
  if (GetID_chare_magic_number(chare->selfID) !=
      GetEnv_chare_magic_number(env)) {
    CmiPrintf("[%d] *** ERROR *** dead chare or bad chareID used at entry point %s.\n", 
	      CmiMyPe(), CsvAccess(EpInfoTable)[GetEnv_EP(env)].name);
    exit(1);
  }
}

void CkProcess_ForChareMsg_to_UVID(env)
ENVELOPE *env;
{
  if(CpvAccess(traceOn))
    trace_begin_execute(env);
  VidEnqueueMsg(env);
  if(CpvAccess(traceOn))
    trace_end_execute(0, ForChareMsg, 0);
  QDCountThisProcessing();
}

void CkProcess_ForChareMsg_to_FVID(env)
ENVELOPE *env;
{
  if(CpvAccess(traceOn))
    trace_begin_execute(env);
  VidForwardMsg(env);
  if(CpvAccess(traceOn))
    trace_end_execute(0, ForChareMsg, 0);
  QDCountThisProcessing();
}

void CkProcess_ForChareMsg_to_Chare(env)
ENVELOPE *env;
{
  CHARE_BLOCK *chareblock;
  int          current_ep;
  EP_STRUCT   *current_epinfo;
  void        *current_usr;
  int          current_magic;
  int          current_msgType;

  chareblock      = GetEnv_chareBlockPtr(env);
  current_ep      = GetEnv_EP(env);
  current_epinfo  = CsvAccess(EpInfoTable) + current_ep;
  current_usr     = USER_MSG_PTR(env);
  current_msgType = GetEnv_msgType(env);

  CpvAccess(currentChareBlock) = (void *)chareblock;
  CpvAccess(nodeforCharesProcessed)++;
  if(CpvAccess(traceOn)) {
    current_magic   = chareblock->selfID.magic;
    trace_begin_execute(env);
    (current_epinfo->function)(current_usr,chareblock->chareptr);
    trace_end_execute(current_magic, current_msgType, current_ep);
  } else {
    (current_epinfo->function)(current_usr,chareblock->chareptr);
  }
  if((current_msgType!=QdBocMsg)&&(current_msgType!=QdBroadcastBocMsg)&&
     (current_msgType!=LdbMsg))
    QDCountThisProcessing();
}

void CkProcess_ForChareMsg(env)
ENVELOPE *env;
{
  CHARE_BLOCK *chare = GetEnv_chareBlockPtr(env);
  CheckMagicNumber(chare, env);
  switch (chare->charekind) {
    case CHAREKIND_UVID: CkProcess_ForChareMsg_to_UVID(env); break;
    case CHAREKIND_FVID: CkProcess_ForChareMsg_to_FVID(env); break;
    case CHAREKIND_CHARE: CkProcess_ForChareMsg_to_Chare(env); break;
    case CHAREKIND_BOCNODE: CkProcess_ForChareMsg_to_Chare(env); break;
  }
}

void CkProcess_BocMsg(env)
ENVELOPE *env;
{
  CHARE_BLOCK *chare;
  chare = GetBocBlockPtr(GetEnv_boc_num(env));
  SetEnv_chareBlockPtr(env, chare);
  CkProcess_ForChareMsg_to_Chare(env);
}

void CkProcess_DynamicBocInitMsg(env)
ENVELOPE *env;
{
  /* ProcessBocInitMsg handles Charm++ bocs properly */
  /* This process of registering the new boc using the */
  /* spanning tree is exactly the same for Charm++ */
  int current_msgType, executing_boc_num;

  current_msgType = GetEnv_msgType(env);
  executing_boc_num = ProcessBocInitMsg(env);
  RegisterDynamicBocInitMsg(&executing_boc_num, NULL);
  if((current_msgType!=QdBocMsg)&&(current_msgType!=QdBroadcastBocMsg)&&
     (current_msgType!=LdbMsg))
    QDCountThisProcessing();
}

void CkProcess_NewChareMsg(env)
ENVELOPE *env;
{
  int current_ep, current_msgType, current_magic, current_chare;
  void *current_usr;
  CHARE_BLOCK *current_block;
  EP_STRUCT *current_epinfo;
  CHARE_BLOCK *CreateChareBlock();

  current_ep = GetEnv_EP(env);
  current_epinfo = CsvAccess(EpInfoTable) + current_ep;
  current_chare = current_epinfo->chareindex;
  current_usr = USER_MSG_PTR(env);
  current_msgType = GetEnv_msgType(env);
  current_magic = CpvAccess(nodecharesProcessed)++;
  CpvAccess(currentChareBlock) = current_block = CreateChareBlock(
				    CsvAccess(ChareSizesTable)[current_chare],
				    CHAREKIND_CHARE, current_magic);

  /* If virtual block exists, get all messages for this chare	*/
  if (GetEnv_vidBlockPtr(env))
    VidRetrieveMessages(current_block,
	    GetEnv_vidPE(env),
	    GetEnv_vidBlockPtr(env));

  /* Now call the entry point : For Charm, this is the actual call to the 
     EP function. For Charm++, this is a call to the translator-generated
     stub function which does "new ChareType(msg)". 
     NOTE: CreateChareBlock sets current_block->chareptr to (current_block + 1)
     because ChareSizesTable[] holds the correct size of the chare for both 
     Charm and Charm++, so it allocates the correct amount of space.
     - Sanjeev 10/10/95 */

  if(CpvAccess(traceOn))
    trace_begin_execute(env);
  (current_epinfo->function)(current_usr, current_block->chareptr);
  if(CpvAccess(traceOn))
    trace_end_execute(current_magic, current_msgType, current_ep);
  if((current_msgType!=QdBocMsg)&&(current_msgType!=QdBroadcastBocMsg)&&
     (current_msgType!=LdbMsg))
    QDCountThisProcessing();
}

void CkProcess_VidSendOverMsg(env)
ENVELOPE *env;
{
  int current_msgType; void *current_usr;
  
  current_msgType = GetEnv_msgType(env);
  current_usr = USER_MSG_PTR(env);
  if(CpvAccess(traceOn))
    trace_begin_execute(env);
  VidSendOverMessages(current_usr, NULL);
  if(CpvAccess(traceOn))
    trace_end_execute(0, current_msgType, 0);
  if((current_msgType!=QdBocMsg)&&(current_msgType!=QdBroadcastBocMsg)&&
     (current_msgType!=LdbMsg))
  QDCountThisProcessing();
}




/* This is the message handler for non-init messages during the
   initialization phase. It simply buffers the messafe */
void BUFFER_INCOMING_MSG(env)
ENVELOPE *env;
{
  if (CpvAccess(CkInitPhase)) {
    CmiGrabBuffer((void **)&env);
    FIFO_EnQueue(CpvAccess(CkBuffQueue),(void *)env);
  } else {
    HANDLE_INCOMING_MSG(env);
  }
}


/* This is the handler function for Charm and Charm++, which is called
   immediately when a message is received (from self or network) */

void HANDLE_INCOMING_MSG(env)
ENVELOPE *env;
{
  CmiGrabBuffer((void **)&env);
  UNPACK(env);
  if(CpvAccess(traceOn))
    trace_enqueue(env);
  switch (GetEnv_msgType(env)) {
  case NewChareMsg:          CkProcess_NewChareMsg(env); break;
  case ForChareMsg:          CkProcess_ForChareMsg(env); break;
  case LdbMsg:               CkProcess_BocMsg(env); break;
  case QdBocMsg:             CkProcess_BocMsg(env); break;
  case QdBroadcastBocMsg:    CkProcess_BocMsg(env); break;
  case ImmBocMsg:            CkProcess_BocMsg(env); break;
  case ImmBroadcastBocMsg:   CkProcess_BocMsg(env); break;
  case BocMsg:               CkProcess_BocMsg(env); break;
  case BroadcastBocMsg:      CkProcess_BocMsg(env); break;
  case VidSendOverMsg:       CkProcess_VidSendOverMsg(env); break;
  case DynamicBocInitMsg:    CkProcess_DynamicBocInitMsg(env); break;
  case NewChareNoBalanceMsg:
    CmiAbort("** ERROR ** obsolete message type.\n");
  default:
    CmiAbort("** ERROR ** bad message type.\n");
  }
}


