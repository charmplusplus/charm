#include "middle.h"

#if CMK_BLUEGENE_CHARM
#include "bgconverse.h"
#endif
#include "ccs-server.h"
#include "conv-ccs.h"

extern "C" void CcsHandleRequest(CcsImplHeader *hdr,const char *reqData);

extern "C" void req_fw_handler(char *msg)
{
  int offset = CmiReservedHeaderSize + sizeof(CcsImplHeader);
  CcsImplHeader *hdr = (CcsImplHeader *)(msg+CmiReservedHeaderSize);
  int destPE = (int)ChMessageInt(hdr->pe);
  if (CmiMyPe() == 0 && destPE == -1) {
    /* Broadcast message to all other processors */
    int len=CmiReservedHeaderSize+sizeof(CcsImplHeader)+ChMessageInt(hdr->len);
    CmiSyncBroadcast(len, msg);
  }
  else if (destPE < -1) {
    /* Multicast the message to your children */
    int len=CmiReservedHeaderSize+sizeof(CcsImplHeader)+ChMessageInt(hdr->len)-destPE*sizeof(ChMessageInt_t);
    int index, child, i;
    int *pes = (int*)(msg+CmiReservedHeaderSize+sizeof(CcsImplHeader));
    ChMessageInt_t *pes_nbo = (ChMessageInt_t *)pes;
    offset -= destPE * sizeof(ChMessageInt_t);
    if (ChMessageInt(pes_nbo[0]) == CmiMyPe()) {
      for (index=0; index<-destPE; ++index) pes[index] = ChMessageInt(pes_nbo[index]);
    }
    for (index=0; index<-destPE; ++index) {
      if (pes[index] == CmiMyPe()) break;
    }
    child = (index << 2) + 1;
    for (i=0; i<4; ++i) {
      if (child+i < -destPE) {
        CmiSyncSend(pes[child+i], len, msg);
      }
    }
  }
  CcsHandleRequest(hdr, msg+offset);
  CmiFree(msg);
}

extern "C" void CcsSendReply(int replyLen, const void *replyData);
extern int rep_fw_handler_idx;
/**
 * Decide if the reply is ready to be forwarded to the waiting client,
 * or if combination is required (for broadcast/multicast CCS requests.
 */
extern "C" int CcsReply(CcsImplHeader *rep,int repLen,const void *repData) {
  int repPE = (int)ChMessageInt(rep->pe);
  if (repPE <= -1) {
    /* Reduce the message to get the final reply */
    CcsHandlerRec *fn;
    int len=CmiReservedHeaderSize+sizeof(CcsImplHeader)+repLen;
    char *msg=(char*)CmiAlloc(len);
    char *r=msg+CmiReservedHeaderSize;
    char *handlerStr;
    rep->len = ChMessageInt_new(repLen);
    *(CcsImplHeader *)r=*rep; r+=sizeof(CcsImplHeader);
    memcpy(r,repData,repLen);
    CmiSetHandler(msg,rep_fw_handler_idx);
    handlerStr=rep->handler;
    fn=(CcsHandlerRec *)CcsGetHandler(handlerStr);
    if (fn->mergeFn == NULL) CmiAbort("Called CCS broadcast with NULL merge function!\n");
    if (repPE == -1) {
      /* CCS Broadcast */
      CkReduce(msg, len, fn->mergeFn);
    } else {
      /* CCS Multicast */
      CmiListReduce(-repPE, (int*)(rep+1), msg, len, fn->mergeFn, fn->redID);
    }
  } else {
    CcsImpl_reply(rep, repLen, repData);
  }
}

/**********************************************
  "ccs_getinfo"-- takes no data
    Return the number of parallel nodes, and
      the number of processors per node as an array
      of 4-byte big-endian ints.
*/

void ccs_getinfo(char *msg)
{
  int nNode=CmiNumNodes();
  int len=(1+nNode)*sizeof(ChMessageInt_t);
  ChMessageInt_t *table=(ChMessageInt_t *)malloc(len);
  int n;
  table[0]=ChMessageInt_new(nNode);
  for (n=0;n<nNode;n++)
    table[1+n]=ChMessageInt_new(CmiNodeSize(n));
  CcsSendReply(len,(const char *)table);
  free(table);
  CmiFree(msg);
}

//////////////////////////////////////////////////////////////////// middle-debug.C

extern "C" {

CpvDeclare(void *, debugQueue);
CpvDeclare(int, freezeModeFlag);

/*
 Start the freeze-- call will not return until unfrozen
 via a CCS request.
 */
void CpdFreeze(void)
{
  CpdNotify(CPD_FREEZE,getpid());
  if (CpvAccess(freezeModeFlag)) return; /*Already frozen*/
  CpvAccess(freezeModeFlag) = 1;
#if ! CMK_BLUEGENE_CHARM
  CpdFreezeModeScheduler();
#endif
}

void CpdUnFreeze(void)
{
  CpvAccess(freezeModeFlag) = 0;
}

int CpdIsFrozen(void) {
  return CpvAccess(freezeModeFlag);
}

}

#if CMK_BLUEGENE_CHARM
#include "blue_impl.h"
void BgProcessMessageFreezeMode(threadInfo *t, char *msg) {
//  CmiPrintf("BgProcessMessageFreezeMode\n");
#if CMK_CCS_AVAILABLE
  void *debugQ=CpvAccess(debugQueue);
  CmiAssert(msg!=NULL);
  int processImmediately = CpdIsDebugMessage(msg);
  if (processImmediately) BgProcessMessageDefault(t, msg);
  while (!CpvAccess(freezeModeFlag) && !CdsFifo_Empty(debugQ)) {
    BgProcessMessageDefault(t, (char*)CdsFifo_Dequeue(debugQ));
  }
  if (!processImmediately) {
    if (!CpvAccess(freezeModeFlag)) BgProcessMessageDefault(t, msg); 
    else CdsFifo_Enqueue(debugQ, msg);
  }
#else
  BgProcessMessageDefault(t, msg);
#endif
}
#endif
