#include <unistd.h>
#include "middle.h"

#include "ccs-server.h"
#include "conv-ccs.h"

#ifdef _WIN32
# include <sys/types.h>
# include <io.h>
# include <process.h>
# define write _write
# define getpid _getpid
#endif

#if CMK_CCS_AVAILABLE
void CcsHandleRequest(CcsImplHeader *hdr,const char *reqData);

void req_fw_handler(char *msg)
{
  int offset = CmiReservedHeaderSize + sizeof(CcsImplHeader);
  CcsImplHeader *hdr = (CcsImplHeader *)(msg+CmiReservedHeaderSize);

  int destPE = (int)ChMessageInt(hdr->pe);
  //  CmiPrintf("[%d] req_fw_handler got msg for pe %d\n",CmiMyPe(),destPE);
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
  
#if CMK_SMP
  // Charmrun gave us a message that is being handled on the rank 0 CCS PE
  // i.e., PE 0, but needs to get forwarded to the real PE destination
  if(destPE>0 && destPE!=CmiMyPe())
    
    {
      int len=CmiReservedHeaderSize+sizeof(CcsImplHeader)+ChMessageInt(hdr->len);
      CmiSyncSend(destPE, len, msg);      
    }
  else
#endif        
    {
      CcsHandleRequest(hdr, msg+offset);
    }
  CmiFree(msg);

}

extern int rep_fw_handler_idx;
/**
 * Decide if the reply is ready to be forwarded to the waiting client,
 * or if combination is required (for broadcast/multicast CCS requests.
 */
int CcsReply(CcsImplHeader *rep,int repLen,const void *repData) {
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
    if (_conditionalDelivery == 0) CcsImpl_reply(rep, repLen, repData);
    else {
      /* We are the child of a conditional delivery, write to the parent the reply */
      if (write(conditionalPipe[1], &repLen, 4) != 4) {
        CmiAbort("CCS> writing reply length to parent failed!");
      }
      if (write(conditionalPipe[1], repData, repLen) != repLen) {
        CmiAbort("CCS> writing reply data to parent failed!");
      }
    }
  }
  return 0;
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

///////////////////////////////// middle-debug.C
#endif
#if ! CMK_HAS_GETPID
typedef int pid_t;
#endif

CpvCExtern(void *, debugQueue);
CpvDeclare(void *, debugQueue);
CpvCExtern(int, freezeModeFlag);
CpvDeclare(int, freezeModeFlag);

/*
 Start the freeze-- call will not return until unfrozen
 via a CCS request.
 */
void CpdFreeze(void)
{
  pid_t pid = 0;
#if CMK_HAS_GETPID
  pid = getpid();
#endif
  CpdNotify(CPD_FREEZE,pid);
  if (CpvAccess(freezeModeFlag)) return; /*Already frozen*/
  CpvAccess(freezeModeFlag) = 1;
  CpdFreezeModeScheduler();
}

void CpdUnFreeze(void)
{
  CpvAccess(freezeModeFlag) = 0;
}

int CpdIsFrozen(void) {
  return CpvAccess(freezeModeFlag);
}


void PrintDebugStackTrace(void *);
void * MemoryToSlot(void *ptr);
int Slot_StackTrace(void *s, void ***stack);
int Slot_ChareOwner(void *s);

#include <stdarg.h>
void CpdNotify(int type, ...) {
  void *ptr; int integer, i;
  pid_t pid=0;
  int levels=64;
  void *stackPtrs[64];
  void *sl;
  va_list list;
  va_start(list, type);
  switch (type) {
  case CPD_ABORT:
    CmiPrintf("CPD: %d Abort %s\n",CmiMyPe(), va_arg(list, char*));
    break;
  case CPD_SIGNAL:
    CmiPrintf("CPD: %d Signal %d\n",CmiMyPe(), va_arg(list, int));
    break;
  case CPD_FREEZE:
#if CMK_HAS_GETPID
    pid = getpid();
#endif
    CmiPrintf("CPD: %d Freeze %d\n", CmiMyPe(), (int)pid);
    break;
  case CPD_BREAKPOINT:
    CmiPrintf("CPD: %d BP %s\n",CmiMyPe(), va_arg(list, char*));
    break;
  case CPD_CROSSCORRUPTION:
    ptr = va_arg(list, void*);
    integer = va_arg(list, int);
    CmiPrintf("CPD: %d Cross %p %d ",CmiMyPe(), ptr, integer);
    sl = MemoryToSlot(ptr);
    if (sl != NULL) {
      int stackLen; void **stackTrace;
      stackLen = Slot_StackTrace(sl, &stackTrace);
      CmiPrintf("%d %d ",Slot_ChareOwner(sl),stackLen);
      for (i=0; i<stackLen; ++i) CmiPrintf("%p ",stackTrace[i]);
    } else {
      CmiPrintf("0 ");
    }
    CmiBacktraceRecord(stackPtrs,1,&levels);
    CmiPrintf("%d ",levels);
    for (i=0; i<levels; ++i) CmiPrintf("%p ",stackPtrs[i]);
    CmiPrintf("\n");
    break;
  }
  va_end(list);
}
