#include "converse.h"
#include "queueing.h"
#include "cldb.h"
#include <time.h>
#include <stdlib.h>
#include <math.h>

void LoadNotifyFn(int l)
{
}

char *CldGetStrategy(void)
{
  return "spray";
}

#define CYCLE_MILLISECONDS 500
#define DEBUGGING_OUTPUT 0

typedef struct 
{
  int mype;
  int EnqueueHandler;
  int ReduceHandler;
  int AverageHandler;
  int HopHandler;
  double load_reported;
  double load_total;
  int    load_count;
  int    spantree_parent;
  int    spantree_children;
  int    spantree_root;
  int    rebalance;
}
peinfo;

CpvStaticDeclare(peinfo, peinf);

struct loadmsg {
  char core[CmiMsgHeaderSizeBytes];
  double load_total;
};

struct reqmsg {
  char core[CmiMsgHeaderSizeBytes];
};

void CldPropagateLoad(double load);

int CldEstimate(void)
{
  return CldLoad();
}

void CldInitiateReduction()
{
  double load = CldEstimate();
  peinfo *pinf = &(CpvAccess(peinf));
  pinf->load_reported = load;
  CldPropagateLoad(load);
}

void CldPropagateLoad(double load)
{
  struct loadmsg msg;
  peinfo *pinf = &(CpvAccess(peinf));
  pinf->load_total += load;
  pinf->load_count ++;
  if (pinf->load_count == pinf->spantree_children + 1) {
    msg.load_total   = pinf->load_total;
    if (pinf->mype == pinf->spantree_root) {
      if (DEBUGGING_OUTPUT) CmiPrintf("---\n");
      CmiSetHandler(&msg, pinf->AverageHandler);
      CmiSyncBroadcastAll(sizeof(msg), &msg);
    } else {
      CmiSetHandler(&msg, pinf->ReduceHandler);
      CmiSyncSend(pinf->spantree_parent, sizeof(msg), &msg);
    }
    pinf->load_total = 0;
    pinf->load_count = 0;
  }
}

void CldReduceHandler(struct loadmsg *msg)
{
  CldPropagateLoad(msg->load_total);
  CmiFree(msg);
}

void CldAverageHandler(struct loadmsg *msg)
{
  peinfo *pinf = &(CpvAccess(peinf));
  double load = CldEstimate();
  double average = (msg->load_total / CmiNumPes());
  int rebalance;
  if (load < (average+10) * 1.2) rebalance=0;
  else rebalance = (int)(load - average);
  if (DEBUGGING_OUTPUT)
    CmiPrintf("PE %d load=%6d average=%6d rebalance=%d\n", 
	      CmiMyPe(), CldEstimate(), (int)average, rebalance);
  pinf->rebalance = rebalance;
  CmiFree(msg);
  CcdCallFnAfter((CcdVoidFn)CldInitiateReduction, 0, CYCLE_MILLISECONDS);
}

void CldEnqueueHandler(char *msg)
{
  int len, queueing, priobits; unsigned int *prioptr;
  CldInfoFn ifn; CldPackFn pfn;
  ifn = (CldInfoFn)CmiHandlerToFunction(CmiGetInfo(msg));
  ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
  CmiSetHandler(msg, CmiGetXHandler(msg));
  CsdEnqueueGeneral(msg, queueing, priobits, prioptr);
}

void CldHopHandler(char *msg)
{
  peinfo *pinf = &(CpvAccess(peinf));
  int len, queueing, priobits; unsigned int *prioptr;
  CldInfoFn ifn; CldPackFn pfn; int pe;

  if (pinf->rebalance) {
    /* do pe = ((lrand48()&0x7FFFFFFF)%CmiNumPes()); */
    do pe = ((CrnRand()&0x7FFFFFFF)%CmiNumPes());
    while (pe == pinf->mype);
    ifn = (CldInfoFn)CmiHandlerToFunction(CmiGetInfo(msg));
    ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    if (pfn && CmiNodeOf(pe) != CmiMyNode()) {
      pfn(&msg);
      ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    }
    CmiSyncSendAndFree(pe, len, msg);
    pinf->rebalance--;
  } else {
    CmiSetHandler(msg, CmiGetXHandler(msg));
    CmiHandleMessage(msg);
  }
}

void CldEnqueueGroup(CmiGroup grp, void *msg, int infofn)
{
  int npes, *pes;
  int len, queueing, priobits,i; unsigned int *prioptr;
  CldInfoFn ifn = (CldInfoFn)CmiHandlerToFunction(infofn);
  peinfo *pinf = &(CpvAccess(peinf));
  CldPackFn pfn;
  ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
  if (pfn) {
    pfn(&msg);
    ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
  }
  CmiSetInfo(msg,infofn);
  CmiSetXHandler(msg, CmiGetHandler(msg));
  CmiSetHandler(msg, pinf->EnqueueHandler);
  CmiLookupGroup(grp, &npes, &pes);
  for(i=0;i<npes;i++) {
    CmiSyncSend(pes[i], len, msg);
  }
  CmiFree(msg);
}

void CldEnqueueMulti(int npes, int *pes, void *msg, int infofn)
{
  int len, queueing, priobits,i; unsigned int *prioptr;
  CldInfoFn ifn = (CldInfoFn)CmiHandlerToFunction(infofn);
  peinfo *pinf = &(CpvAccess(peinf));
  CldPackFn pfn;
  ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
  if (pfn) {
    pfn(&msg);
    ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
  }
  CmiSetInfo(msg,infofn);
  CmiSetXHandler(msg, CmiGetHandler(msg));
  CmiSetHandler(msg, pinf->EnqueueHandler);
  for(i=0;i<npes;i++) {
    CmiSyncSend(pes[i], len, msg);
  }
  CmiFree(msg);
}

void CldEnqueue(int pe, void *msg, int infofn)
{
  int len, queueing, priobits; unsigned int *prioptr;
  CldInfoFn ifn; CldPackFn pfn;
  peinfo *pinf = &(CpvAccess(peinf));
  ifn = (CldInfoFn)CmiHandlerToFunction(infofn);
  ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
  if (pe != CLD_ANYWHERE) {
    if (pfn && (CmiNodeOf(pe) != CmiMyNode())) {
      pfn(&msg);
      ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    }
    CmiSetInfo(msg, infofn);
    CmiSetXHandler(msg, CmiGetHandler(msg));
    CmiSetHandler(msg, pinf->EnqueueHandler);
    if (pe==CLD_BROADCAST) CmiSyncBroadcastAndFree(len, msg);
    else if (pe==CLD_BROADCAST_ALL) CmiSyncBroadcastAllAndFree(len, msg);
    else CmiSyncSendAndFree(pe, len, msg);
  } else {
    CmiSetInfo(msg, infofn);
    CmiSetXHandler(msg, CmiGetHandler(msg));
    CmiSetHandler(msg, pinf->HopHandler);
    CsdEnqueueGeneral(msg, queueing, priobits, prioptr);
  }
}

void CldNodeEnqueue(int node, void *msg, int infofn)
{
  int len, queueing, priobits; unsigned int *prioptr;
  CldInfoFn ifn; CldPackFn pfn;
  peinfo *pinf = &(CpvAccess(peinf));
  ifn = (CldInfoFn)CmiHandlerToFunction(infofn);
  ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
  if (node != CLD_ANYWHERE) {
    if (pfn && (node != CmiMyNode())) {
      pfn(&msg);
      ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    }
    CmiSetInfo(msg, infofn);
    CmiSetXHandler(msg, CmiGetHandler(msg));
    CmiSetHandler(msg, pinf->EnqueueHandler);
    if (node==CLD_BROADCAST) CmiSyncNodeBroadcastAndFree(len, msg);
    else if (node==CLD_BROADCAST_ALL) CmiSyncNodeBroadcastAllAndFree(len, msg);
    else CmiSyncNodeSendAndFree(node, len, msg);
  } else {
    CmiSetInfo(msg, infofn);
    CmiSetXHandler(msg, CmiGetHandler(msg));
    CmiSetHandler(msg, pinf->HopHandler);
    CsdNodeEnqueueGeneral(msg, queueing, priobits, prioptr);
  }
}

void CldModuleInit(char **argv)
{
  peinfo *pinf;
  CpvInitialize(peinfo, peinf);
  /* srand48(time(0)+CmiMyPe()); */
  CrnSrand((int) (time(0)+CmiMyPe()));
  pinf = &CpvAccess(peinf);
  pinf->mype = CmiMyPe();
  pinf->EnqueueHandler = CmiRegisterHandler((CmiHandler)CldEnqueueHandler);
  pinf->ReduceHandler  = CmiRegisterHandler((CmiHandler)CldReduceHandler);
  pinf->AverageHandler = CmiRegisterHandler((CmiHandler)CldAverageHandler);
  pinf->HopHandler     = CmiRegisterHandler((CmiHandler)CldHopHandler);
  pinf->load_total = 0.0;
  pinf->load_count = 0;
  pinf->spantree_children = CmiNumSpanTreeChildren(CmiMyPe());
  pinf->spantree_parent = CmiSpanTreeParent(CmiMyPe());
  pinf->spantree_root = 0;
  pinf->rebalance = 0;
  CldModuleGeneralInit(argv);
  CldInitiateReduction();
}
void CldCallback()
{}
