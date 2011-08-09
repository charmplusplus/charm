#include <stdlib.h>
#include <converse.h>
#include "commbench.h"

#define pva CpvAccess
#define pvd CpvStaticDeclare
#define pvi CpvInitialize

static struct testdata {
  int size;
  int numiter;
} sizes[] = {
  {16,      256},
  //{256,     256},
  //{2048,    64},
  //{4096,    64},
  //{65536,   10},
  //{1048576, 10},
  {-1,      -1},
};

typedef struct message_{
  char core[CmiMsgHeaderSizeBytes];
  int srcpe;
  int idx;
  int data[1];
} Message;

static void fillMessage(Message *msg)
{
  int i, size, items;
  size = sizes[msg->idx].size + sizeof(double);
  items = size/sizeof(double);
  for(i=0;i<items;i++)
    msg->data[i] = i+0x1234;
}

static void checkMessage(Message *msg)
{
  int i, size, items;
  size = sizes[msg->idx].size + sizeof(double);
  items = size/sizeof(double);
  for(i=0;i<items;i++) {
    if(msg->data[i] != (i+0x1234))
      CmiAbort("[pingpong] Data corrupted. Run megacon !!\n");
  }
}

typedef struct Time_{
  char core[CmiMsgHeaderSizeBytes];
  int srcNode;
  double data[1];
} TimeMessage;

pvd(int, nextIter);
pvd(int, nextSize);
pvd(int, nextNbr); 

pvd(double, starttime);
pvd(double, endtime);
pvd(double **, times);/* stores times for all nbrs and sizes */
pvd(int *, nodeList); /* stores nums of pes rank 0 on all nodes*/
pvd(double *, gavg);
pvd(double *, gmax);
pvd(double *, gmin);
pvd(int *, gmaxSrc);
pvd(int *, gmaxDest);
pvd(int *, gminSrc);
pvd(int *, gminDest);

pvd(int, numSizes);    /* how many test sizes exist */
pvd(int, numRecv);

pvd(int, timeHandler);
pvd(int, nodeHandler);
pvd(int, nbrHandler);
pvd(int, sizeHandler);
pvd(int, iterHandler);
pvd(int, bounceHandler);

static void recvTime(TimeMessage *msg)
{
  EmptyMsg m;
  int i, j;
  double time;

  pva(numRecv)++;
  for(i=0;i<CmiNumNodes();i++) {
    if(i==msg->srcNode)
      continue;
    for(j=0;j<pva(numSizes);j++) {
      time = *(msg->data+i*pva(numSizes)+j);
      pva(gavg)[j] += time;
      if(time > pva(gmax)[j]) {
        pva(gmax)[j] = time;
        pva(gmaxSrc)[j] = msg->srcNode;
        pva(gmaxDest)[j] = i;
      }
      if(time < pva(gmin)[j]) {
        pva(gmin)[j] = time;
        pva(gminSrc)[j] = msg->srcNode;
        pva(gminDest)[j] = i;
      }
    }
  }
  if(pva(numRecv)==CmiNumNodes()){
    for(j=0;j<pva(numSizes);j++)
      pva(gavg)[j] /= (CmiNumNodes()*(CmiNumNodes()-1));
    CmiPrintf("[pingpong] CmiSyncSend\n");
    for(j=0;j<pva(numSizes);j++) {
      CmiPrintf("[pingpong] size=%d\taverageTime=%le seconds\n",
                            sizes[j].size, pva(gavg)[j]);
      CmiPrintf("[pingpong] size=%d\tmaxTime=%le seconds\t[%d->%d]\n",
                  sizes[j].size, pva(gmax)[j],pva(gmaxSrc)[j],pva(gmaxDest)[j]);
      CmiPrintf("[pingpong] size=%d\tminTime=%le seconds\t[%d->%d]\n",
                  sizes[j].size, pva(gmin)[j],pva(gminSrc)[j],pva(gminDest)[j]);
    }
    CmiSetHandler(&m, pva(ack_handler));
    CmiSyncSend(0, sizeof(EmptyMsg), &m);
  }
  CmiFree(msg);
}

static void startNextNode(EmptyMsg *msg)
{
  EmptyMsg m;
  CmiFree(msg);
  if((CmiMyNode()+1) != CmiNumNodes()) {
    CmiSetHandler(&m, pva(nbrHandler));
    CmiSyncSend(pva(nodeList)[CmiMyNode()+1], sizeof(EmptyMsg), &m);
  }
}

static void startNextNbr(EmptyMsg *msg)
{
  EmptyMsg m;
  TimeMessage *tm;
  int i, size;

  CmiFree(msg);
  pva(nextNbr)++;
  if(pva(nextNbr) == CmiMyNode()) {
    CmiSetHandler(&m, pva(nbrHandler));
    CmiSyncSend(CmiMyPe(), sizeof(EmptyMsg), &m);
    return;
  }
  if(pva(nextNbr) == CmiNumNodes()) {
    pva(nextNbr) = -1;
    CmiSetHandler(&m, pva(nodeHandler));
    CmiSyncSend(CmiMyPe(), sizeof(EmptyMsg), &m);
    size = sizeof(TimeMessage)+pva(numSizes)*CmiNumNodes()*sizeof(double);
    tm = (TimeMessage *) CmiAlloc(size);
    for(i=0;i<CmiNumNodes();i++)
      memcpy(tm->data+i*pva(numSizes),pva(times)[i],
             sizeof(double)*pva(numSizes));
    tm->srcNode = CmiMyNode();
    CmiSetHandler(tm, pva(timeHandler));
    CmiSyncSendAndFree(0, size, tm);
  } else {
    CmiSetHandler(&m, pva(sizeHandler));
    CmiSyncSend(CmiMyPe(), sizeof(EmptyMsg), &m);
  }
}

static void startNextSize(EmptyMsg *msg)
{
  EmptyMsg m;
  Message *mm;

  pva(nextSize)++;
  if(pva(nextSize) == pva(numSizes)) {
    pva(nextSize) = -1;
    CmiSetHandler(&m, pva(nbrHandler));
    CmiSyncSend(CmiMyPe(), sizeof(EmptyMsg), &m);
  } else {
    int size = sizeof(Message)+sizes[pva(nextSize)].size;
    mm = (Message *) CmiAlloc(size);
    mm->srcpe = CmiMyPe();
    mm->idx = pva(nextSize);
    CmiSetHandler(mm, pva(iterHandler));
    fillMessage(mm);
    CmiSyncSendAndFree(CmiMyPe(), size, mm);
    pva(starttime) = CmiWallTimer();
  }
  CmiFree(msg);
}

static void startNextIter(Message *msg)
{
  EmptyMsg m;

  pva(nextIter)++;
  if(pva(nextIter) > sizes[pva(nextSize)].numiter) {
    pva(endtime) = CmiWallTimer();
    checkMessage(msg);
    pva(times)[pva(nextNbr)][pva(nextSize)] =
      (pva(endtime) - pva(starttime))/(2.0*sizes[pva(nextSize)].numiter);
    pva(nextIter) = -1;
    CmiSetHandler(&m, pva(sizeHandler));
    CmiSyncSend(CmiMyPe(), sizeof(EmptyMsg), &m);
    CmiFree(msg);
  } else {
    CmiSetHandler(msg, pva(bounceHandler));
    CmiSyncSendAndFree(pva(nextNbr), sizeof(Message)+sizes[msg->idx].size, msg);
  }
}

static void bounceMessage(Message *msg)
{
  CmiSetHandler(msg, pva(iterHandler));
  CmiSyncSendAndFree(msg->srcpe, sizeof(Message)+sizes[msg->idx].size, msg);
}

void pingpong_init(void)
{
  EmptyMsg m;

  if(CmiNumNodes()==1) {
    CmiPrintf("[pingpong] This benchmark requires > 1 nodes.\n");
    CmiSetHandler(&m, pva(ack_handler));
    CmiSyncSend(0, sizeof(EmptyMsg), &m);
    return;
  }
  CmiSetHandler(&m, pva(nbrHandler));
  CmiSyncSend(0, sizeof(EmptyMsg), &m);
}

void pingpong_moduleinit(void)
{
  int i,j;
  pvi(int, numRecv);
  pva(numRecv) = 0;
  pvi(int, nextIter);
  pva(nextIter) = -1;
  pvi(int, nextSize);
  pva(nextSize) = -1;
  pvi(int, nextNbr);
  pva(nextNbr) = -1;
  pvi(double, starttime);
  pva(starttime) = 0.0;
  pvi(double, endtime);
  pva(endtime) = 0.0;
  pvi(int, numSizes);
  for(i=0; sizes[i].size != (-1); i++);
  pva(numSizes) = i;
  pvi(double **, times);
  pva(times) = (double **) malloc(CmiNumNodes()*sizeof(double *));
  for(i=0;i<CmiNumNodes();i++)
    pva(times)[i] = (double *) malloc(pva(numSizes)*sizeof(double));
  for(i=0;i<CmiNumNodes();i++)
    for(j=0;j<pva(numSizes);j++)
      pva(times)[i][j] = 0.0;
  pvi(int *, nodeList);
  pva(nodeList) = (int *) malloc(CmiNumNodes()*sizeof(int));
  for(i=0;i<CmiNumNodes();i++)
    pva(nodeList)[i] = CmiNodeFirst(i);
  pvi(double *, gavg);
  pva(gavg) = (double *) malloc(sizeof(double)*pva(numSizes));
  pvi(double *, gmax);
  pva(gmax) = (double *) malloc(sizeof(double)*pva(numSizes));
  pvi(double *, gmin);
  pva(gmin) = (double *) malloc(sizeof(double)*pva(numSizes));
  pvi(int *, gmaxSrc);
  pva(gmaxSrc) = (int *) malloc(sizeof(int)*pva(numSizes));
  pvi(int *, gmaxDest);
  pva(gmaxDest) = (int *) malloc(sizeof(int)*pva(numSizes));
  pvi(int *, gminSrc);
  pva(gminSrc) = (int *) malloc(sizeof(int)*pva(numSizes));
  pvi(int *, gminDest);
  pva(gminDest) = (int *) malloc(sizeof(int)*pva(numSizes));
  for(i=0;i<pva(numSizes);i++) {
    pva(gavg)[i] = 0.0;
    pva(gmax)[i] = 0.0;
    pva(gmin)[i] = 1000000000.0;
    pva(gmaxSrc)[i] = 0;
    pva(gmaxDest)[i] = 0;
    pva(gminSrc)[i] = 0;
    pva(gminDest)[i] = 0;
  }
  pvi(int, timeHandler);
  pva(timeHandler) = CmiRegisterHandler((CmiHandler)recvTime);
  pvi(int, nodeHandler);
  pva(nodeHandler) = CmiRegisterHandler((CmiHandler)startNextNode);
  pvi(int, nbrHandler);
  pva(nbrHandler) = CmiRegisterHandler((CmiHandler)startNextNbr);
  pvi(int, sizeHandler);
  pva(sizeHandler) = CmiRegisterHandler((CmiHandler)startNextSize);
  pvi(int, iterHandler);
  pva(iterHandler) = CmiRegisterHandler((CmiHandler)startNextIter);
  pvi(int, bounceHandler);
  pva(bounceHandler) = CmiRegisterHandler((CmiHandler)bounceMessage);
}
