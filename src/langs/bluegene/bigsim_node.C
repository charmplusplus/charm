
#include "blue.h"
#include "blue_impl.h"    	// implementation header file
#include "blue_timing.h" 	// timing module

#define  DEBUGF(x)      //CmiPrintf x;

/**
  nodeInfo construtor
  class definition in blue_impl.h
*/
nodeInfo::nodeInfo(): lastW(0), udata(NULL), started(0), timeron_flag(0)
{
    int i;

    inBuffer.initialize(INBUFFER_SIZE);
    msgBuffer = CmmNew();

    commThQ = new threadQueue;
    commThQ->initialize(cva(numCth));

    threadTable = new CthThread[cva(numWth)+cva(numCth)];
    _MEMCHECK(threadTable);
    threadinfo = new threadInfo*[cva(numWth)+cva(numCth)];
    _MEMCHECK(threadinfo);

    affinityQ = new ckMsgQueue[cva(numWth)];
    _MEMCHECK(affinityQ);

    // create threadinfo
    for (i=0; i< cva(numWth); i++)
    {
      threadinfo[i] = new workThreadInfo(i, WORK_THREAD, this);
      _MEMCHECK(threadinfo[i]);
    }
    for (i=0; i< cva(numCth); i++)
    {
      threadinfo[i+cva(numWth)] = new commThreadInfo(i+cva(numWth), COMM_THREAD, this);
      _MEMCHECK(threadinfo[i+cva(numWth)]);
    }
#if BLUEGENE_TIMING
    timelines = new BgTimeLineRec[cva(numWth)]; // set default size 1024
    _MEMCHECK(timelines);
#endif
  }

nodeInfo::~nodeInfo() 
{
    if (commThQ) delete commThQ;
    delete [] affinityQ;
    delete [] threadTable;
    delete [] threadinfo;
}


/* add a message to this bluegene node's inbuffer queue */
void nodeInfo::addBgNodeInbuffer(char *msgPtr)
{
  int tags[1];

  /* if inbuffer is full, store in the msgbuffer */
  if (inBuffer.isFull()) {
    tags[0] = id;
    CmmPut(msgBuffer, 1, tags, msgPtr);
  }
  else {
    DEBUGF(("inbuffer is not full.\n"));
    inBuffer.enq(msgPtr);
  }
  /* awake a communication thread to schedule it */
  CthThread t = commThQ->deq();
  if (t) {
    DEBUGF(("activate communication thread on node %d: %p.\n", nodeID, t));
#if 0
    unsigned int prio = 0;
    CthAwakenPrio(t, CQS_QUEUEING_IFIFO, sizeof(int), &prio);
#else
    CthAwaken(t);
#endif
  }
}

/* called by comm thread to poll inBuffer */
char *nodeInfo::getFullBuffer()
{
  int tags[1], ret_tags[1];
  char *data, *mb;

  /* see if we have msg in inBuffer */
  if (inBuffer.isEmpty()) return NULL;
  data = inBuffer.deq(); 

  /* since we have just delete one from inbuffer, fill one from msgbuffer */
  tags[0] = CmmWildCard;
  mb = (char *)CmmGet(msgBuffer, 1, tags, ret_tags);
  if (mb) inBuffer.enq(mb);

  return data;
}

/* add a message to this bluegene node's non-affinity queue */
void nodeInfo::addBgNodeMessage(char *msgPtr)
{
  int i;
  /* find a idle worker thread */
  /* FIXME:  flat search is bad if there is many work threads */
  int wID = lastW;
  for (i=0; i<cva(numWth); i++) 
  {
    wID ++;
    if (wID == cva(numWth)) wID = 0;
    if (affinityQ[wID].length() == 0)
    {
      /* this work thread is idle, schedule the msg here */
      DEBUGF(("activate a work thread %d - %p.\n", wID, threadTable[wID]));
      affinityQ[wID].enq(msgPtr);
#if SCHEDULE_WORK
      double nextT = CmiBgMsgRecvTime(msgPtr);
      CthThread tid = threadTable[wID];
      unsigned int prio = (unsigned int)(nextT*PRIO_FACTOR)+1;
      CthAwakenPrio(tid, CQS_QUEUEING_IFIFO, sizeof(int), &prio);
#else
      CthAwaken(threadTable[wID]);
#endif
      lastW = wID;
      return;
    }
  }
  /* all worker threads are busy */   
  DEBUGF(("all work threads are busy.\n"));
#if SCHEDULE_WORK
#if 0
  DEBUGF(("[N%d] activate all work threads on N%d.\n", id));
  double nextT = CmiBgMsgRecvTime(msgPtr);
  unsigned int prio = (unsigned int)(nextT*PRIO_FACTOR)+1;
  nodeQ.enq(msgPtr);
  for (i=0; i<cva(numWth); i++) 
  {
      CthThread tid = threadTable[i];
      CthAwakenPrio(tid, CQS_QUEUEING_IFIFO, sizeof(int), &prio);
  }
#else
    // only awake rank 0 thread, which is a comm thread
  ((commThreadInfo *)threadinfo[0])->addAffMessage(msgPtr);
/*
  affinityQ[0].enq(msgPtr);
  CthThread tid = threadTable[0];
  CthAwakenPrio(tid, CQS_QUEUEING_IFIFO, sizeof(int), &prio);
*/
#endif

#else
  nodeQ.enq(msgPtr);
#endif
}

