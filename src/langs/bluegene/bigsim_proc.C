
#include "blue.h"
#include "blue_impl.h"    	// implementation header file
#include "blue_timing.h" 	// timing module

#define  DEBUGF(x)      //CmiPrintf x;

extern BgStartHandler  workStartFunc;

void correctMsgTime(char *msg);

/**
  threadInfo methods
*/
void commThreadInfo::addAffMessage(char *msgPtr)
{
  ckMsgQueue &que = myNode->affinityQ[id];
  que.enq(msgPtr);
#if SCHEDULE_WORK
  /* don't awake directly, put into a priority queue sorted by recv time */
  double nextT = CmiBgMsgRecvTime(msgPtr);
  CthThread tid = me;
  unsigned int prio = (unsigned int)(nextT*PRIO_FACTOR)+1;
  DEBUGF(("[%d] awaken worker thread with prio %d.\n", tMYNODEID, prio));
  CthAwakenPrio(tid, CQS_QUEUEING_IFIFO, sizeof(int), &prio);
#else
  if (que.length() == 1) {
    CthAwaken(me);
  }
#endif
}

void commThreadInfo::run()
{
  tSTARTTIME = CmiWallTimer();

  if (!tSTARTED) {
    tSTARTED = 1;
//    InitHandlerTable();
    BgNodeStart(BgGetArgc(), BgGetArgv());
    /* bnv should be initialized */
  }

  threadQueue *commQ = myNode->commThQ;

  for (;;) {
    char *msg = getFullBuffer();
    if (!msg) { 
//      tCURRTIME += (CmiWallTimer()-tSTARTTIME);
      commQ->enq(CthSelf());
      DEBUGF(("[%d] comm thread suspend.\n", BgMyNode()));
      CthSuspend(); 
      DEBUGF(("[%d] comm thread assume.\n", BgMyNode()));
//      tSTARTTIME = CmiWallTimer();
      continue;
    }
    DEBUGF(("[%d] comm thread has a msg.\n", BgMyNode()));
    /* schedule a worker thread, if small work do it itself */
    if (CmiBgMsgType(msg) == SMALL_WORK) {
      if (CmiBgMsgRecvTime(msg) > tCURRTIME)  tCURRTIME = CmiBgMsgRecvTime(msg);
//      tSTARTTIME = CmiWallTimer();
      /* call user registered handler function */
      BgProcessMessage(msg);
    }
    else {
#if BLUEGENE_TIMING
      correctMsgTime(msg);
#endif
      if (CmiBgMsgThreadID(msg) == ANYTHREAD) {
        DEBUGF(("anythread, call addBgNodeMessage\n"));
        addBgNodeMessage(msg);			/* non-affinity message */
      }
      else {
        DEBUGF(("[N%d] affinity msg, call addBgThreadMessage to tID:%d\n", 
			BgMyNode(), CmiBgMsgThreadID(msg)));
        addBgThreadMessage(msg, CmiBgMsgThreadID(msg));
      }
    }
    /* let other communication thread do their jobs */
//    tCURRTIME += (CmiWallTimer()-tSTARTTIME);
#if !SCHEDULE_WORK
    CthYield();
#endif
    tSTARTTIME = CmiWallTimer();
  }
}

void workThreadInfo::run()
{
  tSTARTTIME = CmiWallTimer();

//  InitHandlerTable();
  if (workStartFunc) {
    DEBUGF(("[N%d] work thread %d start.\n", BgMyNode(), id));
    // timing
    startVTimer();
    BG_ENTRYSTART((char*)NULL);
    char **Cmi_argvcopy = CmiCopyArgs(BgGetArgv());
    workStartFunc(BgGetArgc(), Cmi_argvcopy);
    BG_ENTRYEND();
    stopVTimer();
  }

  ckMsgQueue &q1 = myNode->nodeQ;
  ckMsgQueue &q2 = myNode->affinityQ[id];

  for (;;) {
    char *msg=NULL;
    int e1 = q1.isEmpty();
    int e2 = q2.isEmpty();
    int fromQ2 = 0;		// delay the deq of msg from affinity queue

    // not deq from nodeQ assuming no interrupt in the handler
    if (e1 && !e2) { msg = q2[0]; fromQ2 = 1;}
//    else if (e2 && !e1) { msg = q1.deq(); }
    else if (e2 && !e1) { msg = q1[0]; }
    else if (!e1 && !e2) {
      if (CmiBgMsgRecvTime(q1[0]) < CmiBgMsgRecvTime(q2[0])) {
//        msg = q1.deq();
        msg = q1[0];
      }
      else {
        msg = q2[0];
        fromQ2 = 1;
      }
    }
    /* if no msg is ready, put it to sleep */
    if ( msg == NULL ) {
//      tCURRTIME += (CmiWallTimer()-tSTARTTIME);
      CthSuspend();
      DEBUGF(("[N%d] work thread T%d awakened.\n", BgMyNode(), id));
      continue;
    }
#if BLUEGENE_TIMING
    correctMsgTime(msg);
#if THROTTLE_WORK
    if (correctTimeLog) {
      if (CmiBgMsgRecvTime(msg) > gvt+ LEASH) {
	double nextT = CmiBgMsgRecvTime(msg);
	unsigned int prio = (unsigned int)(nextT*PRIO_FACTOR)+1;
// CmiPrintf("Thread %d YieldPrio: %g gvt: %g leash: %g\n", id, nextT, gvt, LEASH);
	CthYieldPrio(CQS_QUEUEING_IFIFO, sizeof(int), &prio);
	continue;
      }
    }
#endif
#endif   /* TIMING */
    DEBUGF(("[N%d] work thread T%d has a msg.\n", BgMyNode(), id));

//if (tMYNODEID==0)
//CmiPrintf("[%d] recvT: %e\n", tMYNODEID, CmiBgMsgRecvTime(msg));

    if (CmiBgMsgRecvTime(msg) > currTime) {
      tCURRTIME = CmiBgMsgRecvTime(msg);
    }

    BG_ENTRYSTART(msg);
    // BgProcessMessage may trap into scheduler
    BgProcessMessage(msg);
    BG_ENTRYEND();

    // counter of processed real mesgs
    stateCounters.realMsgProcCnt++;

    if (fromQ2 == 1) q2.deq();
    else q1.deq();

    DEBUGF(("[N%d] work thread T%d finish a msg.\n", BgMyNode(), id));

    /* let other work thread do their jobs */
#if SCHEDULE_WORK
    DEBUGF(("[N%d] work thread T%d suspend when done - %d to go.\n", BgMyNode(), tMYID, q2.length()));
    CthSuspend();
    DEBUGF(("[N%d] work thread T%d awakened here.\n", BgMyNode(), id));
#else
    CthYield();
#endif
  }
}

