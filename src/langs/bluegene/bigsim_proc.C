
#include "blue.h"
#include "blue_impl.h"    	// implementation header file
#include "blue_timing.h" 	// timing module

#define  DEBUGF(x)      //CmiPrintf x;

extern BgStartHandler  workStartFunc;
extern "C" void CthResumeNormalThread(CthThreadToken* token);

void correctMsgTime(char *msg);

CpvExtern(int      , CthResumeBigSimThreadIdx);

/**
  threadInfo methods
*/
void commThreadInfo::run()
{
  CpvAccess(CthResumeBigSimThreadIdx) = BgRegisterHandler((BgHandler)CthResumeNormalThread);

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
    if (!schedule_flag) CthYield();
    tSTARTTIME = CmiWallTimer();
  }
}

void BgScheduler(int nmsg)
{
  ASSERT(tTHREADTYPE == WORK_THREAD);
  // end current log
  int isinterrupt = 0;
  if (genTimeLog) {
    if (BgIsInALog(tTIMELINEREC)) {
      isinterrupt = 1;
      BgLogEntryCommit(tTIMELINEREC);
      tTIMELINEREC.bgPrevLog = BgLastLog(tTIMELINEREC);
    }
  }
  stopVTimer();

  ((workThreadInfo*)cta(threadinfo))->scheduler(nmsg);

  // begin a new log, and make dependency
  startVTimer();
  if (genTimeLog && isinterrupt) 
  {
    BgTimeLog *curlog = BgLastLog(tTIMELINEREC);
    BgTimeLog *newLog = BgStartLogByName(tTIMELINEREC, -1, "BgSchedulerEnd", BgGetCurTime(), curlog);
  }
}

void BgExitScheduler()
{
  ASSERT(tTHREADTYPE == WORK_THREAD);
  ((workThreadInfo*)cta(threadinfo))->stopScheduler();
}

void BgDeliverMsgs(int nmsg)
{
  if (nmsg == 0) nmsg=1;
  BgScheduler(nmsg);
}

void workThreadInfo::scheduler(int count)
{
  ckMsgQueue &q1 = myNode->nodeQ;
  ckMsgQueue &q2 = myNode->affinityQ[id];

  int cycle = CsdStopFlag;

  int recvd = 0;
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
    /* if no msg is ready, go back to sleep */
    if ( msg == NULL ) {
//      tCURRTIME += (CmiWallTimer()-tSTARTTIME);
      CthSuspend();
      DEBUGF(("[N-%d] work thread T-%d awakened.\n", BgMyNode(), id));
      continue;
    }
#if BLUEGENE_TIMING
    correctMsgTime(msg);
#if THROTTLE_WORK
    if (correctTimeLog) {
      if (CmiBgMsgRecvTime(msg) > gvt+ BG_LEASH) {
	double nextT = CmiBgMsgRecvTime(msg);
	int prio = (int)(nextT*PRIO_FACTOR)+1;
	if (prio < 0) {
	  CmiPrintf("PRIO_FACTOR %e is too small. \n", PRIO_FACTOR);
	  CmiAbort("BigSim time correction abort!\n");
	}
//CmiPrintf("Thread %d YieldPrio: %g gvt: %g leash: %g\n", id, nextT, gvt, BG_LEASH);
	CthYieldPrio(CQS_QUEUEING_IFIFO, sizeof(int), (unsigned int*)&prio);
	continue;
      }
    }
#endif
#endif   /* TIMING */
    DEBUGF(("[N%d] work thread T%d has a msg with recvT:%e msgId:%d.\n", BgMyNode(), id, CmiBgMsgRecvTime(msg), CmiBgMsgID(msg)));

//if (tMYNODEID==0)
//CmiPrintf("[%d] recvT: %e\n", tMYNODEID, CmiBgMsgRecvTime(msg));

    if (CmiBgMsgRecvTime(msg) > currTime) {
      tCURRTIME = CmiBgMsgRecvTime(msg);
    }

#if 1
    if (fromQ2 == 1) q2.deq();
    else q1.deq();
#endif

    BG_ENTRYSTART(msg);
    // BgProcessMessage may trap into scheduler
    BgProcessMessage(msg);
    BG_ENTRYEND();

    // counter of processed real mesgs
    stateCounters.realMsgProcCnt++;

    // NOTE: I forgot why I delayed the dequeue after processing it
#if 0
    if (fromQ2 == 1) q2.deq();
    else q1.deq();
#endif

    DEBUGF(("[N%d] work thread T%d finish a msg.\n", BgMyNode(), id));

    recvd ++;
    if ( recvd == count) return;

    if (cycle != CsdStopFlag) break;

    /* let other work thread do their jobs */
    if (schedule_flag) {
    DEBUGF(("[N%d] work thread T%d suspend when done - %d to go.\n", BgMyNode(), tMYID, q2.length()));
    CthSuspend();
    DEBUGF(("[N%d] work thread T%d awakened here.\n", BgMyNode(), id));
    }
    else {
    CthYield();
    }
  }

  CsdStopFlag --;
}

void workThreadInfo::run()
{
  tSTARTTIME = CmiWallTimer();

    //  register for charm++ applications threads
  CpvAccess(CthResumeBigSimThreadIdx) = BgRegisterHandler((BgHandler)CthResumeNormalThread);

//  InitHandlerTable();
  // before going into scheduler loop, call workStartFunc
  // in bg charm++, it normally is initCharm
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

  scheduler(-1);

  CmiAbort("worker thread should never end!\n");
}

void workThreadInfo::addAffMessage(char *msgPtr)
{
  ckMsgQueue &que = myNode->affinityQ[id];
  que.enq(msgPtr);
  if (schedule_flag) {
  /* don't awake directly, put into a priority queue sorted by recv time */
  double nextT = CmiBgMsgRecvTime(msgPtr);
  CthThread tid = me;
  unsigned int prio = (unsigned int)(nextT*PRIO_FACTOR)+1;
  DEBUGF(("[%d] awaken worker thread with prio %d.\n", tMYNODEID, prio));
  CthAwakenPrio(tid, CQS_QUEUEING_IFIFO, sizeof(int), &prio);
  }
  else {
  if (que.length() == 1) {
    CthAwaken(me);
  }
  }
}

