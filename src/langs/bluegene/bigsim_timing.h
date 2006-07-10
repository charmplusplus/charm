#ifndef BLUE_TIMING_H
#define BLUE_TIMING_H

#include "cklists.h"

#include "bigsim_logs.h"

extern int programExit;
extern double gvt;
//extern int realMsgProcCount,corrMsgProcCount;
extern int  genTimeLog;
extern int  correctTimeLog;
extern int  bgverbose;
extern int  schedule_flag;

CpvExtern(int, msgCounter);

class StateCounters{
  public:
  int realMsgProcCnt,corrMsgProcCnt,corrMsgEnqCnt,corrMsgCCCnt,corrMsgRCCnt,corrMsgCRCnt;
  StateCounters():realMsgProcCnt(0),corrMsgProcCnt(0),
		  corrMsgEnqCnt(0), corrMsgCCCnt(0), 
		  corrMsgRCCnt(0), corrMsgCRCnt(0)
		  {}
  void clear() { realMsgProcCnt=corrMsgProcCnt=corrMsgEnqCnt=
		 corrMsgCCCnt=corrMsgRCCnt=corrMsgCRCnt=0; }
  inline int actionCount() { return realMsgProcCnt+corrMsgProcCnt; }
  inline int operator == (StateCounters &c) {
    return realMsgProcCnt == c.realMsgProcCnt &&
           corrMsgProcCnt == c.corrMsgProcCnt &&
           corrMsgCCCnt == c.corrMsgCCCnt &&
	   corrMsgRCCnt == c.corrMsgRCCnt &&
	   corrMsgCRCnt == c.corrMsgCRCnt;
  }
  inline void add(StateCounters &c) {
    realMsgProcCnt += c.realMsgProcCnt;
    corrMsgProcCnt += c.corrMsgProcCnt;
    corrMsgEnqCnt += c.corrMsgEnqCnt;
    corrMsgCCCnt += c.corrMsgCCCnt;
    corrMsgRCCnt += c.corrMsgRCCnt;
    corrMsgCRCnt += c.corrMsgCRCnt;
  }
};

extern StateCounters stateCounters;
extern double minCorrectTimestamp;

/**
  timing correction message
*/
class bgCorrectionMsg
{
public:
  char     core[CmiBlueGeneMsgHeaderSizeBytes];
  BgMsgID  msgId;
  CmiInt2  tID;		// destination worker thread ID
			// it can be:  -1:   for any thread which was not known
			//            < -100: for each thread except one
  double   tAdjust;	// new absolute value of recvTime at destPe
  int      destNode;
public:
  double key() { return tAdjust; }
  int compareKey(bgCorrectionMsg* otherMsg)
    {
      if(tAdjust < otherMsg->tAdjust) return -1;
      else
	return 1;
    }
};

extern void BgInitTiming();
extern void BgMsgSetTiming(char *msg);
extern int BgAdjustTimelineByIndex(int idxOld, double tAdjustAbs, BgTimeLineRec &tline);
extern int BgAdjustTimeLineInsert(BgTimeLineRec &tline);
extern int BgAdjustTimeLineForward(const BgMsgID &msgId, double tAdjustAbs, BgTimeLineRec &tline, int mynode, int tid);
extern void BgPrintThreadTimeLine(int node, int th, BgTimeLine &tline);
extern void BgFinishCorrection(BgTimeLineRec &tlinerec, int mynode, int tid, int idx, int send=1);
extern void BgSendBufferedCorrMsgs();
extern int BgGetTimeLineIndexByRecvTime(BgTimeLineRec &, BgTimeLog *, int, int);
extern int BgAdjustTimeLineFromIndex(int index, BgTimeLineRec &tlinerec, int mynode);
extern int BgGetIndexFromTime(double effT, int seqno, BgTimeLineRec &tline);
extern void BgSendPendingCorrections(BgTimeLineRec &tlinerec, int mynode);
extern void BgLogEntryCommit(BgTimeLineRec &tlinerec);


#endif
