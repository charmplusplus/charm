#ifndef BLUE_TIMING_H
#define BLUE_TIMING_H

#include "cklists.h"

#include "blue_logs.h"

extern int programExit;
extern double gvt;
//extern int realMsgProcCount,corrMsgProcCount;
extern int  genTimeLog;
extern int  correctTimeLog;
extern int bgSkipEndFlag;

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
  int      msgID;	
  CmiInt2 tID;		// destination worker thread ID
			// it can be:  -1:   for any thread which was not known
			//            < -100: for each thread except one
  double   tAdjust;	// new absolute value of recvTime at destPe
  int 	   destNode;
  int      srcNode;
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
extern int BgAdjustTimeLineForward(int src, int msgID, double tAdjustAbs, BgTimeLineRec &tline, int mynode, int tid);
extern void BgPrintThreadTimeLine(int node, int th, BgTimeLine &tline);
extern void BgWriteThreadTimeLine(char **argv, int x, int y, int z, int th, BgTimeLine &tline);
extern void BgFinishCorrection(BgTimeLineRec &tlinerec, int mynode, int tid, int idx, int send=1);
extern void BgSendBufferedCorrMsgs();
extern bgTimeLog *BgGetTimeLog(BgTimeLineRec *tline, CmiInt2 tID, int srcnode, int msgID, int *index);
extern int BgGetTimeLineIndexByRecvTime(BgTimeLineRec &, bgTimeLog *, int, int);
extern int BgAdjustTimeLineFromIndex(int index, BgTimeLineRec &tlinerec, int mynode);
extern int BgGetIndexFromTime(double effT, int seqno, BgTimeLineRec &tline);
extern void BgSendPendingCorrections(BgTimeLineRec &tlinerec, int mynode);


#if 0
#if BLUEGENE_TIMING

#define BG_ENTRYSTART(handler, m)  \
	tTIMELINEREC.logEntryStart(handler, m);

#define BG_ENTRYEND()  \
	tTIMELINEREC.logEntryCommit();

#define BG_ADDMSG(m, node, tid, local)  	\
        if (genTimeLog)	{ \
          BgGetTime();		\
	  BgMsgSetTiming(m); 	\
	  if (tTHREADTYPE == WORK_THREAD) {	\
            BgTimeLineRec &tlinerec = tTIMELINEREC;	\
            int n = tlinerec.length();			\
            if (n>0) {					\
              bgTimeLog *tlog = tlinerec[n-1];		\
	      if (tlog->endTime == 0.0)			\
                tlog->addMsg(m, node, tid, local);	\
	      else {	 /* standalone msg */		\
		  double curT = CmiBgMsgRecvTime(m);		\
		  bgTimeLog *newLog = new bgTimeLog(-1, "addMsg", curT, curT); \
		  newLog->recvTime = newLog->effRecvTime = curT;	\
                  newLog->addMsg(m, node, tid, local);		\
		  tlinerec.logEntryInsert(newLog);			\
		  tlinerec.clearSendingLogs();		\
		}					\
            }						\
	    /* log[log.length()-1]->print(); */		\
          }	\
	  if (timingMethod == BG_WALLTIME)\
                tSTARTTIME = CmiWallTimer();\
          else if (timingMethod == BG_ELAPSE)\
                tSTARTTIME = tCURRTIME;	\
	}
#else
#define BG_ENTRYSTART(handler, m)
#define BG_ENTRYEND()
#define BG_ADDMSG(m, node, tid)
#endif
#endif

#endif
