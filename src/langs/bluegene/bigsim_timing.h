#ifndef BLUE_TIMING_H
#define BLUE_TIMING_H

#include "cklists.h"

#define BLUEGENE_TIMING     	1

#if CMK_HAS_VALUES_H
#   include <values.h>
#   define INVALIDTIME  MAXDOUBLE
#   define CMK_MAXINT   MAXINT
#else
#   define INVALIDTIME  (9999999999.99)
#   define CMK_MAXINT   (1000000000)
#endif

/* optimization parameters */
#define SCHEDULE_WORK       1
#define USE_MULTISEND	    0		/* must be BATCH_PROCESSING */
#define DELAY_CHECK         1
#define LIMITED_SEND        0		/* BATCH_PROCESSING must be 1 or 2 */
#define THROTTLE_WORK       1
#define DELAY_SEND          1

#define LEASH               0.00005         /* 50us leash window */

#define CHECK_INTERVAL   10
#define CHECK_THRESHOLD  80000

#define CORRECTSENDLEN  5

#define PRIO_FACTOR      (1e8)

#define EPSILON      (1e-9)

#ifdef max
#undef max
#undef min
#endif
#define max(a,b) ((a)>=(b)?(a):(b))
#define min(a,b) ((a)<=(b)?(a):(b))
#define ABS(x)   ((x)>=0?(x):(-x))

extern int bgcorroff;
extern int programExit;
extern double gvt;
//extern int realMsgProcCount,corrMsgProcCount;
extern int  genTimeLog;
extern int  correctTimeLog;
extern int bgSkipEndFlag;

inline int isZero(double input){
  return (input < EPSILON && input > -EPSILON);
}

inline int isLess(double v1, double v2){
  return (v1 < v2-EPSILON);
}

inline int isEqual(double v1, double v2){
  return isZero(v1-v2);
}

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

/**
  a message sent event in timeline
*/
class bgMsgEntry {
public:
  int msgID;
  int dstPe;		// dest bg node in global sequence
  double recvTime;
#if DELAY_SEND
  char *sendMsg;	// real msg
#endif
  CmiInt2 tID;		// destination worker thread ID
//  double sendtime;
public:
  bgMsgEntry(char *msg, int node, int tid, int local);
  inline void print() {
    CmiPrintf("msgID:%d recvtime:%f dstPe:%d\n", msgID, recvTime, dstPe);
  }
  void write(FILE *fp) {
    fprintf(fp, "msgID:%d recvtime:%f dstPe:%d\n", msgID, recvTime, dstPe);
  }
#if DELAY_SEND
  void send();
#endif
};


/**
  event for higher level of tracing like trace projections
*/
class bgEvents {
private:
  void*   data;         // e.g. can be pointer to trace projection log entry
  double  rTime;	// relative time from the start entry
  bgEventCallBackFn  callbackFn;
  void* usrPtr;
  char   eType;
public:
  bgEvents(void *d, double t, bgEventCallBackFn fn, void *uptr, char e): 
	data(d), rTime(t), callbackFn(fn), usrPtr(uptr), eType(e) {}
  inline void update(double startT, double recvT, int e) {
	if (eType==e) callbackFn(data, startT+rTime, recvT, usrPtr);
  }
};

/**
  one time log for an handler function;
  it record a list of message sent events in an execution of handler
*/
class bgTimeLog {
public:
  int ep;
  int seqno;
  int srcnode;        // source bg node  (srcnode,msgID) is the source msg
  int msgID;
  double recvTime;	//Time at which the message was received in 'inbuffer'
  double startTime, endTime;
  double oldStartTime, execTime;
  double effRecvTime;
  CkVec< bgMsgEntry * > msgs;
  CkVec< bgEvents * > evts;
  CkVec< bgTimeLog* > backwardDeps;
  CkVec< bgTimeLog* > forwardDeps;
  char doCorrect;
  char name[20];
public:
  bgTimeLog(bgTimeLog *);
  bgTimeLog(): ep(-1), recvTime(.0), startTime(.0), endTime(.0), msgID(-1), effRecvTime(INVALIDTIME), seqno(0), doCorrect(1) {strcpy(name,"dummyname");}
  bgTimeLog(int epc, char* name, double sTime, double eTime);
  bgTimeLog(int epc, char *msg);
  bgTimeLog(int epc, char* name, double sTime);
  ~bgTimeLog();

  void setExecTime();
  void closeLog();
  inline void addMsg(char *msg, int node, int tid, int local) { msgs.push_back(new bgMsgEntry(msg, node, tid, local)); }
  void print(int node, int th);
  void write(FILE *fp);

  void addBackwardDep(bgTimeLog* log);
  //takes a list of Logs on which this log is dependent (backwardDeps) 
  void addBackwardDeps(CkVec<bgTimeLog*> logs);
  void addBackwardDeps(CkVec<void*> logs);
  //Returns earliest time by which all backward dependents ended  
  double getEndOfBackwardDeps(); 
  int adjustTimeLog(double tAdjust,CkQ<bgTimeLog *> &tline, int mynode, int);
  inline void addEvent(void *data,double absT,bgEventCallBackFn fn,void *p,int e) { 
    evts.push_back(new bgEvents(data, absT-startTime, fn, p, e)); 
  }
  inline void updateEvents(int e) {
    for (int i=0; i<evts.length(); i++)
      evts[i]->update(startTime ,recvTime, e);
  }
  double key() { return effRecvTime; }
  inline int compareKey(bgTimeLog* otherLog){
    if(((isZero(effRecvTime-otherLog->effRecvTime))&&(seqno < otherLog->seqno))
       ||(isLess(effRecvTime,otherLog->effRecvTime)))
      return -1;
    return 1;
  }
#if DELAY_SEND
  void send() {
    for (int i=0; i<msgs.length(); i++)
      msgs[i]->send();
  }
#endif
};


/**
  an entry in a time log
  it record a list of message sent events
*/
typedef CkQ< bgTimeLog *> BgTimeLine;
class BgTimeLineRec {
public:
  BgTimeLine  timeline;
  int         commit;
  int         startIdx;
  int         startCorrFlag;
  int         correctSendIdx;
  int 	      counter;
  double      minCorrection;
  bgTimeLog  *bgCurLog;
#if DELAY_SEND
  CkQ<bgTimeLog *>   sendingLogs;	// send buffered
#endif
public:
  BgTimeLineRec(): timeline(1024), commit(0), counter(1), correctSendIdx(0), startIdx(0), bgCurLog(NULL) {
    if (bgcorroff) startCorrFlag=0; else startCorrFlag=1;
    minCorrection = INVALIDTIME;
  }
  bgTimeLog * operator[](size_t n)
  {
	CmiAssert(n!=-1);
        return timeline[n];
  }
  int length() { return timeline.length(); }
  // special enq which will assign seqno
  void enq(bgTimeLog *log, int isnew) {
	log->seqno = counter++;
  	timeline.enq(log);
#if DELAY_SEND
	if (isnew) sendingLogs.enq(log);
#endif
  }
  void setStartIdx(){
    startIdx = timeline.length();
  }
  double computeUtil(int *numRealMsgs){
    //From startIdx to the end of the timeline
    double total=0.0;
    int tlineLen = length();
    for(int i=0;i<tlineLen;i++) {
      bgTimeLog *log = timeline[i];
      total += log->execTime;
      *numRealMsgs += log->msgs.length();
    }
    return total;
  }
  inline void clearSendingLogs() {
#if DELAY_SEND
    while (!sendingLogs.isEmpty()) {
      bgTimeLog *log = sendingLogs.deq();
      log->send();
    }
#endif
  }
  void logEntryStart(int handler, char *m);
  void logEntryCommit();
  void logEntryInsert(bgTimeLog* log);
  void logEntryStart(bgTimeLog* log);
  void logEntryClose();
  void logEntrySplit();
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
