//Contains the structures for bg logs and timelines
#ifndef BLUE_LOGS_H
#define BLUE_LOGS_H

#include <string.h>

#include "blue.h"
#include "blue_defs.h"
#include "cklists.h"

extern int bglog_version;

extern int bgcorroff;

// identifier for a message which records the source node that generate
// this message and a message sequence number (_msgID)
class BgMsgID
{
private:
  int _pe;		// src PE number where the message is created
  int _msgID;		// local index number on pe

public:
  BgMsgID(): _pe(-1), _msgID(-1) {}
  BgMsgID(int p, int m): _pe(p), _msgID(m) {}
  void pup(PUP::er &p) {
    p|_pe; p|_msgID;
  }
  inline int operator == (const BgMsgID &m) {
    return _pe == m._pe && _msgID == m._msgID;
  }
  int pe() { return _pe; }
  int msgID() { return _msgID; }
  void setPe(int pe) { _pe = pe; }
  void setMsgID(int msgID) { _msgID = msgID; }
};

/**
  a message sent event in timeline
*/
class BgMsgEntry {
  friend class BgTimeLog;
public:
  int msgID;
  int dstNode;          // dest bg node in global sequence
  double sendTime;	// msg sending offset in the event
  double recvTime;	// predicted recv time with delay
#if DELAY_SEND
  char *sendMsg;	// real msg
#endif
  CmiInt2 tID;		// destination worker thread ID
  int msgsize;		// message size
  int group;		// number of messages in this group
private:
  BgMsgEntry() {}
public:
  BgMsgEntry(int seqno, int _msgSize, double _sendTime, double _recvTime, int dstNode, int destrank);
  BgMsgEntry(char *msg, int node, int tid, double sendT, int local, int g=1);
  inline void print() {
    CmiPrintf("msgID:%d sent:%f recvtime:%f dstNode:%d tid:%d group:%d\n", msgID, sendTime, recvTime, dstNode, tID, group);
  }
  void write(FILE *fp) {
    if(dstNode >= 0)
      fprintf(fp, "-msgID:%d sent:%f recvtime:%f dstNode:%d tid:%d size:%d group:%d\n", msgID, sendTime, recvTime, dstNode, tID, msgsize, group);
    if(dstNode == -1)
      fprintf(fp, "-msgID:%d sent:%f recvtime:%f dstNode:BG_BROADCASTALL tid:%d size:%d group:%d\n", msgID, sendTime, recvTime, tID, msgsize, group);
    if(dstNode <= -100)
      fprintf(fp, "-msgID:%d sent:%f recvtime:%f dstNode:BG_BROADCASTALL except %d tid:%d size:%d group:%d\n", msgID, sendTime, recvTime, -100-dstNode, tID, msgsize, group);
    
  }
#if DELAY_SEND
//  void send();
#endif
  void pup(PUP::er &p);
};

/**
  event for higher level of tracing like trace projections
*/
class bgEvents {
private:
  bgEventCallBackFn  callbackFn;
  void* usrPtr;
public:
  void*   data;         // e.g. can be pointer to trace projection log entry
  int     index;		// index of the event to its original log pool.
  double  rTime;	// relative time from the start entry
  char   eType;
  bgEvents(): index(-1) {}
  bgEvents(void *d, int idx, double t, bgEventCallBackFn fn, void *ptr, char e):
	data(d), index(idx), rTime(t), callbackFn(fn), usrPtr(ptr), eType(e) {}
  inline void update(double startT, double recvT, int e) {
	if (eType==e) callbackFn(data, startT+rTime, recvT, usrPtr);
  }
  void print();
  void write(FILE *fp);
  void pup(PUP::er &p);
};

#define BG_STARTSIM     0x1
#define BG_QD           0x2

extern void BgDelaySend(BgMsgEntry *msgEntry);

class BgTimeLineRec;

enum BgMPIOp { MPI_NONE = 0, MPI_BARRIER = 1, MPI_ALLREDUCE = 2 , MPI_ALLTOALL = 3};

#define BGLOG_NAMELEN   20

/**
  one time log for an handler function;
  it records a list of message sent events in an execution of handler
*/
class BgTimeLog {
public:
  CkVec< BgMsgEntry * > msgs;
  CkVec< bgEvents * > evts;
  CkVec< BgTimeLog* > backwardDeps;
  CkVec< BgTimeLog* > forwardDeps;
  CmiObjId objId;
  BgMsgID  msgId;	// incoming message that generates this log

  double recvTime;	//Time at which the message was received in 'inbuffer'
  double startTime, endTime;
  double oldStartTime, execTime;
  double effRecvTime;

  int ep;
  int seqno;
  unsigned int mpiSize;
  unsigned short mpiOp;
  short charm_ep;

  char name[BGLOG_NAMELEN];
  char doCorrect;
  char flag;

  friend class BgTimeLineRec;
public:
  BgTimeLog(BgTimeLog *);
  BgTimeLog(const BgMsgID &msgID);
  BgTimeLog(char *msg, char *str=NULL);
  BgTimeLog();
  BgTimeLog(int epc, const char* name, double sTime, double eTime);
  BgTimeLog(int epc, const char* name, double sTime);
  ~BgTimeLog();

  inline void setName(const char *_name) { CmiAssert(strlen(_name)<20); strcpy(name, _name); }
  inline void setEP(int _ep) { ep = _ep; }
  inline void setCharmEP(short _ep) { charm_ep = _ep; }
  inline void setTime(double stime, double etime) {
         startTime = stime;
         endTime = etime;
         setExecTime();
  }
  inline void setExecTime() {
           execTime = endTime - startTime;
           if(execTime < BG_EPSILON && execTime > -BG_EPSILON)
             execTime = 0.0;
           CmiAssert(execTime >= 0.0);
         }
  inline void addMsg(BgMsgEntry *mentry) {
           msgs.push_back(mentry);
         }
  inline void addMsg(char *msg, int node, int tid, double sendT, int local, int group=1) { 
           msgs.push_back(new BgMsgEntry(msg, node, tid, sendT, local, group)); 
         }
  inline void setObjId(CmiObjId *idx) {
           memcpy(&objId, idx, sizeof(CmiObjId));
         }
  void closeLog();
  void print(int node, int th);
  void write(FILE *fp);

  inline void setStartEvent() { flag |= BG_STARTSIM; }
  inline int isStartEvent() { return (flag & BG_STARTSIM); }
  inline int isQDEvent() { return (flag & BG_QD); }

  // add backward dep of the log corresponent to msg
  void addMsgBackwardDep(BgTimeLineRec &tlinerec, void* msg);
  void addBackwardDep(BgTimeLog* log);
  //takes a list of Logs on which this log is dependent (backwardDeps) 
  void addBackwardDeps(CkVec<BgTimeLog*> logs);
  void addBackwardDeps(CkVec<void*> logs);
  int bDepExists(BgTimeLog* log);			// by guna
  //Returns earliest time by which all backward dependents ended  
  // return the last eff recv time
  double getEndOfBackwardDeps() {
    double maxEndTime =0.0;
    for(int i=0;i<backwardDeps.length();i++)
//    maxEndTime = max(maxEndTime,backwardDeps[i]->endTime);
      maxEndTime = BG_MAX(maxEndTime,backwardDeps[i]->effRecvTime);
      
    return maxEndTime;
  }

  inline void addEvent(void *data,int idx,double absT,bgEventCallBackFn fn,void *p,int e) { 
    evts.push_back(new bgEvents(data, idx, absT-startTime, fn, p, e)); 
  }
  inline void updateEvents(int e) {
    for (int i=0; i<evts.length(); i++)
      evts[i]->update(startTime ,recvTime, e);
  }
  inline double key() { return effRecvTime; }
  inline int compareKey(BgTimeLog* otherLog){
    if(((isZero(effRecvTime-otherLog->effRecvTime))&&(seqno < otherLog->seqno))
       ||(isLess(effRecvTime,otherLog->effRecvTime)))
      return -1;
    return 1;
  }
  inline int isEqual(BgTimeLog* otherLog){
    return (otherLog==this);
  }
  void pupCommon(PUP::er &p);
  void pup(PUP::er &p);
  void winPup(PUP::er &p, int& firstLogToRead, int& numLogsToRead);
#if DELAY_SEND
  void send() {
    for (int i=0; i<msgs.length(); i++)
      BgDelaySend(msgs[i]);
  }
#endif
};


/**
  Timeline for a VP
*/
typedef CkQ< BgTimeLog *> BgTimeLine;

/**
  A wrapper for CkQ of BgTimeLine
*/
class BgTimeLineRec {
public:
  BgTimeLine  timeline;
  int         commit;
  int         startIdx;
  int         startCorrFlag;
  int         correctSendIdx;
  int 	      counter;
  double      minCorrection;
  BgTimeLog  *bgCurLog;		/* current unfinished log */
  BgTimeLog  *bgPrevLog;	/* previous log that should make dependency */
#if DELAY_SEND
  CkQ<BgTimeLog *>   sendingLogs;	// send buffered
#endif
public:
  BgTimeLineRec(): timeline(1024), commit(0), counter(1), correctSendIdx(0), 
		   startIdx(0), bgCurLog(NULL), bgPrevLog(NULL) {
      if (bgcorroff) startCorrFlag=0; else startCorrFlag=1;
      minCorrection = INVALIDTIME;
    }
  ~BgTimeLineRec() {
      for (int i=0; i<timeline.length(); i++)  delete timeline[i];
    }
  BgTimeLog * operator[](size_t n) {
	CmiAssert(n!=(size_t)-1);
        return timeline[n];
    }
  int length() { return timeline.length(); }
  // special enq which will assign seqno
  void enq(BgTimeLog *log, int isnew) {
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
      BgTimeLog *log = timeline[i];
      total += log->execTime;
      *numRealMsgs += log->msgs.length();
    }
    return total;
  }
  inline void clearSendingLogs() {
#if DELAY_SEND
    while (!sendingLogs.isEmpty()) {
      BgTimeLog *log = sendingLogs.deq();
      log->send();
    }
#endif
  }
  void logEntryStart(char *m);
//  void logEntryCommit();
  void logEntryInsert(BgTimeLog* log);
  void logEntryStart(BgTimeLog* log);
  void logEntryClose();
  void logEntrySplit(const char *name = "split-broadcast");
  BgTimeLog *logSplit(const char *name, BgTimeLog **parentlogs, int n);
  BgTimeLog *getTimeLogOnThread(const BgMsgID &msgId, int *index);

  void pup(PUP::er &p);
  void winPup(PUP::er &p, int& firstLogToRead, int& numLogsToRead, int& tLineLength);
};

// BigSim log function API
int BgIsInALog(BgTimeLineRec &tlinerec);
BgTimeLog *BgLastLog(BgTimeLineRec &tlinerec);
void BgAddBackwardDep(BgTimeLog *curlog, BgTimeLog* deplog);
BgTimeLog *BgStartLogByName(BgTimeLineRec &tlinerec, int ep, const char *name, double starttime, BgTimeLog *prevLog);
void BgEndLastLog(BgTimeLineRec &tlinerec);

int BgLogGetThreadEP();
int BgLoadTraceSummary(const char *fname, int &totalWorkerProcs, int &numX, int &numY, int &numZ, int &numCth, int &numWth, int &numEmulatingPes);
int BgReadProc(int procNum, int numWth, int numPes, int totalWorkerProcs, int* allNodeOffsets, BgTimeLineRec& tlinerec);
int BgReadProcWindow(int procNum, int numWth, int numPes, int totalProcs, int* allNodeOffsets, BgTimeLineRec& tlinerec,
		     int& fileLoc, int& totalTlineLength, int firstLog, int numLogs);
int* BgLoadOffsets(int totalProcs, int numPes);
void BgWriteThreadTimeLine(const char *fname, int x, int y, int z, int th, BgTimeLine &tline);
void BgWriteTraceSummary(int numEmulatingPes, int x, int y=1, int z=1, int numWth=1, int numCth=1, const char *fname=NULL, char *traceroot=NULL);
void BgWriteTimelines(int seqno, BgTimeLineRec **tlinerecs, int nlocalNodes, char *traceroot=NULL);
void BgWriteTimelines(int seqno, BgTimeLineRec *tlinerecs, int nlocalNodes, char *traceroot=NULL);
extern "C" void BgGenerateLogs();

#endif
