//Contains the structures for bg logs and timelines
#ifndef BLUE_LOGS_H
#define BLUE_LOGS_H

#include "blue_defs.h"
#include "cklists.h"

extern int bgcorroff;

/**
  a message sent event in timeline
*/
class bgMsgEntry {
  friend class bgTimeLog;
public:
  int msgID;
  int dstPe;		// dest bg node in global sequence
  double recvTime;
#if DELAY_SEND
  char *sendMsg;	// real msg
#endif
  CmiInt2 tID;		// destination worker thread ID
//  double sendtime;
  int msgsize;		// message size
private:
  bgMsgEntry() {}
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
  void pup(PUP::er &p) {
    p|msgID; p|dstPe; p|recvTime; p|tID; p|msgsize;
  }
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

  int index;		// by guna, need to verify, need to use sequence number
  int threadNum;	// by guna, for seq load balancing  ???

  CkVec< bgMsgEntry * > msgs;
  CkVec< bgEvents * > evts;
  CkVec< bgTimeLog* > backwardDeps;
  CkVec< bgTimeLog* > forwardDeps;
  char doCorrect;
  char name[20];

  friend class BgTimeLineRec;
private:
  bgTimeLog(char *msg);
public:
  bgTimeLog(bgTimeLog *);
  bgTimeLog(): ep(-1), recvTime(.0), startTime(.0), endTime(.0), msgID(-1), effRecvTime(INVALIDTIME), seqno(0), doCorrect(1) {strcpy(name,"dummyname");}
  bgTimeLog(int epc, char* name, double sTime, double eTime);
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
  int bDepExists(bgTimeLog* log);			// by guna
  //Returns earliest time by which all backward dependents ended  
  // return the last eff recv time
  double getEndOfBackwardDeps() {
    double maxEndTime =0.0;
    for(int i=0;i<backwardDeps.length();i++)
//    maxEndTime = max(maxEndTime,backwardDeps[i]->endTime);
      maxEndTime = max(maxEndTime,backwardDeps[i]->effRecvTime);
      
    return maxEndTime;
  }

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
  inline int isEqual(bgTimeLog* otherLog){
    return (otherLog==this);
  }
  void pup(PUP::er &p);

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
  void logEntryStart(char *m);
//  void logEntryCommit();
  void logEntryInsert(bgTimeLog* log);
  void logEntryStart(bgTimeLog* log);
  void logEntryClose();
  void logEntrySplit();

  void pup(PUP::er &p){
    int l=length();
    p|l;
    //    CmiPrintf("Puped len: %d\n",l);
    if(!p.isUnpacking()){
      for(int i=0;i<l;i++)
        timeline[i]->index = i;
    }
    else{
      //Timeline is empty when unpacking pup is called
      //timeline.removeFrom(0);
    }

    for (int i=0;i<l;i++) {
        if (p.isUnpacking()) {
                bgTimeLog* t = new bgTimeLog();
                t->pup(p);
                timeline.enq(t);
        }
        else {
          timeline[i]->pup(p);
        }
    }
  }
};

void readProc(int procNum, int numWth ,int numPes, int totalProcs, int* allNodeOffsets, BgTimeLineRec& tlinerec);
int* loadOffsets(int totalProcs, int numPes);

#endif
