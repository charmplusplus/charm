#ifndef BLUE_TIMING_H
#define BLUE_TIMING_H

#include "cklists.h"

#define BLUEGENE_TIMING     	1

/**
  timing correction message
*/
class bgCorrectionMsg
{
public:
  char     core[CmiBlueGeneMsgHeaderSizeBytes];
  int      msgID;	
  CmiUInt2 tID;		// destination worker thread ID
  double   tAdjust;	// new absolute value of recvTime at destPe
  int 	   destNode;
};

/**
  a message sent event in timeline
*/
class bgMsgEntry {
public:
  int msgID;
  CmiUInt2 tID;		// destination worker thread ID
  double sendtime;
  double recvTime;
  int dstPe;
public:
  bgMsgEntry(char *msg);
  inline void print() {
    CmiPrintf("msgID:%d sendtime:%f dstPe:%d\n", msgID, sendtime, dstPe);
  }
  void write(FILE *fp) {
    fprintf(fp, "msgID:%d sendtime:%f dstPe:%d\n", msgID, sendtime, dstPe);
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
public:
  bgEvents(void *d, double t, bgEventCallBackFn fn): data(d), rTime(t), callbackFn(fn) {}
  inline void update(double startT, double recvT, void *usrPtr) {
    callbackFn(data, startT+rTime, recvT, usrPtr);
  }
};

/**
  one time log for an handler function;
  it record a list of message sent events in an execution of handler
*/
class bgTimeLog {
public:
  int ep;
  double recvTime;	//Time at which the message was received in 'inbuffer'
  double startTime, endTime;
  int srcnode;        // source bg node 
  int msgID;
  CkVec< bgMsgEntry * > msgs;
  CkVec< bgEvents * > evts;
public:
  bgTimeLog(): ep(-1), recvTime(.0), startTime(.0), endTime(.0), msgID(-1)  {}
  bgTimeLog(int epc, char *msg);
  ~bgTimeLog();
  void closeLog();
  inline void addMsg(char *msg) { msgs.push_back(new bgMsgEntry(msg)); }
  void print(int node, int th);
  void write(FILE *fp);

  void adjustTimeLog(double tAdjust);
  inline void addEvent(void *data, double absT, bgEventCallBackFn fn) { 
    evts.push_back(new bgEvents(data, absT-startTime, fn)); 
  }
  inline void updateEvents(void *usrPtr) {
    for (int i=0; i<evts.length(); i++)
      evts[i]->update(startTime, recvTime, usrPtr);
  }
};

/**
  an entry in a time log
  it record a list of message sent events
*/
typedef CkQ< bgTimeLog *> BgTimeLine;

CpvExtern(int, bgCorrectionHandler);

extern void BgInitTiming();
extern void BgMsgSetTiming(char *msg);
extern void BgPrintThreadTimeLine(int node, int th, BgTimeLine &tline);
extern void BgWriteThreadTimeLine(char **argv, int x, int y, int z, int th, BgTimeLine &tline);
extern void BgAdjustTimeLineInsert(BgTimeLine &tline);
extern int BgAdjustTimeLineForward(int msgID, double tAdjustAbs, BgTimeLine &tline);

#if BLUEGENE_TIMING

#define BG_ENTRYSTART(handler, m)  \
        if (genTimeLog)	\
	  if (tTHREADTYPE == WORK_THREAD) 	\
	    tTIMELINE.enq(new bgTimeLog(handler, m));

#define BG_ENTRYEND()  \
        if (genTimeLog)	{ \
	  if (tTHREADTYPE == WORK_THREAD) {	\
            BgTimeLine &log = tTIMELINE;	\
            log[log.length()-1]->closeLog();	\
	    if (correctTimeLog) BgAdjustTimeLineInsert(log);	\
          }	\
	}

#define BG_ADDMSG(m)  	\
        if (genTimeLog)	{ \
          BgGetTime();		\
	  BgMsgSetTiming(m); 	\
	  if (tTHREADTYPE == WORK_THREAD) {	\
            BgTimeLine &log = tTIMELINE;	\
            int n = log.length();				\
            if (n>0) {				\
              bgTimeLog *tline = log[n-1];	\
              tline->addMsg(m);				\
            }						\
	    /* log[log.length()-1]->print(); */		\
          }	\
          tSTARTTIME = CmiWallTimer();	\
	}
#else
#define BG_ENTRYSTART(handler, m)
#define BG_ENTRYEND()
#define BG_ADDMSG(m)
#endif

extern int  genTimeLog;
extern int  correctTimeLog;

#endif
