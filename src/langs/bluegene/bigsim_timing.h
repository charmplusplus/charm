#ifndef BLUE_TIMING_H
#define BLUE_TIMING_H

#include "cklists.h"

#define BLUEGENE_TIMING     	1

/**
  a message sent event
*/
class bgMsgEntry {
public:
  int msgID;
  double sendtime;
  int dstPe;
public:
  bgMsgEntry(char *msg);
  void print();
};

/**
  one time log for an handler function;
  it record a list of message sent events in an execution of handler
*/
class bgTimeLog {
public:
  int ep;
  double startTime, endTime;
  int srcpe;                   // source bg node 
  int msgID;
  CkVec< bgMsgEntry * > msgs;
public:
  bgTimeLog(int epc, char *msg);
  ~bgTimeLog();
  void closeLog();
  void addMsg(char *msg);
  void print(int node, int th);

  void adjustTimeLog(double tAdjust);
};

/**
  an entry in a time log
  it record a list of message sent events
*/
typedef CkQ< bgTimeLog *> BgTimeLine;

extern void BgInitTiming();
extern void BgMsgSetTiming(char *msg);
extern void BgPrintThreadTimeLine(int node, int th, BgTimeLine &tline);
extern void BgAdjustTimeLineInit(bgTimeLog* tlog, BgTimeLine &tline);
extern void BgAdjustTimeLineForward(int msgID, double tAdjust, BgTimeLine &tline);

#if BLUEGENE_TIMING

#define BG_ENTRYSTART(handler, m)  \
	if (tTHREADTYPE == WORK_THREAD) tMYNODE->timelines[tMYID].enq(new bgTimeLog(handler, m));

#define BG_ENTRYEND()  \
	if (tTHREADTYPE == WORK_THREAD) {	\
          BgTimeLine &log = tMYNODE->timelines[tMYID];	\
          log[log.length()-1]->closeLog();	\
        }

#define BG_ADDMSG(m)  	\
	BgMsgSetTiming(m); 	\
	if (tTHREADTYPE == WORK_THREAD) {	\
          BgTimeLine &log = tMYNODE->timelines[tMYID];	\
          int n = log.length();				\
          if (n>0) {				\
            bgTimeLog *tline = log[n-1];	\
            tline->addMsg(m);				\
          }						\
	  /* log[log.length()-1]->print(); */		\
        }
#else
#define BG_ENTRYSTART(handler, m)
#define BG_ENTRYEND()
#define BG_ADDMSG(m)
#endif

#endif
