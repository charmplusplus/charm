#ifndef BLUE_TIMING_H
#define BLUE_TIMING_H

#include "cklists.h"

#define BLUEGENE_TIMING     	1

class bgMsgEntry {
public:
  int msgID;
  double sendtime;
  int dstPe;
public:
  bgMsgEntry(char *msg);
  void print();
};

class bgTimingLog {
public:
  int ep;
  double startTime, endTime;
  int srcpe;   // source bg node 
  int msgID;
  CkVec< bgMsgEntry * > msgs;
public:
  bgTimingLog(int epc, char *msg);
  ~bgTimingLog();
  void closeLog();
  void addMsg(char *msg);
  void print(int node, int th);

  void adjustTimingLog(double tAdjust);
};

typedef CkQ< bgTimingLog *> BgTimeLine;

extern void BgInitTiming();
extern void BgMsgSetTiming(char *msg);
extern void BgPrintThreadTimeLine(int node, int th, BgTimeLine &tline);
extern void BgAdjustTimeLineInit(bgTimingLog* tlog, BgTimeLine &tline);
extern void BgAdjustTimeLineForward(int msgID, double tAdjust, BgTimeLine &tline);

#if BLUEGENE_TIMING

#define BG_ENTRYSTART(m)  \
	if (tTHREADTYPE == WORK_THREAD) tMYNODE->timelines[tMYID].enq(new bgTimingLog(handler, m));

#define BG_ENTRYEND()  \
	if (tTHREADTYPE == WORK_THREAD) {	\
          BgTimeLine &log = tMYNODE->timelines[tMYID];	\
          log[log.length()-1]->closeLog();	\
        }

#define BG_ADDMSG(m)  	\
	BgMsgSetTiming(m); 	\
	if (tTHREADTYPE == WORK_THREAD) {	\
          BgTimeLine &log = tMYNODE->timelines[tMYID];	\
          bgTimingLog *tline = log[log.length()-1];	\
          tline->addMsg(m);				\
	  /* log[log.length()-1]->print(); */		\
        }
#else
#define BG_ENTRYSTART(m)
#define BG_ENTRYEND()
#define BG_ADDMSG(m)
#endif

#endif
