#ifndef BLUE_TIMING_H
#define BLUE_TIMING_H

#include "cklists.h"

#define BLUEGENE_TIMING     	1

class bgMsgEntry {
public:
  int msgID;
  double sendtime;
public:
  bgMsgEntry(char *msg);
  void print();
};

class bgTimingLog {
public:
  int ep;
  double time;
  int srcpe;   // source bg node 
  int msgID;
  CkVec< bgMsgEntry * > msgs;
public:
  bgTimingLog(int epc, char *msg);
  ~bgTimingLog();
  void addMsg(char *msg);
  void print();
};

typedef CkQ< bgTimingLog *> BgTimeLine;

extern void BgInitTiming();
extern void BgMsgSetTiming(char *msg);

#if BLUEGENE_TIMING

#define BG_ADDENTRY(m)  \
	if (tTHREADTYPE == WORK_THREAD) tMYNODE->timelines[tMYID].enq(new bgTimingLog(handler, m));

#define BG_ADDMSG(m)  	\
	BgMsgSetTiming(m); 	\
	if (tTHREADTYPE == WORK_THREAD) {	\
          BgTimeLine &log = tMYNODE->timelines[tMYID];	\
          log[log.length()-1]->addMsg(m);	\
	  /* log[log.length()-1]->print(); */		\
        }
#else
#define BG_ADDENTRY(m)
#define BG_ADDMSG(m)
#endif

#endif
