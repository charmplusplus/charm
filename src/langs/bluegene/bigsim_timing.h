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
  char   core[CmiBlueGeneMsgHeaderSizeBytes];
  int    msgID;	
  int    tID;		// destination worker thread ID
  double tAdjust;	// correction in recvTime at destPe
  int 	 destNode;
};

/**
  a message sent event
*/
class bgMsgEntry {
public:
  int msgID;
  int tID;		// destination worker thread ID
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
  double recvTime;	//Time at which the message was received in 'inbuffer'
  double startTime, endTime;
  int srcpe;        // source bg node 
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

CpvExtern(int, bgCorrectionHandler);

extern void BgInitTiming();
extern void BgMsgSetTiming(char *msg);
extern void BgPrintThreadTimeLine(int node, int th, BgTimeLine &tline);
extern void BgGetMsgStartTime(double recvTime, BgTimeLine &tline, double* startTime, int index);
extern void BgAdjustTimeLineInsert(bgTimeLog* tlog, BgTimeLine &tline);
extern void BgAdjustTimeLineForward(int msgID, double tAdjust, BgTimeLine &tline);

#if BLUEGENE_TIMING

#define BG_ENTRYSTART(handler, m)  \
	if (tTHREADTYPE == WORK_THREAD) tMYNODE->timelines[tMYID].enq(new bgTimeLog(handler, m));

#define BG_ENTRYEND()  \
	if (tTHREADTYPE == WORK_THREAD) {	\
          BgTimeLine &log = tMYNODE->timelines[tMYID];	\
          log[log.length()-1]->closeLog();	\
		  BgAdjustTimeLineInsert(log);	\
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
