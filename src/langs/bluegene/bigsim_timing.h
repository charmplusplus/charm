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
  a message sent event
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
  void print();
  void write(FILE *fp);
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
  inline void addMsg(char *msg) { msgs.push_back(new bgMsgEntry(msg)); }
  void print(int node, int th);
  void write(FILE *fp);

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
extern void BgWriteThreadTimeLine(char **argv, int x, int y, int z, int th, BgTimeLine &tline);
extern void BgGetMsgStartTime(double recvTime, BgTimeLine &tline, double* startTime, int index);
extern void BgAdjustTimeLineInsert(BgTimeLine &tline);
extern int BgAdjustTimeLineForward(int msgID, double tAdjustAbs, BgTimeLine &tline);

#if BLUEGENE_TIMING

#define BG_ENTRYSTART(handler, m)  \
        if (genTimeLog)	\
	  if (tTHREADTYPE == WORK_THREAD) 	\
	    tMYNODE->timelines[tMYID].enq(new bgTimeLog(handler, m));

#define BG_ENTRYEND()  \
        if (genTimeLog)	{ \
	  if (tTHREADTYPE == WORK_THREAD) {	\
            BgTimeLine &log = tMYNODE->timelines[tMYID];	\
            log[log.length()-1]->closeLog();	\
	    if (correctTimeLog) BgAdjustTimeLineInsert(log);	\
          }	\
	}

#define BG_ADDMSG(m)  	\
        if (genTimeLog)	{ \
          BgGetTime();		\
	  BgMsgSetTiming(m); 	\
	  if (tTHREADTYPE == WORK_THREAD) {	\
            BgTimeLine &log = tMYNODE->timelines[tMYID];	\
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

#endif
