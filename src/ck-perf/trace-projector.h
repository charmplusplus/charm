/***********Projector Trace File added by Sayantan ***********/
#ifndef _PROJECTOR_H
#define _PROJECTOR_H

#include "trace.h"
#include "ck.h"
#include "stdio.h"
#include "errno.h"
#include "allEvents.h"
#include "trace-common.h"

/// class for recording trace projector events 
/**
  TraceProjector will log Converse/Charm++ events and write into .log files;
  events descriptions will be written into .sts file.
*/

extern void _createTraceprojector(char **argv);

class TraceProjector : public Trace {
  private:
    int traceCoreOn;
  public:
    TraceProjector(char **argv);
    void userEvent(int e);
    void userBracketEvent(int e, double bt, double et);
    void creation(envelope *e, int ep, int num=1);
    void beginExecute(envelope *e);
    void beginExecute(char *) {}
    void beginExecute(CmiObjId  *tid);
    void beginExecute(int event,int msgType,int ep,int srcPe,int ml,CmiObjId *idx=NULL);
    void endExecute(void);
    void messageRecv(char *env, int pe);
    void beginIdle(double curWallTime);
    void endIdle(double curWallTime);
    void beginPack(void);
    void endPack(void);
    void beginUnpack(void);
    void endUnpack(void);
    void enqueue(envelope *e);
    void dequeue(envelope *e);
    void beginComputation(void);
    void endComputation(void);

    int traceRegisterUserEvent(const char*, int);
    void traceClearEps();
    void traceWriteSts();
    void traceClose();
    void traceBegin();
    void traceEnd();
};


#endif

