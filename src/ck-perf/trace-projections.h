/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkPerf
*/
/*@{*/

#ifndef _PROJECTIONS_H
#define _PROJECTIONS_H

#include "trace.h"
#include "ck.h"
#include "stdio.h"
#include "errno.h"

#include "trace-common.h"

/// a log entry in trace projection
class LogEntry {
  public:
    void *operator new(size_t s) {void*ret=malloc(s);_MEMCHECK(ret);return ret;}
    void *operator new(size_t, void *ptr) { return ptr; }
    void operator delete(void *ptr) { free(ptr); }
#ifdef WIN32
    void operator delete(void *, void *) { }
#endif
    LogEntry() {}
    LogEntry(double tm, UChar t, UShort m=0, UShort e=0, int ev=0, int p=0, int ml=0) { 
      type = t; mIdx = m; eIdx = e; event = ev; pe = p; time = tm; msglen = ml;
    }
    double time;
    int event;
    int pe;
    UShort mIdx;
    UShort eIdx;
    UChar type; 
    int msglen;
    void write(FILE *fp);
    void writeBinary(FILE *fp);
};

/// log pool in trace projection
class LogPool {
  private:
    UInt poolSize;
    UInt numEntries;
    LogEntry *pool;
    FILE *fp;
    char *fname;
    int binary;
  public:
    LogPool(char *pgm, int b);
    ~LogPool();
    void write(void);
    void writeBinary(void) {
      for(UInt i=0; i<numEntries; i++)
        pool[i].writeBinary(fp);
    }
    void writeSts(void);
    void add(UChar type,UShort mIdx,UShort eIdx,double time,int event,int pe, int ml=0);
};

/// class for recording trace projections events 
/**
  TraceProjections will log Converse/Charm++ events and write into .log files;
  events descriptions will be written into .sts file.
*/
class TraceProjections : public Trace {
    LogPool* _logPool;        /**<  logpool for all events */
    int curevent;
    int execEvent;
    int execEp;
    int execPe;
    int isIdle;
    int cancel_beginIdle, cancel_endIdle;
  public:
    TraceProjections(char **argv);
    void userEvent(int e);
    void creation(envelope *e, int num=1);
    void beginExecute(envelope *e);
    void beginExecute(int event,int msgType,int ep,int srcPe,int ml);
    void endExecute(void);
    void beginIdle(void);
    void endIdle(void);
    void beginPack(void);
    void endPack(void);
    void beginUnpack(void);
    void endUnpack(void);
    void beginCharmInit(void);
    void endCharmInit(void);
    void enqueue(envelope *e);
    void dequeue(envelope *e);
    void beginComputation(void);
    void endComputation(void);

    int traceRegisterUserEvent(const char*);
    void traceClearEps();
    void traceWriteSts();
    void traceClose();
    void traceBegin();
    void traceEnd();
};


#endif

/*@}*/
