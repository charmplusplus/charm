/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _TRACE_H
#define _TRACE_H

#include "charm++.h"

class envelope;

/**
 \defgroup CkPerf  Charm++ Trace Module
*/
/*@{*/

// An additional interface for summary data
extern "C" void traceClearEps();
extern "C" void traceCommonInit(char **argv);

/// Base class of all tracing strategies.
class Trace {
  public:
    virtual void userEvent(int) {}
    virtual void creation(envelope *, int num=1) {}
    virtual void beginExecute(envelope *) {}
    virtual void beginExecute(int event,int msgType,int ep,int srcPe,int ml) {}
    virtual void endExecute(void) {}
    virtual void beginIdle(void) {}
    virtual void endIdle(void) {}
    virtual void beginPack(void) {}
    virtual void endPack(void) {}
    virtual void beginUnpack(void) {}
    virtual void endUnpack(void) {}
    virtual void beginCharmInit(void) {}
    virtual void endCharmInit(void) {}
    virtual void enqueue(envelope *) {}
    virtual void dequeue(envelope *) {}
    virtual void beginComputation(void) {}
    virtual void endComputation(void) {}

    virtual void traceInit(char **argv) {}
    virtual int traceRegisterUserEvent(const char*) {return 0;}
    virtual void traceClearEps() {}
    virtual void traceWriteSts() {}
    virtual void traceClose() {}
    virtual void traceBegin() {}
    virtual void traceEnd() {}
};

#define ALLDO(x) for (int i=0; i<traces.length(); i++) traces[i]->x

/// Array of Traces modules,  every event raised will go through every Trace module.
class TraceArray {
private:
  CkVec<Trace *>  traces;
  int n;
public:
    TraceArray(): n(0) {}
    inline void addTrace(Trace *tr) { traces.push_back(tr); n++;}
    inline int length() const { return n; }

    inline void userEvent(int e) { ALLDO(userEvent(e));}
    inline void creation(envelope *env, int num=1) { ALLDO(creation(env, num));}
    inline void beginExecute(envelope *env) {ALLDO(beginExecute(env));}
    inline void beginExecute(int event,int msgType,int ep,int srcPe, int mlen) {ALLDO(beginExecute(event, msgType, ep, srcPe, mlen));}
    inline void endExecute(void) {ALLDO(endExecute());}
    inline void beginIdle(void) {ALLDO(beginIdle());}
    inline void endIdle(void) {ALLDO(endIdle());}
    inline void beginPack(void) {ALLDO(beginPack());}
    inline void endPack(void) {ALLDO(endPack());}
    inline void beginUnpack(void) {ALLDO(beginUnpack());}
    inline void endUnpack(void) {ALLDO(endUnpack());}
    inline void beginCharmInit(void) {ALLDO(beginCharmInit()); }
    inline void endCharmInit(void) {ALLDO(endCharmInit());}
    inline void enqueue(envelope *e) {ALLDO(enqueue(e));}
    inline void dequeue(envelope *e) {ALLDO(dequeue(e));}
    inline void beginComputation(void) {ALLDO(beginComputation());}
    inline void endComputation(void) {ALLDO(endComputation());}
    inline void traceInit(char **argv) {ALLDO(traceInit(argv));}
    inline int traceRegisterUserEvent(const char*x) {
	  int eno = 0;
	  for (int i=0; i<traces.length(); i++) {
	    int e = traces[i]->traceRegisterUserEvent(x);
	    if (e) eno = e;
          }
	  return eno;
    }  
    inline void traceClearEps() {ALLDO(traceClearEps());}
    inline void traceWriteSts() {ALLDO(traceWriteSts());}
    inline void traceClose() {ALLDO(traceClose());}
    inline void traceBegin() {ALLDO(traceBegin());}
    inline void traceEnd() {ALLDO(traceEnd());}
};

CpvExtern(TraceArray*, _traces);

CpvExtern(int, CtrLogBufSize);
CpvExtern(char*, traceRoot);
CpvExtern(double, traceInitTime);

extern "C" {
#include "conv-trace.h"
}

inline double TraceTimer() { return CmiWallTimer() - CpvAccess(traceInitTime); }

#ifndef CMK_OPTIMIZE
#  define _TRACE_ONLY(code) do{if(CpvAccess(traceOn)){ code; }} while(0)
#else
#  define _TRACE_ONLY(code) /*empty*/
#endif

#define _TRACE_USER_EVENT(x) _TRACE_ONLY(CpvAccess(_traces)->userEvent(x))
#define _TRACE_CREATION_1(env) _TRACE_ONLY(CpvAccess(_traces)->creation(env))
#define _TRACE_CREATION_N(env, num) _TRACE_ONLY(CpvAccess(_traces)->creation(env, num))
#define _TRACE_BEGIN_EXECUTE(env) _TRACE_ONLY(CpvAccess(_traces)->beginExecute(env))
#define _TRACE_BEGIN_EXECUTE_DETAILED(evt,typ,ep,src,mlen) \
	_TRACE_ONLY(CpvAccess(_traces)->beginExecute(evt,typ,ep,src,mlen))
#define _TRACE_END_EXECUTE() _TRACE_ONLY(CpvAccess(_traces)->endExecute())
#define _TRACE_BEGIN_IDLE() _TRACE_ONLY(CpvAccess(_traces)->beginIdle())
#define _TRACE_END_IDLE() _TRACE_ONLY(CpvAccess(_traces)->endIdle())
#define _TRACE_BEGIN_PACK() _TRACE_ONLY(CpvAccess(_traces)->beginPack())
#define _TRACE_END_PACK() _TRACE_ONLY(CpvAccess(_traces)->endPack())
#define _TRACE_BEGIN_UNPACK() _TRACE_ONLY(CpvAccess(_traces)->beginUnpack())
#define _TRACE_END_UNPACK() _TRACE_ONLY(CpvAccess(_traces)->endUnpack())
#define _TRACE_BEGIN_CHARMINIT() _TRACE_ONLY(CpvAccess(_traces)->beginCharmInit())
#define _TRACE_END_CHARMINIT() _TRACE_ONLY(CpvAccess(_traces)->endCharmInit())
#define _TRACE_BEGIN_COMPUTATION() _TRACE_ONLY(CpvAccess(_traces)->beginComputation())
#define _TRACE_END_COMPUTATION() _TRACE_ONLY(CpvAccess(_traces)->endComputation())
#define _TRACE_ENQUEUE(env) _TRACE_ONLY(CpvAccess(_traces)->enqueue(env))
#define _TRACE_DEQUEUE(env) _TRACE_ONLY(CpvAccess(_traces)->dequeue(env))

#endif


/*@}*/

