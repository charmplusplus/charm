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

// An additional interface for summary data
extern "C" void traceClearEps();
extern "C" void traceCommonInit(char **argv);

// Base class of all tracing strategies.

class Trace {
  public:
    virtual void userEvent(int) {}
    virtual void creation(envelope *, int num=1) {}
    virtual void beginExecute(envelope *) {}
    virtual void beginExecute(int event,int msgType,int ep,int srcPe) {}
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

// Array of Traces so that every event can go through every Trace registered.

class TraceArray {
private:
  CkVec<Trace *>  traces;
public:
    void addTrace(Trace *tr) { traces.push_back(tr); }
    int length() { return traces.length(); }

    void userEvent(int e) { ALLDO(userEvent(e));}
    void creation(envelope *env, int num=1) { ALLDO(creation(env, num));}
    void beginExecute(envelope *env) {ALLDO(beginExecute(env));}
    void beginExecute(int event,int msgType,int ep,int srcPe) {ALLDO(beginExecute(event, msgType, ep, srcPe));}
    void endExecute(void) {ALLDO(endExecute());}
    void beginIdle(void) {ALLDO(beginIdle());}
    void endIdle(void) {ALLDO(endIdle());}
    void beginPack(void) {ALLDO(beginPack());}
    void endPack(void) {ALLDO(endPack());}
    void beginUnpack(void) {ALLDO(beginUnpack());}
    void endUnpack(void) {ALLDO(endUnpack());}
    void beginCharmInit(void) {ALLDO(beginCharmInit()); }
    void endCharmInit(void) {ALLDO(endCharmInit());}
    void enqueue(envelope *e) {ALLDO(enqueue(e));}
    void dequeue(envelope *e) {ALLDO(dequeue(e));}
    void beginComputation(void) {ALLDO(beginComputation());}
    void endComputation(void) {ALLDO(endComputation());}
    void traceInit(char **argv) {ALLDO(traceInit(argv));}
    int traceRegisterUserEvent(const char*x) {ALLDO(traceRegisterUserEvent(x));}
    void traceClearEps() {ALLDO(traceClearEps());}
    void traceWriteSts() {ALLDO(traceWriteSts());}
    void traceClose() {ALLDO(traceClose());}
    void traceBegin() {ALLDO(traceBegin());}
    void traceEnd() {ALLDO(traceEnd());}
};

CpvExtern(TraceArray*, _traces);

CpvExtern(int, CtrLogBufSize);
CpvExtern(char*, traceRoot);

extern "C" {
#include "conv-trace.h"
}

#ifndef CMK_OPTIMIZE
#  define _TRACE_ONLY(code) do{if(CpvAccess(traceOn)){ code; }} while(0)
#else
#  define _TRACE_ONLY(code) /*empty*/
#endif

#define _TRACE_USER_EVENT(x) _TRACE_ONLY(CpvAccess(_traces)->userEvent(x))
#define _TRACE_CREATION_1(env) _TRACE_ONLY(CpvAccess(_traces)->creation(env))
#define _TRACE_CREATION_N(env, num) _TRACE_ONLY(CpvAccess(_traces)->creation(env, num))
#define _TRACE_BEGIN_EXECUTE(env) _TRACE_ONLY(CpvAccess(_traces)->beginExecute(env))
#define _TRACE_BEGIN_EXECUTE_DETAILED(evt,typ,ep,src) \
	_TRACE_ONLY(CpvAccess(_traces)->beginExecute(evt,typ,ep,src))
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


