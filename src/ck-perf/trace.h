/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _TRACE_H
#define _TRACE_H

#include "converse.h"

class envelope;

// An additional interface for summary data
extern "C" void traceClearEps();
extern "C" void traceCommonInit(char **argv,int enabled);

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
};

CpvExtern(Trace*, _trace);

extern "C" {
#include "conv-trace.h"
}

#ifndef CMK_OPTIMIZE
#  define _TRACE_ONLY(code) do{if(CpvAccess(traceOn)){ code; }} while(0)
#else
#  define _TRACE_ONLY(code) /*empty*/
#endif

#define _TRACE_USER_EVENT(x) _TRACE_ONLY(CpvAccess(_trace)->userEvent(x))
#define _TRACE_CREATION_1(env) _TRACE_ONLY(CpvAccess(_trace)->creation(env))
#define _TRACE_CREATION_N(env, num) _TRACE_ONLY(CpvAccess(_trace)->creation(env, num))
#define _TRACE_BEGIN_EXECUTE(env) _TRACE_ONLY(CpvAccess(_trace)->beginExecute(env))
#define _TRACE_BEGIN_EXECUTE_DETAILED(evt,typ,ep,src) \
	_TRACE_ONLY(CpvAccess(_trace)->beginExecute(evt,typ,ep,src))
#define _TRACE_END_EXECUTE() _TRACE_ONLY(CpvAccess(_trace)->endExecute())
#define _TRACE_BEGIN_IDLE() _TRACE_ONLY(CpvAccess(_trace)->beginIdle())
#define _TRACE_END_IDLE() _TRACE_ONLY(CpvAccess(_trace)->endIdle())
#define _TRACE_BEGIN_PACK() _TRACE_ONLY(CpvAccess(_trace)->beginPack())
#define _TRACE_END_PACK() _TRACE_ONLY(CpvAccess(_trace)->endPack())
#define _TRACE_BEGIN_UNPACK() _TRACE_ONLY(CpvAccess(_trace)->beginUnpack())
#define _TRACE_END_UNPACK() _TRACE_ONLY(CpvAccess(_trace)->endUnpack())
#define _TRACE_BEGIN_CHARMINIT() _TRACE_ONLY(CpvAccess(_trace)->beginCharmInit())
#define _TRACE_END_CHARMINIT() _TRACE_ONLY(CpvAccess(_trace)->endCharmInit())
#define _TRACE_BEGIN_COMPUTATION() _TRACE_ONLY(CpvAccess(_trace)->beginComputation())
#define _TRACE_END_COMPUTATION() _TRACE_ONLY(CpvAccess(_trace)->endComputation())
#define _TRACE_ENQUEUE(env) _TRACE_ONLY(CpvAccess(_trace)->enqueue(env))
#define _TRACE_DEQUEUE(env) _TRACE_ONLY(CpvAccess(_trace)->dequeue(env))

#endif


