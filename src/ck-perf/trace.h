#ifndef _TRACE_H
#define _TRACE_H

#include "converse.h"

class envelope;

// Base class of all tracing strategies.

class Trace {
  public:
    virtual void userEvent(int) {}
    virtual void creation(envelope *, int num=1) {}
    virtual void beginExecute(envelope *) {}
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
#define _TRACE_USER_EVENT(x) if(CpvAccess(traceOn)) \
            { CpvAccess(_trace)->userEvent(x); }
#define _TRACE_CREATION_1(env) if(CpvAccess(traceOn)) \
            { CpvAccess(_trace)->creation(env); }
#define _TRACE_CREATION_N(env, num) if(CpvAccess(traceOn)) \
            { CpvAccess(_trace)->creation(env, num); }
#define _TRACE_BEGIN_EXECUTE(env) if(CpvAccess(traceOn)) \
            { CpvAccess(_trace)->beginExecute(env); }
#define _TRACE_END_EXECUTE() if(CpvAccess(traceOn)) \
            { CpvAccess(_trace)->endExecute(); }
#define _TRACE_BEGIN_IDLE() if(CpvAccess(traceOn)) \
            { CpvAccess(_trace)->beginIdle(); }
#define _TRACE_END_IDLE() if(CpvAccess(traceOn)) \
            { CpvAccess(_trace)->endIdle(); }
#define _TRACE_BEGIN_PACK() if(CpvAccess(traceOn)) \
            { CpvAccess(_trace)->beginPack(); }
#define _TRACE_END_PACK() if(CpvAccess(traceOn)) \
            { CpvAccess(_trace)->endPack(); }
#define _TRACE_BEGIN_UNPACK() if(CpvAccess(traceOn)) \
            { CpvAccess(_trace)->beginUnpack(); }
#define _TRACE_END_UNPACK() if(CpvAccess(traceOn)) \
            { CpvAccess(_trace)->endUnpack(); }
#define _TRACE_BEGIN_CHARMINIT() if(CpvAccess(traceOn)) \
            { CpvAccess(_trace)->beginCharmInit(); }
#define _TRACE_END_CHARMINIT() if(CpvAccess(traceOn)) \
            { CpvAccess(_trace)->endCharmInit(); }
#define _TRACE_BEGIN_COMPUTATION() if(CpvAccess(traceOn)) \
            { CpvAccess(_trace)->beginComputation(); }
#define _TRACE_END_COMPUTATION() if(CpvAccess(traceOn)) \
            { CpvAccess(_trace)->endComputation(); }
#define _TRACE_ENQUEUE(env) if(CpvAccess(traceOn)) \
            { CpvAccess(_trace)->enqueue(env); }
#define _TRACE_DEQUEUE(env) if(CpvAccess(traceOn)) \
            { CpvAccess(_trace)->dequeue(env); }
#else
#define _TRACE_USER_EVENT(x)
#define _TRACE_CREATION_1(env)
#define _TRACE_CREATION_N(env, num)
#define _TRACE_BEGIN_EXECUTE(env)
#define _TRACE_END_EXECUTE()
#define _TRACE_BEGIN_IDLE()
#define _TRACE_END_IDLE()
#define _TRACE_BEGIN_PACK()
#define _TRACE_END_PACK()
#define _TRACE_BEGIN_UNPACK()
#define _TRACE_END_UNPACK()
#define _TRACE_BEGIN_CHARMINIT()
#define _TRACE_END_CHARMINIT()
#define _TRACE_BEGIN_COMPUTATION()
#define _TRACE_END_COMPUTATION()
#define _TRACE_ENQUEUE(env)
#define _TRACE_DEQUEUE(env)
#endif

#endif
