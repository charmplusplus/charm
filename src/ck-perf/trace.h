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
#define _TRACE_USER_EVENT(x) \
    do { \
      if(CpvAccess(traceOn)) \
        { CpvAccess(_trace)->userEvent(x); } \
    } while(0)
#define _TRACE_CREATION_1(env) \
    do { \
      if(CpvAccess(traceOn)) \
        { CpvAccess(_trace)->creation(env); } \
    } while(0)
#define _TRACE_CREATION_N(env, num) \
    do { \
      if(CpvAccess(traceOn)) \
      { CpvAccess(_trace)->creation(env, num); } \
    } while(0)
#define _TRACE_BEGIN_EXECUTE(env) \
    do { \
      if(CpvAccess(traceOn)) \
      { CpvAccess(_trace)->beginExecute(env); } \
    } while(0)
#define _TRACE_END_EXECUTE() \
    do { \
      if(CpvAccess(traceOn)) \
        { CpvAccess(_trace)->endExecute(); } \
    } while(0)
#define _TRACE_BEGIN_IDLE() \
    do { \
      if(CpvAccess(traceOn)) \
        { CpvAccess(_trace)->beginIdle(); } \
    } while(0)
#define _TRACE_END_IDLE() \
    do { \
      if(CpvAccess(traceOn)) \
      { CpvAccess(_trace)->endIdle(); } \
    } while(0)
#define _TRACE_BEGIN_PACK() \
    do { \
      if(CpvAccess(traceOn)) \
        { CpvAccess(_trace)->beginPack(); } \
    } while(0)
#define _TRACE_END_PACK() \
    do { \
      if(CpvAccess(traceOn)) \
        { CpvAccess(_trace)->endPack(); } \
    } while(0)
#define _TRACE_BEGIN_UNPACK() \
    do { \
      if(CpvAccess(traceOn)) \
        { CpvAccess(_trace)->beginUnpack(); } \
    } while(0)
#define _TRACE_END_UNPACK() \
    do { \
      if(CpvAccess(traceOn)) \
        { CpvAccess(_trace)->endUnpack(); } \
    } while(0)
#define _TRACE_BEGIN_CHARMINIT() \
    do { \
      if(CpvAccess(traceOn)) \
        { CpvAccess(_trace)->beginCharmInit(); } \
    } while(0)
#define _TRACE_END_CHARMINIT() \
    do { \
      if(CpvAccess(traceOn)) \
        { CpvAccess(_trace)->endCharmInit(); } \
    } while(0)
#define _TRACE_BEGIN_COMPUTATION() \
    do { \
      if(CpvAccess(traceOn)) \
        { CpvAccess(_trace)->beginComputation(); } \
    } while(0)
#define _TRACE_END_COMPUTATION() \
    do { \
      if(CpvAccess(traceOn)) \
        { CpvAccess(_trace)->endComputation(); } \
    } while(0)
#define _TRACE_ENQUEUE(env) \
    do { \
      if(CpvAccess(traceOn)) \
        { CpvAccess(_trace)->enqueue(env); } \
    } while(0)
#define _TRACE_DEQUEUE(env) \
    do { \
      if(CpvAccess(traceOn)) \
        { CpvAccess(_trace)->dequeue(env); } \
    } while(0)
#else
#define _TRACE_USER_EVENT(x) do{}while(0)
#define _TRACE_CREATION_1(env) do{}while(0)
#define _TRACE_CREATION_N(env, num) do{}while(0)
#define _TRACE_BEGIN_EXECUTE(env) do{}while(0)
#define _TRACE_END_EXECUTE() do{}while(0)
#define _TRACE_BEGIN_IDLE() do{}while(0)
#define _TRACE_END_IDLE() do{}while(0)
#define _TRACE_BEGIN_PACK() do{}while(0)
#define _TRACE_END_PACK() do{}while(0)
#define _TRACE_BEGIN_UNPACK() do{}while(0)
#define _TRACE_END_UNPACK() do{}while(0)
#define _TRACE_BEGIN_CHARMINIT() do{}while(0)
#define _TRACE_END_CHARMINIT() do{}while(0)
#define _TRACE_BEGIN_COMPUTATION() do{}while(0)
#define _TRACE_END_COMPUTATION() do{}while(0)
#define _TRACE_ENQUEUE(env) do{}while(0)
#define _TRACE_DEQUEUE(env) do{}while(0)
#endif

#endif
