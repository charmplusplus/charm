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

#endif
