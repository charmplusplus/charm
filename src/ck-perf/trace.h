/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _TRACE_H
#define _TRACE_H

class envelope;

/**
 \defgroup CkPerf  Charm++ Trace Module
*/
/*@{*/

// An additional interface for summary data
extern "C" void traceClearEps();
extern double CmiTraceTimer();

extern int _dummyMsg, _dummyChare, _dummyEP;

// trace_in_charm means only do trace for Charm++ level
// traceOnPe controls if charm pe will generate trace logs
#if CMK_TRACE_IN_CHARM
CkpvExtern(int, traceOnPe);
#  define TRACE_CHARM_PE()  (CkpvAccess(traceOnPe))
#else
          /* skip communication thread */
#  define TRACE_CHARM_PE()  (CkMyRank() != CkMyNodeSize())
#endif

// Base class of all tracing strategies.
// 
class Trace {
  public:
    virtual int traceOnPE() { return TRACE_CHARM_PE(); }
    // turn trace on/off, note that charm will automatically call traceBegin()
    // at the beginning of every run unless the command line option "+traceoff"
    // is specified
    virtual void traceBegin() {}
    virtual void traceEnd() {}
    // registers user event trace module returns int identifier 
    virtual int traceRegisterUserEvent(const char* eventName, int e) { return 0; }
    // a user event has just occured
    virtual void userEvent(int eventID) {}
    // a pair of begin/end user event has just occured
    virtual void userBracketEvent(int eventID, double bt, double et) {}
    // creation of message(s)
    virtual void creation(envelope *, int epIdx, int num=1) {}
    virtual void creationDone(int num=1) {}
    // ???
    virtual void messageRecv(char *env, int pe) {}
    // **************************************************************
    // begin/end execution of a Charm++ entry point
    // NOTE: begin/endPack and begin/endUnpack can be called in between
    //       a beginExecute and its corresponding endExecute.
    virtual void beginExecute(envelope *) {}
    virtual void beginExecute(CmiObjId *tid) {}
    virtual void beginExecute(
      int event,   // event type defined in trace-common.h
      int msgType, // message type
      int ep,      // Charm++ entry point (will correspond to sts file) 
      int srcPe,   // Which PE originated the call
      int ml,      // message size
      CmiObjId* idx)    // index
    { }
    virtual void endExecute(void) {}
    // begin/end idle time for this pe
    virtual void beginIdle(void) {}
    virtual void endIdle(void) {}
    // begin/end the process of packing a message (to send)
    virtual void beginPack(void) {}
    virtual void endPack(void) {}
    // begin/end the process of unpacking a message (can occur before calling
    // a entry point or during an entry point when 
    virtual void beginUnpack(void) {}
    virtual void endUnpack(void) {}
    // ???
    virtual void enqueue(envelope *) {}
    virtual void dequeue(envelope *) {}
    // begin/end of execution
    virtual void beginComputation(void) {}
    virtual void endComputation(void) {}
    // clear all data collected for entry points
    virtual void traceClearEps() {}
    // write the summary sts file for this trace
    virtual void traceWriteSts() {}
    // do any clean-up necessary for tracing
    virtual void traceClose() {}
    // only for Blue Gene
    virtual void tlineEnd(void** parentLogPtr){}
    virtual void bgBeginExec(char* name,void** pLogPtr){}
    virtual void bgEndExec(){}
    virtual void bgPrint(char* str){}
    virtual void getForwardDepForAll(void** logs1,void** logs2, int size ,void* fDepPtr){}
    virtual void userBracketEvent(char* name, double bt, double et, void** parentLogPtr){}
    virtual void userBracketEvent(char* name, double bt, double et, void** parentLogPtr, CkVec<void*> bgLogList){}
};

#define ALLDO(x) for (int i=0; i<length(); i++) if (traces[i]->traceOnPE()) traces[i]->x
#define ALLREVERSEDO(x) for (int i=length()-1; i>=0; i--) if (traces[i]->traceOnPE()) traces[i]->x

/// Array of Traces modules,  every event raised will go through every Trace module.
class TraceArray {
private:
  CkVec<Trace *>  traces;
  int n;
public:
    TraceArray(): n(0) {}
    inline void addTrace(Trace *tr) { traces.push_back(tr); n++;}
    inline void removeTrace(Trace *tr) {    // remove a Trace from TraceArray
        int i;
        for (i=0; i<n; i++) if (tr == traces[i]) break;
        CmiAssert(i<n);
        traces[i] = NULL;
        n--;
    }
    inline void clearTrace() {    // remove void Trace from TraceArray
	int len = traces.length();
	int removed = 0;
        for (int i=0; i<len; i++) {
          if (traces[i] == NULL) { traces.remove(i-removed); removed++; }
        }
    }
    inline const int length() const { return n; }

    inline void userEvent(int e) { ALLDO(userEvent(e));}
    inline void userBracketEvent(int e,double bt, double et) {ALLDO(userBracketEvent(e,bt,et));}
    
    /* Creation needs to access _entryTable, so moved into trace-common.C */
    void creation(envelope *env, int ep, int num=1);
    
    inline void creationDone(int num=1) { ALLDO(creationDone(num)); }
    inline void beginExecute(envelope *env) {ALLDO(beginExecute(env));}
    inline void beginExecute(CmiObjId *tid) {ALLDO(beginExecute(tid));}
    inline void beginExecute(int event,int msgType,int ep,int srcPe, int mlen,CmiObjId *idx=NULL) {ALLDO(beginExecute(event, msgType, ep, srcPe, mlen,idx));}
    inline void endExecute(void) {ALLREVERSEDO(endExecute());}
    inline void messageRecv(char *env, int pe) {ALLDO(messageRecv(env, pe));}
    inline void beginIdle(void) {ALLDO(beginIdle());}
    inline void endIdle(void) {ALLDO(endIdle());}
    inline void beginPack(void) {ALLDO(beginPack());}
    inline void endPack(void) {ALLDO(endPack());}
    inline void beginUnpack(void) {ALLDO(beginUnpack());}
    inline void endUnpack(void) {ALLDO(endUnpack());}
    inline void enqueue(envelope *e) {ALLDO(enqueue(e));}
    inline void dequeue(envelope *e) {ALLDO(dequeue(e));}
    inline void beginComputation(void) {ALLDO(beginComputation());}
    inline void endComputation(void) {ALLDO(endComputation());}
    inline int traceRegisterUserEvent(const char*x, int evt) {
	  int eno = 0;
	  for (int i=0; i<length(); i++) {
	    if (traces[i]->traceOnPE() == 0) continue;
	    int e = traces[i]->traceRegisterUserEvent(x, evt);
	    if (e) eno = e;
          }
	  return eno;
    }  
    inline void traceClearEps() {ALLDO(traceClearEps());}
    inline void traceWriteSts() {ALLDO(traceWriteSts());}
    inline void traceClose() {ALLDO(traceClose()); clearTrace();}
    inline void traceBegin() {ALLDO(traceBegin());}
    inline void traceEnd() {ALLDO(traceEnd());}
    // only for Blue Gene
    inline void tlineEnd(void** pLogPtr){ALLDO(tlineEnd(pLogPtr));}
    inline void bgBeginExec(char* name,void** pLogPtr){ALLDO(bgBeginExec(name,pLogPtr));}
    inline void bgEndExec(){ALLDO(bgEndExec());}
    inline void bgPrint(char* str){ALLDO(bgPrint(str));}
    inline void getForwardDepForAll(void** logs1, void**logs2,int size,void* fDepPtr){ALLDO(getForwardDepForAll(logs1, logs2,size,fDepPtr));}
    inline void userBracketEvent(char* name,double bt, double et, void** pLogPtr) {ALLDO(userBracketEvent(name,bt,et,pLogPtr));}
    inline void userBracketEvent(char* name,double bt, double et, void** pLogPtr, CkVec<void*> bgLogList) {ALLDO(userBracketEvent(name,bt,et,pLogPtr,bgLogList))
;}

};

CkpvExtern(TraceArray*, _traces);

extern "C" {
#include "conv-trace.h"
}

#ifndef CMK_OPTIMIZE
#  define _TRACE_ONLY(code) do{if(CpvAccess(traceOn)){ code; }} while(0)
#else
#  define _TRACE_ONLY(code) /*empty*/
#endif

#define _TRACE_USER_EVENT(x) _TRACE_ONLY(CkpvAccess(_traces)->userEvent(x))
#define _TRACE_USER_EVENT_BRACKET(x,bt,et) _TRACE_ONLY(CkpvAccess(_traces)->userBracketEvent(x,bt,et))
#define _TRACE_CREATION_1(env) _TRACE_ONLY(CkpvAccess(_traces)->creation(env,env->getEpIdx()))
#define _TRACE_CREATION_DETAILED(env,ep) _TRACE_ONLY(CkpvAccess(_traces)->creation(env,ep))
#define _TRACE_CREATION_N(env, num) _TRACE_ONLY(CkpvAccess(_traces)->creation(env,env->getEpIdx(), num))
#define _TRACE_CREATION_DONE(num) _TRACE_ONLY(CkpvAccess(_traces)->creationDone(num))
#define _TRACE_BEGIN_EXECUTE(env) _TRACE_ONLY(CkpvAccess(_traces)->beginExecute(env))
#define _TRACE_BEGIN_EXECUTE_DETAILED(evt,typ,ep,src,mlen,idx) \
	_TRACE_ONLY(CkpvAccess(_traces)->beginExecute(evt,typ,ep,src,mlen,idx))
#define _TRACE_END_EXECUTE() _TRACE_ONLY(CkpvAccess(_traces)->endExecute())
#define _TRACE_MESSAGE_RECV(env, pe) _TRACE_ONLY(CkpvAccess(_traces)->messageRecv(env, pe))
#define _TRACE_BEGIN_IDLE() _TRACE_ONLY(CkpvAccess(_traces)->beginIdle())
#define _TRACE_END_IDLE() _TRACE_ONLY(CkpvAccess(_traces)->endIdle())
#define _TRACE_BEGIN_PACK() _TRACE_ONLY(CkpvAccess(_traces)->beginPack())
#define _TRACE_END_PACK() _TRACE_ONLY(CkpvAccess(_traces)->endPack())
#define _TRACE_BEGIN_UNPACK() _TRACE_ONLY(CkpvAccess(_traces)->beginUnpack())
#define _TRACE_END_UNPACK() _TRACE_ONLY(CkpvAccess(_traces)->endUnpack())
#define _TRACE_BEGIN_COMPUTATION() _TRACE_ONLY(CkpvAccess(_traces)->beginComputation())
#define _TRACE_END_COMPUTATION() _TRACE_ONLY(CkpvAccess(_traces)->endComputation())
#define _TRACE_ENQUEUE(env) _TRACE_ONLY(CkpvAccess(_traces)->enqueue(env))
#define _TRACE_DEQUEUE(env) _TRACE_ONLY(CkpvAccess(_traces)->dequeue(env))

// for Sdag only
// fixme - think of better api for tracing sdag code
#define BgPrint(x)  _TRACE_ONLY(CkpvAccess(_traces)->bgPrint(x))
#define _TRACE_BG_BEGIN_EXECUTE(x,pLogPtr)  _TRACE_ONLY(CkpvAccess(_traces)->bgBeginExec(x,pLogPtr))
#define _TRACE_BG_END_EXECUTE()   _TRACE_ONLY(CkpvAccess(_traces)->bgEndExec())
#define _TRACE_BG_TLINE_END(pLogPtr) _TRACE_ONLY(CkpvAccess(_traces)->tlineEnd(pLogPtr))
#define _TRACE_BG_FORWARD_DEPS(logs1,logs2,size,fDep)  _TRACE_ONLY(CkpvAccess(_traces)->getForwardDepForAll(logs1,logs2, size,fDep))
#define _TRACE_BG_USER_EVENT_BRACKET(x,bt,et,pLogPtr) _TRACE_ONLY(CkpvAccess(_traces)->userBracketEvent(x,bt,et,pLogPtr))
#define _TRACE_BGLIST_USER_EVENT_BRACKET(x,bt,et,pLogPtr,bgLogList) _TRACE_ONLY(CkpvAccess(_traces)->userBracketEvent(x,bt,et,pLogPtr,bgLogList))


#endif


/*@}*/
