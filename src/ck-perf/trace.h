
#ifndef _TRACE_H
#define _TRACE_H

/*
 *   File: trace.h -- header file defines the super class of all trace modules
 *         written by Gengbin Zheng, gzheng@uiuc.edu 
 */ 

#if CMK_HAS_COUNTER_PAPI
#include <papi.h>
#endif

class envelope;

/**
 \defgroup CkPerf  Charm++ Trace Module
*/
/*@{*/

// An additional interface for summary data
extern "C" void traceClearEps();

// **CW** may not be necessary
extern "C" void traceEnableCCS();

extern double CmiTraceTimer();

extern int _sdagMsg, _sdagChare, _sdagEP;

/* CW Support for Thread Listener interface */
extern "C" void traceAddThreadListeners(CthThread tid, envelope *e);

// trace_in_charm means only do trace for Charm++ level, skip converse tracing
// Cpv traceOnPe controls if charm pe will generate trace logs (a default value)
// while traceOnPe flag in each trace module can also control independently if 
// tracing is wanted for each module
CkpvExtern(int, traceOnPe);

// A hack. We need to somehow tell the pup framework what size
// long_long is wrt PAPI.
#if CMK_HAS_COUNTER_PAPI
typedef long_long LONG_LONG_PAPI;
#else
typedef CMK_TYPEDEF_INT8 LONG_LONG_PAPI;
#endif

// Base class of all tracing strategies.
// 
class Trace {

protected:
    int _traceOn;
  
  public:
    Trace(): _traceOn(0) {}
    virtual void setTraceOnPE(int flag) { _traceOn = flag; }
    virtual inline int traceOnPE() { return _traceOn; }
    // turn trace on/off, note that charm will automatically call traceBegin()
    // at the beginning of every run unless the command line option "+traceoff"
    // is specified
    virtual void traceBegin() {}
    virtual void traceEnd() {}

    // for tracing comm thread only
    virtual void traceBeginComm() {}
    virtual void traceEndComm() {}
    virtual void traceBeginOnCommThread() {}   
    virtual void traceEndOnCommThread() {}
    virtual void traceCommSetMsgID(char *) {}
    virtual void traceGetMsgID(char *msg, int *pe, int *event) {
      (void)msg; (void)pe; (void)event;
    }
    virtual void traceSetMsgID(char *msg, int pe, int event) {
       (void)msg; (void)pe; (void)event;
     }

     // registers user event trace module returns int identifier
     virtual int traceRegisterUserEvent(const char* eventName, int e) {
       (void)eventName; (void)e;
       return 0;
     }
     // a user event has just occured
     virtual void userEvent(int eventID) { (void)eventID; }
     // a pair of begin/end user event has just occured
     virtual void userBracketEvent(int eventID, double bt, double et) {
       (void)eventID; (void)bt; (void)et;
     }

    //Register user stat trace module returns int identifier
    virtual int traceRegisterUserStat(const char* evt, int e) {
      return 0;
    }

    //update a user Stat with option to set user specified time
    virtual void updateStatPair(int e, double stat,double time) {}
    virtual void updateStat(int e, double stat) {}

     //interact with application
     virtual void beginAppWork() {}
     virtual void endAppWork() {}

     // a user supplied integer value(likely a timestep)
     virtual void userSuppliedData(int e) { (void)e; }

     // a user supplied integer value(likely a timestep)
     virtual void userSuppliedNote(const char *note) { (void)note; }

     virtual void userSuppliedBracketedNote(const char *note, int eventID, double bt, double et) {
       (void)note; (void)eventID; (void)bt; (void)et;
     }

     // the current memory usage as a double
     virtual void memoryUsage(double currentMemUsage) { (void)currentMemUsage; }

     // creation of message(s)
     virtual void creation(envelope *, int epIdx, int num=1) {
       (void)epIdx; (void)num;
     }
     //epIdx is extracted from the envelope, num is always 1
     virtual void creation(char *) {}
     virtual void creationMulticast(envelope *, int epIdx, int num=1,
                      int *pelist=NULL) {
       (void)epIdx; (void)num; (void)pelist;
     }
     virtual void creationDone(int num=1) { (void)num; }
     // ???
     virtual void messageRecv(char *env, int pe) { (void)env; (void)pe; }
     virtual void beginSDAGBlock(
       int event,   // event type defined in trace-common.h
       int msgType, // message type
       int ep,      // Charm++ entry point (will correspond to sts file)
       int srcPe,   // Which PE originated the call
       int ml,      // message size
       CmiObjId* idx)    // index
     {
       (void)event; (void)msgType; (void)ep; (void)srcPe; (void)ml; (void)idx;
     }
     virtual void endSDAGBlock(void) {}
     // **************************************************************
     // begin/end execution of a Charm++ entry point
     // NOTE: begin/endPack and begin/endUnpack can be called in between
     //       a beginExecute and its corresponding endExecute.
     virtual void beginExecute(envelope *, void *) {}
     virtual void beginExecute(char *) {}
     virtual void beginExecute(CmiObjId *tid) { (void)tid; }
     virtual void beginExecute(
       int event,   // event type defined in trace-common.h
       int msgType, // message type
       int ep,      // Charm++ entry point (will correspond to sts file)
       int srcPe,   // Which PE originated the call
       int ml,      // message size
       CmiObjId* idx,    // index
       void* obj)
    {
      (void)event; (void)msgType; (void)ep; (void)srcPe;
      (void)ml; (void)idx; (void)obj;
    }
    virtual void changeLastEntryTimestamp(double ts) { (void)ts; }
    virtual void endExecute(void) {}
    virtual void endExecute(char *) {}
    // begin/end idle time for this pe
    virtual void beginIdle(double curWallTime) { (void)curWallTime; }
    virtual void endIdle(double curWallTime) { (void)curWallTime; }
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
    // demarkation of a phase boundary
    virtual void endPhase() {}
    // clear all data collected for entry points
    virtual void traceClearEps() {}
    // enable CCS operations if supported on the trace module
    virtual void traceEnableCCS() {}
    // write the summary sts file for this trace
    virtual void traceWriteSts() {}
    // do any clean-up necessary for tracing
    virtual void traceClose() {}
    // flush log buffer immediately
    virtual void traceFlushLog() {}

    //for tracing function calls
    virtual void regFunc(const char *name, int &idx, int idxSpecifiedByUser=0) {
      (void)name; (void)idx; (void)idxSpecifiedByUser;
    }
    virtual void beginFunc(const char *name,const char *file,int line) {
      (void)name; (void)file; (void)line;
    }
    virtual void beginFunc(int idx,const char *file,int line) {
      (void)idx; (void)file; (void)line;
    }
    virtual void endFunc(const char *name) { (void)name; }
    virtual void endFunc(int idx) { (void)idx; }

    /* Memory tracing */
    virtual void malloc(void *where, int size, void **stack, int stackSize) {
      (void)where; (void)size; (void)stack; (void)stackSize;
    }
    virtual void free(void *where, int size) {
      (void)where; (void)size;
    }

    /* for implementing thread listeners */
    virtual void traceAddThreadListeners(CthThread tid, envelope *e) {
      (void)tid; (void)e;
    }

    virtual ~Trace() {} /* for whining compilers */
};

#define ALLDO(x) for (int i=0; i<length(); i++) if (traces[i]->traceOnPE()) traces[i]->x
#define ALLREVERSEDO(x) for (int i=length()-1; i>=0; i--) if (traces[i]->traceOnPE()) traces[i]->x

/// Array of Traces modules,  every event raised will go through every Trace module.
class TraceArray {
private:
  CkVec<Trace *>  traces;
  int n;
  int cancel_beginIdle, cancel_endIdle;
public:
    TraceArray(): n(0) {}
    inline void addTrace(Trace *tr) { traces.push_back(tr); n++;}
    inline void setTraceOnPE(int flag) { for (int i=0; i<length(); i++) traces[i]->setTraceOnPE(flag); }
    // to allow traceCLose() to be called multiple times, remove trace module
    // from the array in each individual trace, and clean up (clearTrace)
    // after the loop.
    inline void removeTrace(Trace *tr) {    // remove a Trace from TraceArray
        int i;
        for (i=0; i<n; i++) if (tr == traces[i]) break;
        CmiAssert(i<n);
        traces[i] = NULL;
    }
    inline void clearTrace() {    // remove holes in TraceArray
	int len = traces.length();
	int removed = 0;
        for (int i=0; i<len; i++) {
          if (traces[i-removed] == NULL) { traces.remove(i-removed); removed++; }
        }
        n -= removed;
    }
    inline int length() const { return n; }

    inline void userEvent(int e) { ALLDO(userEvent(e));}
    inline void userBracketEvent(int e,double bt, double et) {ALLDO(userBracketEvent(e,bt,et));}
    
    inline void beginAppWork() { ALLDO(beginAppWork());}
    inline void endAppWork() { ALLDO(endAppWork());}

	inline void userSuppliedData(int d) { ALLDO(userSuppliedData(d));}

	inline void userSuppliedNote(const char *note) { ALLDO(userSuppliedNote(note));}
	inline void userSuppliedBracketedNote(const char *note, int eventID, double bt, double et) {ALLDO(userSuppliedBracketedNote(note, eventID, bt, et));}


	inline void memoryUsage(double memUsage) { ALLDO(memoryUsage(memUsage));}


    /* Creation needs to access _entryTable, so moved into trace-common.C */
    void creation(envelope *env, int ep, int num=1);
    inline void creation(char *msg){
        /* The check for whether the ep got traced is moved into each elem's
         * creation call as the ep could not be extracted here
         */
        ALLDO(creation(msg));
    }
    void creationMulticast(envelope *env, int ep, int num=1, int *pelist=NULL);
    
    inline void creationDone(int num=1) { ALLDO(creationDone(num)); }
    inline void beginSDAGBlock(int event,int msgType,int ep,int srcPe, int mlen,CmiObjId *idx=NULL) {ALLDO(beginSDAGBlock(event, msgType, ep, srcPe, mlen,idx));}
    inline void endSDAGBlock(void) {ALLREVERSEDO(endExecute());}
    inline void beginExecute(envelope *env, void *obj) {ALLDO(beginExecute(env, obj));}
    inline void beginExecute(char *msg) {ALLDO(beginExecute(msg));}
    inline void beginExecute(CmiObjId *tid) {ALLDO(beginExecute(tid));}
    inline void beginExecute(int event,int msgType,int ep,int srcPe, int mlen,CmiObjId *idx=NULL, void *obj=NULL) {ALLDO(beginExecute(event, msgType, ep, srcPe, mlen,idx, obj));}
    inline void endExecute(void) {ALLREVERSEDO(endExecute());}
    inline void endExecute(char *msg) {ALLREVERSEDO(endExecute(msg));}
    inline void changeLastEntryTimestamp(double ts) {ALLDO(changeLastEntryTimestamp(ts));}
    inline void messageRecv(char *env, int pe) {ALLDO(messageRecv(env, pe));}
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
	    if (traces[i]->traceOnPE() == 0) {
	      continue;
	    }
	    int e = traces[i]->traceRegisterUserEvent(x, evt);
	    if (e) eno = e;
          }
	  return eno;
    } 

//User stat functions for TraceArray
    inline int traceRegisterUserStat(const char*x, int evt) {
	  int eno = 0;
	  for (int i=0; i<length(); i++) {
	    if (traces[i]->traceOnPE() == 0) {
	      continue;
	    }
	    int e = traces[i]->traceRegisterUserStat(x, evt);
	    if (e) eno = e;
          }
	  return eno;
    }
    inline void updateStatPair(int e,double stat,double time) {ALLDO(updateStatPair(e,stat,time));}
    inline void updateStat(int e,double stat) {ALLDO(updateStat(e,stat));}

    inline void traceClearEps() {ALLDO(traceClearEps());}
    inline void traceEnableCCS() {ALLDO(traceEnableCCS());}
    inline void traceWriteSts() {ALLDO(traceWriteSts());}
    inline void traceClose() {ALLDO(traceClose()); clearTrace();}
    inline void traceFlushLog() {ALLDO(traceFlushLog());}
    
    // Tracing module registers *itself* for begin/end idle callbacks:
    inline void beginIdle(double curWallTime) {ALLDO(beginIdle(curWallTime));}
    inline void endIdle(double curWallTime) {ALLDO(endIdle(curWallTime));}
    void traceBegin();    
    void traceEnd();

    // for tracing comm thread only
    void traceBeginComm();
    void traceEndComm();
    void traceBeginOnCommThread();
    void traceEndOnCommThread();
    void traceCommSetMsgID(char *msg)  { ALLDO(traceCommSetMsgID(msg)); }
    void traceGetMsgID(char *msg, int *pe, int *event) { ALLDO(traceGetMsgID(msg, pe, event)); }
    void traceSetMsgID(char *msg, int pe, int event) { ALLDO(traceSetMsgID(msg, pe, event)); }
    /*Calls for tracing function begins and ends*/
    inline void regFunc(const char *name, int &idx, int idxSpecifiedByUser=0){ ALLDO(regFunc(name, idx, idxSpecifiedByUser)); }
    inline void beginFunc(const char *name,const char *file,int line){ ALLDO(beginFunc(name,file,line)); };
    inline void beginFunc(int idx,const char *file,int line){ ALLDO(beginFunc(idx,file,line)); };
    inline void endFunc(const char *name){ ALLDO(endFunc(name)); }
    inline void endFunc(int idx){ ALLDO(endFunc(idx)); }

    /* Phase Demarcation */
    inline void endPhase() { ALLDO(endPhase()); }

    /* Memory tracing */
    inline void malloc(void *where, int size, void **stack, int stackSize){ ALLDO(malloc(where,size,stack,stackSize)); }
    inline void free(void *where, int size){ ALLDO(free(where, size)); }

    /* calls for thread listener registration for each trace module */
    inline void traceAddThreadListeners(CthThread tid, envelope *e) {
      ALLDO(traceAddThreadListeners(tid, e));
    }
};

CkpvExtern(TraceArray*, _traces);

#if CMK_TRACE_ENABLED
#if CMK_BIGSIM_CHARM
extern void    resetVTime();
#  define _TRACE_ONLY(code) do{ BgGetTime(); if(CpvAccess(traceOn) && CkpvAccess(_traces)->length()>0) { code; }  resetVTime(); } while(0)
#else
#  define _TRACE_ONLY(code) do{if(CpvAccess(traceOn) && CkpvAccess(_traces)->length()>0) { code; }} while(0)
#endif
#  define _TRACE_ALWAYS(code) do{ code; } while(0)
#else
#  define _TRACE_ONLY(code) /*empty*/
#  define _TRACE_ALWAYS(code) /*empty*/
#endif

extern "C" {
#include "conv-trace.h"
}

#define _TRACE_USER_EVENT(x) _TRACE_ONLY(CkpvAccess(_traces)->userEvent(x))
#define _TRACE_USER_EVENT_BRACKET(x,bt,et) _TRACE_ONLY(CkpvAccess(_traces)->userBracketEvent(x,bt,et))
#define _TRACE_BEGIN_APPWORK() _TRACE_ONLY(CkpvAccess(_traces)->beginAppWork())
#define _TRACE_END_APPWORK() _TRACE_ONLY(CkpvAccess(_traces)->endAppWork())
#define _TRACE_CREATION_1(env) _TRACE_ONLY(CkpvAccess(_traces)->creation(env,env->getEpIdx()))
#define _TRACE_CREATION_DETAILED(env,ep) _TRACE_ONLY(CkpvAccess(_traces)->creation(env,ep))
#define _TRACE_CREATION_N(env, num) _TRACE_ONLY(CkpvAccess(_traces)->creation(env, env->getEpIdx(), num))
#define _TRACE_CREATION_MULTICAST(env, num, pelist) _TRACE_ONLY(CkpvAccess(_traces)->creationMulticast(env, env->getEpIdx(), num, pelist))
#define _TRACE_CREATION_DONE(num) _TRACE_ONLY(CkpvAccess(_traces)->creationDone(num))
#define _TRACE_BEGIN_SDAG(env) _TRACE_ONLY(CkpvAccess(_traces)->beginSDAGBlock(env))
#define _TRACE_END_SDAG(env) _TRACE_ONLY(CkpvAccess(_traces)->endSDAGBlock(env))
#define _TRACE_BEGIN_EXECUTE(env, obj) _TRACE_ONLY(CkpvAccess(_traces)->beginExecute(env, obj))
#define _TRACE_BEGIN_EXECUTE_DETAILED(evt,typ,ep,src,mlen,idx, obj) _TRACE_ONLY(CkpvAccess(_traces)->beginExecute(evt,typ,ep,src,mlen,idx, obj))
#define _TRACE_END_EXECUTE() _TRACE_ONLY(CkpvAccess(_traces)->endExecute())
#define _TRACE_MESSAGE_RECV(env, pe) _TRACE_ONLY(CkpvAccess(_traces)->messageRecv(env, pe))
#define _TRACE_BEGIN_PACK() _TRACE_ONLY(CkpvAccess(_traces)->beginPack())
#define _TRACE_END_PACK() _TRACE_ONLY(CkpvAccess(_traces)->endPack())
#define _TRACE_BEGIN_UNPACK() _TRACE_ONLY(CkpvAccess(_traces)->beginUnpack())
#define _TRACE_END_UNPACK() _TRACE_ONLY(CkpvAccess(_traces)->endUnpack())
#define _TRACE_BEGIN_COMPUTATION() _TRACE_ALWAYS(CkpvAccess(_traces)->beginComputation())
#define _TRACE_END_COMPUTATION() _TRACE_ALWAYS(CkpvAccess(_traces)->endComputation())
#define _TRACE_ENQUEUE(env) _TRACE_ONLY(CkpvAccess(_traces)->enqueue(env))
#define _TRACE_DEQUEUE(env) _TRACE_ONLY(CkpvAccess(_traces)->dequeue(env))

#define _TRACE_END_PHASE() _TRACE_ONLY(CkpvAccess(_traces)->endPhase())

/* Memory tracing */
#define _TRACE_MALLOC(where, size, stack, stackSize) _TRACE_ONLY(CkpvAccess(_traces)->malloc(where,size,stack,stackSize))
#define _TRACE_FREE(where, size) _TRACE_ONLY(CkpvAccess(_traces)->free(where, size))

#include "trace-bluegene.h"

#endif


/*@}*/
