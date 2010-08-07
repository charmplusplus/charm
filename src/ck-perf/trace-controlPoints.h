#ifndef _VERBOSE_H
#define _VERBOSE_H

#include <stdio.h>
#include <errno.h>

#include "trace.h"
#include "envelope.h"
#include "register.h"
#include "trace-common.h"

/**
 *   \addtogroup ControlPointFramework
 *   @{
 */


/**
 *    An instrumentation module making use of
 *    the tracing framework hooks provided in Charm++. It is used 
 *    by the control point framework to monitor idle time.
 *
 *  Only the more common hooks are listened to in this module.
 */
class TraceControlPoints : public Trace {
 private:

  double lastBeginExecuteTime;
  int lastbeginMessageSize;

  /** The start of the idle region */
  double lastBeginIdle;
  
  /** Amount of time spent so far in untraced regions */
  double totalUntracedTime;

  /** When tracing was suspended (0 if not currently suspended) */
  double whenStoppedTracing;

  /** The amount of time spent executing entry methods since we last reset the counters */
  double totalEntryMethodTime;

  /** The amount of time spent idle since we last reset the counters */
  double totalIdleTime;

  /** The highest seen memory usage  since we last reset the counters */
  double memUsage;

  /** The number of entry method invocations since we last reset the counters */
  long totalEntryMethodInvocations;

    
  /** The time we last rest the counters */
  double lastResetTime;

 public: 
  int b1, b2, b3;
  long b2mlen;
  long b3mlen;

  // In some programs like Changa, entry methods may be nested, and hence we only want to consider the outermost one
  int nesting_level;


 public:
  TraceControlPoints(char **argv);
  
  //begin/end tracing
  void traceBegin(void);
  void traceEnd(void);


  // a user event has just occured
  void userEvent(int eventID);
  // a pair of begin/end user event has just occured
  void userBracketEvent(int eventID, double bt, double et);
  
  // "creation" of message(s) - message Sends
  void creation(envelope *, int epIdx, int num=1);
  void creationMulticast(envelope *, int epIdx, int num=1, int *pelist=NULL);
  void creationDone(int num=1);
  
  void messageRecv(char *env, int pe);
  
  // **************************************************************
  // begin/end execution of a Charm++ entry point
  // NOTE: begin/endPack and begin/endUnpack can be called in between
  //       a beginExecute and its corresponding endExecute.
  void beginExecute(envelope *);
  void beginExecute(CmiObjId *tid);
  void beginExecute(
		    int event,   // event type defined in trace-common.h
		    int msgType, // message type
		    int ep,      // Charm++ entry point id
		    int srcPe,   // Which PE originated the call
		    int ml,      // message size
		    CmiObjId* idx);    // index
  void endExecute(void);
  
  // begin/end idle time for this pe
  void beginIdle(double curWallTime);
  void endIdle(double curWallTime);
  
  // begin/end of execution
  void beginComputation(void);
  void endComputation(void);
  
  /* Memory tracing */
  void malloc(void *where, int size, void **stack, int stackSize);
  void free(void *where, int size);
  
  // do any clean-up necessary for tracing
  void traceClose();


  // ==================================================================
  // The following methods are not required for a tracing module

  /** reset the idle time and entry method execution time accumulators */
  void resetTimings();

  /** Reset the idle, overhead, and memory measurements */
  void resetAll();

  /** Fraction of the time spent idle since resetting the counters */
  double idleRatio(){
    double t = CmiWallTimer() - lastResetTime;
    return (totalIdleTime) /  (t-untracedTime());
  }

  double untracedTime(){
    if(whenStoppedTracing == 0){
      return totalUntracedTime;     
    } else {
      return totalUntracedTime + (CmiWallTimer()-whenStoppedTracing);
    }

  }

  /** Fraction of time spent as overhead since resetting the counters */
  double overheadRatio(){
    double t = CmiWallTimer() - lastResetTime;
    return (t - totalIdleTime - totalEntryMethodTime) / (t-untracedTime());
  }

  /** Highest memory usage (in MB) value we've seen since resetting the counters */
  double memoryUsageMB(){
    return ((double)memUsage) / 1024.0 / 1024.0;
  }

  /** Determine the average grain size since last reset of counters */
  double grainSize(){
    return (double)totalEntryMethodTime / totalEntryMethodInvocations;
  }

  double bytesPerEntry() {
    return (double)(b2mlen + b3mlen) / (double)(b2+b3);
  }


};


TraceControlPoints *localControlPointTracingInstance();


/*! @} */
#endif

