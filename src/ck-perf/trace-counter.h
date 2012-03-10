/**
 * \addtogroup CkPerf
*/
/*@{*/

#ifndef __trace_counter_h__
#define __trace_counter_h__

#include <stdio.h>
#include <errno.h>
#include "trace.h"
#include "trace-common.h"
#include "conv-config.h"

#define MAX_ENTRIES 500

//*******************************************************
//* TIME IS ALWAYS IN SECONDS UNTIL IT IS PRINTED OUT,  *
//* IN WHICH CASE IT IS CONVERTED TO (us)               *
//*******************************************************

//! track statistics for all entry points
class StatTable {
  public:
    StatTable();
    ~StatTable();
    void init(int argc);
    //! one entry is called for 'time' seconds, value is counter reading 
    void setEp(int epidx, int stat, long long value, double time);
    //! write three lines for each stat:
    //!   1. number of calls for each entry
    //!   2. average count for each entry
    //!   3. total time in us spent for each entry
    void write(FILE* fp);
    void clear();
    int numStats() { return numStats_; }
    //! do a reduction across processors to calculate the total count for
    //! each count, and if the count has flops, etc, then calc the 
    //! the flops/s, etc...
    void doReduction(int phase, double idleTime);

  private:
    //! struct to maintain statistics
    struct Statistics {
      char*  name;                    // name of stat being tracked
      char*  desc;                    // description of stat being tracked
      unsigned int numCalled  [MAX_ENTRIES];  // total number times called
      double       avgCount   [MAX_ENTRIES];  // track average of value
      double       stdDevCount[MAX_ENTRIES];  // track stddev of value
      double       totTime    [MAX_ENTRIES];  // total time assoc with counter
      long long    maxCount   [MAX_ENTRIES];  // maximum count among all times
      long long    minCount   [MAX_ENTRIES];  // minimum count among all times

      Statistics(): name(NULL) { }
    };

    Statistics* stats_;             // track stats for each entry point
    int         numStats_;          // size of statistics being tracked
};

//! counter log pool this implements functions for TraceCounter but
//! that needed to be performed on a node-level
class CountLogPool {
  public:
    CountLogPool();
    ~CountLogPool() { }
    // if phase is -1 and always has been, write normal filename
    // if phase not -1, but has been higher before, write filename + "phaseX"
    void write(int phase=-1) ;
    // if phase is -1 and always has been, write normal filename
    // if phase not -1, but has been higher before, write filename + "phaseX"
    void writeSts(int phase=-1);
    FILE* openFile(int phase=-1);
    void setEp(int epidx, 
	       int index1, long long count1, 
	       int index2, long long count2, 
	       double time);
    void clearEps() { stats_.clear(); }
    void init(int argc) { stats_.init(argc); }
    void doReduction(int phase, double idleTime) { 
      stats_.doReduction(phase, idleTime); 
    }

  private:
    StatTable stats_;       // store stats per counter
    int       lastPhase_;   // keep track of last phase for closing behavior
};

//! For each processor, TraceCounter calculates mean, stdev, etc of 
//! CPU performance counters for each entry point.
class TraceCounter : public Trace {
  public:
    TraceCounter();
    ~TraceCounter();
    //! process command line arguments!
    void traceInit(char **argv);
    //! turn trace on/off, note that charm will automatically call traceBegin()
    //! at the beginning of every run unless the command line option "+traceoff"
    //! is specified
    void traceBegin();
    void traceEnd();
    //! registers user event trace module returns int identifier 
    int traceRegisterUserEvent(const char* userEvent) { 
      // CmiPrintf("%d/%d traceRegisterUserEvent(%s)\n", 
      // CkMyPe(), CkNumPes(), userEvent);
      return 0;
    }
    //! a user event has just occured
    void userEvent(int e) { 
      // CmiPrintf("%d/%d userEvent %d\n", CkMyPe(), CkNumPes(), e); 
    }
    //! creation of message(s)
    void creation(envelope *e, int epIdx, int num=1) { }
    //! ???
    void messageRecv(char *env, int pe) { }
    //! begin/end execution of a Charm++ entry point
    //! NOTE: begin/endPack and begin/endUnpack can be called in between
    //!       a beginExecute and its corresponding endExecute.
    void beginExecute(envelope *e);
    void beginExecute(
      int event,   //! event type defined in trace-common.h
      int msgType, //! message type
      int ep,      //! Charm++ entry point (will correspond to sts file) 
      int srcPe,   //! Which PE originated the call
      int ml=0,   //! message size
      CmiObjId *idx=0);   //! array idx
    void endExecute();
    //! begin/end idle time for this pe
    void beginIdle(double curWallTime);
    void endIdle(double curWallTime);
    //! begin/end the process of packing a message (to send)
    void beginPack();
    void endPack();
    //! begin/end the process of unpacking a message (can occur before calling
    //! a entry point or during an entry point when 
    void beginUnpack();
    void endUnpack();
    //! ???
    void enqueue(envelope *e) { }
    void dequeue(envelope *e) { }
    //! begin/end of execution
    void beginComputation();
    void endComputation();
    //! clear all data collected for entry points
    void traceClearEps();
    //! write the summary sts file for this trace
    void traceWriteSts();
    //! do any clean-up necessary for tracing
    void traceClose();

    //! CounterArg is a linked list of strings that allows
    //! processing of command line args
    struct CounterArg {
      int         code;
      char*       arg;
      char*       desc;
      CounterArg* next;
      int         index;  // index into statTable

      CounterArg(): code(-1), arg(NULL), desc(NULL), next(NULL), index(-1) { }
      CounterArg(int c, char* a, char* d):
        code(c), arg(a), desc(d), next(NULL), index(-1) { }
      void setValues(int _code, char* _arg, char* _desc) {
        code = _code;  arg = _arg;  desc = _desc;
      }
    };

  private:
    enum TC_Status { IDLE, WORKING };
    int cancel_beginIdle, cancel_endIdle;
    
    // command line processing
    CounterArg* firstArg_;      // pointer to start of linked list of args
    CounterArg* lastArg_;       // pointer to end of linked list of args
    int         argStrSize_;    // size of max arg string (formatted output)
    CounterArg* commandLine_;   // list of command line args
    int         commandLineSz_; // size of commande line args array
    CounterArg* counter1_;      // point to current counter, circle linked list
    CounterArg* counter2_;      // point to current counter, circle linked list
    int         counter1Sz_;    // size of cycle
    int         counter2Sz_;    // size of cycle
    
    // result of different command line opts
    bool        overview_;      // if true, just measure between phases
    bool        switchRandom_;  // if true, switch counters randomly
    bool        switchByPhase_; // if true, switch counters only at phases
    bool        noLog_;         // if true, don't write a log file
    bool        writeByPhase_;  // if true, write out a log file every phase

    // store between start/stop of counter read
    int         execEP_;        // id currently executing entry point
    double      startEP_;       // start time of currently executing ep
    double      startIdle_;     // start time of currently executing idle
    int         genStart_;      // track value of start_counters

    // store state
    double      idleTime_;        // total idle time
    int         phase_;           // current phase
    int         reductionPhase_;  // for reduction output
    bool        traceOn_;         // true if trace is turned on
    TC_Status   status_;          // to prevent errors
    bool        dirty_;           // true if endExecute called 

    //! start/stop the overall counting ov eps (don't write to logCount, 
    //! just print to screen
    void beginOverview();
    void endOverview();
    //! switch counters by whatever switching strategy 
    void switchCounters();
    //! add the argument parameters to the linked list of args choices
    void registerArg(CounterArg* arg);
    //! see if the arg (str or code) matches any in the linked list of choices
    //! and sets arg->code to the SGI code
    //! return true if arg matches, false otherwise
    bool matchArg(CounterArg* arg);
    //! print out usage argument
    void usage();
    //! print out all arguments in the linked-list of choices
    void printHelp();
};

#endif  // __trace_counter_h__

/*@}*/


