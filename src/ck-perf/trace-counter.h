/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkPerf
*/
/*@{*/

#ifdef CMK_ORIGIN2000

#ifndef __trace_counter_h__
#define __trace_counter_h__

#include <stdio.h>
#include <errno.h>
#include "trace.h"
#include "ck.h"
#include "trace-common.h"

#define MAX_ENTRIES 500

// track statistics for all entry points
class StatTable {
  public:
    StatTable();
    ~StatTable();
    // one entry is called for 'time' seconds, value is counter reading
    void setEp(int epidx, int stat, UInt value, double time) {
      CkAssert(epidx<MAX_ENTRIES);
      CkAssert(stat<numStats_);

      int count = stats_[stat].count[epidx];
      stats_[stat].count[epidx]++;
      double avg = stats_[stat].average[epidx];
      stats_[stat].average[epidx] = (avg * count + value) / (count + 1);
      stats_[stat].totTime[epidx] += time;
    }
    /**
       write three lines for each stat:
       1. number of calls for each entry
       2. average count for each entry
       3. total time in us spent for each entry
    */
    void write(FILE* fp) {
      int i, j;
      for (i=0; i<numStats_; i++) {
	// write number of calls for each entry
	fprintf(fp, "[%s] ", stats_[i].name);
	for (j=0; j<_numEntries; j++) { 
	  fprintf(fp, "%d ", stats_[i].count[j]); 
	}
	fprintf(fp, "\n");
	// write average count for each 
	fprintf(fp, "[%s] ", stats_[i].name);
	for (j=0; j<_numEntries; j++) { 
	  fprintf(fp, "%d ", stats_[i].average[j]); 
	}
	fprintf(fp, "\n");
	// write total time in us spent for each entry
	fprintf(fp, "[%s] ", stats_[i].name);
	for (j=0; j<_numEntries; j++) {
	  fprintf(fp, "%ld ", (long)(stats_[i].totTime[j]*1.0e6));
	}
	fprintf(fp, "\n");
      }
    }

  private:
    // struct to maintain statistics
    struct Statistics {
      char*  name;                  // name of stat being tracked
      UInt   count[MAX_ENTRIES];    // total number times called
      double average[MAX_ENTRIES];  // track average of value
      double totTime[MAX_ENTRIES];  // total time associated with this counter
    };

    Statistics* stats_;             // track stats for each entry point
    int         numStats_;          // size of statistics being tracked
};

// counter log pool
class CountLogPool {
  public:
    CountLogPool(char* pgm);
    ~CountLogPool();
    void write(void) ;
    void writeSts(void);
    void setEp(int epidx, double time);

  private:
    FILE*     fp_;
    StatTable stats_;
};

/**
  For each processor, TraceCounter calculates mean, stdev, etc of 
  CPU performance counters for each entry point.
*/
class TraceCounter : public Trace {
  public:
    TraceCounter() { }
    void userEvent(int e) { }
    void creation(envelope *e, int num=1) { }
    void beginExecute(envelope *e);
    void beginExecute(int event, int msgType, int ep, int srcPe, int mlen=0);
    void endExecute(void);
    void beginIdle(void) { }
    void endIdle(void) { }
    void beginPack(void);
    void endPack(void);
    void beginUnpack(void);
    void endUnpack(void);
    void beginCharmInit(void) { }
    void endCharmInit(void) { }
    void enqueue(envelope *e) { }
    void dequeue(envelope *e) { }
    void beginComputation(void);
    void endComputation(void) { }

    void traceInit(char **argv);
    int traceRegisterUserEvent(const char*) { return 0; }
    void traceClearEps();
    void traceWriteSts();
    void traceClose();
    void traceBegin() { }
    void traceEnd() { }

  private:
    int    execEP_;       // id currently executing entry point
    double startEP_;      // start time of currently executing ep
    double startPack_;    // start time of pack operation
    double startUnpack_;  // start time of unpack operation

    int    msgNum_;
};

#endif  // __trace_counter_h__

#endif // CMK_ORIGIN2000

/*@}*/
