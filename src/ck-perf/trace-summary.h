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

#ifndef _SUMMARY_H
#define _SUMMARY_H

#include <stdio.h>
#include <errno.h>
#include "trace.h"
#include "ck.h"
#include "trace-common.h"

// time in seconds
#define  BIN_SIZE	0.001

#define  MAX_MARKS       256

#define  MAX_PHASES       10

/// Bin entry record CPU time in an interval
class BinEntry {
  public:
    void *operator new(size_t s) {void*ret=malloc(s);_MEMCHECK(ret);return ret;}
    void *operator new(size_t, void *ptr) { return ptr; }
    void operator delete(void *ptr) { free(ptr); }
#ifdef WIN32
    void operator delete(void *, void *) { }
#endif
    BinEntry(): time(0.) {}
    BinEntry(double t): time(t) {}
    inline double getTime() { return time; }
    void setTime(double t) { time = t; }
    double &Time() { return time; }
    void write(FILE *fp);
    void writeU(FILE *fp, int u);
    int  getU();
  private:
    double time;
};

/// a phase entry for trace summary
class PhaseEntry {
  private:
    int nEPs;
    int *count;
    double *times;
    double *maxtimes;
  public:
    PhaseEntry();
    ~PhaseEntry() { delete [] count; delete [] times; delete [] maxtimes; }
    /// one entry is called for 'time' seconds.
    void setEp(int epidx, double time) {
	if (epidx>=nEPs) CmiAbort("Too many entry functions!\n");
	count[epidx]++;
	times[epidx] += time;
	if (maxtimes[epidx] < time) maxtimes[epidx] = time;
    }
    /**
        write two lines for each phase:
        1. number of calls for each entry;
        2. time in us spent for each entry.
    */
    void write(FILE *fp, int seq) {
	int i;
	fprintf(fp, "[%d] ", seq);
	for (i=0; i<_numEntries; i++) 
	    fprintf(fp, "%d ", count[i]);
	fprintf(fp, "\n");

	fprintf(fp, "[%d] ", seq);
	for (i=0; i<_numEntries; i++) 
	    fprintf(fp, "%ld ", (long)(times[i]*1.0e6) );
	fprintf(fp, "\n");

	fprintf(fp, "[%d] ", seq);
	for (i=0; i<_numEntries; i++) 
	    fprintf(fp, "%ld ", (long)(maxtimes[i]*1.0e6) );
	fprintf(fp, "\n");
    }
};

/// table of PhaseEntry
class PhaseTable {
  private:
    PhaseEntry **phases;
    int numPhase;         /**< phase table size */
    int cur_phase;	  /**< current phase */
    int phaseCalled;      /**< total number of phases */
  public:
    PhaseTable(int n): numPhase(n) {
        phases = new PhaseEntry*[n];
        _MEMCHECK(phases);
        for (int i=0; i<n; i++) phases[i] = NULL;
        cur_phase = -1;
        phaseCalled = 0;
    }
    ~PhaseTable() {
	for (int i=0; i<numPhase; i++) delete phases[i];
	delete [] phases;
    }
    inline int numPhasesCalled() { return phaseCalled; };
    /**
      start a phase. If new, create a new PhaseEntry
    */
    void startPhase(int p) { 
	if (p<0 && p>=numPhase) CmiAbort("Invalid Phase number. \n");
	cur_phase = p; 
	if (phases[cur_phase] == NULL) {
	    phases[cur_phase] = new PhaseEntry;
	    _MEMCHECK(phases[cur_phase]);
	    phaseCalled ++;
        }
    }
    void setEp(int epidx, double time) {
	if (cur_phase == -1) return;
	if (phases[cur_phase] == NULL) CmiAbort("No current phase!\n");
	phases[cur_phase]->setEp(epidx, time);
    }
    void write(FILE *fp) {
	for (int i=0; i<numPhase; i++ )
	    if (phases[i]) { 
		phases[i]->write(fp, i);
            }
    }
};

double epThreshold;
double epInterval;

/// info for each entry
class SumEntryInfo {
public:
  double epTime;
  double epMaxTime;
  int epCount;
  enum {HIST_SIZE = 10};
  int hist[HIST_SIZE];
public:
  SumEntryInfo(): epTime(0.), epMaxTime(0.), epCount(0) {}
  void clear() {
    epTime = epMaxTime = 0.;
    epCount = 0;
    for (int i=0; i<HIST_SIZE; i++) hist[i]=0;
  }
  void setTime(double t) {
    epTime += t;
    epCount ++;
    if (epMaxTime < t) epMaxTime = t;
    for (int i=HIST_SIZE-1; i>=0; i--) {
      if (t>epThreshold+i*epInterval) {
        hist[i]++; break;
      }
    }
  }
};

/// summary log pool
class SumLogPool {
  private:
    UInt poolSize;
    UInt numBins;
    BinEntry *pool;	/**< bins */
    FILE *fp, *stsfp ;

    SumEntryInfo  *epInfo;
    UInt epInfoSize;

    /// a mark entry for trace summary
    typedef struct {
      double time;
    } MarkEntry;
    CkVec<MarkEntry *> events[MAX_MARKS];
    int markcount;

    /// for phases
    PhaseTable phaseTab;
  public:
    SumLogPool(char *pgm);
    ~SumLogPool();
    void initMem();
    void write(void) ;
    void writeSts(void);
    void add(double time, int pe);
    void setEp(int epidx, double time);
    void clearEps() {
      for(int i=0; i < epInfoSize; i++) {
	epInfo[i].clear();
      }
    }
    void shrink(void) ;
    void addEventType(int eventType, double time);
    void startPhase(int phase) { phaseTab.startPhase(phase); }
    BinEntry *bins() { return pool; }
    int getNumEntries() { return numBins; }
};

/// class for recording trace summary events 
/**
  TraceSummary calculate CPU utilizations in bins, and will record
  number of calls and total wall time for each entry. 
*/
class TraceSummary : public Trace {
    SumLogPool*  _logPool;
    int curevent;
    int execEvent;
    int execEp;
    int execPe;

    double binStart;
    double start, packstart, unpackstart;
    double bin;
    int msgNum;
  public:
    TraceSummary(char **argv);
    void beginExecute(envelope *e);
    void beginExecute(int event,int msgType,int ep,int srcPe, int mlen=0);
    void endExecute(void);
    void beginPack(void);
    void endPack(void);
    void beginUnpack(void);
    void endUnpack(void);
    void beginComputation(void);
    void endComputation(void);

    void traceClearEps();
    void traceWriteSts();
    void traceClose();

    /**
       for trace summary event mark
    */
    void addEventType(int eventType);
    /** 
       for starting a new phase
    */
    void startPhase(int phase);

    /**
       query utilities
    */
    SumLogPool *pool() { return _logPool; }
};

#endif

/*@}*/
