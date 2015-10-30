/**
 * \addtogroup CkPerf
*/
/*@{*/

#ifndef _SUMMARY_H
#define _SUMMARY_H

#include <stdio.h>
#include <errno.h>

#include "trace.h"
#include "envelope.h"
#include "register.h"
#include "trace-common.h"

// initial bin size, time in seconds
#define  BIN_SIZE	0.001

#define  MAX_MARKS       256

#define  MAX_PHASES       100

/// Bin entry record CPU time in an interval
class BinEntry {
  public:
    void *operator new(size_t s) {void*ret=malloc(s);_MEMCHECK(ret);return ret;}
    void *operator new(size_t, void *ptr) { return ptr; }
    void operator delete(void *ptr) { free(ptr); }
#if defined(WIN32) || CMK_MULTIPLE_DELETE
    void operator delete(void *, void *) { }
#endif
    BinEntry(): _time(0.), _idleTime(0.) {}
    BinEntry(double t, double idleT): _time(t), _idleTime(idleT) {}
    double &time() { return _time; }
    double &getIdleTime() { return _idleTime; }
    void write(FILE *fp);
    int  getU();
    int getUIdle();
  private:
    double _time;
    double _idleTime;
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
	int _numEntries=_entryTable.size();
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
    FILE *fp, *stsfp, *sdfp ;
    char *pgm;

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

    /// for Summary-Detail
    double *cpuTime;    //[MAX_INTERVALS * MAX_ENTRIES];
    int *numExecutions; //[MAX_INTERVALS * MAX_ENTRIES];

  public:
    SumLogPool(char *pgm);
    ~SumLogPool();
    double *getCpuTime() {return cpuTime;}
    void initMem();
    void write(void) ;
    void writeSts(void);
    void add(double time, double idleTime, int pe);
    void setEp(int epidx, double time);
    void clearEps() {
      for(int i=0; i < epInfoSize; i++) {
	epInfo[i].clear();
      }
    }
    void shrink(void);
    void shrink(double max);
    void addEventType(int eventType, double time);
    void startPhase(int phase) { phaseTab.startPhase(phase); }
    BinEntry *bins() { return pool; }
    UInt getNumEntries() { return numBins; }
    UInt getEpInfoSize() {return epInfoSize;} 
    UInt getPoolSize() {return poolSize;}
    // accessors to normal summary data
    inline double getTime(unsigned int interval) {
      return pool[interval].time();
    }


    inline double getCPUtime(unsigned int interval, unsigned int ep){
      if(cpuTime != NULL)
        return cpuTime[interval*epInfoSize+ep]; 
      else 
	return 0.0;
    }
    
    inline void setCPUtime(unsigned int interval, unsigned int ep, double val){
        cpuTime[interval*epInfoSize+ep] = val; }
    inline double addToCPUtime(unsigned int interval, unsigned int ep, double val){
        cpuTime[interval*epInfoSize+ep] += val;
        return cpuTime[interval*epInfoSize+ep]; }
    inline int getNumExecutions(unsigned int interval, unsigned int ep){
        return numExecutions[interval*epInfoSize+ep]; }
    inline void setNumExecutions(unsigned int interval, unsigned int ep, unsigned int val){
        numExecutions[interval*epInfoSize+ep] = val; }
    inline int incNumExecutions(unsigned int interval, unsigned int ep){
        ++numExecutions[interval*epInfoSize+ep];
        return numExecutions[interval*epInfoSize+ep]; }
    inline int getUtilization(int interval, int ep);


    void updateSummaryDetail(int epIdx, double startTime, double endTime);


};

/// class for recording trace summary events 
/**
  TraceSummary calculate CPU utilizations in bins, and will record
  number of calls and total wall time for each entry. 
*/
class TraceSummary : public Trace {
    SumLogPool*  _logPool;
    int execEvent;
    int execEp;
    int execPe;

    /* per-log metadata maintained to derive cross-event information */
    double binStart; /* time of last filled bin? */
    double start, packstart, unpackstart, idleStart;
    double binTime, binIdle;
    int msgNum; /* used to handle multiple endComputation calls?? */
    int inIdle;
    int inExec;
    int depth;
  public:
    TraceSummary(char **argv);
    void creation(envelope *e, int epIdx, int num=1) {}

    void beginExecute(envelope *e, void *obj);
    void beginExecute(char *msg);
    void beginExecute(CmiObjId  *tid);
    void beginExecute(int event,int msgType,int ep,int srcPe, int mlen=0, CmiObjId *idx=NULL, void *obj=NULL);
    void endExecute(void);
    void endExecute(char *msg);
    void beginIdle(double currT);
    void endIdle(double currT);
    void traceBegin(void);
    void traceEnd(void);
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

    /**
     *  Supporting methods for CCS queries
     */
    void traceEnableCCS();
    void fillData(double *buffer, double reqStartTime, 
		  double reqBinSize, int reqNumBins);
};

#endif

/*@}*/
