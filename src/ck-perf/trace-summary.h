/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _SUMMARY_H
#define _SUMMARY_H

#include <stdio.h>
#include <errno.h>

#include "trace.h"
#include "ck.h"

// in second
#define  BIN_SIZE	0.001

#define  MAX_ENTRIES      500

#define  MAX_MARKS       256

#define  MAX_PHASES       10

#define  CREATION           1
#define  BEGIN_PROCESSING   2
#define  END_PROCESSING     3
#define  ENQUEUE            4
#define  DEQUEUE            5
#define  BEGIN_COMPUTATION  6
#define  END_COMPUTATION    7
#define  BEGIN_INTERRUPT    8
#define  END_INTERRUPT      9
#define  USER_EVENT         13
#define  BEGIN_IDLE         14
#define  END_IDLE           15
#define  BEGIN_PACK         16
#define  END_PACK           17
#define  BEGIN_UNPACK       18
#define  END_UNPACK         19

CpvExtern(int, CtrLogBufSize);

class SumLogEntry {
  public:
    void *operator new(size_t s) {void*ret=malloc(s);_MEMCHECK(ret);return ret;}
    void *operator new(size_t, void *ptr) { return ptr; }
    void operator delete(void *ptr) { free(ptr); }
#ifdef WIN32
    void operator delete(void *, void *) { }
#endif
    SumLogEntry() {}
    SumLogEntry(double t, int p=0) { 
      time = t; pe = p;
    }
    double getTime() { return time; }
    void setTime(double t) { time = t; }

    double time;
    int event;
    int pe;
    void write(FILE *fp);
};

typedef struct _MarkEntry {
double time;
struct _MarkEntry *next;
} MarkEntry;

typedef struct _LogMark {
MarkEntry *marks;
} LogMark;

class PhaseEntry {
  private:
    int count[MAX_ENTRIES];
    double times[MAX_ENTRIES];
  public:
    PhaseEntry();
    void setEp(int epidx, double time) {
	if (epidx>=MAX_ENTRIES) CmiAbort("Too many entry functions!\n");
	count[epidx]++;
	times[epidx] += time;
    }
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
    }
};

class PhaseTable {
  private:
    PhaseEntry **phases;
    int numPhase;
    int cur_phase;
    int phaseCalled;
  public:
    PhaseTable(int n); 
    ~PhaseTable() {
	for (int i=0; i<numPhase; i++) delete phases[i];
	delete [] phases;
    }
    int numPhasesCalled() { return phaseCalled; };
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

class SumLogPool {
  private:
    UInt poolSize;
    UInt numEntries;
    SumLogEntry *pool;
    FILE *fp ;

    double  *epTime;
    int *epCount;
    int epSize;

    // for marks
    LogMark events[MAX_MARKS];
    int markcount;

    // for phases
    PhaseTable phaseTab;
  public:
    SumLogPool(char *pgm);
    ~SumLogPool() {
      write();
      fclose(fp);
      // free memory for mark
      if (markcount > 0)
      for (int i=0; i<MAX_MARKS; i++) {
          MarkEntry *e=events[i].marks, *t;
          while (e) {
	      t = e;
	      e = e->next;
              delete t;
          }
      }
      delete[] pool;
    }
    void write(void) ;
    void writeSts(void);
    void add(double time, int pe);
    void setEp(int epidx, double time);
    void clearEps() {
      for(int i=0; i < epSize; i++) {
	epTime[i]  = 0.;
	epCount[i] = 0;
      }
    }
    void shrink(void) ;
    void addEventType(int eventType, double time);
    void startPhase(int phase) { phaseTab.startPhase(phase); }
};

class TraceSummary : public Trace {
    int curevent;
    int execEvent;
    int execEp;
    int execPe;

    double binStart;
    double start, packstart, unpackstart;
    double bin;
    int msgNum;
  public:
    TraceSummary() { curevent=0; msgNum=0; binStart=0.0; bin=0.0;}
    void userEvent(int e);
    void creation(envelope *e, int num=1);
    void beginExecute(envelope *e);
    void beginExecute(int event,int msgType,int ep,int srcPe);
    void endExecute(void);
    void beginIdle(void);
    void endIdle(void);
    void beginPack(void);
    void endPack(void);
    void beginUnpack(void);
    void endUnpack(void);
    void beginCharmInit(void);
    void endCharmInit(void);
    void enqueue(envelope *e);
    void dequeue(envelope *e);
    void beginComputation(void);
    void endComputation(void);

    void traceInit(char **argv);
    int traceRegisterUserEvent(const char*);
    void traceClearEps();
    void traceWriteSts();
    void traceClose();
    void traceBegin();
    void traceEnd();
};

#endif
