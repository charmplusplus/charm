/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _PROJECTIONS_H
#define _PROJECTIONS_H

#include "trace.h"
#include "ck.h"
#include "stdio.h"
#include "errno.h"

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

class LogEntry {
  public:
    void *operator new(size_t s) {void*ret=malloc(s);_MEMCHECK(ret);return ret;}
    void *operator new(size_t, void *ptr) { return ptr; }
    void operator delete(void *ptr) { free(ptr); }
#ifdef WIN32
    void operator delete(void *, void *) { }
#endif
    LogEntry() {}
    LogEntry(double tm, UChar t, UShort m=0, UShort e=0, int ev=0, int p=0) { 
      type = t; mIdx = m; eIdx = e; event = ev; pe = p; time = tm;
    }
    double time;
    int event;
    int pe;
    UShort mIdx;
    UShort eIdx;
    UChar type; 
    void write(FILE *fp);
};

class LogPool {
  private:
    UInt poolSize;
    UInt numEntries;
    LogEntry *pool;
    FILE *fp;
  public:
    LogPool(char *pgm) {
      pool = new LogEntry[CpvAccess(CtrLogBufSize)];
      numEntries = 0;
      poolSize = CpvAccess(CtrLogBufSize);
      char pestr[10];
      sprintf(pestr, "%d", CkMyPe());
      int len = strlen(pgm) + strlen(".log.") + strlen(pestr) + 1;
      char *fname = new char[len];
      sprintf(fname, "%s.%s.log", pgm, pestr);
/*
      fp = fopen(fname, "w");
      delete[] fname;
      if(!fp)
        CmiAbort("Cannot open Projections Trace File for writing...\n");
*/
      do
      {
      fp = fopen(fname, "w+");
      } while (!fp && errno == EINTR);
      delete[] fname;
      if(!fp) {
        CmiAbort("Cannot open Projections Trace File for writing...\n");
      }
      fprintf(fp, "PROJECTIONS-RECORD\n");
    }
    ~LogPool() {
      write();
      fclose(fp);
      delete[] pool;
    }
    void write(void) {
      for(UInt i=0; i<numEntries; i++)
        pool[i].write(fp);
    }
    void writeSts(void);
    void add(UChar type,UShort mIdx,UShort eIdx,double time,int event,int pe) {
      new (&pool[numEntries++])
        LogEntry(time, type, mIdx, eIdx, event, pe);
      if(poolSize==numEntries) {
        double writeTime = CkTimer();
        write();
        numEntries = 0;
        new (&pool[numEntries++]) LogEntry(writeTime, BEGIN_INTERRUPT);
        new (&pool[numEntries++]) LogEntry(CkTimer(), END_INTERRUPT);
      }
    }
};

class TraceProjections : public Trace {
    int curevent;
    int execEvent;
    int execEp;
    int execPe;
  public:
    TraceProjections() { curevent=0; }
    void userEvent(int e);
    void creation(envelope *e, int num=1);
    void beginExecute(envelope *e);
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
};

#endif
