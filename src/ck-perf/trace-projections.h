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

#ifndef _PROJECTIONS_H
#define _PROJECTIONS_H

#include <stdio.h>
#include <errno.h>
#include "trace.h"

#if CMK_PROJECTIONS_USE_ZLIB
#include <zlib.h>
#endif

#include "pup.h"

#define PROJECTION_VERSION  "6.0"

/// a log entry in trace projection
class LogEntry {
  public:
    double time;
    int event;
    int pe;
    unsigned short mIdx;
    unsigned short eIdx;
    unsigned char type; 
    int msglen;
    double recvTime;
    CmiObjId   id;
    int numpes;
    int *pes;
  public:
    LogEntry() {}
    LogEntry(double tm, unsigned char t, unsigned short m=0, unsigned short e=0, int ev=0, int p=0, int ml=0, CmiObjId *d=NULL, double rt=0.) {
      type = t; mIdx = m; eIdx = e; event = ev; pe = p; time = tm; msglen = ml;
      if (d) id = *d; else {id.id[0]=id.id[1]=id.id[2]=0; };
      recvTime = rt; 
    }
    // **CW** new constructor for multicast data
    LogEntry(double tm, unsigned short m, unsigned short e, int ev, int p,
	     int ml, CmiObjId *d, double rt, int num, int *pelist);
    void *operator new(size_t s) {void*ret=malloc(s);_MEMCHECK(ret);return ret;}
    void *operator new(size_t, void *ptr) { return ptr; }
    void operator delete(void *ptr) { free(ptr); }
#if defined(WIN32) || CMK_MULTIPLE_DELETE
    void operator delete(void *, void *) { }
#endif
    void pup(PUP::er &p);
};

/// log pool in trace projection
class LogPool {
  friend class TraceProjections;
  private:
    unsigned int poolSize;
    unsigned int numEntries;
    LogEntry *pool;
    FILE *fp;
    FILE *deltafp;
    FILE *stsfp;
    char *fname;
    char *dfname;
    char *pgmname;
    int binary;
#if CMK_PROJECTIONS_USE_ZLIB
    gzFile deltazfp;
    gzFile zfp;
    int compressed;
#endif
    // **CW** prevTime stores the timestamp of the last event
    // written out to log. This allows the implementation of
    // simple delta encoding and should only be used when
    // writing out logs.
    double prevTime;
    double timeErr;

    int headerWritten;
    void writeHeader();
  public:
    LogPool(char *pgm);
    ~LogPool();
    void setBinary(int b) { binary = b; }
#if CMK_PROJECTIONS_USE_ZLIB
    void setCompressed(int c) { compressed = c; }
#endif
    void creatFiles(char *fix="");
    void openLog(const char *mode);
    void closeLog(void);
    void writeLog(void);
    void write(int writedelta);
    void writeSts(void);
    void add(unsigned char type,unsigned short mIdx,unsigned short eIdx,double time,int event,int pe, int ml=0, CmiObjId* id=0, double recvT=0.);
    void addCreationMulticast(unsigned short mIdx,unsigned short eIdx,double time,int event,int pe, int ml=0, CmiObjId* id=0, double recvT=0., int num=0, int *pelist=NULL);
    void postProcessLog();
};

/// class for recording trace projections events 
/**
  TraceProjections will log Converse/Charm++ events and write into .log files;
  events descriptions will be written into .sts file.
*/
class TraceProjections : public Trace {
    LogPool* _logPool;        /**<  logpool for all events */
    int curevent;
    int execEvent;
    int execEp;
    int execPe;
    int isIdle;
    int cancel_beginIdle, cancel_endIdle;
    int inEntry;
  public:
    TraceProjections(char **argv);
    void userEvent(int e);
    void userBracketEvent(int e, double bt, double et);
    void creation(envelope *e, int epIdx, int num=1);
    void creationMulticast(envelope *e, int epIdx, int num=1, int *pelist=NULL);
    void creationDone(int num=1);
    void beginExecute(envelope *e);
    void beginExecute(CmiObjId  *tid);
    void beginExecute(int event,int msgType,int ep,int srcPe,int ml,CmiObjId *idx=NULL);
    void endExecute(void);
    void messageRecv(char *env, int pe);
    void beginIdle(void);
    void endIdle(void);
    void beginPack(void);
    void endPack(void);
    void beginUnpack(void);
    void endUnpack(void);
    void enqueue(envelope *e);
    void dequeue(envelope *e);
    void beginComputation(void);
    void endComputation(void);

    int traceRegisterUserEvent(const char*, int);
    void traceClearEps();
    void traceWriteSts();
    void traceClose();
    void traceBegin();
    void traceEnd();
};

using namespace PUP;

class toProjectionsFile : public toTextFile {
 protected:
  virtual void bytes(void *p,int n,size_t itemSize,dataType t);
 public:
  //Begin writing to this file, which should be opened for ascii write.
  toProjectionsFile(FILE *f_) :toTextFile(f_) {}
};
class fromProjectionsFile : public fromTextFile {
 protected:
  virtual void bytes(void *p,int n,size_t itemSize,dataType t);
 public:
  //Begin writing to this file, which should be opened for ascii read.
  fromProjectionsFile(FILE *f_) :fromTextFile(f_) {}
};

#if CMK_PROJECTIONS_USE_ZLIB
class toProjectionsGZFile : public PUP::er {
  gzFile f;
 protected:
  virtual void bytes(void *p,int n,size_t itemSize,dataType t);
 public:
  //Begin writing to this gz file, which should be opened for gz write.
  toProjectionsGZFile(gzFile f_) :er(IS_PACKING), f(f_) {}
};
#endif


#endif

/*@}*/
