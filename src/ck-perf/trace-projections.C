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

#include <string.h>
#include "charm++.h"
#include "envelope.h"
#include "trace-common.h"
#include "trace-projections.h"

#define DEBUGF(x)           // CmiPrintf x

// **CW** Simple delta encoding implementation
// delta encoding is on by default. It may be turned off later in
// the runtime.
int deltaLog;
int nonDeltaLog;

int checknested=0;		// check illegal nested begin/end execute 

CkpvStaticDeclare(Trace*, _trace);
CtvStaticDeclare(int,curThreadEvent);

typedef CkVec<char *>  usrEventVec;
CkpvStaticDeclare(usrEventVec, usrEventlist);
class UsrEvent {
public:
  int e;
  char *str;
  UsrEvent(int _e, char* _s): e(_e),str(_s) {}
};
CkpvStaticDeclare(CkVec<UsrEvent *>*, usrEvents);

#ifdef CMK_OPTIMIZE
static int warned=0;
#define OPTIMIZED_VERSION 	\
	if (!warned) { warned=1; 	\
	CmiPrintf("\n\n!!!! Warning: traceUserEvent not available in optimized version!!!!\n\n\n"); }
#else
#define OPTIMIZED_VERSION /*empty*/
#endif

/*
On T3E, we need to have file number control by open/close files only when needed.
*/
#if CMK_TRACE_LOGFILE_NUM_CONTROL
  #define OPEN_LOG openLog("a");
  #define CLOSE_LOG closeLog();
#else
  #define OPEN_LOG
  #define CLOSE_LOG
#endif

#if CMK_HAS_COUNTER_PAPI
int numPAPIEvents = 2;
int papiEvents[] = { PAPI_L3_DCM, PAPI_FP_OPS };
char *papiEventNames[] = {"PAPI_L3_DCM", "PAPI_FP_OPS"};
#endif
 
/* ****** CW TEMPORARY LOCATION ***** Support for thread listeners */

struct TraceThreadListener {
  struct CthThreadListener base;
  int event;
  int msgType;
  int ep;
  int srcPe;
  int ml;
  CmiObjId idx;
};


extern "C"
void traceThreadListener_suspend(struct CthThreadListener *l)
{
  TraceThreadListener *a=(TraceThreadListener *)l;
  /* here, we activate the appropriate trace codes for the appropriate
     registered modules */
  traceSuspend();
}

extern "C"
void traceThreadListener_resume(struct CthThreadListener *l) 
{
  TraceThreadListener *a=(TraceThreadListener *)l;
  /* here, we activate the appropriate trace codes for the appropriate
     registered modules */
  _TRACE_BEGIN_EXECUTE_DETAILED(a->event,a->msgType,a->ep,a->srcPe,a->ml,
				CthGetThreadID(a->base.thread));
  a->event=-1;
  a->srcPe=CkMyPe(); /* potential lie to migrated threads */
  a->ml=0;
}

extern "C"
void traceThreadListener_free(struct CthThreadListener *l) 
{
  TraceThreadListener *a=(TraceThreadListener *)l;
  delete a;
}

void TraceProjections::traceAddThreadListeners(CthThread tid, envelope *e)
{
#ifndef CMK_OPTIMIZE
  /* strip essential information from the envelope */
  TraceThreadListener *a= new TraceThreadListener;
  
  a->base.suspend=traceThreadListener_suspend;
  a->base.resume=traceThreadListener_resume;
  a->base.free=traceThreadListener_free;
  a->event=e->getEvent();
  a->msgType=e->getMsgtype();
  a->ep=e->getEpIdx();
  a->srcPe=e->getSrcPe();
  a->ml=e->getTotalsize();

  CthAddListener(tid, (CthThreadListener *)a);
#endif
}

void LogPool::openLog(const char *mode)
{
#if CMK_PROJECTIONS_USE_ZLIB
  if(compressed) {
    if (nonDeltaLog) {
      do {
	zfp = gzopen(fname, mode);
      } while (!zfp && (errno == EINTR || errno == EMFILE));
      if(!zfp) CmiAbort("Cannot open Projections Compressed Non Delta Trace File for writing...\n");
    }
    if (deltaLog) {
      do {
	deltazfp = gzopen(dfname, mode);
      } while (!deltazfp && (errno == EINTR || errno == EMFILE));
      if (!deltazfp) 
	CmiAbort("Cannot open Projections Compressed Delta Trace File for writing...\n");
    }
  } else {
    if (nonDeltaLog) {
      do {
	fp = fopen(fname, mode);
      } while (!fp && (errno == EINTR || errno == EMFILE));
      if(!fp) CmiAbort("Cannot open Projections Non Delta Trace File for writing...\n");
    }
    if (deltaLog) {
      do {
	deltafp = fopen(dfname, mode);
      } while (!deltafp && (errno == EINTR || errno == EMFILE));
      if (!deltafp) 
	CmiAbort("Cannot open Projections Delta Trace File for writing...\n");
    }
  }
#else
  if (nonDeltaLog) {
    do {
      fp = fopen(fname, mode);
    } while (!fp && (errno == EINTR || errno == EMFILE));
    if(!fp) CmiAbort("Cannot open Projections Non Delta Trace File for writing...\n");
  }
  if (deltaLog) {
    do {
      deltafp = fopen(dfname, mode);
    } while (!deltafp && (errno == EINTR || errno == EMFILE));
    if(!deltafp) 
      CmiAbort("Cannot open Projections Delta Trace File for writing...\n");
  }
#endif
}

void LogPool::closeLog(void)
{
#if CMK_PROJECTIONS_USE_ZLIB
  if(compressed) {
    if (nonDeltaLog) gzclose(zfp);
    if (deltaLog) gzclose(deltazfp);
    return;
  }
#endif
  if (nonDeltaLog) { 
#if !defined(_WIN32) || defined(__CYGWIN__)
    fsync(fileno(fp)); 
#endif
    fclose(fp); 
  }
  if (deltaLog)  { 
#if !defined(_WIN32) || defined(__CYGWIN__)
    fsync(fileno(deltafp)); 
#endif
    fclose(deltafp);  
  }
}

/**
  For each TraceFoo module, _createTraceFoo() must be defined.
  This function is called in _createTraces() generated in moduleInit.C
*/
void _createTraceprojections(char **argv)
{
  DEBUGF(("%d createTraceProjections\n", CkMyPe()));
  CkpvInitialize(CkVec<char *>, usrEventlist);
  CkpvInitialize(CkVec<UsrEvent *>*, usrEvents);
  CkpvAccess(usrEvents) = new CkVec<UsrEvent *>();
#if CMK_BLUEGENE_CHARM
  // CthRegister does not call the constructor
//  CkpvAccess(usrEvents) = CkVec<UsrEvent *>();
#endif
  CkpvInitialize(Trace*, _trace);
  CkpvAccess(_trace) = new  TraceProjections(argv);
  CkpvAccess(_traces)->addTrace(CkpvAccess(_trace));
}

LogPool::LogPool(char *pgm) {
  pool = new LogEntry[CkpvAccess(CtrLogBufSize)];
  numEntries = 0;
  // **CW** for simple delta encoding
  prevTime = 0.0;
  timeErr = 0.0;
  poolSize = CkpvAccess(CtrLogBufSize);
  pgmname = new char[strlen(pgm)+1];
  strcpy(pgmname, pgm);
}

void LogPool::creatFiles(char *fix)
{
  char pestr[10];
  sprintf(pestr, "%d", CkMyPe());
#if CMK_PROJECTIONS_USE_ZLIB
  int len;
  if(compressed)
    len = strlen(pgmname)+strlen(fix)+strlen(".logold")+strlen(pestr)+strlen(".gz")+3;
  else
    len = strlen(pgmname)+strlen(fix)+strlen(".logold")+strlen(pestr)+3;
#else
  int len = strlen(pgmname)+strlen(fix)+strlen(".logold")+strlen(pestr)+3;
#endif
  if (nonDeltaLog) {
    fname = new char[len];
  }
  if (deltaLog) {
    dfname = new char[len];
  }
#if CMK_PROJECTIONS_USE_ZLIB
  if(compressed) {
    if (deltaLog && nonDeltaLog) {
      sprintf(fname, "%s%s.%s.logold.gz", pgmname, fix, pestr);
      sprintf(dfname, "%s%s.%s.log.gz", pgmname, fix, pestr);
    } else {
      if (nonDeltaLog) {
	sprintf(fname, "%s%s.%s.log.gz", pgmname, fix, pestr);
      } else {
	sprintf(dfname, "%s%s.%s.log.gz", pgmname, fix, pestr);
      }
    }
  } else {
    if (deltaLog && nonDeltaLog) {
      sprintf(fname, "%s%s.%s.logold", pgmname, fix, pestr);
      sprintf(dfname, "%s%s.%s.log", pgmname, fix, pestr);
    } else {
      if (nonDeltaLog) {
	sprintf(fname, "%s%s.%s.log", pgmname, fix, pestr);
      } else {
	sprintf(dfname, "%s%s.%s.log", pgmname, fix, pestr);
      }
    }
  }
#else
  if (deltaLog && nonDeltaLog) {
    sprintf(fname, "%s%s.%s.logold", pgmname, fix, pestr);
    sprintf(dfname, "%s%s.%s.log", pgmname, fix, pestr);
  } else {
    if (nonDeltaLog) {
      sprintf(fname, "%s%s.%s.log", pgmname, fix, pestr);
    } else {
      sprintf(dfname, "%s%s.%s.log", pgmname, fix, pestr);
    }
  }
#endif
  openLog("w+");
  CLOSE_LOG 

  if (CkMyPe() == 0) 
  {
    char *fname = new char[strlen(CkpvAccess(traceRoot))+strlen(fix)+strlen(".sts")+2];
    sprintf(fname, "%s%s.sts", CkpvAccess(traceRoot), fix);
    do
    {
      stsfp = fopen(fname, "w");
    } while (!stsfp && (errno == EINTR || errno == EMFILE));
    if(stsfp==0)
      CmiAbort("Cannot open projections sts file for writing.\n");
    delete[] fname;
  }
  headerWritten = 0;
}

LogPool::~LogPool() 
{
  writeLog();

#if CMK_BLUEGENE_CHARM
  extern int correctTimeLog;
  if (correctTimeLog) {
    closeLog();
    creatFiles("-bg");
    writeHeader();
    if (CkMyPe() == 0) writeSts(NULL);
    postProcessLog();
  }
#endif

#if !CMK_TRACE_LOGFILE_NUM_CONTROL
  closeLog();
#endif
  delete[] pool;
  delete [] fname;
}

void LogPool::writeHeader()
{
  if (headerWritten) return;
  headerWritten = 1;
  if(!binary) {
#if CMK_PROJECTIONS_USE_ZLIB
    if(compressed) {
      if (nonDeltaLog) {
	gzprintf(zfp, "PROJECTIONS-RECORD %d\n", numEntries);
      }
      if (deltaLog) {
	gzprintf(deltazfp, "PROJECTIONS-RECORD %d DELTA\n", numEntries);
      }
    } 
    else /* else clause is below... */
#endif
    /*... may hang over from else above */ {
      if (nonDeltaLog) {
	fprintf(fp, "PROJECTIONS-RECORD %d\n", numEntries);
      }
      if (deltaLog) {
	fprintf(deltafp, "PROJECTIONS-RECORD %d DELTA\n", numEntries);
      }
    }
  }
  else { // binary
      if (nonDeltaLog) {
        fwrite(&numEntries,sizeof(numEntries),1,fp);
      }
      if (deltaLog) {
        fwrite(&numEntries,sizeof(numEntries),1,deltafp);
      }
  }
}

void LogPool::writeLog(void)
{
  OPEN_LOG
  writeHeader();
  if (nonDeltaLog) write(0);
  if (deltaLog) write(1);
  CLOSE_LOG
}

void LogPool::write(int writedelta) 
{
  // **CW** Simple delta encoding implementation
  // prevTime has to be maintained as an object variable because
  // LogPool::write may be called several times depending on the
  // +logsize value.
  PUP::er *p = NULL;
  if (binary) {
    p = new PUP::toDisk(writedelta?deltafp:fp);
  }
#if CMK_PROJECTIONS_USE_ZLIB
  else if (compressed) {
    p = new toProjectionsGZFile(writedelta?deltazfp:zfp);
  }
#endif
  else {
    p = new toProjectionsFile(writedelta?deltafp:fp);
  }
  CmiAssert(p);
  for(UInt i=0; i<numEntries; i++) {
    if (!writedelta) {
      pool[i].pup(*p);
    }
    else {	// delta
      double time = pool[i].time;
      if (pool[i].type != BEGIN_COMPUTATION && pool[i].type != END_COMPUTATION)
      {
        double timeDiff = (time-prevTime)*1.0e6;
        UInt intTimeDiff = (UInt)timeDiff;
        timeErr += timeDiff - intTimeDiff; /* timeErr is never >= 2.0 */
        if (timeErr > 1.0) {
          timeErr -= 1.0;
          intTimeDiff++;
        }
        pool[i].time = intTimeDiff/1.0e6;
      }
      pool[i].pup(*p);
      pool[i].time = time;	// restore time value
      prevTime = time;
    }
  }
  delete p;
}

void LogPool::writeSts(void)
{
  // for whining compilers
  int i;
  char name[30];
  // generate an automatic unique ID for each log
  fprintf(stsfp, "PROJECTIONS_ID %s\n", "");
  fprintf(stsfp, "VERSION %s\n", PROJECTION_VERSION);
#if CMK_HAS_COUNTER_PAPI
  fprintf(stsfp, "TOTAL_PAPI_EVENTS %d\n", numPAPIEvents);
  // for now, use i, next time use papiEvents[i].
  // **CW** papi event names is a hack.
  for (i=0;i<numPAPIEvents;i++) {
    fprintf(stsfp, "PAPI_EVENT %d %s\n", i, papiEventNames[i]);
  }
#endif
  traceWriteSTS(stsfp,CkpvAccess(usrEvents)->length());
  for(i=0;i<CkpvAccess(usrEvents)->length();i++){
    fprintf(stsfp, "EVENT %d %s\n", (*CkpvAccess(usrEvents))[i]->e, (*CkpvAccess(usrEvents))[i]->str);
  }	
}

void LogPool::writeSts(TraceProjections *traceProj){
  writeSts();
  if (traceProj != NULL) {
    CkHashtableIterator  *funcIter = traceProj->getfuncIterator();
    funcIter->seekStart();
    int numFuncs = traceProj->getFuncNumber();
    fprintf(stsfp,"TOTAL_FUNCTIONS %d \n",numFuncs);
    while(funcIter->hasNext()) {
      StrKey *key;
      int *obj = (int *)funcIter->next((void **)&key);
      fprintf(stsfp,"FUNCTION %d %s \n",*obj,key->getStr());
    }
  }
  fprintf(stsfp, "END\n");
  fclose(stsfp);
}


#if CMK_BLUEGENE_CHARM
static void updateProjLog(void *data, double t, double recvT, void *ptr)
{
  LogEntry *log = (LogEntry *)data;
  FILE *fp = *(FILE **)ptr;
  log->time = t;
  log->recvTime = recvT<0.0?0:recvT;
//  log->write(fp);
  toProjectionsFile p(fp);
  log->pup(p);
}
#endif

// flush log entries to disk
void LogPool::flushLogBuffer()
{
  if (numEntries) {
    double writeTime = TraceTimer();
    writeLog();
    numEntries = 0;
    new (&pool[numEntries++]) LogEntry(writeTime, BEGIN_INTERRUPT);
    new (&pool[numEntries++]) LogEntry(TraceTimer(), END_INTERRUPT);
  }
}

void LogPool::add(UChar type,UShort mIdx,UShort eIdx,double time,int event,
		  int pe, int ml, CmiObjId *id, double recvT, double cpuT)
{
  new (&pool[numEntries++])
    LogEntry(time, type, mIdx, eIdx, event, pe, ml, id, recvT, cpuT);
  if(poolSize==numEntries) {
    flushLogBuffer();
#if CMK_BLUEGENE_CHARM
    extern int correctTimeLog;
    if (correctTimeLog) CmiAbort("I/O interrupt!\n");
#endif
  }
#if CMK_BLUEGENE_CHARM
  switch (type) {
    case BEGIN_PROCESSING:
      pool[numEntries-1].recvTime = BgGetRecvTime();
    case END_PROCESSING:
    case BEGIN_COMPUTATION:
    case END_COMPUTATION:
    case CREATION:
    case BEGIN_PACK:
    case END_PACK:
    case BEGIN_UNPACK:
    case END_UNPACK:
    case USER_EVENT_PAIR:
      bgAddProjEvent(&pool[numEntries-1], numEntries-1, time, updateProjLog, &fp, BG_EVENT_PROJ);
  }
#endif
}

void LogPool::add(UChar type,double time,UShort funcID,int lineNum,char *fileName){
#ifndef CMK_BLUEGENE_CHARM
  new (&pool[numEntries++])
	LogEntry(time,type,funcID,lineNum,fileName);
  if(poolSize == numEntries){
    flushLogBuffer();
  }
#endif	
}

/* **CW** Not sure if this is the right thing to do. Feels more like
   a hack than a solution to Sameer's request to add the destination
   processor information to multicasts and broadcasts.

   In the unlikely event this method is used for Broadcasts as well,
   pelist == NULL will be used to indicate a global broadcast with 
   num PEs.
*/
void LogPool::addCreationMulticast(UShort mIdx, UShort eIdx, double time,
				   int event, int pe, int ml, CmiObjId *id,
				   double recvT, int num, int *pelist)
{
  new (&pool[numEntries++])
    LogEntry(time, mIdx, eIdx, event, pe, ml, id, recvT, num, pelist);
  if(poolSize==numEntries) {
    flushLogBuffer();
  }
}

void LogPool::postProcessLog()
{
#if CMK_BLUEGENE_CHARM
  bgUpdateProj(1);   // event type
#endif
}

LogEntry::LogEntry(double tm, unsigned short m, unsigned short e, int ev, int p,
	     int ml, CmiObjId *d, double rt, int num, int *pelist) 
{
    type = CREATION_MULTICAST; mIdx = m; eIdx = e; event = ev; pe = p; time = tm; msglen = ml;
    if (d) id = *d; else {id.id[0]=id.id[1]=id.id[2]=0; };
    recvTime = rt; 
    numpes = num;
    if (pelist != NULL) {
	pes = new int[num];
	for (int i=0; i<num; i++) {
	  pes[i] = pelist[i];
	}
    } else {
	pes= NULL;
    }
}

void LogEntry::addPapi(int numPapiEvts, int *papi_ids, LONG_LONG_PAPI *papiVals)
{
#if CMK_HAS_COUNTER_PAPI
  numPapiEvents = numPapiEvts;
  if (papiVals != NULL) {
    papiIDs = new int[numPapiEvents];
    papiValues = new LONG_LONG_PAPI[numPapiEvents];
    for (int i=0; i<numPapiEvents; i++) {
      papiIDs[i] = papi_ids[i];
      papiValues[i] = papiVals[i];
    }
  }
#endif
}

void LogEntry::pup(PUP::er &p)
{
  int i;
  CMK_TYPEDEF_UINT8 itime, irecvtime, icputime;
  char ret = '\n';

  p|type;
  if (p.isPacking()) itime = (CMK_TYPEDEF_UINT8)(1.0e6*time);
  switch (type) {
    case USER_EVENT:
    case USER_EVENT_PAIR:
      p|mIdx; p|itime; p|event; p|pe;
      break;
    case BEGIN_IDLE:
    case END_IDLE:
    case BEGIN_PACK:
    case END_PACK:
    case BEGIN_UNPACK:
    case END_UNPACK:
      p|itime; p|pe; 
      break;
    case BEGIN_PROCESSING:
      if (p.isPacking()) {
        irecvtime = (CMK_TYPEDEF_UINT8)(recvTime==-1?-1:1.0e6*recvTime);
        icputime = (CMK_TYPEDEF_UINT8)(1.0e6*cputime);
      }
      p|mIdx; p|eIdx; p|itime; p|event; p|pe; 
      p|msglen; p|irecvtime; p|id.id[0]; p|id.id[1]; p|id.id[2];
      p|icputime;
#if CMK_HAS_COUNTER_PAPI
      p|numPapiEvents;
      for (i=0; i<numPapiEvents; i++) {
	// not yet!!!
	//	p|papiIDs[i]; 
	p|papiValues[i];
      }
#else
      p|numPapiEvents;     // non papi version has value 0
#endif
      if (p.isUnpacking()) {
	recvTime = irecvtime/1.0e6;
	cputime = icputime/1.0e6;
      }
      break;
    case END_PROCESSING:
      if (p.isPacking()) icputime = (CMK_TYPEDEF_UINT8)(1.0e6*cputime);
      p|mIdx; p|eIdx; p|itime; p|event; p|pe; p|msglen; p|icputime;
#if CMK_HAS_COUNTER_PAPI
      p|numPapiEvents;
      for (i=0; i<numPapiEvents; i++) {
	// not yet!!!
	//	p|papiIDs[i];
	p|papiValues[i];
      }
#else
      p|numPapiEvents;  // non papi version has value 0
#endif
      if (p.isUnpacking()) cputime = icputime/1.0e6;
      break;
    case CREATION:
      if (p.isPacking()) irecvtime = (CMK_TYPEDEF_UINT8)(1.0e6*recvTime);
      p|mIdx; p|eIdx; p|itime;
      p|event; p|pe; p|msglen; p|irecvtime;
      if (p.isUnpacking()) recvTime = irecvtime/1.0e6;
      break;
    case CREATION_MULTICAST:
      if (p.isPacking()) irecvtime = (CMK_TYPEDEF_UINT8)(1.0e6*recvTime);
      p|mIdx; p|eIdx; p|itime;
      p|event; p|pe; p|msglen; p|irecvtime; p|numpes;
      if (pes == NULL) {
        int n=-1;
        p(n);
      }
      else {
	for (i=0; i<numpes; i++) p|pes[i];
      }
      if (p.isUnpacking()) recvTime = irecvtime/1.0e6;
      break;
    case MESSAGE_RECV:
      p|mIdx; p|eIdx; p|itime; p|event; p|pe; p|msglen;
      break;

    case ENQUEUE:
    case DEQUEUE:
      p|mIdx; p|itime; p|event; p|pe;
      break;

    case BEGIN_INTERRUPT:
    case END_INTERRUPT:
      p|itime; p|event; p|pe;
      break;

      // **CW** absolute timestamps are used here to support a quick
      // way of determining the total time of a run in projections
      // visualization.
    case BEGIN_COMPUTATION:
    case END_COMPUTATION:
    case BEGIN_TRACE:
    case END_TRACE:
      p|itime;
      break;
    case BEGIN_FUNC:
	p | itime;
	p | mIdx;
	p | event;
	if(!p.isUnpacking()){
		p(fName,flen-1);
	}
	break;
    case END_FUNC:
	p | itime;
	p | mIdx;
	break;
    default:
      CmiError("***Internal Error*** Wierd Event %d.\n", type);
      break;
  }
  if (p.isUnpacking()) time = itime/1.0e6;
  p|ret;
}

TraceProjections::TraceProjections(char **argv): 
curevent(0), inEntry(0), computationStarted(0)
{
  if (CkpvAccess(traceOnPe) == 0) return;

  CtvInitialize(int,curThreadEvent);
  CtvAccess(curThreadEvent)=0;
  checknested = CmiGetArgFlagDesc(argv,"+checknested","check projections nest begin end execute events");
  int binary = CmiGetArgFlagDesc(argv,"+binary-trace","Write log files in (unreadable) binary format");
#if CMK_PROJECTIONS_USE_ZLIB
  int compressed = CmiGetArgFlagDesc(argv,"+gz-trace","Write log files pre-compressed with gzip");
#endif

  // **CW** default to non delta log encoding. The user may choose to do
  // create both logs (for debugging) or just the old log timestamping
  // (for compatibility).
  // Generating just the non delta log takes precedence over generating
  // both logs (if both arguments appear on the command line).

  // switch to OLD log format until everything works // Gengbin
  nonDeltaLog = 1;
  deltaLog = 0;
  deltaLog = CmiGetArgFlagDesc(argv, "+logDelta",
				  "Generate Delta encoded and simple timestamped log files");

  _logPool = new LogPool(CkpvAccess(traceRoot));
  _logPool->setBinary(binary);
#if CMK_PROJECTIONS_USE_ZLIB
  _logPool->setCompressed(compressed);
#endif
  _logPool->creatFiles();

  funcCount=1;

#if CMK_HAS_COUNTER_PAPI
  // We initialize and create the event sets for use with PAPI here.
  int papiRetValue = PAPI_library_init(PAPI_VER_CURRENT);
  if (papiRetValue != PAPI_VER_CURRENT) {
    CmiAbort("PAPI Library initialization failure!\n");
  }
  // PAPI 3 mandates the initialization of the set to PAPI_NULL
  papiEventSet = PAPI_NULL; 
  if (PAPI_create_eventset(&papiEventSet) != PAPI_OK) {
    CmiAbort("PAPI failed to create event set!\n");
  }
  papiRetValue = PAPI_add_events(papiEventSet, papiEvents, numPAPIEvents);
  if (papiRetValue != PAPI_OK) {
    if (papiRetValue == PAPI_ECNFLCT) {
      CmiAbort("PAPI events conflict! Please re-assign event types!\n");
    } else {
      CmiAbort("PAPI failed to add designated events!\n");
    }
  }
  papiValues = new long_long[numPAPIEvents];
  memset(papiValues, 0, numPAPIEvents*sizeof(long_long));
#endif
}

int TraceProjections::traceRegisterUserEvent(const char* evt, int e)
{
  OPTIMIZED_VERSION
  CkAssert(e==-1 || e>=0);
  CkAssert(evt != NULL);
  int event;
  int biggest = -1;
  for (int i=0; i<CkpvAccess(usrEvents)->length(); i++) {
    int cur = (*CkpvAccess(usrEvents))[i]->e;
    if (cur == e) {
      //CmiPrintf("%s %s\n", (*CkpvAccess(usrEvents))[i]->str, evt);
      if (strcmp((*CkpvAccess(usrEvents))[i]->str, evt) == 0) 
        return e;
      else
        CmiAbort("UserEvent double registered!");
    }
    if (cur > biggest) biggest = cur;
  }
  // if no user events have so far been registered. biggest will be -1
  // and hence newly assigned event numbers will begin from 0.
  if (e==-1) event = biggest+1;  // automatically assign new event number
  else event = e;
  CkpvAccess(usrEvents)->push_back(new UsrEvent(event,(char *)evt));
  return event;
}

void TraceProjections::traceClearEps(void)
{
  // In trace-summary, this zeros out the EP bins, to eliminate noise
  // from startup.  Here, this isn't useful, since we can do that in
  // post-processing
}

void TraceProjections::traceWriteSts(void)
{
  if(CkMyPe()==0)
    _logPool->writeSts(this);
}

void TraceProjections::traceClose(void)
{
  if(CkMyPe()==0){
    _logPool->writeSts(this);
  }
  CkpvAccess(_trace)->endComputation();
  delete _logPool;		// will write
  // remove myself from traceArray so that no tracing will be called.
  CkpvAccess(_traces)->removeTrace(this);
//    delete CkpvAccess(_trace);
//  free(CkpvAccess(traceRoot));
}

void TraceProjections::traceBegin(void)
{
  if (!computationStarted) return;
  _logPool->add(BEGIN_TRACE, 0, 0, TraceTimer(), curevent++, CkMyPe());
}

void TraceProjections::traceEnd(void)
{
  _logPool->add(END_TRACE, 0, 0, TraceTimer(), curevent++, CkMyPe());
}

void TraceProjections::userEvent(int e)
{
  if (!computationStarted) return;
  _logPool->add(USER_EVENT, e, 0, TraceTimer(),curevent++,CkMyPe());
}

void TraceProjections::userBracketEvent(int e, double bt, double et)
{
  if (!computationStarted) return;
  // two events record Begin/End of event e.
  _logPool->add(USER_EVENT_PAIR, e, 0, TraceTimer(bt), curevent, CkMyPe());
  _logPool->add(USER_EVENT_PAIR, e, 0, TraceTimer(et), curevent++, CkMyPe());
}

void TraceProjections::creation(envelope *e, int ep, int num)
{
  double curTime = TraceTimer();
  if(e==0) {
    CtvAccess(curThreadEvent)=curevent;
    _logPool->add(CREATION,ForChareMsg,ep,curTime,
                    curevent++,CkMyPe(), 0, 0, 0.0);
  } else {
    int type=e->getMsgtype();
    e->setEvent(curevent);
    for(int i=0; i<num; i++) {
      _logPool->add(CREATION,type,ep, curTime,
                    curevent+i,CkMyPe(),e->getTotalsize(), 0, 0.0);
    }
    curevent += num;
  }
}

/* **CW** Non-disruptive attempt to add destination PE knowledge to
   Communication Library-specific Multicasts via new event 
   CREATION_MULTICAST.
*/

void TraceProjections::creationMulticast(envelope *e, int ep, int num,
					 int *pelist)
{
  double curTime = TraceTimer();
  if (e==0) {
    CtvAccess(curThreadEvent)=curevent;
    _logPool->addCreationMulticast(ForChareMsg, ep, curTime, curevent++,
				   CkMyPe(), 0, 0, 0.0, num, pelist);
  } else {
    int type=e->getMsgtype();
    e->setEvent(curevent);
    _logPool->addCreationMulticast(type, ep, curTime, curevent++, CkMyPe(),
				   e->getTotalsize(), 0, 0.0, num, pelist);
  }
}

void TraceProjections::creationDone(int num)
{
  // modified the creation done time of the last num log entries
  // FIXME: totally a hack
  double curTime = TraceTimer();
  int idx = _logPool->numEntries-1;
  while (idx >=0 && num >0 ) {
    LogEntry &log = _logPool->pool[idx];
    if (log.type == CREATION) {
      log.recvTime = curTime - log.time;
      num --;
    }
    idx--;
  }
}

void TraceProjections::beginExecute(CmiObjId *tid)
{
#if CMK_HAS_COUNTER_PAPI
  if (PAPI_read(papiEventSet, papiValues) != PAPI_OK) {
    CmiAbort("PAPI failed to read at begin execute!\n");
  }
#endif
  if (checknested && inEntry) CmiAbort("Nested Begin Execute!\n");
  execEvent = CtvAccess(curThreadEvent);
  execEp = (-1);
  _logPool->add(BEGIN_PROCESSING,ForChareMsg,_threadEP,TraceTimer(),
                             execEvent,CkMyPe(), 0, tid);
#if CMK_HAS_COUNTER_PAPI
  _logPool->addPapi(numPAPIEvents, papiEvents, papiValues);
#endif
  inEntry = 1;
}

void TraceProjections::beginExecute(envelope *e)
{
  if(e==0) {
#if CMK_HAS_COUNTER_PAPI
    if (PAPI_read(papiEventSet, papiValues) != PAPI_OK) {
      CmiAbort("PAPI failed to read at begin execute!\n");
    }
#endif
    if (checknested && inEntry) CmiAbort("Nested Begin Execute!\n");
    execEvent = CtvAccess(curThreadEvent);
    execEp = (-1);
    _logPool->add(BEGIN_PROCESSING,ForChareMsg,_threadEP,TraceTimer(),
		  execEvent,CkMyPe(), 0, 0, 0.0, TraceCpuTimer());
#if CMK_HAS_COUNTER_PAPI
    _logPool->addPapi(numPAPIEvents, papiEvents, papiValues);
#endif
    inEntry = 1;
  } else {
    beginExecute(e->getEvent(),e->getMsgtype(),e->getEpIdx(),e->getSrcPe(),e->getTotalsize());
  }
}

void TraceProjections::beginExecute(int event,int msgType,int ep,int srcPe, int mlen,CmiObjId *idx)
{
#if CMK_HAS_COUNTER_PAPI
  if (PAPI_read(papiEventSet, papiValues) != PAPI_OK) {
    CmiAbort("PAPI failed to read at begin execute!\n");
  }
#endif
  if (checknested && inEntry) CmiAbort("Nested Begin Execute!\n");
  execEvent=event;
  execEp=ep;
  execPe=srcPe;
  _logPool->add(BEGIN_PROCESSING,msgType,ep,TraceTimer(),event,
		srcPe, mlen, idx, 0.0, TraceCpuTimer());
#if CMK_HAS_COUNTER_PAPI
  _logPool->addPapi(numPAPIEvents, papiEvents, papiValues);
#endif
  inEntry = 1;
}

void TraceProjections::endExecute(void)
{
#if CMK_HAS_COUNTER_PAPI
  if (PAPI_read(papiEventSet, papiValues) != PAPI_OK) {
    CmiAbort("PAPI failed to read at end execute!\n");
  }
#endif
  if (checknested && !inEntry) CmiAbort("Nested EndExecute!\n");
  double cputime = TraceCpuTimer();
  if(execEp == (-1)) {
    _logPool->add(END_PROCESSING,0,_threadEP,TraceTimer(),
                             execEvent,CkMyPe(),0,0,0.0,cputime);
  } else {
    _logPool->add(END_PROCESSING,0,execEp,TraceTimer(),
                             execEvent,execPe,0,0,0.0,cputime);
  }
#if CMK_HAS_COUNTER_PAPI
  _logPool->addPapi(numPAPIEvents, papiEvents, papiValues);
#endif
  inEntry = 0;
}

void TraceProjections::messageRecv(char *env, int pe)
{
#if 0
  envelope *e = (envelope *)env;
  int msgType = e->getMsgtype();
  int ep = e->getEpIdx();
#if 0
  if (msgType==NewChareMsg || msgType==NewVChareMsg
          || msgType==ForChareMsg || msgType==ForVidMsg
          || msgType==BocInitMsg || msgType==NodeBocInitMsg
          || msgType==ForBocMsg || msgType==ForNodeBocMsg)
    ep = e->getEpIdx();
  else
    ep = _threadEP;
#endif
  _logPool->add(MESSAGE_RECV,msgType,ep,TraceTimer(),
                             curevent++,e->getSrcPe(), e->getTotalsize());
#endif
}

void TraceProjections::beginIdle(double curWallTime)
{
  _logPool->add(BEGIN_IDLE, 0, 0, TraceTimer(curWallTime), 0, CkMyPe());
}

void TraceProjections::endIdle(double curWallTime)
{
  _logPool->add(END_IDLE, 0, 0, TraceTimer(curWallTime), 0, CkMyPe());
}

void TraceProjections::beginPack(void)
{
  _logPool->add(BEGIN_PACK, 0, 0, TraceTimer(), 0, CkMyPe());
}

void TraceProjections::endPack(void)
{
  _logPool->add(END_PACK, 0, 0, TraceTimer(), 0, CkMyPe());
}

void TraceProjections::beginUnpack(void)
{
  _logPool->add(BEGIN_UNPACK, 0, 0, TraceTimer(), 0, CkMyPe());
}

void TraceProjections::endUnpack(void)
{
  _logPool->add(END_UNPACK, 0, 0, TraceTimer(), 0, CkMyPe());
}

void TraceProjections::enqueue(envelope *) {}

void TraceProjections::dequeue(envelope *) {}

void TraceProjections::beginComputation(void)
{
  computationStarted = 1;
//  CkpvAccess(traceInitTime) = TRACE_TIMER();
//  CkpvAccess(traceInitCpuTime) = TRACE_CPUTIMER();
  _logPool->add(BEGIN_COMPUTATION, 0, 0, TraceTimer(), -1, -1);
#if CMK_HAS_COUNTER_PAPI
  // we start the counters here
  if (PAPI_start(papiEventSet) != PAPI_OK) {
    CmiAbort("PAPI failed to start designated counters!\n");
  }
#endif
}

void TraceProjections::endComputation(void)
{
#if CMK_HAS_COUNTER_PAPI
  // we stop the counters here. A silent failure is alright since we
  // are already at the end of the program.
  if (PAPI_stop(papiEventSet, papiValues) != PAPI_OK) {
    CkPrintf("Warning: PAPI failed to stop correctly!\n");
  }
  // NOTE: We should not do a complete close of PAPI until after the
  // sts writer is done.
#endif
  _logPool->add(END_COMPUTATION, 0, 0, TraceTimer(), -1, -1);
}

int TraceProjections::idxRegistered(int idx)
{
    int idxVecLen = idxVec.size();
    for(int i=0; i<idxVecLen; i++)
    {
	if(idx == idxVec[i])
	    return 1;
    }
    return 0;
}

void TraceProjections::regFunc(const char *name, int &idx, int idxSpecifiedByUser){
    StrKey k((char*)name,strlen(name));
    int num = funcHashtable.get(k);
    
    if(num!=0) {
	return;
	//as for mpi programs, the same function may be registered for several times
	//CmiError("\"%s has been already registered! Please change the name!\"\n", name);
    }
    
    int isIdxExisting=0;
    if(idxSpecifiedByUser)
	isIdxExisting=idxRegistered(idx);
    if(isIdxExisting){
	return;
	//same reason with num!=0
	//CmiError("The identifier %d for the trace function has been already registered!", idx);
    }

    if(idxSpecifiedByUser) {
    	char *st = new char[strlen(name)+1];
    	memcpy(st,name,strlen(name)+1);
    	StrKey *newKey = new StrKey(st,strlen(st));
    	int &ref = funcHashtable.put(*newKey);
    	ref=idx;
        funcCount++;
	idxVec.push_back(idx);	
    } else {
    	char *st = new char[strlen(name)+1];
    	memcpy(st,name,strlen(name)+1);
    	StrKey *newKey = new StrKey(st,strlen(st));
    	int &ref = funcHashtable.put(*newKey);
    	ref=funcCount;
    	num = funcCount;
    	funcCount++;
    	idx = num;
	idxVec.push_back(idx);
    }
}

void TraceProjections::beginFunc(char *name,char *file,int line){
	StrKey k(name,strlen(name));	
	unsigned short  num = (unsigned short)funcHashtable.get(k);
	beginFunc(num,file,line);
}

void TraceProjections::beginFunc(int idx,char *file,int line){
	if(idx <= 0){
		CmiError("Unregistered function id %d being used in %s:%d \n",idx,file,line);
	}	
	_logPool->add(BEGIN_FUNC,TraceTimer(),idx,line,file);
}

void TraceProjections::endFunc(char *name){
	StrKey k(name,strlen(name));	
	int num = funcHashtable.get(k);
	endFunc(num);	
}

void TraceProjections::endFunc(int num){
	if(num <= 0){
		printf("endFunc without start :O\n");
	}
	_logPool->add(END_FUNC,TraceTimer(),num,0,NULL);
}

// specialized PUP:ers for handling trace projections logs
void toProjectionsFile::bytes(void *p,int n,size_t itemSize,dataType t)
{
  for (int i=0;i<n;i++) 
    switch(t) {
    case Tchar: CheckAndFPrintF(f,"%c",((char *)p)[i]); break;
    case Tuchar:
    case Tbyte: CheckAndFPrintF(f,"%d",((unsigned char *)p)[i]); break;
    case Tshort: CheckAndFPrintF(f," %d",((short *)p)[i]); break;
    case Tushort: CheckAndFPrintF(f," %u",((unsigned short *)p)[i]); break;
    case Tint: CheckAndFPrintF(f," %d",((int *)p)[i]); break;
    case Tuint: CheckAndFPrintF(f," %u",((unsigned int *)p)[i]); break;
    case Tlong: CheckAndFPrintF(f," %ld",((long *)p)[i]); break;
    case Tulong: CheckAndFPrintF(f," %lu",((unsigned long *)p)[i]); break;
    case Tfloat: CheckAndFPrintF(f," %.7g",((float *)p)[i]); break;
    case Tdouble: CheckAndFPrintF(f," %.15g",((double *)p)[i]); break;
#ifdef CMK_PUP_LONG_LONG
    case Tlonglong: CheckAndFPrintF(f," %lld",((CMK_TYPEDEF_INT8 *)p)[i]); break;
    case Tulonglong: CheckAndFPrintF(f," %llu",((CMK_TYPEDEF_UINT8 *)p)[i]); break;
#endif
    default: CmiAbort("Unrecognized pup type code!");
    };
}

void fromProjectionsFile::bytes(void *p,int n,size_t itemSize,dataType t)
{
  for (int i=0;i<n;i++) 
    switch(t) {
    case Tchar: { 
      char c = fgetc(f);
      if (c==EOF)
	parseError("Could not match character");
      else
        ((char *)p)[i] = c;
      break;
    }
    case Tuchar:
    case Tbyte: ((unsigned char *)p)[i]=(unsigned char)readInt("%d"); break;
    case Tshort:((short *)p)[i]=(short)readInt(); break;
    case Tushort: ((unsigned short *)p)[i]=(unsigned short)readUint(); break;
    case Tint:  ((int *)p)[i]=readInt(); break;
    case Tuint: ((unsigned int *)p)[i]=readUint(); break;
    case Tlong: ((long *)p)[i]=readInt(); break;
    case Tulong:((unsigned long *)p)[i]=readUint(); break;
    case Tfloat: ((float *)p)[i]=(float)readDouble(); break;
    case Tdouble:((double *)p)[i]=readDouble(); break;
#ifdef CMK_PUP_LONG_LONG
    case Tlonglong: ((CMK_TYPEDEF_INT8 *)p)[i]=readLongInt(); break;
    case Tulonglong: ((CMK_TYPEDEF_UINT8 *)p)[i]=readLongInt(); break;
#endif
    default: CmiAbort("Unrecognized pup type code!");
    };
}

#if CMK_PROJECTIONS_USE_ZLIB
void toProjectionsGZFile::bytes(void *p,int n,size_t itemSize,dataType t)
{
  for (int i=0;i<n;i++) 
    switch(t) {
    case Tchar: gzprintf(f,"%c",((char *)p)[i]); break;
    case Tuchar:
    case Tbyte: gzprintf(f,"%d",((unsigned char *)p)[i]); break;
    case Tshort: gzprintf(f," %d",((short *)p)[i]); break;
    case Tushort: gzprintf(f," %u",((unsigned short *)p)[i]); break;
    case Tint: gzprintf(f," %d",((int *)p)[i]); break;
    case Tuint: gzprintf(f," %u",((unsigned int *)p)[i]); break;
    case Tlong: gzprintf(f," %ld",((long *)p)[i]); break;
    case Tulong: gzprintf(f," %lu",((unsigned long *)p)[i]); break;
    case Tfloat: gzprintf(f," %.7g",((float *)p)[i]); break;
    case Tdouble: gzprintf(f," %.15g",((double *)p)[i]); break;
#ifdef CMK_PUP_LONG_LONG
    case Tlonglong: gzprintf(f," %lld",((CMK_TYPEDEF_INT8 *)p)[i]); break;
    case Tulonglong: gzprintf(f," %llu",((CMK_TYPEDEF_UINT8 *)p)[i]); break;
#endif
    default: CmiAbort("Unrecognized pup type code!");
    };
}
#endif

/*@}*/
