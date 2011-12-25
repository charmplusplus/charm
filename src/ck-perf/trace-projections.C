/**
 * \addtogroup CkPerf
*/
/*@{*/

#include <string.h>

#include "charm++.h"
#include "trace-projections.h"
#include "trace-projectionsBOC.h"

#if DEBUG_PROJ
#define DEBUGF(...) CkPrintf(__VA_ARGS__)
#else
#define DEBUGF(...)
#endif
#define DEBUGN(...)  // easy way to selectively disable DEBUGs

#define DefaultLogBufSize      1000000

// **CW** Simple delta encoding implementation
// delta encoding is on by default. It may be turned off later in
// the runtime.
int deltaLog;
int nonDeltaLog;

int checknested=0;		// check illegal nested begin/end execute 

#ifdef PROJ_ANALYSIS
// BOC operations readonlys
CkGroupID traceProjectionsGID;
CkGroupID kMeansGID;

// New reduction type for Outlier Analysis purposes. This is allowed to be
// a global variable according to the Charm++ manual.
CkReductionMsg *outlierReduction(int nMsgs,
				 CkReductionMsg **msgs);
CkReductionMsg *minMaxReduction(int nMsgs,
				CkReductionMsg **msgs);
CkReduction::reducerType outlierReductionType;
CkReduction::reducerType minMaxReductionType;
#endif // PROJ_ANALYSIS

CkpvStaticDeclare(TraceProjections*, _trace);
CtvStaticDeclare(int,curThreadEvent);

CkpvDeclare(CmiInt8, CtrLogBufSize);

typedef CkVec<char *>  usrEventVec;
CkpvStaticDeclare(usrEventVec, usrEventlist);
class UsrEvent {
public:
  int e;
  char *str;
  UsrEvent(int _e, char* _s): e(_e),str(_s) {}
};
CkpvStaticDeclare(CkVec<UsrEvent *>*, usrEvents);

// When tracing is disabled, these are defined as empty static inlines
// in the header, to minimize overhead
#if CMK_TRACE_ENABLED
/// Disable the outputting of the trace logs
void disableTraceLogOutput()
{ 
  CkpvAccess(_trace)->setWriteData(false);
}

/// Enable the outputting of the trace logs
void enableTraceLogOutput()
{
  CkpvAccess(_trace)->setWriteData(true);
}

/// Force the log files to be flushed
void flushTraceLog()
{
  CkpvAccess(_trace)->traceFlushLog();
}
#endif

#if ! CMK_TRACE_ENABLED
static int warned=0;
#define OPTIMIZED_VERSION 	\
	if (!warned) { warned=1; 	\
	CmiPrintf("\n\n!!!! Warning: traceUserEvent not available in optimized version!!!!\n\n\n"); }
#else
#define OPTIMIZED_VERSION /*empty*/
#endif // CMK_TRACE_ENABLED

/*
On T3E, we need to have file number control by open/close files only when needed.
*/
#if CMK_TRACE_LOGFILE_NUM_CONTROL
  #define OPEN_LOG openLog("a");
  #define CLOSE_LOG closeLog();
#else
  #define OPEN_LOG
  #define CLOSE_LOG
#endif //CMK_TRACE_LOGFILE_NUM_CONTROL

#if CMK_HAS_COUNTER_PAPI
int papiEvents[NUMPAPIEVENTS] = { PAPI_L2_DCM, PAPI_FP_OPS };
#endif // CMK_HAS_COUNTER_PAPI

/**
  For each TraceFoo module, _createTraceFoo() must be defined.
  This function is called in _createTraces() generated in moduleInit.C
*/
void _createTraceprojections(char **argv)
{
  DEBUGF("%d createTraceProjections\n", CkMyPe());
  CkpvInitialize(CkVec<char *>, usrEventlist);
  CkpvInitialize(CkVec<UsrEvent *>*, usrEvents);
  CkpvAccess(usrEvents) = new CkVec<UsrEvent *>();
#if CMK_BIGSIM_CHARM
  // CthRegister does not call the constructor
//  CkpvAccess(usrEvents) = CkVec<UsrEvent *>();
#endif //CMK_BIGSIM_CHARM
  CkpvInitialize(TraceProjections*, _trace);
  CkpvAccess(_trace) = new  TraceProjections(argv);
  CkpvAccess(_traces)->addTrace(CkpvAccess(_trace));
  if (CkMyPe()==0) CkPrintf("Charm++: Tracemode Projections enabled.\n");
}
 
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
#if CMK_TRACE_ENABLED
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
      if (!fp) {
	CkPrintf("[%d] Attempting to open file [%s]\n",CkMyPe(),fname);
	CmiAbort("Cannot open Projections Non Delta Trace File for writing...\n");
      }
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
    if (!fp) {
      CkPrintf("[%d] Attempting to open file [%s]\n",CkMyPe(),fname);
      CmiAbort("Cannot open Projections Non Delta Trace File for writing...\n");
    }
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

LogPool::LogPool(char *pgm) {
  pool = new LogEntry[CkpvAccess(CtrLogBufSize)];
  // defaults to writing data (no outlier changes)
  writeData = true;
  numEntries = 0;
  // **CW** for simple delta encoding
  prevTime = 0.0;
  timeErr = 0.0;
  globalStartTime = 0.0;
  globalEndTime = 0.0;
  headerWritten = 0;
  numPhases = 0;
  hasFlushed = false;

  keepPhase = NULL;

  fileCreated = false;
  poolSize = CkpvAccess(CtrLogBufSize);
  pgmname = new char[strlen(pgm)+1];
  strcpy(pgmname, pgm);
}

void LogPool::createFile(const char *fix)
{
  if (fileCreated) {
    return;
  }

  char* filenameLastPart = strrchr(pgmname, PATHSEP) + 1; // Last occurrence of path separator
  char *pathPlusFilePrefix = new char[1024];

  if(nSubdirs > 0){
    int sd = CkMyPe() % nSubdirs;
    char *subdir = new char[1024];
    sprintf(subdir, "%s.projdir.%d", pgmname, sd);
    CmiMkdir(subdir);
    sprintf(pathPlusFilePrefix, "%s%c%s%s", subdir, PATHSEP, filenameLastPart, fix);
    delete[] subdir;
  } else {
    sprintf(pathPlusFilePrefix, "%s%s", pgmname, fix);
  }

  char pestr[10];
  sprintf(pestr, "%d", CkMyPe());
#if CMK_PROJECTIONS_USE_ZLIB
  int len;
  if(compressed)
    len = strlen(pathPlusFilePrefix)+strlen(".logold")+strlen(pestr)+strlen(".gz")+3;
  else
    len = strlen(pathPlusFilePrefix)+strlen(".logold")+strlen(pestr)+3;
#else
  int len = strlen(pathPlusFilePrefix)+strlen(".logold")+strlen(pestr)+3;
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
      sprintf(fname, "%s.%s.logold.gz",  pathPlusFilePrefix, pestr);
      sprintf(dfname, "%s.%s.log.gz", pathPlusFilePrefix, pestr);
    } else {
      if (nonDeltaLog) {
	sprintf(fname, "%s.%s.log.gz", pathPlusFilePrefix,pestr);
      } else {
	sprintf(dfname, "%s.%s.log.gz", pathPlusFilePrefix, pestr);
      }
    }
  } else {
    if (deltaLog && nonDeltaLog) {
      sprintf(fname, "%s.%s.logold", pathPlusFilePrefix, pestr);
      sprintf(dfname, "%s.%s.log", pathPlusFilePrefix, pestr);
    } else {
      if (nonDeltaLog) {
	sprintf(fname, "%s.%s.log", pathPlusFilePrefix, pestr);
      } else {
	sprintf(dfname, "%s.%s.log", pathPlusFilePrefix, pestr);
      }
    }
  }
#else
  if (deltaLog && nonDeltaLog) {
    sprintf(fname, "%s.%s.logold", pathPlusFilePrefix, pestr);
    sprintf(dfname, "%s.%s.log", pathPlusFilePrefix, pestr);
  } else {
    if (nonDeltaLog) {
      sprintf(fname, "%s.%s.log", pathPlusFilePrefix, pestr);
    } else {
      sprintf(dfname, "%s.%s.log", pathPlusFilePrefix, pestr);
    }
  }
#endif
  fileCreated = true;
  delete[] pathPlusFilePrefix;
  openLog("w+");
  CLOSE_LOG 
}

void LogPool::createSts(const char *fix)
{
  CkAssert(CkMyPe() == 0);
  // create the sts file
  char *fname = new char[strlen(CkpvAccess(traceRoot))+strlen(fix)+strlen(".sts")+2];
  sprintf(fname, "%s%s.sts", CkpvAccess(traceRoot), fix);
  do
    {
      stsfp = fopen(fname, "w");
    } while (!stsfp && (errno == EINTR || errno == EMFILE));
  if(stsfp==0){
    CmiPrintf("Cannot open projections sts file for writing due to %s\n", strerror(errno));
    CmiAbort("Error!!\n");
  }
  delete[] fname;
}  

void LogPool::createRC()
{
  // create the projections rc file.
  fname = 
    new char[strlen(CkpvAccess(traceRoot))+strlen(".projrc")+1];
  sprintf(fname, "%s.projrc", CkpvAccess(traceRoot));
  do {
    rcfp = fopen(fname, "w");
  } while (!rcfp && (errno == EINTR || errno == EMFILE));
  if (rcfp==0) {
    CmiAbort("Cannot open projections configuration file for writing.\n");
  }
  delete[] fname;
}

LogPool::~LogPool() 
{
  if (writeData) {
    writeLog();
#if !CMK_TRACE_LOGFILE_NUM_CONTROL
    closeLog();
#endif
  }

#if CMK_BIGSIM_CHARM
  extern int correctTimeLog;
  if (correctTimeLog) {
    createFile("-bg");
    if (CkMyPe() == 0) {
      createSts("-bg");
    }
    writeHeader();
    if (CkMyPe() == 0) writeSts(NULL);
    postProcessLog();
  }
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
  createFile();
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
  int curPhase = 0;
  // **FIXME** - Should probably consider a more sophisticated bounds-based
  //   approach for selective writing instead of making multiple if-checks
  //   for every single event.
  for(UInt i=0; i<numEntries; i++) {
    if (!writedelta) {
      if (keepPhase == NULL) {
	// default case, when no phase selection is required.
	pool[i].pup(*p);
      } else {
	// **FIXME** Might be a good idea to create a "filler" event block for
	//   all the events taken out by phase filtering.
	if (pool[i].type == END_PHASE) {
	  // always write phase markers
	  pool[i].pup(*p);
	  curPhase++;
	} else if (pool[i].type == BEGIN_COMPUTATION ||
		   pool[i].type == END_COMPUTATION) {
	  // always write BEGIN and END COMPUTATION markers
	  pool[i].pup(*p);
	} else if (keepPhase[curPhase]) {
	  pool[i].pup(*p);
	}
      }
    }
    else {	// delta
      // **FIXME** Implement phase-selective writing for delta logs
      //   eventually
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
  delete [] keepPhase;
}

void LogPool::writeSts(void)
{
  // for whining compilers
  int i;
  // generate an automatic unique ID for each log
  fprintf(stsfp, "PROJECTIONS_ID %s\n", "");
  fprintf(stsfp, "VERSION %s\n", PROJECTION_VERSION);
  fprintf(stsfp, "TOTAL_PHASES %d\n", numPhases);
#if CMK_HAS_COUNTER_PAPI
  fprintf(stsfp, "TOTAL_PAPI_EVENTS %d\n", NUMPAPIEVENTS);
  // for now, use i, next time use papiEvents[i].
  // **CW** papi event names is a hack.
  char eventName[PAPI_MAX_STR_LEN];
  for (i=0;i<NUMPAPIEVENTS;i++) {
    PAPI_event_code_to_name(papiEvents[i], eventName);
    fprintf(stsfp, "PAPI_EVENT %d %s\n", i, eventName);
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

void LogPool::writeRC(void)
{
    //CkPrintf("write RC is being executed\n");
#ifdef PROJ_ANALYSIS  
    CkAssert(CkMyPe() == 0);
    fprintf(rcfp,"RC_GLOBAL_START_TIME %lld\n",
  	  (CMK_TYPEDEF_UINT8)(1.0e6*globalStartTime));
    fprintf(rcfp,"RC_GLOBAL_END_TIME   %lld\n",
  	  (CMK_TYPEDEF_UINT8)(1.0e6*globalEndTime));
    /* //Yanhua comment it because isOutlierAutomatic is not a variable in trace
    if (CkpvAccess(_trace)->isOutlierAutomatic()) {
      fprintf(rcfp,"RC_OUTLIER_FILTERED true\n");
    } else {
      fprintf(rcfp,"RC_OUTLIER_FILTERED false\n");
    }
    */
#endif //PROJ_ANALYSIS
  fclose(rcfp);
}


#if CMK_BIGSIM_CHARM
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
    hasFlushed = true;
    numEntries = 0;
    new (&pool[numEntries++]) LogEntry(writeTime, BEGIN_INTERRUPT);
    new (&pool[numEntries++]) LogEntry(TraceTimer(), END_INTERRUPT);
  }
}

void LogPool::add(UChar type, UShort mIdx, UShort eIdx,
		  double time, int event, int pe, int ml, CmiObjId *id, 
		  double recvT, double cpuT, int numPe)
{
  new (&pool[numEntries++])
    LogEntry(time, type, mIdx, eIdx, event, pe, ml, id, recvT, cpuT, numPe);
  if ((type == END_PHASE) || (type == END_COMPUTATION)) {
    numPhases++;
  }
  if(poolSize==numEntries) {
    flushLogBuffer();
#if CMK_BIGSIM_CHARM
    extern int correctTimeLog;
    if (correctTimeLog) CmiAbort("I/O interrupt!\n");
#endif
  }
#if CMK_BIGSIM_CHARM
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
#ifndef CMK_BIGSIM_CHARM
  new (&pool[numEntries++])
	LogEntry(time,type,funcID,lineNum,fileName);
  if(poolSize == numEntries){
    flushLogBuffer();
  }
#endif	
}


  
void LogPool::addMemoryUsage(unsigned char type,double time,double memUsage){
#ifndef CMK_BIGSIM_CHARM
  new (&pool[numEntries++])
	LogEntry(type,time,memUsage);
  if(poolSize == numEntries){
    flushLogBuffer();
  }
#endif	
	
}  



void LogPool::addUserSupplied(int data){
	// add an event
	add(USER_SUPPLIED, 0, 0, TraceTimer(), -1, -1, 0, 0, 0, 0, 0 );

	// set the user supplied value for the previously created event 
	pool[numEntries-1].setUserSuppliedData(data);
  }


void LogPool::addUserSuppliedNote(char *note){
	// add an event
	add(USER_SUPPLIED_NOTE, 0, 0, TraceTimer(), -1, -1, 0, 0, 0, 0, 0 );

	// set the user supplied note for the previously created event 
	pool[numEntries-1].setUserSuppliedNote(note);
  }

void LogPool::addUserSuppliedBracketedNote(char *note, int eventID, double bt, double et){
  //CkPrintf("LogPool::addUserSuppliedBracketedNote eventID=%d\n", eventID);
#ifndef CMK_BIGSIM_CHARM
#if MPI_TRACE_MACHINE_HACK
  //This part of code is used  to combine the contiguous
  //MPI_Test and MPI_Iprobe events to reduce the number of
  //entries
#define MPI_TEST_EVENT_ID 60
#define MPI_IPROBE_EVENT_ID 70 
  int lastEvent = pool[numEntries-1].event;
  if((eventID==MPI_TEST_EVENT_ID || eventID==MPI_IPROBE_EVENT_ID) && (eventID==lastEvent)){
    //just replace the endtime of last event
    //CkPrintf("addUserSuppliedBracketNote: for event %d\n", lastEvent);
    pool[numEntries].endTime = et;
  }else{
    new (&pool[numEntries++])
      LogEntry(bt, et, USER_SUPPLIED_BRACKETED_NOTE, note, eventID);
  }
#else
  new (&pool[numEntries++])
    LogEntry(bt, et, USER_SUPPLIED_BRACKETED_NOTE, note, eventID);
#endif
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
				   double recvT, int numPe, int *pelist)
{
  new (&pool[numEntries++])
    LogEntry(time, mIdx, eIdx, event, pe, ml, id, recvT, numPe, pelist);
  if(poolSize==numEntries) {
    flushLogBuffer();
  }
}

void LogPool::postProcessLog()
{
#if CMK_BIGSIM_CHARM
  bgUpdateProj(1);   // event type
#endif
}

void LogPool::modLastEntryTimestamp(double ts)
{
  pool[numEntries-1].time = ts;
  //pool[numEntries-1].cputime = ts;
}

// /** Constructor for a multicast log entry */
// 
//  THIS WAS MOVED TO trace-projections.h with the other constructors
// 
// LogEntry::LogEntry(double tm, unsigned short m, unsigned short e, int ev, int p,
// 	     int ml, CmiObjId *d, double rt, int numPe, int *pelist) 
// {
//     type = CREATION_MULTICAST; mIdx = m; eIdx = e; event = ev; pe = p; time = tm; msglen = ml;
//     if (d) id = *d; else {id.id[0]=id.id[1]=id.id[2]=id.id[3]=-1; };
//     recvTime = rt; 
//     numpes = numPe;
//     userSuppliedNote = NULL;
//     if (pelist != NULL) {
// 	pes = new int[numPe];
// 	for (int i=0; i<numPe; i++) {
// 	  pes[i] = pelist[i];
// 	}
//     } else {
// 	pes= NULL;
//     }
// }

//void LogEntry::addPapi(LONG_LONG_PAPI *papiVals)
//{
//#if CMK_HAS_COUNTER_PAPI
//   memcpy(papiValues, papiVals, sizeof(LONG_LONG_PAPI)*NUMPAPIEVENTS);
//#endif
//}



void LogEntry::pup(PUP::er &p)
{
  int i;
  CMK_TYPEDEF_UINT8 itime, iEndTime, irecvtime, icputime;
  char ret = '\n';

  p|type;
  if (p.isPacking()) itime = (CMK_TYPEDEF_UINT8)(1.0e6*time);
  if (p.isPacking()) iEndTime = (CMK_TYPEDEF_UINT8)(1.0e6*endTime);

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
      p|msglen; p|irecvtime; 
      p|id.id[0]; p|id.id[1]; p|id.id[2]; p|id.id[3];
      p|icputime;
#if CMK_HAS_COUNTER_PAPI
      //p|numPapiEvents;
      for (i=0; i<NUMPAPIEVENTS; i++) {
	// not yet!!!
	//	p|papiIDs[i]; 
	p|papiValues[i];
	
      }
#else
      //p|numPapiEvents;     // non papi version has value 0
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
      //p|numPapiEvents;
      for (i=0; i<NUMPAPIEVENTS; i++) {
	// not yet!!!
	//	p|papiIDs[i];
	p|papiValues[i];
      }
#else
      //p|numPapiEvents;  // non papi version has value 0
#endif
      if (p.isUnpacking()) cputime = icputime/1.0e6;
      break;
    case USER_SUPPLIED:
	  p|userSuppliedData;
	  p|itime;
	break;
    case USER_SUPPLIED_NOTE:
	  p|itime;
	  int length;
	  if (p.isPacking()) length = strlen(userSuppliedNote);
          p | length;
	  char space;
	  space = ' ';
          p | space;
	  if (p.isUnpacking()) {
	    userSuppliedNote = new char[length+1];
	    userSuppliedNote[length] = '\0';
	  }
   	  PUParray(p,userSuppliedNote, length);
	  break;
    case USER_SUPPLIED_BRACKETED_NOTE:
      //CkPrintf("Writting out a USER_SUPPLIED_BRACKETED_NOTE\n");
	  p|itime;
	  p|iEndTime;
	  p|event;
	  int length2;
	  if (p.isPacking()) length2 = strlen(userSuppliedNote);
          p | length2;
	  char space2;
	  space2 = ' ';
          p | space2;
	  if (p.isUnpacking()) {
	    userSuppliedNote = new char[length+1];
	    userSuppliedNote[length] = '\0';
	  }
   	  PUParray(p,userSuppliedNote, length2);
	  break;
    case MEMORY_USAGE_CURRENT:
      p | memUsage;
      p | itime;
	break;
    case CREATION:
      if (p.isPacking()) irecvtime = (CMK_TYPEDEF_UINT8)(1.0e6*recvTime);
      p|mIdx; p|eIdx; p|itime;
      p|event; p|pe; p|msglen; p|irecvtime;
      if (p.isUnpacking()) recvTime = irecvtime/1.0e6;
      break;
    case CREATION_BCAST:
      if (p.isPacking()) irecvtime = (CMK_TYPEDEF_UINT8)(1.0e6*recvTime);
      p|mIdx; p|eIdx; p|itime;
      p|event; p|pe; p|msglen; p|irecvtime; p|numpes;
      if (p.isUnpacking()) recvTime = irecvtime/1.0e6;
      break;
    case CREATION_MULTICAST:
      if (p.isPacking()) irecvtime = (CMK_TYPEDEF_UINT8)(1.0e6*recvTime);
      p|mIdx; p|eIdx; p|itime;
      p|event; p|pe; p|msglen; p|irecvtime; p|numpes;
      if (p.isUnpacking()) pes = numpes?new int[numpes]:NULL;
      for (i=0; i<numpes; i++) p|pes[i];
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
    case END_PHASE:
      p|eIdx; // FIXME: actually the phase ID
      p|itime;
      break;
    default:
      CmiError("***Internal Error*** Wierd Event %d.\n", type);
      break;
  }
  if (p.isUnpacking()) time = itime/1.0e6;
  p|ret;
}

TraceProjections::TraceProjections(char **argv): 
  curevent(0), inEntry(0), computationStarted(0), 
	converseExit(0), endTime(0.0), traceNestedEvents(0),
	currentPhaseID(0), lastPhaseEvent(NULL)
{
  //  CkPrintf("Trace projections dummy constructor called on %d\n",CkMyPe());

  if (CkpvAccess(traceOnPe) == 0) return;

  CtvInitialize(int,curThreadEvent);
  CkpvInitialize(CmiInt8, CtrLogBufSize);
  CkpvAccess(CtrLogBufSize) = DefaultLogBufSize;
  CtvAccess(curThreadEvent)=0;
  if (CmiGetArgLongDesc(argv,"+logsize",&CkpvAccess(CtrLogBufSize), 
		       "Log entries to buffer per I/O")) {
    if (CkMyPe() == 0) {
      CmiPrintf("Trace: logsize: %ld\n", CkpvAccess(CtrLogBufSize));
    }
  }
  checknested = 
    CmiGetArgFlagDesc(argv,"+checknested",
		      "check projections nest begin end execute events");
  traceNestedEvents = 
    CmiGetArgFlagDesc(argv,"+tracenested",
              "trace projections nest begin/end execute events");
  int binary = 
    CmiGetArgFlagDesc(argv,"+binary-trace",
		      "Write log files in binary format");

  CmiInt8 nSubdirs = 0;
  CmiGetArgLongDesc(argv,"+trace-subdirs", &nSubdirs, "Number of subdirectories into which traces will be written");


#if CMK_PROJECTIONS_USE_ZLIB
  int compressed = CmiGetArgFlagDesc(argv,"+gz-trace","Write log files pre-compressed with gzip");
#else
  // consume the flag so there's no confusing
  CmiGetArgFlagDesc(argv,"+gz-trace",
		    "Write log files pre-compressed with gzip");
  if(CkMyPe() == 0) CkPrintf("Warning> gz-trace is not supported on this machine!\n");
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
  _logPool->setNumSubdirs(nSubdirs);
  _logPool->setBinary(binary);
#if CMK_PROJECTIONS_USE_ZLIB
  _logPool->setCompressed(compressed);
#endif
  if (CkMyPe() == 0) {
    _logPool->createSts();
    _logPool->createRC();
  }
  funcCount=1;

#if CMK_HAS_COUNTER_PAPI
  // We initialize and create the event sets for use with PAPI here.
  int papiRetValue;
  if(CkMyRank()==0){
    papiRetValue = PAPI_library_init(PAPI_VER_CURRENT);
    if (papiRetValue != PAPI_VER_CURRENT) {
      CmiAbort("PAPI Library initialization failure!\n");
    }
#if CMK_SMP
    if(PAPI_thread_init(pthread_self) != PAPI_OK){
      CmiAbort("PAPI could not be initialized in SMP mode!\n");
    }
#endif
  }

#if CMK_SMP
  //PAPI_thread_init has to finish before calling PAPI_create_eventset
  CmiNodeAllBarrier();
#endif
  // PAPI 3 mandates the initialization of the set to PAPI_NULL
  papiEventSet = PAPI_NULL; 
  if (PAPI_create_eventset(&papiEventSet) != PAPI_OK) {
    CmiAbort("PAPI failed to create event set!\n");
  }
  papiRetValue = PAPI_add_events(papiEventSet, papiEvents, NUMPAPIEVENTS);
  if (papiRetValue != PAPI_OK) {
    if (papiRetValue == PAPI_ECNFLCT) {
      CmiAbort("PAPI events conflict! Please re-assign event types!\n");
    } else {
      CmiAbort("PAPI failed to add designated events!\n");
    }
  }
  memset(papiValues, 0, NUMPAPIEVENTS*sizeof(LONG_LONG_PAPI));
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

/** 
 * **IMPT NOTES**:
 *
 * This is called when Converse closes during ConverseCommonExit().
 * **FIXME**(?) - is this also exposed as a tracing-framework API call?
 *
 * Some programs bypass CkExit() (like NAMD, which eventually calls
 * ConverseExit()), modules like traces will have to pretend to shutdown
 * as if CkExit() was called but at the same time avoid making
 * subsequent CkExit() calls (which is usually required for allowing
 * other modules to shutdown).
 *
 * Note that we can only get here if CkExit() was not called, since the
 * trace module will un-register itself from TraceArray if it did.
 *
 */
void TraceProjections::traceClose(void)
{
#ifdef PROJ_ANALYSIS
  // CkPrintf("CkExit was not called on shutdown on [%d]\n", CkMyPe());

  // sets the flag that tells the code not to make the CkExit call later
  converseExit = 1;
  if (CkMyPe() == 0) {
    CProxy_TraceProjectionsBOC bocProxy(traceProjectionsGID);
    bocProxy.traceProjectionsParallelShutdown(-1);
  }
  if(CkMyRank() == CkMyNodeSize()){ //communication thread
    CkpvAccess(_trace)->endComputation();
    delete _logPool;              // will write
    // remove myself from traceArray so that no tracing will be called.
    CkpvAccess(_traces)->removeTrace(this);
  }
#else
  // we've already deleted the logpool, so multiple calls to traceClose
  // are tolerated.
  if (_logPool == NULL) {
    return;
  }
  if(CkMyPe()==0){
    _logPool->writeSts(this);
  }
  CkpvAccess(_trace)->endComputation();
  delete _logPool;              // will write
  // remove myself from traceArray so that no tracing will be called.
  CkpvAccess(_traces)->removeTrace(this);
#endif
}

/**
 *  **IMPT NOTES**:
 *
 *  This is meant to be called internally by the tracing framework.
 *
 */
void TraceProjections::closeTrace() {
  //  CkPrintf("Close Trace called on [%d]\n", CkMyPe());
  if (CkMyPe() == 0) {
    // CkPrintf("Pe 0 will now write sts and projrc files\n");
    _logPool->writeSts(this);
    _logPool->writeRC();
    // CkPrintf("Pe 0 has now written sts and projrc files\n");
  }
  delete _logPool;	 // will write logs to file
}

#if CMK_SMP_TRACE_COMMTHREAD
void TraceProjections::traceBeginOnCommThread()
{
  if (!computationStarted) return;
  _logPool->add(BEGIN_TRACE, 0, 0, TraceTimer(), curevent++, CmiNumPes()+CmiMyNode());
}

void TraceProjections::traceEndOnCommThread()
{
  _logPool->add(END_TRACE, 0, 0, TraceTimer(), curevent++, CmiNumPes()+CmiMyNode());
}
#endif

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

void TraceProjections::userSuppliedData(int d)
{
  if (!computationStarted) return;
  _logPool->addUserSupplied(d);
}

void TraceProjections::userSuppliedNote(char *note)
{
  if (!computationStarted) return;
  _logPool->addUserSuppliedNote(note);
}


void TraceProjections::userSuppliedBracketedNote(char *note, int eventID, double bt, double et)
{
  if (!computationStarted) return;
  _logPool->addUserSuppliedBracketedNote(note,  eventID,  bt, et);
}

void TraceProjections::memoryUsage(double m)
{
  if (!computationStarted) return;
  _logPool->addMemoryUsage(MEMORY_USAGE_CURRENT, TraceTimer(), m );
  
}


void TraceProjections::creation(envelope *e, int ep, int num)
{
  double curTime = TraceTimer();
  if (e == 0) {
    CtvAccess(curThreadEvent) = curevent;
    _logPool->add(CREATION, ForChareMsg, ep, curTime,
		  curevent++, CkMyPe(), 0, NULL, 0, 0.0);
  } else {
    int type=e->getMsgtype();
    e->setEvent(curevent);
    if (num > 1) {
      _logPool->add(CREATION_BCAST, type, ep, curTime,
		    curevent++, CkMyPe(), e->getTotalsize(), 
		    NULL, 0, 0.0, num);
    } else {
      _logPool->add(CREATION, type, ep, curTime,
		    curevent++, CkMyPe(), e->getTotalsize(), 
		    NULL, 0, 0.0);
    }
  }
}

void TraceProjections::creation(char *msg)
{
#if CMK_SMP_TRACE_COMMTHREAD
	//This function is only called from a comm thread
	//in SMP mode. So, it is possible the msg is not
	//a charm msg that contains an envelope, ep idx.
	envelope *e = (envelope *)msg;
	int ep = e->getEpIdx();
	int num = _entryTable.size();
	if(ep<num && ep>=0 && _entryTable[ep]->traceEnabled)
		creation(e, ep, 1);
#endif
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
    if ((log.type == CREATION) ||
	(log.type == CREATION_BCAST) ||
	(log.type == CREATION_MULTICAST)) {
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
  _logPool->addPapi(papiValues);
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
		  execEvent,CkMyPe(), 0, NULL, 0.0, TraceCpuTimer());
#if CMK_HAS_COUNTER_PAPI
    _logPool->addPapi(papiValues);
#endif
    inEntry = 1;
  } else {
    beginExecute(e->getEvent(),e->getMsgtype(),e->getEpIdx(),
		 e->getSrcPe(),e->getTotalsize());
  }
}

void TraceProjections::beginExecute(char *msg){
#if CMK_SMP_TRACE_COMMTHREAD
	//This function is called from comm thread in SMP mode
    envelope *e = (envelope *)msg;
    int num = _entryTable.size();
    int ep = e->getEpIdx();
    if(ep<0 || ep>=num) return;
    if(_entryTable[ep]->traceEnabled)
		beginExecute(e);
#endif
}

void TraceProjections::beginExecute(int event, int msgType, int ep, int srcPe,
				    int mlen, CmiObjId *idx)
{
  if (traceNestedEvents) {
    if (! nestedEvents.isEmpty()) {
      endExecuteLocal();
    }
    nestedEvents.enq(NestedEvent(event, msgType, ep, srcPe, mlen, idx));
  }
  beginExecuteLocal(event, msgType, ep, srcPe, mlen, idx);
}

void TraceProjections::changeLastEntryTimestamp(double ts)
{
  _logPool->modLastEntryTimestamp(ts);
}

void TraceProjections::beginExecuteLocal(int event, int msgType, int ep, int srcPe,
				    int mlen, CmiObjId *idx)
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
  _logPool->addPapi(papiValues);
#endif
  inEntry = 1;
}

void TraceProjections::endExecute(void)
{
  if (traceNestedEvents) nestedEvents.deq();
  endExecuteLocal();
  if (traceNestedEvents) {
    if (! nestedEvents.isEmpty()) {
      NestedEvent &ne = nestedEvents.peek();
      beginExecuteLocal(ne.event, ne.msgType, ne.ep, ne.srcPe, ne.ml, ne.idx);
    }
  }
}

void TraceProjections::endExecute(char *msg)
{
#if CMK_SMP_TRACE_COMMTHREAD
	//This function is called from comm thread in SMP mode
    envelope *e = (envelope *)msg;
    int num = _entryTable.size();
    int ep = e->getEpIdx();
    if(ep<0 || ep>=num) return;
    if(_entryTable[ep]->traceEnabled)
		endExecute();
#endif	
}

void TraceProjections::endExecuteLocal(void)
{
#if CMK_HAS_COUNTER_PAPI
  if (PAPI_read(papiEventSet, papiValues) != PAPI_OK) {
    CmiAbort("PAPI failed to read at end execute!\n");
  }
#endif
  if (checknested && !inEntry) CmiAbort("Nested EndExecute!\n");
  double cputime = TraceCpuTimer();
  if(execEp == (-1)) {
    _logPool->add(END_PROCESSING, 0, _threadEP, TraceTimer(),
		  execEvent, CkMyPe(), 0, NULL, 0.0, cputime);
  } else {
    _logPool->add(END_PROCESSING, 0, execEp, TraceTimer(),
		  execEvent, execPe, 0, NULL, 0.0, cputime);
  }
#if CMK_HAS_COUNTER_PAPI
  _logPool->addPapi(papiValues);
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
  _logPool->add(MESSAGE_RECV, msgType, ep, TraceTimer(),
		curevent++, e->getSrcPe(), e->getTotalsize());
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

  // Executes the callback function provided by the machine
  // layer. This is the proper method to register user events in a
  // machine layer because projections is a charm++ module.
  if (CkpvAccess(traceOnPe) != 0) {
    void (*ptr)() = registerMachineUserEvents();
    if (ptr != NULL) {
      ptr();
    }
  }
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
  endTime = TraceTimer();
  _logPool->add(END_COMPUTATION, 0, 0, endTime, -1, -1);
  /*
  CkPrintf("End Computation [%d] records time as %lf\n", CkMyPe(), 
  	   endTime*1e06);
  */
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

void TraceProjections::endPhase() {
  double currentPhaseTime = TraceTimer();
  if (lastPhaseEvent != NULL) {
  } else {
    if (_logPool->pool != NULL) {
      // assumed to be BEGIN_COMPUTATION
    } else {
      CkPrintf("[%d] Warning: End Phase encountered in an empty log. Inserting BEGIN_COMPUTATION event\n", CkMyPe());
      _logPool->add(BEGIN_COMPUTATION, 0, 0, currentPhaseTime, -1, -1);
    }
  }

  /* Insert endPhase event here. */
  /* FIXME: Format should be TYPE, PHASE#, TimeStamp, [StartTime] */
  /*        We currently "borrow" from the standard add() method. */
  /*        It should really be its own add() method.             */
  /* NOTE: assignment to lastPhaseEvent is "pre-emptive".         */
  lastPhaseEvent = &(_logPool->pool[_logPool->numEntries]);
  _logPool->add(END_PHASE, 0, currentPhaseID, currentPhaseTime, -1, CkMyPe());
  currentPhaseID++;
}

#ifdef PROJ_ANALYSIS
// ***** FROM HERE, ALL BOC-BASED FUNCTIONALITY IS DEFINED *******


// ***@@@@ REGISTRATION FUNCTIONS/METHODS @@@@***

void registerOutlierReduction() {
  outlierReductionType =
    CkReduction::addReducer(outlierReduction);
  minMaxReductionType =
    CkReduction::addReducer(minMaxReduction);
}

/**
 * **IMPT NOTES**:
 *
 * This is the C++ code that is registered to be activated at module
 * shutdown. This is called exactly once on processor 0. Module shutdown
 * is initiated as a result of a CkExit() call by the application code
 * 
 * The exit function must ultimately call CkExit() again to
 * so that other module exit functions may proceed after this module is
 * done.
 *
 */
// FIXME: WHY extern "C"???
extern "C" void TraceProjectionsExitHandler()
{
#if CMK_TRACE_ENABLED
  // CkPrintf("[%d] TraceProjectionsExitHandler called!\n", CkMyPe());
  CProxy_TraceProjectionsBOC bocProxy(traceProjectionsGID);
  bocProxy.traceProjectionsParallelShutdown(CkMyPe());
#else
  CkExit();
#endif
}

// This is called once on each processor but the idiom of use appears
// to be to only have processor 0 register the function.
//
// See initnode in trace-projections.ci
void initTraceProjectionsBOC()
{
  // CkPrintf("[%d] Trace Projections initialization called!\n", CkMyPe());
#ifdef __BIGSIM__
  if (BgNodeRank() == 0) {
#else
    if (CkMyRank() == 0) {
#endif
      registerExitFn(TraceProjectionsExitHandler);
    }
#if 0
  } // this is so indentation does not get messed up
#endif
}

// mainchare for trace-projections BOC-operations. 
// Instantiated at processor 0 and ONLY resides on processor 0 for the 
// rest of its life.
//
// Responsible for:
//   1. Handling commandline arguments
//   2. Creating any objects required for proper BOC operations.
//
TraceProjectionsInit::TraceProjectionsInit(CkArgMsg *msg) {
  /** Options for Outlier Analysis */
  // defaults. Things will change with support for interactive analysis.
  bool findOutliers = false;
  bool outlierAutomatic = true;
  int numKSeeds = 10; 

  int peNumKeep = CkNumPes();  // used as a default
  double entryThreshold = 0.0;
  bool outlierUsePhases = false;
  if (outlierAutomatic) {
    CmiGetArgIntDesc(msg->argv, "+outlierNumSeeds", &numKSeeds,
		     "Number of cluster seeds to apply at outlier analysis.");
    CmiGetArgIntDesc(msg->argv, "+outlierPeNumKeep", 
		     &peNumKeep, "Number of Processors to retain data");
    CmiGetArgDoubleDesc(msg->argv, "+outlierEpThresh", &entryThreshold,
			"Minimum significance of entry points to be considered for clustering (%).");
    findOutliers =
      CmiGetArgFlagDesc(msg->argv,"+outlier", "Find Outliers.");
    outlierUsePhases = 
      CmiGetArgFlagDesc(msg->argv,"+outlierUsePhases",
			"Apply automatic outlier analysis to any available phases.");
    if (outlierUsePhases) {
      // if the user wants to use an outlier feature, it is assumed outlier
      //    analysis is desired.
      findOutliers = true;
    }
  }
  bool findStartTime = (CmiTimerAbsolute()==1);
  traceProjectionsGID = CProxy_TraceProjectionsBOC::ckNew(findOutliers, findStartTime);
  if (findOutliers) {
    kMeansGID = CProxy_KMeansBOC::ckNew(outlierAutomatic,
					numKSeeds,
					peNumKeep,
					entryThreshold,
					outlierUsePhases);
  }
}

// Called on every processor.
void TraceProjectionsBOC::traceProjectionsParallelShutdown(int pe) {
  //CmiPrintf("[%d] traceProjectionsParallelShutdown called from . \n", CkMyPe(), pe);
  endPe = pe;                // the pe that starts CkExit()
  if (CkMyPe() == 0) {
    analysisStartTime = CmiWallTimer();
  }
  CkpvAccess(_trace)->endComputation();
  // no more tracing for projections on this processor after this point. 
  // Note that clear must be called after remove, or bad things will happen.
  CkpvAccess(_traces)->removeTrace(CkpvAccess(_trace));
  CkpvAccess(_traces)->clearTrace();

  // From this point, we start multiple chains of reductions and broadcasts to
  // perform final online analysis activities.

  // Start all parallel operations at once. 
  //   These MUST NOT modify base performance data in LogPool. If they must,
  //   then the parallel operations must be phased (and this code to be
  //   restructured as necessary)
  CProxy_KMeansBOC kMeansProxy(kMeansGID);
  CProxy_TraceProjectionsBOC bocProxy(traceProjectionsGID);
  if (findOutliers) {
    parModulesRemaining++;
    kMeansProxy[CkMyPe()].startKMeansAnalysis();
  }
  parModulesRemaining++;
  if (findStartTime) 
  bocProxy[CkMyPe()].startTimeAnalysis();
  else
  bocProxy[CkMyPe()].startEndTimeAnalysis();
}

// Called on each processor
void KMeansBOC::startKMeansAnalysis() {
  // Initialize all necessary structures
  LogPool *pool = CkpvAccess(_trace)->_logPool;

 if(CkMyPe()==0)     CkPrintf("[%d] KMeansBOC::startKMeansAnalysis time=\t%g\n", CkMyPe(), CkWallTimer() );
  int flushInt = 0;
  if (pool->hasFlushed) {
    flushInt = 1;
  }
  
  CkCallback cb(CkIndex_KMeansBOC::flushCheck(NULL), 
		0, thisProxy);
  contribute(sizeof(int), &flushInt, CkReduction::logical_or, cb);  
}

// Called on processor 0
void KMeansBOC::flushCheck(CkReductionMsg *msg) {
  int someFlush = *((int *)msg->getData());

  // if(CkMyPe()==0) CkPrintf("[%d] KMeansBOC::flushCheck time=\t%g\n", CkMyPe(), CkWallTimer() );
  
  if (someFlush == 0) {
    // Data intact proceed with KMeans analysis
    CProxy_KMeansBOC kMeansProxy(kMeansGID);
    kMeansProxy.flushCheckDone();
  } else {
    // Some processor had flushed it data at some point, abandon KMeans
    CkPrintf("Warning: Some processor has flushed its data. No KMeans will be conducted\n");
    // terminate KMeans
    CProxy_TraceProjectionsBOC bocProxy(traceProjectionsGID);
    bocProxy[0].kMeansDone();
  }
}

// Called on each processor
void KMeansBOC::flushCheckDone() {
  // **FIXME** - more flexible metric collection scheme may be necessary
  //   in the future for production use.
  LogPool *pool = CkpvAccess(_trace)->_logPool;

  // if(CkMyPe()==0)     CkPrintf("[%d] KMeansBOC::flushCheckDone time=\t%g\n", CkMyPe(), CkWallTimer() );

  numEntryMethods = _entryTable.size();
  numMetrics = numEntryMethods + 2; // EPtime + idle and overhead

  // maintained across phases
  markedBegin = false;
  markedIdle = false;
  beginBlockTime = 0.0;
  beginIdleBlockTime = 0.0;
  lastBeginEPIdx = -1; // none

  lastPhaseIdx = 0;
  currentExecTimes = NULL;
  currentPhase = 0;
  selected = false;

  pool->initializePhases();

  // incoming K Seeds and the per-phase filter
  incKSeeds = new double[numK*numMetrics];
  keepMetric = new bool[numMetrics];

  //  Something wrong when call thisProxy[CkMyPe()].getNextPhaseMetrics() !??!
  //  CProxy_KMeansBOC kMeansProxy(kMeansGID);
  //  kMeansProxy[CkMyPe()].getNextPhaseMetrics();
  thisProxy[CkMyPe()].getNextPhaseMetrics();
}

// Called on each processor.
void KMeansBOC::getNextPhaseMetrics() {
  // Assumes the presence of the complete logs on this processor.
  // Assumes first event is always BEGIN_COMPUTATION
  // Assumes each processor sees the same number of phases.
  //
  // In this code, we collect performance data for this processor.
  // All times are in seconds.

  // if(CkMyPe()==0)    CkPrintf("[%d] KMeansBOC::getNextPhaseMetrics time=\t%g\n", CkMyPe(), CkWallTimer() );  

  if (usePhases) {
    DEBUGF("[%d] Using Phases\n", CkMyPe());
  } else {
    DEBUGF("[%d] NOT using Phases\n", CkMyPe());
  }
  
  if (currentExecTimes != NULL) {
    delete [] currentExecTimes;
  }
  currentExecTimes = new double[numMetrics];
  for (int i=0; i<numMetrics; i++) {
    currentExecTimes[i] = 0.0;
  }

  int numEventMethods = _entryTable.size();
  LogPool *pool = CkpvAccess(_trace)->_logPool;
  
  CkAssert(pool->numEntries > lastPhaseIdx);
  double totalPhaseTime = 0.0;
  double totalActiveTime = 0.0; // entry method + idle

  for (int i=lastPhaseIdx; i<pool->numEntries; i++) {
    if (pool->pool[i].type == BEGIN_PROCESSING) {
      // check pairing
      if (!markedBegin) {
	markedBegin = true;
      }
      beginBlockTime = pool->pool[i].time;
      lastBeginEPIdx = pool->pool[i].eIdx;
    } else if (pool->pool[i].type == END_PROCESSING) {
      // check pairing
      // if End without a begin, just ignore
      //   this event. If a phase-boundary is crossed, the Begin
      //   event would be maintained in beginBlockTime, so it is 
      //   not a problem.
      if (markedBegin) {
	markedBegin = false;
	if (pool->pool[i].event < 0)
	{
	  // ignore dummy events. **FIXME** as they have no eIdx?
	  continue;
	}
	currentExecTimes[pool->pool[i].eIdx] += 
	  pool->pool[i].time - beginBlockTime;
	totalActiveTime += pool->pool[i].time - beginBlockTime;
	lastBeginEPIdx = -1;
      }
    } else if (pool->pool[i].type == BEGIN_IDLE) {
      // check pairing
      if (!markedIdle) {
	markedIdle = true;
      }
      beginIdleBlockTime = pool->pool[i].time;
    } else if (pool->pool[i].type == END_IDLE) {
      // check pairing
      if (markedIdle) {
	markedIdle = false;
	currentExecTimes[numEventMethods] += 
	  pool->pool[i].time - beginIdleBlockTime;
	totalActiveTime += pool->pool[i].time - beginIdleBlockTime;
      }
    } else if (pool->pool[i].type == END_PHASE) {
      // ignored when not using phases
      if (usePhases) {
	// when we've not visited this node before
	if (i != lastPhaseIdx) { 
	  totalPhaseTime = 
	    pool->pool[i].time - pool->pool[lastPhaseIdx].time;
	  // it is important that proper accounting of time take place here.
	  // Note that END_PHASE events inevitably occur in the context of
	  //   some entry method by the way the tracing API is designed.
	  if (markedBegin) {
	    CkAssert(lastBeginEPIdx >= 0);
	    currentExecTimes[lastBeginEPIdx] += 
	      pool->pool[i].time - beginBlockTime;
	    totalActiveTime += pool->pool[i].time - beginBlockTime;
	    // this is so the remainder contributes to the next phase
	    beginBlockTime = pool->pool[i].time;
	  }
	  // The following is unlikely, but stranger things have happened.
	  if (markedIdle) {
	    currentExecTimes[numEventMethods] +=
	      pool->pool[i].time - beginIdleBlockTime;
	    totalActiveTime += pool->pool[i].time - beginIdleBlockTime;
	    // this is so the remainder contributes to the next phase
	    beginIdleBlockTime = pool->pool[i].time;
	  }
	  if (totalActiveTime <= totalPhaseTime) {
	    currentExecTimes[numEventMethods+1] = 
	      totalPhaseTime - totalActiveTime;
	  } else {
	    currentExecTimes[numEventMethods+1] = 0.0;
	    CkPrintf("[%d] Warning: Overhead found to be negative for Phase %d!\n",
		     CkMyPe(), currentPhase);
	  }
	  collectKMeansData();
	  // end the loop (and method) and defer the work till the next call
	  lastPhaseIdx = i;
	  break; 
	}
      }
    } else if (pool->pool[i].type == END_COMPUTATION) {
      if (markedBegin) {
	CkAssert(lastBeginEPIdx >= 0);
	currentExecTimes[lastBeginEPIdx] += 
	  pool->pool[i].time - beginBlockTime;
	totalActiveTime += pool->pool[i].time - beginBlockTime;
      }
      if (markedIdle) {
	currentExecTimes[numEventMethods] +=
	  pool->pool[i].time - beginIdleBlockTime;
	totalActiveTime += pool->pool[i].time - beginIdleBlockTime;
      }
      totalPhaseTime = 
	pool->pool[i].time - pool->pool[lastPhaseIdx].time;
      if (totalActiveTime <= totalPhaseTime) {
	currentExecTimes[numEventMethods+1] = totalPhaseTime - totalActiveTime;
      } else {
	currentExecTimes[numEventMethods+1] = 0.0;
	CkPrintf("[%d] Warning: Overhead found to be negative!\n",
		 CkMyPe());
      }
      collectKMeansData();
    }
  }
}

/**
 *  Through a reduction, collectKMeansData aggregates each processors' data
 *  in order for global properties to be determined:
 *  
 *  1. min & max to determine normalization factors.
 *  2. sum to determine global EP averages for possible metric reduction
 *       through thresholding.
 *  3. sum of squares to compute stddev which may be useful in the future.
 *
 *  collectKMeansData will also keep the processor's data for the current
 *    phase so that it may be normalized and worked on subsequently.
 *
 **/
void KMeansBOC::collectKMeansData() {
  int minOffset = numMetrics;
  int maxOffset = 2*numMetrics;
  int sosOffset = 3*numMetrics; // sos = Sum Of Squares

  // if(CkMyPe()==0)    CkPrintf("[%d] KMeansBOC::collectKMeansData time=\tg\n", CkMyPe(), CkWallTimer() );

  double *reductionMsg = new double[numMetrics*4];

  for (int i=0; i<numMetrics; i++) {
    reductionMsg[i] = currentExecTimes[i];
    // duplicate the event times for max and min sections of the reduction
    reductionMsg[minOffset + i] = currentExecTimes[i];
    reductionMsg[maxOffset + i] = currentExecTimes[i];
    // compute squares
    reductionMsg[sosOffset + i] = currentExecTimes[i]*currentExecTimes[i];
  }

  CkCallback cb(CkIndex_KMeansBOC::globalMetricRefinement(NULL), 
		0, thisProxy);
  contribute((numMetrics*4)*sizeof(double), reductionMsg, 
	     outlierReductionType, cb);  
}

// The purpose is mainly to initialize the k seeds and generate
//   normalization parameters for each of the metrics. The k seeds
//   and normalization parameters are broadcast to all processors.
//
// Called on processor 0
void KMeansBOC::globalMetricRefinement(CkReductionMsg *msg) {
  CkAssert(CkMyPe() == 0);
  
  // if(CkMyPe()==0)    CkPrintf("[%d] KMeansBOC::globalMetricRefinement time=\t%g\n", CkMyPe(), CkWallTimer() );

  int sumOffset = 0;
  int minOffset = numMetrics;
  int maxOffset = 2*numMetrics;
  int sosOffset = 3*numMetrics; // sos = Sum Of Squares

  // calculate statistics & boundaries for the k seeds for clustering
  KMeansStatsMessage *outmsg = 
    new (numMetrics, numK*numMetrics, numMetrics*4) KMeansStatsMessage;
  outmsg->numMetrics = numMetrics;
  outmsg->numKPos = numK*numMetrics;
  outmsg->numStats = numMetrics*4;

  // Sum | Min | Max | Sum of Squares
  double *totalExecTimes = (double *)msg->getData();
  double totalTime = 0.0;

  for (int i=0; i<numMetrics; i++) {
    DEBUGN("%lf\n", totalExecTimes[i]);
    totalTime += totalExecTimes[i];

    // calculate event mean over all processors
    outmsg->stats[sumOffset + i] = totalExecTimes[sumOffset + i]/CkNumPes();

    // get the ranges and offsets of each metric. With this, we get
    //   normalization factors that can be sent back to each processor to
    //   be used as necessary. We reuse max for range. Min remains the offset.
    outmsg->stats[minOffset + i] = totalExecTimes[minOffset + i];
    outmsg->stats[maxOffset + i] = totalExecTimes[maxOffset + i] -
      totalExecTimes[minOffset + i];
    
    // calculate stddev (using biased variance)
    outmsg->stats[sosOffset + i] = 
      sqrt((totalExecTimes[sosOffset + i] - 
	    2*(outmsg->stats[i])*totalExecTimes[i] +
	    (outmsg->stats[i])*(outmsg->stats[i])*CkNumPes())/
	   CkNumPes());
  }

  for (int i=0; i<numMetrics; i++) {
    // 1) if the proportion of the max value of the entry method relative to
    //   the average time taken over all entry methods across all processors
    //   is greater than the stipulated percentage threshold ...; AND
    // 2) if the range of values are non-zero.
    //
    // The current assumption is totalTime > 0 (what program has zero total
    //   time from all work?)
    keepMetric[i] = ((totalExecTimes[maxOffset + i]/(totalTime/CkNumPes()) >=
		     entryThreshold) &&
      (totalExecTimes[maxOffset + i] > totalExecTimes[minOffset + i]));
    if (keepMetric[i]) {
      DEBUGF("[%d] Keep EP %d | Max = %lf | Avg Tot = %lf\n", CkMyPe(), i,
	     totalExecTimes[maxOffset + i], totalTime/CkNumPes());
    } else {
      DEBUGN("[%d] DO NOT Keep EP %d\n", CkMyPe(), i);
    }
    outmsg->filter[i] = keepMetric[i];
  }

  delete msg;
  
  // initialize k seeds for this phase
  kSeeds = new double[numK*numMetrics];

  numKReported = 0;
  kNumMembers = new int[numK];

  // Randomly select k processors' metric vectors for the k seeds
  //  srand((unsigned)(CmiWallTimer()*1.0e06));
  srand(11337); // for debugging purposes
  for (int k=0; k<numK; k++) {
    DEBUGF("Seed %d | ", k);
    for (int m=0; m<numMetrics; m++) {
      double factor = totalExecTimes[maxOffset + m] - 
	totalExecTimes[minOffset + m];
      // "uniform" distribution, scaled according to the normalization
      //   factors
      //      kSeeds[numMetrics*k + m] = ((1.0*(k+1))/numK)*factor;
      // Random distribution.
      kSeeds[numMetrics*k + m] =
	((rand()*1.0)/RAND_MAX)*factor;
      if (keepMetric[m]) {
	DEBUGF("[%d|%lf] ", m, kSeeds[numMetrics*k + m]);
      }
      outmsg->kSeedsPos[numMetrics*k + m] = kSeeds[numMetrics*k + m];
    }
    DEBUGF("\n");
    kNumMembers[k] = 0;
  }

  // broadcast statistical values to all processors for cluster discovery
  thisProxy.findInitialClusters(outmsg);
}



// Called on each processor.
void KMeansBOC::findInitialClusters(KMeansStatsMessage *msg) {

 if(CkMyPe()==0)    CkPrintf("[%d] KMeansBOC::findInitialClusters time=\t%g\n", CkMyPe(), CkWallTimer() );

  phaseIter = 0;

  // Get info from stats message
  CkAssert(numMetrics == msg->numMetrics);
  for (int i=0; i<numMetrics; i++) {
    keepMetric[i] = msg->filter[i];
  }

  // Normalize data on local processor.
  // **CWL** See my thesis for detailed discussion of normalization of
  //    performance data.
  // **NOTE** This might change if we want to send data based on the filter
  //   instead of all the data.
  CkAssert(numMetrics*4 == msg->numStats);
  for (int i=0; i<numMetrics; i++) {
    currentExecTimes[i] -= msg->stats[numMetrics + i];  // take offset
    // **CWL** We do not normalize the range. Entry methods that exhibit
    //   large absolute timing variations should be allowed to contribute
    //   more to the Euclidean distance measure!
    // currentExecTimes[i] /= msg->stats[2*numMetrics + i];
  }

  // **NOTE** This might change if we want to send data based on the filter
  //   instead of all the data.
  CkAssert(numK*numMetrics == msg->numKPos);
  for (int i=0; i<msg->numKPos; i++) {
    incKSeeds[i] = msg->kSeedsPos[i];
  }

  // Decide which KSeed this processor belongs to.
  minDistance = calculateDistance(0);
  DEBUGN("[%d] Phase %d Iter %d | Distance from 0 = %lf \n", CkMyPe(), 
	   currentPhase, phaseIter, minDistance);
  minK = 0;
  for (int i=1; i<numK; i++) {
    double distance = calculateDistance(i);
    DEBUGN("[%d] Phase %d Iter %d | Distance from %d = %lf \n", CkMyPe(), 
	     currentPhase, phaseIter, i, distance);
    if (distance < minDistance) {
      minDistance = distance;
      minK = i;
    }
  }

  // Set up a reduction with the modification vector to the root (0).
  //
  // The modification vector sends a negative value for each metric
  //   for the K this processor no longer belongs to and a positive
  //   value to the K the processor now belongs. In addition, a -1.0
  //   is sent to the K it is leaving and a +1.0 to the K it is 
  //   joining.
  //
  // The processor must still contribute a "zero returns" even if
  //   nothing changes. This will be the basis for determine
  //   convergence at the root.
  //
  // The addtional +1 is meant for the count-change that must be
  //   maintained for the special cases at the root when some K
  //   may be deprived of all processor points or go from 0 to a
  //   positive number of processors (see later comments).
  double *modVector = new double[numK*(numMetrics+1)];
  for (int i=0; i<numK; i++) {
    for (int j=0; j<numMetrics+1; j++) {
      modVector[i*(numMetrics+1) + j] = 0.0;
    }
  }
  for (int i=0; i<numMetrics; i++) {
    // for this initialization, only positive values need be sent.
    modVector[minK*(numMetrics+1) + i] = currentExecTimes[i];
  }
  modVector[minK*(numMetrics+1)+numMetrics] = 1.0;

  CkCallback cb(CkIndex_KMeansBOC::updateKSeeds(NULL), 
		0, thisProxy);
  contribute(numK*(numMetrics+1)*sizeof(double), modVector, 
	     CkReduction::sum_double, cb);  
}

double KMeansBOC::calculateDistance(int k) {
  double ret = 0.0;
  for (int i=0; i<numMetrics; i++) {
    if (keepMetric[i]) {
      DEBUGN("[%d] Phase %d Iter %d Metric %d Exec %lf Seed %lf \n", 
	     CkMyPe(), currentPhase, phaseIter, i,
	       currentExecTimes[i], incKSeeds[k*numMetrics + i]);
      ret += pow(currentExecTimes[i] - incKSeeds[k*numMetrics + i], 2.0);
    }
  }
  return sqrt(ret);
}

void KMeansBOC::updateKSeeds(CkReductionMsg *msg) {
  CkAssert(CkMyPe() == 0);

  // if(CkMyPe()==0)    CkPrintf("[%d] KMeansBOC::updateKSeeds time=\t%g\n", CkMyPe(), CkWallTimer() );

  double *modVector = (double *)msg->getData();
  // sanity check
  CkAssert(numK*(numMetrics+1)*sizeof(double) == msg->getSize());

  // A quick convergence test.
  bool hasChanges = false;
  for (int i=0; i<numK; i++) {
    hasChanges = hasChanges || 
      (modVector[i*(numMetrics+1) + numMetrics] != 0.0);
  }
  if (!hasChanges) {
    delete msg;
    findRepresentatives();
  } else {
    int overallChange = 0;
    for (int i=0; i<numK; i++) {
      int change = (int)modVector[i*(numMetrics+1) + numMetrics];
      if (change != 0) {
	overallChange += change;
	// modify the k seeds based on the modification vectors coming in
	//
	// If a seed initially has no members, its contents do not matter and
	//   is simply set to the average of the incoming vector.
	// If the change causes a seed to lose all its members, do nothing.
	//   Its last-known location is kept to allow it to re-capture
	//   membership at the next iteration rather than apply the last
	//   changes (which snaps the point unnaturally to 0,0).
	// Otherwise, apply the appropriate vector changes.
	CkAssert((kNumMembers[i] + change >= 0) &&
		 (kNumMembers[i] + change <= CkNumPes()));
	if (kNumMembers[i] == 0) {
	  CkAssert(change > 0);
	  for (int j=0; j<numMetrics; j++) {
	    kSeeds[i*numMetrics + j] = modVector[i*(numMetrics+1) + j]/change;
	  }
	} else if (kNumMembers[i] + change == 0) {
	  // do nothing.
	} else {
	  for (int j=0; j<numMetrics; j++) {
	    kSeeds[i*numMetrics + j] *= kNumMembers[i];
	    kSeeds[i*numMetrics + j] += modVector[i*(numMetrics+1) + j];
	    kSeeds[i*numMetrics + j] /= kNumMembers[i] + change;
	  }
	}
	kNumMembers[i] += change;
      }
      DEBUGN("[%d] Phase %d Iter %d K = %d Membership Count = %d\n",
	     CkMyPe(), currentPhase, phaseIter, i, kNumMembers[i]);
    }
    delete msg;

    // broadcast the new seed locations.
    KSeedsMessage *outmsg = new (numK*numMetrics) KSeedsMessage;
    outmsg->numKPos = numK*numMetrics;
    for (int i=0; i<numK*numMetrics; i++) {
      outmsg->kSeedsPos[i] = kSeeds[i];
    }

    thisProxy.updateSeedMembership(outmsg);
  }
}

// Called on all processors
void KMeansBOC::updateSeedMembership(KSeedsMessage *msg) {

  // if(CkMyPe()==0)    CkPrintf("[%d] KMeansBOC::updateSeedMembership time=\t%g\n", CkMyPe(), CkWallTimer() );

  phaseIter++;

  // **NOTE** This might change if we want to send data based on the filter
  //   instead of all the data.
  CkAssert(numK*numMetrics == msg->numKPos);
  for (int i=0; i<msg->numKPos; i++) {
    incKSeeds[i] = msg->kSeedsPos[i];
  }

  // Decide which KSeed this processor belongs to.
  lastMinK = minK;
  minDistance = calculateDistance(0);
  DEBUGN("[%d] Phase %d Iter %d | Distance from 0 = %lf \n", CkMyPe(), 
	 currentPhase, phaseIter, minDistance);

  minK = 0;
  for (int i=1; i<numK; i++) {
    double distance = calculateDistance(i);
    DEBUGN("[%d] Phase %d Iter %d | Distance from %d = %lf \n", CkMyPe(), 
	   currentPhase, phaseIter, i, distance);
    if (distance < minDistance) {
      minDistance = distance;
      minK = i;
    }
  }

  double *modVector = new double[numK*(numMetrics+1)];
  for (int i=0; i<numK; i++) {
    for (int j=0; j<numMetrics+1; j++) {
      modVector[i*(numMetrics+1) + j] = 0.0;
    }
  }

  if (minK != lastMinK) {
    for (int i=0; i<numMetrics; i++) {
      modVector[minK*(numMetrics+1) + i] = currentExecTimes[i];
      modVector[lastMinK*(numMetrics+1) + i] = -currentExecTimes[i];
    }
    modVector[minK*(numMetrics+1)+numMetrics] = 1.0;
    modVector[lastMinK*(numMetrics+1)+numMetrics] = -1.0;
  }

  CkCallback cb(CkIndex_KMeansBOC::updateKSeeds(NULL), 
		0, thisProxy);
  contribute(numK*(numMetrics+1)*sizeof(double), modVector, 
	     CkReduction::sum_double, cb);  
}

void KMeansBOC::findRepresentatives() {

  // if(CkMyPe()==0)    CkPrintf("[%d] KMeansBOC::findRepresentatives time=\t%g\n", CkMyPe(), CkWallTimer() );

  int numNonEmptyClusters = 0;
  for (int i=0; i<numK; i++) {
    if (kNumMembers[i] > 0) {
      numNonEmptyClusters++;
    }
  }

  int numRepresentatives = peNumKeep;
  // **FIXME**
  // This is fairly arbitrary. Next time, choose the centers of the top
  //   largest clusters.
  if (numRepresentatives < numNonEmptyClusters) {
    numRepresentatives = numNonEmptyClusters;
  }

  int slotsRemaining = numRepresentatives;

  DEBUGF("Slots = %d | Non-empty = %d \n", slotsRemaining, 
	 numNonEmptyClusters);

  // determine how many exemplars to select per cluster. Currently
  //   hardcoded to 1. Future challenge is to decide on other numbers
  //   or proportionality.
  //
  int exemplarsPerCluster = 1;
  slotsRemaining -= exemplarsPerCluster*numNonEmptyClusters;

  int numCandidateOutliers = CkNumPes() - 
    exemplarsPerCluster*numNonEmptyClusters;

  double *remainders = new double[numK];
  int *assigned = new int[numK];
  exemplarChoicesLeft = new int[numK];
  outlierChoicesLeft = new int[numK];

  for (int i=0; i<numK; i++) {
    assigned[i] = 0;
    remainders[i] = 
      (kNumMembers[i] - exemplarsPerCluster*numNonEmptyClusters) *
      slotsRemaining / numCandidateOutliers;
    if (remainders[i] >= 0.0) {
      assigned[i] = (int)floor(remainders[i]);
      remainders[i] -= assigned[i];
    } else {
      remainders[i] = 0.0;
    }
  }

  for (int i=0; i<numK; i++) {
    slotsRemaining -= assigned[i];
  }
  CkAssert(slotsRemaining >= 0);

  // find clusters to assign the loose slots to, in order of
  // remainder proportion
  while (slotsRemaining > 0) {
    double max = 0.0;
    int winner = 0;
    for (int i=0; i<numK; i++) {
      if (remainders[i] > max) {
	max = remainders[i];
	winner = i;
      }
    }
    assigned[winner]++;
    remainders[winner] = 0.0;
    slotsRemaining--;
  }

  // set up how many reduction cycles of min/max we need to conduct to
  // select the representatives.
  numSelectionIter = exemplarsPerCluster;
  for (int i=0; i<numK; i++) {
    if (assigned[i] > numSelectionIter) {
      numSelectionIter = assigned[i];
    }
  }
  DEBUGF("Selection Iterations = %d\n", numSelectionIter);

  for (int i=0; i<numK; i++) {
    if (kNumMembers[i] > 0) {
      exemplarChoicesLeft[i] = exemplarsPerCluster;
      outlierChoicesLeft[i] = assigned[i];
    } else {
      exemplarChoicesLeft[i] = 0;
      outlierChoicesLeft[i] = 0;
    }
    DEBUGF("%d | Exemplar = %d | Outlier = %d\n", i, exemplarChoicesLeft[i],
	   outlierChoicesLeft[i]);
  }

  delete [] assigned;
  delete [] remainders;

  // send out first broadcast
  KSelectionMessage *outmsg = NULL;
  if (numSelectionIter > 0) {
    outmsg = new (numK, numK, numK) KSelectionMessage;
    outmsg->numKMinIDs = numK;
    outmsg->numKMaxIDs = numK;
    for (int i=0; i<numK; i++) {
      outmsg->minIDs[i] = -1;
      outmsg->maxIDs[i] = -1;
    }
    thisProxy.collectDistances(outmsg);
  } else {
    CkPrintf("Warning: No selection iteration from the start!\n");
    // invoke phase completion on all processors
    thisProxy.phaseDone();
  }
}

/*
 *  lastMin = array of minimum champions of the last tournament
 *  lastMax = array of maximum champions of the last tournament
 *  lastMaxVal = array of last encountered maximum values, allows previous
 *                 minimum winners to eliminate themselves from the next
 *                 minimum race.
 *
 *  Called on all processors.
 */
void KMeansBOC::collectDistances(KSelectionMessage *msg) {

  // if(CkMyPe()==0)    CkPrintf("[%d] KMeansBOC::collectDistances time=\t%g\n", CkMyPe(), CkWallTimer() );

  DEBUGF("[%d] %d | min = %d max = %d\n", CkMyPe(),
	 lastMinK, msg->minIDs[lastMinK], msg->maxIDs[lastMinK]);
  if ((CkMyPe() == msg->minIDs[lastMinK]) || 
      (CkMyPe() == msg->maxIDs[lastMinK])) {
    CkAssert(!selected);
    selected = true;
  }

  // build outgoing reduction structure
  //   format = minVal | ID | maxVal | ID
  double *minMaxAndIDs = NULL;

  minMaxAndIDs = new double[numK*4];
  // initialize to the appropriate out-of-band values (for error checks)
  for (int i=0; i<numK; i++) {
    minMaxAndIDs[i*4] = -1.0; // out-of-band min value
    minMaxAndIDs[i*4+1] = -1.0; // out of band ID
    minMaxAndIDs[i*4+2] = -1.0; // out-of-band max value
    minMaxAndIDs[i*4+3] = -1.0; // out of band ID
  }
  // If I have not won before, I put myself back into the competition
  if (!selected) {
    DEBUGF("[%d] My Contribution = %lf\n", CkMyPe(), minDistance);
    minMaxAndIDs[lastMinK*4] = minDistance;
    minMaxAndIDs[lastMinK*4+1] = CkMyPe();
    minMaxAndIDs[lastMinK*4+2] = minDistance;
    minMaxAndIDs[lastMinK*4+3] = CkMyPe();
  }
  delete msg;

  CkCallback cb(CkIndex_KMeansBOC::findNextMinMax(NULL), 
		0, thisProxy);
  contribute(numK*4*sizeof(double), minMaxAndIDs, 
	     minMaxReductionType, cb);  
}

void KMeansBOC::findNextMinMax(CkReductionMsg *msg) {
  // incoming format:
  //   minVal | minID | maxVal | maxID

  // if(CkMyPe()==0)    CkPrintf("[%d] KMeansBOC::findNextMinMax time=\t%g\n", CkMyPe(), CkWallTimer() );

  if (numSelectionIter > 0) {
    double *incInfo = (double *)msg->getData();
    
    KSelectionMessage *outmsg = new (numK, numK) KSelectionMessage;
    outmsg->numKMinIDs = numK;
    outmsg->numKMaxIDs = numK;
    
    for (int i=0; i<numK; i++) {
      DEBUGF("%d | %lf %d %lf %d \n", i, 
	     incInfo[i*4], (int)incInfo[i*4+1], 
	     incInfo[i*4+2], (int)incInfo[i*4+3]);
    }

    for (int i=0; i<numK; i++) {
      if (exemplarChoicesLeft[i] > 0) {
	outmsg->minIDs[i] = (int)incInfo[i*4+1];
	exemplarChoicesLeft[i]--;
      } else {
	outmsg->minIDs[i] = -1;
      }
      if (outlierChoicesLeft[i] > 0) {
	outmsg->maxIDs[i] = (int)incInfo[i*4+3];
	outlierChoicesLeft[i]--;
      } else {
	outmsg->maxIDs[i] = -1;
      }
    }
    thisProxy.collectDistances(outmsg);
    numSelectionIter--;
  } else {
    // invoke phase completion on all processors
    thisProxy.phaseDone();
  }
}

/**
 *  Completion of the K-Means clustering and data selection of one phase
 *    of the computation.
 *
 *  Called on every processor.
 */
void KMeansBOC::phaseDone() {

  //  if(CkMyPe()==0)    CkPrintf("[%d] KMeansBOC::phaseDone time=\t%g\n", CkMyPe(), CkWallTimer() );

  LogPool *pool = CkpvAccess(_trace)->_logPool;
  CProxy_TraceProjectionsBOC bocProxy(traceProjectionsGID);

  // now decide on what to do with the decision.
  if (!selected) {
    if (usePhases) {
      pool->keepPhase[currentPhase] = false;
    } else {
      // if not using phases, we're working on the whole log
      pool->setAllPhases(false);
    }
  }

  // **FIXME** (?) - All processors have to agree on this, or the reduction
  //   will not be correct! The question is "is this enforcible?"
  if ((currentPhase == (pool->numPhases-1)) || !usePhases) {
    // We're done
    int dummy = 0;
    CkCallback cb(CkIndex_TraceProjectionsBOC::kMeansDone(NULL), 
		  0, bocProxy);
    contribute(sizeof(int), &dummy, CkReduction::sum_int, cb);
  } else {
    // reset all phase-based k-means data and decisions

    // **FIXME**!!!!!    
    
    // invoke the next K-Means computation phase.
    currentPhase++;
    thisProxy[CkMyPe()].getNextPhaseMetrics();
  }
}

void TraceProjectionsBOC::startTimeAnalysis()
{
  double startTime = 0.0;
  if (CkpvAccess(_trace)->_logPool->numEntries>0)
     startTime = CkpvAccess(_trace)->_logPool->pool[0].time;
  CkCallback cb(CkIndex_TraceProjectionsBOC::startTimeDone(NULL), thisProxy);
  contribute(sizeof(double), &startTime, CkReduction::min_double, cb);  
}

void TraceProjectionsBOC::startTimeDone(CkReductionMsg *msg)
{
  // CkPrintf("[%d] TraceProjectionsBOC::startTimeDone time=\t%g parModulesRemaining:%d\n", CkMyPe(), CkWallTimer(), parModulesRemaining);

  if (CkpvAccess(_trace) != NULL) {
    CkpvAccess(_trace)->_logPool->globalStartTime = *(double *)msg->getData();
    CkpvAccess(_trace)->_logPool->setNewStartTime();
    //if (CkMyPe() == 0) CkPrintf("Start time determined to be %lf us\n", (CkpvAccess(_trace)->_logPool->globalStartTime)*1e06);
  }
  delete msg;
  thisProxy[CkMyPe()].startEndTimeAnalysis();
}

void TraceProjectionsBOC::startEndTimeAnalysis()
{
 //CkPrintf("[%d] TraceProjectionsBOC::startEndTimeAnalysis time=\t%g\n", CkMyPe(), CkWallTimer() );

  endTime = CkpvAccess(_trace)->endTime;
  // CkPrintf("[%d] End time is %lf us\n", CkMyPe(), endTime*1e06);

  CkCallback cb(CkIndex_TraceProjectionsBOC::endTimeDone(NULL), 
		0, thisProxy);
  contribute(sizeof(double), &endTime, CkReduction::max_double, cb);  
}

void TraceProjectionsBOC::endTimeDone(CkReductionMsg *msg)
{
 //if(CkMyPe()==0)    CkPrintf("[%d] TraceProjectionsBOC::endTimeDone time=\t%g parModulesRemaining:%d\n", CkMyPe(), CkWallTimer(), parModulesRemaining);

  CkAssert(CkMyPe() == 0);
  parModulesRemaining--;
  if (CkpvAccess(_trace) != NULL) {
    CkpvAccess(_trace)->_logPool->globalEndTime = *(double *)msg->getData() - CkpvAccess(_trace)->_logPool->globalStartTime;
    // CkPrintf("End time determined to be %lf us\n",
    //	     (CkpvAccess(_trace)->_logPool->globalEndTime)*1e06);
  }
  delete msg;
  if (parModulesRemaining == 0) {
    thisProxy[CkMyPe()].finalize();
  }
}

void TraceProjectionsBOC::kMeansDone(CkReductionMsg *msg) {

 if(CkMyPe()==0)  CkPrintf("[%d] TraceProjectionsBOC::kMeansDone time=\t%g\n", CkMyPe(), CkWallTimer() );

  CkAssert(CkMyPe() == 0);
  parModulesRemaining--;
  CkPrintf("K-Means Analysis Time = %lf seconds\n",
	   CmiWallTimer()-analysisStartTime);
  delete msg;
  if (parModulesRemaining == 0) {
    thisProxy[CkMyPe()].finalize();
  }
}

/**
 *
 *  This version is called (on processor 0) only if flushCheck fails.
 *
 */
void TraceProjectionsBOC::kMeansDone() {
  CkAssert(CkMyPe() == 0);
  parModulesRemaining--;
  CkPrintf("K-Means Analysis Aborted because of flush. Time taken = %lf seconds\n",
	   CmiWallTimer()-analysisStartTime);
  if (parModulesRemaining == 0) {
    thisProxy[CkMyPe()].finalize();
  }
}

void TraceProjectionsBOC::finalize()
{
  CkAssert(CkMyPe() == 0);
  //CkPrintf("Total Analysis Time = %lf seconds\n", 
  //	   CmiWallTimer()-analysisStartTime);
  thisProxy.closingTraces();
}

// Called on every processor
void TraceProjectionsBOC::closingTraces() {
  CkpvAccess(_trace)->closeTrace();

    // subtle:  reduction needs to go to the PE which started CkExit()
  int pe = 0;
  if (endPe != -1) pe = endPe;
  CkCallback cb(CkIndex_TraceProjectionsBOC::closeParallelShutdown(NULL), 
		pe, thisProxy); 
  contribute(0, NULL, CkReduction::sum_int, cb);  
}

// The sole purpose of this reduction is to decide whether or not
//   Projections as a module needs to call CkExit() to get other
//   modules to shutdown.
void TraceProjectionsBOC::closeParallelShutdown(CkReductionMsg *msg) {
  CkAssert(endPe == -1 && CkMyPe() ==0 || CkMyPe() == endPe);
  delete msg;
  // decide if CkExit() needs to be called
  if (!CkpvAccess(_trace)->converseExit) {
    CkExit();
  }
}
/*
 *  Registration and definition of the Outlier Reduction callback.
 *  Format: Sum | Min | Max | Sum of Squares
 */
CkReductionMsg *outlierReduction(int nMsgs,
				 CkReductionMsg **msgs) {
  int numBytes = 0;
  int numMetrics = 0;
  double *ret = NULL;

  if (nMsgs == 1) {
    // nothing to do, just pass it on
    return CkReductionMsg::buildNew(msgs[0]->getSize(),msgs[0]->getData());
  }

  if (nMsgs > 1) {
    numBytes = msgs[0]->getSize();
    // sanity checks
    if (numBytes%sizeof(double) != 0) {
      CkAbort("Outlier Reduction Size incompatible with doubles!\n");
    }
    if ((numBytes/sizeof(double))%4 != 0) {
      CkAbort("Outlier Reduction Size Array not divisible by 4!\n");
    }
    numMetrics = (numBytes/sizeof(double))/4;
    ret = new double[numMetrics*4];

    // copy the first message data into the return structure first
    for (int i=0; i<numMetrics*4; i++) {
      ret[i] = ((double *)msgs[0]->getData())[i];
    }

    // Sum | Min | Max | Sum of Squares
    for (int msgIdx=1; msgIdx<nMsgs; msgIdx++) {
      for (int i=0; i<numMetrics; i++) {
	// Sum
	ret[i] += ((double *)msgs[msgIdx]->getData())[i];
	// Min
	ret[numMetrics + i] =
	  (ret[numMetrics + i] < 
	   ((double *)msgs[msgIdx]->getData())[numMetrics + i]) 
	  ? ret[numMetrics + i] : 
	  ((double *)msgs[msgIdx]->getData())[numMetrics + i];
	// Max
	ret[2*numMetrics + i] = 
	  (ret[2*numMetrics + i] >
	   ((double *)msgs[msgIdx]->getData())[2*numMetrics + i])
	  ? ret[2*numMetrics + i] :
	  ((double *)msgs[msgIdx]->getData())[2*numMetrics + i];
	// Sum of Squares (squaring already done at leaf)
	ret[3*numMetrics + i] +=
	  ((double *)msgs[msgIdx]->getData())[3*numMetrics + i];
      }
    }
  }
  
  /* apparently, we do not delete the incoming messages */
  return CkReductionMsg::buildNew(numBytes,ret);
}

/*
 * The only reason we have a user-defined reduction is to support
 *   identification of the "winning" processors as well as to handle
 *   both the min and the max of each "tournament". A simple min/max
 *   discovery cannot handle ties.
 */
CkReductionMsg *minMaxReduction(int nMsgs,
				CkReductionMsg **msgs) {
  CkAssert(nMsgs > 0);

  int numBytes = msgs[0]->getSize();
  CkAssert(numBytes%sizeof(double) == 0);
  int numK = (numBytes/sizeof(double))/4;

  double *ret = new double[numK*4];
  // fill with out-of-band values
  for (int i=0; i<numK; i++) {
    ret[i*4] = -1.0;
    ret[i*4+1] = -1.0;
    ret[i*4+2] = -1.0;
    ret[i*4+3] = -1.0;
  }

  // incoming format K * (minVal | minIdx | maxVal | maxIdx)
  for (int i=0; i<nMsgs; i++) {
    double *temp = (double *)msgs[i]->getData();
    for (int j=0; j<numK; j++) {
      // no previous valid min
      if (ret[j*4+1] < 0) {
	// fill it in only if the incoming min is valid
	if (temp[j*4+1] >= 0) {
	  ret[j*4] = temp[j*4];      // fill min value
	  ret[j*4+1] = temp[j*4+1];  // fill ID
	}
      } else {
	// find Min, only if incoming min is valid
	if (temp[j*4+1] >= 0) {
	  if (temp[j*4] < ret[j*4]) {
	    ret[j*4] = temp[j*4];      // replace min value
	    ret[j*4+1] = temp[j*4+1];  // replace ID
	  }
	}
      }
      // no previous valid max
      if (ret[j*4+3] < 0) {
	// fill only if the incoming max is valid
	if (temp[j*4+3] >= 0) {
	  ret[j*4+2] = temp[j*4+2];  // fill max value
	  ret[j*4+3] = temp[j*4+3];  // fill ID
	}
      } else {
	// find Max, only if incoming max is valid
	if (temp[j*4+3] >= 0) {
	  if (temp[j*4+2] > ret[j*4+2]) {
	    ret[j*4+2] = temp[j*4+2];  // replace max value
	    ret[j*4+3] = temp[j*4+3];  // replace ID
	  }
	}
      }
    }
  }
  CkReductionMsg *redmsg = CkReductionMsg::buildNew(numBytes, ret);
  delete [] ret;
  return redmsg;
}

#include "TraceProjections.def.h"
#endif //PROJ_ANALYSIS

/*@}*/
