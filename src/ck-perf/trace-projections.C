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

CkpvStaticDeclare(CkVec<char *>, usrEventlist);
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
  } else {
    if (nonDeltaLog) fclose(fp);
    if (deltaLog)  fclose(deltafp); 
  }
#else
  if (nonDeltaLog)  fclose(fp);
  if (deltaLog) fclose(deltafp);
#endif
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

extern "C"
void traceProjectionsBeginIdle(void)
{
  CkpvAccess(_trace)->beginIdle();
}

extern "C"
void traceProjectionsEndIdle(void)
{
  CkpvAccess(_trace)->endIdle();
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
    if (CkMyPe() == 0) writeSts();
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
  fprintf(stsfp, "VERSION %s\n", PROJECTION_VERSION);
  traceWriteSTS(stsfp,CkpvAccess(usrEvents)->length());
  for(int i=0;i<CkpvAccess(usrEvents)->length();i++)
    fprintf(stsfp, "EVENT %d %s\n", (*CkpvAccess(usrEvents))[i]->e, (*CkpvAccess(usrEvents))[i]->str);
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

void LogPool::add(UChar type,UShort mIdx,UShort eIdx,double time,int event,int pe, int ml, CmiObjId *id, double recvT) 
{
  new (&pool[numEntries++])
    LogEntry(time, type, mIdx, eIdx, event, pe, ml, id, recvT);
  if(poolSize==numEntries) {
    double writeTime = TraceTimer();
    writeLog();
    numEntries = 0;
    new (&pool[numEntries++]) LogEntry(writeTime, BEGIN_INTERRUPT);
    new (&pool[numEntries++]) LogEntry(TraceTimer(), END_INTERRUPT);
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
      bgAddProjEvent(&pool[numEntries-1], numEntries-1, time, updateProjLog, &fp, 1);
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
    double writeTime = TraceTimer();
    writeLog();
    numEntries = 0;
    new (&pool[numEntries++]) LogEntry(writeTime, BEGIN_INTERRUPT);
    new (&pool[numEntries++]) LogEntry(TraceTimer(), END_INTERRUPT);
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

void LogEntry::pup(PUP::er &p)
{
  int itime, irecvtime;
  char ret = '\n';

  p|type;
  if (p.isPacking()) itime = (int)(1.0e6*time);
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
      if (p.isPacking()) irecvtime = (int)(1.0e6*recvTime);
      p|mIdx; p|eIdx; p|itime; p|event; p|pe; 
      p|msglen; p|irecvtime; p|id.id[0]; p|id.id[1]; p|id.id[2];
      if (p.isUnpacking()) recvTime = irecvtime/1.0e6;
      break;
    case CREATION:
      if (p.isPacking()) irecvtime = (int)(1.0e6*recvTime);
      p|mIdx; p|eIdx; p|itime;
      p|event; p|pe; p|msglen; p|irecvtime;
      if (p.isUnpacking()) recvTime = irecvtime/1.0e6;
      break;
    case CREATION_MULTICAST:
      if (p.isPacking()) irecvtime = (int)(1.0e6*recvTime);
      p|mIdx; p|eIdx; p|itime;
      p|event; p|pe; p|msglen; p|irecvtime; p|numpes;
      if (pes == NULL) {
        int n=-1;
        p(n);
      }
      else {
	for (int i=0; i<numpes; i++) p|pes[i];
      }
      if (p.isUnpacking()) recvTime = irecvtime/1.0e6;
      break;
    case END_PROCESSING:
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
      p|itime;
      break;

    default:
      CmiError("***Internal Error*** Wierd Event %d.\n", type);
      break;
  }
  if (p.isUnpacking()) time = itime/1.0e6;
  p|ret;
}

TraceProjections::TraceProjections(char **argv): curevent(0), isIdle(0), inEntry(0)
{
  if (TRACE_CHARM_PE() == 0) return;

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
}

int TraceProjections::traceRegisterUserEvent(const char* evt, int e)
{
  OPTIMIZED_VERSION
  CkAssert(e==-1 || e>=0);
  CkAssert(evt != NULL);
  int event;
  int biggest = 0;
  for (int i=0; i<CkpvAccess(usrEvents)->length(); i++) {
    int cur = (*CkpvAccess(usrEvents))[i]->e;
    if (cur == e) 
      CmiAbort("UserEvent double registered!");
    if (cur > biggest) biggest = cur;
  }
  if (e==-1) event = biggest;
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
    _logPool->writeSts();
}

void TraceProjections::traceClose(void)
{
  if(CkMyPe()==0){
    _logPool->writeSts();
  }
  if (TRACE_CHARM_PE()) {
    CkpvAccess(_trace)->endComputation();
    delete _logPool;		// will write
    // remove myself from traceArray so that no tracing will be called.
    CkpvAccess(_traces)->removeTrace(this);
//    delete CkpvAccess(_trace);
  }
//  free(CkpvAccess(traceRoot));
}

void TraceProjections::traceBegin(void)
{
#if ! CMK_TRACE_IN_CHARM
  cancel_beginIdle = CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,(CcdVoidFn)traceProjectionsBeginIdle,0);
  cancel_endIdle = CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_BUSY,(CcdVoidFn)traceProjectionsEndIdle,0);
#endif
}

void TraceProjections::traceEnd(void) 
{
#if ! CMK_TRACE_IN_CHARM
  CcdCancelCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE, cancel_beginIdle);
  CcdCancelCallOnConditionKeep(CcdPROCESSOR_BEGIN_BUSY, cancel_endIdle);
#endif
}

void TraceProjections::userEvent(int e)
{
  _logPool->add(USER_EVENT, e, 0, TraceTimer(),curevent++,CkMyPe());
}

void TraceProjections::userBracketEvent(int e, double bt, double et)
{
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
  if (checknested && inEntry) CmiAbort("Nested Begin Execute!\n");
  execEvent = CtvAccess(curThreadEvent);
  execEp = (-1);
  _logPool->add(BEGIN_PROCESSING,ForChareMsg,_threadEP,TraceTimer(),
                             execEvent,CkMyPe(), 0, tid);
  inEntry = 1;
}

void TraceProjections::beginExecute(envelope *e)
{
  if(e==0) {
    if (checknested && inEntry) CmiAbort("Nested Begin Execute!\n");
    execEvent = CtvAccess(curThreadEvent);
    execEp = (-1);
    _logPool->add(BEGIN_PROCESSING,ForChareMsg,_threadEP,TraceTimer(),
                             execEvent,CkMyPe());
    inEntry = 1;
  } else {
    beginExecute(e->getEvent(),e->getMsgtype(),e->getEpIdx(),e->getSrcPe(),e->getTotalsize());
  }
}

void TraceProjections::beginExecute(int event,int msgType,int ep,int srcPe, int mlen,CmiObjId *idx)
{
  if (checknested && inEntry) CmiAbort("Nested Begin Execute!\n");
  execEvent=event;
  execEp=ep;
  execPe=srcPe;
  _logPool->add(BEGIN_PROCESSING,msgType,ep,TraceTimer(),
                             event,srcPe, mlen, idx);
  inEntry = 1;
}

void TraceProjections::endExecute(void)
{
  if (checknested && !inEntry) CmiAbort("Nested EndExecute!\n");
  if(execEp == (-1)) {
    _logPool->add(END_PROCESSING,0,_threadEP,TraceTimer(),
                             execEvent,CkMyPe());
  } else {
    _logPool->add(END_PROCESSING,0,execEp,TraceTimer(),
                             execEvent,execPe);
  }
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

void TraceProjections::beginIdle(void)
{
  if (isIdle == 0) {
    _logPool->add(BEGIN_IDLE, 0, 0, TraceTimer(), 0, CkMyPe());
    isIdle = 1;
  }
}

void TraceProjections::endIdle(void)
{
  if (isIdle) {
    _logPool->add(END_IDLE, 0, 0, TraceTimer(), 0, CkMyPe());
    isIdle = 0;
  }
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
  _logPool->add(BEGIN_COMPUTATION, 0, 0, TraceTimer(), -1, -1);
}

void TraceProjections::endComputation(void)
{
  _logPool->add(END_COMPUTATION, 0, 0, TraceTimer(), -1, -1);
}

// specialized PUP:ers for handling trace projections logs
void toProjectionsFile::bytes(void *p,int n,size_t itemSize,dataType t)
{
  for (int i=0;i<n;i++) 
    switch(t) {
    case Tchar: fprintf(f,"%c",((char *)p)[i]); break;
    case Tuchar:
    case Tbyte: fprintf(f,"%d",((unsigned char *)p)[i]); break;
    case Tshort: fprintf(f," %d",((short *)p)[i]); break;
    case Tushort: fprintf(f," %u",((unsigned short *)p)[i]); break;
    case Tint: fprintf(f," %d",((int *)p)[i]); break;
    case Tuint: fprintf(f," %u",((unsigned int *)p)[i]); break;
    case Tlong: fprintf(f," %ld",((long *)p)[i]); break;
    case Tulong: fprintf(f," %lu",((unsigned long *)p)[i]); break;
    case Tfloat: fprintf(f," %.7g",((float *)p)[i]); break;
    case Tdouble: fprintf(f," %.15g",((double *)p)[i]); break;
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
    default: CmiAbort("Unrecognized pup type code!");
    };
}
#endif

/*@}*/
