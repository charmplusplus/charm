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

#include "trace-projections.h"

#define DEBUGF(x)           // CmiPrintf x

CkpvStaticDeclare(Trace*, _trace);
CtvStaticDeclare(int,curThreadEvent);

static int _numEvents = 0;
static int warned = 0;
static int _threadMsg, _threadChare, _threadEP;

CkpvStaticDeclare(CkVec<char *>, usrEvents);

#ifdef CMK_OPTIMIZE
#define OPTIMIZED_VERSION 	\
	if (!warned) { warned=1; 	\
	CmiPrintf("\n\n!!!! Warning: traceUserEvent not availbale in optimized version!!!!\n\n\n"); }
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
    do {
      zfp = gzopen(fname, mode);
    } while (!zfp && (errno == EINTR || errno == EMFILE));
    if(!zfp) CmiAbort("Cannot open Projections Trace File for writing...\n");
  } else {
    do {
      fp = fopen(fname, mode);
    } while (!fp && (errno == EINTR || errno == EMFILE));
    if(!fp) CmiAbort("Cannot open Projections Trace File for writing...\n");
  }
#else
  do {
    fp = fopen(fname, mode);
  } while (!fp && (errno == EINTR || errno == EMFILE));
  if(!fp) CmiAbort("Cannot open Projections Trace File for writing...\n");
#endif
}

void LogPool::closeLog(void)
{
#if CMK_PROJECTIONS_USE_ZLIB
  if(compressed)
    gzclose(zfp);
  else
    fclose(fp);
#else
    fclose(fp);
#endif
}
/**
  For each TraceFoo module, _createTraceFoo() must be defined.
  This function is called in _createTraces() generated in moduleInit.C
*/
void _createTraceprojections(char **argv)
{
  DEBUGF(("%d createTraceProjections\n", CkMyPe()));
  CkpvInitialize(CkVec<char *>, usrEvents);
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
  poolSize = CkpvAccess(CtrLogBufSize);
  pgmname = new char[strlen(pgm)+1];
  strcpy(pgmname, pgm);
}

void LogPool::init(void)
{
  char pestr[10];
  sprintf(pestr, "%d", CkMyPe());
#if CMK_PROJECTIONS_USE_ZLIB
  int len;
  if(compressed)
    len = strlen(pgmname) + strlen(".log.") + strlen(pestr) + strlen(".gz") + 1;
  else
    len = strlen(pgmname) + strlen(".log.") + strlen(pestr) + 1;
#else
  int len = strlen(pgmname) + strlen(".log.") + strlen(pestr) + 1;
#endif
  fname = new char[len];
#if CMK_PROJECTIONS_USE_ZLIB
  if(compressed)
    sprintf(fname, "%s.%s.log.gz", pgmname, pestr);
  else
    sprintf(fname, "%s.%s.log", pgmname, pestr);
#else
  sprintf(fname, "%s.%s.log", pgmname, pestr);
#endif
  openLog("w+");
  if(!binary) {
#if CMK_PROJECTIONS_USE_ZLIB
    if(compressed)
      gzprintf(zfp, "PROJECTIONS-RECORD\n");
    else
#endif
      fprintf(fp, "PROJECTIONS-RECORD\n");
  }
  CLOSE_LOG 

  if (CkMyPe() == 0) 
  {
    char *fname = new char[strlen(CkpvAccess(traceRoot))+strlen(".sts")+1];
    sprintf(fname, "%s.sts", CkpvAccess(traceRoot));
    do
    {
      stsfp = fopen(fname, "w");
    } while (!stsfp && (errno == EINTR || errno == EMFILE));
    if(stsfp==0)
      CmiAbort("Cannot open projections sts file for writing.\n");
    delete[] fname;
  }
}

LogPool::~LogPool() 
{
  writeLog();
#if !CMK_TRACE_LOGFILE_NUM_CONTROL
  closeLog();
#endif
  delete[] pool;
  delete [] fname;
}

void LogPool::writeLog(void)
{
  OPEN_LOG
  if(binary) writeBinary();
#if CMK_PROJECTIONS_USE_ZLIB
  else if(compressed) writeCompressed();
#endif
  else write();
  CLOSE_LOG
}

void LogPool::write(void) 
{
  for(UInt i=0; i<numEntries; i++)
    pool[i].write(fp);
}

void LogPool::writeBinary(void) {
  for(UInt i=0; i<numEntries; i++)
    pool[i].writeBinary(fp);
}

#if CMK_PROJECTIONS_USE_ZLIB
void LogPool::writeCompressed(void) {
  for(UInt i=0; i<numEntries; i++)
    pool[i].writeCompressed(zfp);
}
#endif

void LogPool::writeSts(void)
{
  fprintf(stsfp, "MACHINE %s\n",CMK_MACHINE_NAME);
  fprintf(stsfp, "PROCESSORS %d\n", CkNumPes());
  fprintf(stsfp, "TOTAL_CHARES %d\n", _numChares);
  fprintf(stsfp, "TOTAL_EPS %d\n", _numEntries);
  fprintf(stsfp, "TOTAL_MSGS %d\n", _numMsgs);
  fprintf(stsfp, "TOTAL_PSEUDOS %d\n", 0);
  fprintf(stsfp, "TOTAL_EVENTS %d\n", _numEvents);
  int i;
  for(i=0;i<_numChares;i++)
    fprintf(stsfp, "CHARE %d %s\n", i, _chareTable[i]->name);
  for(i=0;i<_numEntries;i++)
    fprintf(stsfp, "ENTRY CHARE %d %s %d %d\n", i, _entryTable[i]->name,
                 _entryTable[i]->chareIdx, _entryTable[i]->msgIdx);
  for(i=0;i<_numMsgs;i++)
    fprintf(stsfp, "MESSAGE %d %d\n", i, _msgTable[i]->size);
  for(i=0;i<_numEvents;i++)
    fprintf(stsfp, "EVENT %d %s\n", i, CkpvAccess(usrEvents)[i]);
  fprintf(stsfp, "END\n");
  fclose(stsfp);
}

void LogPool::add(UChar type,UShort mIdx,UShort eIdx,double time,int event,int pe, int ml) 
{
  new (&pool[numEntries++])
    LogEntry(time, type, mIdx, eIdx, event, pe, ml);
  if(poolSize==numEntries) {
    double writeTime = CkTimer();
    writeLog();
    numEntries = 0;
    new (&pool[numEntries++]) LogEntry(writeTime, BEGIN_INTERRUPT);
    new (&pool[numEntries++]) LogEntry(CkTimer(), END_INTERRUPT);
  }
}

void LogEntry::write(FILE* fp)
{
  fprintf(fp, "%d ", type);

  switch (type) {
    case USER_EVENT:
      fprintf(fp, "%d %u %d %d\n", mIdx, (UInt) (time*1.0e6), event, pe);
      break;

    case BEGIN_IDLE:
    case END_IDLE:
    case BEGIN_PACK:
    case END_PACK:
    case BEGIN_UNPACK:
    case END_UNPACK:
      fprintf(fp, "%u %d\n", (UInt) (time*1.0e6), pe);
      break;

    case CREATION:
    case BEGIN_PROCESSING:
    case END_PROCESSING:
    case MESSAGE_RECV:
      fprintf(fp, "%d %d %u %d %d %d\n", mIdx, eIdx, (UInt) (time*1.0e6), event, pe, msglen);
      break;

    case ENQUEUE:
    case DEQUEUE:
      fprintf(fp, "%d %u %d %d\n", mIdx, (UInt) (time*1.0e6), event, pe);
      break;

    case BEGIN_INTERRUPT:
    case END_INTERRUPT:
      fprintf(fp, "%u %d %d\n", (UInt) (time*1.0e6), event, pe);
      break;

    case BEGIN_COMPUTATION:
    case END_COMPUTATION:
    fprintf(fp, "%u\n", (UInt) (time*1.0e6));
      break;

    default:
      CmiError("***Internal Error*** Wierd Event %d.\n", type);
      break;
  }
}

#if CMK_PROJECTIONS_USE_ZLIB
void LogEntry::writeCompressed(gzFile zfp)
{
  gzprintf(zfp, "%d ", type);

  switch (type) {
    case USER_EVENT:
      gzprintf(zfp, "%d %u %d %d\n", mIdx, (UInt) (time*1.0e6), event, pe);
      break;

    case BEGIN_IDLE:
    case END_IDLE:
    case BEGIN_PACK:
    case END_PACK:
    case BEGIN_UNPACK:
    case END_UNPACK:
      gzprintf(zfp, "%u %d\n", (UInt) (time*1.0e6), pe);
      break;

    case CREATION:
    case BEGIN_PROCESSING:
    case END_PROCESSING:
    case MESSAGE_RECV:
      gzprintf(zfp, "%d %d %u %d %d %d\n", mIdx, eIdx, (UInt) (time*1.0e6), event, pe, msglen);
      break;

    case ENQUEUE:
    case DEQUEUE:
      gzprintf(zfp, "%d %u %d %d\n", mIdx, (UInt) (time*1.0e6), event, pe);
      break;

    case BEGIN_INTERRUPT:
    case END_INTERRUPT:
      gzprintf(zfp, "%u %d %d\n", (UInt) (time*1.0e6), event, pe);
      break;

    case BEGIN_COMPUTATION:
    case END_COMPUTATION:
      gzprintf(zfp, "%u\n", (UInt) (time*1.0e6));
      break;

    default:
      CmiError("***Internal Error*** Wierd Event %d.\n", type);
      break;
  }
}
#endif

void LogEntry::writeBinary(FILE* fp)
{
  UInt ttime = (UInt) (time*1.0e6);

  fwrite(&type,sizeof(UChar),1,fp);

  switch (type) {
    case USER_EVENT:
      fwrite(&mIdx,sizeof(UShort),1,fp);
      fwrite(&ttime,sizeof(UInt),1,fp);
      fwrite(&event,sizeof(int),1,fp);
      fwrite(&pe,sizeof(int),1,fp);
      break;

    case BEGIN_IDLE:
    case END_IDLE:
    case BEGIN_PACK:
    case END_PACK:
    case BEGIN_UNPACK:
    case END_UNPACK:
      fwrite(&ttime,sizeof(UInt),1,fp);
      fwrite(&pe,sizeof(int),1,fp);
      break;

    case CREATION:
    case BEGIN_PROCESSING:
    case END_PROCESSING:
      fwrite(&mIdx,sizeof(UShort),1,fp);
      fwrite(&eIdx,sizeof(UShort),1,fp);
      fwrite(&ttime,sizeof(UInt),1,fp);
      fwrite(&event,sizeof(int),1,fp);
      fwrite(&pe,sizeof(int),1,fp);
      fwrite(&msglen,sizeof(int),1,fp);
      break;

    case ENQUEUE:
    case DEQUEUE:
      fwrite(&mIdx,sizeof(UShort),1,fp);
      fwrite(&ttime,sizeof(UInt),1,fp);
      fwrite(&event,sizeof(int),1,fp);
      fwrite(&pe,sizeof(int),1,fp);
      break;

    case BEGIN_INTERRUPT:
    case END_INTERRUPT:
      fwrite(&ttime,sizeof(UInt),1,fp);
      fwrite(&event,sizeof(int),1,fp);
      fwrite(&pe,sizeof(int),1,fp);
      break;

    case BEGIN_COMPUTATION:
    case END_COMPUTATION:
      fwrite(&ttime,sizeof(UInt),1,fp);
      break;

    default:
      CmiError("***Internal Error*** Wierd Event %d.\n", type);
      break;
  }
}

TraceProjections::TraceProjections(char **argv): curevent(0), isIdle(0)
{
  if(CkMyRank()==0) {
    _threadMsg = CkRegisterMsg("dummy_thread_msg", 0, 0, 0, 0);
    _threadChare = CkRegisterChare("dummy_thread_chare", 0);
    _threadEP = CkRegisterEp("dummy_thread_ep", 0, _threadMsg,_threadChare);
  }

  if (TRACE_CHARM_PE() == 0) return;

  CtvInitialize(int,curThreadEvent);
  CtvAccess(curThreadEvent)=0;
  int binary = CmiGetArgFlag(argv,"+binary-trace");
#if CMK_PROJECTIONS_USE_ZLIB
  int compressed = CmiGetArgFlag(argv,"+gz-trace");
#endif
  _logPool = new LogPool(CkpvAccess(traceRoot));
  _logPool->setBinary(binary);
#if CMK_PROJECTIONS_USE_ZLIB
  _logPool->setCompressed(compressed);
#endif
  _logPool->init();
}


int TraceProjections::traceRegisterUserEvent(const char* evt)
{
  OPTIMIZED_VERSION
  if(CkMyPe()==0) {
    CkAssert(evt != NULL);
    CkAssert(CkpvAccess(usrEvents).length() ==  _numEvents);
    CkpvAccess(usrEvents).push_back((char *)evt);
    return _numEvents++;
  }
  else
    return 0;
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
  if(CkMyPe()==0)
    _logPool->writeSts();
  if (TRACE_CHARM_PE()) {
    CkpvAccess(_trace)->endComputation();
    delete _logPool;		// will write
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

void TraceProjections::creation(envelope *e, int num)
{
  if(e==0) {
    CtvAccess(curThreadEvent)=curevent;
    _logPool->add(CREATION,ForChareMsg,_threadEP,TraceTimer(),
                             curevent++,CkMyPe());
  } else {
    int type=e->getMsgtype();
    e->setEvent(curevent);
    for(int i=0; i<num; i++) {
      _logPool->add(CREATION,type,e->getEpIdx(),TraceTimer(),
                               curevent+i,CkMyPe(),e->getTotalsize());
    }
    curevent += num;
  }
}

void TraceProjections::beginExecute(envelope *e)
{
  if(e==0) {
    execEvent = CtvAccess(curThreadEvent);
    execEp = (-1);
    _logPool->add(BEGIN_PROCESSING,ForChareMsg,_threadEP,TraceTimer(),
                             execEvent,CkMyPe());
  } else {
    beginExecute(e->getEvent(),e->getMsgtype(),e->getEpIdx(),e->getSrcPe(),e->getTotalsize());
  }
}
void TraceProjections::beginExecute(int event,int msgType,int ep,int srcPe, int mlen)
{
  execEvent=event;
  execEp=ep;
  execPe=srcPe;
  _logPool->add(BEGIN_PROCESSING,msgType,ep,TraceTimer(),
                             event,srcPe, mlen);
}

void TraceProjections::endExecute(void)
{
  if(execEp == (-1)) {
    _logPool->add(END_PROCESSING,0,_threadEP,TraceTimer(),
                             execEvent,CkMyPe());
  } else {
    _logPool->add(END_PROCESSING,0,execEp,TraceTimer(),
                             execEvent,execPe);
  }
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

void TraceProjections::beginCharmInit(void) {}

void TraceProjections::endCharmInit(void) {}

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

/*@}*/
