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
#define OPEN_LOG  \
  do  {  \
    fp = fopen(fname, "a");   \
  } while (!fp && (errno == EINTR || errno == EMFILE)); 	\
  if(!fp) CmiAbort("Cannot open Projections Trace File for writing...\n");
#define CLOSE_LOG  fclose(fp);
#else
#define OPEN_LOG
#define CLOSE_LOG
#endif

/**
  For each TraceFoo module, _createTraceFoo() must be defined.
  This function is called in _createTraces() generated in moduleInit.C
*/
void _createTraceprojections(char **argv)
{
  DEBUGF(("%d createTraceProjections\n", CkMyPe()));
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

LogPool::LogPool(char *pgm, int b) {
  binary = b;
  pool = new LogEntry[CkpvAccess(CtrLogBufSize)];
  numEntries = 0;
  poolSize = CkpvAccess(CtrLogBufSize);
  char pestr[10];
  sprintf(pestr, "%d", CkMyPe());
  int len = strlen(pgm) + strlen(".log.") + strlen(pestr) + 1;
  fname = new char[len];
  sprintf(fname, "%s.%s.log", pgm, pestr);
  do
  {
    fp = fopen(fname, "w+");
  } while (!fp && (errno == EINTR || errno == EMFILE));
  if(!fp) {
    CmiAbort("Cannot open Projections Trace File for writing...\n");
  }
  if(!binary) {
    fprintf(fp, "PROJECTIONS-RECORD\n");
  }
  CLOSE_LOG 
}

LogPool::~LogPool() 
{
  if(binary) writeBinary();
  else write();
#if !CMK_TRACE_LOGFILE_NUM_CONTROL
  fclose(fp);
#endif
  delete[] pool;
  delete [] fname;
}

void LogPool::write(void) 
{
  OPEN_LOG
  for(UInt i=0; i<numEntries; i++)
    pool[i].write(fp);
  CLOSE_LOG
}

void LogPool::writeSts(void)
{
  char *fname = new char[strlen(CkpvAccess(traceRoot))+strlen(".sts")+1];
  sprintf(fname, "%s.sts", CkpvAccess(traceRoot));
  FILE *sts;
  do
  {
    sts = fopen(fname, "w");
  } while (!sts && (errno == EINTR || errno == EMFILE));
  if(sts==0)
    CmiAbort("Cannot open projections sts file for writing.\n");
  delete[] fname;
  fprintf(sts, "MACHINE %s\n",CMK_MACHINE_NAME);
  fprintf(sts, "PROCESSORS %d\n", CkNumPes());
  fprintf(sts, "TOTAL_CHARES %d\n", _numChares);
  fprintf(sts, "TOTAL_EPS %d\n", _numEntries);
  fprintf(sts, "TOTAL_MSGS %d\n", _numMsgs);
  fprintf(sts, "TOTAL_PSEUDOS %d\n", 0);
  fprintf(sts, "TOTAL_EVENTS %d\n", _numEvents);
  int i;
  for(i=0;i<_numChares;i++)
    fprintf(sts, "CHARE %d %s\n", i, _chareTable[i]->name);
  for(i=0;i<_numEntries;i++)
    fprintf(sts, "ENTRY CHARE %d %s %d %d\n", i, _entryTable[i]->name,
                 _entryTable[i]->chareIdx, _entryTable[i]->msgIdx);
  for(i=0;i<_numMsgs;i++)
    fprintf(sts, "MESSAGE %d %d\n", i, _msgTable[i]->size);
  for(i=0;i<_numEvents;i++)
    fprintf(sts, "EVENT %d Event%d\n", i, i);
  fprintf(sts, "END\n");
  fclose(sts);
}

void LogPool::add(UChar type,UShort mIdx,UShort eIdx,double time,int event,int pe, int ml) 
{
  new (&pool[numEntries++])
    LogEntry(time, type, mIdx, eIdx, event, pe, ml);
  if(poolSize==numEntries) {
    double writeTime = CkTimer();
    if(binary) writeBinary(); else write();
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
  CtvInitialize(int,curThreadEvent);
  CtvAccess(curThreadEvent)=0;
  int binary = CmiGetArgFlag(argv,"+binary-trace");
  _logPool = new LogPool(CkpvAccess(traceRoot),binary);
}


int TraceProjections::traceRegisterUserEvent(const char*)
{
  OPTIMIZED_VERSION
  if(CkMyPe()==0)
    return _numEvents++;
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
  CkpvAccess(_trace)->endComputation();
  if(CkMyPe()==0)
    _logPool->writeSts();
  delete _logPool;		// will write
  delete CkpvAccess(_trace);
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
  if(CkMyRank()==0) {
    _threadMsg = CkRegisterMsg("dummy_thread_msg", 0, 0, 0, 0);
    _threadChare = CkRegisterChare("dummy_thread_chare", 0);
    _threadEP = CkRegisterEp("dummy_thread_ep", 0, _threadMsg,_threadChare);
  }
  _logPool->add(BEGIN_COMPUTATION, 0, 0, TraceTimer(), -1, -1);
}

void TraceProjections::endComputation(void)
{
  _logPool->add(END_COMPUTATION, 0, 0, TraceTimer(), -1, -1);
}

/*@}*/
