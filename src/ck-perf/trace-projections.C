/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "trace-projections.h"

CpvDeclare(Trace*, _trace);
CpvDeclare(int, CtrLogBufSize);
CpvStaticDeclare(LogPool*, _logPool);
CpvStaticDeclare(char*, traceRoot);
CtvStaticDeclare(int,curThreadEvent);

static int _numEvents = 0;
#if CMK_OPTIMIZE
static int warned = 0;
#endif
static int _threadMsg, _threadChare, _threadEP;

#define OPTIMIZED_VERSION 	\
	if (!warned) { warned=1; 	\
	CmiPrintf("\n\n!!!! Warning: traceUserEvent not availbale in optimized version!!!!\n\n\n"); }

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

extern "C" 
void traceInit(char **argv)
{
  traceCommonInit(argv,1);
  CpvInitialize(Trace*, _trace);
  CpvInitialize(LogPool*, _logPool);
  CpvInitialize(int, CtrLogBufSize);
  CpvInitialize(char*, traceRoot);
  CtvInitialize(int,curThreadEvent);
  CtvAccess(curThreadEvent)=0;
  CpvAccess(_trace) = new TraceProjections();
  CpvAccess(CtrLogBufSize) = 10000;
  CmiGetArgInt(argv,"+logsize",&CpvAccess(CtrLogBufSize));
  int binary = CmiGetArgFlag(argv,"+binary-trace");
  char *root;
  if (CmiGetArgString(argv, "+trace-root", &root)) {
    int i;
    for (i=strlen(argv[0])-1; i>=0; i--) if (argv[0][i] == '/') break;
    i++;
    CpvAccess(traceRoot) = (char *)malloc(strlen(argv[0]+i) + strlen(root) + 2);
    _MEMCHECK(CpvAccess(traceRoot));
    strcpy(CpvAccess(traceRoot), root);
    strcat(CpvAccess(traceRoot), "/");
    strcat(CpvAccess(traceRoot), argv[0]+i);
  }
  else {
    CpvAccess(traceRoot) = (char *) malloc(strlen(argv[0])+1);
    _MEMCHECK(CpvAccess(traceRoot));
    strcpy(CpvAccess(traceRoot), argv[0]);
  }
  CpvAccess(_logPool) = new LogPool(CpvAccess(traceRoot),binary);
}

extern "C"
void traceBeginIdle(void)
{
  CpvAccess(_trace)->beginIdle();
}

extern "C"
void traceEndIdle(void)
{
  CpvAccess(_trace)->endIdle();
}

extern "C"
void traceResume(void)
{
  CpvAccess(_trace)->beginExecute(0);
}

extern "C"
void traceSuspend(void)
{
  CpvAccess(_trace)->endExecute();
}

extern "C"
void traceAwaken(CthThread t)
{
  CpvAccess(_trace)->creation(0);
}

extern "C"
void traceUserEvent(int e)
{
#if CMK_OPTIMIZE
  OPTIMIZED_VERSION
#else
  CpvAccess(_trace)->userEvent(e);
#endif
}

extern "C"
int traceRegisterUserEvent(const char*)
{
#if CMK_OPTIMIZE
  OPTIMIZED_VERSION
  return 0;
#else
  if(CmiMyPe()==0)
    return _numEvents++;
  else
    return 0;
#endif
}

extern "C"
void traceClearEps(void)
{
  // In trace-summary, this zeros out the EP bins, to eliminate noise
  // from startup.  Here, this isn't useful, since we can do that in
  // post-processing
}

extern "C"
void traceWriteSts(void)
{
  if(CmiMyPe()==0)
    CpvAccess(_logPool)->writeSts();
}

extern "C"
void traceClose(void)
{
  CpvAccess(_trace)->endComputation();
/*
  if(CmiMyPe()==0)
    CpvAccess(_logPool)->writeSts();
*/
  delete CpvAccess(_logPool);
  delete CpvAccess(_trace);
  free(CpvAccess(traceRoot));
}

LogPool::LogPool(char *pgm, int b) {
  binary = b;
  pool = new LogEntry[CpvAccess(CtrLogBufSize)];
  numEntries = 0;
  poolSize = CpvAccess(CtrLogBufSize);
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
#if CMK_TRACE_LOGFILE_NUM_CONTROL
  CLOSE_LOG 
#endif
  if(!binary) {
    OPEN_LOG
    fprintf(fp, "PROJECTIONS-RECORD\n");
    CLOSE_LOG 
  }
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
  char *fname = new char[strlen(CpvAccess(traceRoot))+strlen(".sts")+1];
  sprintf(fname, "%s.sts", CpvAccess(traceRoot));
  FILE *sts;
  do
  {
    sts = fopen(fname, "w");
  } while (!fp && (errno == EINTR || errno == EMFILE));
  if(sts==0)
    CmiAbort("Cannot open projections sts file for writing.\n");
  delete[] fname;
  fprintf(sts, "MACHINE %s\n",CMK_MACHINE_NAME);
  fprintf(sts, "PROCESSORS %d\n", CmiNumPes());
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

void LogPool::add(UChar type,UShort mIdx,UShort eIdx,double time,int event,int pe) 
{
  new (&pool[numEntries++])
    LogEntry(time, type, mIdx, eIdx, event, pe);
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
      fprintf(fp, "%d %d %u %d %d\n", mIdx, eIdx, (UInt) (time*1.0e6), event, pe);
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

void TraceProjections::userEvent(int e)
{
  CpvAccess(_logPool)->add(USER_EVENT, e, 0, CmiWallTimer(),curevent++,CmiMyPe());
}

void TraceProjections::creation(envelope *e, int num)
{
  if(e==0) {
    CtvAccess(curThreadEvent)=curevent;
    CpvAccess(_logPool)->add(CREATION,ForChareMsg,_threadEP,CmiWallTimer(),
                             curevent++,CmiMyPe());
  } else {
    int type=e->getMsgtype();
    e->setEvent(curevent);
    for(int i=0; i<num; i++) {
      CpvAccess(_logPool)->add(CREATION,type,e->getEpIdx(),CmiWallTimer(),
                               curevent+i,CmiMyPe());
    }
    curevent += num;
  }
}

void TraceProjections::beginExecute(envelope *e)
{
  if(e==0) {
    execEvent = CtvAccess(curThreadEvent);
    execEp = (-1);
    CpvAccess(_logPool)->add(BEGIN_PROCESSING,ForChareMsg,_threadEP,CmiWallTimer(),
                             execEvent,CmiMyPe());
  } else {
    execEvent = e->getEvent();
    int type=e->getMsgtype();
    if(type==BocInitMsg)
      execEvent += CmiMyPe();
    execPe = e->getSrcPe();
    execEp = e->getEpIdx();
    CpvAccess(_logPool)->add(BEGIN_PROCESSING,type,execEp,CmiWallTimer(),
                             execEvent,execPe);
  }
}

void TraceProjections::endExecute(void)
{
  if(execEp == (-1)) {
    CpvAccess(_logPool)->add(END_PROCESSING,0,_threadEP,CmiWallTimer(),
                             execEvent,CmiMyPe());
  } else {
    CpvAccess(_logPool)->add(END_PROCESSING,0,execEp,CmiWallTimer(),
                             execEvent,execPe);
  }
}

void TraceProjections::beginIdle(void)
{
  if (isIdle == 0) {
    CpvAccess(_logPool)->add(BEGIN_IDLE, 0, 0, CmiWallTimer(), 0, CmiMyPe());
    isIdle = 1;
  }
}

void TraceProjections::endIdle(void)
{
  if (isIdle) {
    CpvAccess(_logPool)->add(END_IDLE, 0, 0, CmiWallTimer(), 0, CmiMyPe());
    isIdle = 0;
  }
}

void TraceProjections::beginPack(void)
{
  CpvAccess(_logPool)->add(BEGIN_PACK, 0, 0, CmiWallTimer(), 0, CmiMyPe());
}

void TraceProjections::endPack(void)
{
  CpvAccess(_logPool)->add(END_PACK, 0, 0, CmiWallTimer(), 0, CmiMyPe());
}

void TraceProjections::beginUnpack(void)
{
  CpvAccess(_logPool)->add(BEGIN_UNPACK, 0, 0, CmiWallTimer(), 0, CmiMyPe());
}

void TraceProjections::endUnpack(void)
{
  CpvAccess(_logPool)->add(END_UNPACK, 0, 0, CmiWallTimer(), 0, CmiMyPe());
}

void TraceProjections::beginCharmInit(void) {}

void TraceProjections::endCharmInit(void) {}

void TraceProjections::enqueue(envelope *) {}

void TraceProjections::dequeue(envelope *) {}

void TraceProjections::beginComputation(void)
{
  if(CmiMyRank()==0) {
    _threadMsg = CkRegisterMsg("dummy_thread_msg", 0, 0, 0, 0);
    _threadChare = CkRegisterChare("dummy_thread_chare", 0);
    _threadEP = CkRegisterEp("dummy_thread_ep", 0, _threadMsg,_threadChare);
  }
  CpvAccess(_logPool)->add(BEGIN_COMPUTATION, 0, 0, CmiWallTimer(), -1, -1);
}

void TraceProjections::endComputation(void)
{
  CpvAccess(_logPool)->add(END_COMPUTATION, 0, 0, CmiWallTimer(), -1, -1);
}
