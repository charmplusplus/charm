/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "trace-projections.h"

CpvDeclare(Trace*, _trace);
CpvDeclare(int, traceOn);
CpvDeclare(int, CtrLogBufSize);
CpvStaticDeclare(LogPool*, _logPool);
CpvStaticDeclare(char*, pgmName);
CpvExtern(CthThread, curThread);
static int _numEvents = 0;
static int _threadMsg, _threadChare, _threadEP;

extern "C" void setEvent(CthThread t, int event);
extern "C" int getEvent(CthThread t);

extern "C" 
void traceInit(int* argc, char **argv)
{
  CpvInitialize(Trace*, _trace);
  CpvInitialize(LogPool*, _logPool);
  CpvInitialize(int, traceOn);
  CpvInitialize(int, CtrLogBufSize);
  CpvInitialize(char*, pgmName);
  CpvAccess(_trace) = new TraceProjections();
  CpvAccess(traceOn) = 1;
  CpvAccess(pgmName) = (char *) malloc(strlen(argv[0])+1);
  _MEMCHECK(CpvAccess(pgmName));
  strcpy(CpvAccess(pgmName), argv[0]);
  CpvAccess(CtrLogBufSize) = 10000;
  int i;
  for(i=1;i<*argc;i++) {
    if(strcmp(argv[i], "+logsize")==0) {
      CpvAccess(CtrLogBufSize) = atoi(argv[i+1]);
      break;
    }
  }
  if(i!=*argc) { // +logsize parameter was found, delete it and its arg
    while((i+2)<= *argc) {
      argv[i] = argv[i+2];
      i++;
    }
    *argc -= 2;
  }
  for(i=1;i<*argc;i++) {
    if(strcmp(argv[i], "+traceoff")==0) {
      CpvAccess(traceOn) = 0;
      break;
    }
  }
  if(i!=*argc) { // +traceoff parameter was found, delete it
    while((i+1)<= *argc) {
      argv[i] = argv[i+2];
      i++;
    }
    *argc -= 2;
  }
  CpvAccess(_logPool) = new LogPool(CpvAccess(pgmName));
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
void traceAwaken(void)
{
  CpvAccess(_trace)->creation(0);
}

extern "C"
void traceUserEvent(int e)
{
  CpvAccess(_trace)->userEvent(e);
}

extern "C"
int traceRegisterUserEvent(const char*)
{
  if(CmiMyPe()==0)
    return _numEvents++;
  else
    return 0;
}

extern "C"
void traceClearEps(void)
{
}

extern "C"
void traceClose(void)
{
  CpvAccess(_trace)->endComputation();
  if(CmiMyPe()==0)
    CpvAccess(_logPool)->writeSts();
  delete CpvAccess(_logPool);
}

void LogPool::writeSts(void)
{
  char *fname = new char[strlen(CpvAccess(pgmName))+strlen(".sts")+1];
  sprintf(fname, "%s.sts", CpvAccess(pgmName));
  FILE *sts = fopen(fname, "w");
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

void TraceProjections::userEvent(int e)
{
  CpvAccess(_logPool)->add(USER_EVENT, e, 0, CmiTimer(),curevent++,CmiMyPe());
}

void TraceProjections::creation(envelope *e, int num)
{
  if(e==0) {
    setEvent(CpvAccess(curThread),curevent);
    CpvAccess(_logPool)->add(CREATION,ForChareMsg,_threadEP,CmiTimer(),
                             curevent++,CmiMyPe());
  } else {
    int type=e->getMsgtype();
    e->setEvent(curevent);
    for(int i=0; i<num; i++) {
      CpvAccess(_logPool)->add(CREATION,type,e->getEpIdx(),CmiTimer(),
                               curevent+i,CmiMyPe());
    }
    curevent += num;
  }
}

void TraceProjections::beginExecute(envelope *e)
{
  if(e==0) {
    execEvent = getEvent(CpvAccess(curThread));
    execEp = (-1);
    CpvAccess(_logPool)->add(BEGIN_PROCESSING,ForChareMsg,_threadEP,CmiTimer(),
                             execEvent,CmiMyPe());
  } else {
    execEvent = e->getEvent();
    int type=e->getMsgtype();
    if(type==BocInitMsg)
      execEvent += CmiMyPe();
    execPe = e->getSrcPe();
    execEp = e->getEpIdx();
    CpvAccess(_logPool)->add(BEGIN_PROCESSING,type,execEp,CmiTimer(),
                             execEvent,execPe);
  }
}

void TraceProjections::endExecute(void)
{
  if(execEp == (-1)) {
    CpvAccess(_logPool)->add(END_PROCESSING,0,_threadEP,CmiTimer(),
                             execEvent,CmiMyPe());
  } else {
    CpvAccess(_logPool)->add(END_PROCESSING,0,execEp,CmiTimer(),
                             execEvent,execPe);
  }
}

void TraceProjections::beginIdle(void)
{
  CpvAccess(_logPool)->add(BEGIN_IDLE, 0, 0, CmiTimer(), 0, CmiMyPe());
}

void TraceProjections::endIdle(void)
{
  CpvAccess(_logPool)->add(END_IDLE, 0, 0, CmiTimer(), 0, CmiMyPe());
}

void TraceProjections::beginPack(void)
{
  CpvAccess(_logPool)->add(BEGIN_PACK, 0, 0, CmiTimer(), 0, CmiMyPe());
}

void TraceProjections::endPack(void)
{
  CpvAccess(_logPool)->add(END_PACK, 0, 0, CmiTimer(), 0, CmiMyPe());
}

void TraceProjections::beginUnpack(void)
{
  CpvAccess(_logPool)->add(BEGIN_UNPACK, 0, 0, CmiTimer(), 0, CmiMyPe());
}

void TraceProjections::endUnpack(void)
{
  CpvAccess(_logPool)->add(END_UNPACK, 0, 0, CmiTimer(), 0, CmiMyPe());
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
  CpvAccess(_logPool)->add(BEGIN_COMPUTATION, 0, 0, CmiTimer(), -1, -1);
}

void TraceProjections::endComputation(void)
{
  CpvAccess(_logPool)->add(END_COMPUTATION, 0, 0, CmiTimer(), -1, -1);
}
