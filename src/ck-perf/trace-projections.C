#include "trace-projections.h"

CpvDeclare(Trace*, _trace);
CpvDeclare(int, traceOn);
CpvDeclare(int, CtrLogBufSize);
CpvStaticDeclare(LogPool*, _logPool);
CpvStaticDeclare(char*, pgmName);
static int _numEvents = 0;

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
  strcpy(CpvAccess(pgmName), argv[0]);
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
}

extern "C"
void traceSuspend(void)
{
}

extern "C"
void traceAwaken(void)
{
}

extern "C"
void traceUserEvent(int e)
{
  CpvAccess(_trace)->userEvent(e);
}

extern "C"
int traceRegisterUserEvent(const char*)
{
  if(CkMyPe()==0)
    return _numEvents++;
  else
    return 0;
}

extern "C"
void traceClose(void)
{
  if(CkMyPe()==0)
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

void LogEntry::write(FILE* fp)
{
  fprintf(fp, "%d ", type);

  switch (type) {
    case USER_EVENT:
      fprintf(fp, "%d %u %d %d", mIdx, (int) (time*1.0e6), event, pe);
      break;

    case BEGIN_IDLE:
    case END_IDLE:
    case BEGIN_PACK:
    case END_PACK:
    case BEGIN_UNPACK:
    case END_UNPACK:
      fprintf(fp, "%u %d", (int) (time*1.0e6), pe);
      break;

    case CREATION:
    case BEGIN_PROCESSING:
    case END_PROCESSING:
      fprintf(fp, "%d %d %u %d %d", mIdx, eIdx, (int) (time*1.0e6), event, pe);
      break;

    case ENQUEUE:
    case DEQUEUE:
      fprintf(fp, "%d %u %d %d", mIdx, (int) (time*1.0e6), event, pe);
      break;

    case BEGIN_INTERRUPT:
    case END_INTERRUPT:
      fprintf(fp, "%u %d %d", (int) (time*1.0e6), event, pe);
      break;

    case BEGIN_COMPUTATION:
    case END_COMPUTATION:
    fprintf(fp, "%u", (int) (time*1.0e6));
      break;

    default:
      CkError("***Internal Error*** Wierd Event %d.\n", type);
      break;
  }
}

void TraceProjections::userEvent(int e)
{
  CpvAccess(_logPool)->add(USER_EVENT, e, -1, CkTimer(),curevent++,CkMyPe());
}

void TraceProjections::creation(envelope *e)
{
}

void TraceProjections::beginExecute(envelope *e)
{
}

void TraceProjections::endExecute(void)
{
}

void TraceProjections::beginIdle(void)
{
  CpvAccess(_logPool)->add(BEGIN_IDLE, 0, 0, CkTimer(), 0, CkMyPe());
}

void TraceProjections::endIdle(void)
{
  CpvAccess(_logPool)->add(END_IDLE, 0, 0, CkTimer(), 0, CkMyPe());
}

void TraceProjections::beginPack(void)
{
  CpvAccess(_logPool)->add(BEGIN_PACK, 0, 0, CkTimer(), 0, CkMyPe());
}

void TraceProjections::endPack(void)
{
  CpvAccess(_logPool)->add(END_PACK, 0, 0, CkTimer(), 0, CkMyPe());
}

void TraceProjections::beginUnpack(void)
{
  CpvAccess(_logPool)->add(BEGIN_UNPACK, 0, 0, CkTimer(), 0, CkMyPe());
}

void TraceProjections::endUnpack(void)
{
  CpvAccess(_logPool)->add(END_UNPACK, 0, 0, CkTimer(), 0, CkMyPe());
}

void TraceProjections::beginCharmInit(void)
{
}

void TraceProjections::endCharmInit(void)
{
}

void TraceProjections::enqueue(envelope *e)
{
}

void TraceProjections::dequeue(envelope *e)
{
}

void TraceProjections::beginComputation(void)
{
  CpvAccess(_logPool)->add(BEGIN_COMPUTATION, -1, -1, CkTimer(), -1, -1);
}

void TraceProjections::endComputation(void)
{
  CpvAccess(_logPool)->add(END_COMPUTATION, -1, -1, CkTimer(), -1, -1);
}
