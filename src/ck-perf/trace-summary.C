#include "trace-summary.h"

CpvDeclare(Trace*, _trace);
CpvDeclare(int, traceOn);
CpvDeclare(int, CtrLogBufSize);
CpvStaticDeclare(LogPool*, _logPool);
CpvStaticDeclare(char*, pgmName);
CpvExtern(CthThread, curThread);
static int _numEvents = 0;
static int _threadMsg, _threadChare, _threadEP;
CpvDeclare(double, binSize);

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
  CpvInitialize(double, binSize);
  CpvAccess(_trace) = new TraceProjections();
  CpvAccess(traceOn) = 1;
  CpvAccess(pgmName) = (char *) malloc(strlen(argv[0])+1);
  _MEMCHECK(CpvAccess(pgmName));
  strcpy(CpvAccess(pgmName), argv[0]);
  CpvAccess(CtrLogBufSize) = 10000;
  CpvAccess(binSize) = BIN_SIZE;
  int i;
  for(i=1;i<*argc;i++) {
    if(strcmp(argv[i], "+binsize")==0) {
      double d;
      sscanf(argv[i+1], "%le", &d);
      CpvAccess(binSize) = d;
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
}

extern "C"
void traceEndIdle(void)
{
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
}

extern "C"
int traceRegisterUserEvent(const char*)
{
  return 0;
}

extern "C"
void traceClose(void)
{
  CpvAccess(_trace)->endComputation();
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
  int per = time * 100.0 / CpvAccess(binSize);
//  fprintf(fp, "%d %f%% \n", index, per);
//  fprintf(fp, "%d %4d \n", index, per);
  fprintf(fp, "%4d", per);
}

void TraceProjections::userEvent(int e)
{
}

void TraceProjections::creation(envelope *e, int num)
{
}

void TraceProjections::beginExecute(envelope *e)
{
  double t = CmiTimer();
  msgNum++;
  if (start == t) {
     return;
  }
  start = t;
  int oldIdx = index;
  index = t / CpvAccess(binSize);
  // fill gaps
  for (int i=oldIdx; i<index; i++) {
     CpvAccess(_logPool)->add(i, bin, CmiMyPe());
     bin=0.0;
  }
//CmiPrintf("start: %f index: %d\n", start, index);
}

void TraceProjections::endExecute(void)
{
  msgNum --;
//CmiPrintf("end:msgNum: %d index: %d bin:%f\n", msgNum, index, bin);
  // duplicate messages
  if (msgNum != 0) return;
  double t = CmiTimer();
  double ts = start;
  double nts = index*CpvAccess(binSize);
  while ((nts = nts + CpvAccess(binSize)) < t)
  {
     bin += nts-ts;
     CpvAccess(_logPool)->add(index, bin, CmiMyPe());
//CmiPrintf("add index: %d time: %f\n", index,bin);
     bin = 0;
     ts = nts;
     index++;
  }
//  CpvAccess(_logPool)->add(index, t - ts, CmiMyPe());
  bin += t - ts;
}

void TraceProjections::beginIdle(void)
{
}

void TraceProjections::endIdle(void)
{
}

void TraceProjections::beginPack(void)
{
}

void TraceProjections::endPack(void)
{
}

void TraceProjections::beginUnpack(void)
{
}

void TraceProjections::endUnpack(void)
{
}

void TraceProjections::beginCharmInit(void) {}

void TraceProjections::endCharmInit(void) {}

void TraceProjections::enqueue(envelope *) {}

void TraceProjections::dequeue(envelope *) {}

void TraceProjections::beginComputation(void)
{
}

void TraceProjections::endComputation(void)
{
  if (index!= -1 && msgNum==0) {
//CmiPrintf("Add at last: %d pe:%d time:%f msg:%d\n", index, CmiMyPe(), bin, msgNum);
     CpvAccess(_logPool)->add(index, bin, CmiMyPe());
     msgNum ++;
     // fill gap till end of program
     int curIdx = CmiTimer() / CpvAccess(binSize);
     for (int i=index+1; i<=curIdx; i++) {
        CpvAccess(_logPool)->add(i, 0.0, CmiMyPe());
     }
  }
}

