#include "trace-summary.h"

#define VER   "1.0"

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
    if(strcmp(argv[i], "+binsize")==0) {
      double d;
      sscanf(argv[i+1], "%le", &d);
      CpvAccess(binSize) = d;
      break;
    }
  }
  if(i!=*argc) { // +binsize parameter was found, delete it and its arg
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
  if(CmiMyPe()==0)
      CpvAccess(_logPool)->writeSts();
  // destructor call the write()
  delete CpvAccess(_logPool);
}

void LogPool::write(void) 
{
  int i;
  fprintf(fp, "%d/%d count:%d ep:%d interval:%le ver:%s\n", CmiMyPe(), CmiNumPes(), numEntries, _numEntries, CpvAccess(binSize), VER);
  for(i=0; i<numEntries; i++)
    pool[i].write(fp);
  fprintf(fp, "\n");
  for (i=0; i<_numEntries; i++)
    fprintf(fp, "%ld ", (long)(epTime[i]*1.0e6));
  fprintf(fp, "\n");
  for (i=0; i<_numEntries; i++)
    fprintf(fp, "%d ", epCount[i]);
  fprintf(fp, "\n");
}

void LogPool::writeSts(void)
{
  char *fname = new char[strlen(CpvAccess(pgmName))+strlen(".sts")+1];
  sprintf(fname, "%s.sts", CpvAccess(pgmName));
  FILE *sts = fopen(fname, "w+");
  //CmiPrintf("File: %s \n", fname);
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

void LogPool::shrink(void)
{
  for (int i=0; i<numEntries; i++)
  {
     pool[i].setTime(pool[i*2].getTime() + pool[i*2+1].getTime());
  }
  numEntries /= 2;
  CpvAccess(binSize) *= 2;
CkPrintf("Shrinked binsize: %f !!!!\n", CpvAccess(binSize));
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
  if (e==NULL) {
    execEp = (-1);
  }
  else {
    execEp = e->getEpIdx();
  }
  double t = CmiTimer();
//CmiPrintf("start: %f \n", start);
/*
  msgNum++;
  if (start == t) {
     return;
  }
*/
  start = t;
  double ts = binStart;
  // fill gaps
  while ((ts = ts + CpvAccess(binSize)) < t)
  {
     CpvAccess(_logPool)->add(bin, CmiMyPe());
     bin=0.0;
     binStart = ts;
  }
}

void TraceProjections::endExecute(void)
{
/*
  msgNum --;
  // duplicate messages
  if (msgNum != 0) return;
*/
//CmiPrintf("end:msgNum: %d bin:%f\n", msgNum, bin);
  double t = CmiTimer();
  double ts = start;
  double nts = binStart;

  if (execEp != -1)
  {
    CpvAccess(_logPool)->setEp(execEp, t-ts);
  }

  while ((nts = nts + CpvAccess(binSize)) < t)
  {
     bin += nts-ts;
     binStart  = nts;
     CpvAccess(_logPool)->add(bin, CmiMyPe());
//CmiPrintf("add time: %f\n", bin);
     bin = 0;
     ts = nts;
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
  if (msgNum==0) {
//CmiPrintf("Add at last: %d pe:%d time:%f msg:%d\n", index, CmiMyPe(), bin, msgNum);
     CpvAccess(_logPool)->add(bin, CmiMyPe());
     msgNum ++;
/*
     // fill gap till end of program
     int curIdx = CmiTimer() / CpvAccess(binSize);
     for (int i=index+1; i<=curIdx; i++) {
        CpvAccess(_logPool)->add(i, 0.0, CmiMyPe());
     }
*/
     binStart  += CpvAccess(binSize);
     double t = CmiTimer();
     double ts = binStart;
     while (ts < t)
     {
       CpvAccess(_logPool)->add(bin, CmiMyPe());
       bin=0.0;
       ts += CpvAccess(binSize);
     }
  }
}

/*
void TraceProjections::writeEvent(void)
{
  char pestr[10];
  sprintf(pestr, "%d", CkMyPe());
  int len = strlen(CpvAccess(pgmName)) + strlen(".eps.") + strlen(pestr) + 1;
  char *fname = new char[len];
  sprintf(fname, "%s.%s.eps", CpvAccess(pgmName), pestr);
  FILE *sts = fopen(fname, "w+");
  //CmiPrintf("File: %s \n", fname);
  if(sts==0)
    CmiAbort("Cannot open projections sts file for writing.\n");
  delete[] fname;
  for (int i=0; i<_numEntries; i++)
    fprintf(sts, "%d ", (int)(epTime[i]*1.0e6));
  fprintf(sts, "\n");
  for (int i=0; i<_numEntries; i++)
    fprintf(sts, "%d ", epCount[i]);
  fprintf(sts, "\n");
  fclose(sts);
}
*/
