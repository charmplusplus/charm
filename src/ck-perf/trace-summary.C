/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "trace-summary.h"

#define VER   3.0

CpvDeclare(Trace*, _trace);
CpvDeclare(int, CtrLogBufSize);
CpvStaticDeclare(LogPool*, _logPool);
CpvStaticDeclare(char*, pgmName);
static int _numEvents = 0;
static int _threadMsg, _threadChare, _threadEP;
static int _packMsg, _packChare, _packEP;
static int _unpackMsg, _unpackChare, _unpackEP;
CpvDeclare(double, binSize);
CpvDeclare(double, version);

extern "C" 
void traceInit(char **argv)
{
  char *tmpStr;
  traceCommonInit(argv,1);
  CpvInitialize(Trace*, _trace);
  CpvInitialize(LogPool*, _logPool);
  CpvInitialize(int, CtrLogBufSize);
  CpvInitialize(char*, pgmName);
  CpvInitialize(double, binSize);
  CpvInitialize(double, version);
  CpvAccess(_trace) = new TraceProjections();
  CpvAccess(pgmName) = (char *) malloc(strlen(argv[0])+1);
  _MEMCHECK(CpvAccess(pgmName));
  strcpy(CpvAccess(pgmName), argv[0]);
  CpvAccess(CtrLogBufSize) = LogBufSize;
  CpvAccess(binSize) = BIN_SIZE;
  CpvAccess(version) = VER;
  CmiGetArgInt(argv,"+logsize",&CpvAccess(CtrLogBufSize));
  if (CmiGetArgString(argv,"+binsize",&tmpStr))
  	sscanf(tmpStr,"%lf",&CpvAccess(binSize));
  if (CmiGetArgString(argv,"+version",&tmpStr))
  	sscanf(tmpStr,"%lf",&CpvAccess(version));
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
void traceWriteSts(void)
{
  if(CmiMyPe()==0)
      CpvAccess(_logPool)->writeSts();
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

extern "C"
void traceClearEps(void)
{
  CpvAccess(_logPool)->clearEps();
}

extern "C" 
void CkSummary_StartPhase(int phase)
{
   CpvAccess(_logPool)->startPhase(phase);
}


// for marks
extern "C" 
void CkSummary_MarkEvent(int eventType)
{
   CpvAccess(_logPool)->addEventType(eventType, CmiWallTimer());
}


PhaseEntry::PhaseEntry() 
{
  for (int i=0; i<MAX_ENTRIES; i++) {
    count[i] = 0;
    times[i] = 0.0;
  }
}

PhaseTable::PhaseTable(int n) : numPhase(n)
{
  phases = new PhaseEntry*[n];
  _MEMCHECK(phases);
  for (int i=0; i<n; i++) phases[i] = NULL;
  cur_phase = -1;
  phaseCalled = 0;
}

void LogPool::addEventType(int eventType, double time)
{
   if (eventType <0 || eventType >= MAX_MARKS) {
       CkPrintf("Invalid event type %d!\n", eventType);
       return;
   }
   MarkEntry *e = new MarkEntry;
   e->time = time;
   e->next = events[eventType].marks;
   events[eventType].marks = e;
   markcount ++;
}

LogPool::LogPool(char *pgm) : phaseTab(MAX_PHASES) 
{
    int i;
    poolSize = CpvAccess(CtrLogBufSize);
    if (poolSize % 2) poolSize++;	// make sure it is even
    pool = new LogEntry[poolSize];
    _MEMCHECK(pool);
    numEntries = 0;
    char pestr[10];
    sprintf(pestr, "%d", CkMyPe());
    int len = strlen(pgm) + strlen(".sum.") + strlen(pestr) + 1;
    char *fname = new char[len+1];
    sprintf(fname, "%s.%s.sum", pgm, pestr);
    fp = NULL;
    //CmiPrintf("TRACE: %s:%d\n", fname, errno);
    do {
    fp = fopen(fname, "w+");
    } while (!fp && errno == EINTR);
    delete[] fname;
    if(!fp) {
      CmiAbort("Cannot open Projections Trace File for writing...\n");
    }

    epSize = MAX_ENTRIES;
    epTime = new double[epSize];
    _MEMCHECK(epTime);
    epCount = new int[epSize];
    _MEMCHECK(epCount);
    for (i=0; i< epSize; i++) {
	epTime[i] = 0.0;
	epCount[i] = 0;
    };

    // event
    for (i=0; i<MAX_MARKS; i++) events[i].marks = NULL;
    markcount = 0;
}

void LogPool::write(void) 
{
  int i;
  unsigned int j;
  fprintf(fp, "ver:%3.1f %d/%d count:%d ep:%d interval:%e", CpvAccess(version), CmiMyPe(), CmiNumPes(), numEntries, _numEntries, CpvAccess(binSize));
  if (CpvAccess(version)>=3.0)
  {
    fprintf(fp, " phases:%d", phaseTab.numPhasesCalled());
  }
  fprintf(fp, "\n");
  // write bin time
  for(j=0; j<numEntries; j++)
    pool[j].write(fp);
  fprintf(fp, "\n");
  // write entry execution time
  for (i=0; i<_numEntries; i++)
    fprintf(fp, "%ld ", (long)(epTime[i]*1.0e6));
  fprintf(fp, "\n");
  // write entry function call times
  for (i=0; i<_numEntries; i++)
    fprintf(fp, "%d ", epCount[i]);
  fprintf(fp, "\n");
  // write marks
  if (CpvAccess(version)>=2.0) 
  {
  fprintf(fp, "%d ", markcount);
  for (i=0; i<MAX_MARKS; i++) {
    for(MarkEntry *e = events[i].marks; e; e=e->next)
        fprintf(fp, "%d %f ", i, e->time);
  }
  fprintf(fp, "\n");
  }
  // write phases
  if (CpvAccess(version)>=3.0)
  {
  phaseTab.write(fp);
  }
}

void LogPool::writeSts(void)
{
  char *fname = new char[strlen(CpvAccess(pgmName))+strlen(".sts")+1];
  sprintf(fname, "%s.sts", CpvAccess(pgmName));
  FILE *sts = fopen(fname, "w+");
  //CmiPrintf("File: %s \n", fname);
  if(sts==0)
    CmiAbort("Cannot open summary sts file for writing.\n");
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

void LogPool::add(double time, int pe) 
{
  new (&pool[numEntries++])
  LogEntry(time, pe);
  if(poolSize==numEntries) shrink();
}

void LogPool::setEp(int epidx, double time) 
{
  if (epidx >= epSize) {
        CmiAbort("Too many entry points!!\n");
  }
  //CmiPrintf("set EP: %d %e \n", epidx, time);
  epTime[epidx] += time;
  epCount[epidx] ++;
  // set phase table counter
  phaseTab.setEp(epidx, time);
}

void LogPool::shrink(void)
{
  int entries = numEntries/2;
  for (int i=0; i<entries; i++)
  {
     pool[i].setTime(pool[i*2].getTime() + pool[i*2+1].getTime());
  }
  numEntries = entries;
  CpvAccess(binSize) *= 2;

//CkPrintf("Shrinked binsize: %f entries:%d!!!!\n", CpvAccess(binSize), numEntries);
}

void LogEntry::write(FILE* fp)
{
  int per = (int)(time * 100.0 / CpvAccess(binSize));
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
  // no message means thread execution
  if (e==NULL) {
    beginExecute(-1,-1,_threadEP,-1);
  }
  else {
    beginExecute(-1,-1,e->getEpIdx(),-1);
  }  
}

void TraceProjections::beginExecute(int event,int msgType,int ep,int srcPe)
{
  execEp=ep;
  double t = CmiWallTimer();
//CmiPrintf("start: %f \n", start);

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
//  if (!flag) return;
  double t = CmiWallTimer();
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
     bin = 0;
     ts = nts;
  }
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
    packstart = CmiWallTimer();
}

void TraceProjections::endPack(void)
{
    CpvAccess(_logPool)->setEp(_packEP, CmiWallTimer() - packstart);
}

void TraceProjections::beginUnpack(void)
{
    unpackstart = CmiWallTimer();
}

void TraceProjections::endUnpack(void)
{
    CpvAccess(_logPool)->setEp(_unpackEP, CmiWallTimer()-unpackstart);
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

    _packMsg = CkRegisterMsg("dummy_pack_msg", 0, 0, 0, 0);
    _packChare = CkRegisterChare("dummy_pack_chare", 0);
    _packEP = CkRegisterEp("dummy_pack_ep", 0, _packMsg,_packChare);

    _unpackMsg = CkRegisterMsg("dummy_unpack_msg", 0, 0, 0, 0);
    _unpackChare = CkRegisterChare("dummy_unpack_chare", 0);
    _unpackEP = CkRegisterEp("dummy_unpack_ep", 0, _unpackMsg,_unpackChare);
  }
}

void TraceProjections::endComputation(void)
{
  if (msgNum==0) {
//CmiPrintf("Add at last: %d pe:%d time:%f msg:%d\n", index, CmiMyPe(), bin, msgNum);
     CpvAccess(_logPool)->add(bin, CmiMyPe());
     msgNum ++;

     binStart  += CpvAccess(binSize);
     double t = CmiWallTimer();
     double ts = binStart;
     while (ts < t)
     {
       CpvAccess(_logPool)->add(bin, CmiMyPe());
       bin=0.0;
       ts += CpvAccess(binSize);
     }
  }
}

