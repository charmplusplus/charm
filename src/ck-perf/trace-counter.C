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

#ifdef CMK_ORIGIN2000

#include "trace-counter.h"

#define DEBUGF(x) CmiPrintf x
#define VER 1.0

CpvStaticDeclare(Trace*, _trace);
CpvStaticDeclare(CountLogPool*, _logPool);
CpvStaticDeclare(char*, pgmName);
static int _numEvents = 0;
static int _threadMsg, _threadChare, _threadEP;
static int _packMsg, _packChare, _packEP;
static int _unpackMsg, _unpackChare, _unpackEP;
CpvDeclare(double, binSize);
CpvDeclare(double, version);

static char* helpString =
"0 = Cycles
1 = Issued instructions
2 = Issued loads
3 = Issued stores
4 = Issued store conditionals
5 = Failed store conditionals
6 = Decoded branches.  (This changes meaning in 3.x
    versions of R10000.  It becomes resolved branches).
7 = Quadwords written back from secondary cache
8 = Correctable secondary cache data array ECC errors
9 = Primary (L1) instruction cache misses
10 = Secondary (L2) instruction cache misses
11 = Instruction misprediction from secondary cache way prediction table
12 = External interventions
13 = External invalidations
14 = Virtual coherency conditions.  (This changes meaning in 3.x
     versions of R10000.  It becomes ALU/FPU forward progress
     cycles.  On the R12000, this counter is always 0).
15 = Graduated instructions
16 = Cycles
17 = Graduated instructions
18 = Graduated loads
19 = Graduated stores
20 = Graduated store conditionals
21 = Graduated floating point instructions
22 = Quadwords written back from primary data cache
23 = TLB misses
24 = Mispredicted branches
25 = Primary (L1) data cache misses
26 = Secondary (L2) data cache misses
27 = Data misprediction from secondary cache way prediction table
28 = External intervention hits in secondary cache (L2)
29 = External invalidation hits in secondary cache
30 = Store/prefetch exclusive to clean block in secondary cache
31 = Store/prefetch exclusive to shared block in secondary cache";

void _createTracecounter(char **argv)
{
  DEBUGF(("%d createTraceCounter\n", CkMyPe()));
  CpvInitialize(Trace*, _trace);
  CpvAccess(_trace) = new  TraceCounter();
  CpvAccess(_traces)->addTrace(CpvAccess(_trace));
}

StatTable::StatTable(): stats_(NULL), numStats_(0) 
{
  stats_ = new Statistics[2];
  _MEMCHECK(stats_);
}

StatTable::~StatTable() { if (stats_ != NULL) { delete [] stats_; } }

CountLogPool::CountLogPool(char* pgm)
{
  int i;
  char pestr[10];
  sprintf(pestr, "%d", CkMyPe());
  int len = strlen(pgm) + strlen(".sum.") + strlen(pestr) + 1;
  char* fname = new char[len+1];
  sprintf(fname, "%s.%s.sum", pgm, pestr);
  fp_ = NULL;
  // CmiPrintf("TRACE: %s:%d\n", fname, errno);
  do {
    fp_ = fopen(fname, "w+");
  } while (!fp && errno == EINTR);
  delete[] fname;
  if(!fp) {
    CmiAbort("Cannot open Summary Trace File for writing...\n");
  }
}

CountLogPool::~CountLogPool() 
{
  write();
  fclose(fp_);
}

void CountLogPool::write(void) 
{
  int i;
  unsigned int j;
  fprintf(fp_, "ver:%3.1f %d/%d ep:%d\n", 
	  CpvAccess(version), CmiMyPe(), CmiNumPes(), _numEntries);
  stats_.write(fp_);
}

void CountLogPool::writeSts(void)
{
  char *fname = new char[strlen(CpvAccess(pgmName))+strlen(".sts")+1];
  sprintf(fname, "%s.count.sts", CpvAccess(pgmName));
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

void CountLogPool::setEp(int epidx, intdouble time) 
{
  if (epidx >= MAX_ENTRIES) {
    CmiAbort("CountLogPool::setEp too many entry points!\n");
  }
  // CmiPrintf("set EP: %d %e \n", epidx, time);
  epTime[epidx] += time;
  epCount[epidx] ++;
  // set phase table counter
}

void TraceCounter::traceInit(char **argv)
{
  char* tmpStr;
  CpvInitialize(CountLogPool*, _logPool);
  CpvInitialize(char*, pgmName);
  CpvInitialize(double, version);
  CpvAccess(pgmName) = (char *) malloc(strlen(argv[0])+1);
  _MEMCHECK(CpvAccess(pgmName));
  strcpy(CpvAccess(pgmName), argv[0]);
  CpvAccess(version) = VER;
  if (CmiGetArgString(argv,"+counter1",&tmpStr)) {
  }
  if (CmiGetArgString(argv,"+counter2",&tmpStr)) {
    sscanf(tmpStr,"%lf",&CpvAccess(binSize));
  }
  CpvAccess(_logPool) = new CountLogPool(CpvAccess(pgmName));
}

void TraceCounter::traceClearEps(void)
{
  CpvAccess(_logPool)->clearEps();
}

void TraceCounter::traceWriteSts(void)
{
  if (CmiMyPe()==0) { CpvAccess(_logPool)->writeSts(); }
}

void TraceCounter::traceClose(void)
{
  CpvAccess(_trace)->endComputation();
  if(CmiMyPe()==0) { CpvAccess(_logPool)->writeSts(); }
  // destructor call the write()
  delete CpvAccess(_logPool);
}

void TraceCounter::beginExecute(envelope *e)
{
  // no message means thread execution
  if (e==NULL) { beginExecute(-1,-1,_threadEP,-1); }
  else { beginExecute(-1,-1,e->getEpIdx(),-1); }  
}

void TraceCounter::beginExecute
(
  int event,
  int msgType,
  int ep,
  int srcPe, 
  int mlen
)
{
  execEP_=ep;
  startEP_=TraceTimer();
  // CmiPrintf("start: %f \n", start);
}

void TraceCounter::endExecute(void)
{
  // if (!flag) return;
  double t = TraceTimer();
  double ts = startTime_;

  if (execEp != -1) { CpvAccess(_logPool)->setEp(execEp, t-ts); }
}

void TraceCounter::beginPack(void) { startPack_ = CmiWallTimer(); }

void TraceCounter::endPack(void) {
  CpvAccess(_logPool)->setEp(_packEP, CmiWallTimer() - startPack_);
}

void TraceCounter::beginUnpack(void) { startUnpack_ = CmiWallTimer(); }

void TraceCounter::endUnpack(void) {
  CpvAccess(_logPool)->setEp(_unpackEP, CmiWallTimer()-startUnpack_);
}

void TraceCounter::beginComputation(void)
{
  if (CmiMyRank()==0) {
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

#endif // CMK_ORIGIN2000

/*@}*/
