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
#include "trace-summary.h"
#include "trace-summaryBOC.h"

#define DEBUGF(x)  // CmiPrintf x

#define VER   4.0

#define INVALIDEP     -2

CkpvStaticDeclare(TraceSummary*, _trace);
static int _numEvents = 0;
static int _threadMsg, _threadChare, _threadEP;
static int _packMsg, _packChare, _packEP;
static int _unpackMsg, _unpackChare, _unpackEP;
CkpvDeclare(double, binSize);
CkpvDeclare(double, version);

int sumonly = 0;

/**
  For each TraceFoo module, _createTraceFoo() must be defined.
  This function is called in _createTraces() generated in moduleInit.C
*/
void _createTracesummary(char **argv)
{
  DEBUGF(("%d createTraceSummary\n", CkMyPe()));
  CkpvInitialize(TraceSummary*, _trace);
  CkpvAccess(_trace) = new  TraceSummary(argv);
  CkpvAccess(_traces)->addTrace(CkpvAccess(_trace));
}


/// function call for starting a phase in trace summary logs 
extern "C" 
void CkSummary_StartPhase(int phase)
{
   CkpvAccess(_trace)->startPhase(phase);
}


/// function call for adding an event mark
extern "C" 
void CkSummary_MarkEvent(int eventType)
{
   CkpvAccess(_trace)->addEventType(eventType);
}


PhaseEntry::PhaseEntry() 
{
  nEPs = _numEntries+10;
  count = new int[nEPs];
  times = new double[nEPs];
  maxtimes = new double[nEPs];
  for (int i=0; i<nEPs; i++) {
    count[i] = 0;
    times[i] = 0.0;
    maxtimes[i] = 0.;
  }
}

SumLogPool::~SumLogPool() 
{
  if (!sumonly) {
  write();
  fclose(fp);
  }
  // free memory for mark
  if (markcount > 0)
  for (int i=0; i<MAX_MARKS; i++) {
    for (int j=0; j<events[i].length(); j++)
      delete events[i][j];
  }
  delete[] pool;
  delete[] epInfo;
}

void SumLogPool::addEventType(int eventType, double time)
{
   if (eventType <0 || eventType >= MAX_MARKS) {
       CkPrintf("Invalid event type %d!\n", eventType);
       return;
   }
   MarkEntry *e = new MarkEntry;
   e->time = time;
   events[eventType].push_back(e);
   markcount ++;
}

SumLogPool::SumLogPool(char *pgm) : numBins(0), phaseTab(MAX_PHASES) 
{
   if (TRACE_CHARM_PE() == 0) return;
   int i;
   poolSize = CkpvAccess(CtrLogBufSize);
   if (poolSize % 2) poolSize++;	// make sure it is even
   pool = new BinEntry[poolSize];
   _MEMCHECK(pool);
   char pestr[10];
   sprintf(pestr, "%d", CkMyPe());
   int len = strlen(pgm) + strlen(".sum.") + strlen(pestr) + 1;
   char *fname = new char[len+1];
   sprintf(fname, "%s.%s.sum", pgm, pestr);
   fp = NULL;
   //CmiPrintf("TRACE: %s:%d\n", fname, errno);
   if (!sumonly) {
    do {
      fp = fopen(fname, "w+");
    } while (!fp && errno == EINTR);
    delete[] fname;
    if(!fp) {
      CmiAbort("Cannot open Summary Trace File for writing...\n");
    }
   }

   // event
   markcount = 0;

   if (CkMyPe() == 0)
   {
    char *fname = new char[strlen(CkpvAccess(traceRoot))+strlen(".sum.sts")+1];
    sprintf(fname, "%s.sum.sts", CkpvAccess(traceRoot));
    stsfp = fopen(fname, "w+");
    //CmiPrintf("File: %s \n", fname);
    if(stsfp == 0)
      CmiAbort("Cannot open summary sts file for writing.\n");
    delete[] fname;
   }
}

void SumLogPool::initMem()
{
   epInfoSize = _numEntries+10;
   epInfo = new SumEntryInfo[epInfoSize];
   _MEMCHECK(epInfo);
}

void SumLogPool::write(void) 
{
  int i;
  unsigned int j;

  fprintf(fp, "ver:%3.1f %d/%d count:%d ep:%d interval:%e", CkpvAccess(version), CkMyPe(), CkNumPes(), numBins, _numEntries, CkpvAccess(binSize));
  if (CkpvAccess(version)>=3.0)
  {
    fprintf(fp, " phases:%d", phaseTab.numPhasesCalled());
  }
  fprintf(fp, "\n");

  // write bin time
#if 1
  int last=pool[0].getU();
  pool[0].writeU(fp, last);
  int count=1;
  for(j=1; j<numBins; j++) {
    int u = pool[j].getU();
    if (last == u) {
      count++;
    }
    else {
      if (count > 1) fprintf(fp, "+%d", count);
      pool[j].writeU(fp, u);
      last = u;
      count = 1;
    }
  }
  if (count > 1) fprintf(fp, "+%d", count);
#else
  for(j=0; j<numEntries; j++) 
      pool[j].write(fp);
#endif
  fprintf(fp, "\n");

  // write entry execution time
  fprintf(fp, "EPExeTime: ");
  for (i=0; i<_numEntries; i++)
    fprintf(fp, "%ld ", (long)(epInfo[i].epTime*1.0e6));
  fprintf(fp, "\n");
  // write entry function call times
  fprintf(fp, "EPCallTime: ");
  for (i=0; i<_numEntries; i++)
    fprintf(fp, "%d ", epInfo[i].epCount);
  fprintf(fp, "\n");
  // write max entry function execute times
  fprintf(fp, "MaxEPTime: ");
  for (i=0; i<_numEntries; i++)
    fprintf(fp, "%ld ", (long)(epInfo[i].epMaxTime*1.0e6));
  fprintf(fp, "\n");
#if 0
  for (i=0; i<SumEntryInfo::HIST_SIZE; i++) {
    for (j=0; j<_numEntries; j++) 
      fprintf(fp, "%d ", epInfo[j].hist[i]);
    fprintf(fp, "\n");
  }
#endif
  // write marks
  if (CkpvAccess(version)>=2.0) 
  {
    fprintf(fp, "NumMarks: %d ", markcount);
    for (i=0; i<MAX_MARKS; i++) {
      for(int j=0; j<events[i].length(); j++)
        fprintf(fp, "%d %f ", i, events[i][j]->time);
    }
    fprintf(fp, "\n");
  }
  // write phases
  if (CkpvAccess(version)>=3.0)
  {
    phaseTab.write(fp);
  }
}

void SumLogPool::writeSts(void)
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
    fprintf(stsfp, "EVENT %d Event%d\n", i, i);
  fprintf(stsfp, "END\n");
  fclose(stsfp);
}

void SumLogPool::add(double time, int pe) 
{
  new (&pool[numBins++]) BinEntry(time);
  if(poolSize==numBins) shrink();
}

void SumLogPool::setEp(int epidx, double time) 
{
  if (epidx >= epInfoSize) {
        CmiAbort("Invalid entry point!!\n");
  }
  //CmiPrintf("set EP: %d %e \n", epidx, time);
  epInfo[epidx].setTime(time);
  // set phase table counter
  phaseTab.setEp(epidx, time);
}

void SumLogPool::shrink(void)
{
//  double t = CmiWallTimer();

  int entries = numBins/2;
  for (int i=0; i<entries; i++)
  {
     pool[i].setTime(pool[i*2].getTime() + pool[i*2+1].getTime());
  }
  numBins = entries;
  CkpvAccess(binSize) *= 2;

//CkPrintf("Shrinked binsize: %f entries:%d takes %fs!!!!\n", CkpvAccess(binSize), numEntries, CmiWallTimer()-t);
}

int  BinEntry::getU() { 
  return (int)(time * 100.0 / CkpvAccess(binSize)); 
}

void BinEntry::write(FILE* fp)
{
  int per = (int)(time * 100.0 / CkpvAccess(binSize));
  fprintf(fp, "%4d", per);
}

void BinEntry::writeU(FILE* fp, int u)
{
  fprintf(fp, "%4d", u);
}

TraceSummary::TraceSummary(char **argv):curevent(0),binStart(0.0),bin(0.0),msgNum(0)
{
  char *tmpStr;
  CkpvInitialize(double, binSize);
  CkpvInitialize(double, version);
  CkpvAccess(binSize) = BIN_SIZE;
  CkpvAccess(version) = VER;
  CmiGetArgDoubleDesc(argv,"+binsize",&CkpvAccess(binSize),
  	"CPU usage log time resolution");
  CmiGetArgDoubleDesc(argv,"+version",&CkpvAccess(version),
  	"Write this .sum file version");
  
  epThreshold = 0.001; 
  CmiGetArgDoubleDesc(argv,"+epThreshold",&epThreshold,
  	"Execution time histogram lower bound");
  epInterval = 0.001; 
  CmiGetArgDoubleDesc(argv,"+epInterval",&epInterval,
  	"Execution time histogram bin size");

  sumonly = CmiGetArgFlagDesc(argv, "+sumonly", "merge histogram bins on processor 0");

  _logPool = new SumLogPool(CkpvAccess(traceRoot));
  execEp=INVALIDEP;
}

void TraceSummary::traceClearEps(void)
{
  _logPool->clearEps();
}

void TraceSummary::traceWriteSts(void)
{
  if(CkMyPe()==0)
      _logPool->writeSts();
}

void TraceSummary::traceClose(void)
{
  if(CkMyPe()==0)
      _logPool->writeSts();
  if (TRACE_CHARM_PE()) {
    CkpvAccess(_trace)->endComputation();
    // destructor call the write()
    delete _logPool;
  }
}

void TraceSummary::beginExecute(envelope *e)
{
  // no message means thread execution
  if (e==NULL) {
    beginExecute(-1,-1,_threadEP,-1);
  }
  else {
    beginExecute(-1,-1,e->getEpIdx(),-1);
  }  
}

void TraceSummary::beginExecute(int event,int msgType,int ep,int srcPe, int mlen)
{
  if (execEp != INVALIDEP) {
    CmiPrintf("Warning: TraceSummary two consecutive BEGIN_PROCESSING!\n");
    return;
  }

  execEp=ep;
  double t = TraceTimer();
//CmiPrintf("start: %f \n", start);

  start = t;
  double ts = binStart;
  // fill gaps
  while ((ts = ts + CkpvAccess(binSize)) < t)
  {
     _logPool->add(bin, CkMyPe());
     bin=0.0;
     binStart = ts;
  }
}

void TraceSummary::endExecute(void)
{
//  if (!flag) return;
  double t = TraceTimer();
  double ts = start;
  double nts = binStart;

  if (execEp == INVALIDEP) {
    CmiPrintf("Warning: TraceSummary END_PROCESSING without BEGIN_PROCESSING!\n");
    return;
  }

  if (execEp != -1)
  {
    _logPool->setEp(execEp, t-ts);
  }

  while ((nts = nts + CkpvAccess(binSize)) < t)
  {
     bin += nts-ts;
     binStart  = nts;
     _logPool->add(bin, CkMyPe());
     bin = 0;
     ts = nts;
  }
  bin += t - ts;
  execEp = INVALIDEP;
}

void TraceSummary::beginPack(void)
{
    packstart = CmiWallTimer();
}

void TraceSummary::endPack(void)
{
    _logPool->setEp(_packEP, CmiWallTimer() - packstart);
}

void TraceSummary::beginUnpack(void)
{
    unpackstart = CmiWallTimer();
}

void TraceSummary::endUnpack(void)
{
    _logPool->setEp(_unpackEP, CmiWallTimer()-unpackstart);
}

void TraceSummary::beginComputation(void)
{
  if(CkMyRank()==0) {
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

  // initialze arrays becasue now the number of entries is known.
  _logPool->initMem();
}

void TraceSummary::endComputation(void)
{
  static int done = 0;
  if (done) return;
  done = 1;
  if (msgNum==0) {
//CmiPrintf("Add at last: %d pe:%d time:%f msg:%d\n", index, CkMyPe(), bin, msgNum);
     _logPool->add(bin, CkMyPe());
     msgNum ++;

     binStart  += CkpvAccess(binSize);
     double t = TraceTimer();
     double ts = binStart;
     while (ts < t)
     {
       _logPool->add(bin, CkMyPe());
       bin=0.0;
       ts += CkpvAccess(binSize);
     }
  }
}

void TraceSummary::addEventType(int eventType)
{
  _logPool->addEventType(eventType, TraceTimer());
}

void TraceSummary::startPhase(int phase)
{
   _logPool->startPhase(phase);
}


/// for TraceSummaryBOC

CkGroupID traceSummaryGID;

void TraceSummaryBOC::askSummary()
{
  if (CkpvAccess(_trace) == NULL) return;

#if 0
#if CMK_TRACE_IN_CHARM
  TRACE_CHARM_PE() = 0;
#endif
#endif

  int n=0;
  BinEntry *bins = NULL;
  int traced = TRACE_CHARM_PE();
  if (traced) {
  CkpvAccess(_trace)->endComputation();
  n = CkpvAccess(_trace)->pool()->getNumEntries();
  bins = CkpvAccess(_trace)->pool()->bins();
#if 1
  CmiPrintf("askSummary on [%d] numEntried=%d\n", CkMyPe(), n);
#if 0
  for (int i=0; i<n; i++) 
    CmiPrintf("%4d", bins[i].getU());
  CmiPrintf("\n");
#endif
#endif
  }
  CProxy_TraceSummaryBOC p(traceSummaryGID);
  p[0].sendSummaryBOC(traced, n, bins);
}

extern "C" void _CkExit();

void TraceSummaryBOC::sendSummaryBOC(int traced, int n, BinEntry *b)
{
  int i;
  if (CkpvAccess(_trace) == NULL) return;

  CkAssert(CkMyPe() == 0);

#if 0
#if CMK_TRACE_IN_CHARM
  TRACE_CHARM_PE() = 0;
#endif
#endif

  count ++;
  if (bins == NULL) {
    nBins = CkpvAccess(_trace)->pool()->getNumEntries();
    bins = new BinEntry[nBins];
  }
  if (traced) {
    nTracedPEs ++;
    if (n>nBins) n = nBins;
    for (i=0; i<n; i++) {
      bins[i].Time() += b[i].Time();
    }
  }
  if (count == CkNumPes()) {
    write();
    _CkExit();
  }
}

void TraceSummaryBOC::write(void) 
{
  int i;
  unsigned int j;

  char *fname = new char[strlen(CkpvAccess(traceRoot))+strlen(".sum")+1];
  sprintf(fname, "%s.sum", CkpvAccess(traceRoot));
  FILE *sumfp = fopen(fname, "w+");
  //CmiPrintf("File: %s \n", fname);
  if(sumfp == 0)
      CmiAbort("Cannot open summary sts file for writing.\n");
  delete[] fname;

  fprintf(sumfp, "ver:%3.1f %d/%d count:%d ep:%d interval:%e numTracedPE:%d", CkpvAccess(version), CkMyPe(), CkNumPes(), nBins, _numEntries, CkpvAccess(binSize), nTracedPEs);
  fprintf(sumfp, "\n");

  // write bin time
#if 0
  int last=pool[0].getU();
  pool[0].writeU(fp, last);
  int count=1;
  for(j=1; j<numEntries; j++) {
    int u = pool[j].getU();
    if (last == u) {
      count++;
    }
    else {
      if (count > 1) fprintf(fp, "+%d", count);
      pool[j].writeU(fp, u);
      last = u;
      count = 1;
    }
  }
  if (count > 1) fprintf(fp, "+%d", count);
#else
  for(j=0; j<nBins; j++) {
    bins[j].Time() /= nTracedPEs;
    bins[j].write(sumfp);
  }
#endif
  fprintf(sumfp, "\n");
  fclose(sumfp);

}

extern "C" void CombineSummary()
{
CmiPrintf("[%d] CombineSummary called!\n", CkMyPe());
  if (sumonly) {
CmiPrintf("[%d] Sum Only start!\n", CkMyPe());
    CProxy_TraceSummaryBOC(traceSummaryGID).askSummary();
  }
  else CkExit();
}

void initTraceSummaryBOC()
{
  if (CkMyRank() == 0) {
    registerExitFn(CombineSummary);
  }
}

#include "TraceSummary.def.h"


/*@}*/
