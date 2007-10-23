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

#define VER   7.0

#define INVALIDEP     -2
#define TRACEON_EP     -3

#define DefaultBinCount      10000

CkpvStaticDeclare(TraceSummary*, _trace);
static int _numEvents = 0;
#define NUM_DUMMY_EPS 9
CkpvDeclare(int, binCount);
CkpvDeclare(double, binSize);
CkpvDeclare(double, version);

int sumonly = 0;
int sumDetail = 0;

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

static inline void writeU(FILE* fp, int u)
{
  fprintf(fp, "%4d", u);
}

PhaseEntry::PhaseEntry() 
{
  int _numEntries=_entryTable.size();
  // FIXME: Hopes there won't be more than 10 more EP's registered from now on...
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
    if (sumDetail) fclose(sdfp);
  }
  // free memory for mark
  if (markcount > 0)
  for (int i=0; i<MAX_MARKS; i++) {
    for (int j=0; j<events[i].length(); j++)
      delete events[i][j];
  }
  delete[] pool;
  delete[] epInfo;
  delete[] cpuTime;
  delete[] numExecutions;
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
   // TBD: Can this be moved to initMem?
   poolSize = CkpvAccess(binCount);
   if (poolSize % 2) poolSize++;	// make sure it is even
   pool = new BinEntry[poolSize];
   _MEMCHECK(pool);

   this->pgm = new char[strlen(pgm)+1];
   strcpy(this->pgm,pgm);
   
   // create the sts file
   if (CkMyPe() == 0) {
     char *fname = 
       new char[strlen(CkpvAccess(traceRoot))+strlen(".sum.sts")+1];
     sprintf(fname, "%s.sum.sts", CkpvAccess(traceRoot));
     stsfp = fopen(fname, "w+");
     //CmiPrintf("File: %s \n", fname);
     if (stsfp == 0) {
       CmiAbort("Cannot open summary sts file for writing.\n");
      }
     delete[] fname;
   }
   
   // event
   markcount = 0;
}

void SumLogPool::initMem()
{
   int _numEntries=_entryTable.size();
   epInfoSize = _numEntries + NUM_DUMMY_EPS + 1; // keep a spare EP
   epInfo = new SumEntryInfo[epInfoSize];
   _MEMCHECK(epInfo);

   cpuTime = NULL;
   numExecutions = NULL;
   if (sumDetail) {
       cpuTime = new double[poolSize*epInfoSize];
       _MEMCHECK(cpuTime);
       memset(cpuTime, 0, poolSize*epInfoSize*sizeof(double));
       numExecutions = new int[poolSize*epInfoSize];
       _MEMCHECK(numExecutions);
       memset(numExecutions, 0, poolSize*epInfoSize*sizeof(int));

//         int i, e;
//         for(i=0; i<poolSize; i++) {
//             for(e=0; e< epInfoSize; e++) {
//                 setCPUtime(i,e,0.0);
//                 setNumExecutions(i,e,0);
//             }
//         }
   }
}

int SumLogPool::getUtilization(int interval, int ep) {
    return (int)(getCPUtime(interval, ep) * 100.0 / CkpvAccess(binSize)); 
};

void SumLogPool::write(void) 
{
  int i;
  unsigned int j;
  int _numEntries=_entryTable.size();

  fp = NULL;
  sdfp = NULL;

  // create file(s)
  // CmiPrintf("TRACE: %s:%d\n", fname, errno);
  if (!sumonly) {
    char pestr[10];
    sprintf(pestr, "%d", CkMyPe());
    int len = strlen(pgm) + strlen(".sumd.") + strlen(pestr) + 1;
    char *fname = new char[len+1];
    
    sprintf(fname, "%s.%s.sum", pgm, pestr);
    do {
      fp = fopen(fname, "w+");
    } while (!fp && errno == EINTR);
    if (!fp) {
      CkPrintf("[%d] Attempting to open [%s]\n",CkMyPe(),fname);
      CmiAbort("Cannot open Summary Trace File for writing...\n");
    }
    
    if (sumDetail) {
      sprintf(fname, "%s.%s.sumd", pgm, pestr);
      do {
	sdfp = fopen(fname, "w+");
      } while (!sdfp && errno == EINTR);
      if(!sdfp) {
	CmiAbort("Cannot open Detailed Summary Trace File for writing...\n");
      }
    }
    delete[] fname;
  }

  fprintf(fp, "ver:%3.1f %d/%d count:%d ep:%d interval:%e", CkpvAccess(version), CkMyPe(), CkNumPes(), numBins, _numEntries, CkpvAccess(binSize));
  if (CkpvAccess(version)>=3.0)
  {
    fprintf(fp, " phases:%d", phaseTab.numPhasesCalled());
  }
  fprintf(fp, "\n");

  // write bin time
#if 1
  int last=pool[0].getU();
  writeU(fp, last);
  int count=1;
  for(j=1; j<numBins; j++) {
    int u = pool[j].getU();
    if (last == u) {
      count++;
    }
    else {
      if (count > 1) fprintf(fp, "+%d", count);
      writeU(fp, u);
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

  // write summary details
  if (sumDetail) {
        fprintf(sdfp, "ver:%3.1f cpu:%d/%d numIntervals:%d numEPs:%d intervalSize:%e\n",
                CkpvAccess(version), CkMyPe(), CkNumPes(),
                numBins, _numEntries, CkpvAccess(binSize));

        // Write out cpuTime in microseconds
        // Run length encoding (RLE) along EP axis
        fprintf(sdfp, "ExeTimePerEPperInterval ");
        unsigned int e, i;
        long last= (long) (getCPUtime(0,0)*1.0e6);
        int count=0;
        fprintf(sdfp, "%ld", last);
        for(e=0; e<_numEntries; e++) {
            for(i=0; i<numBins; i++) {

                long u= (long) (getCPUtime(i,e)*1.0e6);
                if (last == u) {
                    count++;
                } else {

                    if (count > 1) fprintf(sdfp, "+%d", count);
                    fprintf(sdfp, " %ld", u);
                    last = u;
                    count = 1;
                }
            }
        }
        if (count > 1) fprintf(sdfp, "+%d", count);
        fprintf(sdfp, "\n");

        // Write out numExecutions
        // Run length encoding (RLE) along EP axis
        fprintf(sdfp, "EPCallTimePerInterval ");
        last= getNumExecutions(0,0);
        count=0;
        fprintf(sdfp, "%d", last);
        for(e=0; e<_numEntries; e++) {
            for(i=0; i<numBins; i++) {

                long u= getNumExecutions(i, e);
                if (last == u) {
                    count++;
                } else {

                    if (count > 1) fprintf(sdfp, "+%d", count);
                    fprintf(sdfp, " %d", u);
                    last = u;
                    count = 1;
                }
            }
        }
        if (count > 1) fprintf(sdfp, "+%d", count);
        fprintf(sdfp, "\n");
  }
}

void SumLogPool::writeSts(void)
{
  traceWriteSTS(stsfp,_numEvents);
  for(int i=0;i<_numEvents;i++)
    fprintf(stsfp, "EVENT %d Event%d\n", i, i);
  fprintf(stsfp, "END\n");
  fclose(stsfp);
}

// Called once per interval
void SumLogPool::add(double time, int pe) 
{
  new (&pool[numBins++]) BinEntry(time);
  if(poolSize==numBins) shrink();
}

// Called once per run of an EP
// adds 'time' to EP's time, increments epCount
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

// Called once from endExecute, endPack, etc. this function updates
// the sumDetail intervals.
void SumLogPool::updateSummaryDetail(int epIdx, double startTime, double endTime)
{
        if (epIdx >= epInfoSize) {
            CmiAbort("Too many entry points!!\n");
        }

        double binSz = CkpvAccess(binSize);
        int startingBinIdx, endingBinIdx;
	startingBinIdx = (int)(startTime/binSz);
        endingBinIdx = (int)(endTime/binSz);
	// shrink if needed
	while (endingBinIdx >= poolSize) {
	  shrink();
	  CmiAssert(CkpvAccess(binSize) > binSz);
          binSz = CkpvAccess(binSize);
	  startingBinIdx = (int)(startTime/binSz);
          endingBinIdx = (int)(endTime/binSz);
	}

        if (startingBinIdx == endingBinIdx) {
            addToCPUtime(startingBinIdx, epIdx, endTime - startTime);
        } else if (startingBinIdx < endingBinIdx) { // EP spans intervals
            addToCPUtime(startingBinIdx, epIdx, (startingBinIdx+1)*binSz - startTime);
            while(++startingBinIdx < endingBinIdx)
                addToCPUtime(startingBinIdx, epIdx, binSz);
            addToCPUtime(endingBinIdx, epIdx, endTime - endingBinIdx*binSz);
        } else {
	  CkPrintf("[%d] EP:%d Start:%lf End:%lf\n",CkMyPe(),epIdx,
		   startTime, endTime);
            CmiAbort("Error: end time of EP is less than start time\n");
        }

        incNumExecutions(startingBinIdx, epIdx);
};

// Shrinks pool[], cpuTime[], and numExecutions[]
void SumLogPool::shrink(void)
{
//  double t = CmiWallTimer();

  // We ensured earlier that poolSize is even; therefore now numBins
  // == poolSize == even.
  int entries = numBins/2;
  for (int i=0; i<entries; i++)
  {
     pool[i].time() = pool[i*2].time() + pool[i*2+1].time();
     if (sumDetail)
     for (int e=0; e < epInfoSize; e++) {
         setCPUtime(i, e, getCPUtime(i*2, e) + getCPUtime(i*2+1, e));
         setNumExecutions(i, e, getNumExecutions(i*2, e) + getNumExecutions(i*2+1, e));
     }
  }
  // zero out the remaining intervals
  if (sumDetail) {
    memset(&cpuTime[entries*epInfoSize], 0, (numBins-entries)*epInfoSize*sizeof(double));
    memset(&numExecutions[entries*epInfoSize], 0, (numBins-entries)*epInfoSize*sizeof(int));
  }
  numBins = entries;
  CkpvAccess(binSize) *= 2;

//CkPrintf("Shrinked binsize: %f entries:%d takes %fs!!!!\n", CkpvAccess(binSize), numEntries, CmiWallTimer()-t);
}

int  BinEntry::getU() 
{ 
  return (int)(_time * 100.0 / CkpvAccess(binSize)); 
}

void BinEntry::write(FILE* fp)
{
  writeU(fp, getU());
}

TraceSummary::TraceSummary(char **argv):binStart(0.0),bin(0.0),msgNum(0)
{
  if (CkpvAccess(traceOnPe) == 0) return;

  CkpvInitialize(int, binCount);
  CkpvInitialize(double, binSize);
  CkpvInitialize(double, version);
  CkpvAccess(binSize) = BIN_SIZE;
  CkpvAccess(version) = VER;
  CkpvAccess(binCount) = DefaultBinCount;
  if (CmiGetArgIntDesc(argv,"+bincount",&CkpvAccess(binCount), "Total number of summary bins"))
    if (CkMyPe() == 0) 
      CmiPrintf("Trace: bincount: %d\n", CkpvAccess(binCount));
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
  // +sumonly overrides +sumDetail
  if (!sumonly)
      sumDetail = CmiGetArgFlagDesc(argv, "+sumDetail", "more detailed summary info");

  _logPool = new SumLogPool(CkpvAccess(traceRoot));
  // assume invalid entry point on start
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
  CkpvAccess(_trace)->endComputation();
  // destructor call the write()
  delete _logPool;
  // remove myself from traceArray so that no tracing will be called.
  CkpvAccess(_traces)->removeTrace(this);
}

void TraceSummary::beginExecute(CmiObjId *tid)
{
  beginExecute(-1,-1,_threadEP,-1);
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

void TraceSummary::beginExecute(int event,int msgType,int ep,int srcPe, int mlen, CmiObjId *idx)
{
  if (execEp != INVALIDEP) {
    TRACE_WARN("Warning: TraceSummary two consecutive BEGIN_PROCESSING!\n");
    return;
  }
  
  execEp=ep;
  double t = TraceTimer();
  //CmiPrintf("start: %f \n", start);
  
  start = t;
  double ts = binStart;
  // fill gaps
  while ((ts = ts + CkpvAccess(binSize)) < t) {
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

  if (execEp == TRACEON_EP) {
    // if trace just got turned on, then one expects to see this
    // END_PROCESSING event without seeing a preceeding BEGIN_PROCESSING
    return;
  }

  if (execEp == INVALIDEP) {
    TRACE_WARN("Warning: TraceSummary END_PROCESSING without BEGIN_PROCESSING!\n");
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
     _logPool->add(bin, CkMyPe()); // This calls shrink() if needed
     bin = 0;
     ts = nts;
  }
  bin += t - ts;

  if (sumDetail)
      _logPool->updateSummaryDetail(execEp, start, t);

  execEp = INVALIDEP;
}

void TraceSummary::beginPack(void)
{
    packstart = CmiWallTimer();
}

void TraceSummary::endPack(void)
{
    _logPool->setEp(_packEP, CmiWallTimer() - packstart);
    if (sumDetail)
        _logPool->updateSummaryDetail(_packEP,  TraceTimer(packstart), TraceTimer(CmiWallTimer()));
}

void TraceSummary::beginUnpack(void)
{
    unpackstart = CmiWallTimer();
}

void TraceSummary::endUnpack(void)
{
    _logPool->setEp(_unpackEP, CmiWallTimer()-unpackstart);
    if (sumDetail)
        _logPool->updateSummaryDetail(_unpackEP,  TraceTimer(unpackstart), TraceTimer(CmiWallTimer()));
}

void TraceSummary::beginComputation(void)
{
  // initialze arrays because now the number of entries is known.
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
     bin = 0.0;
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

void TraceSummaryBOC::startSumOnly()
{
  CmiAssert(CkMyPe() == 0);

  CProxy_TraceSummaryBOC p(traceSummaryGID);
  int size = CkpvAccess(_trace)->pool()->getNumEntries();
  p.askSummary(size);
}

void TraceSummaryBOC::askSummary(int size)
{
  if (CkpvAccess(_trace) == NULL) return;

  int traced = CkpvAccess(_trace)->traceOnPE();

  BinEntry *bins = new BinEntry[size+1];
  bins[size] = traced;		// last element is the traced pe count
  if (traced) {
    CkpvAccess(_trace)->endComputation();
    int n = CkpvAccess(_trace)->pool()->getNumEntries();
    BinEntry *localBins = CkpvAccess(_trace)->pool()->bins();
#if 0
  CmiPrintf("askSummary on [%d] numEntried=%d\n", CkMyPe(), n);
#if 1
  for (int i=0; i<n; i++) CmiPrintf("%4d", localBins[i].getU());
  CmiPrintf("\n");
#endif
#endif
    if (n>size) n=size;
    for (int i=0; i<n; i++) bins[i] = localBins[i];
  }

  contribute(sizeof(BinEntry)*(size+1), bins, CkReduction::sum_double);
  delete [] bins;
}

extern "C" void _CkExit();

void TraceSummaryBOC::sendSummaryBOC(CkReductionMsg *msg)
{
  if (CkpvAccess(_trace) == NULL) return;

  CkAssert(CkMyPe() == 0);

  int n = msg->getSize()/sizeof(BinEntry);
  nBins = n-1;
  bins = (BinEntry *)msg->getData();
  nTracedPEs = (int)bins[n-1].time();
  //CmiPrintf("traced: %d entry:%d\n", nTracedPEs, nBins);

  write();

  delete msg;

  _CkExit();
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

  int _numEntries=_entryTable.size();
  fprintf(sumfp, "ver:%3.1f %d/%d count:%d ep:%d interval:%e numTracedPE:%d", CkpvAccess(version), CkMyPe(), CkNumPes(), nBins, _numEntries, CkpvAccess(binSize), nTracedPEs);
  fprintf(sumfp, "\n");

  // write bin time
#if 0
  int last=pool[0].getU();
  writeU(fp, last);
  int count=1;
  for(j=1; j<numEntries; j++) {
    int u = pool[j].getU();
    if (last == u) {
      count++;
    }
    else {
      if (count > 1) fprintf(fp, "+%d", count);
      writeU(fp, u);
      last = u;
      count = 1;
    }
  }
  if (count > 1) fprintf(fp, "+%d", count);
#else
  for(j=0; j<nBins; j++) {
    bins[j].time() /= nTracedPEs;
    bins[j].write(sumfp);
  }
#endif
  fprintf(sumfp, "\n");
  fclose(sumfp);

}

extern "C" void CombineSummary()
{
#ifndef CMK_OPTIMIZE
  CmiPrintf("[%d] CombineSummary called!\n", CkMyPe());
  if (sumonly) {
    CmiPrintf("[%d] Sum Only start!\n", CkMyPe());
      // pe 0 start the sumonly process
    CProxy_TraceSummaryBOC sumProxy(traceSummaryGID);
    sumProxy[0].startSumOnly();
  }
  else CkExit();
#else
  CkExit();
#endif
}

void initTraceSummaryBOC()
{
#ifdef __BLUEGENE__
  if(BgNodeRank()==0) {
#else
  if (CkMyRank() == 0) {
#endif
    registerExitFn(CombineSummary);
  }
}

#include "TraceSummary.def.h"


/*@}*/
