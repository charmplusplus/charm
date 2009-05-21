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

#define VER   7.1

#define INVALIDEP     -2
#define TRACEON_EP     -3

// 5 minutes of run before it'll fill up:
#define DefaultBinCount      (1000*60*1) 

CkpvStaticDeclare(TraceSummary*, _trace);
static int _numEvents = 0;
#define NUM_DUMMY_EPS 9
CkpvDeclare(int, binCount);
CkpvDeclare(double, binSize);
CkpvDeclare(double, version);


CkpvDeclare(int, previouslySentBins);





/** 
    A class that reads/writes a buffer out of different types of data.

    This class exists because I need to get references to parts of the buffer 
    that have already been used so that I can increment counters inside the buffer.
*/

class compressedBuffer {
public:
  char* buf;
  int pos; ///<< byte position just beyond the previously read/written data

  compressedBuffer(){
    buf = NULL;
    pos = 0;
  }

  compressedBuffer(int bytes){
    buf = (char*)malloc(bytes);
    pos = 0;
  }

  compressedBuffer(void *buffer){
    buf = (char*)buffer;
    pos = 0;
  }
    
  void init(void *buffer){
    buf = (char*)buffer;
    pos = 0;
  }
    
  inline void * currentPtr(){
    return (void*)(buf+pos);
  }

  template <typename T>
  T read(int offset){
    // to resolve unaligned writes causing bus errors, need memcpy
    T v;
    memcpy(&v, buf+offset, sizeof(T));
    return v;
  }
    
  template <typename T>
  void write(T v, int offset){
    T v2 = v; // on stack
    // to resolve unaligned writes causing bus errors, need memcpy
    memcpy(buf+offset, &v2, sizeof(T));
  }
    
  template <typename T>
  void increment(int offset){
    T temp;
    temp = read<T>(offset);
    temp ++;
    write<T>(temp, offset);
  }

  template <typename T>
  void accumulate(T v, int offset){
    T temp;
    temp = read<T>(offset);
    temp += v;
    write<T>(temp, offset);
  }
  
  template <typename T>
  int push(T v){
    int oldpos = pos;
    write<T>(v, pos);
    pos += sizeof(T);
    return oldpos;
  }
  
  template <typename T>
  T pop(){
    T temp = read<T>(pos);
    pos += sizeof(T);
    return temp;
  }

  template <typename T>
  T peek(){
    T temp = read<T>(pos);
    return temp;
  }

  template <typename T0, typename T>
  T peekSecond(){
    T temp;
    memcpy(&temp, buf+pos+sizeof(T0), sizeof(T));
    return temp;
  }

  int datalength(){
    return pos;
  }
     
  void * buffer(){
    return (void*) buf;
  }  

  void freeBuf(){
    free(buf);
  }

  ~compressedBuffer(){
    // don't free the buf because the user my have supplied the buffer
  }
    
};


/** Define the types used in the gathering of sum detail statistics for use with CCS */
#define numBins_T int
#define numProcs_T int
#define entriesInBin_T short
#define ep_T short
#define utilization_T unsigned char
#define other_EP 10000


// Predeclarations of the functions at the bottom of this file
compressedBuffer compressAvailableNewSumDetail(int max=10000);
void mergeCompressedBin(compressedBuffer *srcBufferArray, int numSrcBuf, int *numProcsRepresentedInMessage, int totalProcsAcrossAllMessages, compressedBuffer &destBuffer);
compressedBuffer compressNRecentSumDetail(int desiredBinsToSend);
void printSumDetailInfo(int desiredBinsToSend);
CkReductionMsg *sumDetailCompressedReduction(int nMsg,CkReductionMsg **msgs);
void printCompressedBuf(compressedBuffer b);
compressedBuffer fakeCompressedMessage();
compressedBuffer emptyCompressedBuffer();
void sanityCheckCompressedBuf(compressedBuffer b);
bool isCompressedBufferSane(compressedBuffer b);
double averageUtilizationInBuffer(compressedBuffer b);



/// A reduction type for merging compressed sum detail data
CkReduction::reducerType sumDetailCompressedReducer;
/// An initnode registration function for the reducer
void registerIdleTimeReduction(void) {
  CkAssert(sizeof(short) == 2);
  sumDetailCompressedReducer=CkReduction::addReducer(sumDetailCompressedReduction);
}



// Global Readonly
CkGroupID traceSummaryGID;
bool summaryCcsStreaming;

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
  CkpvInitialize(int, previouslySentBins);
  CkpvAccess(previouslySentBins) = 0;
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
  cpuTime = NULL;
   poolSize = CkpvAccess(binCount);
   if (poolSize % 2) poolSize++;	// make sure it is even
   pool = new BinEntry[poolSize];
   _MEMCHECK(pool);

   this->pgm = new char[strlen(pgm)+1];
   strcpy(this->pgm,pgm);
   
#if 0
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
#endif
   
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
  // write idle time
  if (CkpvAccess(version)>=7.1) {
    fprintf(fp, "IdlePercent: ");
    int last=pool[0].getUIdle();
    writeU(fp, last);
    int count=1;
    for(j=1; j<numBins; j++) {
      int u = pool[j].getUIdle();
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
    fprintf(fp, "\n");
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
    // open sts file
  char *fname = 
       new char[strlen(CkpvAccess(traceRoot))+strlen(".sum.sts")+1];
  sprintf(fname, "%s.sum.sts", CkpvAccess(traceRoot));
  stsfp = fopen(fname, "w+");
  //CmiPrintf("File: %s \n", fname);
  if (stsfp == 0) {
       CmiAbort("Cannot open summary sts file for writing.\n");
  }
  delete[] fname;

  traceWriteSTS(stsfp,_numEvents);
  for(int i=0;i<_numEvents;i++)
    fprintf(stsfp, "EVENT %d Event%d\n", i, i);
  fprintf(stsfp, "END\n");

  fclose(stsfp);
}

// Called once per interval
void SumLogPool::add(double time, double idleTime, int pe) 
{
  new (&pool[numBins++]) BinEntry(time, idleTime);
  if (poolSize==numBins) {
    shrink();
  }
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

int BinEntry::getUIdle() {
  return (int)(_idleTime * 100.0 / CkpvAccess(binSize));
}

void BinEntry::write(FILE* fp)
{
  writeU(fp, getU());
}

TraceSummary::TraceSummary(char **argv):binStart(0.0),
					binTime(0.0),binIdle(0.0),msgNum(0)
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
    /* Keep as a template for error checking. The current form of this check
       is vulnerable to round-off errors (eg. 0.001 vs 0.001 the first time
       I used it). Perhaps an improved form could be used in case vastly
       incompatible EP vs idle times are reported (binSize*2?).

       This check will have to be duplicated before each call to add()

    CkPrintf("[%d] %f vs %f\n", CkMyPe(),
	     binTime + binIdle, CkpvAccess(binSize));
    CkAssert(binTime + binIdle <= CkpvAccess(binSize));
    */
     _logPool->add(binTime, binIdle, CkMyPe()); // add leftovers of last bin
     binTime=0.0;                 // fill all other bins with 0 up till start
     binIdle = 0.0;
     binStart = ts;
  }
}

void TraceSummary::endExecute(void)
{
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
    // fill the bins with time for this entry method
     binTime += nts-ts;
     binStart  = nts;
     // This calls shrink() if needed
     _logPool->add(binTime, binIdle, CkMyPe()); 
     binTime = 0.0;
     binIdle = 0.0;
     ts = nts;
  }
  binTime += t - ts;

  if (sumDetail)
      _logPool->updateSummaryDetail(execEp, start, t);

  execEp = INVALIDEP;
}

void TraceSummary::beginIdle(double currT)
{
  // for consistency with current framework behavior, currT is ignored and
  // independent timing taken by trace-summary.
  double t = TraceTimer();
  
  // mark the time of this idle period. Only the next endIdle should see
  // this value
  idleStart = t; 
  double ts = binStart;
  // fill gaps
  while ((ts = ts + CkpvAccess(binSize)) < t) {
    _logPool->add(binTime, binIdle, CkMyPe()); // add leftovers of last bin
    binTime=0.0;                 // fill all other bins with 0 up till start
    binIdle = 0.0;
    binStart = ts;
  }
}

void TraceSummary::endIdle(double currT)
{
  // again, we ignore the reported currT (see beginIdle)
  double t = TraceTimer();
  double t_idleStart = idleStart;
  double t_binStart = binStart;

  while ((t_binStart = t_binStart + CkpvAccess(binSize)) < t)
  {
    // fill the bins with time for idle
    binIdle += t_binStart - t_idleStart;
    binStart = t_binStart;
    _logPool->add(binTime, binIdle, CkMyPe()); // This calls shrink() if needed
    binTime = 0.0;
    binIdle = 0.0;
    t_idleStart = t_binStart;
  }
  binIdle += t - t_idleStart;
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
     _logPool->add(binTime, binIdle, CkMyPe());
     binTime = 0.0;
     binIdle = 0.0;
     msgNum ++;

     binStart  += CkpvAccess(binSize);
     double t = TraceTimer();
     double ts = binStart;
     while (ts < t)
     {
       _logPool->add(binTime, binIdle, CkMyPe());
       binTime=0.0;
       binIdle = 0.0;
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

void TraceSummary::traceEnableCCS() {
  CProxy_TraceSummaryBOC sumProxy(traceSummaryGID);
  sumProxy.initCCS();
}


void TraceSummary::fillData(double *buffer, double reqStartTime, 
			    double reqBinSize, int reqNumBins) {
  // buffer has to be pre-allocated by the requester and must be an array of
  // size reqNumBins.
  //
  // Assumptions: **CWL** FOR DEMO ONLY - a production-capable version will
  //              need a number of these assumptions dropped:
  //              1) reqBinSize == binSize (unrealistic)
  //              2) bins boundary aligned (ok even under normal circumstances)
  //              3) bins are "factor"-aligned (where reqBinSize != binSize)
  //              4) bins are always available (true unless flush)
  //              5) bins always starts from 0 (unrealistic)

  // works only because of 1)
  // **CWL** - FRACKING STUPID NAME "binStart" has nothing to do with 
  //           "starting" at all!
  int binOffset = (int)(reqStartTime/reqBinSize); 
  for (int i=binOffset; i<binOffset + reqNumBins; i++) {
    // CkPrintf("[%d] %f\n", i, pool()->getTime(i));
    buffer[i-binOffset] = pool()->getTime(i);
  }
}


/// for TraceSummaryBOC

void TraceSummaryBOC::initCCS() {
  if(firstTime){
    CkPrintf("[%d] initCCS() called for first time\n", CkMyPe());
    // initializing CCS-based parameters on all processors
    lastRequestedIndexBlock = 0;
    indicesPerBlock = 1000;
    collectionGranularity = 0.001; // time in seconds
    nBufferedBins = 0;
    
    // initialize buffer, register CCS handler and start the collection
    // pulse only on pe 0.
    if (CkMyPe() == 0) { 
      ccsBufferedData = new CkVec<double>();
    
      CProxy_TraceSummaryBOC sumProxy(traceSummaryGID);
      CkPrintf("Trace Summary now listening in for CCS Client\n");
      CcsRegisterHandler("CkPerfSummaryCcsClientCB", 
			 CkCallback(CkIndex_TraceSummaryBOC::ccsRequestSummaryDouble(NULL), sumProxy[0]));
      CcsRegisterHandler("CkPerfSummaryCcsClientCB uchar", 
			 CkCallback(CkIndex_TraceSummaryBOC::ccsRequestSummaryUnsignedChar(NULL), sumProxy[0])); 
      CcsRegisterHandler("CkPerfSumDetail compressed", 
			 CkCallback(CkIndex_TraceSummaryBOC::ccsRequestSumDetailCompressed(NULL), sumProxy[0])); 

      CkPrintf("[%d] Setting up periodic startCollectData callback\n", CkMyPe());
      CcdCallOnConditionKeep(CcdPERIODIC_1second, startCollectData,
			     (void *)this);
      summaryCcsStreaming = CmiTrue;
    }
    firstTime = false;
  }
}

/** Return summary information as double precision values for each sample period. 
    The actual data collection is in double precision values. 

    The units on the returned values are total execution time across all PEs.
*/
void TraceSummaryBOC::ccsRequestSummaryDouble(CkCcsRequestMsg *m) {
  double *sendBuffer;

  CkPrintf("[%d] Request from Client detected.\n", CkMyPe());

  CkPrintf("Responding ...\n");
  int datalength = 0;
  // if we have no data to send, send an acknowledgement code of -13.37
  // instead.
  if (ccsBufferedData->length() == 0) {
    sendBuffer = new double[1];
    sendBuffer[0] = -13.37;
    datalength = sizeof(double);
    CcsSendDelayedReply(m->reply, datalength, (void *)sendBuffer);
    delete [] sendBuffer;
  } else {
    sendBuffer = ccsBufferedData->getVec();
    datalength = ccsBufferedData->length()*sizeof(double);
    CcsSendDelayedReply(m->reply, datalength, (void *)sendBuffer);
    ccsBufferedData->free();
  }
  CkPrintf("Response Sent. Proceeding with computation.\n");
  delete m;
}


/** Return summary information as unsigned char values for each sample period. 
    The actual data collection is in double precision values.

    This returns the utilization in a range from 0 to 200.
*/
void TraceSummaryBOC::ccsRequestSummaryUnsignedChar(CkCcsRequestMsg *m) {
  unsigned char *sendBuffer;

  CkPrintf("[%d] Request from Client detected. \n", CkMyPe());

  CkPrintf("Responding ...\n");
  int datalength = 0;

  if (ccsBufferedData->length() == 0) {
    sendBuffer = new unsigned char[1];
    sendBuffer[0] = 255;
    datalength = sizeof(unsigned char);
    CcsSendDelayedReply(m->reply, datalength, (void *)sendBuffer);
    delete [] sendBuffer;
  } else {
    double * doubleData = ccsBufferedData->getVec();
    int numData = ccsBufferedData->length();
    
    // pack data into unsigned char array
    sendBuffer = new unsigned char[numData];
    
    for(int i=0;i<numData;i++){
      sendBuffer[i] = 1000.0 * doubleData[i] / (double)CkNumPes() * 200.0; // max = 200 is the same as 100% utilization
      int v = sendBuffer[i];
    }    

    datalength = sizeof(unsigned char) * numData;
    
    CcsSendDelayedReply(m->reply, datalength, (void *)sendBuffer);
    ccsBufferedData->free();
    delete [] sendBuffer;
  }
  CkPrintf("Response Sent. Proceeding with computation.\n");
  delete m;
}


/**

Send back to the client compressed sum-detail style measurements about the 
utilization for each active PE combined across all PEs.

The data format sent by this handler is a bunch of records(one for each bin) of the following format:
   #samples (EP,utilization)* 

One example record for two EPS that executed during the sample period. 
EP 3 used 150/200 of the time while EP 122 executed for 20/200 of the time. 
All of these would be packed as bytes into the message:
2 3 150 122 20

 */
void TraceSummaryBOC::ccsRequestSumDetailCompressed(CkCcsRequestMsg *m) {
  CkPrintf("CCS request for compressed sum detail. (found %d stored in deque)\n",  storedSumDetailResults.size() );
  CkAssert(sumDetail);
  int datalength;

#if 0

  compressedBuffer fakeMessage = fakeCompressedMessage();
  CcsSendDelayedReply(m->reply, fakeMessage.datalength(), fakeMessage.buffer() );
  fakeMessage.freeBuf();

#else

  if (storedSumDetailResults.size()  == 0) {
    compressedBuffer b = emptyCompressedBuffer();
    CcsSendDelayedReply(m->reply, b.datalength(), b.buffer()); 
    b.freeBuf();
  } else {
    CkReductionMsg * msg = storedSumDetailResults.front();
    storedSumDetailResults.pop_front();

    
    void *sendBuffer = (void *)msg->getData();
    datalength = msg->getSize();
    CcsSendDelayedReply(m->reply, datalength, sendBuffer);
    
    delete msg;
  }
    
  
#endif

  CkPrintf("CCS response of %d bytes sent.\n", datalength);
  delete m;
}




void startCollectData(void *data, double currT) {
  CkAssert(CkMyPe() == 0);
  // CkPrintf("startCollectData()\n");
  TraceSummaryBOC *sumObj = (TraceSummaryBOC *)data;
  int lastRequestedIndexBlock = sumObj->lastRequestedIndexBlock;
  double collectionGranularity = sumObj->collectionGranularity;
  int indicesPerBlock = sumObj->indicesPerBlock;
  
  double startTime = lastRequestedIndexBlock*
    collectionGranularity * indicesPerBlock;
  int numIndicesToGet = (int)floor((currT - startTime)/
				   collectionGranularity);
  int numBlocksToGet = numIndicesToGet/indicesPerBlock;
  // **TODO** consider limiting the total number of blocks each collection
  //   request will pick up. This is to limit the amount of perturbation
  //   if it proves to be a problem.
  CProxy_TraceSummaryBOC sumProxy(traceSummaryGID);

//   sumProxy.collectSummaryData(startTime, 
// 		       collectionGranularity,
// 		       numBlocksToGet*indicesPerBlock);

  sumProxy.collectSumDetailData(startTime, 
		       collectionGranularity,
		       1000);

  // assume success
  sumObj->lastRequestedIndexBlock += numBlocksToGet; 
}

void TraceSummaryBOC::collectSummaryData(double startTime, double binSize,
				  int numBins) {
  // CkPrintf("[%d] asked to contribute performance data\n", CkMyPe());

  double *contribution = new double[numBins];
  for (int i=0; i<numBins; i++) {
    contribution[i] = 0.0;
  }
  CkpvAccess(_trace)->fillData(contribution, startTime, binSize, numBins);

  /*
  for (int i=0; i<numBins; i++) {
    CkPrintf("[%d] %f\n", i, contribution[i]);
  }
  */

  CProxy_TraceSummaryBOC sumProxy(traceSummaryGID);
  CkCallback cb(CkIndex_TraceSummaryBOC::summaryDataCollected(NULL), sumProxy[0]);
  contribute(sizeof(double)*numBins, contribution, CkReduction::sum_double, 
	     cb);
}

void TraceSummaryBOC::summaryDataCollected(CkReductionMsg *msg) {
  CkAssert(CkMyPe() == 0);
  // **CWL** No memory management for the ccs buffer for now.

  // CkPrintf("[%d] Reduction completed and received\n", CkMyPe());
  double *recvData = (double *)msg->getData();
  int numBins = msg->getSize()/sizeof(double);

  // if there's an easier way to append a data block to a CkVec, I'll take it
  for (int i=0; i<numBins; i++) {
    ccsBufferedData->insertAtEnd(recvData[i]);
  }
  delete msg;
}


void TraceSummaryBOC::collectSumDetailData(double startTime, double binSize, int numBins) {
  
  
  printSumDetailInfo(numBins);
  compressedBuffer b = compressNRecentSumDetail(numBins);
  //  CkPrintf("[%d] contributing buffer created by compressNRecentSumDetail: \n", CkMyPe());
  //  printCompressedBuf(b);
  
  
  
#if 0
  b = fakeCompressedMessage();
#endif
  
  //  CkPrintf("[%d] contributing %d bytes worth of SumDetail data\n", CkMyPe(), b.datalength());
  
  //  CProxy_TraceSummaryBOC sumProxy(traceSummaryGID);
  CkCallback cb(CkIndex_TraceSummaryBOC::sumDetailDataCollected(NULL), thisProxy[0]);
  contribute(b.datalength(), b.buffer(), sumDetailCompressedReducer, cb);
  
  b.freeBuf();
}


void TraceSummaryBOC::sumDetailDataCollected(CkReductionMsg *msg) {
  CkAssert(CkMyPe() == 0);
  CkPrintf("[%d] Reduction of SumDetail completed. Result stored in storedSumDetailResults deque(sizes=%d)\n", CkMyPe(), storedSumDetailResults.size() );
  fflush(stdout);

  //  printCompressedBuf(msg->getData());
  //  CkPrintf("Sanity Checking buffer before putting in storedSumDetailResults\n");
  compressedBuffer b(msg->getData());
  //  CkPrintf("---------------------------------------- putting CCS reply in queue (average utilization= %lf)\n", averageUtilizationInBuffer(b));
  if(isCompressedBufferSane(b)){
    storedSumDetailResults.push_back(msg); 
  }
}




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

  double *reductionBuffer = new double[size+1];
  reductionBuffer[size] = traced;  // last element is the traced pe count
  if (traced) {
    CkpvAccess(_trace)->endComputation();
    int n = CkpvAccess(_trace)->pool()->getNumEntries();
    BinEntry *localBins = CkpvAccess(_trace)->pool()->bins();
    if (n>size) n=size;
    for (int i=0; i<n; i++) reductionBuffer[i] = localBins[i].time();
  }

  contribute(sizeof(double)*(size+1), reductionBuffer, 
	     CkReduction::sum_double);
  delete [] reductionBuffer;
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







////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


  
/// Compress a buffer by merging all entries in a bin that are less than the threshold into a single "other" category
  compressedBuffer moveTinyEntriesToOther(compressedBuffer src, double threshold){
    //    CkPrintf("[%d] moveTinyEntriesToOther\n", CkMyPe());
    
    // reset the src buffer to the beginning
    src.pos = 0;

    compressedBuffer dest(100000); 
    
    int numBins = src.pop<numBins_T>();
    int numProcs = src.pop<numProcs_T>();
    
    dest.push<numBins_T>(numBins);
    dest.push<numProcs_T>(numProcs);
    
    
    for(int i=0;i<numBins;i++){
      double utilizationInOther = 0.0;
      
      entriesInBin_T numEntriesInSrcBin = src.pop<entriesInBin_T>();
      int numEntriesInDestBinOffset = dest.push<entriesInBin_T>(0);
      
      CkAssert(numEntriesInSrcBin < 200);

      for(int j=0;j<numEntriesInSrcBin;j++){
	ep_T ep = src.pop<ep_T>();
	double v = src.pop<utilization_T>();
	
	if(v < threshold * 250.0){
	  // do not copy bin into destination
	  utilizationInOther += v / 250.0;
	} else {
	  // copy bin into destination
	  dest.increment<entriesInBin_T>(numEntriesInDestBinOffset);
	  dest.push<ep_T>(ep);
	  dest.push<utilization_T>(v);
	}

      }
      
      // if other category has stuff in it, add it to the destination buffer
      if(utilizationInOther > 0.0){
	dest.increment<entriesInBin_T>(numEntriesInDestBinOffset);
	dest.push<ep_T>(other_EP);
	if(utilizationInOther > 1.0)
	  utilizationInOther = 1.0;
	dest.push<utilization_T>(utilizationInOther*250.0);
      }
      
    }
   
    return dest;
  }
  
    





/// A reducer for merging compressed sum detail data
CkReductionMsg *sumDetailCompressedReduction(int nMsg,CkReductionMsg **msgs){
  // CkPrintf("[%d] sumDetailCompressedReduction(nMsgs=%d)\n", CkMyPe(), nMsg);
  
  compressedBuffer *incomingMsgs = new compressedBuffer[nMsg];
  int *numProcsRepresentedInMessage = new int[nMsg];
  
  int numBins = 0;
  int totalsize = 0;
  int totalProcsAcrossAllMessages = 0;
  
  for (int i=0;i<nMsg;i++) {
    incomingMsgs[i].init(msgs[i]->getData());
    
    //  CkPrintf("[%d] Incoming reduction message %d has average utilization %lf\n", CkMyPe(),  i, averageUtilizationInBuffer(incomingMsgs[i])); 
    //   CkPrintf("Is buffer %d sane? %s\n", i, isCompressedBufferSane(incomingMsgs[i]) ? "yes": "no" );


    totalsize += msgs[i]->getSize();
    //  CkPrintf("BEGIN MERGE MESSAGE=========================================================\n");
    //   printCompressedBuf(incomingMsgs[i]);
    
    // Read first value from message. 
    // Make sure all messages have the same number of bins
    if(i==0)
      numBins = incomingMsgs[i].pop<numBins_T>();
    else 
      CkAssert( numBins ==  incomingMsgs[i].pop<numBins_T>() );
    
    // Read second value from message. 
    numProcsRepresentedInMessage[i] = incomingMsgs[i].pop<numProcs_T>();
    totalProcsAcrossAllMessages += numProcsRepresentedInMessage[i];
    //    CkPrintf("Number of procs in message[%d] is %d\n", i,  (int)numProcsRepresentedInMessage[i]);
  }
  
  compressedBuffer dest(totalsize + 100); 
  
  // build a compressed representation of each merged bin
  dest.push<numBins_T>(numBins);
  dest.push<numProcs_T>(totalProcsAcrossAllMessages);
  
  for(int i=0; i<numBins; i++){
    mergeCompressedBin(incomingMsgs, nMsg, numProcsRepresentedInMessage, totalProcsAcrossAllMessages, dest);
  }
  
  // CkPrintf("END MERGE RESULT=========================================================\n");
  // printCompressedBuf(dest);


  //CkPrintf("[%d] Merged buffer has average utilization %lf \n", CkMyPe(), averageUtilizationInBuffer(dest));

  //CkPrintf("Is resulting merged buffer sane? %s\n", isCompressedBufferSane(dest) ? "yes": "no" );  
  
  compressedBuffer dest2 = moveTinyEntriesToOther(dest, 0.15);
  
  //  CkPrintf("Is resulting merged Filtered buffer sane? %s\n", isCompressedBufferSane(dest2) ? "yes": "no" ); 

  //  CkPrintf("[%d] Outgoing reduction (filtered) message has average utilization %lf \n", CkMyPe(), averageUtilizationInBuffer(dest2));

  
  CkReductionMsg *m = CkReductionMsg::buildNew(dest2.datalength(),dest2.buffer());   
  dest.freeBuf();
  delete[] incomingMsgs;
  return m;
}







/// Create fake sum detail data in the compressed format (for debugging)
 compressedBuffer fakeCompressedMessage(){
   CkPrintf("[%d] fakeCompressedMessage\n", CkMyPe());
   
   compressedBuffer fakeBuf(10000);
   
   int numBins = 55;
   int totalsize = 0;
   int numProcs = 1000;

   // build a compressed representation of each merged bin
   fakeBuf.push<numBins_T>(numBins);
   fakeBuf.push<numProcs_T>(numProcs);
   for(int i=0; i<numBins; i++){
     int numRecords = 3;
     fakeBuf.push<entriesInBin_T>(numRecords);
     for(int j=0;j<numRecords;j++){
       fakeBuf.push<ep_T>(j*10+2);
       fakeBuf.push<utilization_T>(120.00);
     }  
   }
   
   //CkPrintf("Fake Compressed Message:=========================================================\n");
   //   printCompressedBuf(fakeBuf);

   CkAssert(isCompressedBufferSane(fakeBuf));

   return fakeBuf;
 }


 /// Create an empty message
 compressedBuffer emptyCompressedBuffer(){
   compressedBuffer result(sizeof(numBins_T));
   result.push<numBins_T>(0);
   return result;
 }




/// Create a compressed buffer of the sum detail samples that occured since the previous call to this function (default max of 10000 bins).
compressedBuffer compressAvailableNewSumDetail(int max){
  const SumLogPool * p = CkpvAccess(_trace)->pool();
  const int numBinsAvailable = p->getNumEntries();
  int binsToSend = numBinsAvailable - CkpvAccess(previouslySentBins);
  if(binsToSend > max)
    binsToSend = max;

  return compressNRecentSumDetail(binsToSend);
}





/** print out the compressed buffer starting from its begining*/
void printCompressedBuf(compressedBuffer b){
  // b should be passed in by value, and hence we can modify it
  b.pos = 0;
  int numEntries = b.pop<numBins_T>();
  CkPrintf("Buffer contains %d records\n", numEntries);
  int numProcs = b.pop<numProcs_T>();
  CkPrintf("Buffer represents an average over %d PEs\n", numProcs);

  for(int i=0;i<numEntries;i++){
    entriesInBin_T recordLength = b.pop<entriesInBin_T>();
    if(recordLength > 0){
      CkPrintf("    Record %d is of length %d : ", i, recordLength);
      
      for(int j=0;j<recordLength;j++){
	ep_T ep = b.pop<ep_T>();
	utilization_T v = b.pop<utilization_T>();
	CkPrintf("(%d,%f) ", ep, v);
      }
    
      CkPrintf("\n");
    }
  }
}



 bool isCompressedBufferSane(compressedBuffer b){
   // b should be passed in by value, and hence we can modify it  
   b.pos = 0;  
   numBins_T numBins = b.pop<numBins_T>();  
   numProcs_T numProcs = b.pop<numProcs_T>();  
   
   if(numBins > 2000){
     ckout << "WARNING: numBins=" << numBins << endl;
     return false;
   }
   
   for(int i=0;i<numBins;i++){  
     entriesInBin_T recordLength = b.pop<entriesInBin_T>();  
     if(recordLength > 200){
       ckout << "WARNING: recordLength=" << recordLength << endl;
       return false;
     }
     
     if(recordLength > 0){  
       
       for(int j=0;j<recordLength;j++){  
         ep_T ep = b.pop<ep_T>();  
         utilization_T v = b.pop<utilization_T>();  
         //      CkPrintf("(%d,%f) ", ep, v);  
	 if(((ep>800 || ep <0 ) && ep != other_EP) || v < 0.0 || v > 251.0){
	   ckout << "WARNING: ep=" << ep << " v=" << v << endl;
	   return false;
	 }
       }  
       
     }  
   }  
   
   return true;
 }



 double averageUtilizationInBuffer(compressedBuffer b){
   // b should be passed in by value, and hence we can modify it  
   b.pos = 0;  
   numBins_T numBins = b.pop<numBins_T>();  
   numProcs_T numProcs = b.pop<numProcs_T>();  
   
   //   CkPrintf("[%d] averageUtilizationInBuffer numProcs=%d   (grep reduction message)\n", CkMyPe(), numProcs);
   
   double totalUtilization = 0.0;
   
   for(int i=0;i<numBins;i++) {  
     entriesInBin_T entriesInBin = b.pop<entriesInBin_T>();     
     for(int j=0;j<entriesInBin;j++){  
       ep_T ep = b.pop<ep_T>();  
       totalUtilization +=  b.pop<utilization_T>();  
     }
   }
   
   return totalUtilization / numBins / 2.5;
 }
 
 

 void sanityCheckCompressedBuf(compressedBuffer b){  
   CkAssert(isCompressedBufferSane(b)); 
 }  
 


 /// Print out some information about the sum detail statistics.
 void printSumDetailInfo(int desiredBinsToSend){
   //   CkPrintf("printSumDetailInfo(desiredBinsToSend=%d)\n", desiredBinsToSend);

   int _numEntries=_entryTable.size();
   SumLogPool * p = CkpvAccess(_trace)->pool();
   int numBinsAvailable = p->getNumEntries();

   int binsToSend = desiredBinsToSend;
   if(binsToSend > numBinsAvailable)
     binsToSend = numBinsAvailable;

   int startBin = numBinsAvailable - binsToSend;
  
   //CkPrintf("printSumDetailInfo() binsToSend=%d\n", binsToSend);

   if (binsToSend < 1) {
     //CkPrintf("printSumDetailInfo() No Bins\n");
   } else {
     double u = 0.0;

     for(int i=0; i<binsToSend; i++) {
       for(int e=0; e<_numEntries; e++) {
	 u += p->getUtilization(i+startBin,e);
       }
     }
    
     double uu = u / binsToSend;
     
     // CkPrintf("printSumDetailInfo()                         uu = %lf\n", uu);
            
   }
   
 }


 
 /// Create a compressed buffer of the n most recent sum detail samples
 compressedBuffer compressNRecentSumDetail(int desiredBinsToSend){
   //   CkPrintf("compressNRecentSumDetail(desiredBinsToSend=%d)\n", desiredBinsToSend);

   int _numEntries=_entryTable.size();
   SumLogPool * p = CkpvAccess(_trace)->pool();
   int numBinsAvailable = p->getNumEntries();

   int binsToSend = desiredBinsToSend;
   if(binsToSend > numBinsAvailable)
     binsToSend = numBinsAvailable;

   int startBin = numBinsAvailable - binsToSend;
  
   //   CkPrintf("compressNRecentSumDetail binsToSend=%d\n", binsToSend);

   if (binsToSend < 1) {
     return emptyCompressedBuffer();
   } else {
     compressedBuffer b(8*(2+_numEntries) * (2+binsToSend)+100);

     b.push<numBins_T>(binsToSend);
     b.push<numProcs_T>(1); // number of processors along reduction subtree. I am just one processor.

     for(int i=0; i<binsToSend; i++) {
       // Create a record for bin i
       //  CkPrintf("Adding record for bin %d\n", i);
       int numEntriesInRecordOffset = b.push<entriesInBin_T>(0); // The number of entries in this record
       
       for(int e=0; e<_numEntries; e++) {
	 double scaledUtilization = p->getUtilization(i+startBin,e) * 2.5; // use range of 0 to 250 for the utilization, so it can fit in an unsigned char
	 if(scaledUtilization > 250.0)
	   scaledUtilization = 250.0;
	 
	 if(scaledUtilization > 0.0) {
	   //	   CkPrintf("Adding non-zero entry (%d,%lf) to bin %d\n", e, scaledUtilization, i);
	   b.push<ep_T>(e);
	   b.push<utilization_T>(scaledUtilization);
	   b.increment<entriesInBin_T>(numEntriesInRecordOffset);
	 } else{
	   
	 }
       }
     }


     //     CkPrintf("[%d] compressNRecentSumDetail resulting buffer: averageUtilizationInBuffer()=%lf\n", CkMyPe(), averageUtilizationInBuffer(b));
     
     CkpvAccess(previouslySentBins) += binsToSend;    
     return b;
   }
   
 }



/** Merge the compressed entries from the first bin in each of the srcBuf buffers.
     
*/
 void mergeCompressedBin(compressedBuffer *srcBufferArray, int numSrcBuf, int *numProcsRepresentedInMessage, int totalProcsAcrossAllMessages, compressedBuffer &destBuf){
  // put a counter at the beginning of destBuf
  int numEntriesInDestRecordOffset = destBuf.push<entriesInBin_T>(0);
  
  //  CkPrintf("BEGIN MERGE------------------------------------------------------------------\n");
  
  // Read off the number of bins in each buffer
  int *remainingEntriesToRead = new int[numSrcBuf];
  for(int i=0;i<numSrcBuf;i++){
    remainingEntriesToRead[i] = srcBufferArray[i].pop<entriesInBin_T>();
  }

  int count = 0;
  // Count remaining entries to process
  for(int i=0;i<numSrcBuf;i++){
    count += remainingEntriesToRead[i];
  }
  
  while (count>0) {
    // find first EP from all buffers (these are sorted by EP already)
    int minEp = 10000;
    for(int i=0;i<numSrcBuf;i++){
      if(remainingEntriesToRead[i]>0){
	int ep = srcBufferArray[i].peek<ep_T>();
	if(ep < minEp){
	  minEp = ep;
	}
      }
    }
    
    //   CkPrintf("[%d] mergeCompressedBin minEp found was %d   totalProcsAcrossAllMessages=%d\n", CkMyPe(), minEp, (int)totalProcsAcrossAllMessages);
    
    destBuf.increment<entriesInBin_T>(numEntriesInDestRecordOffset);

    // Merge contributions from all buffers that list the EP
    double v = 0.0;
    for(int i=0;i<numSrcBuf;i++){
      if(remainingEntriesToRead[i]>0){
	int ep = srcBufferArray[i].peek<ep_T>(); 
	if(ep == minEp){
	  srcBufferArray[i].pop<ep_T>();  // pop ep
	  double util = srcBufferArray[i].pop<utilization_T>();
	  v += util * numProcsRepresentedInMessage[i];
	  remainingEntriesToRead[i]--;
	  count --;
	}
      }
    }

    // create a new entry in the output for this EP.
    destBuf.push<ep_T>(minEp);
    destBuf.push<utilization_T>(v / (double)totalProcsAcrossAllMessages);

  }


  delete [] remainingEntriesToRead;
  // CkPrintf("[%d] End of mergeCompressedBin:\n", CkMyPe() );
  // CkPrintf("END MERGE ------------------------------------------------------------------\n");
 }






#include "TraceSummary.def.h"


/*@}*/




















