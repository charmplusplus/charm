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

// 1 minutes of run before it'll fill up:
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
  if (CkMyPe()==0) CkPrintf("Charm++: Tracemode Summary enabled.\n");
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
}

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

  //CkPrintf("writing to detail file:%d    %d \n", getNumEntries(), numBins);
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
        fprintf(sdfp, "%ld", last);
        for(e=0; e<_numEntries; e++) {
            for(i=0; i<numBins; i++) {

                long u= getNumExecutions(i, e);
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
}

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
     pool[i].getIdleTime() = pool[i*2].getIdleTime() + pool[i*2+1].getIdleTime();
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

void SumLogPool::shrink(double _maxBinSize)
{
    while(CkpvAccess(binSize) < _maxBinSize)
    {
        shrink();
    };
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

TraceSummary::TraceSummary(char **argv):binStart(0.0),idleStart(0.0),
					binTime(0.0),binIdle(0.0),msgNum(0)
{
  if (CkpvAccess(traceOnPe) == 0) return;

    // use absolute time
  if (CmiTimerAbsolute()) binStart = CmiInitTime();

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
  inIdle = 0;
  inExec = 0;
  depth = 0;
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

    delete _logPool;
    CkpvAccess(_traces)->removeTrace(this);
}

void TraceSummary::beginExecute(CmiObjId *tid)
{
  beginExecute(-1,-1,_threadEP,-1);
}

void TraceSummary::beginExecute(envelope *e, void *obj)
{
  // no message means thread execution
  if (e==NULL) {
    beginExecute(-1,-1,_threadEP,-1);
  }
  else {
    beginExecute(-1,-1,e->getEpIdx(),-1);
  }  
}

void TraceSummary::beginExecute(char *msg)
{
#if CMK_SMP_TRACE_COMMTHREAD
    //This function is called from comm thread in SMP mode
    envelope *e = (envelope *)msg;
    int num = _entryTable.size();
    int ep = e->getEpIdx();
    if(ep<0 || ep>=num) return;
    if(_entryTable[ep]->traceEnabled)
        beginExecute(-1,-1,e->getEpIdx(),-1);
#endif
}

void TraceSummary::beginExecute(int event,int msgType,int ep,int srcPe, int mlen, CmiObjId *idx, void *obj)
{
  if (execEp == TRACEON_EP) {
    endExecute();
  }
  CmiAssert(inIdle == 0);
  if (inExec == 0) {
    CmiAssert(depth == 0);
    inExec = 1;
  }
  depth ++;
  // printf("BEGIN exec: %d %d %d\n", inIdle, inExec, depth);

  if (depth > 1) return;          //  nested

/*
  if (execEp != INVALIDEP) {
    TRACE_WARN("Warning: TraceSummary two consecutive BEGIN_PROCESSING!\n");
    return;
  }
*/
  
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

void TraceSummary::endExecute()
{
  CmiAssert(inIdle == 0 && inExec == 1);
  depth --;
  if (depth == 0) inExec = 0;
  CmiAssert(depth >= 0);
  // printf("END exec: %d %d %d\n", inIdle, inExec, depth);

  if (depth != 0) return;
 
  double t = TraceTimer();
  double ts = start;
  double nts = binStart;

/*
  if (execEp == TRACEON_EP) {
    // if trace just got turned on, then one expects to see this
    // END_PROCESSING event without seeing a preceeding BEGIN_PROCESSING
    return;
  }
*/

  if (execEp == INVALIDEP) {
    TRACE_WARN("Warning: TraceSummary END_PROCESSING without BEGIN_PROCESSING!\n");
    return;
  }

  if (execEp >= 0)
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

  if (sumDetail && execEp >= 0 )
      _logPool->updateSummaryDetail(execEp, start, t);

  execEp = INVALIDEP;
}

void TraceSummary::endExecute(char *msg){
#if CMK_SMP_TRACE_COMMTHREAD
    //This function is called from comm thread in SMP mode
    envelope *e = (envelope *)msg;
    int num = _entryTable.size();
    int ep = e->getEpIdx();
    if(ep<0 || ep>=num) return;
    if(_entryTable[ep]->traceEnabled){
        endExecute();
    }
#endif    
}

void TraceSummary::beginIdle(double currT)
{
  if (execEp == TRACEON_EP) {
    endExecute();
  }

  CmiAssert(inIdle == 0 && inExec == 0);
  inIdle = 1;
  //printf("BEGIN idle: %d %d %d\n", inIdle, inExec, depth);

  double t = TraceTimer(currT);
  
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
  CmiAssert(inIdle == 1 && inExec == 0);
  inIdle = 0;
  // printf("END idle: %d %d %d\n", inIdle, inExec, depth);

  double t = TraceTimer(currT);
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

void TraceSummary::traceBegin(void)
{
    // fake as a start of an event, assuming traceBegin is called inside an
    // entry function.
  beginExecute(-1, -1, TRACEON_EP, -1, -1);
}

void TraceSummary::traceEnd(void)
{
  endExecute();
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

void TraceSummaryBOC::traceSummaryParallelShutdown(int pe) {
   
    UInt    numBins = CkpvAccess(_trace)->pool()->getNumEntries();  
    //CkPrintf("trace shut down pe=%d bincount=%d\n", CkMyPe(), numBins);
    CProxy_TraceSummaryBOC sumProxy(traceSummaryGID);
    CkCallback cb(CkIndex_TraceSummaryBOC::maxBinSize(NULL), sumProxy[0]);
    contribute(sizeof(double), &(CkpvAccess(binSize)), CkReduction::max_double, cb);
}

// collect the max bin size
void TraceSummaryBOC::maxBinSize(CkReductionMsg *msg)
{
    double _maxBinSize = *((double *)msg->getData());
    CProxy_TraceSummaryBOC sumProxy(traceSummaryGID);
    sumProxy.shrink(_maxBinSize);
}

void TraceSummaryBOC::shrink(double _mBin){
    UInt    numBins = CkpvAccess(_trace)->pool()->getNumEntries();  
    UInt    epNums  = CkpvAccess(_trace)->pool()->getEpInfoSize();
    _maxBinSize = _mBin;
    if(CkpvAccess(binSize) < _maxBinSize)
    {
        CkpvAccess(_trace)->pool()->shrink(_maxBinSize);
    }
    double *sumData = CkpvAccess(_trace)->pool()->getCpuTime();  
    CProxy_TraceSummaryBOC sumProxy(traceSummaryGID);
    CkCallback cb(CkIndex_TraceSummaryBOC::sumData(NULL), sumProxy[0]);
    contribute(sizeof(double) * numBins * epNums, CkpvAccess(_trace)->pool()->getCpuTime(), CkReduction::sum_double, cb);
}

void TraceSummaryBOC::sumData(CkReductionMsg *msg) {
    double *sumData = (double *)msg->getData();
    int     totalsize = msg->getSize()/sizeof(double);
    UInt    epNums  = CkpvAccess(_trace)->pool()->getEpInfoSize();
    UInt    numBins = totalsize/epNums;  
    int     numEntries = epNums - NUM_DUMMY_EPS - 1; 
    char    *fname = new char[strlen(CkpvAccess(traceRoot))+strlen(".sumall")+1];
    sprintf(fname, "%s.sumall", CkpvAccess(traceRoot));
    FILE *sumfp = fopen(fname, "w+");
    delete [] fname;
    fprintf(sumfp, "ver:%3.1f cpu:%d numIntervals:%d numEPs:%d intervalSize:%e\n",
                CkpvAccess(version), CkNumPes(),
                numBins, numEntries, _maxBinSize);
    for(int i=0; i<numBins; i++){
        for(int j=0; j<numEntries; j++)
        {
            fprintf(sumfp, "%ld ", (long)(sumData[i*epNums+j]*1.0e6));
        }
    }
   fclose(sumfp);
   //CkPrintf("done with analysis\n");
   CkExit();
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

      CkPrintf("[%d] Setting up periodic startCollectData callback\n", CkMyPe());
      CcdCallOnConditionKeep(CcdPERIODIC_1second, startCollectData,
			     (void *)this);
      summaryCcsStreaming = true;
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
    }    

    datalength = sizeof(unsigned char) * numData;
    
    CcsSendDelayedReply(m->reply, datalength, (void *)sendBuffer);
    ccsBufferedData->free();
    delete [] sendBuffer;
  }
  CkPrintf("Response Sent. Proceeding with computation.\n");
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

   sumProxy.collectSummaryData(startTime, 
		       collectionGranularity,
 		       numBlocksToGet*indicesPerBlock);
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
  delete [] contribution;
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

  BinEntry *reductionBuffer = new BinEntry[size+1];
  reductionBuffer[size].time() = traced;  // last element is the traced pe count
  reductionBuffer[size].getIdleTime() = 0;  // last element is the traced pe count
  if (traced) {
    CkpvAccess(_trace)->endComputation();
    int n = CkpvAccess(_trace)->pool()->getNumEntries();
    BinEntry *localBins = CkpvAccess(_trace)->pool()->bins();
    if (n>size) n=size;
    for (int i=0; i<n; i++) reductionBuffer[i] = localBins[i];
  }

  contribute(sizeof(BinEntry)*(size+1), reductionBuffer, 
	     CkReduction::sum_double);
  delete [] reductionBuffer;
}

//extern "C" void _CkExit();

void TraceSummaryBOC::sendSummaryBOC(CkReductionMsg *msg)
{
  if (CkpvAccess(_trace) == NULL) return;

  CkAssert(CkMyPe() == 0);

  int n = msg->getSize()/sizeof(BinEntry);
  nBins = n-1;
  bins = (BinEntry *)msg->getData();
  nTracedPEs = (int)bins[n-1].time();
  // CmiPrintf("traced: %d entry:%d\n", nTracedPEs, nBins);

  write();

  delete msg;

  CkExit();
}

void TraceSummaryBOC::write(void) 
{
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
#if CMK_TRACE_ENABLED
  CmiPrintf("[%d] CombineSummary called!\n", CkMyPe());
  if (sumonly) {
    CmiPrintf("[%d] Sum Only start!\n", CkMyPe());
      // pe 0 start the sumonly process
    CProxy_TraceSummaryBOC sumProxy(traceSummaryGID);
    sumProxy[0].startSumOnly();
  }else if(sumDetail)
  {
      CProxy_TraceSummaryBOC sumProxy(traceSummaryGID);
      sumProxy.traceSummaryParallelShutdown(-1);
  }
  else {
    _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, _threadEP,CkMyPe(), 0, NULL, NULL);
    CkExit();
  }
#else
  CkExit();
#endif
}

void initTraceSummaryBOC()
{
#ifdef __BIGSIM__
  if(BgNodeRank()==0) {
#else
  if (CkMyRank() == 0) {
#endif
    registerExitFn(CombineSummary);
  }
}






#include "TraceSummary.def.h"


/*@}*/




















