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

// http://www.scl.ameslab.gov/Projects/Rabbit/
// http://www.support.compaq.com/nttools

#include "conv-mach.h"
#ifdef CMK_ORIGIN2000
#include "trace-counter.h"

#define DEBUGF(x) // CmiPrintf x
#define VER 1.0

// for performance monitoring
extern "C" int start_counters(int e0, int e1);
extern "C" int read_counters(int e0, long long *c0, int e1, long long *c1);

CpvStaticDeclare(Trace*, _trace);
CpvStaticDeclare(CountLogPool*, _logPool);
CpvStaticDeclare(char*, pgmName);
static int _numEvents = 0;
static int _threadMsg, _threadChare, _threadEP;
static int _packMsg, _packChare, _packEP;
static int _unpackMsg, _unpackChare, _unpackEP;
CpvDeclare(double, version);

//! The following is the list of arguments that can be passed to 
//!   the +counter{1|2} command line arguments.
//! To add or change, change NUM_COUNTER_ARGS and follow the examples
//! Use three constructor arguments:
//!   1) Code (for SGI libperfex) associated with counter.
//!   2) String to be entered on the command line.
//!   3) String that is the description of the counter.
//! All NUM_COUNTER_ARGS are automatically registered via
//!   TraceCounter::TraceCounter() definition.
static const int NUM_COUNTER_ARGS = 32;
static TraceCounter::CounterArg COUNTER_ARG[NUM_COUNTER_ARGS] = 
{ TraceCounter::CounterArg( 0, "CYCLES0",      "Cycles (also see code 16)"),
  TraceCounter::CounterArg( 1, "INSTR",        "Issued instructions"),
  TraceCounter::CounterArg( 2, "LOAD",         "Issued loads"),
  TraceCounter::CounterArg( 3, "STORE",        "Issued stores"),
  TraceCounter::CounterArg( 4, "STORE_COND",   "Issued store conditionals"),
  TraceCounter::CounterArg( 5, "FAIL_COND",    "Failed store conditionals"),
  TraceCounter::CounterArg( 6, "DECODE_BR",    "Decoded branches.  (This changes meaning in 3.x versions of R10000.  It becomes resolved branches)"),
  TraceCounter::CounterArg( 7, "QUADWORDS2",   "Quadwords written back from secondary cache"),
  TraceCounter::CounterArg( 8, "CACHE_ER2",    "Correctable secondary cache data array ECC errors"),
  TraceCounter::CounterArg( 9, "L1_IMISS",     "Primary (L1) instruction cache misses"),
  TraceCounter::CounterArg(10, "L2_IMISS",     "Secondary (L2) instruction cache misses"),
  TraceCounter::CounterArg(11, "INSTRMISPR",   "Instruction misprediction from secondary cache way prediction table"),
  TraceCounter::CounterArg(12, "EXT_INTERV",   "External interventions"),
  TraceCounter::CounterArg(13, "EXT_INVAL",    "External invalidations"),
  TraceCounter::CounterArg(14, "VIRT_COHER",   "Virtual coherency conditions.  (This changes meaning in 3.x versions of R10000.  It becomes ALU/FPU forward progress cycles.  On the R12000, this counter is always 0)."),
  TraceCounter::CounterArg(15, "GR_INSTR15",   "Graduated instructions (also see code 17)"),
  TraceCounter::CounterArg(16, "CYCLES16",     "Cycles (also see code 0)"),
  TraceCounter::CounterArg(17, "GR_INSTR17",   "Graduated instructions (also see code 15)"),
  TraceCounter::CounterArg(18, "GR_LOAD" ,     "Graduated loads"),
  TraceCounter::CounterArg(19, "GR_STORE",     "Graduated stores"),
  TraceCounter::CounterArg(20, "GR_ST_COND",   "Graduated store conditionals"),
  TraceCounter::CounterArg(21, "GR_FLOPS",     "Graduated floating point instructions"),
  TraceCounter::CounterArg(22, "QUADWORDS1",   "Quadwords written back from primary data cache"),
  TraceCounter::CounterArg(23, "TLB_MISS",     "TLB misses"),
  TraceCounter::CounterArg(24, "MIS_BR",       "Mispredicted branches"),
  TraceCounter::CounterArg(25, "L1_DMISS",     "Primary (L1) data cache misses"),
  TraceCounter::CounterArg(26, "L2_DMISS",     "Primary (L2) data cache misses"),
  TraceCounter::CounterArg(27, "DATA_MIS",     "Data misprediction from secondary cache way predicition table"),
  TraceCounter::CounterArg(28, "EXT_INTERV2",  "External intervention hits in secondary cache (L2)"),
  TraceCounter::CounterArg(29, "EXT_INVAL2",   "External invalidation hits in secondary cache"),
  TraceCounter::CounterArg(30, "CLEAN_ST_PRE", "Store/prefetch exclusive to clean block in secondary cache"),
  TraceCounter::CounterArg(31, "SHARE_ST_PRE", "Store/prefetch exclusive to shared block in secondary cache") 
};

void _createTracecounter(char **argv)
{
  DEBUGF(("%d/%d DEBUG: createTraceCounter\n", CkMyPe(), CkNumPes()));
  CpvInitialize(Trace*, _trace);
  TraceCounter* tc = new TraceCounter();  _MEMCHECK(tc);
  tc->traceInit(argv);
  CpvAccess(_trace) = tc;
  CpvAccess(_traces)->addTrace(CpvAccess(_trace));
}

StatTable::StatTable():
  stats_(NULL), numStats_(0)
{
  DEBUGF(("%d/%d DEBUG: StatTable::StatTable %08x size %d\n", 
          CkMyPe(), CkNumPes(), this, argc));
}

void StatTable::init(char** name, char** desc, int argc)
{
  if (argc > numStats_) {
    if (stats_ != NULL) { delete [] stats_; }
    stats_ = new Statistics[argc];  _MEMCHECK(stats_);
    numStats_ = argc;
  }
  for (int i=0; i<argc; i++) { 
    DEBUGF(("%d/%d DEBUG:   %d name %s\n     desc %s\n", 
	    CkMyPe(), CkNumPes(), i, name[i], desc[i]));
    stats_[i].name = name[i]; 
    stats_[i].desc = desc[i];
  }
  clear();
}

StatTable::~StatTable() { if (stats_ != NULL) { delete [] stats_; } }

//! one entry is called for 'time' seconds, value is counter reading
void StatTable::setEp(int epidx, int stat, long long value, double time) 
{
  // CmiPrintf("StatTable::setEp %08x %d %d %d %f\n", 
  //           this, epidx, stat, value, time);

  CkAssert(epidx<MAX_ENTRIES);
  CkAssert(stat<numStats_);
  
  int numCalled = stats_[stat].numCalled[epidx];
  double avg = stats_[stat].avgCount[epidx];
  stats_[stat].numCalled[epidx]++;
  stats_[stat].avgCount[epidx] = (avg * numCalled + value) / (numCalled + 1);
  stats_[stat].totTime[epidx] += time;
}

//! write three lines for each stat:
//!   1. number of calls for each entry
//!   2. average count for each entry
//!   3. total time in us spent for each entry
void StatTable::write(FILE* fp) 
{
  DEBUGF(("%d/%d DEBUG: Writing StatTable\n", CkMyPe(), CkNumPes()));
  int i, j;
  for (i=0; i<numStats_; i++) {
    // write description of the entry
    fprintf(fp, "[%s {%s}]\n", stats_[i].name, stats_[i].desc);
    // write number of calls for each entry
    fprintf(fp, "[%s num_called] ", stats_[i].name);
    for (j=0; j<_numEntries; j++) { 
      fprintf(fp, "%d ", stats_[i].numCalled[j]); 
    }
    fprintf(fp, "\n");
    // write average count for each 
    fprintf(fp, "[%s avg_count] ", stats_[i].name);
    for (j=0; j<_numEntries; j++) { 
      fprintf(fp, "%f ", stats_[i].avgCount[j]); 
    }
    fprintf(fp, "\n");
    // write total time in us spent for each entry
    fprintf(fp, "[%s total_time(us)] ", stats_[i].name);
    for (j=0; j<_numEntries; j++) {
      fprintf(fp, "%f ", stats_[i].totTime[j]*1e6);
    }
    fprintf(fp, "\n");
  }
  DEBUGF(("%d/%d DEBUG: Finished writing StatTable\n", CkMyPe(), CkNumPes()));
}

void StatTable::clear() 
{
  for (int i=0; i<numStats_; i++) {
    for (int j=0; j<MAX_ENTRIES; j++) {
      stats_[i].numCalled[j] = 0;
      stats_[i].avgCount[j] = 0.0;
      stats_[i].totTime[j] = 0.0;
    }
  }
}

CountLogPool::CountLogPool():
  stats_     (),
  traceOn_   (false), 
  lastPhase_ (-1), 
  dirty_     (false)
{
  DEBUGF(("%d/%d DEBUG: CountLogPool::CountLogPool() %08x\n", 
          CkMyPe(), CkNumPes(), this));
}

// open file, if phase is -1, don't add the phase string
FILE* CountLogPool::openFile(int phase) {
  // technically, the sprintf into pestr & phasestr are unnecessary,
  // can just make a limit and check for that

  DEBUGF(("%d CountLogPool::openFile(%d)\n", CkMyPe(), phase));
  const static int strSize = 10;
  char pestr[strSize+1];
  char phasestr[strSize+1];
  snprintf(pestr, strSize, "%d", CkMyPe());
  pestr[strSize] = '\0';
  int len = strlen(CpvAccess(pgmName)) + strlen("phase.count.") + 2*strSize + 1;
  char* fname = new char[len+1];  _MEMCHECK(fname);
  if (phase >= 0) { 
    snprintf(phasestr, strSize, "%d", phase);
    phasestr[strSize] = '\0';
    sprintf(fname, "%s.phase%s.%s.count", CpvAccess(pgmName), phasestr, pestr); 
  }
  else { sprintf(fname, "%s.%s.count", CpvAccess(pgmName), pestr); }
  FILE* fp = NULL;
  DEBUGF(("%d/%d DEBUG: TRACE: %s:%d\n", CkMyPe(), CkNumPes(), fname, errno));
  do {
    fp = fopen(fname, "w+");
  } while (!fp && errno == EINTR);
  delete[] fname;
  if(!fp) {
    CmiAbort("Cannot open Summary Trace File for writing...\n");
  }
  return fp;
}

CountLogPool::~CountLogPool() 
{ 
  // write updates with lastPhase, so call before writeSts()
  if (dirty_) {
    write(); 
    if (CmiMyPe()==0) { 
      CmiPrintf("~CountLogPool: TraceCounter dirty, writing results\n");
      writeSts(); 
    }
  }
}

void CountLogPool::write(int phase) 
{
  if (!dirty_) { return; }

  if (phase >= 0) { lastPhase_ = phase; }
  if (phase < 0 && lastPhase_ >= 0) { lastPhase_++;  phase = lastPhase_; }

  FILE* fp = (phase==-1) ? openFile() : openFile(phase); 
  fprintf(fp, "ver:%3.1f %d/%d ep:%d counters:%d\n", 
	  CpvAccess(version), CmiMyPe(), CmiNumPes(), _numEntries, 
	  stats_.numStats());
  stats_.write(fp);
  fclose(fp);
}

void CountLogPool::writeSts(int phase)
{
  if (!dirty_) { return; }

  // technically, the sprintf into phasestr is unnecessary,
  // can just make a limit and check for that

  if (phase >= 0) { lastPhase_ = phase; }
  DEBUGF(("%d CountLogPool::writeSts(%d)\n", CkMyPe(), phase));

  const static int strSize = 10;
  char phasestr[strSize+1];
  // add strSize for phase number
  char *fname = 
    new char[strlen(CpvAccess(pgmName))+strlen(".count.sts")+strSize];
  _MEMCHECK(fname);
  if (phase < 0 && lastPhase_ >= 0) { phase = lastPhase_; }
  if (phase >= 0) { 
    snprintf(phasestr, strSize, "%d", phase);
    phasestr[strSize] = '\0';
    sprintf(fname, "%s.phase%s.count.sts", CpvAccess(pgmName), phasestr); 
  } 
  else { sprintf(fname, "%s.count.sts", CpvAccess(pgmName)); }
  FILE *sts = fopen(fname, "w+");
  // DEBUGF(("%d/%d DEBUG: File: %s \n", CkMyPe(), CkNumPes(), fname));
  if(sts==0)
    CmiAbort("Cannot open summary sts file for writing.\n");
  CmiPrintf("WRITING FILE=%s\n", fname); 
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
    fprintf(sts, "MESSAGE %d %ld\n", i, _msgTable[i]->size);
  for(i=0;i<_numEvents;i++)
    fprintf(sts, "EVENT %d Event%d\n", i, i);
  fprintf(sts, "END\n");
  fclose(sts);
}

void CountLogPool::setEp(int epidx, long long count1, long long count2, double time) 
{
  dirty_ = true;
  // DEBUGF(("%d/%d DEBUG: CountLogPool::setEp %08x %d %d %d %f\n", 
  //        CkMyPe(), CkNumPes(), this, epidx, count1, count2, time));

  if (epidx >= MAX_ENTRIES) {
    CmiAbort("CountLogPool::setEp too many entry points!\n");
  }
  stats_.setEp(epidx, 0, count1, time);
  stats_.setEp(epidx, 1, count2, time);
}

TraceCounter::TraceCounter() :
  execEP_        (-1),
  startEP_       (0.0),
  startPack_     (0.0),
  startUnpack_   (0.0),
  firstArg_      (NULL),
  lastArg_       (NULL),
  argStrSize_    (0),
  phase_         (-1),
  traceOn_       (false),
  commandLine_   (NULL),
  commandLineSz_ (0),
  counter1_      (NULL),
  counter2_      (NULL),
  status_        (IDLE),
  dirty_         (false)
{
  for (int i=0; i<NUM_COUNTER_ARGS; i++) { registerArg(&COUNTER_ARG[i]); }
}

TraceCounter::~TraceCounter() { 
  if (commandLine_ != NULL) { delete [] commandLine_; }
}

void TraceCounter::traceBegin() {
  DEBUGF(("%d/%d traceBegin called\n", CkMyPe(), CkNumPes()));
  if (traceOn_) { 
      static bool print = true;
      if (print) {
	print = false;
	if (CkMyPe()==0) {
	  CmiPrintf("%d/%d WARN: traceBegin called but trace already on!\n"
		    "            Sure you didn't mean to use +traceoff?\n",
		    CkMyPe(), CkNumPes());
	}
    } 
  }
  else {
    phase_++;
    traceOn_ = true;
    CpvAccess(_logPool)->setTrace(true); 
  }
}

void TraceCounter::traceEnd() {
  DEBUGF(("%d/%d traceEnd called\n", CkMyPe(), CkNumPes()));
  if (!traceOn_) { 
    static bool print = true;
    if (print) {
      print = false;
      if (CkMyPe()==0) {
	CmiPrintf("%d/%d WARN: traceEnd called but trace not on!\n"
		  "            Sure you didn't mean to use +traceoff?\n",
		  CkMyPe(), CkNumPes());
      }
    }
  }
  else {
    traceOn_ = false;
    dirty_ = false;
    if (CmiMyPe()==0) { CpvAccess(_logPool)->writeSts(phase_); }
    CpvAccess(_logPool)->write(phase_); 
    CpvAccess(_logPool)->clearEps(); 
    CpvAccess(_logPool)->setTrace(false);  
    counter1_ = counter1_->next;
    counter2_ = counter2_->next;
    char* name[2];
    char* desc[2];
    name[0] = counter1_->arg;  desc[0] = counter1_->desc;
    name[1] = counter2_->arg;  desc[1] = counter2_->desc;
    CpvAccess(_logPool)->init(name, desc, 2);
    // setTrace must go after the writes otherwise the writes won't go through
    DEBUGF(("%d/%d DEBUG: Created _logPool at %08x\n", 
	    CkMyPe(), CkNumPes(), CpvAccess(_logPool)));
  }
}
  
void TraceCounter::traceInit(char **argv)
{
  CpvInitialize(CountLogPool*, _logPool);
  CpvInitialize(char*, pgmName);
  CpvInitialize(double, version);
  CpvAccess(pgmName) = (char *) malloc(strlen(argv[0])+1);
  _MEMCHECK(CpvAccess(pgmName));
  strcpy(CpvAccess(pgmName), argv[0]);
  CpvAccess(version) = VER;

  int i;
  // parse command line args
  char* counters = NULL;
  CounterArg* commandLine_ = NULL;
  bool badArg = false;
  int numCounters = 0;
  if (CmiGetArgString(argv, "+counters", &counters)) {
    if (CkMyPe()==0) { CmiPrintf("Multiple counters: %s\n", counters); }
    int offset = 0;
    int limit = strlen(counters);
    char* ptr = counters;
    while (offset < limit && 
	   (ptr = strtok(&counters[offset], ",")) != NULL) 
    { 
      offset += strlen(ptr)+1;
      ptr = &ptr[strlen(ptr)+1];
      numCounters++; 
    }
    if (CkMyPe()==0) { 
      CmiPrintf("There are %d counters\n", numCounters); 
    }
    commandLine_ = new CounterArg[numCounters];
    ptr = counters;
    for (i=0; i<numCounters; i++) {
      commandLine_[i].arg = ptr;
      if (!matchArg(&commandLine_[i])) { badArg = true; }
      ptr = &ptr[strlen(ptr)+1];
    }
  }
  commandLineSz_ = numCounters;

  // check to see if args are valid, output if not
  if (badArg || CmiGetArgFlag(argv, "+counterhelp")) {
    if (CkMyPe() == 0) { printHelp(); }
    ConverseExit();  return;
  }
  else if (counters == NULL) {
    if (CkMyPe() == 0) {
      CmiPrintf(
	"ERROR: You've linked with '-tracemode counter', therefore you\n"
	"  must specify '+counters <counters>' where <counters> is\n"
	"  comma delimited list of valid counters\n"
	"Type '+counterhelp' to get list of counters.\n");
    }
    ConverseExit();  return;
  }

  // parse through commandLine_, figure out which belongs on which list (1 vs 2)
  CounterArg* last1 = NULL;
  CounterArg* last2 = NULL;
  CounterArg* tmp = NULL;
  for (i=0; i<commandLineSz_; i++) {
    tmp = &commandLine_[i];
    if (tmp->code < NUM_COUNTER_ARGS/2) {
      if (counter1_ == NULL) { counter1_ = tmp;  last1 = counter1_; }
      else { last1->next = tmp;  last1 = tmp; }
    }
    else {
      if (counter2_ == NULL) { counter2_ = tmp;  last2 = counter2_; }
      else { last2->next = tmp;  last2 = tmp; }
    }
  }
  if (counter1_ == NULL) {
    printHelp();
    CmiPrintf("\nMust specify some counters with code < %d\n", 
	      NUM_COUNTER_ARGS/2);
    ConverseExit();
  }
  if (counter2_ == NULL) {
    printHelp();
    CmiPrintf("\nMust specify some counters with code >= %d\n", 
	      NUM_COUNTER_ARGS/2);
    ConverseExit();
  }
  last1->next = counter1_;
  last2->next = counter2_;

  // all args valid, now set up logging
  if (CkMyPe() == 0) {
    CmiPrintf("Running with tracemode=counter and args:\n");
    // print out counter1 set
    tmp = counter1_;
    i = 0;
    do {
      CmiPrintf("  <counter1-%d>=%d %s %s\n", i, tmp->code, tmp->arg, tmp->desc);
      tmp = tmp->next;
      i++;
    } while (tmp != counter1_);
    // print out counter2 set
    tmp = counter2_;
    i = 0;
    do {
      CmiPrintf("  <counter2-%d>=%d %s %s\n", i, tmp->code, tmp->arg, tmp->desc);
      tmp = tmp->next;
      i++;
    } while (tmp != counter2_);
  }

  // DEBUGF(("    DEBUG: Counter1=%d Counter2=%d\n", counter1_, counter2_));
  char* name[2];  // prepare names for creating log pool
  char* desc[2];  // prepare descriptions for creating log 
  name[0] = counter1_->arg;  desc[0] = counter1_->desc;
  name[1] = counter2_->arg;  desc[1] = counter2_->desc;
  CpvAccess(_logPool) = new CountLogPool();
  _MEMCHECK(CpvAccess(_logPool));
  CpvAccess(_logPool)->init(name, desc, 2);
  DEBUGF(("%d/%d DEBUG: Created _logPool at %08x\n", 
          CkMyPe(), CkNumPes(), CpvAccess(_logPool)));
}

void TraceCounter::traceClearEps(void)
{
  CpvAccess(_logPool)->clearEps();
}

void TraceCounter::traceWriteSts(void)
{
  if (traceOn_) {
    if (CmiMyPe()==0) { CpvAccess(_logPool)->writeSts(); }
  }
}

void TraceCounter::traceClose(void)
{
  CpvAccess(_trace)->endComputation();
  if (dirty_) {
    CpvAccess(_logPool)->write();
    if(CmiMyPe()==0) { 
      CpvAccess(_logPool)->writeSts(); 
      CmiPrintf("TraceCounter dirty, writing results\n");
    }
    CpvAccess(_logPool)->setDirty(false);
    dirty_ = false;
  }
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
  DEBUGF(("%d/%d DEBUG: beginExecute EP %d\n", CkMyPe(), CkNumPes(), ep));

  if (status_!= IDLE) {
    static bool print = true;
    if (print) {
      print = false;
      if (CkMyPe()==0) { 
	CmiPrintf("WARN: %d beginExecute called when status not IDLE!\n", 
		  CkMyPe());
      }
    }
    return;
  }
  else { status_=WORKING; }

  if ((genStart_=start_counters(counter1_->code, counter2_->code))<0)
  {
    CmiPrintf("genStart=%d counter1=%d counter2=%d\n", 
              genStart_, counter1_->code, counter2_->code);
    CmiAbort("ERROR: start_counters() in beginExecute\n");
  }

  DEBUGF(("%d/%d DEBUG:   beginExecute EP %d genStart %d\n", 
          CkMyPe(), CkNumPes(), ep, genStart_));
}

void TraceCounter::endExecute(void)
{
  DEBUGF(("%d/%d DEBUG: endExecute EP %d genStart_ %d\n", 
          CkMyPe(), CkNumPes(), execEP_, genStart_));

  if (status_!=WORKING) {
    static bool print = true;
    if (print) {
      print = false;
      if (CkMyPe()==0) {
	CmiPrintf("WARN: %d endExecute called when status not WORKING!\n", 
		  CkMyPe());
      }
    }
    return;
  }
  else { status_=IDLE; }

  double t = TraceTimer();

  long long value1 = 0, value2 = 0;
  int genRead;
  if ((genRead=read_counters(counter1_->code, &value1, counter2_->code, &value2)) < 0 ||
      genRead != genStart_)
  {
    CmiPrintf("genRead %d genStart_ %d counter1 %ld counter2 %ld\n",
	      genRead, genStart_, value1, value2);
    traceClose();
    CmiAbort("ERROR: read_counters() in endExecute\n");
  }

  DEBUGF((
    "%d/%d DEBUG:   endExecute genRead %d Time %f counter1 %d counter2 %ld\n", 
    CkMyPe(), CkNumPes(), genRead, t-startEP_, value1, value2));
  if (execEP_ != -1) { 
    dirty_ = true;
    CpvAccess(_logPool)->setEp(execEP_, value1, value2, t-startEP_); 
  }
}

void TraceCounter::beginPack(void) 
{ 
  DEBUGF(("%d/%d DEBUG: beginPack\n", CkMyPe(), CkNumPes()));

  /*
    // COMMENT OUT BECAUSE beginPack/endPack can get called between 
    // beginExecute/endExecute and time to call start_counters can be 
    // expensive
    
  if (status_!= IDLE) { 
    CmiPrintf("WARN: %d beginPack called when status not IDLE!\n", CkMyPe());
    return;
  }
  else { status_=WORKING; }

  startPack_ = CmiWallTimer(); 
  if ((genStart_=start_counters(counter1_->code, counter2_->code))<0) {
    CmiPrintf("genStart=%d counter1=%d counter2=%d\n", 
              genStart_, counter1_->code, counter2_->code);
    CmiAbort("ERROR: start_counters() in beginPack\n");
  }
  DEBUGF(("%d/%d DEBUG:   beginPack genStart %d\n", 
          CkMyPe(), CkNumPes(), genStart_));
  */
}

void TraceCounter::endPack(void) 
{
  DEBUGF(("%d/%d DEBUG: endPack\n", CkMyPe(), CkNumPes()));

  /*

    // COMMENT OUT BECAUSE beginPack/endPack can get called between 
    // beginExecute/endExecute and time to call start_counters can be 
    // expensive

  if (status_!=WORKING) {
    CmiPrintf("WARN: %d endPack called when status not WORKING!\n", CkMyPe());
    return;
  }
  else { status_=IDLE; }

  double t = CmiWallTimer();

  long long value1 = 0, value2 = 0;
  int genRead;
  if ((genRead=read_counters(counter1_->code, &value1, counter2_->code, &value2)) < 0 ||
      genRead != genStart_)
  {
    CmiPrintf("genRead %d genStart_ %d counter1 %ld counter2 %ld\n",
	      genRead, genStart_, value1, value2);
    traceClose();
    CmiAbort("ERROR: read_counters() in endPack\n");
  }

  DEBUGF(("%d/%d DEBUG:   endPack genRead %d EP %d Time %f counter1 %d counter2 %ld\n", 
	  CkMyPe(), CkNumPes(), genRead, _packEP, t-startPack_, value1, value2));
  dirty_ = true;
  CpvAccess(_logPool)->setEp(_packEP, value1, value2, t - startPack_);
  */
}

void TraceCounter::beginUnpack(void) 
{ 
  DEBUGF(("%d/%d DEBUG: beginUnpack\n", CkMyPe(), CkNumPes()));

  /*
    // COMMENT OUT BECAUSE beginUnpack/endUnpack can get called between 
    // beginExecute/endExecute and time to call start_counters can be 
    // expensive

  if (status_!= IDLE) { 
    CmiPrintf("WARN: %d beginUnpack called when status not IDLE!\n", CkMyPe());
    return;
  }
  else { status_=WORKING; }

  startUnpack_ = CmiWallTimer(); 
  if ((genStart_=start_counters(counter1_->code, counter2_->code))<0) {
    CmiPrintf("genStart=%d counter1=%d counter2=%d\n", 
              genStart_, counter1_->code, counter2_->code);
    CmiAbort("ERROR: start_counters() in beginUnpack\n");
  }
  DEBUGF(("%d/%d DEBUG:   beginUnpack genStart %d\n", 
          CkMyPe(), CkNumPes(), genStart_));
  */
}

void TraceCounter::endUnpack(void) 
{
  DEBUGF(("%d/%d DEBUG: endUnpack\n", CkMyPe(), CkNumPes()));
     
  /*
    // COMMENT OUT BECAUSE beginUnpack/endUnpack can get called between 
    // beginExecute/endExecute and time to call start_counters can be 
    // expensive

  if (status_!=WORKING) {
    CmiPrintf("WARN: %d endUnack called when status not WORKING!\n", CkMyPe());
    return;
  }
  else { status_=IDLE; }

  double t = CmiWallTimer();

  long long value1 = 0, value2 = 0;
  int genRead;
  if ((genRead=read_counters(counter1_->code, &value1, counter2_->code, &value2))<0 ||
      genRead != genStart_)
  {
    traceWriteSts();
    CmiPrintf("genRead %d genStart_ %d counter1 %ld counter2 %ld\n",
	      genRead, genStart_, value1, value2);
    CmiAbort("ERROR: read_counters() in endUnpack\n");
  }

  DEBUGF(("%d/%d DEBUG:   endUnpack genRead %d EP %d Time %f counter1 %d counter2 %ld\n", 
	  CkMyPe(), CkNumPes(), genRead, _unpackEP, t-startUnpack_, value1, value2));
  dirty_ = true;
  CpvAccess(_logPool)->setEp(_unpackEP, value1, value2, t-startUnpack_);
  */
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

//! add the argument parameters to the linked list of args choices
void TraceCounter::registerArg(CounterArg* arg)
{
  if (firstArg_ == NULL) {
    firstArg_ = lastArg_ = arg;
    argStrSize_ = strlen(arg->arg);
  }
  else { 
    // check to see if any redundancy 
    CounterArg* check = firstArg_;
    while (check != NULL) {
      if (strcmp(check->arg, arg->arg)==0 || check->code == arg->code) {
	if (CkMyPe()==0) { 
	  CmiPrintf("Two args with same name %s or code %d\n", 
		    arg->arg, arg->code); 
	}
	CmiAbort("TraceCounter::registerArg()\n");
      }
      check = check->next;
    }

    lastArg_->next = arg;
    lastArg_ = arg;
    int len = strlen(arg->arg);
    if (len > argStrSize_) { argStrSize_ = len; }
  }
}

//! see if the arg (str or code) matches any in the linked list of choices
//! and sets arg->code to the SGI code
//! return true if arg matches, false otherwise
bool TraceCounter::matchArg(CounterArg* arg)
{
  bool match = false;                // will be set to true if arg matches
  CounterArg* matchArg = firstArg_;  // traverse linked list
  int matchCode = atoi(arg->arg);    // in case user specs num on commline
  if (matchCode == 0) {
    if (arg->arg[0] != '0' || arg->arg[1] != '\0') { matchCode = -1; }
  }
  // DEBUGF(("%d/%d DEBUG: Matching %s or %d\n", CkMyPe(), CkNumPes(), arg->arg, matchCode));
  while (matchArg != NULL && !match) {
    // DEBUGF(("%d/%d DEBUG:   Examining %d %s\n", CkMyPe(), CkNumPes(), matchArg->code, matchArg->arg));
    if (strcmp(matchArg->arg, arg->arg)==0) {
      match = true;
      arg->code = matchArg->code;
      arg->desc = matchArg->desc;
    }
    else if (matchArg->code == matchCode) {
      match = true;
      arg->code = matchArg->code;
      arg->arg = matchArg->arg;
      arg->desc = matchArg->desc;
    }
    matchArg = matchArg->next;
  }
  // DEBUGF(("%d/%d DEBUG: Match = %d\n", CkMyPe(), CkNumPes(), match));
  return match;
}

//! print out all arguments in the linked-list of choices
void TraceCounter::printHelp()
{
  CmiPrintf(
    "Specify one of the following (code or str) after +counters:\n\n"
    "  code  str\n"
    "  ----  ---\n");

  // create a format so that all the str line up 
  char format[64];
  snprintf(format, 64, "    %%2d  %%-%ds  %%s\n", argStrSize_);

  CounterArg* help = firstArg_;
  while (help != NULL) {
    CmiPrintf(format, help->code, help->arg, help->desc);
    help = help->next;
  }
}

#endif // CMK_ORIGIN2000

/*@}*/
