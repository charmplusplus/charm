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

#include "conv-mach.h"
// #define CMK_ORIGIN2000
#ifdef CMK_ORIGIN2000
#include "trace-counter.h"
#include "limits.h"  // for LONGLONG_MAX

#define DEBUGF(x) // CmiPrintf("%d/%d DEBUG: ", CkMyPe(), CkNumPes()); CmiPrintf x
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
  DEBUGF(("%d createTraceCounter\n", CkMyPe()));
  CpvInitialize(Trace*, _trace);
  TraceCounter* tc = new TraceCounter();  _MEMCHECK(tc);
  tc->traceInit(argv);
  CpvAccess(_trace) = tc;
  CpvAccess(_traces)->addTrace(CpvAccess(_trace));
}

StatTable::StatTable(char** args, int argc): stats_(NULL), numStats_(0)
{
  DEBUGF(("StatTable::StatTable %08x size %d\n", this, argc));

  stats_ = new Statistics[argc];  _MEMCHECK(stats_);
  numStats_ = argc;
  for (int i=0; i<argc; i++) { 
    DEBUGF(("  %d name %s\n", i, args[i]));
    stats_[i].name = args[i]; 
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
  
  int count = stats_[stat].count[epidx];
  double avg = stats_[stat].average[epidx];
  stats_[stat].count[epidx]++;
  stats_[stat].average[epidx] = (avg * count + value) / (count + 1);
  stats_[stat].totTime[epidx] += time;
}

//! write three lines for each stat:
//!   1. number of calls for each entry
//!   2. average count for each entry
//!   3. total time in us spent for each entry
void StatTable::write(FILE* fp) 
{
  DEBUGF(("Writing StatTable\n"));
  int i, j;
  for (i=0; i<numStats_; i++) {
    // write number of calls for each entry
    fprintf(fp, "[%s num_called] ", stats_[i].name);
    for (j=0; j<_numEntries; j++) { 
      fprintf(fp, "%d ", stats_[i].count[j]); 
    }
    fprintf(fp, "\n");
    // write average count for each 
    fprintf(fp, "[%s average_count] ", stats_[i].name);
    for (j=0; j<_numEntries; j++) { 
      fprintf(fp, "%f ", stats_[i].average[j]); 
    }
    fprintf(fp, "\n");
    // write total time in us spent for each entry
    fprintf(fp, "[%s total_time] ", stats_[i].name);
    for (j=0; j<_numEntries; j++) {
      fprintf(fp, "%ld ", (long)(stats_[i].totTime[j]*1.0e6));
    }
    fprintf(fp, "\n");
  }
  DEBUGF(("Finished writing StatTable\n"));
}

void StatTable::clear() 
{
  for (int i=0; i<numStats_; i++) {
    for (int j=0; j<MAX_ENTRIES; j++) {
      stats_[i].count[j] = 0;
      stats_[i].average[j] = 0.0;
      stats_[i].totTime[j] = 0.0;
    }
  }
}

CountLogPool::CountLogPool(char* pgm, char** args, int argc): 
  stats_(args, argc)
{
  DEBUGF(("CountLogPool::CountLogPool() %08x\n", this));
  for (int j=0; j<argc; j++) { DEBUGF(("  %d %s\n", j, args[j])); }

  char pestr[10];
  sprintf(pestr, "%d", CkMyPe());
  int len = strlen(pgm) + strlen(".count.") + strlen(pestr) + 1;
  char* fname = new char[len+1];  _MEMCHECK(fname);
  sprintf(fname, "%s.%s.count", pgm, pestr);
  fp_ = NULL;
  DEBUGF(("TRACE: %s:%d\n", fname, errno));
  do {
    fp_ = fopen(fname, "w+");
  } while (!fp_ && errno == EINTR);
  delete[] fname;
  if(!fp_) {
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
  fprintf(fp_, "ver:%3.1f %d/%d ep:%d\n", 
	  CpvAccess(version), CmiMyPe(), CmiNumPes(), _numEntries);
  stats_.write(fp_);
}

void CountLogPool::writeSts(void)
{
  char *fname = new char[strlen(CpvAccess(pgmName))+strlen(".sts")+1];
  _MEMCHECK(fname);
  sprintf(fname, "%s.count.sts", CpvAccess(pgmName));
  FILE *sts = fopen(fname, "w+");
  // DEBUGF(("File: %s \n", fname));
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
    fprintf(sts, "MESSAGE %d %ld\n", i, _msgTable[i]->size);
  for(i=0;i<_numEvents;i++)
    fprintf(sts, "EVENT %d Event%d\n", i, i);
  fprintf(sts, "END\n");
  fclose(sts);
}

void CountLogPool::setEp(int epidx, long long count1, long long count2, double time) 
{
  // DEBUGF(("CountLogPool::setEp %08x %d %d %d %f\n", 
  //         this, epidx, count1, count2, time));

  if (epidx >= MAX_ENTRIES) {
    CmiAbort("CountLogPool::setEp too many entry points!\n");
  }
  stats_.setEp(epidx, 0, count1, time);
  stats_.setEp(epidx, 1, count2, time);
}

TraceCounter::TraceCounter() :
  execEP_      (-1),
  startEP_     (0.0),
  startPack_   (0.0),
  startUnpack_ (0.0),
  firstArg_    (NULL),
  lastArg_     (NULL),
  argStrSize_  (0)
{
  for (int i=0; i<NUM_COUNTER_ARGS; i++) { registerArg(&COUNTER_ARG[i]); }
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

  // parse command line args
  CounterArg counterArg1(0,NULL,NULL);
  CounterArg counterArg2(0,NULL,NULL);
  bool arg1valid = false;
  bool arg2valid = false;
  bool badArg = false;
  if (CmiGetArgString(argv,"+counter1",&counterArg1.arg)) {
    arg1valid = matchArg(&counterArg1);
    if (CkMyPe() == 0) { 
      DEBUGF(("arg1 is %s\n", counterArg1.arg)); 
      if (!arg1valid) { CmiPrintf("Bad +counter1 arg %s\n", counterArg1.arg); }
    }
    badArg = !arg1valid;
  }
  if (CmiGetArgString(argv,"+counter2",&counterArg2.arg)) {
    arg2valid = matchArg(&counterArg2);
    if (CkMyPe() == 0) { 
      DEBUGF(("arg2 is %s\n", counterArg2.arg)); 
      if (!arg2valid) { CmiPrintf("Bad +counter2 arg %s\n", counterArg2.arg); }
    }
    badArg = badArg || !arg2valid;
  }

  // check to see if args are valid, output if not
  if (badArg || CmiGetArgFlag(argv, "+counter-help")) {
    if (CkMyPe() == 0) { printHelp(); }
    ConverseExit();  return;
  }
  else if (!arg1valid || !arg2valid) {
    if (CkMyPe() == 0) {
      CmiPrintf("ERROR: You've linked with '+tracemode counter', therefore you\n"
		"  must specify '+counter1 <arg1>' and '+counter2 <arg2>' to run.\n"
		"  Type '+counter-help' to get list of counters.\n");
    }
    ConverseExit();  return;
  }

  // all args valid, now set up logging
  if (CkMyPe() == 0) {
    CmiPrintf("Running with tracemode=counter and args:\n"
	      "  <counter1>=%s %s\n"
	      "  <counter2>=%s %s\n",
	      counterArg1.arg, counterArg1.desc, 
	      counterArg2.arg, counterArg2.desc);
  }
  counter1_ = counterArg1.code;
  counter2_ = counterArg2.code;
  DEBUGF(("Counter1=%d Counter2=%d\n", counter1_, counter2_));
  char* args[2];  // prepare arguments for creating log pool
  args[0] = counterArg1.arg;
  args[1] = counterArg2.arg;
  CpvAccess(_logPool) = new CountLogPool(CpvAccess(pgmName), args, 2);
  _MEMCHECK(CpvAccess(_logPool));
  DEBUGF(("%d Created _logPool at %08x\n", CkMyPe(), CpvAccess(_logPool)));
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
  // DEBUGF(("start: %f \n", start));

  if ((genStart_=start_counters(counter1_, counter2_)) < 0) {
    CmiPrintf("counter1=%d counter2=%d\n", counter1_, counter2_);
    CmiAbort("ERROR: start_counters()\n");
  }

  DEBUGF(("beginExecute EP %d genStart %d\n", ep, genStart_));
}

void TraceCounter::endExecute(void)
{
  DEBUGF(("endExecute EP %d genStart_ %d ", execEP_, genStart_));

  long long value1 = 0, value2 = 0;
  int genRead;
  if ((genRead=read_counters(counter1_, &value1, counter2_, &value2)) < 0 ||
      genRead != genStart_)
  {
    traceWriteSts();
    CmiPrintf("genRead %d genStart_ %d counter1 %ld counter2 %ld\n",
	      genRead, genStart_, value1, value2);
    CmiAbort("ERROR: read_counters()\n");
  }
  double t = TraceTimer();

  DEBUGF(("genRead %d Time %f counter1 %d counter2 %ld\n", 
	  genRead, t-startEP_, value1, value2));
  if (execEP_ != -1) { 
    CpvAccess(_logPool)->setEp(execEP_, value1, value2, t-startEP_); 
  }
}

void TraceCounter::beginPack(void) 
{ 
  startPack_ = CmiWallTimer(); 
  if ((genStart_=start_counters(counter1_, counter2_)) < 0) {
    CmiPrintf("counter1=%d counter2=%d\n", counter1_, counter2_);
    CmiAbort("ERROR: start_counters()\n");
  }
  DEBUGF(("beginPack genStart %d\n", genStart_));
}

void TraceCounter::endPack(void) 
{
  DEBUGF(("endPack "));

  long long value1 = 0, value2 = 0;
  int genRead;
  if ((genRead=read_counters(counter1_, &value1, counter2_, &value2)) < 0 ||
      genRead != genStart_)
  {
    traceWriteSts();
    CmiPrintf("genRead %d genStart_ %d counter1 %ld counter2 %ld\n",
	      genRead, genStart_, value1, value2);
    CmiAbort("ERROR: read_counters()\n");
  }
  double t = CmiWallTimer();

  DEBUGF(("EP %d Time %f counter1 %d counter2 %ld\n", 
	  _packEP, t-startPack_, value1, value2));
  CpvAccess(_logPool)->setEp(_packEP, value1, value2, t - startPack_);
}

void TraceCounter::beginUnpack(void) 
{ 
  startUnpack_ = CmiWallTimer(); 
  if ((genStart_=start_counters(counter1_, counter2_)) < 0) {
    CmiPrintf("counter1=%d counter2=%d\n", counter1_, counter2_);
    CmiAbort("ERROR: start_counters()\n");
  }
  DEBUGF(("beginUnpack genStart %d\n", genStart_));
}

void TraceCounter::endUnpack(void) 
{
  DEBUGF(("endUnpack "));

  long long value1 = 0, value2 = 0;
  int genRead;
  if ((genRead=read_counters(counter1_, &value1, counter2_, &value2)) < 0 ||
      genRead != genStart_)
  {
    traceWriteSts();
    CmiPrintf("genRead %d genStart_ %d counter1 %ld counter2 %ld\n",
	      genRead, genStart_, value1, value2);
    CmiAbort("ERROR: read_counters()\n");
  }
  double t = CmiWallTimer();

  DEBUGF(("EP %d Time %f counter1 %d counter2 %ld\n", 
	  _unpackEP, t-startUnpack_, value1, value2));
  CpvAccess(_logPool)->setEp(_unpackEP, value1, value2, t-startUnpack_);
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
  // DEBUGF(("Matching %s or %d\n", arg->arg, matchCode));
  while (matchArg != NULL && !match) {
    // DEBUGF(("  Examining %d %s\n", matchArg->code, matchArg->arg));
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
  // DEBUGF(("Match = %d\n", match));
  return match;
}

//! print out all arguments in the linked-list of choices
void TraceCounter::printHelp()
{
  CmiPrintf(
    "Specify one of the following (code or str) after +counter1 and +counter2:\n\n"
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
