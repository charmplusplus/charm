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

#define DEBUGF(x) CmiPrintf("%d/%d DEBUG: ", CkMyPe(), CkNumPes()); CmiPrintf x
#define VER 1.0

// for performance monitoring
extern "C" int start_counters( int e0, int e1 );
extern "C" int read_counters( int e0, long long *c0, int e1, long long *c1);

CpvStaticDeclare(Trace*, _trace);
CpvStaticDeclare(CountLogPool*, _logPool);
CpvStaticDeclare(char*, pgmName);
static int _numEvents = 0;
static int _threadMsg, _threadChare, _threadEP;
static int _packMsg, _packChare, _packEP;
static int _unpackMsg, _unpackChare, _unpackEP;
CpvDeclare(double, version);

//! the following is the list of arguments that can be passed to 
//! the +counter{1|2} command line arguments
//! to add or change, create a CounterArg struct with 
//! three constructor arguments:
//!   1) the code (for SGI libperfex) associated with counter
//!   2) the string to be entered on the command line
//!   3) a string that is the description of the counter
//! then go to the TraceCounter::TraceCounter() definition and make 
//! sure the CounterArg struct is registered via the registerArg call
static TraceCounter::CounterArg arg0(0, "CYCLE",      "Cycles");
static TraceCounter::CounterArg arg1(1, "INSTR",      "Issued instructions");
static TraceCounter::CounterArg arg2(2, "LOAD",       "Issued loads");
static TraceCounter::CounterArg arg3(3, "STORE",      "Issued stores");
static TraceCounter::CounterArg arg4(4, "STORE_COND", "Issued store conditionals");
static TraceCounter::CounterArg arg5(5, "FAIL_COND",  "Failed store conditionals");
static TraceCounter::CounterArg arg6(6, "DECODE_BR",  "Decoded branches.  (This changes meaning in 3.x versions of R10000.  It becomes resolved branches)");
static TraceCounter::CounterArg arg7(7, "QUADWORDS",  "Quadwords written back from secondary cache");

/*
7 = 
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
*/

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
  registerArg(&arg0);
  registerArg(&arg1);
  registerArg(&arg2);
  registerArg(&arg3);
  registerArg(&arg4);
  registerArg(&arg5);
  registerArg(&arg6);
  registerArg(&arg7);
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
    CmiAbort("ERROR: start_counters()");
  }
}

void TraceCounter::endExecute(void)
{
  long long value1 = 0, value2 = 0;
  int genRead;
  if ((genRead=read_counters(counter1_, &value1, counter2_, &value2)) < 0 ||
      genRead != genStart_)
  {
    CmiAbort("ERROR: read_counters()");
  }
  double t = TraceTimer();

  DEBUGF(("EP %d Time %f counter1 %d counter2 %ld\n", 
	  execEP_, t-startEP_, value1, value2));
  if (execEP_ != -1) { 
    CpvAccess(_logPool)->setEp(execEP_, value1, value2, t-startEP_); 
  }
}

void TraceCounter::beginPack(void) 
{ 
  startPack_ = CmiWallTimer(); 
  if ((genStart_=start_counters(counter1_, counter2_)) < 0) {
    CmiAbort("ERROR: start_counters()");
  }
}

void TraceCounter::endPack(void) 
{
  long long value1 = 0, value2 = 0;
  int genRead;
  if ((genRead=read_counters(counter1_, &value1, counter2_, &value2)) < 0 ||
      genRead != genStart_)
  {
    CmiAbort("ERROR: read_counters()");
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
    CmiAbort("ERROR: start_counters()");
  }
}

void TraceCounter::endUnpack(void) 
{
  long long value1 = 0, value2 = 0;
  int genRead;
  if ((genRead=read_counters(counter1_, &value1, counter2_, &value2)) < 0 ||
      genRead != genStart_)
  {
    CmiAbort("ERROR: read_counters()");
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
