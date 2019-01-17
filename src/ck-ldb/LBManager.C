/**
 * \addtogroup CkLdb
*/
/*@{*/

#include <charm++.h>
#include "converse.h"

/*
 * This C++ file contains the Charm stub functions
 */

#include "LBManager.h"
#include "LBSimulation.h"
#include "topology.h"
#include "DistributedLB.h"

CkGroupID _lbmgr;

CkpvDeclare(LBUserDataLayout, lbobjdatalayout);
CkpvDeclare(int, _lb_obj_index);

CkpvDeclare(int, numLoadBalancers);  /**< num of lb created */
CkpvDeclare(bool, lbmanagerInited);  /**< true if lbdatabase is inited */

// command line options
CkLBArgs _lb_args;
int _lb_predict=0;
int _lb_predict_delay=10;
int _lb_predict_window=20;
bool _lb_psizer_on = false;

// registry class stores all load balancers linked and created at runtime
class LBDBRegistry {
friend class LBMgrInit;
friend class LBManager;
private:
  // table for all available LBs linked in
  struct LBDBEntry {
    const char *name;
    LBCreateFn  cfn;
    LBAllocFn   afn;
    const char *help;
    int 	shown;		// if 0, donot show in help page
    LBDBEntry(): name(0), cfn(0), afn(0), help(0), shown(1) {}
    LBDBEntry(int) {}
    LBDBEntry(const char *n, LBCreateFn cf, LBAllocFn af,
              const char *h, int show=1):
      name(n), cfn(cf), afn(af), help(h), shown(show) {};
  };
  CkVec<LBDBEntry> lbtables;	 	// a list of available LBs linked
  CkVec<const char *>   compile_lbs;	// load balancers at compile time
  CkVec<const char *>   runtime_lbs;	// load balancers at run time
public:
  LBDBRegistry() {}
  void displayLBs()
  {
    CmiPrintf("\nAvailable load balancers:\n");
    for (int i=0; i<lbtables.length(); i++) {
      LBDBEntry &entry = lbtables[i];
      if (entry.shown) CmiPrintf("* %s:	%s\n", entry.name, entry.help);
    }
    CmiPrintf("\n");
  }
  void addEntry(const char *name, LBCreateFn fn, LBAllocFn afn, const char *help, int shown) {
    lbtables.push_back(LBDBEntry(name, fn, afn, help, shown));
  }
  void addCompiletimeBalancer(const char *name) {
    compile_lbs.push_back(name);
  }
  void addRuntimeBalancer(const char *name) {
    runtime_lbs.push_back(name);
  }
  LBCreateFn search(const char *name) {
    char *ptr = strpbrk((char *)name, ":,");
    int slen = ptr!=NULL?ptr-name:strlen(name);
    for (int i=0; i<lbtables.length(); i++)
      if (0==strncmp(name, lbtables[i].name, slen)) return lbtables[i].cfn;
    return NULL;
  }
  LBAllocFn getLBAllocFn(const char *name) {
    char *ptr = strpbrk((char *)name, ":,");
    int slen = ptr-name;
    for (int i=0; i<lbtables.length(); i++)
      if (0==strncmp(name, lbtables[i].name, slen)) return lbtables[i].afn;
    return NULL;
  }
};

static LBDBRegistry lbRegistry;
static std::vector<const char*> lbNames;

void LBDefaultCreate(const char *lbname)
{
  lbRegistry.addCompiletimeBalancer(lbname);
}

// default is to show the helper
void LBRegisterBalancer(const char *name, LBCreateFn fn, LBAllocFn afn, const char *help, int shown)
{
  lbRegistry.addEntry(name, fn, afn, help, shown);
}

LBAllocFn getLBAllocFn(const char *lbname) {
    return lbRegistry.getLBAllocFn(lbname);
}

// create a load balancer group using the strategy name
static void createLoadBalancer(const char *lbname)
{
    LBCreateFn fn = lbRegistry.search(lbname);
    if (!fn) {    // invalid lb name
      CmiPrintf("Abort: Unknown load balancer: '%s'!\n", lbname);
      lbRegistry.displayLBs();    // display help page
      CkAbort("Abort");
    }
    // invoke function to create load balancer
    fn();
}

// mainchare
LBMgrInit::LBMgrInit(CkArgMsg *m)
{
#if CMK_LBDB_ON
  _lbmgr = CProxy_LBManager::ckNew();

  // runtime specified load balancer
  if (lbRegistry.runtime_lbs.size() > 0) {
    for (int i=0; i<lbRegistry.runtime_lbs.size(); i++) {
      const char *balancer = lbRegistry.runtime_lbs[i];
      createLoadBalancer(balancer);
    }
  }
  else if (lbRegistry.compile_lbs.size() > 0) {
    for (int i=0; i<lbRegistry.compile_lbs.size(); i++) {
      const char* balancer = lbRegistry.compile_lbs[i];
      createLoadBalancer(balancer);
    }
  }

  // simulation mode
  if (LBSimulation::doSimulation) {
    CmiPrintf("Charm++> Entering Load Balancer Simulation Mode ... \n");
    CProxy_LBManager(_lbmgr).ckLocalBranch()->StartLB();
  }
#endif
  delete m;
}

// called from init.C
void _loadbalancerInit()
{
  CkpvInitialize(bool, lbmanagerInited);
  CkpvAccess(lbmanagerInited) = false;
  CkpvInitialize(int, numLoadBalancers);
  CkpvAccess(numLoadBalancers) = 0;

  CkpvInitialize(LBUserDataLayout, lbobjdatalayout);
  CkpvInitialize(int, _lb_obj_index);
  CkpvAccess(_lb_obj_index) = -1;

  char **argv = CkGetArgv();
  char *balancer = NULL;
  CmiArgGroup("Charm++","Load Balancer");

  // turn on MetaBalancer if set
  _lb_args.metaLbOn() = CmiGetArgFlagDesc(argv, "+MetaLB", "Turn on MetaBalancer");
  CmiGetArgStringDesc(argv, "+MetaLBModelDir", &_lb_args.metaLbModelDir(),
                      "Use this directory to read model for MetaLB");

  if (_lb_args.metaLbOn() && _lb_args.metaLbModelDir() != nullptr) {
#if CMK_USE_ZLIB
    if (CkMyRank() == 0) {
      lbRegistry.addRuntimeBalancer("TreeLB");
      lbNames.push_back("Greedy");
      lbNames.push_back("GreedyRefine");
      lbNames.push_back("DistributedLB");
      lbNames.push_back("Refine");
      lbNames.push_back("Hybrid");
      lbNames.push_back("MetisLB");
      if (CkMyPe() == 0) {
        if (CmiGetArgStringDesc(argv, "+balancer", &balancer, "Use this load balancer"))
          CkPrintf(
              "Warning: Ignoring the +balancer option, since Meta-Balancer's model-based "
              "load balancer selection is enabled.\n");
        CkPrintf(
            "Warning: Automatic strategy selection in MetaLB is activated. This is an "
            "experimental feature.\n");
      }
      while (CmiGetArgStringDesc(argv, "+balancer", &balancer, "Use this load balancer"))
        ;
    }
#else
    if (CkMyPe() == 0)
      CkAbort("MetaLB random forest model not supported because Charm++ was built without zlib support.\n");
#endif
  } else {
    if (CkMyPe() == 0 && _lb_args.metaLbOn())
      CkPrintf(
          "Warning: MetaLB is activated. For Automatic strategy selection in MetaLB, "
          "pass directory of model files using +MetaLBModelDir.\n");
    while (CmiGetArgStringDesc(argv, "+balancer", &balancer, "Use this load balancer")) {
      if (CkMyRank() == 0)
        lbRegistry.addRuntimeBalancer(balancer); /* lbRegistry is a static */
    }
  }

  CmiGetArgDoubleDesc(argv,"+DistLBTargetRatio", &_lb_args.targetRatio(),"The max/avg load ratio that DistributedLB will attempt to achieve");
  CmiGetArgIntDesc(argv,"+DistLBMaxPhases", &_lb_args.maxDistPhases(),"The maximum number of phases that DistributedLB will attempt");

  // set up init value for LBPeriod time in seconds
  // it can also be set by calling LDSetLBPeriod()
  CmiGetArgDoubleDesc(argv,"+LBPeriod", &_lb_args.lbperiod(),"the minimum time period in seconds allowed for two consecutive automatic load balancing");
  _lb_args.loop() = CmiGetArgFlagDesc(argv, "+LBLoop", "Use multiple load balancing strategies in loop");

  // now called in cldb.C: CldModuleGeneralInit()
  // registerLBTopos();
  CmiGetArgStringDesc(argv, "+LBTopo", &_lbtopo, "define load balancing topology");
  //Read the percentage parameter for RefineKLB and GreedyRefineLB
  CmiGetArgIntDesc(argv, "+LBPercentMoves", &_lb_args.percentMovesAllowed() , "Percentage of chares to be moved (used by RefineKLB and GreedyRefineLB) [0-100]");

  /**************** FUTURE PREDICTOR ****************/
  _lb_predict = CmiGetArgFlagDesc(argv, "+LBPredictor", "Turn on LB future predictor");
  CmiGetArgIntDesc(argv, "+LBPredictorDelay", &_lb_predict_delay, "Number of balance steps before learning a model");
  CmiGetArgIntDesc(argv, "+LBPredictorWindow", &_lb_predict_window, "Number of steps to use to learn a model");
  if (_lb_predict_window < _lb_predict_delay) {
    CmiPrintf("LB> [%d] Argument LBPredictorWindow (%d) less than LBPredictorDelay (%d) , fixing\n", CkMyPe(), _lb_predict_window, _lb_predict_delay);
    _lb_predict_delay = _lb_predict_window;
  }

  /******************* SIMULATION *******************/
  // get the step number at which to dump the LB database
  CmiGetArgIntDesc(argv, "+LBVersion", &_lb_args.lbversion(), "LB database file version number");
  CmiGetArgIntDesc(argv, "+LBCentPE", &_lb_args.central_pe(), "CentralLB processor");
  bool _lb_dump_activated = false;
  if (CmiGetArgIntDesc(argv, "+LBDump", &LBSimulation::dumpStep, "Dump the LB state from this step"))
    _lb_dump_activated = true;
  if (_lb_dump_activated && LBSimulation::dumpStep < 0) {
    CmiPrintf("LB> Argument LBDump (%d) negative, setting to 0\n",LBSimulation::dumpStep);
    LBSimulation::dumpStep = 0;
  }
  CmiGetArgIntDesc(argv, "+LBDumpSteps", &LBSimulation::dumpStepSize, "Dump the LB state for this amount of steps");
  if (LBSimulation::dumpStepSize <= 0) {
    CmiPrintf("LB> Argument LBDumpSteps (%d) too small, setting to 1\n",LBSimulation::dumpStepSize);
    LBSimulation::dumpStepSize = 1;
  }
  CmiGetArgStringDesc(argv, "+LBDumpFile", &LBSimulation::dumpFile, "Set the LB state file name");
  // get the simulation flag and number. Now the flag can also be avoided by the presence of the number
  LBSimulation::doSimulation = CmiGetArgIntDesc(argv, "+LBSim", &LBSimulation::simStep, "Read LB state from LBDumpFile since this step");
  // check for stupid LBSim parameter
  if (LBSimulation::doSimulation && LBSimulation::simStep < 0) {
    CkAbort("LB> Argument LBSim (%d) invalid, should be >= 0\n", LBSimulation::simStep);
    return;
  }
  CmiGetArgIntDesc(argv, "+LBSimSteps", &LBSimulation::simStepSize, "Read LB state for this number of steps");
  if (LBSimulation::simStepSize <= 0) {
    CmiPrintf("LB> Argument LBSimSteps (%d) too small, setting to 1\n",LBSimulation::simStepSize);
    LBSimulation::simStepSize = 1;
  }


  LBSimulation::simProcs = 0;
  CmiGetArgIntDesc(argv, "+LBSimProcs", &LBSimulation::simProcs, "Number of target processors.");

  LBSimulation::showDecisionsOnly =
    CmiGetArgFlagDesc(argv, "+LBShowDecisions",
		      "Write to File: Load Balancing Object to Processor Map decisions during LB Simulation");

  // force a global barrier after migration done
  _lb_args.syncResume() = CmiGetArgFlagDesc(argv, "+LBSyncResume",
                  "LB performs a barrier after migration is finished");

  // both +LBDebug and +LBDebug level should work
  if (!CmiGetArgIntDesc(argv, "+LBDebug", &_lb_args.debug(),
                                          "Turn on LB debugging printouts"))
    _lb_args.debug() = CmiGetArgFlagDesc(argv, "+LBDebug",
					     "Turn on LB debugging printouts");

  // getting the size of the team with +teamSize
  if (!CmiGetArgIntDesc(argv, "+teamSize", &_lb_args.teamSize(),
                                          "Team size"))
    _lb_args.teamSize() = 1;

  // ask to print summary/quality of load balancer
  _lb_args.printSummary() = CmiGetArgFlagDesc(argv, "+LBPrintSummary",
		"Print load balancing result summary");

  // to ignore baclground load
  _lb_args.ignoreBgLoad() = CmiGetArgFlagDesc(argv, "+LBNoBackground",
                      "Load balancer ignores the background load.");
#ifdef __BIGSIM__
  _lb_args.ignoreBgLoad() = 1;
#endif
  _lb_args.migObjOnly() = CmiGetArgFlagDesc(argv, "+LBObjOnly",
                      "Only load balancing migratable objects, ignoring all others.");
  if (_lb_args.migObjOnly()) _lb_args.ignoreBgLoad() = 1;

  // assume all CPUs are identical
  _lb_args.testPeSpeed() = CmiGetArgFlagDesc(argv, "+LBTestPESpeed",
                      "Load balancer test all CPUs speed.");
  _lb_args.samePeSpeed() = CmiGetArgFlagDesc(argv, "+LBSameCpus",
                      "Load balancer assumes all CPUs are of same speed.");
  if (!_lb_args.testPeSpeed()) _lb_args.samePeSpeed() = 1;

  _lb_args.useCpuTime() = CmiGetArgFlagDesc(argv, "+LBUseCpuTime",
                      "Load balancer uses CPU time instead of wallclock time.");

  // turn instrumentation off at startup
  _lb_args.statsOn() = !CmiGetArgFlagDesc(argv, "+LBOff",
			"Turn load balancer instrumentation off");

  // turn instrumentation of communicatin off at startup
  _lb_args.traceComm() = !CmiGetArgFlagDesc(argv, "+LBCommOff",
		"Turn load balancer instrumentation of communication off");

  // set alpha and beta
  _lb_args.alpha() = PER_MESSAGE_SEND_OVERHEAD_DEFAULT;
  _lb_args.beta() = PER_BYTE_SEND_OVERHEAD_DEFAULT;
  CmiGetArgDoubleDesc(argv,"+LBAlpha", &_lb_args.alpha(),
                           "per message send overhead");
  CmiGetArgDoubleDesc(argv,"+LBBeta", &_lb_args.beta(),
                           "per byte send overhead");

  if (CkMyPe() == 0) {
    if (_lb_args.debug()) {
      CmiPrintf("CharmLB> Verbose level %d, load balancing period: %g seconds\n", _lb_args.debug(), _lb_args.lbperiod());
    }
    if (_lb_args.debug() > 1) {
      CmiPrintf("CharmLB> Topology %s alpha: %es beta: %es.\n", _lbtopo, _lb_args.alpha(), _lb_args.beta());
    }
    if (_lb_args.printSummary())
      CmiPrintf("CharmLB> Load balancer print summary of load balancing result.\n");
    if (_lb_args.ignoreBgLoad())
      CmiPrintf("CharmLB> Load balancer ignores processor background load.\n");
    if (_lb_args.samePeSpeed())
      CmiPrintf("CharmLB> Load balancer assumes all CPUs are same.\n");
    if (_lb_args.useCpuTime())
      CmiPrintf("CharmLB> Load balancer uses CPU time instead of wallclock time.\n");
    if (LBSimulation::doSimulation)
      CmiPrintf("CharmLB> Load balancer running in simulation mode on file '%s' version %d.\n", LBSimulation::dumpFile, _lb_args.lbversion());
    if (_lb_args.statsOn()==0)
      CkPrintf("CharmLB> Load balancing instrumentation is off.\n");
    if (_lb_args.traceComm()==0)
      CkPrintf("CharmLB> Load balancing instrumentation for communication is off.\n");
    if (_lb_args.migObjOnly())
      CkPrintf("LB> Load balancing strategy ignores non-migratable objects.\n");
  }
}

bool LBManager::manualOn = false;
char *LBManager::avail_vector = NULL;
bool LBManager::avail_vector_set = false;
CmiNodeLock avail_vector_lock;

static LBRealType * _expectedLoad = NULL;

void LBManager::initnodeFn()
{
  int proc;
  int num_proc = CkNumPes();
  avail_vector= new char[num_proc];
  for(proc = 0; proc < num_proc; proc++)
      avail_vector[proc] = 1;
  avail_vector_lock = CmiCreateLock();

  _expectedLoad = new LBRealType[num_proc];
  for (proc=0; proc<num_proc; proc++) _expectedLoad[proc]=0.0;

  _registerCommandLineOpt("+balancer");
  _registerCommandLineOpt("+LBPeriod");
  _registerCommandLineOpt("+LBLoop");
  _registerCommandLineOpt("+LBTopo");
  _registerCommandLineOpt("+LBPercentMoves");
  _registerCommandLineOpt("+LBPredictor");
  _registerCommandLineOpt("+LBPredictorDelay");
  _registerCommandLineOpt("+LBPredictorWindow");
  _registerCommandLineOpt("+LBVersion");
  _registerCommandLineOpt("+LBCentPE");
  _registerCommandLineOpt("+LBDump");
  _registerCommandLineOpt("+LBDumpSteps");
  _registerCommandLineOpt("+LBDumpFile");
  _registerCommandLineOpt("+LBSim");
  _registerCommandLineOpt("+LBSimSteps");
  _registerCommandLineOpt("+LBSimProcs");
  _registerCommandLineOpt("+LBShowDecisions");
  _registerCommandLineOpt("+LBSyncResume");
  _registerCommandLineOpt("+LBDebug");
  _registerCommandLineOpt("+teamSize");
  _registerCommandLineOpt("+LBPrintSummary");
  _registerCommandLineOpt("+LBNoBackground");
  _registerCommandLineOpt("+LBObjOnly");
  _registerCommandLineOpt("+LBTestPESpeed");
  _registerCommandLineOpt("+LBSameCpus");
  _registerCommandLineOpt("+LBUseCpuTime");
  _registerCommandLineOpt("+LBOff");
  _registerCommandLineOpt("+LBCommOff");
  _registerCommandLineOpt("+MetaLB");
  _registerCommandLineOpt("+LBAlpha");
  _registerCommandLineOpt("+LBBeta");
}

void LBManager::callAt(){
  for (std::list<Chare *>::iterator i = chares.begin(); i != chares.end(); ++i)
    if((*i)->atsync_notify)
      (*i)->AtSyncBarrierReached();
  if(nloadbalancers > 0) loadbalancers[0]->InvokeLB();
}

//Called at end of each load balancing cycle
void LBManager::periodicLB(void *in)
{
  ((LBManager*)in)->callAt();
}

void LBManager::setTimer()
{
  CcdCallFnAfterOnPE((CcdVoidFn)periodicLB, (void *)this, 1000*_lb_args.lbperiod(), CkMyPe());
}

void LBManager::reset() {
  if(rank0pe){
    int peFirst = CkNodeFirst(CkMyNode());
    local_pes_to_notify.clear();
    for(int pe=peFirst;pe<peFirst+CkNodeSize(CkMyNode());pe++)
      local_pes_to_notify.insert(local_pes_to_notify.end(),pe);
    received_from_left = false;
    received_from_right = false;
  } else
    received_from_rank0 = false;
}

// called my constructor
void LBManager::init(void)
{
  mystep = 0;
  nloadbalancers = 0;
  new_ld_balancer = 0;
  chare_count = 0;
  metabalancer = nullptr;
  lbdb_obj = new LBDatabase();
  localBarrier.SetMgr(this);
  startedAtSync = false;
  rank0pe = CkMyRank() == 0;
  reset();

#if CMK_LB_CPUTIMER
  obj_cputime = 0;
#endif
  useBarrier = true;
  predictCBFn = nullptr;
  startLBFn_count = 0;

  CkpvAccess(lbmanagerInited) = true;
#if CMK_LBDB_ON
  if (manualOn) TurnManualLBOn();
#endif

  if(_lb_args.lbperiod() != -1.0) // check if user set LBPeriod - fix later
    setTimer();
}

int LBManager::AddStartLBFn(LDStartLBFn fn, void* data)
{
  // Save startLB function
  StartLBCB* callbk = new StartLBCB;

  callbk->fn = fn;
  callbk->data = data;
  callbk->on = 1;
  startLBFnList.push_back(callbk);
  startLBFn_count++;
  return startLBFnList.size()-1;
}

void LBManager::RemoveStartLBFn(LDStartLBFn fn)
{
  for (int i = 0; i < startLBFnList.size(); i++) {
    StartLBCB* callbk = startLBFnList[i];
    if (callbk && callbk->fn == fn) {
      delete callbk;
      startLBFnList[i] = 0;
      startLBFn_count--;
      break;
    }
  }
}

void LBManager::StartLB()
{
  if (startLBFn_count == 0) {
    CmiAbort("StartLB is not supported in this LB");
  }
  for (int i = 0; i < startLBFnList.size(); i++) {
    StartLBCB *startLBFn = startLBFnList[i];
    if (startLBFn && startLBFn->on) startLBFn->fn(startLBFn->data);
  }
}

int LBManager::AddMigrationDoneFn(LDMigrationDoneFn fn, void* data) {
  // Save migrationDone callback function
  MigrationDoneCB* callbk = new MigrationDoneCB;

  callbk->fn = fn;
  callbk->data = data;
  migrationDoneCBList.push_back(callbk);
  return migrationDoneCBList.size()-1;
}

void LBManager::RemoveMigrationDoneFn(LDMigrationDoneFn fn) {
  for (int i = 0; i < migrationDoneCBList.size(); i++) {
    MigrationDoneCB* callbk = migrationDoneCBList[i];
    if (callbk && callbk->fn == fn) {
      delete callbk;
      migrationDoneCBList[i] = 0;
      break;
    }
  }
}

void LBManager::MigrationDone() {
  for (int i = 0; i < migrationDoneCBList.size(); i++) {
    MigrationDoneCB *callbk = migrationDoneCBList[i];
    if (callbk) callbk->fn(callbk->data);
  }
}

void LBManager::SetupPredictor(LDPredictModelFn on, LDPredictWindowFn onWin, LDPredictFn off, LDPredictModelFn change, void* data)
{
  if (predictCBFn == NULL) predictCBFn = new PredictCB;
  predictCBFn->on = on;
  predictCBFn->onWin = onWin;
  predictCBFn->off = off;
  predictCBFn->change = change;
  predictCBFn->data = data;
}

void LBManager::DumpDatabase()
{
#ifdef DEBUG
  CmiPrintf("Database contains %d object managers\n",omCount);
  CmiPrintf("Database contains %d objects\n",objs.size());
#endif
}

void LBManager::Migrated(LDObjHandle h, int waitBarrier)
{
  // Object migrated, inform load balancers
  if(nloadbalancers > 0) loadbalancers[0]->Migrated(waitBarrier);
}

LBManager::LastLBInfo::LastLBInfo()
{
  expectedLoad = _expectedLoad;
}

void LBManager::get_avail_vector(char * bitmap) {
    CmiAssert(bitmap && avail_vector);
    const int num_proc = CkNumPes();
    for(int proc = 0; proc < num_proc; proc++){
      bitmap[proc] = avail_vector[proc];
    }
}

// new_ld == -1(default) : calcualte a new ld
//           -2 : ignore new ld
//           >=0: given a new ld
void LBManager::set_avail_vector(char * bitmap, int new_ld){
    int assigned = 0;
    const int num_proc = CkNumPes();
    if (new_ld == -2) assigned = 1;
    else if (new_ld >= 0) {
      CmiAssert(new_ld < num_proc);
      new_ld_balancer = new_ld;
      assigned = 1;
    }
    CmiAssert(bitmap && avail_vector);
    for(int count = 0; count < num_proc; count++){
        avail_vector[count] = bitmap[count];
        if((bitmap[count] == 1) && !assigned){
            new_ld_balancer = count;
            assigned = 1;
        }
    }
}

// called in CreateFooLB() when multiple load balancers are created
// on PE0, BaseLB of each load balancer applies a ticket number
// and broadcast the ticket number to all processors
int LBManager::getLoadbalancerTicket()  {
  int seq = nloadbalancers;
  nloadbalancers ++;
  loadbalancers.resize(nloadbalancers);
  loadbalancers[seq] = NULL;
  return seq;
}

void LBManager::addLoadbalancer(BaseLB *lb, int seq) {
//  CmiPrintf("[%d] addLoadbalancer for seq %d\n", CkMyPe(), seq);
  if (seq == -1) return;
  if (CkMyPe() == 0) {
    CmiAssert(seq < nloadbalancers);
    if (loadbalancers[seq]) {
      CmiPrintf("Duplicate load balancer created at %d\n", seq);
      CmiAbort("LBManager");
    }
  }
  else
    nloadbalancers ++;
  loadbalancers.resize(seq+1);
  loadbalancers[seq] = lb;
}

// switch strategy in order
void LBManager::nextLoadbalancer(int seq) {
  if (seq == -1) return;		// -1 means this is the only LB
  int next = seq+1;
  if (_lb_args.loop()) {
    if (next == nloadbalancers) next = 0;
  }
  else {
    if (next == nloadbalancers) next --;  // keep using the last one
  }
  if (seq != next) {
    loadbalancers[seq]->turnOff();
    CmiAssert(loadbalancers[next]);
    loadbalancers[next]->turnOn();
  }
}

// switch strategy
void LBManager::switchLoadbalancer(int switchFrom, int switchTo) {
    //TODO: Implement turn off / on
}

// return the seq-th load balancer string name of
// it can be specified in either compile time or runtime
// runtime has higher priority
const char *LBManager::loadbalancer(int seq) {
  if (lbRegistry.runtime_lbs.length()) {
    CmiAssert(seq < lbRegistry.runtime_lbs.length());
    return lbRegistry.runtime_lbs[seq];
  }
  else {
    CmiAssert(seq < lbRegistry.compile_lbs.length());
    return lbRegistry.compile_lbs[seq];
  }
}

void LBManager::pup(PUP::er& p)
{
  IrrGroup::pup(p);
  // the memory should be already allocated
  int np;
  if (!p.isUnpacking()) np = CkNumPes();
  p|np;
  // in case number of processors changes
  if (p.isUnpacking()) {
    CmiLock(avail_vector_lock);
    if(!avail_vector_set){
      avail_vector_set = true;
      CmiAssert(avail_vector);
      if(np>CkNumPes()){
        delete [] avail_vector;
        avail_vector = new char[np];
        for (int i=0; i<np; i++) avail_vector[i] = 1;
      }
      p(avail_vector, np);
    } else{
      char * tmp_avail_vector = new char[np];
      p(tmp_avail_vector, np);
      delete [] tmp_avail_vector;
    }
    CmiUnlock(avail_vector_lock);
  } else{
    CmiAssert(avail_vector);
    p(avail_vector, np);
  }
  p|mystep;
  if(p.isUnpacking()) {
    nloadbalancers = 0;
    if (_lb_args.metaLbOn()) {
      // if unpacking set metabalancer using the id
      metabalancer = (MetaBalancer*)CkLocalBranch(_metalb);
    }
  }
}

void LBManager::ResetAdaptive() {
#if CMK_LBDB_ON
	if (_lb_args.metaLbOn()) {
		if (metabalancer == NULL) {
			metabalancer = CProxy_MetaBalancer(_metalb).ckLocalBranch();
		}
		if (metabalancer != NULL) {
			metabalancer->ResetAdaptive();
		}
	}
#endif
}

void LBManager::ResumeClients() {
#if CMK_LBDB_ON
	if (_lb_args.metaLbOn()) {
		if (metabalancer == NULL) {
			metabalancer = CProxy_MetaBalancer(_metalb).ckLocalBranch();
		}
		if (metabalancer != NULL) {
			metabalancer->ResumeClients();
		}
	}
  localBarrier.ResumeClients();
#endif
}

void LBManager::SetMigrationCost(double cost) {
#if CMK_LBDB_ON
	if (_lb_args.metaLbOn()) {
		if (metabalancer == NULL) {
			metabalancer = (MetaBalancer *)CkLocalBranch(_metalb);
		}
		if (metabalancer != NULL)  {
			metabalancer->SetMigrationCost(cost);
		}
	}
#endif
}

void LBManager::SetStrategyCost(double cost) {
#if CMK_LBDB_ON
	if (_lb_args.metaLbOn()) {
		if (metabalancer == NULL) {
			metabalancer = (MetaBalancer *)CkLocalBranch(_metalb);
		}
		if (metabalancer != NULL)  {
			metabalancer->SetStrategyCost(cost);
		}
	}
#endif
}

void LBManager::UpdateDataAfterLB(double mLoad, double mCpuLoad, double avgLoad) {
#if CMK_LBDB_ON
	if (_lb_args.metaLbOn()) {
		if (metabalancer == NULL) {
			metabalancer = (MetaBalancer *)CkLocalBranch(_metalb);
		}
		if (metabalancer != NULL)  {
			metabalancer->UpdateAfterLBData(mLoad, mCpuLoad, avgLoad);
		}
	}
#endif
}

#if CMK_LBDB_ON
static void work(int iter_block, volatile int* result) {
  int i;
  *result = 1;
  for(i=0; i < iter_block; i++) {
    double b=0.1 + 0.1 * *result;
    *result=(int)(sqrt(1+cos(b * 1.57)));
  }
}

int LDProcessorSpeed()
{
  // for SMP version, if one processor have done this testing,
  // we can skip the other processors by remember the number here
  static int thisProcessorSpeed = -1;

  if (_lb_args.samePeSpeed() || CkNumPes() == 1)  // I think it is safe to assume that we can
    return 1;            // skip this if we are only using 1 PE

  if (thisProcessorSpeed != -1) return thisProcessorSpeed;

  //if (CkMyPe()==0) CkPrintf("Measuring processor speeds...");

  volatile static int result=0;  // I don't care what this is, its just for
			// timing, so this is thread safe.
  int wps = 0;
  const double elapse = 0.4;
  // First, count how many iterations for .2 second.
  // Since we are doing lots of function calls, this will be rough
  const double end_time = CmiCpuTimer()+elapse;
  wps = 0;
  while(CmiCpuTimer() < end_time) {
    work(1000,&result);
    wps+=1000;
  }

  // Now we have a rough idea of how many iterations there are per
  // second, so just perform a few cycles of correction by
  // running for what we think is 1 second.  Then correct
  // the number of iterations per second to make it closer
  // to the correct value

  for(int i=0; i < 2; i++) {
    const double start_time = CmiCpuTimer();
    work(wps,&result);
    const double end_time = CmiCpuTimer();
    const double correction = elapse / (end_time-start_time);
    wps = (int)((double)wps * correction + 0.5);
  }

  // If necessary, do a check now
  //    const double start_time3 = CmiWallTimer();
  //    work(msec * 1e-3 * wps);
  //    const double end_time3 = CmiWallTimer();
  //    CkPrintf("[%d] Work block size is %d %d %f\n",
  //	     thisIndex,wps,msec,1.e3*(end_time3-start_time3));
  thisProcessorSpeed = wps;

  //if (CkMyPe()==0) CkPrintf(" Done.\n");

  return wps;
}

#else
int LDProcessorSpeed() { return 1; }
#endif // CMK_LBDB_ON

int LBManager::ProcessorSpeed() {
  return LDProcessorSpeed();
}

/*
  callable from user's code
*/
void TurnManualLBOn()
{
#if CMK_LBDB_ON
   LBManager * myLbdb = LBManager::Object();
   if (myLbdb) {
     myLbdb->TurnManualLBOn();
   }
   else {
     LBManager::manualOn = true;
   }
#endif
}

void TurnManualLBOff()
{
#if CMK_LBDB_ON
   LBManager * myLbdb = LBManager::Object();
   if (myLbdb) {
     myLbdb->TurnManualLBOff();
   }
   else {
     LBManager::manualOn = true;
   }
#endif
}

extern "C" void LBTurnInstrumentOn() {
#if CMK_LBDB_ON
  if (CkpvAccess(lbmanagerInited))
    LBManager::Object()->CollectStatsOn();
  else
    _lb_args.statsOn() = 1;
#endif
}

extern "C" void LBTurnInstrumentOff() {
#if CMK_LBDB_ON
  if (CkpvAccess(lbmanagerInited))
    LBManager::Object()->CollectStatsOff();
  else
    _lb_args.statsOn() = 0;
#endif
}

extern "C" void LBTurnCommOn() {
#if CMK_LBDB_ON
  _lb_args.traceComm() = 1;
#endif
}

extern "C" void LBTurnCommOff() {
#if CMK_LBDB_ON
  _lb_args.traceComm() = 0;
#endif
}

void LBClearLoads() {
#if CMK_LBDB_ON
  LBManager::Object()->ClearLoads();
#endif
}

void LBTurnPredictorOn(LBPredictorFunction *model) {
#if CMK_LBDB_ON
  LBManager::Object()->PredictorOn(model);
#endif
}

void LBTurnPredictorOn(LBPredictorFunction *model, int wind) {
#if CMK_LBDB_ON
  LBManager::Object()->PredictorOn(model, wind);
#endif
}

void LBTurnPredictorOff() {
#if CMK_LBDB_ON
  LBManager::Object()->PredictorOff();
#endif
}

void LBChangePredictor(LBPredictorFunction *model) {
#if CMK_LBDB_ON
  LBManager::Object()->ChangePredictor(model);
#endif
}

void LBSetPeriod(double second) {
#if CMK_LBDB_ON
  if (CkpvAccess(lbmanagerInited))
    LBManager::Object()->SetLBPeriod(second);
  else
    _lb_args.lbperiod() = second;
#endif
}

int LBRegisterObjUserData(int size)
{
  return CkpvAccess(lbobjdatalayout).claim(size);
}


class client {
  friend class LocalBarrier;
  void* data;
  LDResumeFn fn;
  int refcount;
};
class receiver {
  friend class LocalBarrier;
  void* data;
  LDBarrierFn fn;
  int on;
};

void LBManager::AddClients(Chare* _new_chare, bool atsync, bool atsync_notify) {
  //Chare types: AtSync, reached-notify, resume-notify
  _new_chare->r_count = localBarrier.cur_refcount;
  _new_chare->atsync = atsync;
  _new_chare->atsync_notify = atsync_notify;
  if(_new_chare->atsync)
  localBarrier.client_count++;

  chares.insert(chares.end(), _new_chare);
}

void LBManager::RemoveClients(Chare* _new_chare) {
  if(_new_chare->atsync)
  localBarrier.client_count--;
  chares.remove(_new_chare);
}

LDBarrierReceiver LocalBarrier::AddReceiver(LDBarrierFn fn, void* data)
{
  receiver* new_receiver = new receiver;
  new_receiver->fn = fn;
  new_receiver->data = data;
  new_receiver->on = 1;

  return LDBarrierReceiver(receivers.insert(receivers.end(), new_receiver));
}

void LocalBarrier::RemoveReceiver(LDBarrierReceiver c)
{
  delete *(c.i);
  receivers.erase(c.i);
}

void LocalBarrier::TurnOnReceiver(LDBarrierReceiver c)
{
  (*c.i)->on = 1;
}

void LocalBarrier::TurnOffReceiver(LDBarrierReceiver c)
{
  (*c.i)->on = 0;
}

void LocalBarrier::AtBarrier(Chare* _n_c, int recvd_iter)
{
  _n_c->r_count++;
  at_count++;

  CheckBarrier(recvd_iter);
}

void LocalBarrier::DecreaseBarrier(int c)
{
  at_count-=c;
}

void LBManager::invoke_lb_start(int pe, int lb_step, int myphynode, int mype) {
  thisProxy[pe].recv_lb_start(lb_step, myphynode, mype);
}

void LocalBarrier::propagate_atsync() {
  if(propagated_atsync_step < cur_refcount) {
    int mype = CkMyPe();
    int myphynode = CkNodeOf(mype);
    if(!_mgr->rank0pe) {
      if(!_mgr->received_from_rank0) {
        int node_rank0_pe = CkNodeFirst(myphynode);
        _mgr->invoke_lb_start(node_rank0_pe, cur_refcount, myphynode, mype);
      }
    } else {
      int peFirst = CkNodeFirst(CkMyNode());
      for (std::list<int>::iterator it=_mgr->local_pes_to_notify.begin(); it != _mgr->local_pes_to_notify.end(); ++it)
        _mgr->invoke_lb_start(*it, cur_refcount, myphynode, mype);
      if(!_mgr->received_from_left && myphynode > 0) {
        int pe = CkNodeFirst(myphynode - 1);
        _mgr->invoke_lb_start(pe, cur_refcount, myphynode, mype);
      }
      if(!_mgr->received_from_right && myphynode < CkNumNodes()-1) {
        int pe = CkNodeFirst(myphynode + 1);
        _mgr->invoke_lb_start(pe, cur_refcount, myphynode, mype);
      }
    }
    propagated_atsync_step = cur_refcount;
  }
}

void LBManager::recv_lb_start(int lb_step, int phynode, int pe) {
  if(lb_step != localBarrier.cur_refcount || startedAtSync)
    return;
  int mype = CkMyPe();
  int myphynode = CkNodeOf(mype);
  if(phynode < myphynode)
    received_from_left = true;
  else if (phynode > myphynode)
    received_from_right = true;
  else if (rank0pe)
    local_pes_to_notify.remove(pe);
  else
    received_from_rank0 = true;
  if(localBarrier.client_count == 1) { //dummyAtSync is always a client
    localBarrier.CheckBarrier(1);
  }
}

void LocalBarrier::CheckBarrier(int recvd_iter)
{
  if(_lb_args.lbperiod() != -1.0) return;

  if (!on) return;

  if (client_count == 1 && recvd_iter < 0){
    return;
  }

  // If there are no clients, resume as soon as we're turned on
  //  CkPrintf("\n[PE-%d] at_count=%d, client_count=%d\n", CkMyPe(), at_count, client_count);
  if (client_count == 0) {
    cur_refcount++;
    for (std::list<Chare *>::iterator i = _mgr->chares.begin(); i != _mgr->chares.end(); ++i)
        if((*i)->atsync_notify)
        (*i)->AtSyncBarrierReached();
    return;
  }

  if (at_count >= client_count) {
    bool at_barrier = false;

    for(std::list<Chare*>::iterator i = _mgr->chares.begin(); i != _mgr->chares.end(); ++i)
      if ((*i)->r_count >= cur_refcount && (*i)->atsync)
        at_barrier = true;

    if (at_barrier) {
      _mgr->startedAtSync = true;
      propagate_atsync();
      at_count -= client_count;
      cur_refcount++;
      for (std::list<Chare *>::iterator i = _mgr->chares.begin(); i != _mgr->chares.end(); ++i)
        if((*i)->atsync_notify)
        (*i)->AtSyncBarrierReached();
      if(_mgr->nloadbalancers > 0) _mgr->loadbalancers[0]->InvokeLB();
    }
  }
}

void LocalBarrier::CallReceivers(void)
{
  bool called_receiver=false;

  for (std::list<receiver *>::iterator i = receivers.begin();
       i != receivers.end(); ++i) {
    receiver *recv = *i;
    if (recv->on) {
      recv->fn(recv->data);
      called_receiver = true;
    }
  }

  if (!called_receiver)
    ResumeClients();
}

void LocalBarrier::ResumeClients(void)
{

  if(_lb_args.lbperiod() != -1.0) {
    _mgr->setTimer();
    return;
  }

  for (std::list<Chare *>::iterator i = _mgr->chares.begin(); i != _mgr->chares.end(); ++i)
    if((*i)->atsync)
    (*i)->ResumeFromSync();

  _mgr->reset();
  _mgr->startedAtSync = false;
}

#include "LBManager.def.h"

/*@}*/
