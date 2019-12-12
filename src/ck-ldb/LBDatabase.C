/**
 * \addtogroup CkLdb
*/
/*@{*/

#include <charm++.h>
#include "converse.h"

/*
 * This C++ file contains the Charm stub functions
 */

#include "LBDatabase.h"
#include "LBSimulation.h"
#include "topology.h"
#include "DistributedLB.h"

#include "NullLB.h"

CkGroupID _lbdb;

CkpvDeclare(LBUserDataLayout, lbobjdatalayout);
CkpvDeclare(int, _lb_obj_index);

CkpvDeclare(int, numLoadBalancers);  /**< num of lb created */
CkpvDeclare(bool, hasNullLB);         /**< true if NullLB is created */
CkpvDeclare(bool, lbdatabaseInited);  /**< true if lbdatabase is inited */

// command line options
CkLBArgs _lb_args;
int _lb_predict=0;
int _lb_predict_delay=10;
int _lb_predict_window=20;
bool _lb_psizer_on = false;

// registry class stores all load balancers linked and created at runtime
class LBDBRegistry {
friend class LBDBInit;
friend class LBDatabase;
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
LBDBInit::LBDBInit(CkArgMsg *m)
{
#if CMK_LBDB_ON
  _lbdb = CProxy_LBDatabase::ckNew();

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
  else {
    // NullLB is the default when none of above lb created
    // note user may create his own load balancer in his code manually like
    // in NAMD, but never mind NullLB can disable itself if there is
    // a non NULL LB.
    createLoadBalancer("NullLB");
  }

  // simulation mode
  if (LBSimulation::doSimulation) {
    CmiPrintf("Charm++> Entering Load Balancer Simulation Mode ... \n");
    CProxy_LBDatabase(_lbdb).ckLocalBranch()->StartLB();
  }
#endif
  delete m;
}

// called from init.C
void _loadbalancerInit()
{
  CkpvInitialize(bool, lbdatabaseInited);
  CkpvAccess(lbdatabaseInited) = false;
  CkpvInitialize(int, numLoadBalancers);
  CkpvAccess(numLoadBalancers) = 0;
  CkpvInitialize(bool, hasNullLB);
  CkpvAccess(hasNullLB) = false;

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
      lbRegistry.addRuntimeBalancer("GreedyLB");
      lbRegistry.addRuntimeBalancer("GreedyRefineLB");
      lbRegistry.addRuntimeBalancer("DistributedLB");
      lbRegistry.addRuntimeBalancer("RefineLB");
      lbRegistry.addRuntimeBalancer("HybridLB");
      lbRegistry.addRuntimeBalancer("MetisLB");
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

bool LBDatabase::manualOn = false;
char *LBDatabase::avail_vector = NULL;
bool LBDatabase::avail_vector_set = false;
CmiNodeLock avail_vector_lock;

static LBRealType * _expectedLoad = NULL;

void LBDatabase::initnodeFn()
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

/*************************************************************
 * Set up the builtin barrier-- the load balancer needs somebody
 * to call AtSync on each PE in case there are no atSync array
 * elements.  The builtin-atSync caller (batsyncer) does this.
 */

//Called periodically-- starts next load balancing cycle
void LBDatabase::batsyncer::gotoSync(void *bs)
{
  LBDatabase::batsyncer *s = (LBDatabase::batsyncer *)bs;
  s->gotoSyncCalled = true;
  s->db->AtLocalBarrier(s->BH);
}
//Called at end of each load balancing cycle
void LBDatabase::batsyncer::resumeFromSync(void *bs)
{
  LBDatabase::batsyncer *s = (LBDatabase::batsyncer *)bs;

#if 0
  double curT = CmiWallTimer();
  if (s->nextT<curT)  s->period *= 2;
  s->nextT = curT + s->period;
#endif

  if (s->gotoSyncCalled) {
    CcdCallFnAfterOnPE((CcdVoidFn)gotoSync, (void *)s, 1000*s->period, CkMyPe());
    s->gotoSyncCalled = false;
  }
}

// initPeriod in seconds
void LBDatabase::batsyncer::init(LBDatabase *_db, double initPeriod)
{
  db = _db;
  period = initPeriod;
  nextT = CmiWallTimer() + period;
  BH = db->AddLocalBarrierClient((LDResumeFn)resumeFromSync, (void*)(this));
  gotoSyncCalled = true;
  //This just does a CcdCallFnAfter
  resumeFromSync((void *)this);
}



// called by constructor
void LBDatabase::init(void)
{
  mystep = 0;
  nloadbalancers = 0;
  new_ld_balancer = 0;
  metabalancer = nullptr;
  objsEmptyHead = -1;
  omCount = omsRegistering = 0;

  obj_walltime = 0;
#if CMK_LB_CPUTIMER
  obj_cputime = 0;
#endif
  useBarrier = true;
  statsAreOn = false;
  obj_running = false;
  predictCBFn = nullptr;
  batsync.init(this, _lb_args.lbperiod());	    // original 1.0 second
  commTable = new LBCommTable;
  startLBFn_count = 0;

  CkpvAccess(lbdatabaseInited) = true;
#if CMK_LBDB_ON
  if (manualOn) TurnManualLBOn();
#endif
}

LDOMHandle LBDatabase::RegisterOM(LDOMid userID, void* userPtr, LDCallbacks cb) {
  LDOMHandle newHandle;
  newHandle.id = userID;

  LBOM* om = new LBOM(this, userID, userPtr, cb);
  if (om != nullptr) {
    newHandle.handle = oms.size();
    oms.push_back(om);
  } else newHandle.handle = -1;
  om->DepositHandle(newHandle);
  omCount++;
  return newHandle;
}

void LBDatabase::UnregisterOM(LDOMHandle omh) {
  delete oms[omh.handle];
  oms[omh.handle] = nullptr;
  omCount--;
}

void LBDatabase::RegisteringObjects(LDOMHandle omh) {
  // for an unregistered anonymous OM to join and control the barrier
  if (omh.id.id.idx == 0) {
    if (omsRegistering == 0)
      LocalBarrierOff();
    omsRegistering++;
  }
  else {
    LBOM* om = oms[omh.handle];
    if (!om->RegisteringObjs()) {
      if (omsRegistering == 0)
        LocalBarrierOff();
      omsRegistering++;
      om->SetRegisteringObjs(true);
    }
  }
}

void LBDatabase::DoneRegisteringObjects(LDOMHandle omh)
{
  // for an unregistered anonymous OM to join and control the barrier
  if (omh.id.id.idx == 0) {
    omsRegistering--;
    if (omsRegistering == 0)
      LocalBarrierOn();
  }
  else {
    LBOM* om = oms[omh.handle];
    if (om->RegisteringObjs()) {
      omsRegistering--;
      if (omsRegistering == 0)
        LocalBarrierOn();
      om->SetRegisteringObjs(false);
    }
  }
}

#if CMK_BIGSIM_CHARM
#define LBOBJ_OOC_IDX 0x1
#endif

LDObjHandle LBDatabase::RegisterObj(LDOMHandle omh, CmiUInt8 id,
                                    void* userPtr, int migratable) {
  LDObjHandle newhandle;

  newhandle.omhandle = omh;
  newhandle.id = id;

#if CMK_BIGSIM_CHARM
  if (_BgOutOfCoreFlag==2){ //taking object into memory
    //first find the first (LBOBJ_OOC_IDX) in objs and insert the object at that position
    int newpos = -1;
    for (int i = 0; i < objs.size(); i++) {
      if (objs[i].obj == (LBObj *)LBOBJ_OOC_IDX) {
        newpos = i;
        break;
      }
    }
    if (newpos == -1) newpos = objs.size();
    newhandle.handle = newpos;
    LBObj *obj = new LBObj(newhandle, userPtr, migratable);
    if (newpos == objs.size()) {
      objs.emplace_back(obj);
    } else {
      objs[newpos].obj = obj;
    }
    //objCount is not increased since it's the original object which is pupped
    //through out-of-core emulation.
    //objCount++;
  } else
#endif
  {
    // objsEmptyHead maintains a linked list of empty positions within the objs array
    // If objsEmptyHead == -1, there are no vacant positions, so add to the back
    // If objsEmptyHead > -1, we place the new object at index objsEmptyHead and advance
    // objsEmptyHead to the next empty position.
    if (objsEmptyHead == -1) {
      newhandle.handle = objs.size();
      LBObj *obj = new LBObj(newhandle, userPtr, migratable);
      objs.emplace_back(obj);
    } else {
      newhandle.handle = objsEmptyHead;
      LBObj *obj = new LBObj(newhandle, userPtr, migratable);
      objs[objsEmptyHead].obj = obj;

      objsEmptyHead = objs[objsEmptyHead].next;
    }
  }

  return newhandle;
}

void LBDatabase::UnregisterObj(LDObjHandle h)
{
  delete objs[h.handle].obj;

#if CMK_BIGSIM_CHARM
  //hack for BigSim out-of-core emulation.
  //we want the chare array object to keep at the same
  //position even going through the pupping routine.
  if (_BgOutOfCoreFlag == 1) { //in taking object out of memory
    objs[h.handle].obj = (LBObj *)(LBOBJ_OOC_IDX);
  } else
#endif
  {
    objs[h.handle].obj = NULL;
    // Maintain the linked list of empty positions by adding the newly removed
    // index as the new objsEmptyHead
    objs[h.handle].next = objsEmptyHead;
    objsEmptyHead = h.handle;
  }
}

void LBDatabase::Send(const LDOMHandle &destOM, const CmiUInt8 &destID, unsigned int bytes, int destObjProc, int force)
{
  if (force || (StatsOn() && _lb_args.traceComm())) {
    LBCommData* item_ptr;

    if (obj_running) {
      const LDObjHandle &runObj = RunningObj();

      // Don't record self-messages from an object to an object
      if (runObj.omhandle.id == destOM.id
          && runObj.id == destID )
        return;

      // In the future, we'll have to eliminate processor to same
      // processor messages as well

      LBCommData item(runObj, destOM.id, destID, destObjProc);
      item_ptr = commTable->HashInsertUnique(item);
    } else {
      LBCommData item(CkMyPe(), destOM.id, destID, destObjProc);
      item_ptr = commTable->HashInsertUnique(item);
    }
    item_ptr->addMessage(bytes);
  }
}

void LBDatabase::MulticastSend(const LDOMHandle &destOM, CmiUInt8 *destIDs, int nDests, unsigned int bytes, int nMsgs)
{
  if (StatsOn() && _lb_args.traceComm()) {
    LBCommData* item_ptr;
    if (obj_running) {
      const LDObjHandle &runObj = RunningObj();

      LBCommData item(runObj, destOM.id, destIDs, nDests);
      item_ptr = commTable->HashInsertUnique(item);
      item_ptr->addMessage(bytes, nMsgs);
    }
  }
}

void LBDatabase::DumpDatabase()
{
#ifdef DEBUG
  CmiPrintf("Database contains %d object managers\n",omCount);
  CmiPrintf("Database contains %d objects\n",objs.size());
#endif
}

int LBDatabase::NotifyMigrated(LDMigratedFn fn, void* data)
{
  // Save migration function
  MigrateCB* callbk = new MigrateCB;

  callbk->fn = fn;
  callbk->data = data;
  callbk->on = 1;
  migrateCBList.push_back(callbk);
  return migrateCBList.size()-1;
}

void LBDatabase::RemoveNotifyMigrated(int handle)
{
  MigrateCB* callbk = migrateCBList[handle];
  migrateCBList[handle] = NULL;
  delete callbk;
}

int LBDatabase::AddStartLBFn(LDStartLBFn fn, void* data)
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

void LBDatabase::RemoveStartLBFn(LDStartLBFn fn)
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

void LBDatabase::StartLB()
{
  if (startLBFn_count == 0) {
    CmiAbort("StartLB is not supported in this LB");
  }
  for (int i = 0; i < startLBFnList.size(); i++) {
    StartLBCB *startLBFn = startLBFnList[i];
    if (startLBFn && startLBFn->on) startLBFn->fn(startLBFn->data);
  }
}

int LBDatabase::AddMigrationDoneFn(LDMigrationDoneFn fn, void* data) {
  // Save migrationDone callback function
  MigrationDoneCB* callbk = new MigrationDoneCB;

  callbk->fn = fn;
  callbk->data = data;
  migrationDoneCBList.push_back(callbk);
  return migrationDoneCBList.size()-1;
}

void LBDatabase::RemoveMigrationDoneFn(LDMigrationDoneFn fn) {
  for (int i = 0; i < migrationDoneCBList.size(); i++) {
    MigrationDoneCB* callbk = migrationDoneCBList[i];
    if (callbk && callbk->fn == fn) {
      delete callbk;
      migrationDoneCBList[i] = 0;
      break;
    }
  }
}

void LBDatabase::MigrationDone() {
  for (int i = 0; i < migrationDoneCBList.size(); i++) {
    MigrationDoneCB *callbk = migrationDoneCBList[i];
    if (callbk) callbk->fn(callbk->data);
  }
}

void LBDatabase::SetupPredictor(LDPredictModelFn on, LDPredictWindowFn onWin, LDPredictFn off, LDPredictModelFn change, void* data)
{
  if (predictCBFn == NULL) predictCBFn = new PredictCB;
  predictCBFn->on = on;
  predictCBFn->onWin = onWin;
  predictCBFn->off = off;
  predictCBFn->change = change;
  predictCBFn->data = data;
}

int LBDatabase::GetObjDataSz()
{
  int nitems = 0;
  int i;
  if (_lb_args.migObjOnly()) {
  for(i = 0; i < objs.size(); i++)
    if (objs[i].obj && (objs[i].obj)->data.migratable)
      nitems++;
  } else {
  for(i = 0; i < objs.size(); i++)
    if (objs[i].obj)
      nitems++;
  }
  return nitems;
}

void LBDatabase::GetObjData(LDObjData *dp)
{
  if (_lb_args.migObjOnly()) {
    for (int i = 0; i < objs.size(); i++) {
      LBObj* obj = objs[i].obj;
      if (obj && obj->data.migratable)
        *dp++ = obj->ObjData();
    }
  } else {
    for (int i = 0; i < objs.size(); i++) {
      LBObj* obj = objs[i].obj;
      if (obj)
        *dp++ = obj->ObjData();
    }
  }
}

void LBDatabase::BackgroundLoad(LBRealType* walltime, LBRealType* cputime)
{
  LBRealType total_walltime;
  LBRealType total_cputime;
  TotalTime(&total_walltime, &total_cputime);

  LBRealType idletime;
  IdleTime(&idletime);

  *walltime = total_walltime - idletime - obj_walltime;
  if (*walltime < 0) *walltime = 0.;
#if CMK_LB_CPUTIMER
  *cputime = total_cputime - obj_cputime;
#else
  *cputime = *walltime;
#endif
}

void LBDatabase::GetTime(LBRealType *total_walltime, LBRealType *total_cputime,
                         LBRealType *idletime, LBRealType *bg_walltime,
                         LBRealType *bg_cputime)
{
  TotalTime(total_walltime,total_cputime);

  IdleTime(idletime);

  *bg_walltime = *total_walltime - *idletime - obj_walltime;
  if (*bg_walltime < 0) *bg_walltime = 0.;
#if CMK_LB_CPUTIMER
  *bg_cputime = *total_cputime - obj_cputime;
#else
  *bg_cputime = *bg_walltime;
#endif
  //CkPrintf("HERE [%d] total: %f %f obj: %f %f idle: %f bg: %f\n", CkMyPe(), *total_walltime, *total_cputime, obj_walltime, obj_cputime, *idletime, *bg_walltime);
}

void LBDatabase::ClearLoads(void)
{
  int i;
  for (i = 0; i < objs.size(); i++) {
    LBObj *obj = objs[i].obj;
    if (obj)
    {
      if (obj->data.wallTime > 0.0) {
        obj->lastWallTime = obj->data.wallTime;
#if CMK_LB_CPUTIMER
        obj->lastCpuTime = obj->data.cpuTime;
#endif
      }
      obj->data.wallTime = 0.0;
#if CMK_LB_CPUTIMER
      obj->data.cpuTime = 0.0;
#endif
    }
  }
  delete commTable;
  commTable = new LBCommTable;
  machineUtil.Clear();
  obj_walltime = 0;
#if CMK_LB_CPUTIMER
  obj_cputime = 0;
#endif
}

int LBDatabase::Migrate(LDObjHandle h, int dest)
{
  if (h.handle >= objs.size()) {
    CmiAbort("[%d] LBDB::Migrate: Handle %d out of range 0-%zu\n",CkMyPe(),h.handle,objs.size());
  }
  else if (!(objs[h.handle].obj)) {
    CmiAbort("[%d] LBDB::Migrate: Handle %d no longer registered, range 0-%zu\n", CkMyPe(),h.handle,objs.size());
  }

  LBOM *const om = oms[(objs[h.handle].obj)->parentOM().handle];
  om->Migrate(h, dest);
  return 1;
}

void LBDatabase::Migrated(LDObjHandle h, int waitBarrier)
{
  // Object migrated, inform load balancers

  // subtle: callback may change (on) when switching LBs
  // call in reverse order
  //for(int i=0; i < migrateCBList.length(); i++) {
  for(int i = migrateCBList.size()-1; i >= 0; i--) {
    MigrateCB* cb = (MigrateCB*)migrateCBList[i];
    if (cb && cb->on) (cb->fn)(cb->data, h, waitBarrier);
  }
}

void LBDatabase::MetaLBResumeWaitingChares(int lb_ideal_period) {
#if CMK_LBDB_ON
  for (int i = 0; i < objs.size(); i++) {
    LBObj* obj = objs[i].obj;
    if (obj) {
      LBOM *om = oms[obj->parentOM().handle];
      LDObjHandle h = obj->GetLDObjHandle();
      om->MetaLBResumeWaitingChares(h, lb_ideal_period);
    }
  }
#endif
}

void LBDatabase::MetaLBCallLBOnChares() {
#ifdef CMK_LBDB_ON
  for (int i = 0; i < objs.size(); i++) {
    LBObj* obj = objs[i].obj;
    if (obj) {
      LBOM *om = oms[obj->parentOM().handle];
      LDObjHandle h = obj->GetLDObjHandle();
      om->MetaLBCallLBOnChares(h);
    }
  }
#endif
}

int LBDatabase::useMem() {
  int size = sizeof(LBDatabase);
  size += oms.size() * sizeof(LBOM);
  size += GetObjDataSz() * sizeof(LBObj);
  size += migrateCBList.size() * sizeof(MigrateCB);
  size += startLBFnList.size() * sizeof(StartLBCB);
  size += commTable->useMem();
  return size;
}

LBDatabase::LastLBInfo::LastLBInfo()
{
  expectedLoad = _expectedLoad;
}

void LBDatabase::get_avail_vector(char * bitmap) {
    CmiAssert(bitmap && avail_vector);
    const int num_proc = CkNumPes();
    for(int proc = 0; proc < num_proc; proc++){
      bitmap[proc] = avail_vector[proc];
    }
}

// new_ld == -1(default) : calcualte a new ld
//           -2 : ignore new ld
//           >=0: given a new ld
void LBDatabase::set_avail_vector(char * bitmap, int new_ld){
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
int LBDatabase::getLoadbalancerTicket()  {
  int seq = nloadbalancers;
  nloadbalancers ++;
  loadbalancers.resize(nloadbalancers);
  loadbalancers[seq] = NULL;
  return seq;
}

void LBDatabase::addLoadbalancer(BaseLB *lb, int seq) {
//  CmiPrintf("[%d] addLoadbalancer for seq %d\n", CkMyPe(), seq);
  if (seq == -1) return;
  if (CkMyPe() == 0) {
    CmiAssert(seq < nloadbalancers);
    if (loadbalancers[seq]) {
      CmiPrintf("Duplicate load balancer created at %d\n", seq);
      CmiAbort("LBDatabase");
    }
  }
  else
    nloadbalancers ++;
  loadbalancers.resize(seq+1);
  loadbalancers[seq] = lb;
}

// switch strategy in order
void LBDatabase::nextLoadbalancer(int seq) {
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
void LBDatabase::switchLoadbalancer(int switchFrom, int switchTo) {
  if (switchTo != switchFrom) {
    if (switchFrom != -1) loadbalancers[switchFrom]->turnOff();
    CmiAssert(loadbalancers[switchTo]);
    loadbalancers[switchTo]->turnOn();
  }
}

// return the seq-th load balancer string name of
// it can be specified in either compile time or runtime
// runtime has higher priority
const char *LBDatabase::loadbalancer(int seq) {
  if (lbRegistry.runtime_lbs.length()) {
    CmiAssert(seq < lbRegistry.runtime_lbs.length());
    return lbRegistry.runtime_lbs[seq];
  }
  else {
    CmiAssert(seq < lbRegistry.compile_lbs.length());
    return lbRegistry.compile_lbs[seq];
  }
}

void LBDatabase::pup(PUP::er& p)
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


void LBDatabase::EstObjLoad(const LDObjHandle &_h, double cputime)
{
#if CMK_LBDB_ON
  LBObj *const obj = LbObj(_h);

  CmiAssert(obj != NULL);
  obj->setTiming(cputime);
#endif
}

void LBDatabase::ResetAdaptive() {
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

void LBDatabase::ResumeClients() {
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

void LBDatabase::SetMigrationCost(double cost) {
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

void LBDatabase::SetStrategyCost(double cost) {
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

void LBDatabase::UpdateDataAfterLB(double mLoad, double mCpuLoad, double avgLoad) {
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

int LBDatabase::ProcessorSpeed() {
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
int LBDatabase::ProcessorSpeed() { return 1; }
#endif // CMK_LBDB_ON

/*
  callable from user's code
*/
void TurnManualLBOn()
{
#if CMK_LBDB_ON
   LBDatabase * myLbdb = LBDatabase::Object();
   if (myLbdb) {
     myLbdb->TurnManualLBOn();
   }
   else {
     LBDatabase::manualOn = true;
   }
#endif
}

void TurnManualLBOff()
{
#if CMK_LBDB_ON
   LBDatabase * myLbdb = LBDatabase::Object();
   if (myLbdb) {
     myLbdb->TurnManualLBOff();
   }
   else {
     LBDatabase::manualOn = true;
   }
#endif
}

extern "C" void LBTurnInstrumentOn() {
#if CMK_LBDB_ON
  if (CkpvAccess(lbdatabaseInited))
    LBDatabase::Object()->CollectStatsOn();
  else
    _lb_args.statsOn() = 1;
#endif
}

extern "C" void LBTurnInstrumentOff() {
#if CMK_LBDB_ON
  if (CkpvAccess(lbdatabaseInited))
    LBDatabase::Object()->CollectStatsOff();
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
  LBDatabase::Object()->ClearLoads();
#endif
}

void LBTurnPredictorOn(LBPredictorFunction *model) {
#if CMK_LBDB_ON
  LBDatabase::Object()->PredictorOn(model);
#endif
}

void LBTurnPredictorOn(LBPredictorFunction *model, int wind) {
#if CMK_LBDB_ON
  LBDatabase::Object()->PredictorOn(model, wind);
#endif
}

void LBTurnPredictorOff() {
#if CMK_LBDB_ON
  LBDatabase::Object()->PredictorOff();
#endif
}

void LBChangePredictor(LBPredictorFunction *model) {
#if CMK_LBDB_ON
  LBDatabase::Object()->ChangePredictor(model);
#endif
}

void LBSetPeriod(double second) {
#if CMK_LBDB_ON
  if (CkpvAccess(lbdatabaseInited))
    LBDatabase::Object()->SetLBPeriod(second);
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

LDBarrierClient LocalBarrier::AddClient(LDResumeFn fn, void* data)
{
  client* new_client = new client;
  new_client->fn = fn;
  new_client->data = data;
  new_client->refcount = cur_refcount;

#if CMK_BIGSIM_CHARM
  if(_BgOutOfCoreFlag!=2){
    //during out-of-core emualtion for BigSim, if taking procs from disk to mem,
    //client_count should not be increased
    client_count++;
  }
#else
  client_count++;
#endif

  return LDBarrierClient(clients.insert(clients.end(), new_client));
}

void LocalBarrier::RemoveClient(LDBarrierClient c)
{
  delete *(c.i);
  clients.erase(c.i);

#if CMK_BIGSIM_CHARM
  //during out-of-core emulation for BigSim, if taking procs from mem to disk,
  //client_count should not be increased
  if(_BgOutOfCoreFlag!=1)
  {
    client_count--;
  }
#else
  client_count--;
#endif
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

void LocalBarrier::AtBarrier(LDBarrierClient h)
{
  (*h.i)->refcount++;
  at_count++;
  CheckBarrier();
}

void LocalBarrier::DecreaseBarrier(LDBarrierClient h, int c)
{
  at_count-=c;
}

void LocalBarrier::CheckBarrier()
{
  if (!on) return;

  // If there are no clients, resume as soon as we're turned on
  if (client_count == 0) {
    cur_refcount++;
    CallReceivers();
  }

  // If there have been enough AtBarrier calls, check to see if all clients have
  // made it to the barrier. It's possible to have gotten multiple AtSync calls
  // from a single client, which is why this check is necessary.
  if (at_count >= client_count) {
    bool at_barrier = true;

    for (auto& c : clients) {
      if (c->refcount < cur_refcount) {
        at_barrier = false;
        break;
      }
    }

    if (at_barrier) {
      at_count -= client_count;
      cur_refcount++;
      CallReceivers();
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
  for (std::list<client *>::iterator i = clients.begin(); i != clients.end(); ++i)
    (*i)->fn((*i)->data);
}

#include "LBDatabase.def.h"

/*@}*/
