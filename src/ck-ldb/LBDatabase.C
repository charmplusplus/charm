/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "converse.h"

/*
 * This C++ file contains the Charm stub functions
 */

#include "LBDatabase.h"
#include "LBSimulation.h"
#include "topology.h"

#include "NullLB.h"

CkGroupID _lbdb;

CkpvDeclare(LBUserDataLayout, lbobjdatalayout);
CkpvDeclare(int, _lb_obj_index);

CkpvDeclare(int, numLoadBalancers);  /**< num of lb created */
CkpvDeclare(int, hasNullLB);         /**< true if NullLB is created */
CkpvDeclare(int, lbdatabaseInited);  /**< true if lbdatabase is inited */

// command line options
CkLBArgs _lb_args;
int _lb_predict=0;
int _lb_predict_delay=10;
int _lb_predict_window=20;

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
  CkpvInitialize(int, lbdatabaseInited);
  CkpvAccess(lbdatabaseInited) = 0;
  CkpvInitialize(int, numLoadBalancers);
  CkpvAccess(numLoadBalancers) = 0;
  CkpvInitialize(int, hasNullLB);
  CkpvAccess(hasNullLB) = 0;

  CkpvInitialize(LBUserDataLayout, lbobjdatalayout);
  CkpvInitialize(int, _lb_obj_index);
  CkpvAccess(_lb_obj_index) = -1;

  char **argv = CkGetArgv();
  char *balancer = NULL;
  CmiArgGroup("Charm++","Load Balancer");
  while (CmiGetArgStringDesc(argv, "+balancer", &balancer, "Use this load balancer")) {
    if (CkMyRank() == 0)
      lbRegistry.addRuntimeBalancer(balancer);   /* lbRegistry is a static */
  }

  // set up init value for LBPeriod time in seconds
  // it can also be set by calling LDSetLBPeriod()
  CmiGetArgDoubleDesc(argv,"+LBPeriod", &_lb_args.lbperiod(),"the minimum time period in seconds allowed for two consecutive automatic load balancing");
  _lb_args.loop() = CmiGetArgFlagDesc(argv, "+LBLoop", "Use multiple load balancing strategies in loop");

  // now called in cldb.c: CldModuleGeneralInit()
  // registerLBTopos();
  CmiGetArgStringDesc(argv, "+LBTopo", &_lbtopo, "define load balancing topology");
  //Read the K parameter for RefineKLB
  CmiGetArgIntDesc(argv, "+LBNumMoves", &_lb_args.percentMovesAllowed() , "Percentage of chares to be moved (used by RefineKLB) [0-100]");

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
  int _lb_dump_activated = 0;
  if (CmiGetArgIntDesc(argv, "+LBDump", &LBSimulation::dumpStep, "Dump the LB state from this step"))
    _lb_dump_activated = 1;
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
    CmiPrintf("LB> Argument LBSim (%d) invalid, should be >= 0\n");
    CkExit();
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

	// turn on MetaBalancer if set
	_lb_args.metaLbOn() = CmiGetArgFlagDesc(argv, "+MetaLB",
		"Turn on MetaBalancer");

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

int LBDatabase::manualOn = 0;
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
  _registerCommandLineOpt("+LBNumMoves");
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

// called my constructor
void LBDatabase::init(void)
{
  myLDHandle = LDCreate();
  mystep = 0;
  nloadbalancers = 0;
  new_ld_balancer = 0;
	metabalancer = NULL;

  CkpvAccess(lbdatabaseInited) = 1;
#if CMK_LBDB_ON
  if (manualOn) TurnManualLBOn();
#endif
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
  LBDB *const db = (LBDB*)(_h.omhandle.ldb.handle);
  LBObj *const obj = db->LbObj(_h);

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
  LDResumeClients(myLDHandle);
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
     LBDatabase::manualOn = 1;
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
     LBDatabase::manualOn = 0;
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

#include "LBDatabase.def.h"

/*@}*/
