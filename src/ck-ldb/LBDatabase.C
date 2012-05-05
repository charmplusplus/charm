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

#include "limits.h"

#include "NullLB.h"

#define VEC_SIZE 500
#define IMB_TOLERANCE 1.1
#define OUTOFWAY_TOLERANCE 2
#define UTILIZATION_THRESHOLD 0.7
#define NEGLECT_IDLE 2 // Should never be == 1

#   define DEBAD(x) /*CkPrintf x*/
#   define EXTRA_FEATURE 0

struct AdaptiveData {
  double iteration;
  double max_load;
  double avg_load;
  double utilization;
  double idle_time;
};

struct AdaptiveLBDatabase {
  std::vector<AdaptiveData> history_data;
} adaptive_lbdb;

struct AdaptiveLBInfo {
  AdaptiveLBInfo() {
    max_avg_ratio = 1;
    remote_local_ratio = 1;
  }
  double max_avg_ratio;
  double remote_local_ratio;
};

// TODO: Separate out the datastructure required by just the central and on all
// processors
struct AdaptiveLBStructure {
  int tentative_period;
  int final_lb_period;
  // This is based on the linear extrapolation
  int lb_calculated_period;
  int lb_iteration_no;
  // This is set when all the processor sends the maximum iteration no
  int global_max_iter_no;
  // This keeps track of what was the max iteration no we had previously
  // received. TODO: Mostly global_max_iter_no should be sufficied.
  int tentative_max_iter_no;
  // TODO: Use reduction to collect max iteration. Then we don't need this
  // counter.
  int global_recv_iter_counter;
  // true indicates it is in Inform->ReceiveMaxIter->FinalLBPeriod stage.
  bool in_progress;
  double lb_strategy_cost;
  double lb_migration_cost;
  bool doCommStrategy;
  int lb_msg_send_no;
  int lb_msg_recv_no;
  // Total AtSync calls from all the chares residing on the processor
  int total_syncs_called;
  int last_lb_type;
  AdaptiveLBInfo greedy_info;
  AdaptiveLBInfo refine_info;
  AdaptiveLBInfo comm_info;
  AdaptiveLBInfo comm_refine_info;
} adaptive_struct;


CkReductionMsg* lbDataCollection(int nMsg, CkReductionMsg** msgs) {
  double lb_data[6];
  lb_data[1] = 0.0; // total number of processors contributing
  lb_data[2] = 0.0; // total load
  lb_data[3] = 0.0; // max load
  lb_data[4] = 0.0; // idle time
  lb_data[5] = 1.0; // utilization
  for (int i = 0; i < nMsg; i++) {
    CkAssert(msgs[i]->getSize() == 6*sizeof(double));
    if (msgs[i]->getSize() != 6*sizeof(double)) {
      CkPrintf("Error!!! Reduction not correct. Msg size is %d\n", msgs[i]->getSize());
    }
    double* m = (double *)msgs[i]->getData();
    // Total count
    lb_data[1] += m[1];
    // Avg load
    lb_data[2] += m[2];
    // Max load
    lb_data[3] = ((m[3] > lb_data[3])? m[3] : lb_data[3]);
    // Avg idle
    lb_data[4] += m[4];
    // Get least utilization
    lb_data[5] = ((m[5] < lb_data[5]) ? m[5] : lb_data[5]);
    if (i == 0) {
      // Iteration no
      lb_data[0] = m[0];
    }
    if (m[0] != lb_data[0]) {
      CkPrintf("Error!!! Reduction is intermingled between iteration %lf and\
      %lf\n", lb_data[0], m[0]);
    }
  }
  return CkReductionMsg::buildNew(6*sizeof(double), lb_data);
}

/*global*/ CkReduction::reducerType lbDataCollectionType;
/*initcall*/ void registerLBDataCollection(void) {
  lbDataCollectionType = CkReduction::addReducer(lbDataCollection);
}

CkGroupID _lbdb;

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

LBAllocFn getLBAllocFn(char *lbname) {
    return lbRegistry.getLBAllocFn(lbname);
}

LBCreateFn getLBCreateFn(const char *lbname) {
    return lbRegistry.search(lbname);
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

  // set alpha and beeta
  _lb_args.alpha() = PER_MESSAGE_SEND_OVERHEAD_DEFAULT;
  _lb_args.beeta() = PER_BYTE_SEND_OVERHEAD_DEFAULT;
  CmiGetArgDoubleDesc(argv,"+LBAlpha", &_lb_args.alpha(),
                           "per message send overhead");
  CmiGetArgDoubleDesc(argv,"+LBBeta", &_lb_args.beeta(),
                           "per byte send overhead");

  if (CkMyPe() == 0) {
    if (_lb_args.debug()) {
      CmiPrintf("CharmLB> Verbose level %d, load balancing period: %g seconds\n", _lb_args.debug(), _lb_args.lbperiod());
    }
    if (_lb_args.debug() > 1) {
      CmiPrintf("CharmLB> Topology %s alpha: %es beta: %es.\n", _lbtopo, _lb_args.alpha(), _lb_args.beeta());
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

  //CkPrintf("Total objs in %d is %d\n", CkMyPe(), getLBDB()->ObjDataCount());
}

// called my constructor
void LBDatabase::init(void)
{
  //thisProxy = CProxy_LBDatabase(thisgroup);
  myLDHandle = LDCreate();
  mystep = 0;
  nloadbalancers = 0;
  new_ld_balancer = 0;

  CkpvAccess(lbdatabaseInited) = 1;
#if CMK_LBDB_ON
  if (manualOn) TurnManualLBOn();
#endif

  total_load_vec.resize(VEC_SIZE, 0.0);
  total_count_vec.resize(VEC_SIZE, 0);
  max_iteration = -1;
  prev_idle = 0.0;
  alpha_beta_cost_to_load = 1.0; // Some random value. TODO: Find the actual

  // If metabalancer enabled, initialize the variables
  adaptive_struct.tentative_period =  INT_MAX;
  adaptive_struct.final_lb_period =  INT_MAX;
  adaptive_struct.lb_calculated_period = INT_MAX;
  adaptive_struct.lb_iteration_no = -1;
  adaptive_struct.global_max_iter_no = 0;
  adaptive_struct.tentative_max_iter_no = -1;
  adaptive_struct.global_recv_iter_counter = 0;
  adaptive_struct.in_progress = false;
  adaptive_struct.lb_strategy_cost = 0.0;
  adaptive_struct.lb_migration_cost = 0.0;
  adaptive_struct.lb_msg_send_no = 0;
  adaptive_struct.lb_msg_recv_no = 0;
  adaptive_struct.total_syncs_called = 0;
  adaptive_struct.last_lb_type = -1;

  // This is indicating if the load balancing strategy and migration started.
  // This is mainly used to register callbacks for noobj pes. They would
  // register as soon as resumefromsync is called. On receiving the handles at
  // the central pe, it clears the previous handlers and sets lb_in_progress
  // to false so that it doesn't clear the handles.
  lb_in_progress = false;

  is_prev_lb_refine = -1;
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
	CmiAssert(avail_vector);
	// in case number of processors changes
	if (p.isUnpacking() && np > CkNumPes()) {
		CmiLock(avail_vector_lock);
		delete [] avail_vector;
		avail_vector = new char[np];
		for (int i=0; i<np; i++) avail_vector[i] = 1;
		CmiUnlock(avail_vector_lock);
	}
	p(avail_vector, np);
	p|mystep;
	if(p.isUnpacking()) nloadbalancers = 0;
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

void LBDatabase::ResumeClients() {
  // If metabalancer enabled, initialize the variables
  adaptive_lbdb.history_data.clear();

  adaptive_struct.tentative_period =  INT_MAX;
  adaptive_struct.final_lb_period =  INT_MAX;
  adaptive_struct.lb_calculated_period = INT_MAX;
  adaptive_struct.lb_iteration_no = -1;
  adaptive_struct.global_max_iter_no = 0;
  adaptive_struct.tentative_max_iter_no = -1;
  adaptive_struct.global_recv_iter_counter = 0;
  adaptive_struct.in_progress = false;
  adaptive_struct.lb_strategy_cost = 0.0;
  adaptive_struct.lb_migration_cost = 0.0;
  adaptive_struct.lb_msg_send_no = 0;
  adaptive_struct.lb_msg_recv_no = 0;
  adaptive_struct.total_syncs_called = 0;

  total_load_vec.clear();
  total_count_vec.clear();
  prev_idle = 0.0;
  if (lb_in_progress) {
    lbdb_no_obj_callback.clear();
    lb_in_progress = false;
  }

  total_load_vec.resize(VEC_SIZE, 0.0);
  total_count_vec.resize(VEC_SIZE, 0);

  // While resuming client, if we find that there are no objects, then handle
  // the case accordingly.
  if (getLBDB()->ObjDataCount() == 0) {
    HandleAdaptiveNoObj();
  }
  LDResumeClients(myLDHandle);
}

bool LBDatabase::AddLoad(int iteration, double load) {
  total_count_vec[iteration]++;
  adaptive_struct.total_syncs_called++;
  DEBAD(("At PE %d Total contribution for iteration %d is %d total objs %d\n", CkMyPe(), iteration,
    total_count_vec[iteration], getLBDB()->ObjDataCount()));

  if (iteration > adaptive_struct.lb_iteration_no) {
    adaptive_struct.lb_iteration_no = iteration;
  }
  total_load_vec[iteration] += load;
  if (total_count_vec[iteration] == getLBDB()->ObjDataCount()) {
    double idle_time;
    IdleTime(&idle_time);

    if (iteration < NEGLECT_IDLE) {
      prev_idle = idle_time;
    }
    idle_time -= prev_idle;

    // The chares do not contribute their 0th iteration load. So the total syncs
    // in reality is total_syncs_called + obj_counts
    int total_countable_syncs = adaptive_struct.total_syncs_called +
        (1 - NEGLECT_IDLE) * getLBDB()->ObjDataCount(); // TODO: Fix me! weird!
    if (total_countable_syncs != 0) {
      idle_time = idle_time * getLBDB()->ObjDataCount() / total_countable_syncs;
    }
    //CkPrintf("[%d] Idle time %lf and countable %d for iteration %d\n", CkMyPe(), idle_time, total_countable_syncs, iteration);

    double lb_data[6];
    lb_data[0] = iteration;
    lb_data[1] = 1;
    lb_data[2] = total_load_vec[iteration]; // For average load
    lb_data[3] = total_load_vec[iteration]; // For max load
    lb_data[4] = idle_time;
    // Set utilization
    if (total_load_vec[iteration] == 0.0) {
      lb_data[5] = 0.0;
    } else {
      lb_data[5] = total_load_vec[iteration]/(idle_time + total_load_vec[iteration]);
    }

    //CkPrintf("   [%d] sends total load %lf idle time %lf ratio of idle/load %lf at iter %d\n", CkMyPe(),
    //    total_load_vec[iteration], idle_time,
    //    idle_time/total_load_vec[iteration], adaptive_struct.lb_iteration_no);

    CkCallback cb(CkIndex_LBDatabase::ReceiveMinStats((CkReductionMsg*)NULL), thisProxy[0]);
    contribute(6*sizeof(double), lb_data, lbDataCollectionType, cb);
  }
  return true;
}

void LBDatabase::ReceiveMinStats(CkReductionMsg *msg) {
  double* load = (double *) msg->getData();
  double avg = load[2]/load[1];
  double max = load[3];
  double avg_idle = load[4]/load[1];
  double utilization = load[5];
  int iteration_n = load[0];
  DEBAD(("** [%d] Iteration Avg load: %lf Max load: %lf Avg Idle : %lf Max Idle : %lf for %lf procs\n",iteration_n, avg, max, avg_idle, utilization, load[1]));
  CkPrintf("** [%d] Iteration Avg load: %lf Max load: %lf Avg Idle : %lf Max Idle : %lf for %lf procs\n",iteration_n, avg, max, avg_idle, utilization, load[1]);
  delete msg;

#if EXTRA_FEATURE
  if (adaptive_struct.final_lb_period != iteration_n) {
    for (int i = 0; i < lbdb_no_obj_callback.size(); i++) {
      thisProxy[lbdb_no_obj_callback[i]].TriggerAdaptiveReduction();
    }
  }
#endif

  // Store the data for this iteration
  adaptive_struct.lb_iteration_no = iteration_n;
  AdaptiveData data;
  data.iteration = adaptive_struct.lb_iteration_no;
  data.max_load = max;
  data.avg_load = avg;
  data.utilization = utilization;
  data.idle_time = avg_idle;
  adaptive_lbdb.history_data.push_back(data);

  // If lb period inform is in progress, dont inform again.
  // If this is the lb data corresponding to the final lb period informed, then
  // don't recalculate as some of the processors might have already gone into
  // LB_STAGE.
  if (adaptive_struct.in_progress || (adaptive_struct.final_lb_period == iteration_n)) {
    return;
  }

  double utilization_threshold = UTILIZATION_THRESHOLD;

#if EXTRA_FEATURE
  CkPrintf("alpha_beta_to_load %lf\n", alpha_beta_cost_to_load);
  if (alpha_beta_cost_to_load < 0.1) {
    // Ignore the effect of idle time and there by lesser utilization. So we
    // assign utilization threshold to be 0.0
    CkPrintf("Changing the idle load tolerance coz this isn't communication intensive benchmark\n");
    utilization_threshold = 0.0;
  }
#endif

  // First generate the lb period based on the cost of lb. Find out what is the
  // expected imbalance at the calculated lb period.
  int period;
  // This is based on the new max load after load balancing. So technically, it
  // is calculated based on the shifter up avg curve.
  double ratio_at_t = 1.0;
  int tmp_lb_type;
  double tmp_max_avg_ratio, tmp_comm_ratio;
  GetPrevLBData(tmp_lb_type, tmp_max_avg_ratio, tmp_comm_ratio);
  double tolerate_imb = IMB_TOLERANCE * tmp_max_avg_ratio;

  if (generatePlan(period, ratio_at_t)) {
    DEBAD(("Generated period and calculated %d and period %d max iter %d\n",
      adaptive_struct.lb_calculated_period, period,
      adaptive_struct.tentative_max_iter_no));
    // set the imbalance tolerance to be ratio_at_calculated_lb_period
    if (ratio_at_t != 1.0) {
      CkPrintf("Changed tolerance to %lf after line eq whereas max/avg is %lf\n", ratio_at_t, max/avg);
      // Since ratio_at_t is shifter up, max/(tmp_max_avg_ratio * avg) should be
      // compared with the tolerance
      tolerate_imb = ratio_at_t * tmp_max_avg_ratio * OUTOFWAY_TOLERANCE;
    }

    CkPrintf("Prev LB Data Type %d, max/avg %lf, local/remote %lf\n", tmp_lb_type, tmp_max_avg_ratio, tmp_comm_ratio);

    if ((utilization < utilization_threshold || max/avg >= tolerate_imb) && adaptive_lbdb.history_data.size() > 6) {
      CkPrintf("Trigger soon even though we calculated lbperiod max/avg(%lf) and utilization ratio (%lf)\n", max/avg, utilization);
      TriggerSoon(iteration_n, max/avg, tolerate_imb);
      return;
    }

    // If the new lb period from linear extrapolation is greater than maximum
    // iteration known from previously collected data, then inform all the
    // processors about the new calculated period.
    if (period > adaptive_struct.tentative_max_iter_no) {
      adaptive_struct.doCommStrategy = false;
      adaptive_struct.lb_calculated_period = period;
      adaptive_struct.in_progress = true;
      CkPrintf("Sticking to the calculated period %d\n",
        adaptive_struct.lb_calculated_period);
      thisProxy.LoadBalanceDecision(adaptive_struct.lb_msg_send_no++,
        adaptive_struct.lb_calculated_period);
      return;
    }
    // TODO: Shouldn't we return from here??
  }

  CkPrintf("Prev LB Data Type %d, max/avg %lf, local/remote %lf\n", tmp_lb_type, tmp_max_avg_ratio, tmp_comm_ratio);

  // This would be called when the datasize is not enough to calculate lb period
  if ((utilization < utilization_threshold || max/avg >= tolerate_imb) && adaptive_lbdb.history_data.size() > 4) {
    CkPrintf("Carry out load balancing step at iter max/avg(%lf) and utilization ratio (%lf)\n", max/avg, utilization);
    TriggerSoon(iteration_n, max/avg, tolerate_imb);
    return;
  }

}

void LBDatabase::TriggerSoon(int iteration_n, double imbalance_ratio,
    double tolerate_imb) {

  // If the previously calculated_period (not the final decision) is greater
  // than the iter +1 and if it is greater than the maximum iteration we have
  // seen so far, then we can inform this
  if ((iteration_n + 1 > adaptive_struct.tentative_max_iter_no) &&
      (iteration_n+1 < adaptive_struct.lb_calculated_period)) {
    if (imbalance_ratio < tolerate_imb) {
      adaptive_struct.doCommStrategy = true;
      CkPrintf("No load imbalance but idle time\n");
    } else {
      adaptive_struct.doCommStrategy = false;
      CkPrintf("load imbalance \n");
    }
    adaptive_struct.lb_calculated_period = iteration_n + 1;
    adaptive_struct.in_progress = true;
    CkPrintf("Informing everyone the lb period is %d\n",
        adaptive_struct.lb_calculated_period);
    thisProxy.LoadBalanceDecision(adaptive_struct.lb_msg_send_no++,
        adaptive_struct.lb_calculated_period);
  }
}

bool LBDatabase::generatePlan(int& period, double& ratio_at_t) {
  if (adaptive_lbdb.history_data.size() <= 4) {
    return false;
  }

  // Some heuristics for lbperiod
  // If constant load or almost constant,
  // then max * new_lb_period > avg * new_lb_period + lb_cost
  double max = 0.0;
  double avg = 0.0;
  AdaptiveData data;
  for (int i = 0; i < adaptive_lbdb.history_data.size(); i++) {
    data = adaptive_lbdb.history_data[i];
    max += data.max_load;
    avg += data.avg_load;
    //DEBAD(("max (%d, %lf) avg (%d, %lf)\n", i, data.max_load, i, data.avg_load));
    //CkPrintf("max (%d, %lf) avg (%d, %lf)\n", i, data.max_load, i, data.avg_load);
  }
//  max /= (adaptive_struct.lb_iteration_no - adaptive_lbdb.history_data[0].iteration);
//  avg /= (adaptive_struct.lb_iteration_no - adaptive_lbdb.history_data[0].iteration);

  // If linearly varying load, then find lb_period
  // area between the max and avg curve
  // If we can attain perfect balance, then the new load is close to the
  // average. Hence we pass 1, else pass in some other value which would be the
  // new max_load after load balancing.
  int tmp_lb_type;
  double tmp_max_avg_ratio, tmp_comm_ratio;
  double tolerate_imb;

#if EXTRA_FEATURE
  // First get the data for refine.
  GetLBDataForLB(1, tmp_max_avg_ratio, tmp_comm_ratio);
  tolerate_imb = tmp_max_avg_ratio;

  // If RefineLB does a good job, then find the period considering RefineLB
  if (tmp_max_avg_ratio <= 1.01) {
    if (max/avg < tolerate_imb) {
      CkPrintf("Resorting to imb = 1.0 coz max/avg (%lf) < imb(%lf)\n", max/avg, tolerate_imb);
      tolerate_imb = 1.0;
    }
    CkPrintf("Will generate plan for refine %lf imb and %lf overhead\n", tolerate_imb, 0.2);
    return getPeriodForStrategy(tolerate_imb, 0.2, period, ratio_at_t);
  }
#endif

  GetLBDataForLB(0, tmp_max_avg_ratio, tmp_comm_ratio);
  tolerate_imb = tmp_max_avg_ratio;
  if (max/avg < tolerate_imb) {
    CkPrintf("Resorting to imb = 1.0 coz max/avg (%lf) < imb(%lf)\n", max/avg, tolerate_imb);
    tolerate_imb = 1.0;
  }

  return getPeriodForStrategy(tolerate_imb, 1, period, ratio_at_t);
}

bool LBDatabase::getPeriodForStrategy(double new_load_percent,
    double overhead_percent, int& period, double& ratio_at_t) {
  double mslope, aslope, mc, ac;
  getLineEq(new_load_percent, aslope, ac, mslope, mc);
  CkPrintf("new load percent %lf\n", new_load_percent);
  CkPrintf("\n max: %fx + %f; avg: %fx + %f\n", mslope, mc, aslope, ac);
  double a = (mslope - aslope)/2;
  double b = (mc - ac);
  double c = -(adaptive_struct.lb_strategy_cost +
      adaptive_struct.lb_migration_cost) * overhead_percent;
  bool got_period = getPeriodForLinear(a, b, c, period);
  if (!got_period) {
    return false;
  }

  if (mslope < 0) {
    if (period > (-mc/mslope)) {
      CkPrintf("Max < 0 Period set when max load is -ve\n");
      return false;
    }
  }

  if (aslope < 0) {
    if (period > (-ac/aslope)) {
      CkPrintf("Avg < 0 Period set when avg load is -ve\n");
      return false;
    }
  }

  int intersection_t = (mc-ac) / (aslope - mslope);
  if (intersection_t > 0 && period > intersection_t) {
    CkPrintf("Avg | Max Period set when curves intersect\n");
    return false;
  }
  ratio_at_t = ((mslope*period + mc)/(aslope*period + ac));
  CkPrintf("Ratio at t (%lf*%d + %lf) / (%lf*%d+%lf) = %lf\n", mslope, period, mc, aslope, period, ac, ratio_at_t);
  return true;
}

bool LBDatabase::getPeriodForLinear(double a, double b, double c, int& period) {
  CkPrintf("Quadratic Equation %lf X^2 + %lf X + %lf\n", a, b, c);
  if (a == 0.0) {
    period = (-c / b);
    CkPrintf("Ideal period for linear load %d\n", period);
    return true;
  }
  int x;
  double t = (b * b) - (4*a*c);
  if (t < 0) {
    CkPrintf("(b * b) - (4*a*c) is -ve sqrt : %lf\n", sqrt(t));
    return false;
  }
  t = (-b + sqrt(t)) / (2*a);
  x = t;
  if (x < 0) {
    CkPrintf("boo!!! x (%d) < 0\n", x);
    x = 0;
    return false;
  }
  period = x;
  CkPrintf("Ideal period for linear load %d\n", period);
  return true;
}

bool LBDatabase::getLineEq(double new_load_percent, double& aslope, double& ac, double& mslope, double& mc) {
  int total = adaptive_lbdb.history_data.size();
  int iterations = 1 + adaptive_lbdb.history_data[total - 1].iteration -
      adaptive_lbdb.history_data[0].iteration;
  double a1 = 0;
  double m1 = 0;
  double a2 = 0;
  double m2 = 0;
  AdaptiveData data;
  int i = 0;
  for (i = 0; i < total/2; i++) {
    data = adaptive_lbdb.history_data[i];
    m1 += data.max_load;
    a1 += data.avg_load;
    CkPrintf("max (%d, %lf) avg (%d, %lf) adjusted_avg (%d, %lf)\n", i, data.max_load, i, data.avg_load, i, new_load_percent*data.avg_load);
  }
  m1 /= i;
  a1 = (a1 * new_load_percent) / i;

  for (i = total/2; i < total; i++) {
    data = adaptive_lbdb.history_data[i];
    m2 += data.max_load;
    a2 += data.avg_load;
    CkPrintf("max (%d, %lf) avg (%d, %lf) adjusted_avg (%d, %lf)\n", i, data.max_load, i, data.avg_load, i, new_load_percent*data.avg_load);
  }
  m2 /= (i - total/2);
  a2 = (a2 * new_load_percent) / (i - total/2);

  aslope = 2 * (a2 - a1) / iterations;
  mslope = 2 * (m2 - m1) / iterations;
  ac = adaptive_lbdb.history_data[0].avg_load * new_load_percent;
  mc = adaptive_lbdb.history_data[0].max_load;

  //ac = (adaptive_lbdb.history_data[1].avg_load * new_load_percent - aslope);
  //mc = (adaptive_lbdb.history_data[1].max_load - mslope);

  return true;
}

void LBDatabase::LoadBalanceDecision(int req_no, int period) {
  if (req_no < adaptive_struct.lb_msg_recv_no) {
    CkPrintf("Error!!! Received a request which was already sent or old\n");
    return;
  }
  //CkPrintf("[%d] Load balance decision made cur iteration: %d period:%d state: %d\n",CkMyPe(), adaptive_struct.lb_iteration_no, period, local_state);
  adaptive_struct.tentative_period = period;
  adaptive_struct.lb_msg_recv_no = req_no;
  thisProxy[0].ReceiveIterationNo(req_no, adaptive_struct.lb_iteration_no);
}

void LBDatabase::LoadBalanceDecisionFinal(int req_no, int period) {
  if (req_no < adaptive_struct.lb_msg_recv_no) {
    return;
  }
  DEBAD(("[%d] Final Load balance decision made cur iteration: %d period:%d \n",CkMyPe(), adaptive_struct.lb_iteration_no, period));
  adaptive_struct.tentative_period = period;
  adaptive_struct.final_lb_period = period;
  LDOMAdaptResumeSync(myLDHandle, period);
}


void LBDatabase::ReceiveIterationNo(int req_no, int local_iter_no) {
  CmiAssert(CkMyPe() == 0);

  adaptive_struct.global_recv_iter_counter++;
  if (local_iter_no > adaptive_struct.global_max_iter_no) {
    adaptive_struct.global_max_iter_no = local_iter_no;
  }

  int period;
  if (CkNumPes() == adaptive_struct.global_recv_iter_counter) {

    if (adaptive_struct.global_max_iter_no > adaptive_struct.tentative_max_iter_no) {
      adaptive_struct.tentative_max_iter_no = adaptive_struct.global_max_iter_no;
    }
    period = (adaptive_struct.tentative_period > adaptive_struct.global_max_iter_no) ? adaptive_struct.tentative_period : adaptive_struct.global_max_iter_no + 1;
    // If no one has gone into load balancing stage, then we can safely change
    // the period otherwise keep the old period.
    if (adaptive_struct.global_max_iter_no < adaptive_struct.final_lb_period) {
      adaptive_struct.tentative_period = period;
      CkPrintf("Final lb_period CHANGED!%d\n", adaptive_struct.tentative_period);
    } else {
      adaptive_struct.tentative_period = adaptive_struct.final_lb_period;
      CkPrintf("Final lb_period NOT CHANGED!%d\n", adaptive_struct.tentative_period);
    }
    thisProxy.LoadBalanceDecisionFinal(req_no, adaptive_struct.tentative_period);
    adaptive_struct.in_progress = false;
    adaptive_struct.global_recv_iter_counter = 0;
  }
}

int LBDatabase::getPredictedLBPeriod(bool& is_tentative) {
  // If tentative and final_lb_period are the same, then the decision has been
  // made but if not, they are in the middle of consensus, hence return the
  // lease of the two
  if (adaptive_struct.tentative_period < adaptive_struct.final_lb_period) {
    is_tentative = true;
    return adaptive_struct.tentative_period;
   } else {
     is_tentative = false;
     return adaptive_struct.final_lb_period;
   }
}

// Called by CentralLB to indicate that the LB strategy and migration is in
// progress.
void LBDatabase::ResetAdaptive() {
  adaptive_struct.lb_iteration_no = -1;
  lb_in_progress = true;
}

void LBDatabase::HandleAdaptiveNoObj() {
#if EXTRA_FEATURE
  adaptive_struct.lb_iteration_no++;
  //CkPrintf("HandleAdaptiveNoObj %d\n", adaptive_struct.lb_iteration_no);
  thisProxy[0].RegisterNoObjCallback(CkMyPe());
  TriggerAdaptiveReduction();
#endif
}

void LBDatabase::RegisterNoObjCallback(int index) {
#if EXTRA_FEATURE
  if (lb_in_progress) {
    lbdb_no_obj_callback.clear();
    //CkPrintf("Clearing and registering\n");
    lb_in_progress = false;
  }
  lbdb_no_obj_callback.push_back(index);
  CkPrintf("Registered %d to have no objs.\n", index);

  // If collection has already happened and this is second iteration, then
  // trigger reduction.
  if (adaptive_struct.lb_iteration_no != -1) {
    //CkPrintf("Collection already started now %d so kick in\n", adaptive_struct.lb_iteration_no);
    thisProxy[index].TriggerAdaptiveReduction();
  }
#endif
}

void LBDatabase::TriggerAdaptiveReduction() {
#if EXTRA_FEATURE
  adaptive_struct.lb_iteration_no++;
  //CkPrintf("Trigger adaptive for %d\n", adaptive_struct.lb_iteration_no);
  double lb_data[6];
  lb_data[0] = adaptive_struct.lb_iteration_no;
  lb_data[1] = 1;
  lb_data[2] = 0.0;
  lb_data[3] = 0.0;
  lb_data[4] = 0.0;
  lb_data[5] = 0.0;

  // CkPrintf("   [%d] sends total load %lf idle time %lf ratio of idle/load %lf at iter %d\n", CkMyPe(),
  //     total_load_vec[iteration], idle_time,
  //     idle_time/total_load_vec[iteration], adaptive_struct.lb_iteration_no);

  CkCallback cb(CkIndex_LBDatabase::ReceiveMinStats((CkReductionMsg*)NULL), thisProxy[0]);
  contribute(6*sizeof(double), lb_data, lbDataCollectionType, cb);
#endif
}


bool LBDatabase::isStrategyComm() {
  return adaptive_struct.doCommStrategy;
}

void LBDatabase::SetMigrationCost(double lb_migration_cost) {
  adaptive_struct.lb_migration_cost = lb_migration_cost;
}

void LBDatabase::SetStrategyCost(double lb_strategy_cost) {
  adaptive_struct.lb_strategy_cost = lb_strategy_cost;
}

void LBDatabase::UpdateAfterLBData(int lb, double lb_max, double lb_avg, double
    local_comm, double remote_comm) {
  adaptive_struct.last_lb_type = lb;
  if (lb == 0) {
    adaptive_struct.greedy_info.max_avg_ratio = lb_max/lb_avg;
  } else if (lb == 1) {
    adaptive_struct.refine_info.max_avg_ratio = lb_max/lb_avg;
  } else if (lb == 2) {
    adaptive_struct.comm_info.remote_local_ratio = remote_comm/local_comm;
  } else if (lb == 3) {
    adaptive_struct.comm_refine_info.remote_local_ratio =
    remote_comm/local_comm;
  }
}

void LBDatabase::UpdateAfterLBData(double max_load, double max_cpu, double
avg_load) {
  if (adaptive_struct.last_lb_type == -1) {
    adaptive_struct.last_lb_type = 0;
  }
  int lb = adaptive_struct.last_lb_type;
  //CkPrintf("Storing data after lb ratio %lf for lb %d\n", max_load/avg_load, lb);
  if (lb == 0) {
    adaptive_struct.greedy_info.max_avg_ratio = max_load/avg_load;
  } else if (lb == 1) {
    adaptive_struct.refine_info.max_avg_ratio = max_load/avg_load;
  } else if (lb == 2) {
    adaptive_struct.comm_info.max_avg_ratio = max_load/avg_load;
  } else if (lb == 3) {
    adaptive_struct.comm_refine_info.max_avg_ratio = max_load/avg_load;
  }
}

void LBDatabase::UpdateAfterLBComm(double alpha_beta_to_load) {
  CkPrintf("Setting alpha beta %lf\n", alpha_beta_to_load);
  alpha_beta_cost_to_load = alpha_beta_to_load;
}


void LBDatabase::GetPrevLBData(int& lb_type, double& lb_max_avg_ratio, double&
    remote_local_comm_ratio) {
  lb_type = adaptive_struct.last_lb_type;
  lb_max_avg_ratio = 1;
  remote_local_comm_ratio = 1;
  GetLBDataForLB(lb_type, lb_max_avg_ratio, remote_local_comm_ratio);
}

void LBDatabase::GetLBDataForLB(int lb_type, double& lb_max_avg_ratio, double&
    remote_local_comm_ratio) {
  if (lb_type == 0) {
    lb_max_avg_ratio = adaptive_struct.greedy_info.max_avg_ratio;
  } else if (lb_type == 1) {
    lb_max_avg_ratio = adaptive_struct.refine_info.max_avg_ratio;
  } else if (lb_type == 2) {
    remote_local_comm_ratio = adaptive_struct.comm_info.remote_local_ratio;
  } else if (lb_type == 3) {
    remote_local_comm_ratio =
       adaptive_struct.comm_refine_info.remote_local_ratio;
  }
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

#include "LBDatabase.def.h"

/*@}*/
