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
#define IDLE_LOAD_TOLERANCE 0.3

struct AdaptiveData {
  int iteration;
  double max_load;
  double avg_load;
  double max_idle_load_ratio;
  double idle_time;
};

struct AdaptiveLBDatabase {
  std::vector<AdaptiveData> history_data;
} adaptive_lbdb;

struct AdaptiveLBInfo {
  double max_avg_ratio;
};

struct AdaptiveLBStructure {
  int lb_ideal_period;
  int lb_calculated_period;
  int lb_no_iterations;
  int global_max_iter_no;
  int global_recv_iter_counter;
  bool in_progress;
  double lb_strategy_cost;
  double lb_migration_cost;
  bool lb_period_informed;
  bool doCommStrategy;
  int lb_msg_send_no;
  int lb_msg_recv_no;
  int total_syncs_called;
  int last_lb_type;
  AdaptiveLBInfo greedy_info;
  AdaptiveLBInfo refine_info;
  AdaptiveLBInfo metis_info;
} adaptive_struct;


CkReductionMsg* lbDataCollection(int nMsg, CkReductionMsg** msgs) {
  double lb_data[6];
  lb_data[1] = 0.0;
  lb_data[2] = 0.0;
  lb_data[3] = 0.0;
  lb_data[4] = 0.0;
  lb_data[5] = 0.0;
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
    // Max idle
    lb_data[5] = ((m[5] > lb_data[5]) ? m[5] : lb_data[5]);
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
  
  max_load_vec.resize(VEC_SIZE, 0.0);
  total_load_vec.resize(VEC_SIZE, 0.0);
  total_contrib_vec.resize(VEC_SIZE, 0.0);
  max_iteration = -1;

  // If metabalancer enabled, initialize the variables
  adaptive_struct.lb_ideal_period =  INT_MAX;
  adaptive_struct.lb_calculated_period = INT_MAX;
  adaptive_struct.lb_no_iterations = -1;
  adaptive_struct.global_max_iter_no = 0;
  adaptive_struct.global_recv_iter_counter = 0;
  adaptive_struct.in_progress = false;
  adaptive_struct.lb_strategy_cost = 0.0;
  adaptive_struct.lb_migration_cost = 0.0;
  adaptive_struct.lb_msg_send_no = 0;
  adaptive_struct.lb_msg_recv_no = 0;
  adaptive_struct.total_syncs_called = 0;
  adaptive_struct.last_lb_type = -1;

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

  adaptive_struct.lb_ideal_period =  INT_MAX;
  adaptive_struct.lb_calculated_period = INT_MAX;
  adaptive_struct.lb_no_iterations = -1;
  adaptive_struct.global_max_iter_no = 0;
  adaptive_struct.global_recv_iter_counter = 0;
  adaptive_struct.in_progress = false;
  adaptive_struct.lb_strategy_cost = 0.0;
  adaptive_struct.lb_migration_cost = 0.0;
  adaptive_struct.lb_msg_send_no = 0;
  adaptive_struct.lb_msg_recv_no = 0;
  adaptive_struct.total_syncs_called = 0;
  
  max_load_vec.clear();
  total_load_vec.clear();
  total_contrib_vec.clear();

  max_load_vec.resize(VEC_SIZE, 0.0);
  total_load_vec.resize(VEC_SIZE, 0.0);
  total_contrib_vec.resize(VEC_SIZE, 0.0);

  LDResumeClients(myLDHandle);
}

bool LBDatabase::AddLoad(int iteration, double load) {
  total_contrib_vec[iteration]++;
  adaptive_struct.total_syncs_called++;
  //CkPrintf("At PE %d Total contribution for iteration %d is %lf total objs %d\n", CkMyPe(), iteration,
  //total_contrib_vec[iteration], getLBDB()->ObjDataCount());

  if (iteration > adaptive_struct.lb_no_iterations) {
    adaptive_struct.lb_no_iterations = iteration;
  }
  total_load_vec[iteration] += load;
 // if (max_load_vec[iteration] < load) {
 //   max_load_vec[iteration] = load;
 // }
  if (total_contrib_vec[iteration] == getLBDB()->ObjDataCount()) {
    double idle_time;
    IdleTime(&idle_time);
    //CkPrintf("[%d] Idle time %lf for iteration %d\n", CkMyPe(), idle_time, iteration);
    // Skips the 0th iteration collection of stats hence...
    idle_time = idle_time * getLBDB()->ObjDataCount() /
       (adaptive_struct.total_syncs_called + getLBDB()->ObjDataCount());

    double lb_data[6];
    lb_data[0] = iteration;
    lb_data[1] = 1;
    lb_data[2] = total_load_vec[iteration];
    //lb_data[2] = max_load_vec[iteration];
    lb_data[3] = total_load_vec[iteration];
    //lb_data[3] = getLBDB()->ObjDataCount();
    lb_data[4] = idle_time;
    if (total_load_vec[iteration] == 0.0) {
      lb_data[5] = idle_time;
    } else {
      lb_data[5] = idle_time/total_load_vec[iteration];
    }

   // CkPrintf("[%d] sends total load %lf idle time %lf ratio of idle/load %lf at iter %d\n", CkMyPe(),
   //     total_load_vec[iteration], idle_time,
   //     idle_time/total_load_vec[iteration], adaptive_struct.lb_no_iterations);

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
  double max_idle_load_ratio = load[5];
  int iteration_n = load[0];
  CkPrintf("** [%d] Iteration Avg load: %lf Max load: %lf Avg Idle : %lf Max Idle : %lf for %lf procs\n",iteration_n, avg, max, avg_idle, max_idle_load_ratio, load[1]);
  delete msg;
 
  // Store the data for this iteration
  adaptive_struct.lb_no_iterations = iteration_n;
  AdaptiveData data;
  data.iteration = adaptive_struct.lb_no_iterations;
  data.max_load = max;
  data.avg_load = avg;
  data.max_idle_load_ratio = max_idle_load_ratio;
  data.idle_time = avg_idle;
  adaptive_lbdb.history_data.push_back(data);

  // If lb period inform is in progress, dont inform again
  if (adaptive_struct.in_progress) {
    return;
  }

//  if (adaptive_struct.lb_period_informed) {
//    return;
//  }

  // If the max/avg ratio is greater than the threshold and also this is not the
  // step immediately after load balancing, carry out load balancing
  //if (max/avg >= 1.1 && adaptive_lbdb.history_data.size() > 4) {
  int tmp1;
  double tmp2;
  GetPrevLBData(tmp1, tmp2);
  double tolerate_imb = IMB_TOLERANCE * tmp2;

  if ((max_idle_load_ratio >= IDLE_LOAD_TOLERANCE || max/avg >= tolerate_imb) && adaptive_lbdb.history_data.size() > 4) {
    CkPrintf("Carry out load balancing step at iter max/avg(%lf) and max_idle_load_ratio ratio (%lf)\n", max/avg, max_idle_load_ratio);
//    if (!adaptive_struct.lb_period_informed) {
//      // Just for testing
//      adaptive_struct.lb_calculated_period = 40;
//      adaptive_struct.lb_period_informed = true;
//      thisProxy.LoadBalanceDecision(adaptive_struct.lb_calculated_period);
//      return;
//    }



    // If the new lb period is less than current set lb period
    if (adaptive_struct.lb_calculated_period > iteration_n + 1) {
      if (max/avg < tolerate_imb) {
        adaptive_struct.doCommStrategy = true;
        CkPrintf("No load imbalance but idle time\n");
      } else {
        adaptive_struct.doCommStrategy = false;
        CkPrintf("load imbalance \n");
      }
      adaptive_struct.lb_calculated_period = iteration_n + 1;
      adaptive_struct.lb_period_informed = true;
      adaptive_struct.in_progress = true;
      CkPrintf("Informing everyone the lb period is %d\n",
          adaptive_struct.lb_calculated_period);
      thisProxy.LoadBalanceDecision(adaptive_struct.lb_msg_send_no++, adaptive_struct.lb_calculated_period);
    }
    return;
  }

  // Generate the plan for the adaptive strategy
  int period;
  if (generatePlan(period)) {
    //CkPrintf("Carry out load balancing step at iter\n");

    // If the new lb period is less than current set lb period
    if (adaptive_struct.lb_calculated_period > period) {
      adaptive_struct.doCommStrategy = false;
      adaptive_struct.lb_calculated_period = period;
      adaptive_struct.in_progress = true;
      adaptive_struct.lb_period_informed = true;
      CkPrintf("Informing everyone the lb period is %d\n",
          adaptive_struct.lb_calculated_period);
      thisProxy.LoadBalanceDecision(adaptive_struct.lb_msg_send_no++, adaptive_struct.lb_calculated_period);
    }
  }
}

bool LBDatabase::generatePlan(int& period) {
  if (adaptive_lbdb.history_data.size() <= 8) {
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
    CkPrintf("max (%d, %lf) avg (%d, %lf)\n", i, data.max_load, i, data.avg_load);
  }
//  max /= (adaptive_struct.lb_no_iterations - adaptive_lbdb.history_data[0].iteration);
//  avg /= (adaptive_struct.lb_no_iterations - adaptive_lbdb.history_data[0].iteration);
//
//  adaptive_struct.lb_ideal_period = (adaptive_struct.lb_strategy_cost +
//  adaptive_struct.lb_migration_cost) / (max - avg);
//  CkPrintf("max : %lf, avg: %lf, strat cost: %lf, migration_cost: %lf, idealperiod : %d \n",
//      max, avg, adaptive_struct.lb_strategy_cost, adaptive_struct.lb_migration_cost, adaptive_struct.lb_ideal_period);
//

  // If linearly varying load, then find lb_period
  // area between the max and avg curve 
  // If we can attain perfect balance, then the new load is close to the
  // average. Hence we pass 1, else pass in some other value which would be the
  // new max_load after load balancing.
  int tmp1;
  double tmp2;
  GetPrevLBData(tmp1, tmp2);
  
  double tolerate_imb = tmp2;
  if (max/avg < tolerate_imb) {
    tolerate_imb = 1.0;
  }

  return getPeriodForStrategy(tolerate_imb, 1, period);

//  int refine_period, scratch_period;
//  bool obtained_refine, obtained_scratch;
//  obtained_refine = getPeriodForStrategy(1, 1, refine_period);
//  obtained_scratch = getPeriodForStrategy(1, 1, scratch_period);
//
//  if (obtained_refine) {
//    if (!obtained_scratch) {
//      period = refine_period;
//      adaptive_struct.isRefine = true;
//      return true;
//    }
//    if (scratch_period < 1.1*refine_period) {
//      adaptive_struct.isRefine = false;
//      period = scratch_period;
//      return true;
//    }
//    period = refine_period;
//    adaptive_struct.isRefine = true;
//    return true;
//  }
//
//  if (obtained_scratch) {
//    period = scratch_period;
//    adaptive_struct.isRefine = false;
//    return true;
//  }
//  return false;
}

bool LBDatabase::getPeriodForStrategy(double new_load_percent, double overhead_percent, int& period) {
  double mslope, aslope, mc, ac;
  getLineEq(new_load_percent, aslope, ac, mslope, mc);
  CkPrintf("new load percent %lf\n", new_load_percent);
  CkPrintf("\n max: %fx + %f; avg: %fx + %f\n", mslope, mc, aslope, ac);
  double a = (mslope - aslope)/2;
  double b = (mc - ac);
  double c = -(adaptive_struct.lb_strategy_cost +
      adaptive_struct.lb_migration_cost) * overhead_percent;
  //c = -2.5;
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
  }
  m1 /= i;
  a1 = (a1 * new_load_percent) / i;

  for (i = total/2; i < total; i++) {
    data = adaptive_lbdb.history_data[i];
    m2 += data.max_load;
    a2 += data.avg_load;
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
  //CkPrintf("[%d] Load balance decision made cur iteration: %d period:%d state: %d\n",CkMyPe(), adaptive_struct.lb_no_iterations, period, local_state);
  adaptive_struct.lb_ideal_period = period;
  //local_state = ON;
  adaptive_struct.lb_msg_recv_no = req_no;
  thisProxy[0].ReceiveIterationNo(req_no, adaptive_struct.lb_no_iterations);
}

void LBDatabase::LoadBalanceDecisionFinal(int req_no, int period) {
  if (req_no < adaptive_struct.lb_msg_recv_no) {
    return;
  }
//  CkPrintf("[%d] Final Load balance decision made cur iteration: %d period:%d \n",CkMyPe(), adaptive_struct.lb_no_iterations, period);
  adaptive_struct.lb_ideal_period = period;
  LDOMAdaptResumeSync(myLDHandle, period);

//  if (local_state == ON) {
//    local_state = DECIDED;
//    return;
//  }

  // If the state is PAUSE, then its waiting for the final decision from central
  // processor. If the decision is that the ideal period is in the future,
  // resume. If the ideal period is now, then carry out load balancing.
//  if (local_state == PAUSE) {
//    if (adaptive_struct.lb_no_iterations < adaptive_struct.lb_ideal_period) {
//      local_state = DECIDED;
//      //SendMinStats();
//      //FIX ME!!! ResumeClients(0);
//    } else {
//      local_state = LOAD_BALANCE;
//      //FIX ME!!! ProcessAtSync();
//    }
//    return;
//  }
//  CkPrintf("Error!!! Final decision received but the state is invalid %d\n", local_state);
}


void LBDatabase::ReceiveIterationNo(int req_no, int local_iter_no) {
  CmiAssert(CkMyPe() == 0);

  adaptive_struct.global_recv_iter_counter++;
  if (local_iter_no > adaptive_struct.global_max_iter_no) {
    adaptive_struct.global_max_iter_no = local_iter_no;
  }
  if (CkNumPes() == adaptive_struct.global_recv_iter_counter) {
    adaptive_struct.lb_ideal_period = (adaptive_struct.lb_ideal_period > adaptive_struct.global_max_iter_no) ? adaptive_struct.lb_ideal_period : adaptive_struct.global_max_iter_no + 1;
    thisProxy.LoadBalanceDecisionFinal(req_no, adaptive_struct.lb_ideal_period);
    CkPrintf("Final lb_period %d\n", adaptive_struct.lb_ideal_period);
    adaptive_struct.in_progress = false;
    adaptive_struct.global_max_iter_no = 0;
    adaptive_struct.global_recv_iter_counter = 0;
  }
}

int LBDatabase::getPredictedLBPeriod() {
  return adaptive_struct.lb_ideal_period;
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

void LBDatabase::UpdateAfterLBData(int lb, double lb_max, double lb_avg) {
  adaptive_struct.last_lb_type = lb;
  if (lb == 0) {
    adaptive_struct.greedy_info.max_avg_ratio = lb_max/lb_avg;
  } else if (lb == 1) {
    adaptive_struct.refine_info.max_avg_ratio = lb_max/lb_avg;
  }
}

void LBDatabase::GetPrevLBData(int& lb_type, double& lb_max_avg_ratio) {
  lb_type = adaptive_struct.last_lb_type;
  lb_max_avg_ratio = 1;
  if (lb_type == 0) {
    lb_max_avg_ratio = adaptive_struct.greedy_info.max_avg_ratio;
  } else if (lb_type == 1) {
    lb_max_avg_ratio = adaptive_struct.refine_info.max_avg_ratio;
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
