/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkLdb
*/
/*@{*/

#include <converse.h>

/*
 * This C++ file contains the Charm stub functions
 */

#include "LBDatabase.h"
#include "LBDatabase.def.h"
#include "LBSimulation.h"

#include "NullLB.h"

CkGroupID lbdb;

CkpvDeclare(int, numLoadBalancers);  /**< num of lb created */
CkpvDeclare(int, hasNullLB);         /**< true if NullLB is created */
CkpvDeclare(int, lbdatabaseInited);  /**< true if lbdatabase is inited */

// command line options
double autoLbPeriod = 0.0;		// in seconds
int lb_debug=0;
int lb_ignoreBgLoad=0;

class LBDBResgistry {
friend class LBDBInit;
private:
  // table for all available LBs linked in
  struct LBDBEntry {
    const char *name;
    LBCreateFn  fn;
    const char *help;
    int 	shown;		// if 0, donot show in help page
    LBDBEntry(): name(0), fn(0), help(0), shown(1) {}
    LBDBEntry(int) {}
    LBDBEntry(const char *n, LBCreateFn f, const char *h, int show=1):
      name(n), fn(f), help(h), shown(show) {};
  };
  CkVec<LBDBEntry> lbtables;	 	// a list of available LBs linked
  CkVec<const char *>   compile_lbs;	// load balancers at compile time
  CkVec<const char *>   runtime_lbs;	// load balancers at run time
public:
  LBDBResgistry() {}
  void displayLBs()
  {
    CmiPrintf("\nAvailable load balancers:\n");
    for (int i=0; i<lbtables.length(); i++) {
      LBDBEntry &entry = lbtables[i];
      if (entry.shown) CmiPrintf("* %s:	%s\n", entry.name, entry.help);
    }
    CmiPrintf("\n");
  }
  void addEntry(const char *name, LBCreateFn fn, const char *help, int shown) {
    lbtables.push_back(LBDBEntry(name, fn, help, shown));
  }
  void addCompiletimeBalancer(const char *name) {
    compile_lbs.push_back(name); 
  }
  void addRuntimeBalancer(const char *name) {
    runtime_lbs.push_back(name); 
  }
  LBCreateFn search(const char *name) {
    for (int i=0; i<lbtables.length(); i++)
      if (0==strcmp(name, lbtables[i].name)) return lbtables[i].fn;
    return NULL;
  }
};

static LBDBResgistry lbRegistry;

void LBDefaultCreate(const char *lbname)
{
  lbRegistry.addCompiletimeBalancer(lbname);
}

// default is to show the helper
void LBRegisterBalancer(const char *name, LBCreateFn fn, const char *help, int shown)
{
  lbRegistry.addEntry(name, fn, help, shown);
}

// create a load balancer group using the strategy name
static void createLoadBalancer(const char *lbname)
{
    LBCreateFn fn = lbRegistry.search(lbname);
    if (!fn) {
      CmiPrintf("Abort: Unknown load balancer: '%s'!\n", lbname);
      lbRegistry.displayLBs();
      CkExit();
    }
    fn();
}

LBDBInit::LBDBInit(CkArgMsg *m)
{
#if CMK_LBDB_ON
  lbdb = CProxy_LBDatabase::ckNew();

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
    // NullLB is the default
    // user may create his own load balancer in his code, but never mind
    // NullLB can disable itself if there is a non NULL LB.
    createLoadBalancer("NullLB");
  }

  if (LBSimulation::doSimulation) {
    CmiPrintf("Charm++> Entering Load Balancer Simulation Mode ... \n");
    CProxy_LBDatabase(lbdb).ckLocalBranch()->StartLB();
  }
#endif
  delete m;
}


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
    lbRegistry.addRuntimeBalancer(balancer);
  }

  // set up init value for LBPeriod time in seconds
  // it can also be set calling LDSetLBPeriod()
  CmiGetArgDoubleDesc(argv,"+LBPeriod", &autoLbPeriod,"specify the period for automatic load balancing in seconds (for non atSync mode)");

  /******************* SIMULATION *******************/
  // get the step number at which to dump the LB database
  CmiGetArgIntDesc(argv, "+LBDump", &LBSimulation::dumpStep, "Dump the LB state at this step");
  CmiGetArgStringDesc(argv, "+LBDumpFile", &LBSimulation::dumpFile, "Set the LB state file name");
  // get the simulation flag
  LBSimulation::doSimulation = CmiGetArgFlagDesc(argv, "+LBSim", "Read LB state from LBDumpFile");
  LBSimulation::simProcs = 0;
  CmiGetArgIntDesc(argv, "+LBSimProcs", &LBSimulation::simProcs, "Number of target processors.");

  lb_debug = CmiGetArgFlagDesc(argv, "+LBDebug", "Turn on LB debugging printouts");
  lb_ignoreBgLoad = CmiGetArgFlagDesc(argv, "+LBObjOnly", "Load balancer only balance migratable object without considering the background load, etc");
  if (CkMyPe() == 0) {
    if (lb_debug) {
      CmiPrintf("LB> Load balancer running with verbose mode, period time: %gs.\n", autoLbPeriod);
    }
    if (lb_ignoreBgLoad)
      CmiPrintf("LB> Load balancer only balance migratable object.\n");
    if (LBSimulation::doSimulation)
      CmiPrintf("LB> Load balancer running in simulation mode.\n");
  }
}

int LBDatabase::manualOn = 0;

void LBDatabase::init(void) 
{
  myLDHandle = LDCreate();

  mystep = 0;
  nloadbalancers = 0;

  int num_proc = CkNumPes();
  avail_vector = new char[num_proc];
  for(int proc = 0; proc < num_proc; proc++)
      avail_vector[proc] = 1;
  new_ld_balancer = 0;

  CkpvAccess(lbdatabaseInited) = 1;
#if CMK_LBDB_ON
  if (manualOn) TurnManualLBOn();
#endif
}

void LBDatabase::get_avail_vector(char * bitmap) {
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
    for(int count = 0; count < num_proc; count++){
        avail_vector[count] = bitmap[count];
        if((bitmap[count] == 1) && !assigned){
            new_ld_balancer = count;
            assigned = 1;
        }
    }
}

int LBDatabase::getLoadbalancerTicket()  { 
  int seq = nloadbalancers;
  nloadbalancers ++;
  loadbalancers.growAtLeast(nloadbalancers); 
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
  loadbalancers.growAtLeast(seq); 
  loadbalancers[seq] = lb;
}

void LBDatabase::nextLoadbalancer(int seq) {
  if (seq == -1) return;		// -1 means this is the only LB
  int next = seq+1;
  if (next == nloadbalancers) next --;
  CmiAssert(loadbalancers[next]);
  if (seq != next) {
    loadbalancers[seq]->turnOff();
    loadbalancers[next]->turnOn();
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

/*@}*/
