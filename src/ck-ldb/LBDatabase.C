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

double autoLbPeriod = 0.0;
int lb_debug=0;
int lb_ignoreBgLoad=0;

static LBDefaultCreateFn defaultCreate=NULL;
void LBSetDefaultCreate(LBDefaultCreateFn f)
{
  if (defaultCreate) CmiAbort("Error: try to create multiple load balancer strategies!");
  defaultCreate=f;
}

class LBDBResgistry {
private:
  class LBDBEntry {
  public:
    const char *name;
    LBDefaultCreateFn  fn;
    const char *help;

    LBDBEntry(): name(0), fn(0), help(0) {}
    LBDBEntry(int) {}
    LBDBEntry(const char *n, LBDefaultCreateFn f, const char *h):
      name(n), fn(f), help(h) {};
  };
  CkVec<LBDBEntry> lbtables;	 // a list of available LBs
  char *defaultBalancer;
public:
  LBDBResgistry() { defaultBalancer=NULL; }
  void displayLBs()
  {
    CmiPrintf("\nAvailable load balancers:\n");
    for (int i=0; i<lbtables.length(); i++) {
      CmiPrintf("* %s:	%s\n", lbtables[i].name, lbtables[i].help);
    }
    CmiPrintf("\n");
  }
  void add(const char *name, LBDefaultCreateFn fn, const char *help) {
    lbtables.push_back(LBDBEntry(name, fn, help));
  }
  LBDefaultCreateFn search(const char *name) {
    for (int i=0; i<lbtables.length(); i++)
      if (0==strcmp(name, lbtables[i].name)) return lbtables[i].fn;
    return NULL;
  }
  char *& defaultLB() { return defaultBalancer; };
};

static LBDBResgistry  lbRegistry;

void LBRegisterBalancer(const char *name, LBDefaultCreateFn fn, const char *help)
{
  lbRegistry.add(name, fn, help);
}

LBDBInit::LBDBInit(CkArgMsg *m)
{
#if CMK_LBDB_ON
  lbdb = CProxy_LBDatabase::ckNew();

  LBDefaultCreateFn lbFn = defaultCreate;

  char *balancer = lbRegistry.defaultLB();
  if (balancer) {
    LBDefaultCreateFn fn = lbRegistry.search(balancer);
    if (!fn) {
      lbRegistry.displayLBs();
      CmiPrintf("Abort: Unknown load balancer: '%s'!\n", balancer);
      CkExit();
    }
    else  // overwrite defaultCreate.
      lbFn = fn;
  }

  // NullLB is the default
  if (!lbFn) lbFn = CreateNullLB;
  (lbFn)();

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
  if (CmiGetArgStringDesc(argv, "+balancer", &balancer, "Use this load balancer")) {
    lbRegistry.defaultLB() = balancer;
  }

  // set up init value for LBPeriod time in milliseconds
  // it can also be set calling LDSetLBPeriod()
  CmiGetArgDoubleDesc(argv,"+LBPeriod", &autoLbPeriod,"specify the period for automatic load balancing (non atSync mode)");

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
      CmiPrintf("LB> Load balancer running with verbose mode, period time: %gms.\n", autoLbPeriod);
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
