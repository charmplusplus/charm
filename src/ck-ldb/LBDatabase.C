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

#include "NullLB.h"

CkGroupID lbdb;

CkpvDeclare(int, numLoadBalancers);  /**< num of lb created */
CkpvDeclare(int, hasNullLB);         /**< true if NullLB is created */
CkpvDeclare(int, lbdatabaseInited);  /**< true if lbdatabase is inited */
CkpvDeclare(int, dumpStep);			 /**< the load balancing step at which to dump data */
CkpvDeclare(char*, dumpFile);		 /**< the name of the file in which the data will be dumped */

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
  CkpvInitialize(int, dumpStep);
  CkpvAccess(dumpStep) = -1;
  CkpvInitialize(char*, dumpFile);
  CkpvAccess(dumpFile) = NULL;

  char **argv = CkGetArgv();
  char *balancer = NULL;
  if (CmiGetArgString(argv, "+balancer", &balancer)) {
    lbRegistry.defaultLB() = balancer;
  }

  // get the step number at which to dump the LB database
  CmiGetArgInt(argv, "+LBDump", &CkpvAccess(dumpStep));
  CmiGetArgString(argv, "+LBDumpFile", &CkpvAccess(dumpFile));
}

int LBDatabase::manualOn = 0;

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
