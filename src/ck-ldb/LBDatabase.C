/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <converse.h>


/*
 * This C++ file contains the Charm stub functions
 */

#include "LBDatabase.h"
#include "LBDatabase.def.h"

#include "NullLB.h"

CkGroupID lbdb;

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
  CkVec<LBDBEntry> lbtables;
public:
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

  char *balancer = NULL;
  if (CmiGetArgString(m->argv, "+balancer", &balancer)) {
    LBDefaultCreateFn fn = lbRegistry.search(balancer);
    if (!fn) { 
      lbRegistry.displayLBs(); 
      CmiAbort("Unknown load balancer!"); 
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
