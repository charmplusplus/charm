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

/*
Status:
  * support nonmigratable attrib
  * does not support processor avail bitvector
*/

#include <charm++.h>

#include "ComboCentLB.h"

extern LBAllocFn getLBAllocFn(char *lbname);

CreateLBFunc_Def(ComboCentLB);

static void lbinit(void) {
  LBRegisterBalancer("ComboCentLB", 
		     CreateComboCentLB, 
		     AllocateComboCentLB, 
		     "Allow multiple strategies to work serially");
}

ComboCentLB::ComboCentLB(const CkLBOptions &opt): CentralLB(opt)
{
  lbname = "ComboCentLB";
  const char *lbs = theLbdb->loadbalancer(opt.getSeqNo());
  if (CkMyPe() == 0)
    CkPrintf("[%d] ComboCentLB created with %s\n",CkMyPe(), lbs);
  
  char *lbcopy = strdup(lbs);
  char *p = strchr(lbcopy, ':');
  if (p==NULL) return;
  p = strtok(p+1, ",");
  while (p) {
    LBAllocFn fn = getLBAllocFn(p);
    if (fn == NULL) {
      CkPrintf("LB> Invalid load balancer: %s.\n", p);
      CmiAbort("");
    }
    BaseLB *alb = fn();
    clbs.push_back((CentralLB*)alb);
    p = strtok(NULL, ",");
  }
}

void ComboCentLB::work(CentralLB::LDStats* stats, int count)
{
  int nlbs = clbs.length();
  for (int i=0; i<nlbs; i++) {
    clbs[i]->work(stats, count);
    if (i!=nlbs-1) {
      for (int obj=0; obj<stats->n_objs; obj++) 
        stats->from_proc[obj] = stats->to_proc[obj];
    }
  }
}

#include "ComboCentLB.def.h"


/*@}*/
