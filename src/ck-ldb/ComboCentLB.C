/**
 * \addtogroup CkLdb
*/
/*@{*/

/*
Status:
  * support nonmigratable attrib
  * does not support processor avail bitvector
*/

#include "ComboCentLB.h"

extern LBAllocFn getLBAllocFn(const char *lbname);

CreateLBFunc_Def(ComboCentLB, "Allow multiple strategies to work serially")

ComboCentLB::ComboCentLB(const CkLBOptions &opt): CBase_ComboCentLB(opt)
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

void ComboCentLB::work(LDStats* stats)
{
  int nlbs = clbs.length();
  int n_objs = stats->n_objs;
  int *from_orig = new int[n_objs];
  int obj;

  // stats->from_proc should remain untouched at end
  for (obj=0; obj<n_objs; obj++) from_orig[obj] = stats->from_proc[obj];

  for (int i=0; i<nlbs; i++) {
    clbs[i]->work(stats);
    if (i!=nlbs-1) {
      for (obj=0; obj<stats->n_objs; obj++) 
        stats->from_proc[obj] = stats->to_proc[obj];
    }
  }

  for (obj=0; obj<n_objs; obj++) stats->from_proc[obj] = from_orig[obj];

  delete [] from_orig;
}

#include "ComboCentLB.def.h"


/*@}*/
