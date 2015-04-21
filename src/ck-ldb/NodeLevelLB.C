/**
 * Author Harshitha Menon (gplkrsh2@illinois.edu)
 * Node level load balancer which first performs load balancing across nodes and
 * then within a node.
 * Eg Usage: +balancer NodeLevelLB:MetisLB,RefineLB where first MetisLB will be
 * applied across nodes followed by RefineLB within a node
*/
#include "NodeLevelLB.h"

extern LBAllocFn getLBAllocFn(const char *lbname);

CreateLBFunc_Def(NodeLevelLB, "Node level load balancer")

#if defined(_WIN32) && ! defined(__CYGWIN__)
  /* strtok is thread safe in Windows */
#define strtok_r(x,y,z) strtok(x,y)
#endif

NodeLevelLB::NodeLevelLB(const CkLBOptions &opt): CBase_NodeLevelLB(opt) {
  lbname = "NodeLevelLB";
  const char *lbs = theLbdb->loadbalancer(opt.getSeqNo());
  if (CkMyPe() == 0)
    CkPrintf("[%d] NodeLevelLB created with %s\n",CkMyPe(), lbs);

  char *lbcopy = strdup(lbs);
  char *p = strchr(lbcopy, ':');
  char *ptr = NULL;
  char *lbname;
  if (p==NULL) {
    CmiAbort("LB> Nodelevel load balancer not specified\n");
  }
  lbname = strtok_r(p+1, ",", &ptr);
  while (lbname) {
    LBAllocFn fn = getLBAllocFn(lbname);
    if (fn == NULL) {
      CkPrintf("LB> Invalid load balancer: %s.\n", lbname);
      CmiAbort("");
    }
    BaseLB *alb = fn();
    clbs.push_back((CentralLB*)alb);
    lbname = strtok_r(NULL, ",", &ptr);
  }

  // HybridBaseLB constructs a default tree
  if (tree) {
    delete tree;
  }

  // Construct a tree with three levels where the lowest level is at the node
  // level
  tree = new ThreeLevelTree(CmiMyNodeSize());
  num_levels = tree->numLevels();
  initTree();
}

void NodeLevelLB::work(LDStats* stats) {
  if (currentLevel > 2) {
    CkAbort("NodeLevelLB> Maximum levels can only be 3\n");
  }

  CentralLB* clb;
  int idx_lb = num_levels - currentLevel - 1;
  if (clbs.length() > idx_lb) {
    clb = clbs[idx_lb];
  } else {
    clb = clbs[clbs.length() - 1];
  }
  clb->work(stats);
}
#include "NodeLevelLB.def.h"
