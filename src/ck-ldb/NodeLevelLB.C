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

NodeLevelLB::NodeLevelLB(const CkLBOptions &opt): CBase_NodeLevelLB(opt) {
  lbname = "NodeLevelLB";
  const char *lbs = theLbdb->loadbalancer(opt.getSeqNo());
  if (CkMyPe() == 0)
    CkPrintf("[%d] NodeLevelLB created with %s\n",CkMyPe(), lbs);

  char *lbcopy = strdup(lbs);
  char *p = strchr(lbcopy, ':');
  if (p==NULL) {
    CmiAbort("LB> Nodelevel load balancer not specified\n");
  }
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
