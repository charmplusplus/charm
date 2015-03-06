/**
 * \addtogroup CkLdb
*/
/*@{*/

/*
  A Hybrid strategy that uses 3 level tree
  The top level applies refinement strategy
  The test applies greedy strategy
*/

#include "HybridLB.h"
#include "LBDBManager.h"

#include "GreedyLB.h"
#include "GreedyCommLB.h"
#include "RefineCommLB.h"
#include "RefineLB.h"
#include "MetisLB.h"

#define  DEBUGF(x)      // CmiPrintf x;

CreateLBFunc_Def(HybridLB, "Hybrid load balancer")

HybridLB::HybridLB(const CkLBOptions &opt): CBase_HybridLB(opt)
{
#if CMK_LBDB_ON
  lbname = (char *)"HybridLB";

  // defines topology in base class
//  tree = new ThreeLevelTree;

  // decide which load balancer to call
  // IMPORTANT: currently, the greedy LB must allow objects that
  // are not from existing processors.
  refine = (CentralLB *)AllocateRefineLB();
//  greedy = (CentralLB *)AllocateMetisLB();
  greedy = (CentralLB *)AllocateGreedyLB();

  initTree();
#endif
}

HybridLB::~HybridLB()
{
  delete greedy;
  delete refine;
}

void HybridLB::work(LDStats* stats)
{
#if CMK_LBDB_ON
	theLbdb->ResetAdaptive();
  LevelData *lData = levelData[currentLevel];

  // TODO: let's generate LBMigrateMsg ourself
  //  take into account the outObjs
  //if (currentLevel == tree->numLevels()-1) 
  if (currentLevel == 1) 
    greedy->work(stats);
  else
    refine->work(stats);
#endif
}
  
#include "HybridLB.def.h"

/*@{*/

