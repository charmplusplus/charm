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
  A Hybrid strategy that uses 3 level tree
  The top level applies refinement strategy
  The test applies greedy strategy
*/

#include "charm++.h"
#include "HybridLB.h"
#include "LBDBManager.h"

#include "GreedyLB.h"
#include "GreedyCommLB.h"
#include "RefineCommLB.h"
#include "RefineLB.h"

#define  DEBUGF(x)      // CmiPrintf x;

CreateLBFunc_Def(HybridLB, "Hybrid load balancer");

HybridLB::HybridLB(const CkLBOptions &opt): HybridBaseLB(opt)
{
#if CMK_LBDB_ON
  lbname = (char *)"HybridLB";

  // defines topology in base class
//  tree = new ThreeLevelTree;

  // decide which load balancer to call
  greedy = (CentralLB *)AllocateGreedyLB();
  refine = (CentralLB *)AllocateRefineLB();
#endif
}

HybridLB::~HybridLB()
{
  delete greedy;
  delete refine;
}

void HybridLB::work(LDStats* stats,int count)
{
#if CMK_LBDB_ON
  LevelData *lData = levelData[currentLevel];

  // TODO: let's generate LBMigrateMsg ourself
  //  take into account the outObjs
  if (currentLevel == tree->numLevels()-1) 
    refine->work(stats, count);
  else
    greedy->work(stats, count);
#endif
}
  
#include "HybridLB.def.h"

/*@{*/

