/**
 * \addtogroup CkLdb
*/
/*@{*/

/**
 * Author: jjgalvez@illinois.edu (Juan Galvez)
 * Greedy algorithm to minimize cpu max_load and object migrations.
 * Can find solution equal or close to regular Greedy with less (sometimes much less) migrations.
 * The amount of migrations that the user can tolerate is passed via the command-line
 * option +LBPercentMoves (as percentage of chares that can be moved).
 *
 * If LBPercentMoves is not passed, strategy assumes it can move all objects.
 * In this case, the algorithm will give preference to minimizing cpu max_load.
 * It will still move less than greedy, but the amount of migrations
 * will depend very much on the particular case (object load distribution and processor background loads),
 *
 * supports processor avail bitvector
 * supports nonmigratable attrib
 *
*/

#ifndef _GREEDY_REFINE_LB_H_
#define _GREEDY_REFINE_LB_H_

#include "CentralLB.h"
#include "GreedyRefineCentralLB.decl.h"

void CreateGreedyRefineCentralLB();
BaseLB *AllocateGreedyRefineCentralLB();

#define __DEBUG_GREEDY_REFINE_ 0

class GreedyRefineCentralLB : public CBase_GreedyRefineCentralLB {
public:
  GreedyRefineCentralLB(const CkLBOptions &);
  GreedyRefineCentralLB(CkMigrateMessage *m);
  void work(LDStats* stats);
  void receiveSolutions(CkReductionMsg *msg);
  void receiveTotalTime(double time);
  void setMigrationTolerance(float tol) { migrationTolerance = tol; }

private:
  bool QueryBalanceNow(int step) { return true; }

  class GProc {
  public:
    GProc() : available(true), load(0) {}
    int id;
    bool available;
    int pos;    // position in min heap
    double load;
    double bgload; // background load
    float speed;
  };

  class GObj {
  public:
    int id;
    double load;
    int oldPE;
  };

  class ObjLoadGreater {
  public:
    inline bool operator() (const GObj *o1, const GObj *o2) const {
      return (o1->load > o2->load);
    }
  };

  class PHeap;
  class Solution;

  double fillData(LDStats *stats,
                  std::vector<GObj> &objs,
                  std::vector<GObj*> &pobjs,
                  std::vector<GProc> &procs,
                  PHeap &procHeap);

  double greedyLB(const std::vector<GObj*> &pobjs, PHeap &procHeap, const BaseLB::LDStats *stats) const;
  void sendSolution(double maxLoad, int migrations);

  double strategyStartTime;
  double totalObjLoad;
  int availablePes;
  float migrationTolerance;
  int totalObjs;

#if __DEBUG_GREEDY_REFINE_
  void dumpObjLoads(std::vector<GObj> &objs);
  void dumpProcLoads(std::vector<GProc> &procs);
#endif
};

#endif

/*@}*/