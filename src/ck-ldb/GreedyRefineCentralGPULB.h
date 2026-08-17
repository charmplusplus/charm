/**
 * \addtogroup CkLdb
*/
/*@{*/

/**
 * Two-level GPU-aware variant of GreedyRefineCentralLB.
 *
 * Objects are balanced at two granularities:
 *   - Across GPU groups (the set of PEs that share a gpu_device_id, i.e. the
 *     PEs of a process bound to one GPU): balanced by GPU load (gpuTime).
 *   - Within a GPU group, across that group's PEs: balanced by CPU load
 *     (wallTime).
 *
 * This matches a setup where each process owns a GPU. The coarse decision of
 * which GPU/process an object lands on is driven by the object's GPU work; the
 * fine decision of which PE inside that process runs the object's host-side
 * code is driven by its CPU work.
 *
 * Inherits the greedy-refine migration-minimization machinery and the
 * concurrent multi-parameter solution search from GreedyRefineCentralLB.
 *
 * supports processor avail bitvector
 * supports nonmigratable attrib
*/

#ifndef _GREEDY_REFINE_GPU_LB_H_
#define _GREEDY_REFINE_GPU_LB_H_

#include "CentralLB.h"
#include "GreedyRefineCentralGPULB.decl.h"

void CreateGreedyRefineCentralGPULB();
BaseLB *AllocateGreedyRefineCentralGPULB();

#define __DEBUG_GREEDY_REFINE_GPU_ 0

class GreedyRefineCentralGPULB : public CBase_GreedyRefineCentralGPULB {
public:
  GreedyRefineCentralGPULB(const CkLBOptions &);
  GreedyRefineCentralGPULB(CkMigrateMessage *m);
  void work(LDStats* stats);
  void receiveSolutions(CkReductionMsg *msg);
  void receiveTotalTime(double time);
  void setMigrationTolerance(float tol) { migrationTolerance = tol; }

private:
  bool QueryBalanceNow(int step) { return true; }

  class GProc {
  public:
    GProc() : available(true), load(0), bgload(0), cpuLoad(0), bg_walltime(0) {}
    int id;
    bool available;
    int pos;    // position in min heap
    double load;       // GPU load accumulator (used by non-CUDA PE-level path)
    double bgload;     // background GPU load (group level under CUDA)
    double cpuLoad;    // CPU/wall load accumulator for within-group balancing
    double bg_walltime;
    float speed;
  };

  class GObj {
  public:
    int id;
    double load;     // GPU load (gpuTime) — drives cross-group assignment
    double cpuLoad;  // CPU load (wallTime) — drives within-group PE assignment
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

#if __DEBUG_GREEDY_REFINE_GPU_
  void dumpObjLoads(std::vector<GObj> &objs);
  void dumpProcLoads(std::vector<GProc> &procs);
#endif
};

#endif

/*@}*/
