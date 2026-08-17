/**
 * \addtogroup CkLdb
*/
/*@{*/

/**
 * Two-level GPU-aware greedy-refine load balancer.
 *
 * Cross-group (per-GPU/per-process) balancing uses GPU load (gpuTime);
 * within-group (per-PE) balancing uses CPU load (wallTime).  See the header
 * for the full description.  Derived from GreedyRefineCentralLB
 * (jjgalvez@illinois.edu).
*/

#include "charm++.h"
#include "ckgraph.h"
#include "GreedyRefineCentralGPULB.h"

#include <float.h>
#include <limits.h>
#include <algorithm>
#include <math.h>
#if CMK_CUDA
#include <unordered_map>
#endif

extern int quietModeRequested;

// a solution is feasible if num migrations <= user-specified limit
// LOAD_MIG_BAL is used to control tradeoff between maxload and migrations
// when selecting solutions from the feasible set
#define LOAD_MIG_BAL 1.003

using namespace std;

class GreedyRefineCentralGPULB::Solution {
public:
  Solution() {}
  Solution(int pe, double maxLoad, int nmoves) : pe(pe), max_load(maxLoad), migrations(nmoves) {}
  int pe; // pe who produced this solution
  float max_load;
  int migrations;

  void pup(PUP::er &p) {
    p|pe;
    p|max_load;
    p|migrations;
  }
};

// custom heap to allow removal of processors from any position
class GreedyRefineCentralGPULB::PHeap {
public:
  PHeap(int numpes) {
    Q.reserve(numpes+1);
    Q.push_back(NULL);  // first element of the array is NULL
  }

  void addProcessors(std::vector<GreedyRefineCentralGPULB::GProc> &procs, bool bgLoadZero, bool insert=true) {
    for (int i=0; i < procs.size(); i++) {
      GreedyRefineCentralGPULB::GProc &p = procs[i];
      if (p.available) {
        p.load = p.bgload;
        if (insert) {
          Q.push_back(&p);
          p.pos = Q.size()-1;
        }
      }
    }
    if (!bgLoadZero) buildMinHeap();
  }

  inline GreedyRefineCentralGPULB::GProc *top() const {
    CkAssert(Q.size() > 1);
    return Q[1];
  }

  inline void push(GreedyRefineCentralGPULB::GProc *p) {
    Q.push_back(p);
    p->pos = Q.size()-1;
    siftUp(p->pos);
  }

  inline GreedyRefineCentralGPULB::GProc *pop() {
    if (Q.size() == 1) return NULL;
    GreedyRefineCentralGPULB::GProc *retval;
    if (Q.size() == 2) {
      retval = Q[1];
      Q.pop_back();
      return retval;
    }
    retval = Q[1];
    Q[1] = Q.back();
    Q.pop_back();
    Q[1]->pos = 1;
    siftDown(1);
    return retval;
  }

  // remove processor from any position in the heap
  void remove(GreedyRefineCentralGPULB::GProc *p) {
    int pos = p->pos;
    if ((Q.size() == 2) || (pos == Q.size()-1)) return Q.pop_back();
    if (pos == 1) { pop(); return; }
    Q[pos] = Q.back();
    Q.pop_back();
    Q[pos]->pos = pos;
    if (Q[pos/2]->load > Q[pos]->load) siftUp(pos);
    else siftDown(pos);
  }

  inline void clear() {
    Q.clear();
    Q.push_back(NULL);
  }

private:

  void min_heapify(int i) {
    const int left = 2*i;
    const int right = 2*i + 1;
    int smallest = i;
    if ((left < Q.size()) && (Q[left]->load < Q[smallest]->load)) smallest = left;
    if ((right < Q.size()) && (Q[right]->load < Q[smallest]->load)) smallest = right;
    if (smallest != i) {
      swap(i,smallest);
      Q[i]->pos = i;
      Q[smallest]->pos = smallest;
      min_heapify(smallest);
    }
  }

  void inline buildMinHeap() {
    for (int i=Q.size()/2; i > 0; i--) min_heapify(i);
  }

  inline void swap(int pos1, int pos2) {
    GreedyRefineCentralGPULB::GProc *t = Q[pos1];
    Q[pos1] = Q[pos2];
    Q[pos2] = t;
  }

  void siftUp(int pos) {
    if (pos == 1) return;   // reached root
    int ppos = pos/2;
    if (Q[ppos]->load > Q[pos]->load) {
      swap(ppos,pos);
      Q[ppos]->pos = ppos;
      Q[pos]->pos = pos;
      siftUp(ppos);
    }
  }

  inline int minChild(int pos) const {
    int c1 = pos*2;
    int c2 = pos*2 + 1;
    if (c1 >= Q.size()) return -1;
    if (c2 >= Q.size()) return c1;
    if (Q[c1]->load < Q[c2]->load) return c1;
    else return c2;
  }

  void siftDown(int pos) {
    int cpos = minChild(pos);
    if (cpos == -1) return;
    if (Q[pos]->load > Q[cpos]->load) {
      swap(pos,cpos);
      Q[cpos]->pos = cpos;
      Q[pos]->pos = pos;
      siftDown(cpos);
    }
  }

  std::vector<GreedyRefineCentralGPULB::GProc*> Q;
};

CreateLBFunc_Def(GreedyRefineCentralGPULB, "Two-level GPU-aware greedy refinement-based algorithm")

GreedyRefineCentralGPULB::GreedyRefineCentralGPULB(const CkLBOptions &opt): CBase_GreedyRefineCentralGPULB(opt), migrationTolerance(1.0)
{
  lbname = "GreedyRefineCentralGPULB";
  if ((CkMyPe() == 0) && !quietModeRequested)
    CkPrintf("CharmLB> GreedyRefineCentralGPULB created.\n");
  if (_lb_args.percentMovesAllowed() < 100) {
    migrationTolerance = float(_lb_args.percentMovesAllowed())/100.0;
  }
  concurrent = true;
}

GreedyRefineCentralGPULB::GreedyRefineCentralGPULB(CkMigrateMessage *m): CBase_GreedyRefineCentralGPULB(m), migrationTolerance(1.0) {
  lbname = "GreedyRefineCentralGPULB";
  if (_lb_args.percentMovesAllowed() < 100)
    migrationTolerance = float(_lb_args.percentMovesAllowed())/100.0;
  concurrent = true;
}

// ------------------------------------------------

// regular greedy lb algorithm (CPU-only / non-CUDA path)
double GreedyRefineCentralGPULB::greedyLB(const std::vector<GreedyRefineCentralGPULB::GObj*> &pobjs,
              GreedyRefineCentralGPULB::PHeap &procHeap,
              const BaseLB::LDStats *stats) const
{
  double max_load = 0;
  int nmoves = 0;
  for (int i=0; i < pobjs.size(); i++) {
    const GreedyRefineCentralGPULB::GObj *obj = pobjs[i];
    GreedyRefineCentralGPULB::GProc *p = procHeap.pop();  // least loaded processor
    // update processor load
    p->load += (obj->load / p->speed);
    procHeap.push(p);

    if (p->id != obj->oldPE) nmoves++;
    if (p->load > max_load) max_load = p->load;
  }

  if ((CkMyPe() == cur_ld_balancer+1) && (_lb_args.debug() > 1)) {
    CkPrintf("[%d] %f : Greedy strategy nmoves=%d, max_load=%f\n", CkMyPe(),
             CkWallTimer() - strategyStartTime, nmoves, max_load);
  }
  return max_load;
}

// -----------------------------------------------
#if __DEBUG_GREEDY_REFINE_GPU_
#include <fstream>
void GreedyRefineCentralGPULB::dumpObjLoads(std::vector<GreedyRefineCentralGPULB::GObj> &objs) {
  std::ofstream outfile("objloads.txt");
  outfile << objs.size() << std::endl;
  for (int i=0; i < objs.size(); i++) {
    GreedyRefineCentralGPULB::GObj &obj = objs[i];
    if ((i > 0) && (i % 100 == 0)) outfile << obj.load << std::endl;
    else outfile << obj.load << " ";
  }
  outfile.close();
}
void GreedyRefineCentralGPULB::dumpProcLoads(std::vector<GreedyRefineCentralGPULB::GProc> &procs) {
  std::ofstream outfile("proc_bg_loads.txt");
  outfile << procs.size() << std::endl;
  for (int i=0; i < procs.size(); i++) {
    GreedyRefineCentralGPULB::GProc &p = procs[i];
    if ((i > 0) && (i % 100 == 0)) outfile << p.load << std::endl;
    else outfile << p.load << " ";
  }
  outfile.close();
}
#endif

double GreedyRefineCentralGPULB::fillData(LDStats *stats,
                            std::vector<GreedyRefineCentralGPULB::GObj> &objs,
                            std::vector<GreedyRefineCentralGPULB::GObj*> &pobjs,
                            std::vector<GreedyRefineCentralGPULB::GProc> &procs,
                            PHeap &procHeap)
{
  const int n_pes = stats->nprocs();
  const int n_objs = stats->n_migrateobjs;
  // most of these variables are just for printing stats when _lb_args.debug()
  int unmigratableObjs = 0;
  availablePes = 0; totalObjLoad = 0;
  double minBGLoad = DBL_MAX; double avgBGLoad = 0; double maxBGLoad = 0;
  double minSpeed  = DBL_MAX; double maxSpeed  = 0; double avgSpeed  = 0;
  double minOload  = DBL_MAX; double maxOload  = 0;

  for (int pe=0; pe < n_pes; pe++) {
    GreedyRefineCentralGPULB::GProc &p = procs[pe];
    p.id = pe;
    p.available = stats->procs[pe].available;
    p.speed = stats->procs[pe].pe_speed;
    if (p.available) {
      availablePes++;
      #ifndef CMK_CUDA
        p.bgload = stats->procs[pe].bg_walltime;
        if (p.bgload > maxBGLoad) maxBGLoad = p.bgload;
      #else
        // bgload is the background GPU load (aggregated per group below).
        // cpuLoad is the background CPU/wall load, used for within-group
        // PE-level balancing; seed it with this PE's background walltime.
        p.bgload = 0.0;
        p.bg_walltime = stats->procs[pe].bg_walltime;
        p.cpuLoad = stats->procs[pe].bg_walltime;
      #endif
      if (_lb_args.debug() > 1) {
        double &speed = stats->procs[pe].pe_speed;
        if (speed < minSpeed) minSpeed = speed;
        if (speed > maxSpeed) maxSpeed = speed;
        avgSpeed += speed;
      }
    }
  }
  if (!availablePes) CkAbort("GreedyRefineCentralGPULB: No available processors\n");

  for (int i=0; i < n_objs; i++) {
    LDObjData &oData = stats->objData[i];
    GreedyRefineCentralGPULB::GObj &obj = objs[i];
    int pe = stats->from_proc[i];
    obj.id = i;
    obj.oldPE = pe;
    CkAssert(pe >= 0 && pe <= n_pes);
    if (pe == n_pes) obj.oldPE = -1; // this can happen in HybridLB if object comes from outside group. mark oldPE as -1 in this situation
    if (!oData.migratable) {
      CkAssert(pe < n_pes);
      unmigratableObjs++;
      GreedyRefineCentralGPULB::GProc &p = procs[pe];
      if (!p.available)
        CkAbort("GreedyRefineCentralGPULB: nonmigratable object on unavailable processor\n");
#if CMK_CUDA
      // GPU load is background for the PE's GPU group; CPU load is background
      // for the PE itself.
      p.bgload  += oData.gpuTime;
      p.cpuLoad += oData.wallTime;
      if (p.bgload > maxBGLoad) maxBGLoad = p.bgload;
#else
      double nmObjLoad = oData.wallTime;
      p.bgload += nmObjLoad; // take non-migratable object load as background load
      if (p.bgload > maxBGLoad) maxBGLoad = p.bgload;
#endif
    } else {
#if CMK_CUDA
      // load -> GPU work (cross-group objective); cpuLoad -> CPU work (intra-group)
      obj.load    = oData.gpuTime;
      obj.cpuLoad = oData.wallTime;
#else
      obj.load    = oData.wallTime * stats->procs[pe].pe_speed;
      obj.cpuLoad = oData.wallTime;
#endif
      pobjs.push_back(&obj);
      totalObjLoad += obj.load;
      if (_lb_args.debug() > 1) {
        if (obj.load < minOload) minOload = obj.load;
        if (obj.load > maxOload) maxOload = obj.load;
      }
    }
  }

  procHeap.addProcessors(procs, (maxBGLoad <= 0.001), true);

  // ---- print some stats ----
  if ((_lb_args.debug() > 1) && (!concurrent || (CkMyPe() == cur_ld_balancer))) {
    for (int pe=0; pe < n_pes; pe++) {
      GreedyRefineCentralGPULB::GProc &p = procs[pe];
      if (!p.available) continue;
      if (p.bgload < minBGLoad) minBGLoad = p.bgload;
      avgBGLoad += p.bgload;
    }
    CkPrintf("[%d] GreedyRefineCentralGPULB: num pes=%d, num objs=%d\n", CkMyPe(), n_pes, n_objs);
    CkPrintf("[%d] Unavailable processors=%d, Unmigratable objs=%d\n", CkMyPe(), n_pes - availablePes, unmigratableObjs);
    CkPrintf("[%d] min_bgload=%f mean_bgload=%f max_bgload=%f\n", CkMyPe(), minBGLoad, (avgBGLoad / availablePes), maxBGLoad);
    CkPrintf("[%d] min_oload=%f mean_oload=%f max_oload=%f\n", CkMyPe(), minOload, (totalObjLoad / (n_objs - unmigratableObjs)), maxOload);
    CkPrintf("[%d] min_speed=%f mean_speed=%f max_speed=%f\n", CkMyPe(), minSpeed, (avgSpeed / availablePes), maxSpeed);
  }

  return maxBGLoad;
}

static const float Avals[] = {1.0, 1.005, 1.01, 1.015, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.16, 1.20, 1.30};
static const float Bvals[] = {FLT_MAX, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3};
#define Avals_len 14
#define Bvals_len 16
#define NUM_SOLUTIONS Avals_len*Bvals_len+1
static void getGreedyRefineParams(int rank, float &A, float &B) {
  if (rank == 0) { A = 0; B = -1; return; } // causes PE0 to run regular greedy
  rank--;
  int x = rank / Bvals_len;
  if (x >= Avals_len) {
    A = B = -1;
  } else {
    A = Avals[x];
    B = Bvals[rank % Bvals_len];
  }
}

void GreedyRefineCentralGPULB::sendSolution(double maxLoad, int migrations)
{
  // gather results in central PE, who will decide which solution is the best
  // only the objective values of the solutions are sent, not the whole solutions

  GreedyRefineCentralGPULB::Solution sol(CkMyPe(), maxLoad, migrations);
  size_t buf_size = sizeof(GreedyRefineCentralGPULB::Solution);
  void *buffer = malloc(buf_size);
  PUP::toMem pd(buffer);
  pd|sol;

  CkCallback cb(CkIndex_GreedyRefineCentralGPULB::receiveSolutions((CkReductionMsg*)NULL), thisProxy[cur_ld_balancer]);
  contribute(buf_size, buffer, CkReduction::set, cb);

  if ((_lb_args.debug() > 1) && (CkMyPe() == cur_ld_balancer)) {
    CkPrintf("[%d] %f : Called gather/reduction\n", CkMyPe(), CkWallTimer() - strategyStartTime);
  }

  free(buffer);
}

void GreedyRefineCentralGPULB::work(LDStats *stats)
{
  strategyStartTime = CkWallTimer();
  float A = 1.001, B = FLT_MAX; // Use A=0, B=-1 to imitate regular Greedy (ignore migrations)
  if (concurrent) {
    getGreedyRefineParams(CkMyPe(), A, B);
    if (A < 0) {
      sendSolution(-1,-1);  // send empty response to PE0
      return;
    }
  }

  const int n_pes = stats->nprocs();
  totalObjs = stats->n_migrateobjs;

  std::vector<GreedyRefineCentralGPULB::GObj> objs(totalObjs);
  // will sort pobjs instead of objs (faster swapping). will only contain pointers
  // to migratable objects
  std::vector<GreedyRefineCentralGPULB::GObj*> pobjs;
  pobjs.reserve(totalObjs);

  std::vector<GreedyRefineCentralGPULB::GProc> procs(n_pes);
  PHeap procHeap(n_pes);

  // fill data structures used by algorithm
  double maxLoad = fillData(stats, objs, pobjs, procs, procHeap);

  // ------------ apply greedy refine algorithm --------------

  // Sort by GPU load (GObj::load) — the cross-group objective.
  std::sort(pobjs.begin(), pobjs.end(), GreedyRefineCentralGPULB::ObjLoadGreater());

  int nmoves = 0;
  double greedyMaxLoad = 0;

#if CMK_CUDA
  // ---- Two-level GPU-aware path ----
  //
  // Outer: balance GPU load (GObj::load) across GPU groups. A GPU group is the
  //        set of available PEs that share a gpu_device_id (the PEs of one
  //        process bound to one GPU).
  // Inner: within the chosen group, balance CPU load (GObj::cpuLoad / GProc::cpuLoad)
  //        across the group's PEs.

  struct GPUGrp {
    uint64_t gpu_id;
    double load;                        // aggregate GPU load across PEs in this group
    std::vector<int> peIds;             // indices into procs[]
  };

  std::vector<GPUGrp> gpuGroups;
  std::unordered_map<uint64_t, int> gpuIdToIdx;

  for (int pe = 0; pe < n_pes; pe++) {
    GreedyRefineCentralGPULB::GProc &p = procs[pe];
    if (!p.available) continue;
    uint64_t devId = stats->procs[pe].gpu_device_id;

    auto it = gpuIdToIdx.find(devId);
    if (it == gpuIdToIdx.end()) {
      gpuIdToIdx[devId] = gpuGroups.size();
      GPUGrp g;
      g.gpu_id = devId;
      g.load = p.bgload;                // background GPU load
      g.peIds.push_back(pe);
      gpuGroups.push_back(std::move(g));
    } else {
      gpuGroups[it->second].load += p.bgload;
      gpuGroups[it->second].peIds.push_back(pe);
    }
  }
  int nGroups = gpuGroups.size();

  if ((_lb_args.debug() > 1) && (CkMyPe() == cur_ld_balancer)) {
    CkPrintf("[%d] GreedyRefineCentralGPULB: %d GPU group(s), %d available PEs, %d migratable objs\n",
             CkMyPe(), nGroups, availablePes, (int)pobjs.size());
    for (auto &g : gpuGroups)
      CkPrintf("[%d]   GPU %llu: %d PEs, bgGpuLoad=%.6f\n",
               CkMyPe(), (unsigned long long)g.gpu_id, (int)g.peIds.size(), g.load);
  }

  // --- Greedy preprocessing at GPU-group level to establish target M ---
  double M = 0;
  {
    std::vector<double> grpLoad(nGroups);
    for (int gi = 0; gi < nGroups; gi++) grpLoad[gi] = gpuGroups[gi].load;

    for (int i = 0; i < (int)pobjs.size(); i++) {
      int lightest = 0;
      for (int gi = 1; gi < nGroups; gi++)
        if (grpLoad[gi] < grpLoad[lightest]) lightest = gi;
      grpLoad[lightest] += pobjs[i]->load;
      if (grpLoad[lightest] > M) M = grpLoad[lightest];
    }
    greedyMaxLoad = M;
  }
  M *= A;

  // Reset GPU group loads back to bg-only for the real assignment pass.
  for (int gi = 0; gi < nGroups; gi++) {
    gpuGroups[gi].load = 0;
    for (int pe : gpuGroups[gi].peIds)
      gpuGroups[gi].load += procs[pe].bgload;
  }

  // Reverse map: PE index -> GPU group index
  std::unordered_map<int, int> peToGrpIdx;
  for (int gi = 0; gi < nGroups; gi++)
    for (int pe : gpuGroups[gi].peIds)
      peToGrpIdx[pe] = gi;

  // Two dimensions are tracked separately. maxLoad follows the GPU-group
  // aggregate and keeps driving the refine target M below; maxCpuLoad follows
  // the busiest PE's CPU load. Only their combination is reported as the
  // solution objective -- see the note after the assignment loop.
  maxLoad = 0;
  for (int gi = 0; gi < nGroups; gi++)
    if (gpuGroups[gi].load > maxLoad) maxLoad = gpuGroups[gi].load;

  double maxCpuLoad = 0;
  for (int pe = 0; pe < n_pes; pe++)
    if (procs[pe].available && procs[pe].cpuLoad > maxCpuLoad)
      maxCpuLoad = procs[pe].cpuLoad;

  // --- Main assignment loop ---
  for (int i = 0; i < (int)pobjs.size(); i++) {
    const GreedyRefineCentralGPULB::GObj *obj = pobjs[i];
    double obj_gpu = obj->load;
    double obj_cpu = obj->cpuLoad;

    // ----- Outer: choose a GPU group by GPU load -----
    int lightest_gi = 0;
    for (int gi = 1; gi < nGroups; gi++)
      if (gpuGroups[gi].load < gpuGroups[lightest_gi].load) lightest_gi = gi;

    int chosen_gi = lightest_gi;
    if (obj->oldPE >= 0) {
      auto curIt = peToGrpIdx.find(obj->oldPE);
      if (curIt != peToGrpIdx.end()) {
        int cur_gi = curIt->second;
        GPUGrp &curGrp = gpuGroups[cur_gi];
        // Keep on current GPU group if it is within B tolerance of the lightest
        // group and adding this object keeps it under the target M.
        if ((curGrp.load <= (gpuGroups[lightest_gi].load + 0.01) * B) &&
            (curGrp.load + obj_gpu <= M))
          chosen_gi = cur_gi;
      }
    }

    GPUGrp &g = gpuGroups[chosen_gi];

    // ----- Inner: choose a PE within the group by CPU load -----
    int bestPe = g.peIds[0];
    for (int pe : g.peIds)
      if (procs[pe].cpuLoad < procs[bestPe].cpuLoad) bestPe = pe;

    // Keep the object on its old PE if that PE is in the chosen group and its
    // CPU load is within B tolerance of the lightest PE — reduces migrations.
    if (obj->oldPE >= 0 && peToGrpIdx.count(obj->oldPE) &&
        peToGrpIdx[obj->oldPE] == chosen_gi &&
        procs[obj->oldPE].cpuLoad <= (procs[bestPe].cpuLoad + 0.01) * B)
      bestPe = obj->oldPE;

    GreedyRefineCentralGPULB::GProc *p = &procs[bestPe];

    // Update GPU group aggregate (GPU load) and PE CPU load.
    g.load += obj_gpu;
    p->cpuLoad += obj_cpu / p->speed;

    if (g.load > maxLoad) {
      maxLoad = g.load;
      if (maxLoad > M) M = maxLoad;
    }
    if (p->cpuLoad > maxCpuLoad) maxCpuLoad = p->cpuLoad;

    if (bestPe != obj->oldPE) {
      nmoves++;
      stats->to_proc[obj->id] = bestPe;
    }
  }

  // A step is not over until both resources are done with it, and because
  // kernels are launched asynchronously the two overlap, so a group's step time
  // is set by whichever dimension is busier: its GPU or its busiest PE.
  // Reporting only the GPU term hides every intra-group difference -- with a
  // single GPU group that term is the same for every (A,B) candidate, so
  // receiveSolutions falls through to its fewest-migrations tie-break and picks
  // the do-nothing solution. That is why a one-GPU run never migrated anything.
  maxLoad = std::max(maxLoad, maxCpuLoad);

  if ((_lb_args.debug() > 1) && (CkMyPe() == cur_ld_balancer)) {
    CkPrintf("[%d] --- Per-GPU-group GPU load after LB ---\n", CkMyPe());
    for (int gi = 0; gi < nGroups; gi++)
      CkPrintf("[%d]   GPU %llu: aggregate gpuLoad=%.6f\n",
               CkMyPe(), (unsigned long long)gpuGroups[gi].gpu_id, gpuGroups[gi].load);
  }

#else
  // ---- Original PE-level greedy refine (non-GPU path, CPU only) ----

  double M = 0;
  if (B > 0) {
    M = greedyLB(pobjs, procHeap, stats);
    greedyMaxLoad = M;
    procHeap.addProcessors(procs, (maxLoad <= 0.001), false);
  }

  M *= A;
  for (int i=0; i < pobjs.size(); i++) {
    const GreedyRefineCentralGPULB::GObj *obj = pobjs[i];
    GreedyRefineCentralGPULB::GProc *llp = procHeap.top();
    GreedyRefineCentralGPULB::GProc *prevPe = NULL;
    if (obj->oldPE >= 0) prevPe = &(procs[obj->oldPE]);

    GreedyRefineCentralGPULB::GProc *p = llp;
    if (prevPe && (prevPe->load <= (llp->load+0.01)*B) && (prevPe->load + obj->load <= M) && (prevPe->available))
      p = prevPe;

    procHeap.remove(p);
    p->load += (obj->load / p->speed);
    procHeap.push(p);

    if (p->id != obj->oldPE) {
      nmoves++;
      stats->to_proc[obj->id] = p->id;
    }
    if (p->load > maxLoad) {
      maxLoad = p->load;
      if (maxLoad > M) M = maxLoad;
    }
  }
#endif
  // ----------------------------------------------

  if (concurrent) {
    sendSolution(maxLoad, nmoves);

#if __DEBUG_GREEDY_REFINE_GPU_
    CkCallback cb(CkReductionTarget(GreedyRefineCentralGPULB, receiveTotalTime), thisProxy[cur_ld_balancer]);
    contribute(sizeof(double), &strategyStartTime, CkReduction::sum_double, cb);
#endif
  } else if (_lb_args.debug() > 0) {
    double greedyRatio = 1.0;
    if (greedyMaxLoad > 0) greedyRatio = maxLoad / greedyMaxLoad;
    double migrationRatio = nmoves/double(pobjs.size());
    (void)greedyRatio; (void)migrationRatio;
  }
}

void GreedyRefineCentralGPULB::receiveTotalTime(double time)
{
  CkPrintf("Avg start time of GreedyRefineCentralGPULB strategy is %f\n", time / CkNumPes());
}

// decide which solution among all PEs is best and apply it
void GreedyRefineCentralGPULB::receiveSolutions(CkReductionMsg *msg)
{
  std::vector<GreedyRefineCentralGPULB::Solution> results(NUM_SOLUTIONS);

  int migrationsAllowed = totalObjs * migrationTolerance;
  // feasible solutions are those satistying user's migration constraint
  bool feasibleSolutions = false;
  float lowest_max_load = FLT_MAX;    // lowest max load of all solutions
  float lowest_max_load_f = FLT_MAX;  // lowest max load of feasible solution set
  float highest_max_load = 0;         // highest max load of all solutions
  int lowestMigrations = INT_MAX;     // lowest num migrations of all solutions
  const GreedyRefineCentralGPULB::Solution *bestSol = NULL; // best solution

  // first pass. Will record solution with lowest migrations as the best, in case
  // there is no feasible solution
  CkReduction::setElement *current = (CkReduction::setElement*)msg->getData();  // Get the first element in the set
  int numSolutions = 0;
  for ( ; current && (numSolutions < NUM_SOLUTIONS); current = current->next()) {
    PUP::fromMem pd(&current->data);
    pd|results[numSolutions]; // store result
    if (results[numSolutions].migrations >= 0) {  // valid result
      const GreedyRefineCentralGPULB::Solution &r = results[numSolutions++];
      if ((r.migrations <= migrationsAllowed) && (r.max_load < lowest_max_load_f)) {
        lowest_max_load_f = r.max_load;
        feasibleSolutions = true;
      }

      if ((r.migrations < lowestMigrations) ||
        ((r.migrations == lowestMigrations) && (r.max_load < bestSol->max_load))) {
        lowestMigrations = r.migrations;
        bestSol = &r;
      }

      if (r.max_load < lowest_max_load) lowest_max_load = r.max_load;
      if (r.max_load > highest_max_load) highest_max_load = r.max_load;
    }
  }
  results.resize(numSolutions); // for cases where CkNumPes() < NUM_SOLUTIONS
  CkAssert(numSolutions > 0);

  if (feasibleSolutions) {
    // second pass, get solution with low max load and migrations from feasible set
    int bestMigrations = INT_MAX;  // num migrations of best solution
    for (int i=0; i < results.size(); i++) {
      const GreedyRefineCentralGPULB::Solution &r = results[i];
      if ((r.migrations < bestMigrations && r.max_load <= lowest_max_load_f*LOAD_MIG_BAL) ||
          (r.migrations == bestMigrations && r.max_load < bestSol->max_load)) {
        bestMigrations = r.migrations;
        bestSol = &r;
      }
    }
  }
  // else: can't satisfy user migration constraint (for this lb step),
  // so just use solution with lowest num migrations

  if (_lb_args.debug() > 1) {
    CkPrintf("GreedyRefineCentralGPULB: Lowest max_load is %f, worst max_load is %f, lowest migrations=%d\n",
             lowest_max_load, highest_max_load, lowestMigrations);
    CkPrintf("GreedyRefineCentralGPULB: Got %d solutions at %f\nBest one is from PE %d with max_load=%f, migrations=%d\n",
             numSolutions, CkWallTimer(), bestSol->pe, bestSol->max_load, bestSol->migrations);
    float A, B;
    getGreedyRefineParams(bestSol->pe, A, B);
    CkPrintf("Best PE used params A=%f B=%f\n", A, B);
  }

  // notify PE that produced the best solution
  thisProxy[bestSol->pe].ApplyDecision();
}

#include "GreedyRefineCentralGPULB.def.h"

/*@}*/
