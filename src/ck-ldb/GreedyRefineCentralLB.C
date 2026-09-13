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

#include "charm++.h"
#include "ckgraph.h"
#include "GreedyRefineCentralLB.h"

#include <float.h>
#include <limits.h>
#include <algorithm>
#include <math.h>
#if CMK_CUDA || CMK_HIP
CkpvExtern(int, _lb_obj_index);
#include <unordered_map>
#endif

extern int quietModeRequested;

// a solution is feasible if num migrations <= user-specified limit
// LOAD_MIG_BAL is used to control tradeoff between maxload and migrations
// when selecting solutions from the feasible set
#define LOAD_MIG_BAL 1.003

using namespace std;

class GreedyRefineCentralLB::Solution {
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
class GreedyRefineCentralLB::PHeap {
public:
  PHeap(int numpes) {
    Q.reserve(numpes+1);
    Q.push_back(NULL);  // first element of the array is NULL
  }

  void addProcessors(std::vector<GreedyRefineCentralLB::GProc> &procs, bool bgLoadZero, bool insert=true) {
    for (int i=0; i < procs.size(); i++) {
      GreedyRefineCentralLB::GProc &p = procs[i];
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

  inline GreedyRefineCentralLB::GProc *top() const {
    CkAssert(Q.size() > 1);
    return Q[1];
  }

  inline void push(GreedyRefineCentralLB::GProc *p) {
    Q.push_back(p);
    p->pos = Q.size()-1;
    siftUp(p->pos);
  }

  inline GreedyRefineCentralLB::GProc *pop() {
    if (Q.size() == 1) return NULL;
    GreedyRefineCentralLB::GProc *retval;
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
  void remove(GreedyRefineCentralLB::GProc *p) {
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
    GreedyRefineCentralLB::GProc *t = Q[pos1];
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

  std::vector<GreedyRefineCentralLB::GProc*> Q;
};

CreateLBFunc_Def(GreedyRefineCentralLB, "Greedy refinement-based algorithm")

GreedyRefineCentralLB::GreedyRefineCentralLB(const CkLBOptions &opt): CBase_GreedyRefineCentralLB(opt), migrationTolerance(1.0)
{
  lbname = "GreedyRefineCentralLB";
  if ((CkMyPe() == 0) && !quietModeRequested)
    CkPrintf("CharmLB> GreedyRefineCentralLB created.\n");
  if (_lb_args.percentMovesAllowed() < 100) {
    migrationTolerance = float(_lb_args.percentMovesAllowed())/100.0;
  }
  concurrent = true;
}

GreedyRefineCentralLB::GreedyRefineCentralLB(CkMigrateMessage *m): CBase_GreedyRefineCentralLB(m), migrationTolerance(1.0) {
  lbname = "GreedyRefineCentralLB";
  if (_lb_args.percentMovesAllowed() < 100)
    migrationTolerance = float(_lb_args.percentMovesAllowed())/100.0;
  concurrent = true;
}

// ------------------------------------------------

// regular greedy lb algorithm
double GreedyRefineCentralLB::greedyLB(const std::vector<GreedyRefineCentralLB::GObj*> &pobjs,
              GreedyRefineCentralLB::PHeap &procHeap,
              const BaseLB::LDStats *stats) const
{
  double max_load = 0;
  int nmoves = 0;
  for (int i=0; i < pobjs.size(); i++) {
    const GreedyRefineCentralLB::GObj *obj = pobjs[i];
    GreedyRefineCentralLB::GProc *p = procHeap.pop();  // least loaded processor
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
#if __DEBUG_GREEDY_REFINE_
#include <fstream>
void GreedyRefineCentralLB::dumpObjLoads(std::vector<GreedyRefineCentralLB::GObj> &objs) {
  std::ofstream outfile("objloads.txt");
  outfile << objs.size() << std::endl;
  for (int i=0; i < objs.size(); i++) {
    GreedyRefineCentralLB::GObj &obj = objs[i];
    if ((i > 0) && (i % 100 == 0)) outfile << obj.load << std::endl;
    else outfile << obj.load << " ";
  }
  outfile.close();
}
void GreedyRefineCentralLB::dumpProcLoads(std::vector<GreedyRefineCentralLB::GProc> &procs) {
  std::ofstream outfile("proc_bg_loads.txt");
  outfile << procs.size() << std::endl;
  for (int i=0; i < procs.size(); i++) {
    GreedyRefineCentralLB::GProc &p = procs[i];
    if ((i > 0) && (i % 100 == 0)) outfile << p.load << std::endl;
    else outfile << p.load << " ";
  }
  outfile.close();
}
#endif

double GreedyRefineCentralLB::fillData(LDStats *stats,
                            std::vector<GreedyRefineCentralLB::GObj> &objs,
                            std::vector<GreedyRefineCentralLB::GObj*> &pobjs,
                            std::vector<GreedyRefineCentralLB::GProc> &procs,
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
    GreedyRefineCentralLB::GProc &p = procs[pe];
    p.id = pe;
    p.available = stats->procs[pe].available;
    p.speed = stats->procs[pe].pe_speed;
    if (p.available) {
      availablePes++;
      #if !(CMK_CUDA || CMK_HIP)
        p.bgload = stats->procs[pe].bg_walltime;
        if (p.bgload > maxBGLoad) maxBGLoad = p.bgload;
      #else
        p.bgload = 0.0;
      #endif

      #if (CMK_CUDA || CMK_HIP)
        p.bg_walltime = stats->procs[pe].bg_walltime;
        // CmiPrintf("[%d] settign bg_walltime to %f\n", pe, p.bg_walltime);
      #endif
      if (_lb_args.debug() > 1) {
        double &speed = stats->procs[pe].pe_speed;
        if (speed < minSpeed) minSpeed = speed;
        if (speed > maxSpeed) maxSpeed = speed;
        avgSpeed += speed;
      }
    }
  }
  if (!availablePes) CkAbort("GreedyRefineCentralLB: No available processors\n");

  for (int i=0; i < n_objs; i++) {
    LDObjData &oData = stats->objData[i];
    GreedyRefineCentralLB::GObj &obj = objs[i];
    int pe = stats->from_proc[i];
    obj.id = i;
    obj.oldPE = pe;
#if (CMK_CUDA || CMK_HIP) && CMK_LB_USER_DATA
    // LDObjData::gpuPupSize and the GPU allocation size registered by
    // CentralLB::initLB only exist in GPU builds with LB user data.
    obj.gpuPupSize = oData.gpuPupSize;
    obj.gpuAllocSize = *(size_t *)oData.getUserData(CkpvAccess(_lb_obj_index));
#else
    obj.gpuPupSize = 0;
    obj.gpuAllocSize = 0;
#endif
    CkAssert(pe >= 0 && pe <= n_pes);
    if (pe == n_pes) obj.oldPE = -1; // this can happen in HybridLB if object comes from outside group. mark oldPE as -1 in this situation
    if (!oData.migratable) {
      CkAssert(pe < n_pes);
      unmigratableObjs++;
      GreedyRefineCentralLB::GProc &p = procs[pe];
      if (!p.available)
        CkAbort("GreedyRefineCentralLB: nonmigratable object on unavailable processor\n");
#if CMK_CUDA || CMK_HIP
      double nmObjLoad = oData.gpuTime;
#else
      double nmObjLoad = oData.wallTime;
#endif
      p.bgload += nmObjLoad; // take non-migratable object load as background load
      //is the non migratable obj load correct
      CkPrintf("[%d] Obj %d on PE %d is non-migratable, load=%.6f\n", CkMyPe(), i, pe, nmObjLoad);
      if (p.bgload > maxBGLoad) maxBGLoad = p.bgload;
    } else {
#if CMK_CUDA || CMK_HIP
      obj.load = oData.gpuTime * stats->procs[pe].pe_speed;
#else
      obj.load = oData.wallTime * stats->procs[pe].pe_speed;
#endif
        // CkPrintf("[%d] Obj %d on PE %d is migratable, load=%.6f, GPU pup size=%ld, GPU alloc size=%ld\n", CkMyPe(), i, pe, obj.load, oData.gpuPupSize, obj.gpuAllocSize);
      pobjs.push_back(&obj);
      totalObjLoad += obj.load;
      if (_lb_args.debug() > 1) {
        if (obj.load < minOload) minOload = obj.load;
        if (obj.load > maxOload) maxOload = obj.load;
#if CMK_CUDA || CMK_HIP
        // CkPrintf("[%d] Obj %d (PE %d): wallTime=%.6f gpuTime=%.6f effectiveLoad=%.6f\n",
        //          CkMyPe(), i, pe, oData.wallTime, oData.gpuTime, obj.load);
#endif
      }
    }
  }

  procHeap.addProcessors(procs, (maxBGLoad <= 0.001), true);

  // ---- print some stats ----
  // CkPrintf("here\n")
  if ((_lb_args.debug() > 1) && (!concurrent || (CkMyPe() == cur_ld_balancer))) {
    for (int pe=0; pe < n_pes; pe++) {
      GreedyRefineCentralLB::GProc &p = procs[pe];
      if (!p.available) continue;
      if (p.bgload < minBGLoad) minBGLoad = p.bgload;
      avgBGLoad += p.bgload;
    }
    CkPrintf("[%d] GreedyRefineCentralLB: num pes=%d, num objs=%d\n", CkMyPe(), n_pes, n_objs);
    CkPrintf("[%d] Unavailable processors=%d, Unmigratable objs=%d\n", CkMyPe(), n_pes - availablePes, unmigratableObjs);
    CkPrintf("[%d] min_bgload=%f mean_bgload=%f max_bgload=%f\n", CkMyPe(), minBGLoad, (avgBGLoad / availablePes), maxBGLoad);
    CkPrintf("[%d] min_oload=%f mean_oload=%f max_oload=%f\n", CkMyPe(), minOload, (totalObjLoad / (n_objs - unmigratableObjs)), maxOload);
    CkPrintf("[%d] min_speed=%f mean_speed=%f max_speed=%f\n", CkMyPe(), minSpeed, (avgSpeed / availablePes), maxSpeed);

    double maxLoad = 0;
    double minLoad = FLT_MAX;
    std::vector<double> ploads(n_pes, -1);
    for (int i=0; i < n_objs; i++) {
      GreedyRefineCentralLB::GObj &o = objs[i];
      int pe = o.oldPE;
      if (pe < 0) continue;
      if (ploads[pe] < 0) ploads[pe] = procs[pe].bgload;
      if (stats->objData[i].migratable)  // load for this object is already counted if !migratable
        ploads[pe] += o.load;
      if (ploads[pe] > maxLoad) maxLoad = ploads[pe];
      if (ploads[pe] < minLoad) minLoad = ploads[pe];
    }
    CkPrintf("[%d] maxload with current map=%f\n", CkMyPe(), maxLoad);
    CkPrintf("[%d] minload with current map=%f\n", CkMyPe(), minLoad);

    // CkPrintf("[%d] --- Per-PE loads before LB ---\n", CkMyPe());
    // for (int pe=0; pe < n_pes; pe++) {
    //   if (ploads[pe] >= 0)
    //     CkPrintf("[%d]   PE %d: totalLoad=%.6f bgLoad=%.6f\n",
    //              CkMyPe(), pe, ploads[pe], procs[pe].bgload);
    // }

    //CkPrintf("[%d] %f : Filled proc and obj stats\n", CkMyPe(), CkWallTimer() - strategyStartTime);
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

void GreedyRefineCentralLB::sendSolution(double maxLoad, int migrations)
{
  // gather results in central PE, who will decide which solution is the best
  // only the objective values of the solutions are sent, not the whole solutions

  GreedyRefineCentralLB::Solution sol(CkMyPe(), maxLoad, migrations);
  size_t buf_size = sizeof(GreedyRefineCentralLB::Solution);
  void *buffer = malloc(buf_size);
  PUP::toMem pd(buffer);
  pd|sol;

  CkCallback cb(CkIndex_GreedyRefineCentralLB::receiveSolutions((CkReductionMsg*)NULL), thisProxy[cur_ld_balancer]);
  contribute(buf_size, buffer, CkReduction::set, cb);

  if ((_lb_args.debug() > 1) && (CkMyPe() == cur_ld_balancer)) {
    CkPrintf("[%d] %f : Called gather/reduction\n", CkMyPe(), CkWallTimer() - strategyStartTime);
  }

  free(buffer);
}

void GreedyRefineCentralLB::work(LDStats *stats)
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

  std::vector<GreedyRefineCentralLB::GObj> objs(totalObjs);
  // will sort pobjs instead of objs (faster swapping). will only contain pointers
  // to migratable objects
  std::vector<GreedyRefineCentralLB::GObj*> pobjs;
  pobjs.reserve(totalObjs);

  std::vector<GreedyRefineCentralLB::GProc> procs(n_pes);
  PHeap procHeap(n_pes);

  // fill data structures used by algorithm
  double maxLoad = fillData(stats, objs, pobjs, procs, procHeap);

  // ------------ apply greedy refine algorithm --------------

  std::sort(pobjs.begin(), pobjs.end(), GreedyRefineCentralLB::ObjLoadGreater());

  int nmoves = 0;
  double greedyMaxLoad = 0;

#if CMK_CUDA || CMK_HIP
  // ---- GPU-aware path: balance at GPU-group level ----
  //
  // Group PEs by gpu_device_id.  M tracks the max *GPU-group* aggregate load.
  // greedyLB preprocessing computes M at GPU-group level.
  // Main loop: pop lightest GPU group, assign object to lightest PE in that group.

  // --- Build GPU groups from the per-PE procs vector ---

  struct GPUGrp {
    uint64_t gpu_id;
    double load;                        // aggregate load across PEs in this group
    std::vector<int> peIds;             // indices into procs[]
    size_t gpu_mem_remaining;
    size_t pool_buff_mem_remaining;
  };

  std::vector<GPUGrp> gpuGroups;
  std::unordered_map<uint64_t, int> gpuIdToIdx;

  for (int pe = 0; pe < n_pes; pe++) {
    GreedyRefineCentralLB::GProc &p = procs[pe];
    if (!p.available) continue;
    uint64_t devId = stats->procs[pe].gpu_device_id;
    size_t gpu_mem_remaining = stats->procs[pe].gpu_mem_remaining;
    size_t pool_buff_mem_remaining = stats->procs[pe].pool_buff_mem_remaining;
    // printf("pe gpu_id %ld\n", devId);

    auto it = gpuIdToIdx.find(devId);
    if (it == gpuIdToIdx.end()) {
      gpuIdToIdx[devId] = gpuGroups.size();
      GPUGrp g;
      g.gpu_id = devId;
      g.load = p.bgload;
      g.peIds.push_back(pe);
      g.gpu_mem_remaining = gpu_mem_remaining;
      g.pool_buff_mem_remaining = pool_buff_mem_remaining;
      gpuGroups.push_back(std::move(g));
    } else {
      gpuGroups[it->second].load += p.bgload;
      gpuGroups[it->second].peIds.push_back(pe);
    }
  }
  int nGroups = gpuGroups.size();

  // CkPrintf("[%d] GreedyRefineCentralLB: %d GPU group(s), %d available PEs, %d migratable objs\n",
  //          CkMyPe(), nGroups, availablePes, (int)pobjs.size());
  // for (auto &g : gpuGroups)
  //   CkPrintf("[%d]   GPU %llu: %d PEs, bgload=%.6f, gpu_mem_remaining=%ld, pool_buff_mem_remaining=%ld\n",
  //            CkMyPe(), g.gpu_id, (int)g.peIds.size(), g.load, g.gpu_mem_remaining, g.pool_buff_mem_remaining);

  // --- Greedy preprocessing at GPU-group level to establish target M ---
  // Reset group loads to bg only, then greedily assign objects to get M.
  double M = 0;
  {
    // Save a copy of group bg loads
    std::vector<double> grpLoad(nGroups);
    for (int gi = 0; gi < nGroups; gi++) grpLoad[gi] = gpuGroups[gi].load;

    for (int i = 0; i < (int)pobjs.size(); i++) {
      // Find lightest GPU group
      int lightest = 0;
      for (int gi = 1; gi < nGroups; gi++) {
        if (grpLoad[gi] < grpLoad[lightest]) lightest = gi;
      }
      grpLoad[lightest] += pobjs[i]->load;
      if (grpLoad[lightest] > M) M = grpLoad[lightest];
    }
    greedyMaxLoad = M;
  }
  M *= A;
  // CkPrintf("M is %f\n", M);

  // Reset GPU group loads back to bg-only for the real assignment pass
  for (int gi = 0; gi < nGroups; gi++) {
    gpuGroups[gi].load = 0;
    for (int pe : gpuGroups[gi].peIds)
      gpuGroups[gi].load += procs[pe].bgload;
  }
  // Also reset per-PE loads in procHeap to bgload
  procHeap.addProcessors(procs, (maxLoad <= 0.001), false);

  // if ((_lb_args.debug() > 0) && (CkMyPe() == cur_ld_balancer))
  //   CkPrintf("[%d] GPU greedy-refine: M(target)=%.6f, A=%.3f, B=%.3f\n", CkMyPe(), M, A, B);

  // Reverse map: PE index -> GPU group index
  std::unordered_map<int, int> peToGrpIdx;
  for (int gi = 0; gi < nGroups; gi++)
    for (int pe : gpuGroups[gi].peIds)
      peToGrpIdx[pe] = gi;

  for (int i = 0; i < (int)pobjs.size(); i++) {
    const GreedyRefineCentralLB::GObj *obj = pobjs[i];
    double obj_load = obj->load;

    int lightest_gi = 0;
    for (int gi = 1; gi < nGroups; gi++) {
      if (gpuGroups[gi].load < gpuGroups[lightest_gi].load)
        lightest_gi = gi;
    }

    int src_gi = -1;
    if (obj->oldPE >= 0) {
      auto srcIt = peToGrpIdx.find(obj->oldPE);
      if (srcIt != peToGrpIdx.end()) src_gi = srcIt->second;
    }

    // Refinement: if object's current GPU group is close enough, keep it there
    int chosen_gi = lightest_gi;
    if (src_gi >= 0) {
      GPUGrp &curGrp = gpuGroups[src_gi];
      if ((curGrp.load <= (gpuGroups[lightest_gi].load + 0.01) * B) && (curGrp.load + obj_load <= M))
            chosen_gi = src_gi;
    }

    // Pool buffer constraint
    if (chosen_gi != src_gi && src_gi >= 0 && obj->gpuPupSize > 0) {
      if (gpuGroups[src_gi].pool_buff_mem_remaining < obj->gpuPupSize || gpuGroups[chosen_gi].pool_buff_mem_remaining < obj->gpuPupSize)
        chosen_gi = src_gi;

      if((size_t)(0.95 * gpuGroups[chosen_gi].gpu_mem_remaining) <  obj->gpuAllocSize )//95% of the rest of the memory can be filled
        chosen_gi = src_gi;
    }

    GPUGrp &g = gpuGroups[chosen_gi];

    int bestPe = g.peIds[0];

    //find the PE with the least walltime
    for(int pe : g.peIds) {
      if(procs[pe].load < procs[bestPe].load) {
        bestPe = pe;
      }
    }

    if(obj->oldPE >= 0 && peToGrpIdx[obj->oldPE] == chosen_gi)
      bestPe = obj->oldPE;
      
    GreedyRefineCentralLB::GProc *p = &procs[bestPe];
    double scaled = obj->load / p->speed;

    // Update PE load
    procHeap.remove(p);
    p->load += scaled;
    procHeap.push(p);

    // Update GPU group aggregate
    g.load += scaled;

    if (chosen_gi != src_gi && src_gi >= 0 && obj->gpuPupSize > 0)
    {
      gpuGroups[src_gi].pool_buff_mem_remaining -= obj->gpuPupSize;
      gpuGroups[chosen_gi].pool_buff_mem_remaining -= obj->gpuPupSize;
      gpuGroups[chosen_gi].gpu_mem_remaining-= obj->gpuAllocSize;
    }

    // Track max GPU-group load; expand M if exceeded
    if (g.load > maxLoad) {
      maxLoad = g.load;
      if (maxLoad > M) M = maxLoad;
    }

    // Record migration if PE changed
    if (bestPe != obj->oldPE) {
      nmoves++;
      stats->to_proc[obj->id] = bestPe;
      // if (_lb_args.debug() > 2)
        // CkPrintf("[%d] Migrating obj %d: PE %d -> PE %d (GPU %d, objLoad=%.6f, gpuGrpLoad=%.6f)\n",
        //          CkMyPe(), obj->id, obj->oldPE, bestPe, g.gpu_id, obj_load, g.load);
    }
  }

  // Print per-GPU-group loads after LB
  CkPrintf("[%d] --- Per-GPU-group loads after LB ---\n", CkMyPe());
  for (int gi = 0; gi < nGroups; gi++)
    CkPrintf("[%d]   GPU %llu: aggregate load=%.6f\n",
             CkMyPe(), gpuGroups[gi].gpu_id, gpuGroups[gi].load);

#else
  // ---- Original PE-level greedy refine (non-GPU path) ----

  double M = 0;
  if (B > 0) {
    M = greedyLB(pobjs, procHeap, stats);
    greedyMaxLoad = M;
    procHeap.addProcessors(procs, (maxLoad <= 0.001), false);
  }

  M *= A;
  // if ((_lb_args.debug() > 1) && (CkMyPe() == cur_ld_balancer)) {
  //   CkPrintf("maxLoad=%f totalObjLoad=%f M=%f A=%f B=%f\n", maxLoad, totalObjLoad, M, A, B);
  // }
  for (int i=0; i < pobjs.size(); i++) {
    const GreedyRefineCentralLB::GObj *obj = pobjs[i];
    GreedyRefineCentralLB::GProc *llp = procHeap.top();
    GreedyRefineCentralLB::GProc *prevPe = NULL;
    if (obj->oldPE >= 0) prevPe = &(procs[obj->oldPE]);

    GreedyRefineCentralLB::GProc *p = llp;
    if (prevPe && (prevPe->load <= (llp->load+0.01)*B) && (prevPe->load + obj->load <= M) && (prevPe->available))
      p = prevPe;

    procHeap.remove(p);
    p->load += (obj->load / p->speed);
    procHeap.push(p);

    // if (p->id != obj->oldPE) {
    //   nmoves++;
    //   stats->to_proc[obj->id] = p->id;
    //   if (_lb_args.debug() > 1) {
    //     CkPrintf("[%d] Migrating obj %d: PE %d -> PE %d (objLoad=%.6f, destPELoad=%.6f)\n",
    //              CkMyPe(), obj->id, obj->oldPE, p->id, obj->load, p->load);
    //   }
    // }
    if (p->load > maxLoad) {
      maxLoad = p->load;
      if (maxLoad > M) M = maxLoad;
    }
  }
#endif
  // ----------------------------------------------
  // if (_lb_args.debug() > 1 && (!concurrent || (CkMyPe() == cur_ld_balancer))) {
  //   CkPrintf("[%d] --- Per-PE loads after LB ---\n", CkMyPe());
  //   for (int pe=0; pe < n_pes; pe++) {
  //     GreedyRefineCentralLB::GProc &p = procs[pe];
  //     if (p.available)
  //       CkPrintf("[%d]   PE %d: totalLoad=%.6f bgLoad=%.6f\n",
  //                CkMyPe(), pe, p.load, p.bgload);
  //   }
  //   CkPrintf("[%d] After LB: max_load=%.6f, migrations=%d/%d (%.2f%%)\n",
  //            CkMyPe(), maxLoad, nmoves, (int)pobjs.size(),
  //            100.0 * nmoves / double(pobjs.size()));
  // }

  if (concurrent) {

    sendSolution(maxLoad, nmoves);

#if __DEBUG_GREEDY_REFINE_
    CkCallback cb(CkReductionTarget(GreedyRefineCentralLB, receiveTotalTime), thisProxy[cur_ld_balancer]);
    contribute(sizeof(double), &strategyStartTime, CkReduction::sum_double, cb);
#endif
  } else if (_lb_args.debug() > 0) {
    double greedyRatio = 1.0;
    if (greedyMaxLoad > 0) greedyRatio = maxLoad / greedyMaxLoad;
    double migrationRatio = nmoves/double(pobjs.size());
    // if ((greedyRatio > 1.03) && (migrationRatio < migrationTolerance)) {
    //   CkPrintf("[%d] GreedyRefine: WARNING - migration ratio is %.3f (within user-specified tolerance).\n"
    //            "but maxload after lb is %f higher than greedy. Consider testing with A=0, B=-1\n",
    //            CkMyPe(), migrationRatio, greedyRatio);
    // }
    // CkPrintf("[%d] GreedyRefineCentralLB: after lb, max_load=%.3f, migrations=%d(%.2f%%), ratioToGreedy=%.3f\n",
    //          CkMyPe(), maxLoad, nmoves, 100.0*migrationRatio, greedyRatio);
  }
}

void GreedyRefineCentralLB::receiveTotalTime(double time)
{
  CkPrintf("Avg start time of GreedyRefineCentralLB strategy is %f\n", time / CkNumPes());
}

// decide which solution among all PEs is best and apply it
void GreedyRefineCentralLB::receiveSolutions(CkReductionMsg *msg)
{
  std::vector<GreedyRefineCentralLB::Solution> results(NUM_SOLUTIONS);

  int migrationsAllowed = totalObjs * migrationTolerance;
  ckout<<"migrations allowed "<<migrationsAllowed<<" out of "<<totalObjs<<" total objs"<<endl;
  // feasible solutions are those satistying user's migration constraint
  bool feasibleSolutions = false;
  float lowest_max_load = FLT_MAX;    // lowest max load of all solutions
  float lowest_max_load_f = FLT_MAX;  // lowest max load of feasible solution set
  float highest_max_load = 0;         // highest max load of all solutions
  int lowestMigrations = INT_MAX;     // lowest num migrations of all solutions
  const GreedyRefineCentralLB::Solution *bestSol = NULL; // best solution

  // first pass. Will record solution with lowest migrations as the best, in case
  // there is no feasible solution
  CkReduction::setElement *current = (CkReduction::setElement*)msg->getData();  // Get the first element in the set
  int numSolutions = 0;
  for ( ; current && (numSolutions < NUM_SOLUTIONS); current = current->next()) {
    PUP::fromMem pd(&current->data);
    pd|results[numSolutions]; // store result
    if (results[numSolutions].migrations >= 0) {  // valid result
      const GreedyRefineCentralLB::Solution &r = results[numSolutions++];
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
      const GreedyRefineCentralLB::Solution &r = results[i];
      // Select if we find (fewer migrations and load within tolerance) or
      // (same as lowest migration and better load).  Since we know a feasible
      // solution exists and we only minimize here, we guarantee that we'll end
      // with a feasible solution.
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
    CkPrintf("GreedyRefineCentralLB: Lowest max_load is %f, worst max_load is %f, lowest migrations=%d\n",
             lowest_max_load, highest_max_load, lowestMigrations);

    CkPrintf("GreedyRefineCentralLB: Got %d solutions at %f\nBest one is from PE %d with max_load=%f, migrations=%d\n",
             numSolutions, CkWallTimer(), bestSol->pe, bestSol->max_load, bestSol->migrations);
    float A, B;
    getGreedyRefineParams(bestSol->pe, A, B);
    CkPrintf("Best PE used params A=%f B=%f\n", A, B);
  }

  // notify PE that produced the best solution
  thisProxy[bestSol->pe].ApplyDecision();
}

#include "GreedyRefineCentralLB.def.h"

/*@}*/