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
#include "GreedyRefineLB.h"

#include <float.h>
#include <limits.h>
#include <algorithm>
#include <math.h>

extern int quietModeRequested;

// a solution is feasible if num migrations <= user-specified limit
// LOAD_MIG_BAL is used to control tradeoff between maxload and migrations
// when selecting solutions from the feasible set
#define LOAD_MIG_BAL 1.003

using namespace std;

class GreedyRefineLB::Solution {
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
class GreedyRefineLB::PHeap {
public:
  PHeap(int numpes) {
    Q.reserve(numpes+1);
    Q.push_back(NULL);  // first element of the array is NULL
  }

  void addProcessors(std::vector<GreedyRefineLB::GProc> &procs, bool bgLoadZero, bool insert=true) {
    for (int i=0; i < procs.size(); i++) {
      GreedyRefineLB::GProc &p = procs[i];
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

  inline GreedyRefineLB::GProc *top() const {
    CkAssert(Q.size() > 1);
    return Q[1];
  }

  inline void push(GreedyRefineLB::GProc *p) {
    Q.push_back(p);
    p->pos = Q.size()-1;
    siftUp(p->pos);
  }

  inline GreedyRefineLB::GProc *pop() {
    if (Q.size() == 1) return NULL;
    GreedyRefineLB::GProc *retval;
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
  void remove(GreedyRefineLB::GProc *p) {
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
    GreedyRefineLB::GProc *t = Q[pos1];
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

  std::vector<GreedyRefineLB::GProc*> Q;
};

CreateLBFunc_Def(GreedyRefineLB, "Greedy refinement-based algorithm")

GreedyRefineLB::GreedyRefineLB(const CkLBOptions &opt): CBase_GreedyRefineLB(opt), migrationTolerance(1.0)
{
  lbname = "GreedyRefineLB";
  if ((CkMyPe() == 0) && !quietModeRequested)
    CkPrintf("CharmLB> GreedyRefineLB created.\n");
  if (_lb_args.percentMovesAllowed() < 100) {
    migrationTolerance = float(_lb_args.percentMovesAllowed())/100.0;
  }
  concurrent = true;
}

GreedyRefineLB::GreedyRefineLB(CkMigrateMessage *m): CBase_GreedyRefineLB(m), migrationTolerance(1.0) {
  lbname = "GreedyRefineLB";
  if (_lb_args.percentMovesAllowed() < 100)
    migrationTolerance = float(_lb_args.percentMovesAllowed())/100.0;
  concurrent = true;
}

// ------------------------------------------------

// regular greedy lb algorithm
double GreedyRefineLB::greedyLB(const std::vector<GreedyRefineLB::GObj*> &pobjs,
              GreedyRefineLB::PHeap &procHeap,
              const BaseLB::LDStats *stats) const
{
  double max_load = 0;
  int nmoves = 0;
  for (int i=0; i < pobjs.size(); i++) {
    const GreedyRefineLB::GObj *obj = pobjs[i];
    GreedyRefineLB::GProc *p = procHeap.pop();  // least loaded processor
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
void GreedyRefineLB::dumpObjLoads(std::vector<GreedyRefineLB::GObj> &objs) {
  std::ofstream outfile("objloads.txt");
  outfile << objs.size() << std::endl;
  for (int i=0; i < objs.size(); i++) {
    GreedyRefineLB::GObj &obj = objs[i];
    if ((i > 0) && (i % 100 == 0)) outfile << obj.load << std::endl;
    else outfile << obj.load << " ";
  }
  outfile.close();
}
void GreedyRefineLB::dumpProcLoads(std::vector<GreedyRefineLB::GProc> &procs) {
  std::ofstream outfile("proc_bg_loads.txt");
  outfile << procs.size() << std::endl;
  for (int i=0; i < procs.size(); i++) {
    GreedyRefineLB::GProc &p = procs[i];
    if ((i > 0) && (i % 100 == 0)) outfile << p.load << std::endl;
    else outfile << p.load << " ";
  }
  outfile.close();
}
#endif

double GreedyRefineLB::fillData(LDStats *stats,
                            std::vector<GreedyRefineLB::GObj> &objs,
                            std::vector<GreedyRefineLB::GObj*> &pobjs,
                            std::vector<GreedyRefineLB::GProc> &procs,
                            PHeap &procHeap)
{
  const int n_pes = stats->nprocs();
  const int n_objs = stats->n_objs;
  // most of these variables are just for printing stats when _lb_args.debug()
  int unmigratableObjs = 0;
  availablePes = 0; totalObjLoad = 0;
  double minBGLoad = DBL_MAX; double avgBGLoad = 0; double maxBGLoad = 0;
  double minSpeed  = DBL_MAX; double maxSpeed  = 0; double avgSpeed  = 0;
  double minOload  = DBL_MAX; double maxOload  = 0;

  for (int pe=0; pe < n_pes; pe++) {
    GreedyRefineLB::GProc &p = procs[pe];
    p.id = pe;
    p.available = stats->procs[pe].available;
    p.speed = stats->procs[pe].pe_speed;
    if (p.available) {
      availablePes++;
      p.bgload = stats->procs[pe].bg_walltime;
      if (p.bgload > maxBGLoad) maxBGLoad = p.bgload;
      if (_lb_args.debug() > 1) {
        double &speed = stats->procs[pe].pe_speed;
        if (speed < minSpeed) minSpeed = speed;
        if (speed > maxSpeed) maxSpeed = speed;
        avgSpeed += speed;
      }
    }
  }
  if (!availablePes) CkAbort("GreedyRefineLB: No available processors\n");

  for (int i=0; i < n_objs; i++) {
    LDObjData &oData = stats->objData[i];
    GreedyRefineLB::GObj &obj = objs[i];
    int pe = stats->from_proc[i];
    obj.id = i;
    obj.oldPE = pe;
    CkAssert(pe >= 0 && pe <= n_pes);
    if (pe == n_pes) obj.oldPE = -1; // this can happen in HybridLB if object comes from outside group. mark oldPE as -1 in this situation
    if (!oData.migratable) {
      CkAssert(pe < n_pes);
      unmigratableObjs++;
      GreedyRefineLB::GProc &p = procs[pe];
      if (!p.available)
        CkAbort("GreedyRefineLB: nonmigratable object on unavailable processor\n");
      p.bgload += oData.wallTime; // take non-migratable object load as background load
      if (p.bgload > maxBGLoad) maxBGLoad = p.bgload;
    } else {
      obj.load = oData.wallTime * stats->procs[pe].pe_speed;
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
      GreedyRefineLB::GProc &p = procs[pe];
      if (!p.available) continue;
      if (p.bgload < minBGLoad) minBGLoad = p.bgload;
      avgBGLoad += p.bgload;
    }
    CkPrintf("[%d] GreedyRefineLB: num pes=%d, num objs=%d\n", CkMyPe(), n_pes, n_objs);
    CkPrintf("[%d] Unavailable processors=%d, Unmigratable objs=%d\n", CkMyPe(), n_pes - availablePes, unmigratableObjs);
    CkPrintf("[%d] min_bgload=%f mean_bgload=%f max_bgload=%f\n", CkMyPe(), minBGLoad, (avgBGLoad / availablePes), maxBGLoad);
    CkPrintf("[%d] min_oload=%f mean_oload=%f max_oload=%f\n", CkMyPe(), minOload, (totalObjLoad / (n_objs - unmigratableObjs)), maxOload);
    CkPrintf("[%d] min_speed=%f mean_speed=%f max_speed=%f\n", CkMyPe(), minSpeed, (avgSpeed / availablePes), maxSpeed);

    double maxLoad = 0;
    std::vector<double> ploads(n_pes, -1);
    for (int i=0; i < n_objs; i++) {
      GreedyRefineLB::GObj &o = objs[i];
      int pe = o.oldPE;
      if (pe < 0) continue;
      if (ploads[pe] < 0) ploads[pe] = procs[pe].bgload;
      if (stats->objData[i].migratable)  // load for this object is already counted if !migratable
        ploads[pe] += o.load;
      if (ploads[pe] > maxLoad) maxLoad = ploads[pe];
    }
    CkPrintf("[%d] maxload with current map=%f\n", CkMyPe(), maxLoad);

    //CkPrintf("[%d] %f : Filled proc and obj stats\n", CkMyPe(), CkWallTimer() - strategyStartTime);
  }

  return maxBGLoad;
}

static const float Avals[] = {1.0, 1.005, 1.01, 1.015, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.16, 1.20, 1.30};
static const float Bvals[] = {1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, FLT_MAX};
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

void GreedyRefineLB::sendSolution(double maxLoad, int migrations)
{
  // gather results in central PE, who will decide which solution is the best
  // only the objective values of the solutions are sent, not the whole solutions

  GreedyRefineLB::Solution sol(CkMyPe(), maxLoad, migrations);
  size_t buf_size = sizeof(GreedyRefineLB::Solution);
  void *buffer = malloc(buf_size);
  PUP::toMem pd(buffer);
  pd|sol;

  CkCallback cb(CkIndex_GreedyRefineLB::receiveSolutions((CkReductionMsg*)NULL), thisProxy[cur_ld_balancer]);
  contribute(buf_size, buffer, CkReduction::set, cb);

  if ((_lb_args.debug() > 1) && (CkMyPe() == cur_ld_balancer)) {
    CkPrintf("[%d] %f : Called gather/reduction\n", CkMyPe(), CkWallTimer() - strategyStartTime);
  }

  free(buffer);
}

void GreedyRefineLB::work(LDStats *stats)
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
  totalObjs = stats->n_objs;

  std::vector<GreedyRefineLB::GObj> objs(totalObjs);
  // will sort pobjs instead of objs (faster swapping). will only contain pointers
  // to migratable objects
  std::vector<GreedyRefineLB::GObj*> pobjs;
  pobjs.reserve(totalObjs);

  std::vector<GreedyRefineLB::GProc> procs(n_pes);
  PHeap procHeap(n_pes);

  // fill data structures used by algorithm
  double maxLoad = fillData(stats, objs, pobjs, procs, procHeap);

  // ------------ apply greedy refine algorithm --------------

  std::sort(pobjs.begin(), pobjs.end(), GreedyRefineLB::ObjLoadGreater());

  double M = 0, greedyMaxLoad = 0;
  if (B > 0) {
    // greedy preprocessing: tells me what max_load to aim for
    M = greedyLB(pobjs, procHeap, stats);
    greedyMaxLoad = M;
    procHeap.addProcessors(procs, (maxLoad <= 0.001), false);
  }

  // maxLoad at this point is based only on bg load
  int nmoves = 0;
  /*
  // this would be in case I decide to drop greedyLB preprocessing, but the preprocessing
  // seems useful and doesn't add much time
  //if (M <= 0) M = std::max(totalObjLoad/availablePes, maxLoad) * A;
  else*/ M *= A;
  if ((_lb_args.debug() > 1) && (CkMyPe() == cur_ld_balancer)) {
    CkPrintf("maxLoad=%f totalObjLoad=%f M=%f A=%f B=%f\n", maxLoad, totalObjLoad, M, A, B);
  }
  for (int i=0; i < pobjs.size(); i++) {
    const GreedyRefineLB::GObj *obj = pobjs[i];
    GreedyRefineLB::GProc *llp = procHeap.top();          // least loaded processor
    GreedyRefineLB::GProc *prevPe = NULL;
    if (obj->oldPE >= 0) prevPe = &(procs[obj->oldPE]); // current processor

    // choose processor
    GreedyRefineLB::GProc *p = llp;
    if (prevPe && (prevPe->load <= (llp->load+0.01)*B) && (prevPe->load + obj->load <= M) && (prevPe->available))
      p = prevPe;  // use same PE

    // update processor load
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
  // ----------------------------------------------

  if (concurrent) {

    sendSolution(maxLoad, nmoves);

#if __DEBUG_GREEDY_REFINE_
    CkCallback cb(CkReductionTarget(GreedyRefineLB, receiveTotalTime), thisProxy[cur_ld_balancer]);
    contribute(sizeof(double), &strategyStartTime, CkReduction::sum_double, cb);
#endif
  } else if (_lb_args.debug() > 0) {
    double greedyRatio = 1.0;
    if (B > 0) greedyRatio = maxLoad / greedyMaxLoad;
    double migrationRatio = nmoves/double(pobjs.size());
    if ((greedyRatio > 1.03) && (migrationRatio < migrationTolerance)) {
      CkPrintf("[%d] GreedyRefine: WARNING - migration ratio is %.3f (within user-specified tolerance).\n"
               "but maxload after lb is %f higher than greedy. Consider testing with A=0, B=-1\n",
               CkMyPe(), migrationRatio, greedyRatio);
    }
    CkPrintf("[%d] GreedyRefineLB: after lb, max_load=%.3f, migrations=%d(%.2f%%), ratioToGreedy=%.3f\n",
             CkMyPe(), maxLoad, nmoves, 100.0*migrationRatio, greedyRatio);
  }
}

void GreedyRefineLB::receiveTotalTime(double time)
{
  CkPrintf("Avg start time of GreedyRefineLB strategy is %f\n", time / CkNumPes());
}

// decide which solution among all PEs is best and apply it
void GreedyRefineLB::receiveSolutions(CkReductionMsg *msg)
{
  std::vector<GreedyRefineLB::Solution> results(NUM_SOLUTIONS);

  int migrationsAllowed = totalObjs * migrationTolerance;
  // feasible solutions are those satistying user's migration constraint
  bool feasibleSolutions = false;
  float lowest_max_load = FLT_MAX;    // lowest max load of all solutions
  float lowest_max_load_f = FLT_MAX;  // lowest max load of feasible solution set
  float highest_max_load = 0;         // highest max load of all solutions
  int lowestMigrations = INT_MAX;     // lowest num migrations of all solutions
  const GreedyRefineLB::Solution *bestSol = NULL; // best solution

  // first pass. Will record solution with lowest migrations as the best, in case
  // there is no feasible solution
  CkReduction::setElement *current = (CkReduction::setElement*)msg->getData();  // Get the first element in the set
  int numSolutions = 0;
  for ( ; current && (numSolutions < NUM_SOLUTIONS); current = current->next()) {
    PUP::fromMem pd(&current->data);
    pd|results[numSolutions]; // store result
    if (results[numSolutions].migrations >= 0) {  // valid result
      const GreedyRefineLB::Solution &r = results[numSolutions++];
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
      const GreedyRefineLB::Solution &r = results[i];
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
    CkPrintf("GreedyRefineLB: Lowest max_load is %f, worst max_load is %f, lowest migrations=%d\n",
             lowest_max_load, highest_max_load, lowestMigrations);

    CkPrintf("GreedyRefineLB: Got %d solutions at %f\nBest one is from PE %d with max_load=%f, migrations=%d\n",
             numSolutions, CkWallTimer(), bestSol->pe, bestSol->max_load, bestSol->migrations);
    float A, B;
    getGreedyRefineParams(bestSol->pe, A, B);
    CkPrintf("Best PE used params A=%f B=%f\n", A, B);
  }

  // notify PE that produced the best solution
  thisProxy[bestSol->pe].ApplyDecision();
}

#include "GreedyRefineLB.def.h"

/*@}*/
