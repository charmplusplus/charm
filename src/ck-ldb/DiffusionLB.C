/** \file DiffusionLB.C
 *  Authors: Monika G
 *           Kavitha C
 *
 */

/**
 *  1. Each node has a list of neighbors (bi-directional) (either topology-based
 *     or other mechanisms like k highest communicating nodes)
 *  2. Over multiple iterations, each node diffuses load to neighbor nodes
 *     by only passing load tokens (not actual objects)
 *  3. Once the diffusion iterations converge (load imbalance threshold is reached),
 *     actual load balancing is done by taking object communication into account
 */

#include "DiffusionLB.h"
#include "LBSimulation.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "ck.h"
#include "ckgraph.h"
#include "envelope.h"
// #include "LBDBManager.h"
// #include "LBSimulation.h"
#include "DiffusionHelper.C"
#include "elements.h"

#define DEBUGF(x) CmiPrintf x;
#define DEBUGR(x)  // CmiPrintf x;
#define DEBUGL(x) /*CmiPrintf x*/;
// Rounds of the pseudo-load diffusion loop. Fixed, with no convergence check:
// every round costs two neighbour exchanges and the SDAG waits between them, so
// the strategy pays all 40 even when the load is already even. On a GPU-bound
// run where diffusion wants to move ~1% of the load, that is the single largest
// cost of load balancing -- larger than the migration it decides on.
// CHARM_LB_DIFFUSION_ITERS overrides it so the trade can be measured.
static int diffusionIterations() {
  static const int n = []() {
    const char* s = getenv("CHARM_LB_DIFFUSION_ITERS");
    const int v = s ? atoi(s) : 40;
    return v > 0 ? v : 40;
  }();
  return n;
}
#define ITERATIONS (diffusionIterations())

// Two job-wide hops used to sit between the strategy's phases: PE 0 counting
// every node out of the pseudo rounds before broadcasting AcrossNodeLB, and PE 0
// counting every node's within-node phase before broadcasting
// ProcessMigrations. Neither is needed. The convergence verdict a member leaves
// the rounds on was reduced over every member, so no round message can still be
// in flight; and a node's move list is final once its own within-node handoffs
// are acked, because every LoadReceived that targets a PE of this node was
// issued by this node's rank0PE (see migMaybeDone). Under +LBAsync each hop made
// every node wait for the slowest before issuing a single move, while the
// application kept running around it. Both are now node-local;
// CHARM_DIFFUSION_GLOBAL_PHASES=1 restores the PE 0 barriers for bisecting.
static bool diffusionGlobalPhases() {
  static const bool on = (getenv("CHARM_DIFFUSION_GLOBAL_PHASES") != nullptr);
  return on;
}

#include "DiffusionCostModel.h"

// The across-node transfer-cost table, loaded once per process from the file
// named by +LBCostConfig. Shared by every DiffusionLB branch in the process;
// nothing mutates it after the load.
DiffusionCostConfig diffusionCostCfg;

#include "DiffusionCostModel.C"
#include "DiffusionMetric.C"
#include "DiffusionNeighbors.C"
#include "DiffusionPseudo.C"
#include "DiffusionCore.C"

// Percentage of error acceptable. Only the pseudo-round convergence ratio
// still derives its default from this; the decision floors themselves come
// from +LBDiffusionMinImbalance (effMinImbalance).
#define THRESHOLD 2

// Diffusion rounds stop once no node wants to shift more than this fraction of
// its own load. Measured on a GPU-bound run, diffusion asks to move ~1.6% on the
// first round and converges immediately after, so the fixed 40 rounds were
// almost entirely wasted.
static double pseudoConvergeRatio() {
  static const double r = []() {
    const char* s = getenv("CHARM_LB_DIFFUSION_CONVERGE");
    const double v = s ? atof(s) : (THRESHOLD / 100.0);
    return v > 0.0 ? v : (THRESHOLD / 100.0);
  }();
  return r;
}
#define PSEUDO_CONVERGE_RATIO (pseudoConvergeRatio())

// Initialize static Diffusion timing variables
double DiffusionLB::totalNeighborTime = 0.0;
double DiffusionLB::totalLBTime = 0.0;
double DiffusionLB::totalStartTime = 0.0;
double DiffusionLB::totalPseudoLBTime = 0.0;
  double DiffusionLB::totalAcrossTime = 0.0;
double DiffusionLB::totalWithinTime = 0.0;
  double DiffusionLB::phaseStartTime = 0.0;

static bool diffusionTimingExitRegistered = false;

static void printDiffusionTimingAtExit() {
  DiffusionLB::printDiffusionTiming();
  CkContinueExit();
}

// CreateLBFunc_Def(DiffusionLB, "The distributed graph refinement load balancer")
static void lbinit()
{
  LBRegisterBalancer<DiffusionLB>("DiffusionLB",
                                  "The distributed graph refine load balancer");

  numPes = CkNumPes();
  
  // Register exit function to print Diffusion timing
  if (!diffusionTimingExitRegistered) {
    registerExitFn(printDiffusionTimingAtExit);
    diffusionTimingExitRegistered = true;
  }
}

using std::vector;


DiffusionLB::DiffusionLB(const CkLBOptions& opt) : CBase_DiffusionLB(opt)
{
  // Zero until an application registers a position and initializeCentroid (or
  // the first centroid to arrive) establishes the width. Left uninitialised it
  // read as 3, and every non-3 width -- a 1-D ordering key, say -- was
  // rejected as a size mismatch.
  position_dim = 0;
  // Once per process: every branch shares one table, and the loader prints on
  // PE 0 only. Left uncalibrated when no +LBCostConfig was given, in which case
  // the across-node phase behaves exactly as it did before the model existed.
  static bool costCfgLoadAttempted = false;
  if (!costCfgLoadAttempted)
  {
    costCfgLoadAttempted = true;
    if (_lb_args.costConfig() != NULL)
      diffusionCostCfg.load(_lb_args.costConfig());
  }

  nodeSize = CkNodeSize(0);
  myNodeId = CkMyPe() / nodeSize;
  acks = 0;
  max = 0;
  hs_asksOut = 0;
  hs_confirmOut = 0;
  hs_phaseOwed = false;
  hs_barrierOwed = false;
  roundsDoneCount = 0;
  hs_graphCached = false;
  mig_acksOut = 0;
  across_owed = false;
  within_owed = false;
  acrossDoneCount = 0;
  withinDoneCount = 0;
  acrossNbrDoneCount = 0;
  acrossSelfDone = false;
  withinSelfDone = false;
  withinNbrDoneCount = 0;
  pseudoContribCount = 0;
  pseudoMaxMetric = 0.0;
  effMinImbalance = _lb_args.diffusionMinImbalance();
  quietSteps = 0;
  stepEndTime = 0.0;
  thisInterval = 0.0;
  lastInterval = 0.0;
  lastStepMoves = 0;
  lastMigratesIssued = 0;
  revertThisStep = false;
  lastStepWasRevert = false;
  keyed1D = false;
  myKeyLo = myKeyHi = 0.0;
  round = 0;
  hs_asksOut = 0;
  hs_confirmOut = 0;
  hs_phaseOwed = false;
  hs_barrierOwed = false;
  roundsDoneCount = 0;
  hs_graphCached = false;
  mig_acksOut = 0;
  across_owed = false;
  within_owed = false;
  acrossDoneCount = 0;
  withinDoneCount = 0;
  acrossNbrDoneCount = 0;
  acrossSelfDone = false;
  statsReceived = 0;
  rank0_barrier_counter = 0;

  myNodeInternalBytes = 0.0;
  myNodeExternalBytes = 0.0;

  num_migrations = 0;
  pseudoSectionBuilt = false;

#if CMK_LBDB_ON
  lbname = "DiffusionLB";
  if (_lb_args.statsOn())
    lbmgr->CollectStatsOn();
  // Every comm-aware balancer in this tree enables communication instrumentation
  // from its own constructor rather than requiring the user to pass +LBCommOn --
  // MetisLB.C:23, ScotchLB.C:23, ScotchTopoLB.C:25, ScotchRefineLB.C:20,
  // RecBipartLB.C:119, ZoltanLB.C:62. Without this the comm graph is empty and
  // MetricComm scores every object identically.
  LBTurnCommOn();
  thisProxy = CProxy_DiffusionLB(thisgroup);
  numNodes = CkNumPes() / nodeSize;  // CkNumNodes();
  myStats = new DistBaseLB::LDStats;

  rank0PE = myNodeId * nodeSize;  // CkNodeFirst(CkMyNode());
  if (CkMyPe() == rank0PE)
  {
    statsList = new CLBStatsMsg*[nodeSize];
    nodeStats = new BaseLB::LDStats(nodeSize);
    numObjects.resize(nodeSize);
    prefixObjects.resize(nodeSize);
    pe_load.resize(nodeSize);
  }
  if (CkMyPe() == 0)
  {
    fullStats = new BaseLB::LDStats(CkNumPes());
  }
#endif
}

DiffusionLB::DiffusionLB(CkMigrateMessage* m) : CBase_DiffusionLB(m) {}

DiffusionLB::~DiffusionLB()
{
#if CMK_LBDB_ON
  delete[] statsList;
  delete nodeStats;
  delete myStats;
  delete[] gain_val;
  lbmgr = CProxy_LBManager(_lbmgr).ckLocalBranch();
  if (lbmgr)
    lbmgr->RemoveStartLBFn(startLbFnHdl);
#endif
}

// Main entry point for the load balancer
void DiffusionLB::Strategy(const DistBaseLB::LDStats* const stats)
{
  startOverallTiming();
  total_migrates = 0;
  total_crossnode_migrates = 0;

  if (CkMyPe() == 0 && _lb_args.debug() >= 1)
  {
    double start_time = CmiWallTimer();
  }
  statsmsg = AssembleStats();
  if (statsmsg == NULL)
    CkAbort("Error: statsmsg is NULL\n");

  // start stats assembly on rank0PE
  marshmsg = new CkMarshalledCLBStatsMessage(statsmsg);

  // reset variables (necessary for mutliple LB rounds)
  acks = 0;
  max = 0;
  round = 0;
  hs_asksOut = 0;
  hs_confirmOut = 0;
  hs_phaseOwed = false;
  hs_barrierOwed = false;
  roundsDoneCount = 0;
  // hs_graphCached deliberately NOT reset here: the graph survives steps.
  mig_acksOut = 0;
  across_owed = false;
  within_owed = false;
  acrossDoneCount = 0;
  withinDoneCount = 0;
  acrossNbrDoneCount = 0;
  acrossSelfDone = false;
  withinSelfDone = false;
  withinNbrDoneCount = 0;
  pseudoContribCount = 0;
  pseudoMaxMetric = 0.0;
  rank0_barrier_counter = 0;
  pseudo_done = true;

  num_migrations = 0;

  mig_id_map.clear();
  objectHandles.clear();
  objectSrcIds.clear();
  objSenderPEs.clear();
  objectLoads.clear();
  objectGLoads.clear();
  objectKeys.clear();

  // The regret loop's two inputs, reduced ahead of statsAssembled. Group
  // reductions complete in contribution order, and every PE contributes these
  // two before its statsAssembled contribution (a rank0PE's comes later, from
  // ReceiveStats), so both verdict inputs are on every PE by the time the
  // strategy starts. The interval is this PE's wall time since the previous
  // step ended; zero on the first step, which decideRegret treats as unknown.
  {
    const double interval = (stepEndTime > 0.0) ? (CmiWallTimer() - stepEndTime) : 0.0;
    CkCallback cbI(CkReductionTarget(DiffusionLB, regretInterval), thisProxy);
    contribute(sizeof(double), &interval, CkReduction::max_double, cbI);
    CkCallback cbM(CkReductionTarget(DiffusionLB, regretMoves), thisProxy);
    contribute(sizeof(int), &lastMigratesIssued, CkReduction::sum_int, cbM);
  }

  thisProxy[rank0PE].ReceiveStats(*marshmsg);

  if (CkMyPe() != rank0PE)
  {
    CkCallback cb(CkReductionTarget(DiffusionLB, statsAssembled), thisProxy);
    contribute(cb);
  }

  if (CkMyPe() == rank0PE) {
      for (int i = 0; i < nodeSize; i++) pe_load[i] = 0;
      myNodeInternalBytes = 0.0;
      myNodeExternalBytes = 0.0;
  }
}

/*Entry method called on each rank0PE to collect all node-relevant stats. On completion,
 * all PEs call statsAssembled().*/
void DiffusionLB::ReceiveStats(CkMarshalledCLBStatsMessage&& data)
{
  // TODO: why is this in CMK_LBDB_ON? needs to be done always?
#if CMK_LBDB_ON
  CLBStatsMsg* m = data.getMessage();
  CmiAssert(CkMyPe() == rank0PE);

  // store the message
  int fromRank = m->from_pe - rank0PE;
  statsReceived++;

  // Clear nodeStats at the start of each new round to prevent accumulation
  if (statsReceived == 1) {
    nodeStats->objData.clear();
    nodeStats->from_proc.clear();
    nodeStats->to_proc.clear();
    nodeStats->commData.clear();
    nodeStats->n_migrateobjs = 0;
    nodeStats->deleteCommHash();
  }

  AddToList(m, fromRank);

  if (statsReceived == nodeSize)
  {
    // build LDStats
    BuildStats();
    CkCallback cb(CkReductionTarget(DiffusionLB, statsAssembled), thisProxy);
    contribute(cb);
    statsReceived = 0;
  }
#endif
}

/*Once stats are assembled on rank0PEs, can begin finding Nbors*/
void DiffusionLB::statsAssembled()
{
  // Same inputs on every PE, so the same verdict everywhere.
  decideRegret();
  if (CkMyPe() == rank0PE)
  {
    findNBors(1);
  }
}

void DiffusionLB::regretInterval(double maxInterval)
{
  lastInterval = thisInterval;
  thisInterval = maxInterval;
}

void DiffusionLB::regretMoves(int moves) { lastStepMoves = moves; }

// Judge the previous step by what it did to the application. thisInterval
// covers the iterations run under the previous step's placement, lastInterval
// the ones before it; if the placement made those iterations slower by more
// than the tolerance, the step is taken back and the floor raised so the same
// noise is not chased again next time. Only a step that actually moved
// something is judged, and never one that was itself a revert -- the interval
// after a revert is compared against the bad placement and would always look
// like an improvement, so judging it could only ever undo the undo.
void DiffusionLB::decideRegret()
{
  const double tol = _lb_args.diffusionRegret();
  const bool judge = tol > 0.0 && !lastStepWasRevert && lastStepMoves > 0 &&
                     lastInterval > 0.0 && thisInterval > 0.0;
  revertThisStep = judge && thisInterval > lastInterval * (1.0 + tol);

  if (revertThisStep)
  {
    effMinImbalance = std::min(2.0 * effMinImbalance, 1.0);
    quietSteps = 0;
  }
  else if (lastStepMoves == 0)
  {
    // Two quiet steps in a row: relax the floor back toward its configured
    // value one halving at a time.
    if (++quietSteps >= 2)
    {
      effMinImbalance = std::max(_lb_args.diffusionMinImbalance(), 0.5 * effMinImbalance);
      quietSteps = 0;
    }
  }
  else
    quietSteps = 0;

  if (CkMyPe() == 0 && _lb_args.debug() > 0)
    CkPrintf("[DiffusionLB] step %d: interval %.4fs (previous %.4fs), previous step "
             "moved %d%s -> %s, floor %.3f\n",
             step(), thisInterval, lastInterval, lastStepMoves,
             lastStepWasRevert ? " (a revert)" : "",
             revertThisStep ? "REVERT" : "balance", effMinImbalance);
}

double DiffusionLB::keyOf(const LDObjData& od)
{
  if (od.position.size() != 1) return std::numeric_limits<double>::quiet_NaN();
  return (double)od.position[0];
}

bool DiffusionLB::nborKeyAdjacent(int nbor) const
{
  if (!keyed1D) return true;
  if (nbor < 0 || nbor >= (int)nborKeyLo.size()) return true;
  const double nlo = nborKeyLo[nbor], nhi = nborKeyHi[nbor];
  if (nlo != nlo || nhi != nhi) return true;
  const bool below = nhi <= myKeyLo;
  const bool above = nlo >= myKeyHi;
  if (!below && !above) return true;  // overlapping: nothing to preserve
  for (int j = 0; j < (int)nborKeyLo.size(); j++)
  {
    if (j == nbor) continue;
    const double jlo = nborKeyLo[j], jhi = nborKeyHi[j];
    if (jlo != jlo || jhi != jhi) continue;
    // Another neighbour sits between us and this one.
    if (below && jlo >= nhi && jhi <= myKeyLo) return false;
    if (above && jhi <= nlo && jlo >= myKeyHi) return false;
  }
  return true;
}

void DiffusionLB::allowedEndsFor(int nbor, std::vector<char>& allowed)
{
  const int n = nodeStats->objData.size();
  allowed.assign(n, 1);
  if (!keyed1D) return;
  if (!nborKeyAdjacent(nbor))
  {
    // Nothing may cross to a non-adjacent interval.
    std::fill(allowed.begin(), allowed.end(), 0);
    return;
  }
  int loEnd = -1, hiEnd = -1;
  double loKey = 0.0, hiKey = 0.0;
  for (int i = 0; i < n; i++)
  {
    if (objs[i].getCurrPe() == -1 || !nodeStats->objData[i].migratable) continue;
    const double k = keyOf(nodeStats->objData[i]);
    if (k != k) continue;
    if (loEnd == -1 || k < loKey) { loEnd = i; loKey = k; }
    if (hiEnd == -1 || k > hiKey) { hiEnd = i; hiKey = k; }
  }
  std::fill(allowed.begin(), allowed.end(), 0);
  if (loEnd == -1) return;
  // Which side of us does this neighbour sit on? Its interval ends came with
  // its round-0 load. An unkeyed or overlapping neighbour gets both ends and
  // the metric's own distance rule decides.
  bool below = false, above = false;
  if (nbor >= 0 && nbor < (int)nborKeyLo.size())
  {
    const double nlo = nborKeyLo[nbor], nhi = nborKeyHi[nbor];
    if (nlo == nlo && nhi == nhi)
    {
      below = nhi <= loKey;
      above = nlo >= hiKey;
    }
  }
  if (below && !above) allowed[loEnd] = 1;
  else if (above && !below) allowed[hiEnd] = 1;
  else { allowed[loEnd] = 1; allowed[hiEnd] = 1; }
}

// Undo the previous step: every object it moved onto this node goes back to
// the PE it came from. Objects are addressed directly (the token form of the
// handoff, only_mcount=1), whether the previous home is in this node or not,
// so the within-node phase has nothing to retarget. Rank0PE only.
int DiffusionLB::revertPreviousStep()
{
  const int n = nodeStats->objData.size();
  const int prevStepNo = step() - 1;
  int reverted = 0;
  for (int j = 0; j < n; j++)
  {
    const LDObjData& od = nodeStats->objData[j];
    if (od.prevStep != prevStepNo || od.prevPe < 0 || od.prevPe >= numPes) continue;
    if (!od.migratable || objs[j].getCurrPe() == -1) continue;
    const int rank = GetRank(j);
    const int donorPE = rank0PE + rank;
    const int destPE = od.prevPe;
    if (destPE == donorPE) continue;
    const int pe_local_id = j - (rank > 0 ? prefixObjects[rank - 1] : 0);
    objs[j].setCurrPe(-1);
    mig_acksOut += 2;
    thisProxy[destPE].LoadMetaInfo(od.handle, pe_local_id, objs[j].getCompLoad(),
                                   diffusionObjLoad(od), donorPE, 1, CkMyPe(), keyOf(od));
    thisProxy[donorPE].LoadReceived(pe_local_id, destPE, CkMyPe());
    nodeStats->to_proc[j] = destPE;
    reverted++;
  }
  if (_lb_args.debug() > 0)
    CkPrintf("[DiffusionLB node %d] step %d: reverting %d object(s) moved by step %d\n",
             myNodeId, step(), reverted, prevStepNo);
  return reverted;
}

void DiffusionLB::InitializeObjHeap(int n)
{
  obj_heap.resize(n);
  heap_pos.resize(n);
  for (int i = 0; i < n; i++)
  {
    obj_heap[i] = i;
    heap_pos[i] = i;
  }
  heapify(obj_heap, ObjCompareOperator(&objects, gain_val), heap_pos);
}

// Create a migrate message for this obj from resident PE to rank0PE
// objId should be PE local id of the object
void DiffusionLB::LoadReceived(int objId, int destPE, int ackPE)
{
  thisProxy[ackPE].migMsgAck();
  int sourcePE = CkMyPe();
  
  if (objId < 0 || objId >= myStats->objData.size()) {
    CkAbort("Error: objId %d out of bounds for size %d on PE %d\n", objId, (int)myStats->objData.size(), CkMyPe());
  }
  // load is received, hence create a migrate message for the object with id objId.
  auto it = mig_id_map.find(objId);
  if(it!=mig_id_map.end()) {
    MigrateInfo* migrateMe = it->second;
    migrateMe->to_pe = destPE;
  } else {
    MigrateInfo* migrateMe = new MigrateInfo;
    migrateMe->obj = myStats->objData[objId].handle;
    migrateMe->from_pe = CkMyPe();
    migrateMe->to_pe = destPE;
    // migrateMe->async_arrival = myStats->objData[objId].asyncArrival;
    migrateInfo.push_back(migrateMe);
    mig_id_map.emplace(objId, migrateMe);
    total_migrates++;

    if (CkMyPe() / nodeSize != destPE / nodeSize)
      total_crossnode_migrates++;
  }

  if (_lb_args.debug() > 2) CkPrintf("[%d] Completing LoadReceived for objId %d to %d\n", CkMyPe(), objId, destPE);
}

void DiffusionLB::update_peload(int rank, double load) {
  pe_load[rank] -= load;
}

/* Load has been logically sent from overloaded to underloaded nodes in LoadBalance().
 * Now we should load balance the PE's within the node. This function should only be
 * called by rank0PE.
 *
 * At a high level, this does the following:
 * - find overloaded and underloaded PEs on my node
 * - create minheap of PEs sorted by load
 * - create maxheap of objects (using ckheap) sorted by load
 * - iterate through objects in maxheap and offload based on minheap (via LoadReceived)
 *
 * MEMORY CONTRACT (LBMemoryContract.h) -- integration design, not yet wired:
 * DiffusionLB does not pass through CentralLB::Strategy, so the contract
 * verifier does not cover it. The decentralized form of the contract is
 * receiver-side acceptance: (1) each PE advertises its device's free memory
 * on the neighbor load-exchange messages it already sends; (2) a transfer
 * becomes an offer that the RECEIVER accepts or refuses against a local
 * ledger of headroom minus committed arrivals (receiver-side serialization
 * resolves concurrent senders; a refusal is ordinary diffusion back-pressure
 * -- the object stays and later rounds retry); (3) I-batch is local: each PE
 * bounds its own round's outgoing staged bytes by its staging reserve. The
 * offer/refusal round-trip touches this file, DiffusionNeighbors, and the
 * LoadMetaInfo/LoadReceived protocol, and must be validated on a multi-device
 * run -- deliberately not implemented blind on a single-GPU machine.
 * */
void DiffusionLB::WithinNodeLB()
{

   endAcrossTiming();
      startWithinTiming();
  if (thisIndex == 0)
    if (_lb_args.debug() == 3) CkPrintf("--------STARTING WITHIN NODE LB--------\n");

  // The next phase now starts from the withinDone barrier (see withinDone),
  // which makes the same three-way choice the quiescence callback made here.

  if( nodeSize == 1) {
      if (_lb_args.debug() == 3) CkPrintf("--------Node size is 1--------\n");
    // Every PE is its own node's rank0PE: nothing to balance within it, but a
    // revert step still has to hand back what the previous step moved here.
    if (revertThisStep) revertPreviousStep();
    withinNodeReport();
    return;
  }
  if (CkMyPe() == rank0PE)
  {
   
    // A revert step moves only what the previous step moved; see decideRegret.
    if (revertThisStep)
    {
      revertPreviousStep();
      endWithinTiming();
      withinNodeReport();
      return;
    }

    const int n = nodeStats->objData.size();

    // ---- which resource does this phase balance? -------------------------
    // PEs of a process share a device, so moving a chare between them relieves
    // the host only. When the diffused dimension is host time that is the whole
    // story. When it is GPU time, host work matters only if some PE's host time
    // exceeds the node's device time; otherwise the device is the bottleneck,
    // and per-PE host time is mostly driver overhead plus group work charged to
    // whatever object happened to be current -- measured on barnes as a 100x
    // spread between PEs of one node, all of it noise. Cutting the placement
    // for that shredded it (1 object on one PE, 65 on the next) for nothing.
    // The device dimension is then the smoother, real signal: equal device
    // work per PE keeps the per-PE driving cost even as well.
    bool onHost = true;
#if CMK_CUDA
    if (_lb_args.diffusionGpuDim())
    {
      double maxPeHost = 0.0, nodeGpu = 0.0;
      for (int r = 0; r < nodeSize; r++) maxPeHost = std::max(maxPeHost, pe_load[r]);
      for (int j = 0; j < n; j++)
        if (objs[j].getCurrPe() != -1) nodeGpu += diffusionObjLoad(nodeStats->objData[j]);
      for (size_t t = 0; t < objectGLoads.size(); t++) nodeGpu += objectGLoads[t];
      onHost = maxPeHost > nodeGpu;
    }
#endif
    if (!onHost)
    {
      // Re-express every within-node figure in the device dimension, with a
      // mean floor of its own (the across-node phase applies none; see
      // BuildStats). From here
      // on pe_load, the CkVertex loads and the token loads are what this phase
      // balances; nothing after this point reads them as host time.
      double sum = 0.0;
      int cnt = 0;
      for (int j = 0; j < n; j++)
        if (objs[j].getCurrPe() != -1) { sum += diffusionObjLoad(nodeStats->objData[j]); cnt++; }
      for (size_t t = 0; t < objectGLoads.size(); t++) { sum += objectGLoads[t]; cnt++; }
      const double floor = cnt > 0 ? sum / cnt : 0.0;
      for (int r = 0; r < nodeSize; r++) pe_load[r] = 0.0;
      for (int j = 0; j < n; j++)
      {
        const double g = std::max(diffusionObjLoad(nodeStats->objData[j]), floor);
        objs[j].setCompLoad(g);
        if (objs[j].getCurrPe() != -1) pe_load[GetRank(j)] += g;
      }
      for (size_t t = 0; t < objectGLoads.size(); t++)
      {
        objectLoads[t] = std::max(objectGLoads[t], floor);
        pe_load[0] += objectLoads[t];
      }
    }

    double avgPE = averagePE();
    double maxPE = 0.0;
    for (int r = 0; r < nodeSize; r++) maxPE = std::max(maxPE, pe_load[r]);

    // ---- the floor -------------------------------------------------------
    // Below it there is nothing to balance, only noise to chase.
    if (avgPE <= 0.0 || maxPE <= avgPE * (1.0 + effMinImbalance))
    {
      if (_lb_args.debug() > 1)
        CkPrintf("[WITHIN node %d] max/avg %.3f under floor %.3f (%s): no moves\n",
                 myNodeId, avgPE > 0.0 ? maxPE / avgPE : 0.0, 1.0 + effMinImbalance,
                 onHost ? "host" : "device");
      endWithinTiming();
      withinNodeReport();
      return;
    }

    // Create a max heap and min heap for pe loads
    std::vector<double> objectSizes;
    std::vector<int> objectIds;
    std::vector<int> objectPEs;
    std::vector<LDObjHandle> objectHdl;
    std::vector<int> isToken;
    minHeap minPes(nodeSize);
    double threshold = effMinImbalance * avgPE;

    // ---- interval repartition ----------------------------------------
    // When the application registered a 1-D ordering key, do not shuffle
    // individual objects between this node's PEs by load. Cut the node's key
    // interval into nodeSize CONTIGUOUS pieces of equal load instead.
    //
    // The distinction matters because peBoxes -- what the neighbour exchange
    // prunes against -- is per PE, while the across-node interval rule only
    // keeps a NODE contiguous. A node can hold one clean interval while its
    // PEs hold interleaved fragments of it, and it is this phase that
    // interleaves them. Partitioning gives every PE one interval by
    // construction, which is the property blockmap had before any balancing.
    //
    // Tokens -- objects another node handed this one this step -- take part:
    // their key travels in LoadMetaInfo, so the cut is over the complete set.
    // This used to be skipped whenever a token had arrived, and the heap path
    // below then shuffled the node instead (measured: 13 of 24 phases).
    {
      bool allKeyed = keyed1D;
      for (size_t t = 0; t < objectKeys.size() && allKeyed; t++)
        if (objectKeys[t] != objectKeys[t]) allKeyed = false;

      if (allKeyed)
      {
        struct Ent { double key; double load; int idx; bool token; };
        std::vector<Ent> ents;
        ents.reserve(n + objectKeys.size());
        for (int j = 0; j < n; j++)
        {
          if (objs[j].getCurrPe() == -1) continue;
          ents.push_back({keyOf(nodeStats->objData[j]), objs[j].getCompLoad(), j, false});
        }
        for (size_t t = 0; t < objectKeys.size(); t++)
          ents.push_back({objectKeys[t], objectLoads[t], (int)t, true});
        std::sort(ents.begin(), ents.end(),
                  [](const Ent& a, const Ent& b) { return a.key < b.key; });

        const int m = ents.size();
        double total = 0.0;
        for (int k = 0; k < m; k++) total += ents[k].load;
        const double share = total / (double)nodeSize;

        int moved = 0;
        double acc = 0.0;
        int target = 0;
        for (int k = 0; k < m; k++)
        {
          const Ent& e = ents[k];
          // Advance the cut once this chunk has its share, but never past the
          // last rank, and leave at least one object for each remaining rank.
          while (target < nodeSize - 1 && acc >= share * (target + 1) &&
                 (m - k) > (nodeSize - 1 - target))
            target++;
          acc += e.load;
          const int destPE = rank0PE + target;

          if (e.token)
          {
            // Addressed to this rank0PE by the across-node phase; retarget it
            // to its cut, the way the heap path hands tokens on.
            if (target == 0) continue;
            const int t = e.idx;
            const int donorPE = objSenderPEs[t];
            migrates_expected--;
            mig_acksOut += 2;
            thisProxy[destPE].LoadMetaInfo(objectHandles[t], objectSrcIds[t], objectLoads[t],
                                           objectGLoads[t], donorPE, 1, CkMyPe(), objectKeys[t]);
            thisProxy[donorPE].LoadReceived(objectSrcIds[t], destPE, CkMyPe());
            pe_load[0] -= objectLoads[t];
            pe_load[target] += objectLoads[t];
            moved++;
            continue;
          }

          const int j = e.idx;
          const int rank = GetRank(j);
          if (rank == target) continue;
          if (!nodeStats->objData[j].migratable) continue;

          const int pe_local_id = j - (rank > 0 ? prefixObjects[rank - 1] : 0);
          const int donorPE = rank0PE + rank;
          if (donorPE == destPE) continue;

          mig_acksOut += 2;
          thisProxy[destPE].LoadMetaInfo(nodeStats->objData[j].handle,
                                         pe_local_id, objs[j].getCompLoad(),
                                         diffusionObjLoad(nodeStats->objData[j]),
                                         donorPE, 1, CkMyPe(), e.key);
          thisProxy[donorPE].LoadReceived(pe_local_id, destPE, CkMyPe());
          nodeStats->to_proc[j] = destPE;
          pe_load[rank] -= objs[j].getCompLoad();
          pe_load[target] += objs[j].getCompLoad();
          moved++;
        }
        if (_lb_args.debug() > 1)
          CkPrintf("[WITHIN node %d] interval repartition (%s): %d of %d objects "
                   "moved, share=%.6f\n", myNodeId, onHost ? "host" : "device",
                   moved, m, share);

        endWithinTiming();
        withinNodeReport();
        return;
      }
    }
    // ------------------------------------------------------------------

    // for each pe... find overload, something with prefix sum?
    // and store the underloaded pes
    for (int rank = 0; rank < nodeSize; rank++)
    {
      if (_lb_args.debug() == 3) CkPrintf("\nOrig PE load with node LB [%d] = %lf", rank+rank0PE, pe_load[rank]);
      if (pe_load[rank] > avgPE + threshold)
      {
        double overLoad = pe_load[rank] - avgPE;
        int start = 0;
        if (rank != 0)
        {
          start = prefixObjects[rank - 1];
        }
        // Collect from the ENDS of this rank's key interval, not in array
        // order. Same reasoning as the across-node interval rule: the volume
        // that neighbour exchange costs is set by each PE's EXTENT, so the
        // objects worth giving up are the ones at the edges. Taking them
        // end-inward leaves the retained set contiguous; array order punches
        // holes in it, which cost nothing in load but the full extent in
        // volume. Only when the application registered a 1-D ordering key.
        std::vector<int> _ord;
        for (int j2 = start; j2 < prefixObjects[rank]; j2++) _ord.push_back(j2);
        const bool _ordered =
            !_ord.empty() && nodeStats->objData[_ord[0]].position.size() == 1;
        if (_ordered)
          std::sort(_ord.begin(), _ord.end(), [&](int a, int b) {
            return nodeStats->objData[a].position[0] <
                   nodeStats->objData[b].position[0];
          });
        int _lo = 0, _hi = (int)_ord.size() - 1;
        bool _takeLo = true;
        while (_lo <= _hi)
        {
          int j;
          if (_ordered)
          {
            j = _takeLo ? _ord[_lo] : _ord[_hi];
            if (_takeLo) _lo++; else _hi--;
            _takeLo = !_takeLo;
          }
          else { j = _ord[_lo]; _lo++; }
        {
          // getCompLoad(), not getVertexLoad(): this weighs an object's load against
          // a budget in seconds (overLoad), and getVertexLoad()'s MAX(compLoad, 0.1)
          // floor reports every object as 0.1s whenever real per-object load is
          // smaller -- the common case. With a typical overLoad well under 0.1s the
          // test then fails for every object on every step, and within-node balancing
          // silently does nothing while reporting that it ran.
          // An object carrying no measured load cannot relieve the donor's
          // overload, and moving it changes neither PE's load -- so overLoad
          // never shrinks and this loop hands over EVERY object on the PE.
          // On a GPU-resident application, where host time per object is
          // ~0, that is nearly all of them: the donor is stripped to a
          // handful while the receiver piles them up (measured: an even
          // 43-44 objects/PE became 1 vs 98 after one round, and the load
          // spread it was minimising got worse, not better). Require a
          // positive contribution, and stop once the budget is spent.
          if (overLoad <= 0.0) break;
          if (objs[j].getCompLoad() <= 0.0) continue;
          if (objs[j].isMigratable() && objs[j].getCurrPe() != -1 && objs[j].getCompLoad() <= overLoad)
          {
            objectSizes.push_back(objs[j].getCompLoad());

            int pe_local_id = j;
            if (rank != 0) {
              pe_local_id = j - prefixObjects[rank - 1];
            }
            objectIds.push_back(pe_local_id);
            objectPEs.push_back(rank+rank0PE);
            objectHdl.push_back(nodeStats->objData[j].handle);
            isToken.push_back(0);
            overLoad -= objs[j].getCompLoad();
          }
        }
        }
        if(rank==0) {
          // Objects migrating in. They land here because across-node migration
          // addresses rank0PE, not because rank0PE should keep them; now that
          // an arrival actually raises pe_load[0] (see the mean-floor in
          // BuildStats) this branch is reachable and hands them on.
          for(int i=0;i<objectLoads.size();i++) {
            if(overLoad <= 0.0) break;
            if(objectLoads[i] <= overLoad) {
              objectSizes.push_back(objectLoads[i]);
              objectIds.push_back(objectSrcIds[i]); // this is pe local id
              objectHdl.push_back(objectHandles[i]);
              objectPEs.push_back(objSenderPEs[i]);
              isToken.push_back(1);
              overLoad -= objectLoads[i];
            }
          }
        }
      }
      else if (pe_load[rank] < avgPE - threshold)
      {
        InfoRecord* itemMin = new InfoRecord;
        itemMin->load = pe_load[rank];
        itemMin->Id = rank;
        minPes.insert(itemMin);
      }
    }

    // build heap of objects
    maxHeap objects(objectIds.size());
    for (int i = 0; i < objectIds.size(); i++)
    {
      InfoRecord* item = new InfoRecord;
      item->load = objectSizes[i];  // sorting factor in maxheap
      item->Id = objectIds[i];
      item->pe = objectPEs[i]; // sending pe
      item->handle = objectHdl[i];
      item->token = false;
      if(isToken[i])
        item->token = true;
      objects.insert(item);
    }

    // pop object from priority queue and migrate to most underloaded PE
    // TODO: this needs a strategy update
    InfoRecord* minPE = NULL;
    while (objects.numElements() > 0 &&
           ((minPE == NULL && minPes.numElements() > 0) || minPE != NULL))
    {
      InfoRecord* maxObj = objects.deleteMax();
      if (minPE == NULL)
        minPE = minPes.deleteMin();
      double diff = avgPE - minPE->load;
      if(diff < 0) {
        minPE = minPes.deleteMin();
        continue;
      }
      int objId = maxObj->Id; // this is the pe local id of the object
      int nodeObjId = objId; // node local id (to be cmoputed below)

      // TODO!!!! objID must be the pe_local one the whole time!! otherwise, might tyr to compute pe index here for non local object!!
      int rank = maxObj->pe % nodeSize; // donor PE rank (original)
      bool is_local = !maxObj->token;

      if (!is_local) rank = 0; // coming from rank0PE (as a token)

      if (rank > 0)
        nodeObjId += prefixObjects[rank - 1];

      if (maxObj->load > diff || pe_load[rank] < avgPE - threshold)
      {
        delete maxObj;
        continue;
      }

      int destPE = rank0PE + minPE->Id;
      int donorPE = maxObj->pe;

      if (is_local && donorPE != rank + rank0PE) {
        CkAbort("Error: donorPE %d does not match object PE %d = %d + %d\n", donorPE, rank+rank0PE, rank, rank0PE);
      }

      LDObjHandle objHandle = maxObj->handle;
      double currLoad = maxObj->load;     
 
      if(!is_local) {
        // Diagnostic only, like every migrates_expected adjustment: the step's
        // barrier is the source-side move ledger, not this arrival count.
        migrates_expected--;
        //subtract from intermediate PE (rank0, i.e. me)
      } else {
        nodeStats->to_proc[nodeObjId] = destPE;
      }
    
      if (objId < 0) {
        CkAbort("Error: objId %d is negative for objId %d on donorPE %d (rank0PE %d)\n",
                objId, objId, donorPE, rank0PE);
      }
      mig_acksOut += 2;
      // Count-only handoff: the receiver keeps neither the loads nor the key.
      thisProxy[destPE].LoadMetaInfo(objHandle, objId, currLoad, 0.0, donorPE, 1, CkMyPe(),
                                     std::numeric_limits<double>::quiet_NaN()); // to the receiving PE (mig++)
      thisProxy[donorPE].LoadReceived(objId, destPE, CkMyPe());
      pe_load[minPE->Id] += maxObj->load;
      pe_load[rank] -= maxObj->load;
      if (pe_load[minPE->Id] < avgPE)
      {
        minPE->load += maxObj->load;//= pe_load[minPE->Id];
        minPes.insert(minPE);
      }
      else
        delete minPE;
      minPE = NULL;
    }
    //may be clearing heaps for next LB step, after intra-node LB is done above
    // TODO: clear the heaps? why?
    while (minPes.numElements() > 0)
    {
      InfoRecord* minPE = minPes.deleteMin();
      delete minPE;
    }
    while (objects.numElements() > 0)
    {
      InfoRecord* maxObj = objects.deleteMax();
      delete maxObj;
    }

    endWithinTiming();
    // Held until every handoff issued above is acked as processed -- the
    // property the quiescence detector used to provide.
    withinNodeReport();
  }
}

void DiffusionLB::ProcessMigrations()
{
  if (CkMyPe() == 0)
  BaseLB::endLBStrategyTiming();

  // SAME AS IN PACKANDSENDMIGRATEMSGS
  LBMigrateMsg* msg = new (total_migrates, CkNumPes(), CkNumPes(), 0) LBMigrateMsg;
  msg->n_moves = total_migrates;
  // This PE's contribution to the next step's regret verdict.
  lastMigratesIssued = total_migrates;
  if (_lb_args.debug() > 1) CkPrintf("PE-%d with %d migrates and %d cross-node migrates\n", CkMyPe(), total_migrates, total_crossnode_migrates);
  for (int i = 0; i < total_migrates; i++)
  {
    MigrateInfo* item = (MigrateInfo*)migrateInfo[i];
    msg->moves[i] = *item;
    delete item;
    migrateInfo[i] = 0;
  }
  migrateInfo.clear();
  // The idempotency map still points at the MigrateInfo objects deleted just
  // above, and total_migrates still counts them; both were only reset at the
  // next Strategy(). A LoadReceived straggling in after this point therefore
  // wrote through a dangling pointer (map hit) or desynchronized the
  // vector/counter pair for the following round (map miss). The phase barrier
  // is supposed to make such a straggler impossible; reset here so that if it
  // ever is not, the straggler creates a fresh, harmless entry instead of
  // corrupting the heap.
  mig_id_map.clear();
  total_migrates = 0;

  // if we don't do the barrier here, must be done with LBSyncResume so that it is done in
  // MigrationDone
  if (!_lb_args.syncResume())
  {
  // SAME AS IN PROCESSMIGRATIONDECISION
  const int me = CkMyPe();
  for (int i = 0; i < msg->n_moves; i++)
  {
    MigrateInfo& move = msg->moves[i];
     if (_lb_args.debug() == 3) CkPrintf("\n[PE-%d] Migrating obj from %d to %d", CkMyPe(), move.from_pe,
               move.to_pe);
    if (move.from_pe == me)
    {
      if (move.to_pe == me)
      {
          CkAbort("[%i] Error, attempting to migrate object myself to myself\n",
                  CkMyPe());
      }
     
      lbmgr->Migrate(move.obj, move.to_pe);
    }
    else if (move.from_pe != me)
    {
      CkAbort("Trying to move objs not on my PE\n");
    }
  }

#if CMK_GLOBAL_LOCATION_UPDATE
  // SAME AS IN PROCESSMIGRATIONDECISION
  if (!_lb_args.lbPeerDecision()) BroadcastLocationUpdate(msg);
#endif

    CkCallback cb(CkIndex_DiffusionLB::MigrationDoneWrapper(), thisProxy);
    contribute(cb);
    // Nothing holds the message past this point: the moves have been issued and
    // BroadcastLocationUpdate copies. The syncResume branch below hands
    // ownership to ProcessMigrationDecision, which frees it.
    delete msg;
  }
  else
    ProcessMigrationDecision(msg);

}

void DiffusionLB::CascadingMigration(LDObjHandle h, double load)
{
#if 0
  CkAbort("CASCADING: we don't understand this implementation yet\n");
  double threshold = THRESHOLD * avgLoadNeighbor / 100.0;
  int minNode = 0;
  int myPos = 0;  // neighborPos[CkNodeOf(rank0PE)];

  if (loadReceivers > 0)
  {
    double minLoad;
    // Send to max underloaded node
    for (int i = 0; i < neighborCount; i++)
    {
      if (toSendLoad[i] >= threshold && load <= toSendLoad[i] &&
          (minNode == -1 || minLoad < toSendLoad[i]))
      {
        minNode = i;
        minLoad = toSendLoad[i];
      }
    }
    if (minNode != -1 && minNode != myPos)
    {
      // Send load info to receiving load
      toSendLoad[minNode] -= load;
      if (toSendLoad[minNode] < threshold)
      {
        loadReceivers--;
      }
      thisProxy[sendToNeighbors[minNode] *
                nodeSize /*CkNodeFirst(sendToNeighbors[minNode])*/]
          .LoadMetaInfo(h, 0, load, CkMyPe(), 0);
      const int acrossNodeToPe = sendToNeighbors[minNode] *
                            nodeSize /*CkNodeFirst(sendToNeighbors[minNode])*/;
      lbmgr->Migrate(h, acrossNodeToPe);
#if CMK_GLOBAL_LOCATION_UPDATE
      // This is the across-node move, so the object lands in a different
      // process. Bystanders have to learn the new location, or a GPU-direct
      // sender keeps picking its transfer mode for the process the object just
      // left -- and a MEMCPY chosen that way hands the receiver a pointer into
      // an address space it cannot read. Migrating one object at a time means
      // the move-list broadcast never sees these.
      //
      // +LBPeerDecision drops the broadcast: CkLocMgr's process residency table
      // answers the "is it on my GPU" question exactly, without anyone having
      // to be told, so the mode decision no longer depends on every PE holding
      // a fresh cache entry. Ordinary messages route by home as always.
      if (!_lb_args.lbPeerDecision())
        BroadcastSingleLocationUpdate(h, acrossNodeToPe);
#endif
    }
  }
  if (loadReceivers <= 0 || minNode == myPos || minNode == -1)
  {
    int minRank = -1;
    double minLoad = 0;
    for (int i = 0; i < nodeSize; i++)
    {
      if (minRank == -1 || pe_load[i] < minLoad)
      {
        minRank = i;
        minLoad = pe_load[i];
      }
    }

    pe_load[minRank] += load;
    if (minRank > 0)
    {
      lbmgr->Migrate(h, rank0PE + minRank);
#if CMK_GLOBAL_LOCATION_UPDATE
      // Within-node move: same process, so no transfer mode changes meaning,
      // but other PEs still cache a location that is now wrong.
      if (!_lb_args.lbPeerDecision())
        BroadcastSingleLocationUpdate(h, rank0PE + minRank);
#endif
    }
  }
#endif
}


// A rank0PE reports its within-node handoffs done (Fix D arms this; until
// then the quiescence detector in WithinNodeLB still drives the transition).
void DiffusionLB::withinNodeReport()
{
  within_owed = true;
  migMaybeDone();
}

// PE 0: every node's within-node handoffs are applied. The same three-way
// choice the quiescence callback used to make.
void DiffusionLB::withinDone()
{
  if (++withinDoneCount < numNodes) return;
  withinDoneCount = 0;
  if (step() == LBSimulation::dumpStep)
    thisProxy.ProcessFinalStats();
  else if (_lb_args.debug() > 0)
    thisProxy.CollectStats();
  else
    thisProxy.ProcessMigrations();
}

void DiffusionLB::MigrationDoneWrapper()
{
  int balancing = 1;
  MigrationDone(balancing);  // call DistBaseLB version

  // End LB timing instrumentation
}

void DiffusionLB::MigrationDone(int balancing)
{
  // The interval the next step judges starts here, once this step's moves
  // are done; and remember whether this step was a revert, so the next one
  // does not judge it. This is the one point every end-of-step path passes
  // through -- the move ledger's resume calls it directly.
  stepEndTime = CmiWallTimer();
  lastStepWasRevert = revertThisStep;
  DistBaseLB::MigrationDone(balancing);
}

void DiffusionLB::printDiffusionTiming()
{
  if (CkMyPe() == 0 && (totalNeighborTime > 0 || totalPseudoLBTime > 0 || totalAcrossTime > 0 || totalWithinTime > 0))
  {
    CkPrintf("\n[DiffusionLB Timing] Neighbor Selection: %.6f seconds\n", totalNeighborTime);
    CkPrintf("[DiffusionLB Timing] Pseudo LB: %.6f seconds\n", totalPseudoLBTime);
    CkPrintf("[DiffusionLB Timing] Across Node: %.6f seconds\n", totalAcrossTime);
    CkPrintf("[DiffusionLB Timing] Within Node: %.6f seconds\n", totalWithinTime);
    CkPrintf("[DiffusionLB Timing] Total: %.6f seconds (sum of phases %.6f)\n", 
             totalLBTime, 
             totalNeighborTime + totalPseudoLBTime + totalAcrossTime + totalWithinTime);
  }
}

#include "DiffusionLB.def.h"
