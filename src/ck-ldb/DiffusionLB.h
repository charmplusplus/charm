/*Distributed Graph Refinement Strategy*/
#ifndef _DISTLB_H_
#define _DISTLB_H_

#include "BaseLB.h"
#include "CentralLB.h"
#include "DistBaseLB.h"
#include "TopoManager.h"
#include "charm++.h"
#include "ckgraph.h"

#include "ckheap.h"
#include "topology.h"

#include "Heap_helper.C"

#include <queue>
#include <unordered_map>
#include <vector>

#include "ckmulticast.h"

#include "DiffusionLB.decl.h"

int numPes;

void CreateDiffusionLB();

// DiffusionLB balances two different resources at its two levels, and they are not
// interchangeable:
//
//   Across nodes. Under one process per device a node IS a GPU, so the scarce
//   resource is device occupancy and the quantity to equalise is the sum of GPU
//   time over the node's objects. Selected with +LBDiffusionGpuDim.
//
//   Within a node. The PEs of a process SHARE that device, so moving a chare from
//   one PE to another does not relieve the GPU by a microsecond -- the kernel still
//   runs on the same card. Only host-side work relocates. The intra-node heap must
//   therefore balance CPU time alone; charging it GPU time would have it believe it
//   is rebalancing something it structurally cannot.
//
// Hence two accessors. diffusionObjLoad() is the diffused dimension (what crosses
// node boundaries); diffusionObjCpuLoad() is always host time (what moves between
// PEs inside a node).
//
// Note deliberately NOT max(cpu, gpu): summing per-object maxima over-counts every
// object whose two timelines overlap. A node's step time is
// max(sum of gpuTime, max over PEs of sum of wallTime) -- aggregate first, then take
// the max, never the other way round.

// The dimension diffused across nodes. Defaults to host time so that CPU-only
// workloads keep working; +LBDiffusionGpuDim switches it to device occupancy for
// GPU-bound runs. An automatic choice would have to be identical on every node --
// nodes disagreeing about which resource they are equalising would diffuse
// incoherently -- so it is an explicit flag rather than a local heuristic.
static inline double diffusionObjLoad(const LDObjData& o)
{
#if CMK_CUDA
  if (_lb_args.diffusionGpuDim()) return o.gpuTime;
#endif
  return o.wallTime;
}

// Host time, always. Used for per-PE totals and the within-node heap, which can only
// ever move host work between PEs that share a device.
static inline double diffusionObjCpuLoad(const LDObjData& o) { return o.wallTime; }

/// for backward compatibility
typedef LBMigrateMsg NLBMigrateMsg;

// Multicast that seeds the pseudo-LB section: its CkMcastBaseMsg base carries
// the cookie each member contributes against, and it hands over the multicast
// manager's id so members can reach their local branch.
class PseudoRoundMsg : public CkMcastBaseMsg, public CMessage_PseudoRoundMsg
{
public:
  CkGroupID mcastGid;
  // Verdict payload, used when this message carries the convergence result back
  // to the section. Unused (and ignored) by the round-start multicast.
  double maxRatio;
};

class DiffusionLB : public CBase_DiffusionLB
{
public:
  // Entry methods for the pseudo-LB section reduction; public so the
  // generated dispatch code can reach them.
  void beginPseudoRounds();
  void pseudoRoundStart(PseudoRoundMsg* m);
  void pseudoVerdictRoot(double maxRatio);
  void pseudoConvergeResult(PseudoRoundMsg* m);
  void pseudoMetricContribute(double metric);
  DiffusionLB_SDAG_CODE DiffusionLB(const CkLBOptions&);
  DiffusionLB(CkMigrateMessage* m);
  ~DiffusionLB();
    static void printDiffusionTiming();


  // void MigratedHelper(LDObjHandle h, int waitBarrier);
  // void Migrated(LDObjHandle h, int waitBarrier = 1);
  void createCommList();
  void findNBors(int do_again);
  void beginMST();
  void findNBorsRound();
  void startFirstRound();
  void proposeNbor(int nborId);
  void askNbor(int nbor, int rnd);
  void okayNbor(int agree, int nborId);
  void ackNbor(int nbor);
  void ackNborDone();
  // Counting barrier that replaces the quiescence detector between the pseudo
  // rounds and AcrossNodeLB. The convergence reduction already proves every
  // round message was processed before the verdict goes out, so all that is
  // left to know is that every member has left the loop.
  void roundsDone();
  int roundsDoneCount;
  // Migration-handoff completion counting, replacing the quiescence detectors
  // between AcrossNodeLB -> WithinNodeLB -> ProcessMigrations. Each
  // LoadMetaInfo/LoadReceived a rank0PE sends is acked by its receiver after
  // processing; the phase's barrier contribution is held until every ack is
  // back, so barrier completion proves every handoff has been applied.
  int mig_acksOut;
  bool across_owed;
  bool within_owed;
  void migMsgAck();
  void migMaybeDone();
  void acrossDone();
  void nbrAcrossDone();
  int acrossDoneCount;
  // Across-node completion, tracked against this node's own neighbours instead
  // of the whole job. Load only ever moves between neighbours, so a node needs
  // to know that ITS neighbours have stopped sending to it -- not that every
  // node everywhere has finished. acrossSelfDone is this node's own handoffs
  // acked; acrossNbrDoneCount counts the neighbours that told us the same.
  int acrossNbrDoneCount;
  bool acrossSelfDone;
  void maybeStartWithin();
  // Within-node completion, tracked the same way. A node's within-node phase
  // can RETARGET a token it received across nodes, and that retarget is a
  // LoadReceived to the donor PE on the node that sent the token -- so a node's
  // own move list is not final until every neighbour it sent tokens to has
  // finished its within-node phase. Only neighbours can hold its tokens.
  bool withinSelfDone;
  int withinNbrDoneCount;
  void nbrWithinDone();
  void maybeStartMigrations();
  void withinDone();
  int withinDoneCount;
  void withinNodeReport();
  // The neighbour graph survives across LB steps: topology-derived and
  // comm-derived neighbourhoods change slowly, and rebuilding the graph is the
  // most expensive phase of a step. Set once the first step's rounds start
  // (the graph is final by then); CHARM_DIFFUSION_GRAPH_REBUILD=1 restores a
  // rebuild every step.
  bool hs_graphCached;
  void statsAssembled();
  void startStrategy();
  void startStrategyBarrier();
  void next_phase(int val);
  void sortArr(long arr[], int n, int* nbors);

  void startMSTBarrier();

  // pseudolb_barrier removed with the global convergence check (DiffusionPseudo.C)

  void MigrationDoneWrapper();  // Call when migration is complete
  void ReceiveStats(CkMarshalledCLBStatsMessage&& data);
  void ReceiveFinalStats(std::vector<bool> isMigratable, std::vector<int> from_proc,
                         std::vector<int> to_proc, int n_migrateobjs,
                         std::vector<std::vector<LBRealType>> positions,
                         std::vector<double> load,
                         std::vector<LDCommData> commData);

  void buildMSTinRounds(double best_weight, int best_from, int best_to);
  void next_MSTphase(double newcost, int newparent, int newto);

  void LoadReceived(int objId, int fromPE, int ackPE);
  void update_peload(int rank, double load);
  void AcrossNodeLB();

  void ProcessMigrations();
  void ProcessFinalStats();
  void CollectStats();
  void WithinNodeLB();

  void print_max_load(double max);
    void print_avg_load(double sum);
    void print_external_comm(double sum);
    void print_internal_comm(double sum);
    void print_num_migrations(int sum);

  // A token: an object another PE is handing this node (or this PE). `load` is
  // host time, `gload` the diffused dimension, `key` the object's 1-D ordering
  // key or NaN when it registered none -- carried so a token can take part in
  // the within-node interval repartition instead of forcing it to be skipped.
  void LoadMetaInfo(LDObjHandle h, int objId, double load, double gload, int senderPE,
                    int only_mcount, int ackPE, double key);

protected:
  virtual bool QueryBalanceNow(int) { return true; };

private:
  CProxy_DiffusionLB thisProxy;

  // phase 0: set up stats structures --------------------------------
  CLBStatsMsg* statsmsg;
  CkMarshalledCLBStatsMessage* marshmsg;
  CLBStatsMsg** statsList;  // used in DiffusionHelper
  BaseLB::LDStats* nodeStats;
  DistBaseLB::LDStats* myStats;

  BaseLB::LDStats* fullStats;

  int statsReceived;

  std::vector<int> numObjects;
  std::vector<int> prefixObjects;
  std::vector<double> pe_load;

  // general state --------------------------------
  double my_load;
  double my_loadAfterTransfer;
  int rank0PE;
  int nodeSize;
  int numNodes;
  int myNodeId;

  double myNodeInternalBytes;
  double myNodeExternalBytes;

  double num_migrations;

  // centroid setup --------------------------------
  std::vector<std::vector<LBRealType>> allNodeCentroids;
  std::vector<int> allNodeObjCount;
  std::vector<double> allNodeDistances;
  std::vector<std::vector<LBRealType>> nborCentroids;
  std::vector<double> nborDistances;
  std::vector<int> nborObjCount;
  std::vector<LBRealType> myCentroid;
  int position_dim;
  int centReceiveNode;

  void addNeighbor(int nbor);
  // Connectivity backbone for the diffusion graph; replaces the MST. See the
  // definition in DiffusionNeighbors.C for why.
  void buildRingBackbone();
  void pairedSort(int* A, std::vector<double> B);

  // phase 1: build neighbor list --------------------------------
  int rank0_barrier_counter;
  int neighborCount;
  std::vector<int> sendToNeighbors;  // Neighbors to which curr node has to send load.
  int* node_idx;//nbors;

  std::vector<int> mstVisitedPes;
  std::unordered_map<int, double> cost_for_neighbor;

  double best_weight;
  int best_from;
  int best_to;
  int all_tos_negative;

  bool visited;
  int pick;
  int round;
  int requests_sent;
  int acks, max;

  // Handshake completion counting, which replaces the quiescence detection that
  // used to drain the ask/okay/ack exchange (see hsMaybeAdvance in
  // DiffusionNeighbors.C). All on the node's rank0PE.
  int hs_asksOut;      // askNbor sent, okayNbor not yet back
  int hs_confirmOut;   // ackNbor sent, ackNborDone not yet back
  bool hs_phaseOwed;   // this round's next_phase() contribution is being held
  int hs_phaseVal;     //   ... and the value it will carry
  bool hs_barrierOwed; // the final startStrategyBarrier() is being held
  void hsMaybeAdvance();

  // phase 2: pseudo load balancing --------------------------------
  void PseudoLoadBalancing();

  std::vector<double> toSendLoad;
  std::vector<double> toReceiveLoad;
  // Flow sent to each neighbour in the previous pseudo-LB round. Second-order
  // diffusion carries a fraction of it into this round as momentum; see
  // PseudoLoadBalancing.
  std::vector<double> prevRoundToSend;
  std::vector<double> loadNeighbors;
  double avgLoadNeighbor;  // Average load of the neighbor group
  double my_pseudo_load;

  int pseudo_itr;  // iteration count
  int temp_itr;
  bool pseudo_done;
  // Global convergence state for the pseudo-LB rounds. pseudo_metric is this
  // node's "how much load do I still want to shift", expressed as a fraction of
  // its own load so nodes of different sizes are comparable; the reduction takes
  // the max across nodes and every PE gets the same verdict back.
  double pseudo_metric;
  // Section over the one PE per node that actually diffuses, plus the multicast
  // cookie its reductions run on. Reducing over these numNodes members rather
  // than the whole group keeps the collective the width of the algorithm, and
  // keeps PEs with no part in diffusion out of its lockstep entirely.
  CProxySection_DiffusionLB pseudoSection;
  CkSectionInfo pseudoCookie;
  CkGroupID pseudoMcastGid;
  bool pseudoSectionBuilt;

  double prev_pseudo_load;  // my_pseudo_load at the end of the previous round
  bool pseudo_converged;
  // PE 0's counting reduction over the diffusing PEs' convergence metrics, one
  // per round. Replaces the CkMulticast section reduction and verdict
  // multicast: those travel as ordinary messages and sat behind the
  // application's queue on every hop under +LBAsync, which is what stretched a
  // 1 ms pseudo phase to 100-200 ms of wall time. numNodes point-to-point
  // expedited messages into PE 0 and out again cost nothing at this width.
  int pseudoContribCount;
  double pseudoMaxMetric;
  // Only one PE per node drives diffusion; the rest join the convergence
  // reduction with a neutral value so the collective is over the whole group.
  bool isPseudoRoot;

  // phase 3: across node LB --------------------------------
  void buildObjComms(int nobjs);
  void buildGainValues(int nobjs);
  void buildGainValuesNbor(int nobjs, int nbor);

  int getBestNeighbor();
  int getBestObject(int nbor);

  int* gain_val;
  int loadReceivers;
  int *holds;

  std::vector<std::vector<int>> objectComms;

  // heap things
  std::vector<CkVertex> objs;
  std::vector<int> obj_heap;  // TODO: replace with ckheap
  std::vector<int> heap_pos;
  void InitializeObjHeap(int size);
  std::vector<CkVertex> objects;  // this is only used to pass in to ObjCompareOperator,
                                  // but not initialzied??

  // phase 4: within node LB --------------------------------
  double averagePE();

  // phase 5: migration --------------------------------
  std::vector<MigrateInfo*> migrateInfo;
  int total_migrates;
  int total_crossnode_migrates;

  // ---- decision floor and ordering keys ----------------------------------
  //
  // The floor. An imbalance below effMinImbalance (a fraction of the mean) is
  // treated as noise at every level: the pseudo rounds send nothing for it,
  // the across-node phase sheds nothing for it, the within-node phase moves
  // nothing for it. +LBDiffusionMinImbalance, fixed for the run.
  double effMinImbalance;

  // 1-D ordering keys. When every object on this node registered a position
  // of width 1, the node holds an interval of an ordering and the rules that
  // keep it one (only the ends leave, and only toward the neighbour on that
  // side) apply to both metrics. Neighbours' intervals travel with their
  // loads in the pseudo rounds.
  bool keyed1D;
  double myKeyLo, myKeyHi;
  // The node's mean per-object load in the diffused dimension (BuildStats).
  // An object measured at zero retires this much of a shed budget, so that a
  // budget is retired in proportion to objects moved rather than never.
  double objLoadFloor;
  std::vector<double> nborKeyLo, nborKeyHi;
  std::vector<double> objectKeys;    // per token, parallel to objectLoads
  std::vector<double> objectGLoads;  // per token, diffused dimension
  static double keyOf(const LDObjData& od);
  // Marks the objects that may leave for neighbour `nbor` this pop: the low
  // end of this node's interval if the neighbour lies below it, the high end
  // if above, both when that cannot be told. Everything else is refused.
  void allowedEndsFor(int nbor, std::vector<char>& allowed);
  // Whether neighbour `nbor` holds the interval next to this node's, with no
  // other neighbour's interval between them. Load may only flow between
  // adjacent intervals: the ring backbone joins the first and last node too,
  // and a move over that edge lands the top of the key space next to the
  // bottom, which is what inflates a domain box. An unkeyed or overlapping
  // neighbour counts as adjacent (nothing to be preserved there).
  bool nborKeyAdjacent(int nbor) const;

  // Diffusion-specific timing instrumentation
  static double totalNeighborTime;
  static double totalPseudoLBTime;
  static double totalAcrossTime;
  static double totalWithinTime;
  static double phaseStartTime;
  static double totalStartTime;
  static double totalLBTime;
  
  static void startOverallTiming() { totalStartTime = CmiWallTimer(); }
  static void startNeighborTiming() { phaseStartTime = CmiWallTimer(); }
  static void endNeighborTiming() { totalNeighborTime += CmiWallTimer() - phaseStartTime; }
  static void startPseudoLBTiming() { phaseStartTime = CmiWallTimer(); }
  static void endPseudoLBTiming() { totalPseudoLBTime += CmiWallTimer() - phaseStartTime; }
  static void startAcrossTiming() { phaseStartTime = CmiWallTimer(); }
  static void endAcrossTiming() { totalAcrossTime += CmiWallTimer() - phaseStartTime;}
  static void startWithinTiming() { phaseStartTime = CmiWallTimer(); }
  static void endWithinTiming() { 
    totalWithinTime += CmiWallTimer() - phaseStartTime;
    totalLBTime += CmiWallTimer() - totalStartTime; 
  }

  // main entry point
  void Strategy(const DistBaseLB::LDStats* const stats);

  // helper functions
  int findNborIdx(int node);
  double avgNborLoad();  // used in pseudoLB only
  int GetRank(int obj_id);
  void BuildStats();
  CLBStatsMsg* AssembleStats();
  void AddToList(CLBStatsMsg* m, int rank);

  // Cascading migrations / not used (because cascading migration doesn't make sense?)
  std::vector<LDObjHandle> objectHandles;
  std::vector<double> objectLoads;
  std::vector<int> objectSrcIds;
  std::vector<int> objSenderPEs;
  std::unordered_map<int, MigrateInfo*> mig_id_map;
  int FindObjectHandle(LDObjHandle h);
  void CascadingMigration(LDObjHandle h, double load);

   void processReceiveCentroid(int node, std::vector<LBRealType> centroid, int objCount);
   void resetVarsMST();
   void initializeCentroid();
   void finishCentroidList();

   int writeStatsMsgs(BaseLB::LDStats* statsData);
   int writeStatsMsgsJSON(BaseLB::LDStats* statsData);
};

#endif /* _DistributedLB_H_ */
