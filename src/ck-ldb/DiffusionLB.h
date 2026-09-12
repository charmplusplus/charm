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

// The two load accessors, diffusionObjLoad() and diffusionObjCpuLoad(), live in
// DiffusionLoad.h so that the metric, the flow arithmetic and the offline
// simulator can use them without this header.
#include "DiffusionLoad.h"
// The round arithmetic and the plan summary the remap decision is made on.
#include "DiffusionFlow.h"

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
  // Bytes this node's objects sent to each other node last interval, into
  // cost_for_neighbor and `ebytes` (both indexed by node). Runs every step,
  // cached graph or not, because nborCommAdjacent reads the result.
  void countCommToNodes(std::vector<long>& ebytes);
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

  // ---- the remap decision (+LBDiffusionRemapAbove) -------------------------
  //
  // Diffusion executes its plan one hop: a node hands over objects it holds,
  // and cannot forward what it receives in the same step. When the plan is
  // deeper than that -- a hot region whose interior nodes see no gradient --
  // the step cannot reduce the maximum, however many objects it moves, and
  // the balanced partition is one most regions must relocate to reach. That
  // is a scratch-remap's job: partition from scratch, relabel for overlap,
  // migrate once. Measured on the 256-node stencil, one step from 4.49x to
  // 1.12x with 0 detached pieces, where eight diffusion steps and five times
  // the migrations reached 1.9x.
  //
  // After the rounds every node reports what its plan predicts; PE 0 sums,
  // decides on the predicted max/avg against the flag, and tells every PE.
  // On a hand-off DiffusionLB steps aside and a MetisLB it created for the
  // purpose -- outside the manager's balancer sequence -- finishes the step
  // as a central balancer does. Three things make that hand-off clean:
  // this balancer keeps its own hold on the AtSync barrier until the central
  // step's migration-done callback, so no new step can start in between; it
  // forwards the arrivals the manager routes to it, since the hidden balancer
  // is never the manager's current one; and it clears its started flag so
  // its next step is not refused.
  void proceedAfterPlan();
  void planReport(double load, double predicted, double planIn, double planOut);
  void planVerdict(int remap, CkGroupID remapGid);
  void remapHandoff();
  void onCentralStepDone();
  void Migrated(int waitBarrier) override;
  double remapAbove;
  bool remapHandoffActive;
  bool remapGidValid;
  CkGroupID remapGid;
  DiffusionPlanSummary planSummary;
  int planReports;

  // ---- the load dimension (LBLoadDim.h) ------------------------------------
  //
  // Which of the two dimensions the rounds diffuse is decided once per step,
  // job-wide, so that no two nodes equalise different things: each rank-0 PE
  // reports its node's totals, PE 0 resolves, every PE takes the verdict
  // (diffusionLoadDimDevice), and the rank-0 PEs price their node in it
  // before releasing the stats barrier.
  void loadDimReport(double sumHost, double maxHost, double sumDev, double maxDev);
  void loadDimVerdict(int device, double alphaHost, double alphaDev);
  LBCriticality loadDimCrit;  // PE 0: the reports folded so far
  int loadDimReports;
  // This node's totals in both dimensions, from BuildStats.
  double nodeHostSum, nodeHostMax, nodeDevSum, nodeDevMax;
  // Per neighbour, refreshed every round: host time per PE and device time,
  // so the across-node phase can bound what a neighbour takes in the
  // dimension not being diffused.
  std::vector<double> nborHostPerPe, nborDev;
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
  // Whether this node's objects sent anything to neighbour `nbor`'s in the
  // last interval, i.e. the two nodes share a boundary of the application's
  // communication graph. True whenever there is no comm data to say otherwise.
  bool nborCommAdjacent(int nbor) const;
  // Whether load may flow from this node to neighbour `nbor`: nborKeyAdjacent
  // and nborCommAdjacent together, with the comm rule waived for a node that
  // borders none of its neighbours. The pseudo rounds consult this; a
  // neighbour that fails it receives no flow and therefore no objects.
  bool nborFlowAdjacent(int nbor) const;

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
