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

class DiffusionLB : public CBase_DiffusionLB
{
public:
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

  void LoadReceived(int objId, int fromPE);
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

  void LoadMetaInfo(LDObjHandle h, int objId, double load, int senderPE, int only_mcount);

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
