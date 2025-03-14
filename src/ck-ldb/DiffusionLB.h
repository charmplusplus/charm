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

#include "DiffusionMetric.h"
#include "Heap_helper.C"

#include <queue>
#include <unordered_map>
#include <vector>

#include "DiffusionLB.decl.h"

#include "DiffusionJSON.h"

#define SELF_IDX NUM_NEIGHBORS
#define EXT_IDX NUM_NEIGHBORS + 1
#define NUM_NEIGHBORS 2

void CreateDiffusionLB();

/// for backward compatibility
typedef LBMigrateMsg NLBMigrateMsg;

class DiffusionLB : public CBase_DiffusionLB
{
public:
  DiffusionLB_SDAG_CODE DiffusionLB(const CkLBOptions&);
  DiffusionLB(CkMigrateMessage* m);
  ~DiffusionLB();

  // void MigratedHelper(LDObjHandle h, int waitBarrier);
  // void Migrated(LDObjHandle h, int waitBarrier = 1);
  void createCommList();
  void findNBors(int do_again);
  void findNBorsRound(int do_again);
  void startFirstRound();
  void proposeNbor(int nborId);
  void okayNbor(int agree, int nborId);
  void statsAssembled();
  void startStrategy();
  void next_phase(int val);
  void sortArr(long arr[], int n, int* nbors);

  void MigrationDoneWrapper();  // Call when migration is complete
  void ReceiveStats(CkMarshalledCLBStatsMessage&& data);
  void ReceiveFinalStats(std::vector<bool> isMigratable, std::vector<int> from_proc,
                         std::vector<int> to_proc, std::vector<LDCommData> commData,
                         int n_migrateobjs,
                         std::vector<std::vector<LBRealType>> positions);

  void buildMSTinRounds(double best_weight, int best_from, int best_to);
  void next_MSTphase(double newcost, int newparent, int newto);

  void LoadReceived(int objId, int fromPE);
  void AcrossNodeLB();

  void ProcessMigrations();
  void WithinNodeLB();

  void LoadMetaInfo(LDObjHandle h, double load);

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

  // phase 1: build neighbor list --------------------------------
  int rank0_barrier_counter;
  int neighborCount;
  std::vector<int> sendToNeighbors;  // Neighbors to which curr node has to send load.
  int* nbors;

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
  std::vector<double> loadNeighbors;
  double avgLoadNeighbor;  // Average load of the neighbor group
  double my_pseudo_load;

  int pseudo_itr;  // iteration count
  int temp_itr;

  // phase 3: across node LB --------------------------------
  void buildObjComms(int nobjs);
  void buildGainValues(int nobjs);
  void buildGainValuesNbor(int nobjs, int nbor);

  int getBestNeighbor();
  int getBestObject(int nbor);

  int* gain_val;
  int loadReceivers;

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

  // main entry point
  void Strategy(const DistBaseLB::LDStats* const stats);

  // helper functions
  int findNborIdx(int node);
  double avgNborLoad();  // used in pseudoLB only
  int GetPENumber(int& obj_id);
  void BuildStats();
  CLBStatsMsg* AssembleStats();
  void AddToList(CLBStatsMsg* m, int rank);

  // Cascading migrations / not used (because cascading migration doesn't make sense?)
  std::vector<LDObjHandle> objectHandles;
  std::vector<double> objectLoads;
  int FindObjectHandle(LDObjHandle h);
  void CascadingMigration(LDObjHandle h, double load);
};

#endif /* _DistributedLB_H_ */
