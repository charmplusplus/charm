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

#include <queue>
#include <unordered_map>
#include <vector>

#include "DiffusionLB.decl.h"

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

  // void ReceiveLoadInfo(int itr, double load, int node);
  void MigrationDoneWrapper();  // Call when migration is complete
  void ReceiveStats(CkMarshalledCLBStatsMessage&& data);

  void buildMSTinRounds(double best_weight, int best_from, int best_to);
  void next_MSTphase(double newcost, int newparent, int newto);

  // void LoadTransfer(double load, int initPE, int objId);
  void LoadReceived(int objId, int fromPE);
  // void PseudoLoad(int itr, double load, int node);
  void AcrossNodeLB();

  // void MigrationInfo(int to, int from);

  // TODO: Can be made private or not entry methods as well.
  void ProcessMigrations();
  void WithinNodeLB();

  // void ResumeClients(CkReductionMsg* msg);
  // void ResumeClients(int balancing);
  // void CallResumeClients();
  // void PrintDebugMessage(int len, double* result);
  void LoadMetaInfo(LDObjHandle h, double load);

protected:
  virtual bool QueryBalanceNow(int) { return true; };

private:
  CProxy_DiffusionLB thisProxy;

  // phase 0: set up stats structures
  CLBStatsMsg* statsmsg;
  CkMarshalledCLBStatsMessage* marshmsg;
  CLBStatsMsg** statsList;  // used in DiffusionHelper
  BaseLB::LDStats* nodeStats;
  DistBaseLB::LDStats* myStats;

  // general info
  double my_load;
  double my_loadAfterTransfer;
  int rank0PE;
  int nodeSize;
  int numNodes;

  // phase 1: build neighbor list
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

  // phase 2: pseudo load balancing
  std::vector<double> toSendLoad;
  std::vector<double> toReceiveLoad;
  std::vector<double> loadNeighbors;
  double avgLoadNeighbor;  // Average load of the neighbor group
  double my_pseudo_load;

  // aggregate load received
  int itr;  // iteration count
  int temp_itr;
  int pick;
  int notif;
  int statsReceived;
  int loadReceived;
  int do_again = 1;
  int round, requests_sent;
  int myNodeId;

  std::vector<int> numObjects;
  std::vector<int> prefixObjects;
  std::vector<double> pe_load;
  std::vector<double> pe_loadBefore;

  int acks, max;

  int edgeCount;
  std::vector<int> edge_indices;
  // const DistBaseLB::LDStats* statsData;

  // heap
  int* obj_arr;
  int* gain_val;
  std::vector<CkVertex> objs;
  std::vector<int> obj_heap;  // TODO: replace with ckheap
  std::vector<int> heap_pos;

  int loadReceivers;
  std::vector<bool> balanced;

  // migration
  int total_migrates;
  int total_migratesActual;
  int migrates_completed;
  int migrates_expected;
  std::vector<MigrateInfo*> migrateInfo;
  std::vector<int> migratedTo;
  std::vector<int> migratedFrom;
  std::vector<CkVertex> objects;
  std::vector<std::vector<int>> objectComms;
  int finalBalancing;  // TODO get rid of this

  // stats
  double strat_end_time;
  double start_lb_time;
  double end_lb_time;
  // debug messages vars and functions.
  // load
  double maxB;
  double minB;
  double avgB;
  double maxA;
  double minA;
  double avgA;
  int maxPEB;
  int maxPEA;
  int minPEB;
  int minPEA;
  // communication
  double internalBefore;
  double externalBefore;
  double internalAfter;
  double externalAfter;
  double internalBeforeFinal;
  double externalBeforeFinal;
  double internalAfterFinal;
  double externalAfterFinal;
  int receivedStats;
  int migrates;      // number of objects migrated across node
  int migratesNode;  // number of objects migrated within node

  // helper functions
  CLBStatsMsg* AssembleStats();
  void AddToList(CLBStatsMsg* m, int rank);
  void BuildStats();
  int findNborIdx(int node);
  bool AggregateToSend();
  double avgNborLoad();

  // state helpers
  int GetPENumber(int& obj_id);
  double averagePE();

  // Cascading migrations
  std::vector<LDObjHandle> objectHandles;
  std::vector<double> objectLoads;
  int FindObjectHandle(LDObjHandle h);
  void CascadingMigration(LDObjHandle h, double load);

  // main functions
  void Strategy(const DistBaseLB::LDStats* const stats);
  void PseudoLoadBalancing();

  // Heap functions
  void InitializeObjHeap(BaseLB::LDStats* stats, int* obj_arr, int size, int* gain_val);
};

#endif /* _DistributedLB_H_ */
