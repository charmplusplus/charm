/**
 * Author: gplkrsh2@illinois.edu (Harshitha Menon)
 *
 * A distributed load balancer which consists of two phases.
 * 1. Information propagation
 * 2. Probabilistic transfer of load
 *
 * Information propagation is done using Gossip protocol where the information
 * about the underloaded processors in the system is spread to the overloaded
 * processors.
 * Once the overloaded processors receive the partial information about the
 * underloaded processors in the system, it does probabilistic load transfer.
 * The probability of a PE receiving load is inversely proportional to its
 * current load.
*/

#ifndef _DISTLB_H_
#define _DISTLB_H_

#include "DistBaseLB.h"
#include "DistributedLB.decl.h"

#include "ckheap.h"

#include <vector>

void CreateDistributedLB();

class DistributedLB : public CBase_DistributedLB {
public:
  DistributedLB(const CkLBOptions &);
  DistributedLB(CkMigrateMessage *m);
  static void initnodeFn(void);

  void GossipLoadInfo(int, int, int[], double[]);
  void LoadReduction(CkReductionMsg* msg);
  void AfterLBReduction(CkReductionMsg* msg);
  void DoneGossip();
  void InformMigration(int obj_id, int from_pe, double obj_load, bool force);
  void RecvAck(int obj_id, int assigned_pe, bool can_accept);
  void turnOn();
  void turnOff();

private:
  // Load information obtained via gossipping
  int underloaded_pe_count;
  std::vector<int> pe_no;
  std::vector<double> loads;
  std::vector<double> distribution;

  minHeap* objs;

  int total_migrates_ack;
  int total_migrates;
  std::vector<MigrateInfo*> migrateInfo;

  // Constant variables for configuring how the strategy works
  bool kUseAck;
  int kPartialInfoCount;
  int kMaxTrials;
  int kMaxGossipMsgCount;
  int kMaxObjPickTrials;
  int kMaxPhases;
  double kTargetRatio;

  // Global stats about the load gather from reductions
  double avg_load;
  double max_load;
  double load_ratio;

  // Information about this PEs load
  double my_load;
  double init_load;
  double b_load;
  double b_load_per_obj;

  // Control flow variables
  bool lb_started;
  int phase_number;
  int gossip_msg_count;
  int objs_count;
  int negack_count;
  double transfer_threshold;

  double start_time;

  const DistBaseLB::LDStats* my_stats;

  void InitLB(const CkLBOptions &);
  void SendLoadInfo();
  void LoadBalance();
  void LoadBalance(CkVec<int> &obj_no, CkVec<int> &obj_pe_no);
  void MapObjsToPe(minHeap *objs, CkVec<int> &obj_no, CkVec<int> &obj_pe_no);
	int PickRandReceiverPeIdx() const;
	void CalculateCumulateDistribution();
  void Strategy(const DistBaseLB::LDStats* const stats);
  void Setup();
  void Cleanup();
  void PackAndSendMigrateMsgs();
  void StartNextLBPhase();
  void DoneWithLBPhase();

  // TODO: Should this use the lb_started flag?
  bool QueryBalanceNow(int step) { return true; };
};

#endif /* _DistributedLB_H_ */
