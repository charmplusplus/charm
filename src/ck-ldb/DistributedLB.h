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
  void GossipLoadInfo(int, int, int, int[], double[]);
  void AvgLoadReduction(double x);
  void DoneGossip();
  void InformMigration(int obj_id, int from_pe, double obj_load, bool force);
  void RecvAck(int obj_id, int assigned_pe, bool can_accept);
  void SendAfterBarrier(CkReductionMsg *msg);

private:
  CProxy_DistributedLB thisProxy;

  int underloaded_pe_count;
  std::vector<int> pe_no;
  std::vector<double> loads;
  std::vector<double> distribution;

  int total_migrates_ack;
  int total_migrates;
  std::vector<MigrateInfo*> migrateInfo;
  LBMigrateMsg* msg;
  
  bool kUseAck;
  double kTransferThreshold;
  int kPartialInfoCount;
  int kMaxTrials;
  int kMaxGossipMsgCount;

  int gossip_msg_count;
  bool lb_started;
  int objs_count;
	double my_load;
	double avg_load;
  double b_load;
  double b_load_per_obj;
  double thr_avg;
	int req_hop;
  int negack_count;

  const DistBaseLB::LDStats* my_stats;

  void InitLB(const CkLBOptions &);
  void SendLoadInfo();
  void LoadBalance();
  void LoadBalance(CkVec<int> &obj_no, CkVec<int> &obj_pe_no);
  void MapObjsToPe(minHeap &objs, CkVec<int> &obj_no, CkVec<int> &obj_pe_no);
	int PickRandReceiverPeIdx() const;
	void CalculateCumulateDistribution();
  void Strategy(const DistBaseLB::LDStats* const stats);

  bool QueryBalanceNow(int step) { return true; };
};

#endif /* _DistributedLB_H_ */
