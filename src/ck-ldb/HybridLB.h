/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef HYBRIDLB_H
#define HYBRIDLB_H

#include "BaseLB.h"
#include "CentralLB.h"
#include "HybridLB.decl.h"

#include "topology.h"

void CreateHybridLB();

/// for backward compatibility
typedef LBMigrateMsg NLBMigrateMsg;

class HybridLB : public BaseLB
{
public:
  HybridLB(const CkLBOptions &);
  HybridLB(CkMigrateMessage *m):BaseLB(m) {}
  ~HybridLB();

  static void staticAtSync(void*);
  void AtSync(void); // Everything is at the PE barrier

  void ReceiveStats(CkMarshalledCLBStatsMessage &m); 	// Receive stats on PE 0
  void ResumeClients(CkReductionMsg *msg);
  void ResumeClients();
  void ReceiveMigration(LBMigrateMsg *); 	// Receive migration data

  // Migrated-element callback
  static void staticMigrated(void* me, LDObjHandle h, int waitBarrier);
  void Migrated(LDObjHandle h, int waitBarrier);

  void MigrationDone(void);  // Call when migration is complete
  void NotifyMigrationDone(void);  // Call when migration is complete
  void Loadbalancing(void);	// start load balancing
  int step() { return mystep; };

private:
  CProxy_HybridLB  thisProxy;
  LBTopology       *topo;
  int              parent;
  CkVec<int>       children;
  int              foundNeighbors;
  int		   recvslot;
  int		   loadbalancing;
protected:
  virtual CmiBool QueryBalanceNow(int) { return CmiTrue; };  
  virtual CmiBool QueryMigrateStep(int) { return CmiTrue; };  
  virtual LBMigrateMsg* Strategy(LDStats* stats,int count);

  int NeighborIndex(int pe);   // return the neighbor array index

  LDStats *statsData;

private:
  void FindNeighbors();
  CLBStatsMsg* AssembleStats();
  void buildStats();
  CLBStatsMsg * buildCombinedLBStatsMessage();
  void depositLBStatsMessage(CLBStatsMsg *msg);

  int mystep;
  int stats_msg_count;
  CLBStatsMsg** statsMsgsList;
  LDStats* statsDataList;
  int migrates_completed;
  int migrates_expected;
  LBMigrateMsg** mig_msgs;
  int mig_msgs_received;
  int mig_msgs_expected;
  int receive_stats_ready;
  int cur_ld_balancer;
  double start_lb_time;
};

/*
class NLBStatsMsg {
public:
  int from_pe;
  int serial;
  int pe_speed;
  double total_walltime;
  double total_cputime;
  double idletime;
  double bg_walltime;
  double bg_cputime;
  double obj_walltime;   // may not needed
  double obj_cputime;   // may not needed
  int n_objs;
  LDObjData *objData;
  int n_comm;
  LDCommData *commData;
public:
  NLBStatsMsg(int osz, int csz);
  NLBStatsMsg(NLBStatsMsg *s);
  NLBStatsMsg()  {}
  ~NLBStatsMsg();
  void pup(PUP::er &p);
}; 
*/

#endif /* NBORBASELB_H */

/*@}*/
