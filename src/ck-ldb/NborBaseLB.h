/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef NBORBASELB_H
#define NBORBASELB_H

#include "BaseLB.h"
#include "NborBaseLB.decl.h"

#include "topology.h"

void CreateNborBaseLB();

/// for backward compatibility
typedef LBMigrateMsg NLBMigrateMsg;

class NLBStatsMsg;

class NborBaseLB : public BaseLB
{
private:
  CProxy_NborBaseLB  thisProxy;
  LBTopology         *topo;
public:
  NborBaseLB(const CkLBOptions &);
  NborBaseLB(CkMigrateMessage *m):BaseLB(m) {}
  ~NborBaseLB();

  static void staticAtSync(void*);
  void AtSync(void); // Everything is at the PE barrier

  void ReceiveStats(CkMarshalledNLBStatsMessage &m); 		// Receive stats on PE 0
  void ResumeClients(CkReductionMsg *msg);
  void ResumeClients(int balancing);
  void ReceiveMigration(LBMigrateMsg *); 	// Receive migration data

  // Migrated-element callback
  static void staticMigrated(void* me, LDObjHandle h, int waitBarrier);
  void Migrated(LDObjHandle h, int waitBarrier);

  void MigrationDone(int balancing);  // Call when migration is complete

  struct LDStats {  // Passed to Strategy
    int from_pe;
    double total_walltime;
    double total_cputime;
    double idletime;
    double bg_walltime;
    double bg_cputime;
    double obj_walltime;
    double obj_cputime;
    int pe_speed;
    CmiBool available;
    CmiBool move;

    int n_objs;
    LDObjData* objData;
    int n_comm;
    LDCommData* commData;

    inline void clearBgLoad() {
      bg_walltime = bg_cputime = idletime = 0.0;
    }
  };

protected:
  virtual CmiBool QueryBalanceNow(int) { return CmiTrue; };  
  virtual CmiBool QueryMigrateStep(int) { return CmiTrue; };  
  virtual LBMigrateMsg* Strategy(LDStats* stats, int n_nbrs);

  int NeighborIndex(int pe);   // return the neighbor array index

  LDStats myStats;

private:
  void FindNeighbors();
  NLBStatsMsg* AssembleStats();

  int stats_msg_count;
  NLBStatsMsg** statsMsgsList;
  LDStats* statsDataList;
  int migrates_completed;
  int migrates_expected;
  LBMigrateMsg** mig_msgs;
  int mig_msgs_received;
  int mig_msgs_expected;
  int* neighbor_pes;
  int receive_stats_ready;
  double start_lb_time;
};

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

#endif /* NBORBASELB_H */

/*@}*/
