#ifndef NEIGHBORLB_H
#define NEIGHBORLB_H

#include <LBDatabase.h>
#include "NeighborLB.decl.h"


void CreateNeighborLB();

class NLBStatsMsg;
class NLBMigrateMsg;

class NeighborLB : public Group
{
public:
  NeighborLB();
  ~NeighborLB();
  static void staticAtSync(void*);
  void AtSync(void); // Everything is at the PE barrier

  void ReceiveStats(NLBStatsMsg *); 		// Receive stats on PE 0
  void ResumeClients();
  void ReceiveMigration(NLBMigrateMsg *); 	// Receive migration data

  // Migrated-element callback
  static void staticMigrated(void* me, LDObjHandle h);
  void Migrated(LDObjHandle h);

  void MigrationDone(void);  // Call when migration is complete
  int step() { return mystep; };

  struct MigrateInfo {  // Used in NLBMigrateMsg
    LDObjHandle obj;
    int from_pe;
    int to_pe;
  };

  struct LDStats {  // Passed to Strategy
    int from_pe;
    double total_walltime;
    double total_cputime;
    double idletime;
    double bg_walltime;
    double bg_cputime;
    double obj_walltime;
    double obj_cputime;
    int proc_speed;
  };

protected:
  virtual CmiBool QueryBalanceNow(int) { return CmiTrue; };  
  virtual NLBMigrateMsg* Strategy(LDStats* stats,int count);

  virtual int num_neighbors() {
    if (CmiNumPes() > 2) return 2;
    else return (CmiNumPes()-1);
  };

  virtual void neighbors(int* _n) {
    _n[0] = (CmiMyPe() + CmiNumPes() -1) % CmiNumPes();
    _n[1] = (CmiMyPe() + 1) % CmiNumPes();
  };

  LBDatabase* theLbdb;
  struct {
    int proc_speed;
    double total_walltime;
    double total_cputime;
    double idletime;
    double bg_walltime;
    double bg_cputime;
    int obj_data_sz;
    LDObjData* objData;
    int comm_data_sz;
    LDCommData* commData;
    double obj_walltime;
    double obj_cputime;
  } myStats;

private:
  void FindNeighbors();
  NLBStatsMsg* AssembleStats();

  int mystep;
  int stats_msg_count;
  NLBStatsMsg** statsMsgsList;
  LDStats* statsDataList;
  int migrates_completed;
  int migrates_expected;
  NLBMigrateMsg** mig_msgs;
  int mig_msgs_received;
  int mig_msgs_expected;
  int* neighbor_pes;
  int receive_stats_ready;
  double start_lb_time;
};

class NLBStatsMsg : public CMessage_NLBStatsMsg {
public:
  int from_pe;
  int serial;
  int proc_speed;
  double total_walltime;
  double total_cputime;
  double idletime;
  double bg_walltime;
  double bg_cputime;
  double obj_walltime;
  double obj_cputime;
}; 

class NLBMigrateMsg : public CMessage_NLBMigrateMsg {
public:
  int n_moves;
  NeighborLB::MigrateInfo* moves;

  // Other methods & data members 
  
  static void* alloc(int msgnum, size_t size, int* array, int priobits); 
  static void* pack(NLBMigrateMsg* in); 
  static NLBMigrateMsg* unpack(void* in); 
}; 

#endif /* NEIGHBORLB_H */
