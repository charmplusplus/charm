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

#ifndef NEIGHBORLB_H
#define NEIGHBORLB_H

#include <BaseLB.h>
#include "WSLB.decl.h"

void CreateWSLB();

class WSLBStatsMsg;

class WSLB : public CBase_WSLB
{
public:
  WSLB();
  WSLB(CkMigrateMessage *m):CBase_WSLB(m) {}
  ~WSLB();
  static void staticAtSync(void*);
  void AtSync(void); // Everything is at the PE barrier

  void ReceiveStats(WSLBStatsMsg *); 		// Receive stats on PE 0
  void ResumeClients();
  void ReceiveMigration(LBMigrateMsg *); 	// Receive migration data

  // Migrated-element callback
  static void staticMigrated(void* me, LDObjHandle h);
  void Migrated(LDObjHandle h);

  void MigrationDone(void);  // Call when migration is complete
  int step() { return mystep; };

  struct LDStats {  // Passed to Strategy
    int from_pe;
    double total_walltime;
    double total_cputime;
    double idletime;
    double bg_walltime;
    double bg_cputime;
    double obj_walltime;
    double obj_cputime;
    double usage;
    int proc_speed;
    CmiBool vacate_me;
  };

protected:
  virtual CmiBool QueryBalanceNow(int);
  virtual LBMigrateMsg* Strategy(LDStats* stats,int count);
  virtual int num_neighbors() {
    return (CkNumPes() > 5) ? 4 : (CkNumPes()-1);
  };
  virtual void neighbors(int* _n) {
    const int me = CkMyPe();
    const int npe = CkNumPes();
    if (npe > 1)
      _n[0] = (me + npe - 1) % npe;
    if (npe > 2)
      _n[1] = (me + 1) % npe;

    int bigstep = (npe - 1) / 3 + 1;
    if (bigstep == 1) bigstep++;

    if (npe > 3)
      _n[2] = (me + bigstep) % npe;
    if (npe > 4)
      _n[3] = (me + npe - bigstep) % npe;
  };

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
  WSLBStatsMsg* AssembleStats();

  int mystep;
  int stats_msg_count;
  WSLBStatsMsg** statsMsgsList;
  LDStats* statsDataList;
  int migrates_completed;
  int migrates_expected;
  LBMigrateMsg** mig_msgs;
  int mig_msgs_received;
  int mig_msgs_expected;
  int* neighbor_pes;
  int receive_stats_ready;
  double start_lb_time;
  double first_step_time;
  double usage;
  double usage_int_err;
  CmiBool vacate;
};

class WSLBStatsMsg : public CMessage_WSLBStatsMsg {
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
  double usage;
  CmiBool vacate_me;
}; 


#endif /* NEIGHBORLB_H */


/*@}*/
