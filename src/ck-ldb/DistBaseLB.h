/**
 * Author: gplkrsh2@illinois.edu (Harshitha Menon)
 * Base class for distributed load balancer.
*/

#ifndef _DISTBASELB_H
#define _DISTBASELB_H

#include "BaseLB.h"
#include "DistBaseLB.decl.h"


void CreateDistBaseLB();

class DistBaseLB : public CBase_DistBaseLB {
public:
  DistBaseLB(const CkLBOptions &);
  DistBaseLB(CkMigrateMessage *m) : CBase_DistBaseLB(m) {}
  ~DistBaseLB();

  void InvokeLB(void);
  void barrierDone(); // Everything is at the PE barrier
  void LoadBalance();
  void ResumeClients();
  void ResumeClients(int balancing);
  // Migrated-element callback
  void Migrated(int waitBarrier);
#if CMK_GLOBAL_LOCATION_UPDATE
  void ReceiveLocationUpdate(LBMigrateMsg* msg);
#endif

  struct LDStats {  // Passed to Strategy
    int from_pe;
    LBRealType total_walltime;
    LBRealType idletime;
    LBRealType bg_walltime;
    LBRealType obj_walltime;
#if CMK_LB_CPUTIMER
    LBRealType total_cputime;
    LBRealType bg_cputime;
    LBRealType obj_cputime;
#endif
    int pe_speed;
    bool available;
    bool move;

    std::vector<LDObjData> objData;
    std::vector<LDCommData> commData;

    inline void clearBgLoad() {
      bg_walltime = idletime = 0.0;
#if CMK_LB_CPUTIMER
      bg_cputime = 0.0;
#endif
    }
  };

protected:
  virtual void Strategy(const LDStats* const myStats);
  void ProcessMigrationDecision(LBMigrateMsg* migrateMsg);
  void MigrationDone(int balancing);  // Call when migration is complete
#if CMK_GLOBAL_LOCATION_UPDATE
  // Broadcasts this PE's own moves (the only ones it knows about) so every
  // other PE's location cache stays current for them too. Safe to call with
  // zero moves; a no-op in that case.
  void BroadcastLocationUpdate(LBMigrateMsg* migrateMsg);
  // Same, for a strategy that migrates objects one at a time with
  // lbmgr->Migrate() instead of handing down a move list; those migrations are
  // invisible to the call above.
  void BroadcastSingleLocationUpdate(const LDObjHandle& h, int to_pe);
#endif

  LDStats myStats;
  int migrates_expected;

private:
	bool lb_started;
  double start_lb_time;
  double strat_start_time;
  double strat_end_time;

  int migrates_completed;
  LBMigrateMsg** mig_msgs;

  void AssembleStats();
};

#endif /* _DISTBASELB_H */
