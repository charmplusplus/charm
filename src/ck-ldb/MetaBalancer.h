/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef METABALANCER_H
#define METABALANCER_H


//#include <charm++.h>
//#include "ckreduction.h" 

//#include "lbdb.h"
//#include "LBDBManager.h"
//#include "lbdb++.h"
#include "LBDatabase.h"

#include <vector>

#include "MetaBalancer.decl.h"

extern CkGroupID _metalb;

CkpvExtern(int, metalbInited);

void _MetaLBInit();

// main chare
class MetaLBInit : public Chare {
  public:
    MetaLBInit(CkArgMsg*);
    MetaLBInit(CkMigrateMessage *m):Chare(m) {}
};

class MetaBalancer : public CBase_MetaBalancer {
public:
  MetaBalancer(void)  { init(); }
  MetaBalancer(CkMigrateMessage *m)  { init(); }
  ~MetaBalancer()  {}
  
private:
  void init();
public:
  inline static MetaBalancer * Object() { return CkpvAccess(metalbInited)?(MetaBalancer *)CkLocalBranch(_metalb):NULL; }

  static void initnodeFn(void);

  void pup(PUP::er& p);

  void ResumeClients();

/*  inline void SetLBPeriod(double s) { LDSetLBPeriod(myLDHandle, s);}
  inline double GetLBPeriod() { return LDGetLBPeriod(myLDHandle);}*/ //NOTE

  void ResetAdaptive();
  int get_iteration();
  bool AddLoad(int iteration, double load);
  void ReceiveMinStats(CkReductionMsg *);
  void TriggerSoon(int iteration_no, double imbalance_ratio, double tolerate_imb);
  void LoadBalanceDecision(int, int);
  void LoadBalanceDecisionFinal(int, int);
  void ReceiveIterationNo(int, int); // Receives the current iter no
  void HandleAdaptiveNoObj();
  void RegisterNoObjCallback(int index);
  void TriggerAdaptiveReduction();

  bool generatePlan(int& period, double& ratio_at_t);
  bool getLineEq(double new_load_percent, double& aslope, double& ac, double& mslope, double& mc);
  bool getPeriodForLinear(double a, double b, double c, int& period);
  bool getPeriodForStrategy(double new_load, double overhead_percent, int& period, double& ratio_at_t);
  int getPredictedLBPeriod(bool& is_tentative);

  bool isStrategyComm();

  void UpdateAfterLBData(int is_lb_refine, double lb_max, double lb_avg, double
      local_comm, double remote_comm);

  void UpdateAfterLBData(double max_load, double max_cpu, double avg_load);
  void UpdateAfterLBComm(double alpha_beta_cost);
  void GetPrevLBData(int& lb_type, double& lb_max_avg_ratio, double&
      local_remote_comm_ratio);
  void GetLBDataForLB(int lb_type, double& lb_max_avg_ratio, double&
      local_remote_comm_ratio);

  void SetMigrationCost(double lb_migration_cost);
  void SetStrategyCost(double lb_strategy_cost);

private:
  //CProxy_MetaBalancer thisProxy;
  LBDatabase* lbdatabase;
  int mystep;
  std::vector<double> total_load_vec;
  // Keeps track of how many local chares contributed
  std::vector<int> total_count_vec;
  std::vector<int> lbdb_no_obj_callback;
  int max_iteration;

  double after_lb_max;
  double after_lb_avg;
  double prev_idle;
  double alpha_beta_cost_to_load;
  int is_prev_lb_refine;

public:
  bool lb_in_progress;

public:
};

inline MetaBalancer* MetaBalancerObj() { return MetaBalancer::Object(); }

#endif /* LDATABASE_H */

/*@}*/
