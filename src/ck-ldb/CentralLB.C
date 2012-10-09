
/**
 * \addtogroup CkLdb
*/
/*@{*/

#include <charm++.h>
#include "ck.h"
#include "envelope.h"
#include "CentralLB.h"
#include "LBDBManager.h"
#include "LBSimulation.h"

//#include "limits.h"
#include <vector>

#define alpha 4.0e-6
#define beta 2.67e-9
#define percent_overhead 10


#define  DEBUGF(x)       // CmiPrintf x;
#define  DEBUG(x)        // x;
#define  DEBAD(x)        // CmiPrintf x

#if CMK_MEM_CHECKPOINT
   /* can not handle reduction in inmem FT */
#define USE_REDUCTION         0
#define USE_LDB_SPANNING_TREE 0
#elif defined(_FAULT_MLOG_)
/* can not handle reduction in inmem FT */
#define USE_REDUCTION         0
#define USE_LDB_SPANNING_TREE 0
#else
#define USE_REDUCTION         1
#define USE_LDB_SPANNING_TREE 1
#endif

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
extern int _restartFlag;
extern void getGlobalStep(CkGroupID );
extern void initMlogLBStep(CkGroupID );
extern int globalResumeCount;
extern void sendDummyMigrationCounts(int *);
#endif

#if CMK_GRID_QUEUE_AVAILABLE
CpvExtern(void *, CkGridObject);
#endif

CkGroupID loadbalancer;
int * lb_ptr;
int load_balancer_created;

//struct AdaptiveData {
//  int iteration;
//  double max_load;
//  double avg_load;
//};
//
//struct AdaptiveLBDatabase {
//  std::vector<AdaptiveData> history_data;
//} adaptive_lbdb;
//
//enum state {
//  OFF,
//  ON,
//  PAUSE,
//  DECIDED,
//  LOAD_BALANCE
//} local_state;
//
//struct AdaptiveLBStructure {
//  int lb_ideal_period;
//  int lb_calculated_period;
//  int lb_no_iterations;
//  int global_max_iter_no;
//  int global_recv_iter_counter;
//  bool in_progress;
//  double prev_load;
//  double lb_strategy_cost;
//  double lb_migration_cost;
//  bool lb_period_informed;
//  int lb_msg_send_no;
//  int lb_msg_recv_no;
//} adaptive_struct;

CreateLBFunc_Def(CentralLB, "CentralLB base class")

static void getPredictedLoadWithMsg(BaseLB::LDStats* stats, int count, 
		             LBMigrateMsg *, LBInfo &info, int considerComm);

//CkReductionMsg* lbDataCollection(int nMsg, CkReductionMsg** msgs) {
//  double lb_data[4];
//  lb_data[0] = 0;
//  lb_data[1] = 0;
//  lb_data[2] = 0;
//  for (int i = 0; i < nMsg; i++) {
//    CkAssert(msgs[i]->getSize() == 4*sizeof(double));
//    double* m = (double *)msgs[i]->getData();
//    lb_data[0] += m[0];
//    lb_data[1] = ((m[1] > lb_data[1])? m[1] : lb_data[1]);
//    lb_data[2] += m[2];
//    if (i == 0) {
//      lb_data[3] = m[3];
//    }
//    if (m[3] != lb_data[3]) {
//      CkPrintf("Error!!! Reduction is intermingled between iteration %lf and\
//      %lf\n", lb_data[3], m[3]);
//    }
//  }
//  return CkReductionMsg::buildNew(4*sizeof(double), lb_data);
//}
//
///*global*/ CkReduction::reducerType lbDataCollectionType;
///*initcall*/ void registerLBDataCollection(void) {
//  lbDataCollectionType = CkReduction::addReducer(lbDataCollection);
//}

/*
void CreateCentralLB()
{
  CProxy_CentralLB::ckNew(0);
}
*/

void CentralLB::staticStartLB(void* data)
{
  CentralLB *me = (CentralLB*)(data);
  me->StartLB();
}

void CentralLB::staticMigrated(void* data, LDObjHandle h, int waitBarrier)
{
  CentralLB *me = (CentralLB*)(data);
  me->Migrated(h, waitBarrier);
}

void CentralLB::staticAtSync(void* data)
{
  CentralLB *me = (CentralLB*)(data);
  me->AtSync();
}

void CentralLB::initLB(const CkLBOptions &opt)
{
#if CMK_LBDB_ON
  lbname = "CentralLB";
  thisProxy = CProxy_CentralLB(thisgroup);
  //  CkPrintf("Construct in %d\n",CkMyPe());

  // create and turn on by default
  receiver = theLbdb->
    AddLocalBarrierReceiver((LDBarrierFn)(staticAtSync),(void*)(this));
  notifier = theLbdb->getLBDB()->
    NotifyMigrated((LDMigratedFn)(staticMigrated),(void*)(this));
  startLbFnHdl = theLbdb->getLBDB()->
    AddStartLBFn((LDStartLBFn)(staticStartLB),(void*)(this));

  // CkPrintf("[%d] CentralLB initLB \n",CkMyPe());
  if (opt.getSeqNo() > 0) turnOff();

  stats_msg_count = 0;
  statsMsgsList = NULL;
  statsData = NULL;

  storedMigrateMsg = NULL;
  reduction_started = 0;

  // for future predictor
  if (_lb_predict) predicted_model = new FutureModel(_lb_predict_window);
  else predicted_model=0;
  // register user interface callbacks
  theLbdb->getLBDB()->SetupPredictor((LDPredictModelFn)(staticPredictorOn),(LDPredictWindowFn)(staticPredictorOnWin),(LDPredictFn)(staticPredictorOff),(LDPredictModelFn)(staticChangePredictor),(void*)(this));

  myspeed = theLbdb->ProcessorSpeed();

  migrates_completed = 0;
  future_migrates_completed = 0;
  migrates_expected = -1;
  future_migrates_expected = -1;
  cur_ld_balancer = _lb_args.central_pe();      // 0 default
  lbdone = 0;
  count_msgs=0;
  statsMsg = NULL;

  if (_lb_args.statsOn()) theLbdb->CollectStatsOn();

  load_balancer_created = 1;

  // If metabalancer enabled, initialize the variables
 // adaptive_struct.lb_ideal_period =  INT_MAX;
 // adaptive_struct.lb_calculated_period = INT_MAX;
 // adaptive_struct.lb_no_iterations = -1;
 // adaptive_struct.global_max_iter_no = 0;
 // adaptive_struct.global_recv_iter_counter = 0;
 // adaptive_struct.in_progress = false;
 // adaptive_struct.prev_load = 0.0;
 // adaptive_struct.lb_strategy_cost = 0.0;
 // adaptive_struct.lb_migration_cost = 0.0;
 // adaptive_struct.lb_msg_send_no = 0;
 // adaptive_struct.lb_msg_recv_no = 0;
 // local_state = OFF;
#endif
}

CentralLB::~CentralLB()
{
#if CMK_LBDB_ON
  delete [] statsMsgsList;
  delete statsData;
  theLbdb = CProxy_LBDatabase(_lbdb).ckLocalBranch();
  if (theLbdb) {
    theLbdb->getLBDB()->
      RemoveNotifyMigrated(notifier);
    theLbdb->
      RemoveStartLBFn((LDStartLBFn)(staticStartLB));
  }
#endif
}

void CentralLB::turnOn() 
{
#if CMK_LBDB_ON
  theLbdb->getLBDB()->
    TurnOnBarrierReceiver(receiver);
  theLbdb->getLBDB()->
    TurnOnNotifyMigrated(notifier);
  theLbdb->getLBDB()->
    TurnOnStartLBFn(startLbFnHdl);
#endif
}

void CentralLB::turnOff() 
{
#if CMK_LBDB_ON
  theLbdb->getLBDB()->
    TurnOffBarrierReceiver(receiver);
  theLbdb->getLBDB()->
    TurnOffNotifyMigrated(notifier);
  theLbdb->getLBDB()->
    TurnOffStartLBFn(startLbFnHdl);
#endif
}

void CentralLB::AtSync()
{
//  CkPrintf("AtSync CEntral LB [%d]\n", CkMyPe());
#if CMK_LBDB_ON
//  DEBUGF(("[%d] CentralLB AtSync step %d!!!!!\n",CkMyPe(),step()));

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	CpvAccess(_currentObj)=this;
#endif

  // if num of processor is only 1, nothing should happen
  if (!QueryBalanceNow(step()) || CkNumPes() == 1) {
    MigrationDone(0);
    return;
  }
  if(CmiNodeAlive(CkMyPe())){
    thisProxy [CkMyPe()].ProcessAtSync();
  }
#endif
}

#include "ComlibStrategy.h"

void CentralLB::ProcessAtSync()
{



#if CMK_LBDB_ON
  if (reduction_started) return;              // reducton in progress

  CmiAssert(CmiNodeAlive(CkMyPe()));
  if (CkMyPe() == cur_ld_balancer) {
    start_lb_time = CkWallTimer();
  }
 double total_load;
 double idle_time;
 double bg_walltime;
 theLbdb->GetTime(&total_load,&total_load, &idle_time, &bg_walltime, &bg_walltime);
 theLbdb->IdleTime(&idle_time);
 DEBAD(("Total walltime [%d] %lf: %lf: %lf final laod: %lf\n", CkMyPe(),
    total_load, idle_time, bg_walltime, (total_load - idle_time - bg_walltime)));

// CkPrintf("Total walltime [%d] %lf: %lf: %lf final laod: %lf\n", CkMyPe(),
//    total_load, idle_time, bg_walltime, (total_load - idle_time - bg_walltime));


#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	initMlogLBStep(thisgroup);
#endif

  // build message
  BuildStatsMsg();

#if USE_REDUCTION
    // reduction to get total number of objects and comm
    // so that processor 0 can pre-allocate load balancing database
  int counts[2];
  counts[0] = theLbdb->GetObjDataSz();
  counts[1] = theLbdb->GetCommDataSz();

  CkCallback cb(CkIndex_CentralLB::ReceiveCounts((CkReductionMsg*)NULL), 
                  thisProxy[0]);
  contribute(2*sizeof(int), counts, CkReduction::sum_int, cb);
  reduction_started = 1;
#else
  SendStats();
#endif
#endif
}

//void CentralLB::ProcessAtSyncMin()
//{
//#if CMK_LBDB_ON
//  if (reduction_started) return;              // reducton in progress
//
//  CmiAssert(CmiNodeAlive(CkMyPe()));
//  if (CkMyPe() == cur_ld_balancer) {
//    start_lb_time = CkWallTimer();
//  }
//
//
//#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
//	initMlogLBStep(thisgroup);
//#endif
//  
//  adaptive_struct.lb_no_iterations++;
// // CkPrintf("[%d] ProcessAtSyncMin lb_iteration [%d] adaptive_struct.lb_ideal_period [%d]\n", CkMyPe(),
// //     adaptive_struct.lb_no_iterations, adaptive_struct.lb_ideal_period);
//
//  // If decision has been made and has reached the lb_period, then do load
//  // balancing, else if hasn't reached ideal_period, then resume.
//  if (local_state == DECIDED) {
//    if (adaptive_struct.lb_no_iterations < adaptive_struct.lb_ideal_period) {
// //     CkPrintf("[%d] Decision is made but lagging\n", CkMyPe());
//      SendMinStats();
//      ResumeClients(0);
//    } else {
//      local_state = LOAD_BALANCE;
// //     CkPrintf("[%d] Decision is made and do LB\n", CkMyPe());
//      ProcessAtSync();
//    }
//    return;
//  }
//   
//  // If the state is ON and not DECIDED, then if havn't reached lb_period, then
//  // move ahead. If has reached lb_ideal_period, then change state to PAUSE and
//  // dont resume client.
//  if (local_state == ON) {
//    if (adaptive_struct.lb_no_iterations < adaptive_struct.lb_ideal_period) {
//      SendMinStats();
//      ResumeClients(0);
//    } else {
//      local_state = PAUSE;
//    }
//    return;
//  }
//
//  SendMinStats();
//  ResumeClients(0);
//#endif
//}
//
//void CentralLB::SendMinStats() {
//
// double total_load;
// double idle_time;
// double bg_walltime;
//  theLbdb->GetTime(&total_load,&total_load, &idle_time, &bg_walltime, &bg_walltime);
// // CkPrintf("Total walltime [%d] %lf: %lf: %lf final laod: %lf\n", CkMyPe(), total_load, idle_time, bg_walltime, (total_load - idle_time));
//
//  // Since the total_load is cumulative since the last load balancing stage,
//  // Hence it is subtracted from the previous load.
//  total_load -= idle_time;
//  double tmp = total_load;
//  total_load -= adaptive_struct.prev_load;
//  adaptive_struct.prev_load = tmp; 
//
//  double lb_data[4];
//  lb_data[0] = total_load;
//  lb_data[1] = total_load;
//  lb_data[2] = 1;
//  lb_data[3] = adaptive_struct.lb_no_iterations;
//  //CkPrintf("[%d] sends total load %lf at iter %d\n", CkMyPe(), total_load, adaptive_struct.lb_no_iterations);
//
//  if (adaptive_struct.lb_no_iterations != 0) {
//    CkCallback cb(CkIndex_CentralLB::ReceiveMinStats((CkReductionMsg*)NULL), 
//        thisProxy[0]);
////    contribute(4*sizeof(double), lb_data, lbDataCollectionType, cb);
//  }
//
////    int tmp1 = adaptive_struct.lb_no_iterations;
////    CkPrintf("[%d] contribution iteration_no: %d\n",CkMyPe(), tmp1);
////    // Send the current iteration no
////    CkCallback cb1(CkIndex_CentralLB::ReceiveIterationNo((CkReductionMsg*)NULL), 
////        thisProxy[0]);
////    contribute(sizeof(int), &tmp1, CkReduction::max_int, cb1);
//}
//
//void CentralLB::ReceiveMinStats(CkReductionMsg *msg) {
//  CmiAssert(CkMyPe() == 0);
//  double* load = (LBRealType *) msg->getData();
//  double max = load[1];
//  double avg = load[0]/load[2];
//  int iteration_n = load[3];
//  CkPrintf("Iteration %d Total load : %lf Avg load: %lf Max load: %lf for %lf procs\n",iteration_n, load[0], load[0]/load[2], load[1], load[2]);
//  CkPrintf("Current calculated period %d\n", adaptive_struct.lb_calculated_period);
//  delete msg;
//
//  // Store the data for this iteration
//  AdaptiveData data;
//  data.iteration = adaptive_struct.lb_no_iterations;
//  data.max_load = max;
//  data.avg_load = avg;
//  adaptive_lbdb.history_data.push_back(data);
//
//  // If lb period inform is in progress, dont inform again
//  if (adaptive_struct.in_progress) {
//    return;
//  }
//
////  if (adaptive_struct.lb_period_informed) {
////    return;
////  }
//
//  // If the max/avg ratio is greater than the threshold and also this is not the
//  // step immediately after load balancing, carry out load balancing
//  //if (max/avg >= 1.1 && adaptive_lbdb.history_data.size() > 4) {
//  if (max/avg >= 1.5 && adaptive_lbdb.history_data.size() > 4) {
//    CkPrintf("Carry out load balancing step at iter max/avg(%lf) > 1.1\n", max/avg);
////    if (!adaptive_struct.lb_period_informed) {
////      // Just for testing
////      adaptive_struct.lb_calculated_period = 40;
////      adaptive_struct.lb_period_informed = true;
////      thisProxy.LoadBalanceDecision(adaptive_struct.lb_calculated_period);
////      return;
////    }
//
//    // If the new lb period is less than current set lb period
//    if (adaptive_struct.lb_calculated_period > iteration_n + 1) {
//      adaptive_struct.lb_calculated_period = iteration_n + 1;
//      adaptive_struct.lb_period_informed = true;
//      adaptive_struct.in_progress = true;
//      CkPrintf("Informing everyone the lb period is %d\n",
//          adaptive_struct.lb_calculated_period);
//      thisProxy.LoadBalanceDecision(adaptive_struct.lb_msg_send_no++, adaptive_struct.lb_calculated_period);
//    }
//    return;
//  }
//
//  // Generate the plan for the adaptive strategy
//  int period;
//  if (generatePlan(period)) {
//    //CkPrintf("Carry out load balancing step at iter\n");
//
//    // If the new lb period is less than current set lb period
//    if (adaptive_struct.lb_calculated_period > period) {
//      adaptive_struct.lb_calculated_period = period;
//      adaptive_struct.in_progress = true;
//      adaptive_struct.lb_period_informed = true;
//      CkPrintf("Informing everyone the lb period is %d\n",
//          adaptive_struct.lb_calculated_period);
//      thisProxy.LoadBalanceDecision(adaptive_struct.lb_msg_send_no++, adaptive_struct.lb_calculated_period);
//    }
//  }
//}
//
//bool CentralLB::generatePlan(int& period) {
//  if (adaptive_lbdb.history_data.size() <= 8) {
//    return false;
//  }
//
//  // Some heuristics for lbperiod
//  // If constant load or almost constant,
//  // then max * new_lb_period > avg * new_lb_period + lb_cost
//  double max = 0.0;
//  double avg = 0.0;
//  AdaptiveData data;
//  for (int i = 0; i < adaptive_lbdb.history_data.size(); i++) {
//    data = adaptive_lbdb.history_data[i];
//    max += data.max_load;
//    avg += data.avg_load;
//    CkPrintf("max (%d, %lf) avg (%d, %lf)\n", i, data.max_load, i, data.avg_load);
//  }
////  max /= (adaptive_struct.lb_no_iterations - adaptive_lbdb.history_data[0].iteration);
////  avg /= (adaptive_struct.lb_no_iterations - adaptive_lbdb.history_data[0].iteration);
////
////  adaptive_struct.lb_ideal_period = (adaptive_struct.lb_strategy_cost +
////  adaptive_struct.lb_migration_cost) / (max - avg);
////  CkPrintf("max : %lf, avg: %lf, strat cost: %lf, migration_cost: %lf, idealperiod : %d \n",
////      max, avg, adaptive_struct.lb_strategy_cost, adaptive_struct.lb_migration_cost, adaptive_struct.lb_ideal_period);
////
//  // If linearly varying load, then find lb_period
//  // area between the max and avg curve 
//  double mslope, aslope, mc, ac;
//  getLineEq(aslope, ac, mslope, mc);
//  CkPrintf("\n max: %fx + %f; avg: %fx + %f\n", mslope, mc, aslope, ac);
//  double a = (mslope - aslope)/2;
//  double b = (mc - ac);
//  double c = -(adaptive_struct.lb_strategy_cost + adaptive_struct.lb_migration_cost);
//  //c = -2.5;
//  bool got_period = getPeriodForLinear(a, b, c, period);
//  if (!got_period) {
//    return false;
//  }
//  
//  if (mslope < 0) {
//    if (period > (-mc/mslope)) {
//      CkPrintf("Max < 0 Period set when max load is -ve\n");
//      return false;
//    }
//  }
//
//  if (aslope < 0) {
//    if (period > (-ac/aslope)) {
//      CkPrintf("Avg < 0 Period set when avg load is -ve\n");
//      return false;
//    }
//  }
//
//  int intersection_t = (mc-ac) / (aslope - mslope);
//  if (intersection_t > 0 && period > intersection_t) {
//    CkPrintf("Avg | Max Period set when curves intersect\n");
//    return false;
//  }
//  return true;
//}
//
//bool CentralLB::getPeriodForLinear(double a, double b, double c, int& period) {
//  CkPrintf("Quadratic Equation %lf X^2 + %lf X + %lf\n", a, b, c);
//  if (a == 0.0) {
//    period = (-c / b);
//    CkPrintf("Ideal period for linear load %d\n", period);
//    return true;
//  }
//  int x;
//  double t = (b * b) - (4*a*c);
//  if (t < 0) {
//    CkPrintf("(b * b) - (4*a*c) is -ve sqrt : %lf\n", sqrt(t));
//    return false;
//  }
//  t = (-b + sqrt(t)) / (2*a);
//  x = t;
//  if (x < 0) {
//    CkPrintf("boo!!! x (%d) < 0\n", x);
//    x = 0;
//    return false;
//  }
//  period = x;
//  CkPrintf("Ideal period for linear load %d\n", period);
//  return true;
//}
//
//bool CentralLB::getLineEq(double& aslope, double& ac, double& mslope, double& mc) {
//  int total = adaptive_lbdb.history_data.size();
//  int iterations = 1 + adaptive_lbdb.history_data[total - 1].iteration -
//      adaptive_lbdb.history_data[0].iteration;
//  double a1 = 0;
//  double m1 = 0;
//  double a2 = 0;
//  double m2 = 0;
//  AdaptiveData data;
//  int i = 0;
//  for (i = 0; i < total/2; i++) {
//    data = adaptive_lbdb.history_data[i];
//    m1 += data.max_load;
//    a1 += data.avg_load;
//  }
//  m1 /= i;
//  a1 /= i;
//
//  for (i = total/2; i < total; i++) {
//    data = adaptive_lbdb.history_data[i];
//    m2 += data.max_load;
//    a2 += data.avg_load;
//  }
//  m2 /= (i - total/2);
//  a2 /= (i - total/2);
//
//  aslope = 2 * (a2 - a1) / iterations;
//  mslope = 2 * (m2 - m1) / iterations;
//  ac = adaptive_lbdb.history_data[0].avg_load;
//  mc = adaptive_lbdb.history_data[0].max_load;
//  return true;
//}
//
//void CentralLB::LoadBalanceDecision(int req_no, int period) {
//  if (req_no < adaptive_struct.lb_msg_recv_no) {
//    CkPrintf("Error!!! Received a request which was already sent or old\n");
//    return;
//  }
//  //CkPrintf("[%d] Load balance decision made cur iteration: %d period:%d state: %d\n",CkMyPe(), adaptive_struct.lb_no_iterations, period, local_state);
//  adaptive_struct.lb_ideal_period = period;
//  local_state = ON;
//  adaptive_struct.lb_msg_recv_no = req_no;
//  thisProxy[0].ReceiveIterationNo(req_no, adaptive_struct.lb_no_iterations);
//}
//
//void CentralLB::LoadBalanceDecisionFinal(int req_no, int period) {
//  if (req_no < adaptive_struct.lb_msg_recv_no) {
//    return;
//  }
//  //CkPrintf("[%d] Final Load balance decision made cur iteration: %d period:%d state: %d\n",CkMyPe(), adaptive_struct.lb_no_iterations, period, local_state);
//  adaptive_struct.lb_ideal_period = period;
//
//  if (local_state == ON) {
//    local_state = DECIDED;
//    return;
//  }
//
//  // If the state is PAUSE, then its waiting for the final decision from central
//  // processor. If the decision is that the ideal period is in the future,
//  // resume. If the ideal period is now, then carry out load balancing.
//  if (local_state == PAUSE) {
//    if (adaptive_struct.lb_no_iterations < adaptive_struct.lb_ideal_period) {
//      local_state = DECIDED;
//      SendMinStats();
//      ResumeClients(0);
//    } else {
//      local_state = LOAD_BALANCE;
//      ProcessAtSync();
//    }
//    return;
//  }
//  CkPrintf("Error!!! Final decision received but the state is invalid %d\n", local_state);
//}
//
//
//void CentralLB::ReceiveIterationNo(int req_no, int local_iter_no) {
//  CmiAssert(CkMyPe() == 0);
//
//  adaptive_struct.global_recv_iter_counter++;
//  if (local_iter_no > adaptive_struct.global_max_iter_no) {
//    adaptive_struct.global_max_iter_no = local_iter_no;
//  }
//  if (CkNumPes() == adaptive_struct.global_recv_iter_counter) {
//    adaptive_struct.lb_ideal_period = (adaptive_struct.lb_ideal_period > adaptive_struct.global_max_iter_no) ? adaptive_struct.lb_ideal_period : adaptive_struct.global_max_iter_no + 1;
//    thisProxy.LoadBalanceDecisionFinal(req_no, adaptive_struct.lb_ideal_period);
//    CkPrintf("Final lb_period %d\n", adaptive_struct.lb_ideal_period);
//    adaptive_struct.in_progress = false;
//    adaptive_struct.global_max_iter_no = 0;
//    adaptive_struct.global_recv_iter_counter = 0;
//  }
//}

// called only on 0
void CentralLB::ReceiveCounts(CkReductionMsg  *msg)
{
  CmiAssert(CkMyPe() == 0);
  if (statsData == NULL) statsData = new LDStats;

  int *counts = (int *)msg->getData();
  int n_objs = counts[0];
  int n_comm = counts[1];

    // resize database
  statsData->objData.resize(n_objs);
  statsData->from_proc.resize(n_objs);
  statsData->to_proc.resize(n_objs);
  statsData->commData.resize(n_comm);

  DEBUGF(("[%d] ReceiveCounts: n_objs:%d n_comm:%d\n",CkMyPe(), n_objs, n_comm));
	
    // broadcast call to let everybody start to send stats
  thisProxy.SendStats();
}

void CentralLB::BuildStatsMsg()
{
#if CMK_LBDB_ON
  // build and send stats
  const int osz = theLbdb->GetObjDataSz();
  const int csz = theLbdb->GetCommDataSz();

  int npes = CkNumPes();
  CLBStatsMsg* msg = new CLBStatsMsg(osz, csz);
  _MEMCHECK(msg);
  msg->from_pe = CkMyPe();
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	msg->step = step();
#endif
  //msg->serial = CrnRand();

/*
  theLbdb->TotalTime(&msg->total_walltime,&msg->total_cputime);
  theLbdb->IdleTime(&msg->idletime);
  theLbdb->BackgroundLoad(&msg->bg_walltime,&msg->bg_cputime);
*/
#if CMK_LB_CPUTIMER
  theLbdb->GetTime(&msg->total_walltime,&msg->total_cputime,
		   &msg->idletime, &msg->bg_walltime,&msg->bg_cputime);
#else
  theLbdb->GetTime(&msg->total_walltime,&msg->total_walltime,
		   &msg->idletime, &msg->bg_walltime,&msg->bg_walltime);
#endif

  msg->pe_speed = myspeed;
  DEBUGF(("Processor %d Total time (wall,cpu) = %f %f Idle = %f Bg = %f %f\n", CkMyPe(),msg->total_walltime,msg->total_cputime,msg->idletime,msg->bg_walltime,msg->bg_cputime));

  msg->n_objs = osz;
  theLbdb->GetObjData(msg->objData);
  msg->n_comm = csz;
  theLbdb->GetCommData(msg->commData);
//  theLbdb->ClearLoads();
  DEBUGF(("PE %d BuildStatsMsg %d objs, %d comm\n",CkMyPe(),msg->n_objs,msg->n_comm));

  if(CkMyPe() == cur_ld_balancer) {
    msg->avail_vector = new char[CkNumPes()];
    LBDatabaseObj()->get_avail_vector(msg->avail_vector);
    msg->next_lb = LBDatabaseObj()->new_lbbalancer();
  }

  CmiAssert(statsMsg == NULL);
  statsMsg = msg;
#endif
}


// called on every processor
void CentralLB::SendStats()
{
#if CMK_LBDB_ON
  CmiAssert(statsMsg != NULL);
  reduction_started = 0;

#if USE_LDB_SPANNING_TREE
  if(CkNumPes()>1024)
  {
    if (CkMyPe() == cur_ld_balancer)
      thisProxy[CkMyPe()].ReceiveStats(statsMsg);
    else
      thisProxy[CkMyPe()].ReceiveStatsViaTree(statsMsg);
  }
  else
#endif
  {
    DEBUGF(("[%d] calling ReceiveStats on step %d \n",CmiMyPe(),step()));
    thisProxy[cur_ld_balancer].ReceiveStats(statsMsg);
  }

  statsMsg = NULL;

#ifdef __BIGSIM__
  BgEndStreaming();
#endif

  {
  // enfore the barrier to wait until centralLB says no
  LDOMHandle h;
  h.id.id.idx = 0;
  theLbdb->getLBDB()->RegisteringObjects(h);
  }
#endif
}

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
extern int donotCountMigration;
#endif

void CentralLB::Migrated(LDObjHandle h, int waitBarrier)
{
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    if(donotCountMigration){
        return ;
    }
#endif

#if CMK_LBDB_ON
  if (waitBarrier) {
	    migrates_completed++;
      DEBUGF(("[%d] An object migrated! %d %d\n",CkMyPe(),migrates_completed,migrates_expected));
    if (migrates_completed == migrates_expected) {
      MigrationDone(1);
    }
  }
  else {
    future_migrates_completed ++;
    DEBUGF(("[%d] An object migrated with no barrier! %d expected: %d\n",CkMyPe(),future_migrates_completed,future_migrates_expected));
    if (future_migrates_completed == future_migrates_expected)  {
	CheckMigrationComplete();
    }
  }
#endif
}

void CentralLB::MissMigrate(int waitForBarrier)
{
  LDObjHandle h;
  Migrated(h, waitForBarrier);
}

// build a complete data from bufferred messages
// not used when USE_REDUCTION = 1
void CentralLB::buildStats()
{
    statsData->nprocs() = stats_msg_count;
    // allocate space
    statsData->objData.resize(statsData->n_objs);
    statsData->from_proc.resize(statsData->n_objs);
    statsData->to_proc.resize(statsData->n_objs);
    statsData->commData.resize(statsData->n_comm);

    int nobj = 0;
    int ncom = 0;
    int nmigobj = 0;
    // copy all data in individule message to this big structure
    for (int pe=0; pe<CkNumPes(); pe++) {
       int i;
       CLBStatsMsg *msg = statsMsgsList[pe];
       if(msg == NULL) continue;
       for (i=0; i<msg->n_objs; i++) {
         statsData->from_proc[nobj] = statsData->to_proc[nobj] = pe;
	 statsData->objData[nobj] = msg->objData[i];
         if (msg->objData[i].migratable) nmigobj++;
	 nobj++;
       }
       for (i=0; i<msg->n_comm; i++) {
	 statsData->commData[ncom] = msg->commData[i];
	 ncom++;
       }
       // free the memory
       delete msg;
       statsMsgsList[pe]=0;
    }
    statsData->n_migrateobjs = nmigobj;
}

// deposit one processor data at a time, note database is pre-allocated
// to have enough space
// used when USE_REDUCTION = 1
void CentralLB::depositData(CLBStatsMsg *m)
{
  int i;
  if (m == NULL) return;

  const int pe = m->from_pe;
  struct ProcStats &procStat = statsData->procs[pe];
  procStat.pe = pe;
  procStat.total_walltime = m->total_walltime;
  procStat.idletime = m->idletime;
  procStat.bg_walltime = m->bg_walltime;
#if CMK_LB_CPUTIMER
  procStat.total_cputime = m->total_cputime;
  procStat.bg_cputime = m->bg_cputime;
#endif
  procStat.pe_speed = m->pe_speed;
  //procStat.utilization = 1.0;
  procStat.available = CmiTrue;
  procStat.n_objs = m->n_objs;

  int &nobj = statsData->n_objs;
  int &nmigobj = statsData->n_migrateobjs;
  for (i=0; i<m->n_objs; i++) {
      statsData->from_proc[nobj] = statsData->to_proc[nobj] = pe;
      statsData->objData[nobj] = m->objData[i];
      if (m->objData[i].migratable) nmigobj++;
      nobj++;
      CmiAssert(nobj <= statsData->objData.capacity());
  }
  int &n_comm = statsData->n_comm;
  for (i=0; i<m->n_comm; i++) {
      statsData->commData[n_comm] = m->commData[i];
      n_comm++;
      CmiAssert(n_comm <= statsData->commData.capacity());
  }
  delete m;
}

void CentralLB::ReceiveStats(CkMarshalledCLBStatsMessage &msg)
{
#if CMK_LBDB_ON
  if (statsMsgsList == NULL) {
    statsMsgsList = new CLBStatsMsg*[CkNumPes()];
    CmiAssert(statsMsgsList != NULL);
    for(int i=0; i < CkNumPes(); i++)
      statsMsgsList[i] = 0;
  }
  if (statsData == NULL) statsData = new LDStats;

    //  loop through all CLBStatsMsg in the incoming msg
  int count = msg.getCount();
  for (int num = 0; num < count; num++) 
  {
    CLBStatsMsg *m = msg.getMessage(num);
    CmiAssert(m!=NULL);
    const int pe = m->from_pe;
    DEBUGF(("Stats msg received, %d %d %d %p step %d\n", pe,stats_msg_count,m->n_objs,m,step()));
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))     
/*      
 *  if(m->step < step()){
 *    //TODO: if a processor is redoing an old load balance step..
 *    //tell it that the step is done and that it should not perform any migrations
 *      thisProxy[pe].ReceiveDummyMigration();
 *  }*/
#endif
	
    if(!CmiNodeAlive(pe)){
	DEBUGF(("[%d] ReceiveStats called from invalidProcessor %d\n",CkMyPe(),pe));
	continue;
    }
	
    if (m->avail_vector!=NULL) {
      LBDatabaseObj()->set_avail_vector(m->avail_vector,  m->next_lb);
    }

    if (statsMsgsList[pe] != 0) {
      CkPrintf("*** Unexpected CLBStatsMsg in ReceiveStats from PE %d ***\n",
	     pe);
    } else {
      statsMsgsList[pe] = m;
#if USE_REDUCTION
      depositData(m);
#else
      // store per processor data right away
      struct ProcStats &procStat = statsData->procs[pe];
      procStat.pe = pe;
      procStat.total_walltime = m->total_walltime;
      procStat.idletime = m->idletime;
      procStat.bg_walltime = m->bg_walltime;
#if CMK_LB_CPUTIMER
      procStat.total_cputime = m->total_cputime;
      procStat.bg_cputime = m->bg_cputime;
#endif
      procStat.pe_speed = m->pe_speed;
      //procStat.utilization = 1.0;
      procStat.available = CmiTrue;
      procStat.n_objs = m->n_objs;

      statsData->n_objs += m->n_objs;
      statsData->n_comm += m->n_comm;
#endif
      stats_msg_count++;
    }
  }    // end of for

  const int clients = CkNumValidPes();
  DEBUGF(("THIS POINT count = %d, clients = %d\n",stats_msg_count,clients));
 
  if (stats_msg_count == clients) {
	DEBUGF(("[%d] All stats messages received \n",CmiMyPe()));
    statsData->nprocs() = stats_msg_count;
    thisProxy[CkMyPe()].LoadBalance();
  }
#endif
}

/** added by Abhinav for receiving msgs via spanning tree */
void CentralLB::ReceiveStatsViaTree(CkMarshalledCLBStatsMessage &msg)
{
#if CMK_LBDB_ON
	CmiAssert(CkMyPe() != 0);
	bufMsg.add(msg);         // buffer messages
	count_msgs++;
	//CkPrintf("here %d\n", CkMyPe());
	if (count_msgs == st.numChildren+1) {
		if(st.parent == 0)
		{
			thisProxy[0].ReceiveStats(bufMsg);
			//CkPrintf("from %d\n", CkMyPe());
		}
		else
			thisProxy[st.parent].ReceiveStatsViaTree(bufMsg);
		count_msgs = 0;
                bufMsg.free();
	} 
#endif
}

void CentralLB::LoadBalance()
{
#if CMK_LBDB_ON
  int proc;
  const int clients = CkNumPes();

#if ! USE_REDUCTION
  // build data
  buildStats();
#else
  for (proc = 0; proc < clients; proc++) statsMsgsList[proc] = NULL;
#endif

//NOTE  theLbdb->ResetAdaptive();
  if (!_lb_args.samePeSpeed()) statsData->normalize_speed();

  if (_lb_args.debug()) 
      CmiPrintf("\nCharmLB> %s: PE [%d] step %d starting at %f Memory: %f MB\n",
		  lbname, cur_ld_balancer, step(), start_lb_time,
		  CmiMemoryUsage()/(1024.0*1024.0));

  // if we are in simulation mode read data
  if (LBSimulation::doSimulation) simulationRead();

  char *availVector = LBDatabaseObj()->availVector();
  for(proc = 0; proc < clients; proc++)
      statsData->procs[proc].available = (CmiBool)availVector[proc];

  preprocess(statsData);

//    CkPrintf("Before Calling Strategy\n");

  if (_lb_args.printSummary()) {
      LBInfo info(clients);
        // not take comm data
      info.getInfo(statsData, clients, 0);
      LBRealType mLoad, mCpuLoad, totalLoad;
      info.getSummary(mLoad, mCpuLoad, totalLoad);
      int nmsgs, nbytes;
      statsData->computeNonlocalComm(nmsgs, nbytes);
      CkPrintf("[%d] Load Summary (before LB): max (with bg load): %f max (obj only): %f average: %f at step %d nonlocal: %d msgs %.2fKB.\n", CkMyPe(), mLoad, mCpuLoad, totalLoad/clients, step(), nmsgs, 1.0*nbytes/1024);
//      if (_lb_args.debug() > 1) {
//        for (int i=0; i<statsData->n_objs; i++)
//          CmiPrintf("[%d] %.10f %.10f\n", i, statsData->objData[i].minWall, statsData->objData[i].maxWall);
//      }
  }

#if CMK_REPLAYSYSTEM
  LDHandle *loadBalancer_pointers;
  if (_replaySystem) {
    loadBalancer_pointers = (LDHandle*)malloc(CkNumPes()*sizeof(LDHandle));
    for (int i=0; i<statsData->n_objs; ++i) loadBalancer_pointers[statsData->from_proc[i]] = statsData->objData[i].handle.omhandle.ldb;
  }
#endif
  
  LBMigrateMsg* migrateMsg = Strategy(statsData);
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	migrateMsg->step = step();
#endif

#if CMK_REPLAYSYSTEM
  CpdHandleLBMessage(&migrateMsg);
  if (_replaySystem) {
    for (int i=0; i<migrateMsg->n_moves; ++i) migrateMsg->moves[i].obj.omhandle.ldb = loadBalancer_pointers[migrateMsg->moves[i].from_pe];
    free(loadBalancer_pointers);
  }
#endif
  
  LBDatabaseObj()->get_avail_vector(migrateMsg->avail_vector);
  migrateMsg->next_lb = LBDatabaseObj()->new_lbbalancer();

  // if this is the step at which we need to dump the database
  simulationWrite();

//  calculate predicted load
//  very time consuming though, so only happen when debugging is on
  if (_lb_args.printSummary()) {
      LBInfo info(clients);
        // not take comm data
      getPredictedLoadWithMsg(statsData, clients, migrateMsg, info, 0);
      LBRealType mLoad, mCpuLoad, totalLoad;
      info.getSummary(mLoad, mCpuLoad, totalLoad);
      int nmsgs, nbytes;
      statsData->computeNonlocalComm(nmsgs, nbytes);
      CkPrintf("[%d] Load Summary (after LB): max (with bg load): %f max (obj only): %f average: %f at step %d nonlocal: %d msgs %.2fKB useMem: %.2fKB.\n", CkMyPe(), mLoad, mCpuLoad, totalLoad/clients, step(), nmsgs, 1.0*nbytes/1024, (1.0*useMem())/1024);
      for (int i=0; i<clients; i++)
        migrateMsg->expectedLoad[i] = info.peLoads[i];
  }

  DEBUGF(("[%d]calling recv migration\n",CkMyPe()));
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_)) 
    lbDecisionCount++;
    migrateMsg->lbDecisionCount = lbDecisionCount;
#endif

  envelope *env = UsrToEnv(migrateMsg);
  if (1) {
      // broadcast
    thisProxy.ReceiveMigration(migrateMsg);
  }
  else {
    // split the migration for each processor
    for (int p=0; p<CkNumPes(); p++) {
      LBMigrateMsg *m = extractMigrateMsg(migrateMsg, p);
      thisProxy[p].ReceiveMigration(m);
    }
    delete migrateMsg;
  }

  // Zero out data structures for next cycle
  // CkPrintf("zeroing out data\n");
  statsData->clear();
  stats_msg_count=0;
#endif
}

// test if sender and receiver in a commData is nonmigratable.
static int isMigratable(LDObjData **objData, int *len, int count, const LDCommData &commData)
{
#if CMK_LBDB_ON
  for (int pe=0 ; pe<count; pe++)
  {
    for (int i=0; i<len[pe]; i++)
      if (LDObjIDEqual(objData[pe][i].objID(), commData.sender.objID()) ||
          LDObjIDEqual(objData[pe][i].objID(), commData.receiver.get_destObj().objID())) 
      return 0;
  }
#endif
  return 1;
}

// rebuild LDStats and remove all non-migratble objects and related things
void CentralLB::removeNonMigratable(LDStats* stats, int count)
{
  int i;

  // check if we have non-migratable objects
  int have = 0;
  for (i=0; i<stats->n_objs; i++) 
  {
    LDObjData &odata = stats->objData[i];
    if (!odata.migratable) {
      have = 1; break;
    }
  }
  if (have == 0) return;

  CkVec<LDObjData> nonmig;
  CkVec<int> new_from_proc, new_to_proc;
  nonmig.resize(stats->n_migrateobjs);
  new_from_proc.resize(stats->n_migrateobjs);
  new_to_proc.resize(stats->n_migrateobjs);
  int n_objs = 0;
  for (i=0; i<stats->n_objs; i++) 
  {
    LDObjData &odata = stats->objData[i];
    if (odata.migratable) {
      nonmig[n_objs] = odata;
      new_from_proc[n_objs] = stats->from_proc[i];
      new_to_proc[n_objs] = stats->to_proc[i];
      n_objs ++;
    }
    else {
      stats->procs[stats->from_proc[i]].bg_walltime += odata.wallTime;
#if CMK_LB_CPUTIMER
      stats->procs[stats->from_proc[i]].bg_cputime += odata.cpuTime;
#endif
    }
  }
  CmiAssert(stats->n_migrateobjs == n_objs);

  stats->makeCommHash();
  
  CkVec<LDCommData> newCommData;
  newCommData.resize(stats->n_comm);
  int n_comm = 0;
  for (i=0; i<stats->n_comm; i++) 
  {
    LDCommData& cdata = stats->commData[i];
    if (!cdata.from_proc()) 
    {
      int idx = stats->getSendHash(cdata);
      CmiAssert(idx != -1);
      if (!stats->objData[idx].migratable) continue;
    }
    switch (cdata.receiver.get_type()) {
    case LD_PROC_MSG:
      break;
    case LD_OBJ_MSG:  {
      int idx = stats->getRecvHash(cdata);
      if (stats->complete_flag)
        CmiAssert(idx != -1);
      else if (idx == -1) continue;          // receiver not in this group
      if (!stats->objData[idx].migratable) continue;
      break;
      }
    case LD_OBJLIST_MSG:    // object message FIXME add multicast
      break;
    }
    newCommData[n_comm] = cdata;
    n_comm ++;
  }

  if (n_objs != stats->n_objs) CmiPrintf("Removed %d nonmigratable %d comms - n_objs:%d migratable:%d\n", stats->n_objs-n_objs, stats->n_objs, stats->n_migrateobjs, stats->n_comm-n_comm);

  // swap to new data
  stats->objData = nonmig;
  stats->from_proc = new_from_proc;
  stats->to_proc = new_to_proc;
  stats->n_objs = n_objs;

  stats->commData = newCommData;
  stats->n_comm = n_comm;

  stats->deleteCommHash();
  stats->makeCommHash();

}


#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
extern int restarted;
#endif

void CentralLB::ReceiveMigration(LBMigrateMsg *m)
{
  storedMigrateMsg = m;
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	ProcessReceiveMigration((CkReductionMsg*)NULL);
#else
  CkCallback cb(CkIndex_CentralLB::ProcessReceiveMigration((CkReductionMsg*)NULL),
                  thisProxy);
  contribute(0, NULL, CkReduction::max_int, cb);

#endif
}

void CentralLB::ProcessReceiveMigration(CkReductionMsg  *msg)
{
#if CMK_LBDB_ON
	int i;
        LBMigrateMsg *m = storedMigrateMsg;
        CmiAssert(m!=NULL);
        delete msg;

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	int *dummyCounts;

	DEBUGF(("[%d] Starting ReceiveMigration WITH step %d m->step %d\n",CkMyPe(),step(),m->step));
	// CmiPrintf("[%d] Starting ReceiveMigration step %d m->step %d\n",CkMyPe(),step(),m->step);
	if(step() > m->step){
		char str[100];
		envelope *env = UsrToEnv(m);
		CmiPrintf("[%d] Object %s tProcessed %d m->TN %d\n",CmiMyPe(),mlogData->objID.toString(str),mlogData->tProcessed,env->TN);
		return;
	}
	lbDecisionCount = m->lbDecisionCount;
#endif

  if (_lb_args.debug() > 1) 
    if (CkMyPe()%1024==0) CmiPrintf("[%d] Starting ReceiveMigration step %d at %f\n",CkMyPe(),step(), CmiWallTimer());

  for (i=0; i<CkNumPes(); i++) theLbdb->lastLBInfo.expectedLoad[i] = m->expectedLoad[i];
  CmiAssert(migrates_expected <= 0 || migrates_completed == migrates_expected);
/*FAULT_EVAC*/
  if(!CmiNodeAlive(CkMyPe())){
	delete m;
	return;
  }
  migrates_expected = 0;
  future_migrates_expected = 0;
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	int sending=0;
    int dummy=0;
	LBDB *_myLBDB = theLbdb->getLBDB();
	if(_restartFlag){
        dummyCounts = new int[CmiNumPes()];
        bzero(dummyCounts,sizeof(int)*CmiNumPes());
    }
#endif
  for(i=0; i < m->n_moves; i++) {
    MigrateInfo& move = m->moves[i];
    const int me = CkMyPe();
    if (move.from_pe == me && move.to_pe != me) {
      DEBUGF(("[%d] migrating object to %d\n",move.from_pe,move.to_pe));
      // migrate object, in case it is already gone, inform toPe
#if (!defined(_FAULT_MLOG_) && !defined(_FAULT_CAUSAL_))
      if (theLbdb->Migrate(move.obj,move.to_pe) == 0) 
         thisProxy[move.to_pe].MissMigrate(!move.async_arrival);
#else
            if(_restartFlag == 0){
                DEBUG(CmiPrintf("[%d] need to move object from %d to %d \n",CkMyPe(),move.from_pe,move.to_pe));
                theLbdb->Migrate(move.obj,move.to_pe);
                sending++;
            }else{
                if(_myLBDB->validObjHandle(move.obj)){
                    DEBUG(CmiPrintf("[%d] need to move object from %d to %d \n",CkMyPe(),move.from_pe,move.to_pe));
                    theLbdb->Migrate(move.obj,move.to_pe);
                    sending++;
                }else{
                    DEBUG(CmiPrintf("[%d] dummy move to pe %d detected after restart \n",CmiMyPe(),move.to_pe));
                    dummyCounts[move.to_pe]++;
                    dummy++;
                }
            }
#endif
    } else if (move.from_pe != me && move.to_pe == me) {
       DEBUGF(("[%d] expecting object from %d\n",move.to_pe,move.from_pe));
      if (!move.async_arrival) migrates_expected++;
      else future_migrates_expected++;
    }
  }
  DEBUGF(("[%d] in ReceiveMigration %d moves expected: %d future expected: %d\n",CkMyPe(),m->n_moves, migrates_expected, future_migrates_expected));
  // if (_lb_debug) CkPrintf("[%d] expecting %d objects migrating.\n", CkMyPe(), migrates_expected);

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	if(_restartFlag){
		sendDummyMigrationCounts(dummyCounts);
		_restartFlag  =0;
    	delete []dummyCounts;
	}
#endif


#if 0
  if (m->n_moves ==0) {
    theLbdb->SetLBPeriod(theLbdb->GetLBPeriod()*2);
  }
#endif
  cur_ld_balancer = m->next_lb;
  if((CkMyPe() == cur_ld_balancer) && (cur_ld_balancer != 0)){
      LBDatabaseObj()->set_avail_vector(m->avail_vector, -2);
  }

  if (migrates_expected == 0 || migrates_completed == migrates_expected)
    MigrationDone(1);
  delete m;

//	CkEvacuatedElement();
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
//  migrates_expected = 0;
//  //  ResumeClients(1);
#endif
#endif
}

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
void CentralLB::ReceiveDummyMigration(int globalDecisionCount){
    DEBUGF(("[%d] ReceiveDummyMigration called for step %d with globalDecisionCount %d\n",CkMyPe(),step(),globalDecisionCount));
    //TODO: this is gonna be important when a crash happens during checkpoint
    //the globalDecisionCount would have to be saved and compared against
    //a future recvMigration
                
	thisProxy[CkMyPe()].ResumeClients(1);
}
#endif

void CentralLB::MigrationDone(int balancing)
{
#if CMK_LBDB_ON
  migrates_completed = 0;
  migrates_expected = -1;
  // clear load stats
  if (balancing) theLbdb->ClearLoads();
  // Increment to next step
  theLbdb->incStep();
	DEBUGF(("[%d] Incrementing Step %d \n",CkMyPe(),step()));
  // if sync resume, invoke a barrier

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    savedBalancing = balancing;
    startLoadBalancingMlog(&resumeCentralLbAfterChkpt,(void *)this);
#endif

  LBDatabase::Object()->MigrationDone();    // call registered callbacks

  LoadbalanceDone(balancing);        // callback
#if (!defined(_FAULT_MLOG_) && !defined(_FAULT_CAUSAL_))
  // if sync resume invoke a barrier
  if (balancing && _lb_args.syncResume()) {
    CkCallback cb(CkIndex_CentralLB::ResumeClients((CkReductionMsg*)NULL), 
                  thisProxy);
    contribute(0, NULL, CkReduction::sum_int, cb);
  }
  else{	
    if(CmiNodeAlive(CkMyPe())){
	thisProxy [CkMyPe()].ResumeClients(balancing);
    }	
  }	
#if CMK_GRID_QUEUE_AVAILABLE
  CmiGridQueueDeregisterAll ();
  CpvAccess(CkGridObject) = NULL;
#endif
#endif 
#endif
}

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
void CentralLB::endMigrationDone(int balancing){
    DEBUGF(("[%d] CentralLB::endMigrationDone step %d\n",CkMyPe(),step()));


  if (balancing && _lb_args.syncResume()) {
    CkCallback cb(CkIndex_CentralLB::ResumeClients((CkReductionMsg*)NULL),
                  thisProxy);
    contribute(0, NULL, CkReduction::sum_int, cb);
  }
  else{
    if(CmiNodeAlive(CkMyPe())){
    DEBUGF(("[%d] Sending ResumeClients balancing %d \n",CkMyPe(),balancing));
    thisProxy [CkMyPe()].ResumeClients(balancing);
    }
  }

}
#endif

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
void resumeCentralLbAfterChkpt(void *_lb){
    CentralLB *lb= (CentralLB *)_lb;
    CpvAccess(_currentObj)=lb;
    lb->endMigrationDone(lb->savedBalancing);
}
#endif


void CentralLB::ResumeClients(CkReductionMsg *msg)
{
  ResumeClients(1);
  delete msg;
}

void CentralLB::ResumeClients(int balancing)
{
#if CMK_LBDB_ON
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    resumeCount++;
    globalResumeCount = resumeCount;
#endif
  DEBUGF(("[%d] Resuming clients. balancing:%d.\n",CkMyPe(),balancing));
  if (balancing && _lb_args.debug() && CkMyPe() == cur_ld_balancer) {
    double end_lb_time = CkWallTimer();
  }

#if (!defined(_FAULT_MLOG_) && !defined(_FAULT_CAUSAL_))
  if (balancing) ComlibNotifyMigrationDone();  
#endif

  theLbdb->ResumeClients();
  if (balancing)  {

    CheckMigrationComplete();
    if (future_migrates_expected == 0 || 
            future_migrates_expected == future_migrates_completed) {
      CheckMigrationComplete();
    }
  }
#endif
}

/*
  migration of objects contains two different kinds:
  (1) objects want to make a barrier for migration completion
      (waitForBarrier is true)
      migrationDone() to finish and resumeClients
  (2) objects don't need a barrier
  However, next load balancing can only happen when both migrations complete
*/ 
void CentralLB::CheckMigrationComplete()
{
#if CMK_LBDB_ON
  lbdone ++;
  if (lbdone == 2) {
    if (_lb_args.debug() && CkMyPe()==0) {
      double end_lb_time = CkWallTimer();
      CkPrintf("CharmLB> %s: PE [%d] step %d finished at %f duration %f s\n\n",
                lbname, cur_ld_balancer, step()-1, end_lb_time,
		end_lb_time-start_lb_time);
    }

    //FIX ME!!! adaptive_struct.lb_migration_cost = (CkWallTimer() - start_lb_time);
//NOTE    theLbdb->SetMigrationCost(CkWallTimer() - start_lb_time);

    lbdone = 0;
    future_migrates_expected = -1;
    future_migrates_completed = 0;


    DEBUGF(("[%d] Migration Complete\n", CkMyPe()));
    // release local barrier  so that the next load balancer can go
    LDOMHandle h;
    h.id.id.idx = 0;
    theLbdb->getLBDB()->DoneRegisteringObjects(h);
    // switch to the next load balancer in the list
    // subtle: called from Migrated() may result in Migrated() called in next LB
    theLbdb->nextLoadbalancer(seqno);
  }
#endif
}

void CentralLB::preprocess(LDStats* stats)
{
  if (_lb_args.ignoreBgLoad())
    stats->clearBgLoad();

  // Call the predictor for the future
  if (_lb_predict) FuturePredictor(statsData);
}

// default load balancing strategy
LBMigrateMsg* CentralLB::Strategy(LDStats* stats)
{
#if CMK_LBDB_ON
  double strat_start_time = CkWallTimer();
  if (_lb_args.debug())
    CkPrintf("CharmLB> %s: PE [%d] strategy starting at %f\n", lbname, cur_ld_balancer, strat_start_time);

  work(stats);


  if (_lb_args.debug()>2)  {
    CkPrintf("CharmLB> Obj Map:\n");
    for (int i=0; i<stats->n_objs; i++) CkPrintf("%d ", stats->to_proc[i]);
    CkPrintf("\n");
  }

  LBMigrateMsg *msg = createMigrateMsg(stats);

  int clients = CkNumPes();
  LBInfo info(clients);
  getPredictedLoadWithMsg(stats, clients, msg, info, 0);
  LBRealType mLoad, mCpuLoad, totalLoad, totalLoadWComm;
  info.getSummary(mLoad, mCpuLoad, totalLoad);
  //CkPrintf("CharmLB> Max load w/o comm %lf Max cpu load %lf Avg load %lf\n", mLoad, mCpuLoad, totalLoad/clients);
  //info.print();
//NOTE!!  theLbdb->UpdateAfterLBData(mLoad, mCpuLoad, totalLoad/clients);

  //getPredictedLoadWithMsg(stats, clients, msg, info,1);
  //info.getSummary(mLoad, mCpuLoad, totalLoadWComm);
  //info.print();
  //CkPrintf("CharmLB> Max load with comm %lf Max cpu load %lf Avg load %lf\n", mLoad, mCpuLoad, totalLoad/clients);
  //int nmsgs, nbytes;
  //stats->computeNonlocalComm(nmsgs, nbytes);
  //CkPrintf("CharmLB> Non local communication %d msg and %d bytes\n", nmsgs, nbytes);


  //long msg_n;
  //long long bytes_n;
  //stats->computeComm(msg_n, bytes_n);
  //CkPrintf("CharmLB> Total communication %ld msg and %lld bytes\n", nmsgs, nbytes);

  //double alpha_beta_cost = (msg_n * alpha) + (bytes_n * beta);
  //theLbdb->UpdateAfterLBComm(alpha_beta_cost/totalLoad);

  if (_lb_args.debug()) {
    double strat_end_time = CkWallTimer();
    envelope *env = UsrToEnv(msg);

    double lbdbMemsize = LBDatabase::Object()->useMem()/1000;
    CkPrintf("CharmLB> %s: PE [%d] Memory: LBManager: %d KB CentralLB: %d KB\n",
  	      lbname, cur_ld_balancer, (int)lbdbMemsize, (int)(useMem()/1000));
    CkPrintf("CharmLB> %s: PE [%d] #Objects migrating: %d, LBMigrateMsg size: %.2f MB\n", lbname, cur_ld_balancer, msg->n_moves, env->getTotalsize()/1024.0/1024.0);
    CkPrintf("CharmLB> %s: PE [%d] strategy finished at %f duration %f s\n",
	      lbname, cur_ld_balancer, strat_end_time, strat_end_time-strat_start_time);
    //CkPrintf("Strategy cost %f %f %f\n", strat_end_time, strat_start_time, adaptive_struct.lb_strategy_cost);
//NOTE    theLbdb->SetStrategyCost(strat_end_time - strat_start_time);
  }
  return msg;
#else
  return NULL;
#endif
}

void CentralLB::work(LDStats* stats)
{
  // does nothing but print the database
  stats->print();
}

// generate migrate message from stats->from_proc and to_proc
LBMigrateMsg * CentralLB::createMigrateMsg(LDStats* stats)
{
  int i;
  CkVec<MigrateInfo*> migrateInfo;
  for (i=0; i<stats->n_objs; i++) {
    LDObjData &objData = stats->objData[i];
    int frompe = stats->from_proc[i];
    int tope = stats->to_proc[i];
    if (frompe != tope) {
      //      CkPrintf("[%d] Obj %d migrating from %d to %d\n",
      //         CkMyPe(),obj,pe,dest);
      MigrateInfo *migrateMe = new MigrateInfo;
      migrateMe->obj = objData.handle;
      migrateMe->from_pe = frompe;
      migrateMe->to_pe = tope;
      migrateMe->async_arrival = objData.asyncArrival;
      migrateInfo.insertAtEnd(migrateMe);
    }
  }

  int migrate_count=migrateInfo.length();
  LBMigrateMsg* msg = new(migrate_count,CkNumPes(),CkNumPes(),0) LBMigrateMsg;
  msg->n_moves = migrate_count;
  for(i=0; i < migrate_count; i++) {
    MigrateInfo* item = (MigrateInfo*) migrateInfo[i];
    msg->moves[i] = *item;
    delete item;
    migrateInfo[i] = 0;
  }
  return msg;
}

LBMigrateMsg * CentralLB::extractMigrateMsg(LBMigrateMsg *m, int p)
{
  int nmoves = 0;
  int nunavail = 0;
  int i;
  for (i=0; i<m->n_moves; i++) {
    MigrateInfo* item = (MigrateInfo*) &m->moves[i];
    if (item->from_pe == p || item->to_pe == p) nmoves++;
  }
  for (i=0; i<CkNumPes();i++) {
    if (!m->avail_vector[i]) nunavail++;
  }
  LBMigrateMsg* msg;
  if (nunavail) msg = new(nmoves,CkNumPes(),CkNumPes(),0) LBMigrateMsg;
  else msg = new(nmoves,0,0,0) LBMigrateMsg;
  msg->n_moves = nmoves;
  msg->level = m->level;
  msg->next_lb = m->next_lb;
  for (i=0,nmoves=0; i<m->n_moves; i++) {
    MigrateInfo* item = (MigrateInfo*) &m->moves[i];
    if (item->from_pe == p || item->to_pe == p) {
      msg->moves[nmoves] = *item;
      nmoves++;
    }
  }
  // copy processor data
  if (nunavail)
  for (i=0; i<CkNumPes();i++) {
    msg->avail_vector[i] = m->avail_vector[i];
    msg->expectedLoad[i] = m->expectedLoad[i];
  }
  return msg;
}

void CentralLB::simulationWrite() {
  if(step() == LBSimulation::dumpStep)
  {
    // here we are supposed to dump the database
    int dumpFileSize = strlen(LBSimulation::dumpFile) + 4;
    char *dumpFileName = (char *)malloc(dumpFileSize);
    while (sprintf(dumpFileName, "%s.%d", LBSimulation::dumpFile, LBSimulation::dumpStep) >= dumpFileSize) {
      free(dumpFileName);
      dumpFileSize+=3;
      dumpFileName = (char *)malloc(dumpFileSize);
    }
    writeStatsMsgs(dumpFileName);
    free(dumpFileName);
    CmiPrintf("LBDump: Dumped the load balancing data at step %d.\n",LBSimulation::dumpStep);
    ++LBSimulation::dumpStep;
    --LBSimulation::dumpStepSize;
    if (LBSimulation::dumpStepSize <= 0) { // prevent stupid step sizes
      CmiPrintf("Charm++> Exiting...\n");
      CkExit();
    }
    return;
  }
}

void CentralLB::simulationRead() {
  LBSimulation *simResults = NULL, *realResults;
  LBMigrateMsg *voidMessage = new (0,0,0,0) LBMigrateMsg();
  voidMessage->n_moves=0;
  for ( ;LBSimulation::simStepSize > 0; --LBSimulation::simStepSize, ++LBSimulation::simStep) {
    // here we are supposed to read the data from the dump database
    int simFileSize = strlen(LBSimulation::dumpFile) + 4;
    char *simFileName = (char *)malloc(simFileSize);
    while (sprintf(simFileName, "%s.%d", LBSimulation::dumpFile, LBSimulation::simStep) >= simFileSize) {
      free(simFileName);
      simFileSize+=3;
      simFileName = (char *)malloc(simFileSize);
    }
    readStatsMsgs(simFileName);

    // allocate simResults (only the first step)
    if (simResults == NULL) {
      simResults = new LBSimulation(LBSimulation::simProcs);
      realResults = new LBSimulation(LBSimulation::simProcs);
    }
    else {
      // should be the same number of procs of the original simulation!
      if (!LBSimulation::procsChanged) {
	// it means we have a previous step, so in simResults there is data.
	// we can now print the real effects of the load balancer during the simulation
	// or print the difference between the predicted data and the real one.
	realResults->reset();
	// reset to_proc of statsData to be equal to from_proc
	for (int k=0; k < statsData->n_objs; ++k) statsData->to_proc[k] = statsData->from_proc[k];
	findSimResults(statsData, LBSimulation::simProcs, voidMessage, realResults);
	simResults->PrintDifferences(realResults,statsData);
      }
      simResults->reset();
    }

    // now pass it to the strategy routine
    double startT = CkWallTimer();
    preprocess(statsData);
    CmiPrintf("%s> Strategy starts ... \n", lbname);
    LBMigrateMsg* migrateMsg = Strategy(statsData);
    CmiPrintf("%s> Strategy took %fs memory usage: CentralLB: %d KB.\n",
               lbname, CkWallTimer()-startT, (int)(useMem()/1000));

    // now calculate the results of the load balancing simulation
    findSimResults(statsData, LBSimulation::simProcs, migrateMsg, simResults);

    // now we have the simulation data, so print it and loop
    CmiPrintf("Charm++> LBSim: Simulation of load balancing step %d done.\n",LBSimulation::simStep);
    // **CWL** Officially recording my disdain here for using ints for bool
    if (LBSimulation::showDecisionsOnly) {
      simResults->PrintDecisions(migrateMsg, simFileName, 
				 LBSimulation::simProcs);
    } else {
      simResults->PrintSimulationResults();
    }

    free(simFileName);
    delete migrateMsg;
    CmiPrintf("Charm++> LBSim: Passing to the next step\n");
  }
  // deallocate simResults
  delete simResults;
  CmiPrintf("Charm++> Exiting...\n");
  CkExit();
}

void CentralLB::readStatsMsgs(const char* filename) 
{
#if CMK_LBDB_ON
  int i;
  FILE *f = fopen(filename, "r");
  if (f==NULL) {
    CmiPrintf("Fatal Error> Cannot open LB Dump file %s!\n", filename);
    CmiAbort("");
  }

  // at this stage, we need to rebuild the statsMsgList and
  // statsDataList structures. For that first deallocate the
  // old structures
  if (statsMsgsList) {
    for(i = 0; i < stats_msg_count; i++)
      delete statsMsgsList[i];
    delete[] statsMsgsList;
    statsMsgsList=0;
  }

  PUP::fromDisk pd(f);
  PUP::machineInfo machInfo;

  pd((char *)&machInfo, sizeof(machInfo));	// read machine info
  PUP::xlater p(machInfo, pd);

  if (_lb_args.lbversion() > 1) {
    p|_lb_args.lbversion();		// write version number
    CkPrintf("LB> File version detected: %d\n", _lb_args.lbversion());
    CmiAssert(_lb_args.lbversion() <= LB_FORMAT_VERSION);
  }
  p|stats_msg_count;

  CmiPrintf("readStatsMsgs for %d pes starts ... \n", stats_msg_count);
  if (LBSimulation::simProcs == 0) LBSimulation::simProcs = stats_msg_count;
  if (LBSimulation::simProcs != stats_msg_count) LBSimulation::procsChanged = true;

  // LBSimulation::simProcs must be set
  statsData->pup(p);

  CmiPrintf("Simulation for %d pes \n", LBSimulation::simProcs);
  CmiPrintf("n_obj: %d n_migratble: %d \n", statsData->n_objs, statsData->n_migrateobjs);

  // file f is closed in the destructor of PUP::fromDisk
  CmiPrintf("ReadStatsMsg from %s completed\n", filename);
#endif
}

void CentralLB::writeStatsMsgs(const char* filename) 
{
#if CMK_LBDB_ON
  FILE *f = fopen(filename, "w");
  if (f==NULL) {
    CmiPrintf("Fatal Error> writeStatsMsgs failed to open the output file %s!\n", filename);
    CmiAbort("");
  }

  const PUP::machineInfo &machInfo = PUP::machineInfo::current();
  PUP::toDisk p(f);
  p((char *)&machInfo, sizeof(machInfo));	// machine info

  p|_lb_args.lbversion();		// write version number
  p|stats_msg_count;
  statsData->pup(p);

  fclose(f);

  CmiPrintf("WriteStatsMsgs to %s succeed!\n", filename);
#endif
}

// calculate the predicted wallclock/cpu load for every processors
// considering communication overhead if considerComm is true
void getPredictedLoadWithMsg(BaseLB::LDStats* stats, int count, 
                      LBMigrateMsg *msg, LBInfo &info, 
		      int considerComm)
{
#if CMK_LBDB_ON
	stats->makeCommHash();

 	// update to_proc according to migration msgs
	for(int i = 0; i < msg->n_moves; i++) {
	  MigrateInfo &mInfo = msg->moves[i];
	  int idx = stats->getHash(mInfo.obj.objID(), mInfo.obj.omID());
	  CmiAssert(idx != -1);
          stats->to_proc[idx] = mInfo.to_pe;
	}

	info.getInfo(stats, count, considerComm);
#endif
}


void CentralLB::findSimResults(LDStats* stats, int count, LBMigrateMsg* msg, LBSimulation* simResults)
{
    CkAssert(simResults != NULL && count == simResults->numPes);
    // estimate the new loads of the processors. As a first approximation, this is the
    // sum of the cpu times of the objects on that processor
    double startT = CkWallTimer();
    getPredictedLoadWithMsg(stats, count, msg, simResults->lbinfo, 1);
    CmiPrintf("getPredictedLoad finished in %fs\n", CkWallTimer()-startT);
}

void CentralLB::pup(PUP::er &p) { 
  BaseLB::pup(p); 
  if (p.isUnpacking())  {
    initLB(CkLBOptions(seqno)); 
  }
  p|reduction_started;
  int has_statsMsg=0;
  if (p.isPacking()) has_statsMsg = (statsMsg!=NULL);
  p|has_statsMsg;
  if (has_statsMsg) {
    if (p.isUnpacking())
      statsMsg = new CLBStatsMsg;
    statsMsg->pup(p);
  }
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
  p | lbDecisionCount;
  p | resumeCount;
#endif
	
}

int CentralLB::useMem() { 
  return sizeof(CentralLB) + statsData->useMem() + 
         CkNumPes() * sizeof(CLBStatsMsg *);
}


/**
  CLBStatsMsg is not a real message now.
  CLBStatsMsg is used for all processors to fill in their local load and comm
  statistics and send to processor 0
*/

CLBStatsMsg::CLBStatsMsg(int osz, int csz) {
  n_objs = osz;
  n_comm = csz;
  objData = new LDObjData[osz];
  commData = new LDCommData[csz];
  avail_vector = NULL;
}

CLBStatsMsg::~CLBStatsMsg() {
  delete [] objData;
  delete [] commData;
  delete [] avail_vector;
}

void CLBStatsMsg::pup(PUP::er &p) {
  int i;
  p|from_pe;
  p|pe_speed;
  p|total_walltime;
  p|idletime;
  p|bg_walltime;
#if CMK_LB_CPUTIMER
  p|total_cputime;
  p|bg_cputime;
#endif
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
  p | step;
#endif
  p|n_objs;
  if (p.isUnpacking()) objData = new LDObjData[n_objs];
  for (i=0; i<n_objs; i++) p|objData[i];
  p|n_comm;
  if (p.isUnpacking()) commData = new LDCommData[n_comm];
  for (i=0; i<n_comm; i++) p|commData[i];

  int has_avail_vector;
  if (!p.isUnpacking()) has_avail_vector = (avail_vector != NULL);
  p|has_avail_vector;
  if (p.isUnpacking()) {
    if (has_avail_vector) avail_vector = new char[CkNumPes()];
    else avail_vector = NULL;
  }
  if (has_avail_vector) p(avail_vector, CkNumPes());

  p(next_lb);
}

// CkMarshalledCLBStatsMessage is used in the marshalled parameter in
// the entry function, it is just used to use to pup.
// I don't use CLBStatsMsg directly as marshalled parameter because
// I want the data pointer stored and not to be freed by the Charm++.
void CkMarshalledCLBStatsMessage::free() { 
  int count = msgs.size();
  for  (int i=0; i<count; i++) {
    delete msgs[i];
    msgs[i] = NULL;
  }
  msgs.free();
}

void CkMarshalledCLBStatsMessage::add(CkMarshalledCLBStatsMessage &m)
{
  int count = m.getCount();
  for (int i=0; i<count; i++) add(m.getMessage(i));
}

void CkMarshalledCLBStatsMessage::pup(PUP::er &p)
{
  int count = msgs.size();
  p|count;
  for (int i=0; i<count; i++) {
    CLBStatsMsg *msg;
    if (p.isUnpacking()) msg = new CLBStatsMsg;
    else { 
      msg = msgs[i]; CmiAssert(msg!=NULL);
    }
    msg->pup(p);
    if (p.isUnpacking()) add(msg);
  }
}

SpanningTree::SpanningTree()
{
	double sq = sqrt(CkNumPes()*4.0-3.0) - 1; // 1 + arity + arity*arity = CkNumPes()
	arity = (int)ceil(sq/2);
	calcParent(CkMyPe());
	calcNumChildren(CkMyPe());
}

void SpanningTree::calcParent(int n)
{
	parent=-1;
	if(n != 0  && arity > 0)
		parent = (n-1)/arity;
}

void SpanningTree::calcNumChildren(int n)
{
	numChildren = 0;
	if (arity == 0) return;
	int fullNode=(CkNumPes()-1-arity)/arity;
	if(n <= fullNode)
		numChildren = arity;
	if(n == fullNode+1)
		numChildren = CkNumPes()-1-(fullNode+1)*arity;
	if(n > fullNode+1)
		numChildren = 0;
}

#include "CentralLB.def.h"
 
/*@}*/
