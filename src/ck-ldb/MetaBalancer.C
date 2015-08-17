/**
 * Author: Harshitha Menon
 * Date created: 2012
*/
/*@{*/

#include "converse.h"

/*
 */

#include "MetaBalancer.h"
#include "topology.h"

#include "limits.h"

#define VEC_SIZE 50
#define IMB_TOLERANCE 1.1
#define OUTOFWAY_TOLERANCE 2
#define UTILIZATION_THRESHOLD 0.7
#define NEGLECT_IDLE 2 // Should never be == 1
#define MIN_STATS 6
#define STATS_COUNT 8 // The number of stats collected during reduction

#define DEBAD(x) /*CkPrintf x*/
#define DEBADDETAIL(x) /*CkPrintf x*/
#define EXTRA_FEATURE 0

CkReductionMsg* lbDataCollection(int nMsg, CkReductionMsg** msgs) {
  double lb_data[STATS_COUNT];
  lb_data[1] = 0.0; // total number of processors contributing
  lb_data[2] = 0.0; // total load
  lb_data[3] = 0.0; // max load
  lb_data[4] = 0.0; // idle time
  lb_data[5] = 1.0; // utilization
  lb_data[6] = 0.0; // total load with bg
  lb_data[7] = 0.0; // max load with bg
  for (int i = 0; i < nMsg; i++) {
    CkAssert(msgs[i]->getSize() == STATS_COUNT*sizeof(double));
    if (msgs[i]->getSize() != STATS_COUNT*sizeof(double)) {
      CkPrintf("Error!!! Reduction not correct. Msg size is %d\n",
          msgs[i]->getSize());
      CkAbort("Incorrect Reduction size in MetaBalancer\n");
    }
    double* m = (double *)msgs[i]->getData();
    // Total count
    lb_data[1] += m[1];
    // Avg load
    lb_data[2] += m[2];
    // Max load
    lb_data[3] = ((m[3] > lb_data[3])? m[3] : lb_data[3]);
    // Avg idle
    lb_data[4] += m[4];
    // Get least utilization
    lb_data[5] = ((m[5] < lb_data[5]) ? m[5] : lb_data[5]);
    // Get Avg load with bg
    lb_data[6] += m[6];
    // Get Max load with bg
    lb_data[7] = ((m[7] > lb_data[7])? m[7] : lb_data[7]);
    if (i == 0) {
      // Iteration no
      lb_data[0] = m[0];
    }
    if (m[0] != lb_data[0]) {
      CkPrintf("Error!!! Reduction is intermingled between iteration %lf \
        and %lf\n", lb_data[0], m[0]);
      CkAbort("Intermingling iterations in MetaBalancer\n");
    }
  }
  return CkReductionMsg::buildNew(STATS_COUNT*sizeof(double), lb_data);
}

/*global*/ CkReduction::reducerType lbDataCollectionType;
/*initnode*/ void registerLBDataCollection(void) {
  lbDataCollectionType = CkReduction::addReducer(lbDataCollection);
}

CkGroupID _metalb;
CkGroupID _metalbred;

CkpvDeclare(int, metalbInited);  /**< true if metabalancer is inited */

double _nobj_timer = 10.0;

// mainchare
MetaLBInit::MetaLBInit(CkArgMsg *m) {
#if CMK_LBDB_ON
  _metalbred = CProxy_MetaBalancerRedn::ckNew();
  _metalb = CProxy_MetaBalancer::ckNew();
#endif
  delete m;
}

// called from init.C
void _metabalancerInit() {
  _registerCommandLineOpt("+MetaLBNoObjTimer");
  CkpvInitialize(int, metalbInited);
  CkpvAccess(metalbInited) = 0;
  char **argv = CkGetArgv();
	CmiGetArgDoubleDesc(argv, "+MetaLBNoObjTimer", &_nobj_timer,
			"Time in seconds before triggering reduction for no objs");
}

void MetaBalancer::initnodeFn() {
}

// called by my constructor
void MetaBalancer::init(void) {
  lbdatabase = (LBDatabase *)CkLocalBranch(_lbdb);
  CkpvAccess(metalbInited) = 1;
  total_load_vec.resize(VEC_SIZE, 0.0);
  total_count_vec.resize(VEC_SIZE, 0);
  prev_idle = 0.0;
  alpha_beta_cost_to_load = 1.0; // Some random value. TODO: Find the actual

  metaRdnGroup = (MetaBalancerRedn*)CkLocalBranch(_metalbred);

  adaptive_lbdb.lb_iter_no = -1;

  // If metabalancer enabled, initialize the variables
  adaptive_struct.tentative_period =  INT_MAX;
  adaptive_struct.final_lb_period =  INT_MAX;
  adaptive_struct.lb_calculated_period = INT_MAX;
  adaptive_struct.lb_iteration_no = -1;
  adaptive_struct.finished_iteration_no = -1;
  adaptive_struct.global_max_iter_no = 0;
  adaptive_struct.tentative_max_iter_no = -1;
  adaptive_struct.in_progress = false;
  adaptive_struct.lb_strategy_cost = 0.0;
  adaptive_struct.lb_migration_cost = 0.0;
  adaptive_struct.lb_msg_send_no = 0;
  adaptive_struct.lb_msg_recv_no = 0;
  adaptive_struct.total_syncs_called = 0;
  adaptive_struct.last_lb_type = -1;


  // This is indicating if the load balancing strategy and migration started.
  // This is mainly used to register callbacks for noobj pes. They would
  // register as soon as resumefromsync is called. On receiving the handles at
  // the central pe, it clears the previous handlers and sets lb_in_progress
  // to false so that it doesn't clear the handles.
  lb_in_progress = false;

  is_prev_lb_refine = -1;
  if (_lb_args.metaLbOn()) {
    periodicCall((void *) this);
  }
}

void MetaBalancer::pup(PUP::er& p) {
  if (p.isUnpacking()) {
    lbdatabase = (LBDatabase *)CkLocalBranch(_lbdb);
    metaRdnGroup = (MetaBalancerRedn*)CkLocalBranch(_metalbred);
  }
  p|prev_idle;
  p|alpha_beta_cost_to_load;
  p|is_prev_lb_refine;
  p|lb_in_progress;
}


void MetaBalancer::ResumeClients() {
  // If metabalancer enabled, initialize the variables
  adaptive_lbdb.history_data.clear();

  adaptive_struct.tentative_period =  INT_MAX;
  adaptive_struct.final_lb_period =  INT_MAX;
  adaptive_struct.lb_calculated_period = INT_MAX;
  adaptive_struct.lb_iteration_no = -1;
  adaptive_struct.finished_iteration_no = -1;
  adaptive_struct.global_max_iter_no = 0;
  adaptive_struct.tentative_max_iter_no = -1;
  adaptive_struct.in_progress = false;
  adaptive_struct.lb_strategy_cost = 0.0;
  adaptive_struct.lb_migration_cost = 0.0;
  adaptive_struct.lb_msg_send_no = 0;
  adaptive_struct.lb_msg_recv_no = 0;
  adaptive_struct.total_syncs_called = 0;

  prev_idle = 0.0;
  if (lb_in_progress) {
    lbdb_no_obj_callback.clear();
    lb_in_progress = false;
  }
  HandleAdaptiveNoObj();
}

int MetaBalancer::get_iteration() {
  return adaptive_struct.lb_iteration_no;
}

int MetaBalancer::get_finished_iteration() {
  return adaptive_struct.finished_iteration_no;
}

void MetaBalancer::AdjustCountForNewContributor(int it_n) {
#if CMK_LBDB_ON
  int index;

  // it_n is the first iteration this chare will contribute to.
  // If the finished_iteration_no is < it_n, then we need to update the counts
  // of all the other iterations from finished_iteration_no + 1 to it_n to
  // discount the newly added chares.
  for (int i = (get_finished_iteration()+1); i <= it_n; i++) {
    index = i % VEC_SIZE;
    total_count_vec[index]++;
  }
#endif
}

void MetaBalancer::AdjustCountForDeadContributor(int it_n) {
#if CMK_LBDB_ON
  int index;
  // it_n is the last iteration this chare contributed to.
  // If the finished_iteration_no is < it_n, then we need to update the counts
  // of all the other iterations from finished_iteration_no + 1 to it_n.
  for (int i = (get_finished_iteration() + 1); i <= it_n; i++) {
    index = i % VEC_SIZE;
    total_count_vec[index]--;
  }

  // Check whether any of the future iterations now become valid
  for (int i = (it_n + 1); i <= adaptive_struct.lb_iteration_no; i++) {
    index = i % VEC_SIZE;
    // When this contributor dies, the objDataCount gets updated only later so
    // we need to account for that by -1
    if (total_count_vec[index] == (lbdatabase->getLBDB()->ObjDataCount() - 1)){
      ContributeStats(i);
    }
  }
#endif
}

bool MetaBalancer::AddLoad(int it_n, double load) {
#if CMK_LBDB_ON
  int index = it_n % VEC_SIZE;
  total_count_vec[index]++;
  adaptive_struct.total_syncs_called++;
  DEBADDETAIL(("At PE %d Total contribution for iteration %d is %d \
      total objs %d\n", CkMyPe(), it_n, total_count_vec[index],
      lbdatabase->getLBDB()->ObjDataCount()));

  if (it_n <= adaptive_struct.finished_iteration_no) {
    CkAbort("Error!! Received load for iteration that has contributed\n");
  }
  if (it_n > adaptive_struct.lb_iteration_no) {
    adaptive_struct.lb_iteration_no = it_n;
  }
  total_load_vec[index] += load;
  if (total_count_vec[index] > lbdatabase->getLBDB()->ObjDataCount()) {
    CkPrintf("iteration %d received %d contributions and expected %d\n", it_n,
        total_count_vec[index], lbdatabase->getLBDB()->ObjDataCount());
    CkAbort("Abort!!! Received more contribution");
  }
  if ((total_count_vec[index] == lbdatabase->getLBDB()->ObjDataCount())){
    ContributeStats(it_n);
  }
#endif
  return true;
}

void MetaBalancer::ContributeStats(int it_n) {
#if CMK_LBDB_ON
  int index = it_n % VEC_SIZE;

  double idle_time, bg_walltime, cpu_bgtime;
  lbdatabase->IdleTime(&idle_time);
  lbdatabase->BackgroundLoad(&bg_walltime, &cpu_bgtime);

  int sync_for_bg = adaptive_struct.total_syncs_called +
    lbdatabase->getLBDB()->ObjDataCount();
  bg_walltime = bg_walltime * lbdatabase->getLBDB()->ObjDataCount() / sync_for_bg;

  if (it_n < NEGLECT_IDLE) {
    prev_idle = idle_time;
  }
  idle_time -= prev_idle;

  // The chares do not contribute their 0th iteration load. So the total syncs
  // in reality is total_syncs_called + obj_counts
  int total_countable_syncs = adaptive_struct.total_syncs_called +
    (1 - NEGLECT_IDLE) * lbdatabase->getLBDB()->ObjDataCount(); // TODO: Fix me!
  if (total_countable_syncs != 0) {
    idle_time = idle_time * lbdatabase->getLBDB()->ObjDataCount() / total_countable_syncs;
  }

  double lb_data[STATS_COUNT];
  lb_data[0] = it_n;
  lb_data[1] = 1;
  lb_data[2] = total_load_vec[index]; // For average load
  lb_data[3] = total_load_vec[index]; // For max load
  // Set utilization
  if (total_load_vec[index] == 0.0) {
    lb_data[4] = 0.0;
    lb_data[5] = 0.0;
  } else {
    lb_data[4] = total_load_vec[index]/(idle_time + total_load_vec[index]);
    lb_data[5] = total_load_vec[index]/(idle_time + total_load_vec[index]);
  }
  lb_data[6] = lb_data[2] + bg_walltime; // For Avg load with bg
  lb_data[7] = lb_data[6]; // For Max load with bg
  total_load_vec[index] = 0.0;
  total_count_vec[index] = 0;

  adaptive_struct.finished_iteration_no = it_n;

  DEBADDETAIL(("[%d] sends total load %lf idle time %lf utilization %lf at iter %d\n",
        CkMyPe(), total_load_vec[index], idle_time,
        lb_data[5], adaptive_struct.finished_iteration_no));

  CkCallback cb(CkIndex_MetaBalancer::ReceiveMinStats((CkReductionMsg*)NULL), thisProxy[0]);
  contribute(STATS_COUNT*sizeof(double), lb_data, lbDataCollectionType, cb);

#endif
}

void MetaBalancer::ReceiveMinStats(CkReductionMsg *msg) {
  double* load = (double *) msg->getData();
  double avg = load[2]/load[1];
  double max = load[3];
  double avg_utilization = load[4]/load[1];
  double min_utilization = load[5];
  int iteration_n = (int) load[0];
  double avg_load_bg = load[6]/load[1];
  double max_load_bg = load[7];
	// Set the max and avg to be the load with background
	max = max_load_bg;
	avg = avg_load_bg;
  DEBAD(("** [%d] Iteration Avg load: %lf Max load: %lf Avg Util : %lf \
      Min Util : %lf for %lf procs\n",iteration_n, avg, max, avg_utilization,
      min_utilization, load[1]));
  delete msg;

  // For processors with  no  objs, trigger MetaBalancer reduction
  if (adaptive_struct.final_lb_period != iteration_n) {
    for (int i = 0; i < lbdb_no_obj_callback.size(); i++) {
      thisProxy[lbdb_no_obj_callback[i]].TriggerAdaptiveReduction();
    }
  }

  // Store the data for this iteration
  adaptive_lbdb.lb_iter_no = iteration_n;
  AdaptiveData data;
  data.iteration = adaptive_lbdb.lb_iter_no;
  data.max_load = max;
  data.avg_load = avg;
  data.min_utilization = min_utilization;
  data.avg_utilization = avg_utilization;
  adaptive_lbdb.history_data.push_back(data);

  if (iteration_n == 1) {
    adaptive_struct.info_first_iter.max_avg_ratio = max/avg;
  }


  if (adaptive_struct.final_lb_period == iteration_n) {
    thisProxy.MetaLBCallLBOnChares();
  }

  // If lb period inform is in progress, dont inform again.
  // If this is the lb data corresponding to the final lb period informed, then
  // don't recalculate as some of the processors might have already gone into
  // LB_STAGE.
  if (adaptive_struct.in_progress || 
      (adaptive_struct.final_lb_period == iteration_n)) {
    return;
  }

	// If the utilization is beyond 90%, then do nothing
	if (data.avg_utilization >= 0.90) {
		return;
	}

  double utilization_threshold = UTILIZATION_THRESHOLD;

#if EXTRA_FEATURE
  DEBAD(("alpha_beta_to_load %lf\n", alpha_beta_cost_to_load));
  if (alpha_beta_cost_to_load < 0.1) {
    // Ignore the effect of idle time and there by lesser utilization. So we
    // assign utilization threshold to be 0.0
    DEBAD(("Changing the idle load tolerance coz this isn't \
        communication intensive benchmark\n"));
    utilization_threshold = 0.0;
  }
#endif

  // First generate the lb period based on the cost of lb. Find out what is the
  // expected imbalance at the calculated lb period.
  int period;
  // This is based on the new max load after load balancing. So technically, it
  // is calculated based on the shifter up avg curve.
  double ratio_at_t = 1.0;
  int tmp_lb_type;
  double tmp_max_avg_ratio, tmp_comm_ratio;
  GetPrevLBData(tmp_lb_type, tmp_max_avg_ratio, tmp_comm_ratio);

  double tolerate_imb = IMB_TOLERANCE * tmp_max_avg_ratio;

  if (generatePlan(period, ratio_at_t)) {
    DEBAD(("Generated period and calculated %d and period %d max iter %d\n",
      adaptive_struct.lb_calculated_period, period,
      adaptive_struct.tentative_max_iter_no));
    // set the imbalance tolerance to be ratio_at_calculated_lb_period
    if (ratio_at_t != 1.0) {
      DEBAD(("Changed tolerance to %lf after line eq whereas max/avg is %lf\n",
        ratio_at_t, max/avg));
      // Since ratio_at_t is shifter up, max/(tmp_max_avg_ratio * avg) should be
      // compared with the tolerance
      tolerate_imb = ratio_at_t * tmp_max_avg_ratio * OUTOFWAY_TOLERANCE;
    }

    DEBAD(("Prev LB Data Type %d, max/avg %lf, local/remote %lf\n",
      tmp_lb_type, tmp_max_avg_ratio, tmp_comm_ratio));

#if EXTRA_FEATURE
    if ((avg_utilization < utilization_threshold || max/avg >= tolerate_imb) &&
          adaptive_lbdb.history_data.size() > MIN_STATS) {
      DEBAD(("Trigger soon even though we calculated lbperiod max/avg(%lf) and \
					utilization ratio (%lf)\n", max/avg, avg_utilization));
      TriggerSoon(iteration_n, max/avg, tolerate_imb);
      return;
    }
#endif

    // If the new lb period from linear extrapolation is greater than maximum
    // iteration known from previously collected data, then inform all the
    // processors about the new calculated period.
    if (period > adaptive_struct.tentative_max_iter_no && period !=
          adaptive_struct.final_lb_period) {
      adaptive_struct.doCommStrategy = false;
      adaptive_struct.lb_calculated_period = period;
      adaptive_struct.in_progress = true;
      DEBAD(("Sticking to the calculated period %d\n",
          adaptive_struct.lb_calculated_period));
      thisProxy.LoadBalanceDecision(adaptive_struct.lb_msg_send_no++,
        adaptive_struct.lb_calculated_period);
      return;
    }
    return;
  }

  DEBAD(("Prev LB Data Type %d, max/avg %lf, local/remote %lf\n", tmp_lb_type,
      tmp_max_avg_ratio, tmp_comm_ratio));

#if EXTRA_FEATURE
  // This would be called when linear extrapolation did not provide suitable
  // period provided there is enough  historical data 
  if ((avg_utilization < utilization_threshold || max/avg >= tolerate_imb) &&
			adaptive_lbdb.history_data.size() > 4) {
    DEBAD(("Carry out load balancing step at iter max/avg(%lf) and utilization \
      ratio (%lf)\n", max/avg, avg_utilization));
    TriggerSoon(iteration_n, max/avg, tolerate_imb);
    return;
  }
#endif

}

void MetaBalancer::TriggerSoon(int iteration_n, double imbalance_ratio,
    double tolerate_imb) {

  // If the previously calculated_period (not the final decision) is greater
  // than the iter +1 and if it is greater than the maximum iteration we have
  // seen so far, then we can inform this
  if ((iteration_n + 1 > adaptive_struct.tentative_max_iter_no) &&
      (iteration_n+1 < adaptive_struct.lb_calculated_period) &&
      (iteration_n + 1 != adaptive_struct.final_lb_period)) {
    if (imbalance_ratio < tolerate_imb) {
      adaptive_struct.doCommStrategy = true;
      DEBAD(("No load imbalance but idle time\n"));
    } else {
      adaptive_struct.doCommStrategy = false;
      DEBAD(("load imbalance \n"));
    }
    adaptive_struct.lb_calculated_period = iteration_n + 1;
    adaptive_struct.in_progress = true;
    DEBAD(("Informing everyone the lb period is %d\n",
        adaptive_struct.lb_calculated_period));
    thisProxy.LoadBalanceDecision(adaptive_struct.lb_msg_send_no++,
        adaptive_struct.lb_calculated_period);
  }
}

bool MetaBalancer::generatePlan(int& period, double& ratio_at_t) {
  if (adaptive_lbdb.history_data.size() <= 4) {
    return false;
  }

  // Some heuristics for lbperiod
  // If constant load or almost constant,
  // then max * new_lb_period > avg * new_lb_period + lb_cost
  double max = 0.0;
  double avg = 0.0;
  AdaptiveData data;
  for (int i = 0; i < adaptive_lbdb.history_data.size(); i++) {
    data = adaptive_lbdb.history_data[i];
    max += data.max_load;
    avg += data.avg_load;
    DEBAD(("max (%d, %lf) avg (%d, %lf)\n", i, data.max_load, i, data.avg_load));
  }
//  max /= (adaptive_struct.lb_iteration_no - adaptive_lbdb.history_data[0].iteration);
//  avg /= (adaptive_struct.lb_iteration_no - adaptive_lbdb.history_data[0].iteration);

  // If linearly varying load, then find lb_period
  // area between the max and avg curve
  // If we can attain perfect balance, then the new load is close to the
  // average. Hence we pass 1, else pass in some other value which would be the
  // new max_load after load balancing.
  int tmp_lb_type;
  double tmp_max_avg_ratio, tmp_comm_ratio;
  double tolerate_imb;

#if EXTRA_FEATURE
  // First get the data for refine.
  GetLBDataForLB(1, tmp_max_avg_ratio, tmp_comm_ratio);
  tolerate_imb = tmp_max_avg_ratio;

  // If RefineLB does a good job, then find the period considering RefineLB
  if (tmp_max_avg_ratio <= 1.01) {
    if (max/avg < tolerate_imb) {
      DEBAD(("Resorting to imb = 1.0 coz max/avg (%lf) < imb(%lf)\n", max/avg,
          tolerate_imb));
      tolerate_imb = 1.0;
    }
    DEBAD(("Will generate plan for refine %lf imb and %lf overhead\n",
        tolerate_imb, 0.2));
    return getPeriodForStrategy(tolerate_imb, 0.2, period, ratio_at_t);
  }

  GetLBDataForLB(0, tmp_max_avg_ratio, tmp_comm_ratio);
#endif

  GetPrevLBData(tmp_lb_type, tmp_max_avg_ratio, tmp_comm_ratio);
  tolerate_imb = tmp_max_avg_ratio;
//  if (max/avg < tolerate_imb) {
//    CkPrintf("Resorting to imb = 1.0 coz max/avg (%lf) < imb(%lf)\n", max/avg, tolerate_imb);
//    tolerate_imb = 1.0;
//  }
  if (max/avg > tolerate_imb) {
    if (getPeriodForStrategy(tolerate_imb, 1, period, ratio_at_t)) {
      return true;
    }
  }

  max = 0.0;
  avg = 0.0;
  for (int i = 0; i < adaptive_lbdb.history_data.size(); i++) {
    data = adaptive_lbdb.history_data[i];
    max += data.max_load;
    avg += data.avg_load*tolerate_imb;
  }
  max /= adaptive_lbdb.history_data.size();
  avg /= adaptive_lbdb.history_data.size();
  double cost = adaptive_struct.lb_strategy_cost + adaptive_struct.lb_migration_cost;
  period = (int) (cost/(max - avg));
  DEBAD(("Obtained period %d from constant prediction tolerated \
			imbalance(%f)\n", period, tolerate_imb));
  if (period < 0) { 
    period = adaptive_struct.final_lb_period;
    DEBAD(("Obtained -ve period from constant prediction so changing to prev %d\n", period));
  } 
  ratio_at_t = max / avg;
  return true;
}

bool MetaBalancer::getPeriodForStrategy(double new_load_percent,
    double overhead_percent, int& period, double& ratio_at_t) {
  double mslope, aslope, mc, ac;
  getLineEq(new_load_percent, aslope, ac, mslope, mc);
  DEBAD(("new load percent %lf\n", new_load_percent));
  DEBAD(("\n max: %fx + %f; avg: %fx + %f\n", mslope, mc, aslope, ac));
  double a = (mslope - aslope)/2;
  double b = (mc - ac);
  double c = -(adaptive_struct.lb_strategy_cost +
      adaptive_struct.lb_migration_cost) * overhead_percent;
  DEBAD(("cost %f\n",
      (adaptive_struct.lb_strategy_cost+adaptive_struct.lb_migration_cost)));
  bool got_period = getPeriodForLinear(a, b, c, period);
  if (!got_period) {
    return false;
  }

  if (mslope < 0) {
    if (period > (-mc/mslope)) {
      DEBAD(("Max < 0 Period set when max load is -ve\n"));
      return false;
    }
  }

  if (aslope < 0) {
    if (period > (-ac/aslope)) {
      DEBAD(("Avg < 0 Period set when avg load is -ve\n"));
      return false;
    }
  }

  int intersection_t = (int) ((mc-ac) / (aslope - mslope));
  if (intersection_t > 0 && period > intersection_t) {
    DEBAD(("Avg | Max Period set when curves intersect\n"));
    return false;
  }
  ratio_at_t = ((mslope*period + mc)/(aslope*period + ac));
  DEBAD(("Ratio at t (%lf*%d + %lf) / (%lf*%d+%lf) = %lf\n", mslope, period, mc, aslope, period, ac, ratio_at_t));
  return true;
}

bool MetaBalancer::getPeriodForLinear(double a, double b, double c, int& period) {
  DEBAD(("Quadratic Equation %lf X^2 + %lf X + %lf\n", a, b, c));
  if (a == 0.0) {
    period = (int) (-c / b);
    if (period < 0) {
      DEBAD(("-ve period for -c/b (%d)\n", period));
      return false;
    }
    DEBAD(("Ideal period for linear load %d\n", period));
    return true;
  }
  int x;
  double t = (b * b) - (4*a*c);
  if (t < 0) {
    DEBAD(("(b * b) - (4*a*c) is -ve sqrt : %lf\n", sqrt(t)));
    return false;
  }
  t = (-b + sqrt(t)) / (2*a);
  x = (int) t;
  if (x < 0) {
    DEBAD(("Oops!!! x (%d) < 0\n", x));
    x = 0;
    return false;
  }
  period = x;
  DEBAD(("Ideal period for linear load %d\n", period));
  return true;
}

bool MetaBalancer::getLineEq(double new_load_percent, double& aslope, double& ac, double& mslope, double& mc) {
  int total = adaptive_lbdb.history_data.size();
  int iterations = (int) (1 + adaptive_lbdb.history_data[total - 1].iteration -
      adaptive_lbdb.history_data[0].iteration);
  double a1 = 0;
  double m1 = 0;
  double a2 = 0;
  double m2 = 0;
  AdaptiveData data;
  int i = 0;
  for (i = 0; i < total/2; i++) {
    data = adaptive_lbdb.history_data[i];
    m1 += data.max_load;
    a1 += data.avg_load;
    DEBAD(("max (%d, %lf) avg (%d, %lf) adjusted_avg (%d, %lf)\n", i, data.max_load, i, data.avg_load, i, new_load_percent*data.avg_load));
  }
  m1 /= i;
  a1 = (a1 * new_load_percent) / i;

  for (i = total/2; i < total; i++) {
    data = adaptive_lbdb.history_data[i];
    m2 += data.max_load;
    a2 += data.avg_load;
    DEBAD(("max (%d, %lf) avg (%d, %lf) adjusted_avg (%d, %lf)\n", i, data.max_load, i, data.avg_load, i, new_load_percent*data.avg_load));
  }
  m2 /= (i - total/2);
  a2 = (a2 * new_load_percent) / (i - total/2);

  aslope = 2 * (a2 - a1) / iterations;
  mslope = 2 * (m2 - m1) / iterations;
  ac = adaptive_lbdb.history_data[0].avg_load * new_load_percent;
  mc = adaptive_lbdb.history_data[0].max_load;

  ac = a1 - ((aslope * total)/4);
  mc = m1 - ((mslope * total)/4);

  //ac = (adaptive_lbdb.history_data[1].avg_load * new_load_percent - aslope);
  //mc = (adaptive_lbdb.history_data[1].max_load - mslope);

  return true;
}

void MetaBalancer::LoadBalanceDecision(int req_no, int period) {
  if (req_no < adaptive_struct.lb_msg_recv_no) {
    DEBAD(("Error!!! Received a request which was already sent or old\n"));
    return;
  }
  DEBADDETAIL(("[%d] Load balance decision made cur iteration: %d period:%d\n",
			CkMyPe(), adaptive_struct.lb_iteration_no, period));
  adaptive_struct.tentative_period = period;
  adaptive_struct.lb_msg_recv_no = req_no;
  if (metaRdnGroup == NULL) {
    metaRdnGroup = (MetaBalancerRedn*)CkLocalBranch(_metalbred);
  }
  if (metaRdnGroup != NULL) {
    metaRdnGroup->getMaxIter(adaptive_struct.lb_iteration_no);
  }
}

void MetaBalancer::LoadBalanceDecisionFinal(int req_no, int period) {
  if (req_no < adaptive_struct.lb_msg_recv_no) {
    return;
  }
  DEBADDETAIL(("[%d] Final Load balance decision made cur iteration: %d \
			period:%d \n",CkMyPe(), adaptive_struct.lb_iteration_no, period));
  adaptive_struct.tentative_period = period;
  adaptive_struct.final_lb_period = period;
  lbdatabase->MetaLBResumeWaitingChares(period);
}

void MetaBalancer::MetaLBCallLBOnChares() {
  lbdatabase->MetaLBCallLBOnChares();
}

void MetaBalancer::ReceiveIterationNo(int local_iter_no) {
  CmiAssert(CkMyPe() == 0);

  if (local_iter_no > adaptive_struct.global_max_iter_no) {
    adaptive_struct.global_max_iter_no = local_iter_no;
  }

  int period;

    if (adaptive_struct.global_max_iter_no > adaptive_struct.tentative_max_iter_no) {
      adaptive_struct.tentative_max_iter_no = adaptive_struct.global_max_iter_no;
    }
    period = (adaptive_struct.tentative_period > adaptive_struct.global_max_iter_no) ?
				adaptive_struct.tentative_period : adaptive_struct.global_max_iter_no + 1;
    // If no one has gone into load balancing stage, then we can safely change
    // the period otherwise keep the old period.
    if (adaptive_struct.global_max_iter_no < adaptive_struct.final_lb_period) {
      adaptive_struct.tentative_period = period;
      DEBAD(("Final lb_period CHANGED!%d\n", adaptive_struct.tentative_period));
    } else {
      adaptive_struct.tentative_period = adaptive_struct.final_lb_period;
      DEBAD(("Final lb_period NOT CHANGED!%d\n", adaptive_struct.tentative_period));
    }
    thisProxy.LoadBalanceDecisionFinal(adaptive_struct.lb_msg_recv_no, adaptive_struct.tentative_period);
    adaptive_struct.in_progress = false;
}

int MetaBalancer::getPredictedLBPeriod(bool& is_tentative) {
  // If tentative and final_lb_period are the same, then the decision has been
  // made but if not, they are in the middle of consensus, hence return the
  // lease of the two
  if (adaptive_struct.tentative_period != adaptive_struct.final_lb_period) {
    is_tentative = true;
  } else {
    is_tentative = false;
  }
  if (adaptive_struct.tentative_period < adaptive_struct.final_lb_period) {
    return adaptive_struct.tentative_period;
   } else {
     return adaptive_struct.final_lb_period;
   }
}

// Called by CentralLB to indicate that the LB strategy and migration is in
// progress.
void MetaBalancer::ResetAdaptive() {
  adaptive_lbdb.lb_iter_no = -1;
  lb_in_progress = true;
}

// This is required for PEs with no objs
void MetaBalancer::periodicCall(void *ad) {
  MetaBalancer *s = (MetaBalancer *)ad;
  CcdCallFnAfterOnPE((CcdVoidFn)checkForNoObj, (void *)s, _nobj_timer, CkMyPe());
}

void MetaBalancer::checkForNoObj(void *ad) {
  MetaBalancer *s = (MetaBalancer *) ad;
  s->HandleAdaptiveNoObj();
}

// Called by LBDatabase to indicate that no objs are there in this processor
void MetaBalancer::HandleAdaptiveNoObj() {
#if CMK_LBDB_ON
  if (lbdatabase->getLBDB()->ObjDataCount() == 0) {
    adaptive_struct.finished_iteration_no++;
    adaptive_struct.lb_iteration_no++;
    DEBAD(("(%d) --HandleAdaptiveNoObj %d\n", CkMyPe(),
          adaptive_struct.finished_iteration_no));
    thisProxy[0].RegisterNoObjCallback(CkMyPe());
    TriggerAdaptiveReduction();
  }
#endif
}

void MetaBalancer::RegisterNoObjCallback(int index) {
  // If the load balancing process (migration) is going on and in the meantime
  // one of the processor finishes everything and finds that there are no objs
  // in it, then it registers a callback. So clear the old data in case it
  // hasn't been done.
  if (lb_in_progress) {
    lbdb_no_obj_callback.clear();
    lb_in_progress = false;
  }
  lbdb_no_obj_callback.push_back(index);
  DEBAD(("Registered %d to have no objs.\n", index));

  // If collection has already happened and this is second iteration, then
  // trigger reduction.
  if (adaptive_lbdb.lb_iter_no != -1) {
    DEBAD(("Collection already started now %d so kick in\n",
        adaptive_struct.finished_iteration_no));
    //thisProxy[index].TriggerAdaptiveReduction();
  }
}

void MetaBalancer::TriggerAdaptiveReduction() {
#if CMK_LBDB_ON
  if (lbdatabase->getLBDB()->ObjDataCount() == 0) {
    adaptive_struct.finished_iteration_no++;
    adaptive_struct.lb_iteration_no++;
    double lb_data[STATS_COUNT];
    lb_data[0] = adaptive_struct.finished_iteration_no;
    lb_data[1] = 1;
    lb_data[2] = 0.0;
    lb_data[3] = 0.0;
    lb_data[4] = 0.0;
    lb_data[5] = 0.0;
    lb_data[6] = 0.0;
    lb_data[7] = 0.0;

    DEBAD(("[%d] Triggered adaptive reduction for noobj %d\n", CkMyPe(),
          adaptive_struct.finished_iteration_no));

    CkCallback cb(CkIndex_MetaBalancer::ReceiveMinStats((CkReductionMsg*)NULL),
        thisProxy[0]);
    contribute(STATS_COUNT*sizeof(double), lb_data, lbDataCollectionType, cb);
  }
#endif
}


bool MetaBalancer::isStrategyComm() {
  return adaptive_struct.doCommStrategy;
}

void MetaBalancer::SetMigrationCost(double lb_migration_cost) {
  adaptive_struct.lb_migration_cost = lb_migration_cost;
}

void MetaBalancer::SetStrategyCost(double lb_strategy_cost) {
  adaptive_struct.lb_strategy_cost = lb_strategy_cost;
}

void MetaBalancer::UpdateAfterLBData(int lb, double lb_max, double lb_avg, double
    local_comm, double remote_comm) {
  adaptive_struct.last_lb_type = lb;
  if (lb == 0) {
    adaptive_struct.greedy_info.max_avg_ratio = lb_max/lb_avg;
  } else if (lb == 1) {
    adaptive_struct.refine_info.max_avg_ratio = lb_max/lb_avg;
  } else if (lb == 2) {
    adaptive_struct.comm_info.remote_local_ratio = remote_comm/local_comm;
  } else if (lb == 3) {
    adaptive_struct.comm_refine_info.remote_local_ratio =
    remote_comm/local_comm;
  }
}

void MetaBalancer::UpdateAfterLBData(double max_load, double max_cpu,
    double avg_load) {
  if (adaptive_struct.last_lb_type == -1) {
    adaptive_struct.last_lb_type = 0;
  }
  int lb = adaptive_struct.last_lb_type;
  if (lb == 0) {
    adaptive_struct.greedy_info.max_avg_ratio = max_load/avg_load;
  } else if (lb == 1) {
    adaptive_struct.refine_info.max_avg_ratio = max_load/avg_load;
  } else if (lb == 2) {
    adaptive_struct.comm_info.max_avg_ratio = max_load/avg_load;
  } else if (lb == 3) {
    adaptive_struct.comm_refine_info.max_avg_ratio = max_load/avg_load;
  }
}

void MetaBalancer::UpdateAfterLBComm(double alpha_beta_to_load) {
  DEBAD(("Setting alpha beta %lf\n", alpha_beta_to_load));
  alpha_beta_cost_to_load = alpha_beta_to_load;
}


void MetaBalancer::GetPrevLBData(int& lb_type, double& lb_max_avg_ratio,
    double& remote_local_comm_ratio) {
  lb_type = adaptive_struct.last_lb_type;
  lb_max_avg_ratio = 1;
  remote_local_comm_ratio = 1;
  GetLBDataForLB(lb_type, lb_max_avg_ratio, remote_local_comm_ratio);

  // Based on the first iteration
  lb_max_avg_ratio = adaptive_struct.info_first_iter.max_avg_ratio;
}

void MetaBalancer::GetLBDataForLB(int lb_type, double& lb_max_avg_ratio,
    double& remote_local_comm_ratio) {
  if (lb_type == 0) {
    lb_max_avg_ratio = adaptive_struct.greedy_info.max_avg_ratio;
  } else if (lb_type == 1) {
    lb_max_avg_ratio = adaptive_struct.refine_info.max_avg_ratio;
  } else if (lb_type == 2) {
    remote_local_comm_ratio = adaptive_struct.comm_info.remote_local_ratio;
  } else if (lb_type == 3) {
    remote_local_comm_ratio =
       adaptive_struct.comm_refine_info.remote_local_ratio;
  }
}

void MetaBalancerRedn::init() {
//  metabalancer = (MetaBalancer *)CkLocalBranch(_metalb);
  metabalancer = NULL;
}

void MetaBalancerRedn::pup(PUP::er& p) {
}

void MetaBalancerRedn::ReceiveIterNo(int max_iter) {
  CkAssert(CkMyPe() == 0);
  if (metabalancer == NULL) {
    metabalancer = (MetaBalancer*)CkLocalBranch(_metalb);
  }
  if (metabalancer != NULL) {
    metabalancer->ReceiveIterationNo(max_iter);
  }
}

void MetaBalancerRedn::getMaxIter(int max_iter) {
  CkCallback cb(CkReductionTarget(MetaBalancerRedn, ReceiveIterNo), thisProxy[0]);
  contribute(sizeof(int), &max_iter, CkReduction::max_int, cb);
}

#include "MetaBalancer.def.h"

/*@}*/
