/**
 * Author: Jonathan Lifflander
 *
 * Hierarchical Greedy LB based on the description in this paper:
 *
 * HPDC 12:
 *   "Work Stealing and Persistence-based Load Balancers for Iterative
 *    Overdecomposed Applications"
 *    Jonathan Lifflander, Sriram Krishnamoorthy, Laxmikant V. Kale
 *
*/

#if ! defined _HIERLB_H_
#define _HIERLB_H_

#include "HierarchicalLBTypes.h"
#include "DistBaseLB.h"
#include "HierarchicalLB.decl.h"

#include "ckheap.h"

#include <vector>
#include <unordered_map>
#include <map>
#include <list>
#include <limits>

void CreateHierarchicalLB();

struct ChildLoadInfo : HierLBTypes {
  double cur_load = 0.0;
  int node_size = 0, pe = 0;
  bool final_child = false;
  cont_hier_objid_t recs;

  ChildLoadInfo() = default;
};

struct HierarchicalLB : HierLBTypes, CBase_HierarchicalLB {
  using dist_base_lb_t = DistBaseLB;
  using lb_stats_t = dist_base_lb_t::LDStats;
  using child_load_t = ChildLoadInfo;

  HierarchicalLB(const CkLBOptions &);
  HierarchicalLB(CkMigrateMessage *m);

  void avg_load_reduction(double x);
  void done_hier();
  void turnOn();
  void turnOff();

  static hier_objid_t convert_to_hier_objid(int const& obj_id, int const& pe);
  static int hier_objid_get_pe(hier_objid_t const& id);
  static int hier_objid_get_id(hier_objid_t const& id);

  void setup_tree();
  void calc_load_over();
  void lb_tree_msg(
    double const child_load, int const child, cont_hier_objid_t load,
    int child_size
  );
  child_load_t* find_min_child();
  void down_tree_msg(
    int const& from, cont_hier_objid_t excess_load, bool final_child
  );
  void transfer_objects(
    int const& to_pe, std::vector<int> const& lst
  );
  void finished_transfer_requests();

  void send_down_tree();
  void distribute_amoung_children();

  bool QueryBalanceNow(int step) { return true; };

private:
  int rank = 0, nproc = 0;
  double my_load = 0.0, avg_load = 0.0, thr_avg = 0.0, total_child_load = 0.0;
  bool tree_is_setup = false;

  // instead of heap use a sample-based method or organize tasks
  cont_hier_objid_t obj_sample, given_objs, taken_objs, load_over;
  std::unordered_map<int, child_load_t*> children, live_children;
  std::unordered_map<int, std::vector<int>> transfers;
  int transfer_count = 0;
  int parent = -1, bottomParent = -1, child_msgs = 0, agg_node_size = 0;

  void clear_obj_map(cont_hier_objid_t& objs);

  lb_stats_t const* my_stats = nullptr;

  void InitLB(const CkLBOptions &);
  void LoadBalance();
  void Strategy(const DistBaseLB::LDStats* const stats);
  int histogram_time_sample(double const& time_milli);
};

#endif /*_HIERLB_H_ */
