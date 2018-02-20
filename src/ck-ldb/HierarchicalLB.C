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

#include "HierarchicalLB.h"
#include "HierarchicalLBTypes.h"

#include "elements.h"

CreateLBFunc_Def(
  HierarchicalLB, "The scalable hierarchical greedy load balancer"
)

#define HIER_LB_THRESHOLD 1.005
#define HIER_LB_NARY 8
#define HIER_LB_ROOT 0
#define HIER_LB_BIN_SIZE 10

#if DEBUG_HIER_LB_ON
  #define DEBUG_HIER_LB(...) CkPrintf(__VA_ARGS__)
#else
  #define DEBUG_HIER_LB(...)
#endif

HierarchicalLB::HierarchicalLB(CkMigrateMessage *m)
  : CBase_HierarchicalLB(m)
{ }

HierarchicalLB::HierarchicalLB(const CkLBOptions &opt)
  : CBase_HierarchicalLB(opt)
{
  lbname = "HierarchicalLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] HierarchicalLB created\n",CkMyPe());
  InitLB(opt);
}

void HierarchicalLB::turnOn()
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

void HierarchicalLB::turnOff()
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

void HierarchicalLB::InitLB(const CkLBOptions &opt) {
  thisProxy = CProxy_HierarchicalLB(thisgroup);
  if (opt.getSeqNo() > 0) turnOff();
}

int
HierarchicalLB::histogram_time_sample(double const& time_milli) {
  int const bin_size = HIER_LB_BIN_SIZE;
  int const bin = (
    (static_cast<int>(time_milli)) / bin_size * bin_size
  ) + bin_size;
  return bin;
}

/*static*/
HierarchicalLB::hier_objid_t
HierarchicalLB::convert_to_hier_objid(int const& obj_id, int const& pe) {
  hier_objid_t pe_conv = (hier_objid_t)pe;
  return ((pe_conv << 32) | (hier_objid_t)obj_id);
}

/*static*/
int
HierarchicalLB::hier_objid_get_pe(hier_objid_t const& id) {
  return (int)(id >> 32);
}

/*static*/
int
HierarchicalLB::hier_objid_get_id(hier_objid_t const& id) {
  return (int)(id);
}

void
HierarchicalLB::Strategy(const DistBaseLB::LDStats* const stats) {
  if (CkMyPe() == 0) {
    CkPrintf("[%d] In HierarchicalLB strategy\n", CkMyPe());
  }

  my_stats = stats;

  my_load = 0.0;
  for (int i = 0; i < my_stats->n_objs; i++) {
    auto const& time = my_stats->objData[i].wallTime;
    // convert time to millisecond to bin times
    auto time_milli = time * 1000;
    auto bin = histogram_time_sample(time_milli);
    // loads are all in millisecond units for uniformity with the binning
    // process to make the logic simpler
    my_load += time_milli;

    if (my_stats->objData[i].migratable) {
      hier_objid_t obj_id = HierarchicalLB::convert_to_hier_objid(i, CkMyPe());
      obj_sample[bin].push_back(obj_id);
    }

    // DEBUG_HIER_LB(
    //   "%d: my_load=%f, obj=%d, time=%f, time_milli=%f, bin=%d, obj_id=%ld, "
    //   "pe=%d, obj=%d\n",
    //   CkMyPe(), my_load, i, time, time_milli, bin, obj_id,
    //   hier_objid_get_pe(obj_id), hier_objid_get_id(obj_id)
    // );
  }

  DEBUG_HIER_LB(
    "%d: my_load=%f, n_objs=%d\n",
    CkMyPe(), my_load, my_stats->n_objs
  );

  if (!tree_is_setup) {
    setup_tree();
    tree_is_setup = true;
  }

  // Use reduction to obtain the average load in the system
  CkCallback cb(CkReductionTarget(HierarchicalLB, avg_load_reduction), thisProxy);
  contribute(sizeof(double), &my_load, CkReduction::sum_double, cb);
}

void
HierarchicalLB::setup_tree() {
  CkAssert(
    tree_is_setup == false &&
    "Tree must not already be set up when is this called"
  );

  rank = CkMyPe();
  nproc = CkNumPes();

  for (int i = 0; i < HIER_LB_NARY; i++) {
    int const child = rank * HIER_LB_NARY + i + 1;
    if (child < nproc) {
      children[child] = new ChildLoadInfo();
      DEBUG_HIER_LB("\t%d: child = %d\n", rank, child);
    }
  }

  if (children.size() == 0) {
    for (int i = 0; i < HIER_LB_NARY; i++) {
      int factorProcs = nproc / HIER_LB_NARY * HIER_LB_NARY;
      if (factorProcs < nproc) {
        factorProcs += HIER_LB_NARY;
      }
      int const child = (rank * HIER_LB_NARY + i + 1) - factorProcs - 1;
      //int child = (rank + 1 + theTwoSided().numProcRanks().toInt() - 1) / Nary;
      if (child < nproc && child >= 0) {
        children[child] = new ChildLoadInfo();
        children[child]->final_child = true;
        DEBUG_HIER_LB("\t%d: child-x = %d\n", rank, child);
      }
    }
  }

  parent = (rank - 1) / HIER_LB_NARY;

  int factorProcs = nproc / HIER_LB_NARY * HIER_LB_NARY;
  if (factorProcs < nproc) {
    factorProcs += HIER_LB_NARY;
  }

  bottomParent = ((rank + 1 + factorProcs) - 1) / HIER_LB_NARY;

  DEBUG_HIER_LB(
    "\t%d: parent=%d, bottomParent=%d, children.size()=%ld\n",
    rank, parent, bottomParent, children.size()
  );
}

void
HierarchicalLB::calc_load_over() {
  auto cur_item = obj_sample.begin();
  auto threshold = HIER_LB_THRESHOLD * avg_load;

  DEBUG_HIER_LB(
    "%d: calc_load_over: my_load=%f, avg_load=%f, threshold=%f\n",
    rank, my_load, avg_load, threshold
  );

  while (my_load > threshold && cur_item != obj_sample.end()) {
    if (cur_item->second.size() != 0) {
      auto const obj = cur_item->second.back();

      load_over[cur_item->first].push_back(obj);
      cur_item->second.pop_back();

      auto to_id = hier_objid_get_id(obj);
      auto const& obj_time_milli = my_stats->objData[to_id].wallTime * 1000;

      my_load -= obj_time_milli;

      DEBUG_HIER_LB(
        "%d: calc_load_over: my_load=%f, threshold=%f, adding unit, bin=%d\n",
        rank, my_load, threshold, cur_item->first
      );
    } else {
      cur_item++;
    }
  }

  for (auto i = 0; i < obj_sample.size(); i++) {
    if (obj_sample[i].size() == 0) {
      obj_sample.erase(obj_sample.find(i));
    }
  }
}

void
HierarchicalLB::lb_tree_msg(
  double const child_load, int const child, cont_hier_objid_t load,
  int child_size
) {
  DEBUG_HIER_LB(
    "%d: lb_tree_msg: child=%d, child_load=%f, child_size=%d, "
    "child_msgs=%d, children.size()=%ld, agg_node_size=%d, "
    "avg_load=%f, child_avg=%f, incoming load.size=%ld\n",
    rank, child, child_load, child_size, child_msgs+1, children.size(),
    agg_node_size + child_size, avg_load, child_load/child_size,
    load.size()
  );

  if (load.size() > 0) {
    for (auto& bin : load) {
      //auto load_iter = load.find(i);

      DEBUG_HIER_LB(
        "%d: \t lb_tree_msg: combining bins for bin=%d, size=%ld\n",
        rank, bin.first, bin.second.size()
      );

      if (bin.second.size() > 0) {
        // splice in the new list to accumulated work units that fall in a
        // common histrogram bin
        auto given_iter = given_objs.find(bin.first);

        if (given_iter == given_objs.end()) {
          // do the insertion here
          given_objs.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(bin.first),
            std::forward_as_tuple(cont_hier_bin_t{})
          );

          given_iter = given_objs.find(bin.first);

          CkAssert(
            given_iter != given_objs.end() &&
            "An insertion just took place so this must not fail"
          );
        }

        // add in the load that was just received
        total_child_load += bin.first * bin.second.size();

        given_iter->second.splice(
          given_iter->second.begin(), bin.second
        );
      }
    }
  }

  agg_node_size += child_size;

  auto child_iter = children.find(child);

  CkAssert(
    child_iter != children.end() && "Entry must exist in children map"
  );

  child_iter->second->node_size = child_size;
  child_iter->second->cur_load = child_load;
  child_iter->second->pe = child;

  total_child_load += child_load;

  child_msgs++;

  if (child_size > 0 && child_load != 0.0) {
    auto live_iter = live_children.find(child);
    if (live_iter == live_children.end()) {
      live_children.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(child),
        std::forward_as_tuple(child_iter->second)
      );
    }
  }

  CkAssert(
    child_msgs <= children.size() &&
    "Number of children must be greater or less than"
  );

  if (child_msgs == children.size()) {
    if (rank == HIER_LB_ROOT) {
      DEBUG_HIER_LB(
        "lb_tree_msg: %d: reached root!: total_load=%f, avg=%f\n",
        rank, total_child_load, total_child_load/agg_node_size
      );
      send_down_tree();
    } else {
      distribute_amoung_children();
    }
  }
}

ChildLoadInfo*
HierarchicalLB::find_min_child() {
  if (live_children.size() == 0) {
    return nullptr;
  }

  ChildLoadInfo* cur = live_children.begin()->second;
  //double cur_min = std::numerical_limits<double>::max();

  DEBUG_HIER_LB(
    "%d: find_min_child, cur.pe=%d, load=%f\n",
    rank, cur->pe, cur->cur_load
  );

  for (auto&& c : live_children) {
    auto const& load = c.second->cur_load / c.second->node_size;
    auto const& cur_load = cur->cur_load / cur->node_size;
    if (load <  cur_load || cur->node_size == 0) {
      cur = c.second;
    }
  }

  return cur;
}

void
HierarchicalLB::down_tree_msg(
  int const& from, cont_hier_objid_t excess_load, bool final_child
) {
  DEBUG_HIER_LB(
    "%d: down_tree_msg: from=%d, bottomParent=%d: load=%ld\n",
    rank, from, bottomParent, excess_load.size()
  );

  if (final_child) {
    // take the load
    taken_objs = std::move(excess_load);

    int total_taken_load = 0;
    for (auto&& item : taken_objs) {
      total_taken_load = item.first * item.second.size();

      DEBUG_HIER_LB(
        "%d: down_tree_msg: from=%d, taken_bin=%d, taken_bin_count=%ld, "
        "total_taken_load=%d\n",
        rank, from, item.first, item.second.size(), total_taken_load
      );
    }

    my_load += total_taken_load;

    DEBUG_HIER_LB(
      "%d: down_tree_msg: new load profile=%f, total_taken_load=%d, "
      "avg_load=%f\n",
      rank, my_load, total_taken_load, avg_load
    );
  } else {
    given_objs = std::move(excess_load);
    send_down_tree();
  }
}

void
HierarchicalLB::send_down_tree() {
  DEBUG_HIER_LB("%d: send_down_tree\n", rank);

  auto cIter = given_objs.rbegin();

  while (cIter != given_objs.rend()) {
    ChildLoadInfo* c = find_min_child();
    int const weight = c->node_size;
    double const threshold = avg_load * weight * HIER_LB_THRESHOLD;

    DEBUG_HIER_LB(
      "\t %d: distribute min child: c=%p, child=%d, cur_load=%f, "
      "weight=%d, avg_load=%f, threshold=%f\n",
      rank, c, c ? c->pe : -1, c ? c->cur_load : -1.0,
      weight, avg_load, threshold
    );

    if (c == nullptr || weight == 0) {
      break;
    } else {
      if (cIter->second.size() != 0) {
        DEBUG_HIER_LB(
          "\t distribute: %d, child=%d, cur_load=%f, time=%d\n",
          rank, c->pe, c->cur_load, cIter->first
       );

        // @todo agglomerate units into this bin together to increase efficiency
        auto task = cIter->second.back();
        c->recs[cIter->first].push_back(task);
        c->cur_load += cIter->first;
        // remove from list
        cIter->second.pop_back();
      } else {
        cIter++;
      }
    }
  }

  clear_obj_map(given_objs);

  for (auto& c : children) {
    // ??
    // if (c.first == CkMyPe()) {
    //   // take the load
    // } else {
    thisProxy[c.second->pe].down_tree_msg(
      CkMyPe(), c.second->recs, c.second->final_child
    );
    c.second->recs.clear();
    // }
  }
}

void
HierarchicalLB::clear_obj_map(cont_hier_objid_t& objs) {
  // @todo fix this crap

  // for (auto iter = objs.begin(); iter != objs.end(); ) {
  //   if (iter->second.size() == 0) {
  //     iter = objs.erase(iter);
  //   } else {
  //     ++iter;
  //   }
  // }

  std::vector<int> to_remove{};
  for (auto&& bin : objs) {
    if (bin.second.size() == 0) {
      to_remove.push_back(bin.first);
    }
  }

  for (auto&& r : to_remove) {
    auto giter = objs.find(r);
    CkAssert(
      giter != objs.end() && "Must exist"
    );
    objs.erase(giter);
  }
}

void
HierarchicalLB::distribute_amoung_children() {
  DEBUG_HIER_LB("distribute_amoung_children: %d, parent=%d\n", rank, parent);

  auto cIter = given_objs.rbegin();

  while (cIter != given_objs.rend()) {
    ChildLoadInfo* c = find_min_child();
    int const weight = c->node_size;
    double const threshold = avg_load * weight * HIER_LB_THRESHOLD;

    DEBUG_HIER_LB(
      "\t %d: distribute min child: c=%p, child=%d, cur_load=%f, "
      "weight=%d, avg_load=%f, threshold=%f\n",
      rank, c, c ? c->pe : -1, c ? c->cur_load : -1.0,
      weight, avg_load, threshold
    );

    if (c == nullptr || c->cur_load > threshold || weight == 0) {
      break;
    } else {
      if (cIter->second.size() != 0) {
        DEBUG_HIER_LB(
          "\t distribute: %d, child=%d, cur_load=%f, time=%d\n",
          rank, c->pe, c->cur_load, cIter->first
        );

        // @todo agglomerate units into this bin together to increase efficiency
        auto task = cIter->second.back();
        c->recs[cIter->first].push_back(task);
        c->cur_load += cIter->first;
        // remove from list
        cIter->second.pop_back();
      } else {
        cIter++;
      }
    }
  }

  clear_obj_map(given_objs);

  thisProxy[parent].lb_tree_msg(
    total_child_load, rank, given_objs, agg_node_size
  );

  given_objs.clear();
}

void
HierarchicalLB::avg_load_reduction(double x) {
  avg_load = x/CkNumPes();
  // Calculate the average load by considering the threshold for imbalance
  thr_avg = HIER_LB_THRESHOLD * avg_load;

  DEBUG_HIER_LB(
    "avg_load_reduction: %d: total=%f, avg_load=%f, thr_avg=%f\n",
    rank, x, avg_load, thr_avg
  );

  calc_load_over();

  thisProxy[bottomParent].lb_tree_msg(my_load, rank, load_over, 1);

  if (children.size() == 0) {
    cont_hier_objid_t empty_obj{};
    // send an empty msg up tree
    thisProxy[parent].lb_tree_msg(0.0, rank, empty_obj, agg_node_size);
  }

  // Start quiescence detection at PE 0.
  if (CkMyPe() == 0) {
    CkCallback cb(CkIndex_HierarchicalLB::done_hier(), thisProxy);
    CkStartQD(cb);
  }
}

void HierarchicalLB::done_hier() {
  DEBUG_HIER_LB(
    "%d: done_hier: given_objs.size()=%ld, taken_objs.size()=%ld\n",
    rank, given_objs.size(), taken_objs.size()
  );

  LoadBalance();
  theLbdb->nextLoadbalancer(seqno);
}

void
HierarchicalLB::transfer_objects(
  int const& to_pe, std::vector<int> const& lst
) {
  auto trans_iter = transfers.find(to_pe);

  CkAssert(
    trans_iter == transfers.end() &&
    "There must not be an entry"
  );

  transfers[to_pe] = lst;
  transfer_count += lst.size();
}

void
HierarchicalLB::finished_transfer_requests() {
  LBMigrateMsg* msg = new (transfer_count,CkNumPes(),CkNumPes(),0) LBMigrateMsg;
  msg->n_moves = transfer_count;

  int i = 0;
  for (auto&& t : transfers) {
    for (auto&& obj_id : t.second) {
      MigrateInfo mig;
      mig.obj = my_stats->objData[obj_id].handle;
      mig.from_pe = rank;
      mig.to_pe = t.first;

      msg->moves[i] = std::move(mig);
      i++;
    }
  }

  DEBUG_HIER_LB(
    "%d: finished_transfer_requests, i=%d\n", rank, i
  );

  ProcessMigrationDecision(msg);

  // reset data structures
  transfers.clear();
  transfer_count = 0;

  given_objs.clear();
  taken_objs.clear();
  obj_sample.clear();
  load_over.clear();
  my_load = avg_load = total_child_load = thr_avg = 0.0;
  agg_node_size = child_msgs = 0;
}

void HierarchicalLB::LoadBalance() {
  // if (lb_started) {
  //   return;
  // }
  // lb_started = true;

  std::map<int, std::vector<int>> transfer_list;

  for (auto&& bin : taken_objs) {
    for (auto&& obj_id : bin.second) {
      auto pe = hier_objid_get_pe(obj_id);
      auto id = hier_objid_get_id(obj_id);

      if (pe != rank) {
        migrates_expected++;

        DEBUG_HIER_LB(
          "%d: LoadBalance, obj_id=%ld, pe=%d, id=%d\n",
          rank, obj_id, pe, id
        );

        transfer_list[pe].push_back(id);
      }
    }
  }

  DEBUG_HIER_LB(
    "%d: LoadBalance, transfer_list=%ld\n",
    rank, transfer_list.size()
  );

  for (auto&& trans : transfer_list) {
    thisProxy[trans.first].transfer_objects(rank, trans.second);
  }

  // set callback on CB
  if (CkMyPe() == 0) {
    CkCallback cb(
      CkIndex_HierarchicalLB::finished_transfer_requests(), thisProxy
    );
    CkStartQD(cb);
  }
}

#include "HierarchicalLB.def.h"
