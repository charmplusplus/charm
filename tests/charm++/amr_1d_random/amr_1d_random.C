// This example is a simplified version of the algorithm being developed
// for adaptive mesh refinement for SpECTRE.  It does random refinement
// (splitting and joining elements) in 1D. 

// The array chare DgElement represent non-overlapping elements that cover the
// unit interval.  The id of an element is determined by its refinement level L
// and its index I.  There are 2^L possible elements for refinement level L,
// indexed left-to-right from [0, 2^L -1].  The unique id of an element is
// given by id = 2^L + I
//
// Level 0 |---------------------------------------------------------------|
//                id = 1 = 2^0 + 0    
// Level 1 |-------------------------------|-------------------------------|
//                id = 2 = 2^1 + 0               id = 3 = 2^1 + 1
// Level 2 |---------------|---------------|---------------|---------------|
//          id = 4 = 2^2+0  id = 5 = 2^2+1  id = 6 = 2^2+2  id = 7 = 2^2+3
// Level 3 |-------|-------|-------|-------|-------|-------|-------|-------|
//          id = 8  id = 9  id = 10 id = 11 id = 12 id = 13 id = 14 id = 15
// Level 4 |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
//           16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
//
// As the elements covering the interval are refined, we require that
// neighboring elements be within one refinement level of each other.
// (This is referred to as 2:1 balance.)  Splitting has a higher priority than
// joining, so an elements desired refinement level will be increased in order
// to maintain 2:1 balance.
//
// The interval is considered to be periodic, so that the element on the
// lower boundary is a neighbor of an element on the upper boundary.  Thus
// the element with id 8 will have a lower neighbor whose id is either 7, 15,
// or 31.
//
// Note that two elements can only join if they are siblings, i.e. they had
// a common parent (i.e. elements 9 and 10 cannot join).
//
// This example creates an empty array chare DgElement.
// Then it creates the 2^L elements on the initial_refinement_level
// Then it performs a given number_of_iterations
// 
// In each iteration (with quiessence detected between these phases):
//   - Each element is asked to ping its neighbors.
//
//   - Each element is asked to contribute its 1D volume to a reduction.
//     Each element also checks that it was pinged by two neighbors.
//     The main chare checks that the total volume contributed by the elements
//     is one.
//
//   - Each element is asked to randomly determine their AMR decision.
//     The element then communicates this decision to its neighbors.
//     Upon receiving a neighbor's decision, an element checks whether it
//     needs to adjust its decision in order to preserve 2:1 balance or
//     because it cannot join its sibling.  If a decision is updated, it
//     is communicated to the neighbors of the element.
//
//   - Each element is asked to execute its AMR decision.
//     - If it wants to split, it creates the lower child, which then creates
//       the upper child, which then calls the original element to communicate
//       its data to each child and then delete itself.  The children then
//       initialize themselves with the data provided.
//
//      - If it wants to join and is the lower child, it creates the parent.
//        (The upper child does nothing.)  The parent then calls the lower child
//        which calls the upper child, passing along data to eventually
//        contribute to the new parent.  The lower child then deletes itself.
//        The upper child adds its data, communicates all the data to the new
//        parent, and deletes itself.  The parent then initializes itself with
//        the provided data.
//
//      - If it neither want to join nor split, the element updates its data
//        (i.e. determines its new neighbors).

#include "amr_1d_random.h"

#include <charm++.h>
#include <ostream>
#include <vector>
#include <numeric>

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_DgElement dgElementProxy;

namespace {
static constexpr int initial_refinement_level = 0;

static constexpr int number_of_iterations = 10;
static constexpr int maximum_refinement_level = 1;
static const double do_something_fraction = 1.0;

static constexpr bool output_iteration = true;
static constexpr bool output_phase = true;
static constexpr bool output_action = true;
static constexpr bool output_data = false;

constexpr int two_to_the(int n) { return 1 << n; }

constexpr int initial_number_of_elements = two_to_the(initial_refinement_level);

int refinement_level(const ElementId_t id) {
  CkAssert(id > 0);
  return std::floor(std::log2(id));
}

int index(const ElementId_t id) {
  CkAssert(id > 0);
  return id % two_to_the(refinement_level(id));
}

ElementId_t id(const int refinement_level, const int index) {
  return two_to_the(refinement_level) + index;
}

double fraction_of_block_volume(const ElementId_t& element_id) {
  return 1.0 / two_to_the(refinement_level(element_id));
}

Flag_t random_flag(const int current_refinement_level) {
  static std::random_device r;
  static const auto seed = r();
  static std::mt19937 generator(seed);
  static std::uniform_real_distribution<> distribution(0.0, 1.0);

  const double join_fraction =
      current_refinement_level / static_cast<double>(maximum_refinement_level);

  const double random_number = distribution(generator);
  if (random_number > do_something_fraction) {
    return 0;
  }
  if (random_number < join_fraction * do_something_fraction) {
    return -1;
  }
  return 1;
}

bool are_siblings(const ElementId_t& id, const ElementId_t& neighbor_id) {
  CkAssert(id > 1);
  CkAssert(neighbor_id > 1);
  return id / 2 == neighbor_id / 2;
}

// An abutting nibling is the neighbor child of my sibling
bool neighbor_is_my_abutting_nibling(const ElementId_t& id,
                                     const ElementId_t& neighbor_id) {
  CkAssert(id > 1);
  CkAssert(neighbor_id > 1);
  if(neighbor_id < 4) {
    return false;
  }
  return neighbor_id / 4 == id / 2;
}

bool check_amr_decision(Flag_t& flag, const ElementId_t& id,
                        const ElementId_t& neighbor_id,
                        const Flag_t& neighbor_flag) {
  CkAssert(flag > -2 && flag < 2);
  CkAssert(neighbor_flag > -2 && neighbor_flag < 2);

  // Check if desired refinement level is two or three less than that of
  // neighbor.  If so, change flag so it is within one level.
  const int desired_level = refinement_level(id) + flag;
  CkAssert(desired_level > -1 && desired_level <= maximum_refinement_level);
  const int neighbor_desired_level =
      refinement_level(neighbor_id) + neighbor_flag;
  CkAssert(neighbor_desired_level > -1 &&
           neighbor_desired_level <= maximum_refinement_level);
  if (desired_level + 1 < neighbor_desired_level) {
    flag = neighbor_desired_level - 1 - refinement_level(id);
    return true;
  }

  // Next check if I want to join and the neighbor is either a sibling that
  // does not want to join or an abutting nibling.  If so, then change flag
  // to do nothing.
  if (flag == -1 && ((are_siblings(id, neighbor_id) && neighbor_flag != -1) ||
                      neighbor_is_my_abutting_nibling(id, neighbor_id))) {
    flag = 0;
    return true;
  }

  return false;
}

ElementId_t new_lower_neighbor_id(const ElementId_t& old_id,
                                  const Flag_t& old_flag) {
  if (old_flag == 1) {
    return 2 * old_id + 1;
  }
  if (old_flag == -1) {
    return old_id / 2;
  }
  return old_id;
}

ElementId_t new_upper_neighbor_id(const ElementId_t& old_id,
                                  const Flag_t& old_flag) {
  if (old_flag == 1) {
    return 2 * old_id;
  }
  if (old_flag == -1) {
    return old_id / 2;
  }
  return old_id;
}

void print_iteration(const int iteration) {
  if (output_iteration) {
    CkPrintf("\n\n------------\nIteration %i\n------------\n\n\n", iteration);
  }
}

void print_phase(const std::string& phase_name) {
  if (output_phase) {
    CkPrintf("Main is in phase %s\n", phase_name.c_str());
  }
}

void print_action(const std::string& action, const ElementId_t id) {
  if (output_action) {
    CkPrintf("Element (L%i,I%i) %s on (N%i, C%i)\n", refinement_level(id),
             index(id), action.c_str(), CkMyNode(), CkMyPe());
  }
}
}  // namespace

// Creates an empty array chare DgElement
Main::Main(CkArgMsg* msg) {
  delete msg;
  dgElementProxy = CProxy_DgElement::ckNew();
  const int number_of_procs = CkNumPes();
  int which_proc = 0;
  std::vector<ElementId_t> element_ids(initial_number_of_elements);
  std::iota(element_ids.begin(), element_ids.end(), initial_number_of_elements);
  for (size_t j = 0; j < element_ids.size(); ++j) {
    dgElementProxy(element_ids[j]).insert(which_proc);
    which_proc = (which_proc + 1 == number_of_procs ? 0 : which_proc + 1);
  }
  dgElementProxy.doneInserting();
  CkStartQD(CkCallback(CkIndex_Main::initialize(), mainProxy));
}

// Creates the 2^L elements on the initial_refinement_level distributed
// round-robin among the PEs
void Main::initialize() { 
  print_phase("initialize");
  dgElementProxy.initialize_initial_elements();
  CkStartQD(CkCallback(CkIndex_Main::check_neighbors(), mainProxy));
}

// Requests every element of the DgElement chare array to ping its neighboring
// elements (The interval is considered to be periodic, so the element on the
// lower boundary of the interval is a neighbor of the element on upper boundary
// of the interval)
void Main::check_neighbors() {
  print_phase("check neighbors");
  dgElementProxy.ping_neighbors();
  CkStartQD(CkCallback(CkIndex_Main::check_domain(), mainProxy));
}

// Requests every element to contribute the 1D volume of the interval that it
// covers.
void Main::check_domain() {
  print_phase("check");
  dgElementProxy.send_volume();

  if (iteration >= number_of_iterations) {
    CkStartQD(CkCallback(CkIndex_Main::exit(), mainProxy));
  } else {
    CkStartQD(CkCallback(CkIndex_Main::evaluate_amr_criteria(), mainProxy));
  }
}

// Requests every element to decide whether to split, join, or do no refinement
void Main::evaluate_amr_criteria() {
  ++iteration;
  print_iteration(iteration);
  print_phase("evaluate refinement criteria");
  dgElementProxy.evaluate_refinement_criteria();
  CkStartQD(CkCallback(CkIndex_Main::begin_inserting(), mainProxy));
}

void Main::begin_inserting() {
  print_phase("begin inserting");
  dgElementProxy.beginInserting();
  CkStartQD(CkCallback(CkIndex_Main::create_new_elements(), mainProxy));
}

// Creates new elements for elements that are split or joined
void Main::create_new_elements() {
  print_phase("create new elements");
  dgElementProxy.create_new_elements();
  CkStartQD(CkCallback(CkIndex_Main::done_inserting(), mainProxy));
}

void Main::done_inserting() {
  print_phase("done inserting");
  dgElementProxy.doneInserting();
  CkStartQD(CkCallback(CkIndex_Main::count_elements(), mainProxy));
}

void Main::count_elements() {
  print_phase("count elements");
  dgElementProxy.count_elements();
}

// Initialize new elements and update the neighbors of unrefined elements
void Main::adjust_domain(int num_elements) {
  print_phase("adjust domain");
  CkPrintf("Total elements = %i\n", num_elements);
  dgElementProxy.adjust_domain();
  CkStartQD(CkCallback(CkIndex_Main::delete_old_elements(), mainProxy));
}

// Creates new elements for elements that are split or joined
void Main::delete_old_elements() {
  print_phase("delete old elements");
  dgElementProxy.delete_old_elements();
  CkStartQD(CkCallback(CkIndex_Main::check_neighbors(), mainProxy));
}

// Cleanly ends the executable
void Main::exit() {
  print_phase("exit");
  CkExit();
}

// Callback executed after all elements have contributed their volume.
// The total volume should be 1.0
void Main::check_volume(const double volume) {
  print_phase("check volume");
  if (volume != 1.0) {
    CkAbort("Volume %f is not 1.0\n", volume);
  }
}

// Constructor for initial elements.  All initial elements are on the
// same refinement level
DgElement::DgElement() {
  print_action("created", thisIndex);
}

void DgElement::count_elements() {
  CkCallback cb(CkReductionTarget(Main, adjust_domain), mainProxy);;
  int result = 1;
  contribute(sizeof(int), &result, CkReduction::sum_int, cb);
}

// Adjusts the domain based on the final AMR decisions of each element
void DgElement::adjust_domain() {
  CkPrintf("Flag for %i is %i\n", thisIndex, flag_);

  if (flag_ == -2) {
    // this is a newly created element, do nothing
  } else if (flag_ == 1) {
    send_data_to_children();
  } else if (flag_ == -1) {
    // Element wants to join, if it is the lower child, create the parent
    print_action("adjusting domain (join)", thisIndex);
    if (thisIndex % 2 == 0) {
      collect_data_from_children({thisIndex + 1}, {{0, 0}});
    }
  } else {
    // Element is neither splitting nor joining.  Update the ids of its
    // neighbors and reset/clear the AMR flags
    print_action("adjusting domain (do nothing)", thisIndex);
    if(neighbor_flags_.count(neighbors_[0]) == 0 ||
       neighbor_flags_.count(neighbors_[1]) == 0) {
      CkPrintf("On Element %i, lower neighbor = %i, upper neighbor = %i \n"
	       "Neighbor flags = ", thisIndex, neighbors_[0], neighbors_[1]);
      for (const auto& neighbor_flag : neighbor_flags_) {
	CkPrintf("(%i, %i) ", neighbor_flag.first, neighbor_flag.second);
      }
    }
    neighbors_[0] =
        new_lower_neighbor_id(neighbors_[0], neighbor_flags_.at(neighbors_[0]));
    neighbors_[1] =
        new_upper_neighbor_id(neighbors_[1], neighbor_flags_.at(neighbors_[1]));
    flag_ = -2;
    neighbor_flags_.clear();
  }
}

// Collect data from children to send to their newly created parent after a
// join.  Then delete the child.
//
// sibling_ids_to_collect is the list of further children to collect data from
// parent_neighbors is the data being collected
void DgElement::collect_data_from_children(
    std::deque<ElementId_t> sibling_ids_to_collect,
    std::array<ElementId_t, 2> parent_neighbors) {
  print_action("collecting data from children", thisIndex);
  if (sibling_ids_to_collect.empty()) {
    // I am the upper child
    parent_neighbors[1] =
        new_upper_neighbor_id(neighbors_[1], neighbor_flags_.at(neighbors_[1]));
    CkPrintf("initialize_parent called\n");
    thisProxy[thisIndex / 2].initialize_parent(parent_neighbors);
  } else {
    // I am the lower child
    parent_neighbors[0] =
        new_lower_neighbor_id(neighbors_[0], neighbor_flags_.at(neighbors_[0]));
    const auto next_child_id = sibling_ids_to_collect.front();
    sibling_ids_to_collect.pop_front();
    thisProxy[next_child_id].collect_data_from_children(sibling_ids_to_collect,
                                                        parent_neighbors);
  }
}

void DgElement::create_new_elements() {
  if (flag_ == 1) {
    // Element wants to split, create the lower child
    print_action("adjusting domain (split)", thisIndex);
    thisProxy[2 * thisIndex].insert();
    thisProxy[2 * thisIndex + 1].insert();
  } else if (flag_ == -1) {
    // Element wants to join, if it is the lower child, create the parent
    print_action("adjusting domain (join)", thisIndex);
    if (thisIndex % 2 == 0) {
      thisProxy[thisIndex / 2].insert();
    }
  } else {
    // Element is neither splitting nor joining.
  }
}

void DgElement::delete_old_elements() {
  if (flag_ == 1 || flag_ == -1) {
    print_action("deleting", thisIndex);
    thisProxy[thisIndex].ckDestroy();
  }
}

// Evaluate the AMR criteria (for this example, randomly choose whether to
// join, split, or remain the same).  Then communicate these decisions to
// the neighbors of the element in case elements need to adjust their decisions
void DgElement::evaluate_refinement_criteria() {
  print_action("evaluating refinement criteria", thisIndex);
  flag_ = random_flag(refinement_level(thisIndex));

  // Actions can be executed in any order.  Check if I have already received
  // flags from my neighbors, and if so check if my decision needs to be
  // changed.
  if (!neighbor_flags_.empty()) {
    for (const auto& neighbor_id_flag : neighbor_flags_) {
      check_amr_decision(flag_, thisIndex, neighbor_id_flag.first,
                         neighbor_id_flag.second);
    }
  }

  thisProxy[neighbors_[0]].update_amr_decision(thisIndex, flag_);
  thisProxy[neighbors_[1]].update_amr_decision(thisIndex, flag_);
  print_data("Evaluate criteria");
}

// Initialize the data held by a newly created child element.
void DgElement::initialize_child(const ElementId_t& nonsibling_neighbor_id) {
  print_action("initializing child", thisIndex);
  if (thisIndex % 2 == 0) {
    // I am the lower child
    neighbors_[0] = nonsibling_neighbor_id;
    neighbors_[1] = thisIndex + 1;
  } else {
    // I am the upper child
    neighbors_[0] = thisIndex - 1;
    neighbors_[1] = nonsibling_neighbor_id;
  }
}

// Initialize the initial elements
void DgElement::initialize_initial_elements() {
  const int L = refinement_level(thisIndex);
  const int I = index(thisIndex);
  const int Imax = two_to_the(L) - 1;
  // The interval is considered to be periodic so that the element on the
  // lower boundary of the interval is a neighbor of the element on the
  // upper boundary of the interval
  neighbors_[0] = (I == 0 ? id(L, Imax) : thisIndex - 1);
  neighbors_[1] = (I == Imax ? id(L, 0) : thisIndex + 1);
}

// Initialize the data held by a newly created parent element.
void DgElement::initialize_parent(
    const std::array<ElementId_t, 2>& parent_neighbors) {
  print_action("initializing parent", thisIndex);
  neighbors_ = parent_neighbors;
}

// Receive a ping from a neighboring element.
//
// pinger is the id of the neighbor
// index is 0 for the lower neighbor, 1 for the upper neighbor
void DgElement::ping(const ElementId_t& pinger, const size_t index) {
  print_action("pinged", thisIndex);
  if (neighbors_[index] != pinger) {
    CkPrintf("On Element %i, Neighbors %zu = %i, pinger = %i\n", thisIndex,
	     index, neighbors_[index], pinger);
  }
  CkAssert(neighbors_[index] == pinger);
  // Set pings_received to 2 if this is the second ping, to index if this
  // is the first ping. A value of -1 indicates no previous pings.
  if (index == 0) {
    CkAssert(pings_received_ == -1 || pings_received_ == 1);
    ++pings_received_;
  } else {
    CkAssert(pings_received_ == -1 || pings_received_ == 0);
    pings_received_ += 2;
  }
}

// Ping each neighboring element with my id as well as if I am the
// lower (0) or upper (1) neighbor of that neighbor
void DgElement::ping_neighbors() {
  print_data("pinging neighbors");
  CkAssert(neighbors_[0] > 0 && neighbors_[1] > 0);
  thisProxy[neighbors_[0]].ping(thisIndex, 1);
  thisProxy[neighbors_[1]].ping(thisIndex, 0);
}

// After splitting, send data to all the newly created child elements of this
// element. Then delete the element.
void DgElement::send_data_to_children() {
  print_action("sending data to children", thisIndex);
  // Send data to lower child
  CkPrintf("%i, Neighbors = %i, %i\n", thisIndex, neighbors_[0], neighbors_[1]);
  thisProxy[2 * thisIndex].initialize_child(
      new_lower_neighbor_id(neighbors_[0], neighbor_flags_.at(neighbors_[0])));
  // Send data to upper child
  thisProxy[2 * thisIndex + 1].initialize_child(
      new_upper_neighbor_id(neighbors_[1], neighbor_flags_.at(neighbors_[1])));
}

// Contribute the volume of the element to a reduction.  Also check that
// the element received two pings during the check of neighbors and reset
// ping counter.
void DgElement::send_volume() {
  print_data("sending volume");
  CkAssert(pings_received_ == 2);
  pings_received_ = -1;
  const double volume = fraction_of_block_volume(thisIndex);
  contribute(sizeof(double), &volume, CkReduction::sum_double,
             CkCallback(CkReductionTarget(Main, check_volume), mainProxy));
}

// Possibly update the AMR decision of this element based on the decision of
// one of its neighbors.  There are two reasons an element will update its
// decision:
//   - The desired refinement level of the neighbor is two or more levels
//     higher than the desired refinement level of this element.  In this
//     case change the decision so that the desired level of this element
//     is one lower than the desired refinement level of the neighbor
//   - It want to join, but its sibling either does not want to join, or
//     does not exist (i.e. it is currently refined).  In this case the
//     decision is changed to do nothing
//
// If the decision is updated, the new decison is sent to the neighbors.
//
// neighbor_id is the id of the neighbor
// neighbor_flag is its AMR decision
void DgElement::update_amr_decision(const ElementId_t& neighbor_id,
                                    const Flag_t& neighbor_flag) {
  print_action("updating AMR decision", thisIndex);

  // Actions can be executed in any order.  Therefore we need to check:
  // - If we received flags from a neighbor multiple times, but not in the
  //   order they were sent.  Neighbor flags should only be sent again if
  //   they have changed to a higher priority (i.e. higher integral value of
  //   the flag).
  if (1 == neighbor_flags_.count(neighbor_id)) {
    const auto previously_received_flag = neighbor_flags_.at(neighbor_id);
    if (previously_received_flag >= neighbor_flag) {
      print_data("Update decision (old decision)");
      return;
    }
  }

  neighbor_flags_[neighbor_id] = neighbor_flag;

  // Actions can be executed in any order.  Therefore we need to check:
  // - If we have evaluated our own AMR decision.  If not, return.
  if (flag_ == -2) {
    print_data("Update decision (waiting for evaluate");
    return;
  }

  const bool my_amr_decision_changed =
      check_amr_decision(flag_, thisIndex, neighbor_id, neighbor_flag);

  if (my_amr_decision_changed) {
    thisProxy[neighbors_[0]].update_amr_decision(thisIndex, flag_);
    thisProxy[neighbors_[1]].update_amr_decision(thisIndex, flag_);
    print_data("Update decision (changed)");
  } else {
    print_data("Update decision (not changed)");
  }
}

void DgElement::print_data(const std::string& action_name) {
  if (output_data) {
    CkPrintf(
        "Element %i is executing %s.\nNeighbors: (%i, %i)\nFlag: %i\nNeighbor "
        "flags: (%i,%i)\n",
        thisIndex, action_name.c_str(), neighbors_[0], neighbors_[1], flag_,
        neighbor_flags_.count(neighbors_[0]) == 1
            ? neighbor_flags_.at(neighbors_[0])
            : -2,
        neighbor_flags_.count(neighbors_[1]) == 1
            ? neighbor_flags_.at(neighbors_[1])
            : -2);
  }
}

#include "random_amr.def.h"
