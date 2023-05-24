// Tests a simple 1D AMR that was written as a reproduction for issue #3660

#include "amr_1d_simple.h"

#include <charm++.h>
#include <ostream>
#include <vector>
#include <numeric>

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_DgElement dgElementProxy;

namespace {
static constexpr int initial_refinement_level = 0;

static constexpr int number_of_iterations = 3;
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
    dgElementProxy(element_ids[j]).insert(0);
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
  CkStartQD(CkCallback(CkIndex_Main::create_new_elements(), mainProxy));
}

// Creates new elements for elements that are split or joined
void Main::create_new_elements() {
  print_phase("create new elements");
  ++iteration;
  if (iteration == number_of_iterations)
    CkStartQD(CkCallback(CkIndex_Main::exit(), mainProxy));
  print_iteration(iteration);

  dgElementProxy.beginInserting();
  dgElementProxy.create_new_elements(iteration);
  dgElementProxy.doneInserting();
  CkStartQD(CkCallback(CkIndex_Main::adjust_domain(), mainProxy));
}

// Initialize new elements and update the neighbors of unrefined elements
void Main::adjust_domain() {
  print_phase("adjust domain");
  dgElementProxy.adjust_domain();
  CkStartQD(CkCallback(CkIndex_Main::delete_old_elements(), mainProxy));
}

// Creates new elements for elements that are split or joined
void Main::delete_old_elements() {
  print_phase("delete old elements");
  dgElementProxy.delete_old_elements();
  CkStartQD(CkCallback(CkIndex_Main::create_new_elements(), mainProxy));
}

// Cleanly ends the executable
void Main::exit() {
  print_phase("exit");
  CkExit();
}

// Constructor for initial elements.  All initial elements are on the
// same refinement level
DgElement::DgElement() {
  print_action("created", thisIndex);
  init_flag = false;
}

// Adjusts the domain based on the final AMR decisions of each element
void DgElement::adjust_domain() {
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

void DgElement::create_new_elements(int iteration) {

  flag_ = iteration % 2 ? 1 : -1;

  neighbor_flags_[neighbors_[0]] = flag_;
  neighbor_flags_[neighbors_[1]] = flag_;

  if (!init_flag)
    CkAbort("Initialize on parent or children wasn't called\n");

  if (flag_ == 1) {
    // Element wants to split, create the lower child
    print_action("adjusting domain (split)", thisIndex);
    thisProxy[2 * thisIndex].insert(1);
    thisProxy[2 * thisIndex + 1].insert(0);
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
void DgElement::evaluate_refinement_criteria(int iteration) {
  print_action("evaluating refinement criteria", thisIndex);
  flag_ = iteration % 2 ? 1 : -1; //random_flag(refinement_level(thisIndex));

  neighbor_flags_[neighbors_[0]] = flag_;
  neighbor_flags_[neighbors_[1]] = flag_;

  print_data("Evaluate criteria");
}

// Initialize the data held by a newly created child element.
void DgElement::initialize_child(const ElementId_t& nonsibling_neighbor_id) {
  print_action("initializing child", thisIndex);
  init_flag = true;
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
  init_flag = true;
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
  init_flag = true;
  print_action("initializing parent", thisIndex);
  neighbors_ = parent_neighbors;
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
