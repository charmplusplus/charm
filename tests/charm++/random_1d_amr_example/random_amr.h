#pragma once

#include <array>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>

// Chare array index for an element (0 is reserved for unknown)
using ElementId_t = int;
// A flag denoting the current AMR decision of an element
// -2 = NoDecision; -1 = Join; 0 = DoNothing; 1 = Split
using Flag_t = int;
using NeighborFlags_t = std::unordered_map<ElementId_t, Flag_t>;

#include "random_amr.decl.h"

class Main : public CBase_Main {
 private:
  int iteration{0};

 public:
  Main(CkArgMsg* msg);
  void initialize();
  void check_neighbors();
  void check_domain();
  void evaluate_amr_criteria();
  void begin_inserting();
  void create_new_elements();
  void done_inserting();
  void count_elements();
  void adjust_domain(int num_elements);
  void delete_old_elements();
  void exit();
  void check_volume(const double volume);
};

class DgElement : public CBase_DgElement {
 public:
  DgElement();
  void count_elements();
  void adjust_domain();
  void collect_data_from_children(
      std::deque<ElementId_t> sibling_ids_to_collect,
      std::array<ElementId_t, 2> parent_neighbors);
  void create_new_elements();
  void delete_old_elements();
  void evaluate_refinement_criteria();
  void initialize_child(const ElementId_t& nonsibling_neighbor_id);
  void initialize_initial_elements();
  void initialize_parent(const std::array<ElementId_t, 2>& parent_neighbors);
  void ping(const ElementId_t& pinger, const size_t index);
  void ping_neighbors();
  void send_data_to_children();
  void send_volume();
  void update_amr_decision(const ElementId_t& neighbor_id,
                           const Flag_t& neighbor_flag);

 private:
  void print_data(const std::string& action_name);
  // Current AMR decision of element (see Flag_t above for meaning of values)
  Flag_t flag_{-2};
  // The ids of the lower and upper neighbors
  std::array<ElementId_t, 2> neighbors_{{0, 0}};
  // The current AMR decision of neighboring elements
  NeighborFlags_t neighbor_flags_{};
  // The number of pings received from neighbors during the check of neighbors
  // The value is set to -1 prior to the check; 0 (lower neighbor) or 1 (upper
  // neighbor) after the first ping; and 2 after both neighbors have pinged
  int pings_received_{-1};
};
