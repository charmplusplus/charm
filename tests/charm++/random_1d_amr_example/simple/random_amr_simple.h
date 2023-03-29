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

#include "random_amr_simple.decl.h"

class Main : public CBase_Main {
 public:
  int itercount;

  Main(CkArgMsg* msg);
  void iterate();
  void init_new();
  void delete_old();
};

class DgElement : public CBase_DgElement {
 public:
  DgElement();
  void iterate(int start);
  void split();
  void join();
  void init_new();
  void init_child(int itercount_);
  void init_parent(int itercount_);
  void delete_old();

 private:

  Flag_t flag_{-2};
  int itercount;
  int child_count;
};
