// This test tests dynamic insertion and deletion of array elements meant to
// be used in an AMR code.

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

// This test creates an empty array chare DgElement.
// Then it creates the 2^L elements on the initial_refinement_level
// Then it performs a given number_of_iterations
// 
// In each iteration:
//   - It first checks the volume covered by the elements,
//     by performing a reduction over all elements.  This volume should be 1.
//
//   - Next an unordered_map is built on the main chare by having all elements
//     send their PE to main
//
//   - Next, the main chare loops over the unordered_map and pings each element.
//     When an element is pinged, it pings the main chare to remove the element
//     from the unordered_map
//
//   - Next new elements are created.  On odd iterations, each element creates
//     its two children.  On even iterations, the lower child creates its
//     parent
//
//   - Finally the old elements are deleted.   Thus the AMR grid is cycling
//     back and forth between two refinement levels.

#include "dynamic_insertion_deletion.h"
#include <numeric>

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_DgElement dgElementProxy;

namespace {
static constexpr bool print_iteration = true;
static constexpr bool print_phase = true;
static constexpr bool print_action = true;
static constexpr bool print_ping = true;

static constexpr size_t initial_refinement_level = 2;

static constexpr size_t number_of_iterations = 40;

constexpr size_t two_to_the(size_t n) { return 1 << n; }

constexpr size_t initial_number_of_elements =
    two_to_the(initial_refinement_level);

size_t refinement_level(const ElementId_t id) {
  return std::floor(std::log2(id));
}

size_t index(const ElementId_t id) {
  return static_cast<size_t>(id) % two_to_the(refinement_level(id));
}

double fraction_of_block_volume(const ElementId_t& element_id) {
  return 1.0 / two_to_the(refinement_level(element_id));
}

}  // namespace

Main::Main(CkArgMsg* msg) {
  delete msg;
  dgElementProxy = CProxy_DgElement::ckNew();
  dgElementProxy.doneInserting();
  CkStartQD(CkCallback(CkIndex_Main::initialize(), mainProxy));
}

void Main::initialize() {
  if (print_phase) {
    CkPrintf("Main is in phase initialize\n");
  }
  const int number_of_procs = CkNumPes();
  int which_proc = 0;
  std::vector<ElementId_t> element_ids(initial_number_of_elements);
  std::iota(element_ids.begin(), element_ids.end(), initial_number_of_elements);
  for (size_t j = 0; j < element_ids.size(); ++j) {
    dgElementProxy(element_ids[j]).insert(iteration, which_proc);
    which_proc = (which_proc + 1 == number_of_procs ? 0 : which_proc + 1);
  }
  CkStartQD(CkCallback(CkIndex_Main::check_domain(), mainProxy));
}

void Main::check_domain() {
  ++iteration;
  if (print_iteration) {
    CkPrintf("\n\n------------\nIteration %zu\n------------\n\n\n",
                     iteration);
  }
  if (print_phase) {
    CkPrintf("Main is in phase check\n");
  }
  dgElementProxy.send_volume(iteration);
  
  if (iteration > number_of_iterations) {
    if(possible_hangs.size() > 1) {
      CkPrintf("\n\n\n\nPossible hang didn't hang!\n\n\n\n\n\n");
    }
    CkStartQD(CkCallback(CkIndex_Main::exit(), mainProxy));
  } else {
    CkStartQD(CkCallback(CkIndex_Main::build_proc_map(), mainProxy));
  }
}

void Main::build_proc_map() {
  if (print_phase) {
    CkPrintf("Main is in phase build proc map\n");
  }
  dgElementProxy.send_proc_to_main(iteration);
  CkStartQD(CkCallback(CkIndex_Main::ping_elements(), mainProxy));
}

void Main::ping_elements() {
  if (print_phase) {
    CkPrintf("Main is in phase ping\n");
  }
  if (print_ping) {
    CkPrintf("Potential hangs: ");
    for (const auto& id_proc : possible_hangs) {
      const auto& id = id_proc.first;
      const auto& proc = id_proc.second;
      if(0 != proc) {
	CkPrintf("(%i,%i) ", id, proc);
      }
    }
    CkPrintf("\n");
  }
  for (const auto& id_proc : proc_map) {
    const auto& id = id_proc.first;
    const auto& proc = id_proc.second;
    dgElementProxy[id].receive_ping(iteration);
  }
  CkStartQD(CkCallback(CkIndex_Main::create_new_elements(), mainProxy));
}

void Main::create_new_elements() {
  if (print_phase) {
    CkPrintf("Main is in phase create\n");
  }
  dgElementProxy.create_new_elements(iteration);
  CkStartQD(CkCallback(CkIndex_Main::delete_old_elements(), mainProxy));
}

void Main::delete_old_elements() {
  if (print_phase) {
    CkPrintf("Main is in phase delete\n");
  }
  dgElementProxy.delete_old_elements(iteration);
  CkStartQD(CkCallback(CkIndex_Main::check_domain(), mainProxy));
}

void Main::exit() {
  if (print_phase) {
    CkPrintf("Main is in phase exit\n");
  }
  CkExit();
}

void Main::check_volume(const double volume) {
  CkPrintf("Volume = %f at iteration %zu\n", volume, iteration);
  if (volume != 1.0) {
    CkAbort("Volume %f is not 1.0\n", volume);
  }
}

void Main::add_proc_to_map(const ElementId_t& id, const int proc) {
  if(iteration == 1) {
    initial_proc_map[id] = proc;
  } else {
    if (refinement_level(id) == initial_refinement_level &&
	proc != initial_proc_map[id]) {
      possible_hangs[id] = proc;
    }
  }
  proc_map[id] = proc;
}

void Main::remove_proc_from_map(const ElementId_t& id, const int proc) {
  CkAssert(proc == proc_map.at(id));
  proc_map.erase(id);
}

DgElement::DgElement(const size_t iteration)
    : iteration_at_creation(iteration) {
  if (print_action) {
    CkPrintf("Created element %i on (N%i, C%i) on iteration %zu\n",
	     thisIndex, CkMyNode(), CkMyPe(), iteration);
  }
}

void DgElement::create_new_elements(const size_t iteration) {
  if (print_action) {
    CkPrintf("Creating new elements %i on (N%i, C%i) on iteration %zu\n",
	     thisIndex, CkMyNode(), CkMyPe(), iteration);
  }
  if (iteration % 2 == 1) {
    thisProxy(2 * thisIndex).insert(iteration);
    thisProxy(2 * thisIndex + 1).insert(iteration);
  } else {
    if (thisIndex % 2 == 0) {
      thisProxy(thisIndex / 2).insert(iteration);
    }
  }
}

void DgElement::delete_old_elements(const size_t iteration) {
  if (print_action) {
    CkPrintf("Adjusting element %i on (N%i, C%i) on iteration %zu\n", thisIndex,
                     CkMyNode(), CkMyPe(), iteration);
  }
  const auto lev = refinement_level(thisIndex);
  if (iteration % 2 == 1 && lev == initial_refinement_level) {
    thisProxy[thisIndex].ckDestroy();
  } else if (iteration % 2 == 0 && lev == initial_refinement_level + 1) {
    thisProxy[thisIndex].ckDestroy();
  }
}

void DgElement::send_volume(const size_t iteration) {
  const double volume = fraction_of_block_volume(thisIndex);
  if (print_action) {
    CkPrintf("Sending volume %f from element %i on (N%i, C%i) on iter %zu\n",
	     volume, thisIndex, CkMyNode(), CkMyPe(), iteration);
  }
  contribute(sizeof(double), &volume, CkReduction::sum_double,
             CkCallback(CkReductionTarget(Main, check_volume), mainProxy));
}

void DgElement::send_proc_to_main(const size_t iteration) {
  if (print_ping) {
    CkPrintf("Element %i sending proc %i to main on (N%i, C%i) on iter %zu\n",
	     thisIndex, CkMyPe(), CkMyNode(), CkMyPe(), iteration);
  }
  mainProxy.add_proc_to_map(thisIndex, CkMyPe());
}

void DgElement::receive_ping(const size_t iteration) {
  if (print_ping) {
    CkPrintf("Pinged element %i on (N%i, C%i) on iteration %zu\n", thisIndex,
	     CkMyNode(), CkMyPe(), iteration);
  }
  mainProxy.remove_proc_from_map(thisIndex, CkMyPe());
}

#include "dynamic_insertion_deletion.def.h"
