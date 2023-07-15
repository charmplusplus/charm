// This test creates an empty array chare DgElement.
//
// Next it inserts an element on each PE.
//
// Then it performs a number of iterations.
//
// In each iteration, all elements are pinged which triggers a reduction
// checking that the sum of contributed ints is equal to the expected value.
//
// During some iterations (controlled by strides), CkNumPes() new elements are
// inserted into the array chare in one of three ways:
//  - the main chare inserts the new element
//  - a specific array element inserts the new element
//  - a specific member of a group chare inserts the new element
//
// Each array element stores the iteration on which it is created
// The id of a newly created array element is iteration*CkNumPes() + i
// where 0 <= i < CkNumPes()
//
// During a reduction each element contributes thisIndex*iteration

#include "dynamic_insertion.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_DgGroup dgGroupProxy;
/*readonly*/ CProxy_DgElement dgElementProxy;

namespace {
static constexpr bool print_phase = true;
static constexpr bool print_action = true;
static constexpr int main_creation_stride = 3;
static constexpr int array_creation_stride = 5;
static constexpr int group_creation_stride = 7;
static constexpr int number_of_iterations = 10;
}  // namespace

Main::Main(CkArgMsg* msg) {
  delete msg;
  dgGroupProxy = CProxy_DgGroup::ckNew();
  dgElementProxy = CProxy_DgElement::ckNew();
  dgElementProxy.doneInserting();
  CkStartQD(CkCallback(CkIndex_Main::initialize(), mainProxy));
}

void Main::initialize() {
  if (print_phase) {
    CkPrintf("Main is in phase initialize during iteration %i\n", iteration);
  }
  const int number_of_procs = CkNumPes();
  for (int i = 0; i < number_of_procs; ++i) {
    dgElementProxy(i).insert(iteration, i);
    sum_of_indices += i;
  }
  CkStartQD(CkCallback(CkIndex_Main::end_array_insertion(), mainProxy));
}

void Main::ping_elements() {
  ++iteration;
  if (print_phase) {
    CkPrintf("Main is in phase ping during iteration %i\n", iteration);
  }
  dgElementProxy.ping(iteration);
  CkStartQD(CkCallback(CkIndex_Main::begin_array_insertion(), mainProxy));
}

void Main::begin_array_insertion() {
  dgGroupProxy.begin_array_insertion();
  CkStartQD(CkCallback(CkIndex_Main::create_new_elements(), mainProxy));
}

void Main::create_new_elements() {
  if (print_phase) {
    CkPrintf("Main is in phase create during iteration %i\n", iteration);
  }
  if (iteration % main_creation_stride == 0 ||
      iteration % group_creation_stride == 0 ||
      iteration % array_creation_stride == 0) {
    const int number_of_procs = CkNumPes();
    const int offset = iteration*number_of_procs;
    for (int i = 0; i < number_of_procs; ++i) {
      if (iteration % array_creation_stride == 0) {
	// create a new element on a specific existing element
        dgElementProxy(i).create_new_element(iteration, offset + i);
      } else if (iteration % main_creation_stride == 0) {
	// create a new element directly on main chare
        dgElementProxy(offset + i).insert(iteration, i);
      } else {
	// create a new element from a specific group member
	dgGroupProxy[i].create_new_element(iteration, offset + i);
      }
      sum_of_indices += offset + i;
    }
  }
  CkStartQD(CkCallback(CkIndex_Main::end_array_insertion(), mainProxy));
}

void Main::end_array_insertion() {
  dgGroupProxy.end_array_insertion();
  if (iteration >= number_of_iterations) {
    CkStartQD(CkCallback(CkIndex_Main::exit(), mainProxy));
  } else {
    CkStartQD(CkCallback(CkIndex_Main::ping_elements(), mainProxy));
  }
}

void Main::exit() {
  if (print_phase) {
    CkPrintf("Main is in phase exit during iteration %i\n", iteration);
  }
  CkExit();
}

void Main::check_sum(const int sum) {
  CkPrintf("Main is checking sum %i during iteration %i is %i\n", sum,
           iteration, iteration * sum_of_indices);
  CkAssert(sum == iteration*sum_of_indices);
}

DgGroup::DgGroup() {}

void DgGroup::begin_array_insertion() {
  dgElementProxy.beginInserting();
}

void DgGroup::end_array_insertion() {
  dgElementProxy.doneInserting();
}

void DgGroup::create_new_element(const int iteration, const int new_id) {
  dgElementProxy(new_id).insert(iteration);
}

DgElement::DgElement(const int iteration) : iteration_at_creation(iteration) {
  if (print_action) {
    CkPrintf("Created element %i on (N%i, C%i) during iteration %i\n",
             thisIndex, CkMyNode(), CkMyPe(), iteration);
  }
}

void DgElement::ping(const int iteration) {
  if (print_action) {
    CkPrintf("Pinged element %i on (N%i, C%i) during iteration %i\n", thisIndex,
             CkMyNode(), CkMyPe(), iteration);
  }
  CkAssert(iteration > iteration_at_creation);
  int value = thisIndex*iteration;
  contribute(sizeof(int), &value, CkReduction::sum_int,
             CkCallback(CkReductionTarget(Main, check_sum), mainProxy));
}

void DgElement::create_new_element(const int iteration, const int new_id) {
  thisProxy(new_id).insert(iteration, CkMyPe());
}

#include "dynamic_insertion.def.h"
