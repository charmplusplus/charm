#include "random_amr_simple.h"

#include <charm++.h>
#include <ostream>
#include <vector>

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_DgElement dgElementProxy;

constexpr int initial_number_of_elements = 10;

Main::Main(CkArgMsg* msg) {
  delete msg;
  dgElementProxy = CProxy_DgElement::ckNew();
  const int number_of_procs = CkNumPes();
  int which_proc = 0;
  for (size_t j = 0; j < initial_number_of_elements; ++j) {
    dgElementProxy(j).insert(which_proc);
    which_proc = (which_proc + 1 == number_of_procs ? 0 : which_proc + 1);
  }
  dgElementProxy.doneInserting();
  dgElementProxy.iterate(initial_number_of_elements);
}

DgElement::DgElement() {
  itercount = 0;
  flag_ = 0;
  child_count = 0;
}

void DgElement::iterate(int nelements) {
  if (thisIndex == 0 || thisIndex == 100)
    CkPrintf("Iteration %i, number of elements %i\n", itercount, nelements);

  if (itercount == 10)
    CkExit();

  if (itercount++ % 2 == 0) {
    split();
  } else {
    join();
  }
}

void DgElement::split() {
  flag_ = 1;
  thisProxy[100 + 2 * thisIndex].insert();
  thisProxy[100 + 2 * thisIndex + 1].insert();
  
  thisProxy[100 + 2 * thisIndex].init_child(itercount);
  thisProxy[100 + 2 * thisIndex + 1].init_child(itercount);

  CkCallback cb(CkReductionTarget(DgElement, delete_old), dgElementProxy);;
  int result = 1;
  contribute(sizeof(int), &result, CkReduction::sum_int, cb);
}

void DgElement::init_child(int itercount_) {
  itercount = itercount_;
  CkCallback cb(CkReductionTarget(DgElement, delete_old), dgElementProxy);;
  int result = 1;
  contribute(sizeof(int), &result, CkReduction::sum_int, cb);
}

void DgElement::join() {
  flag_ = -1;
  if (thisIndex % 2 == 0)
    thisProxy[(thisIndex - 100) / 2].insert();

  thisProxy[(thisIndex - 100) / 2].init_parent(itercount);
  
  CkCallback cb(CkReductionTarget(DgElement, delete_old), dgElementProxy);;
  int result = 1;
  contribute(sizeof(int), &result, CkReduction::sum_int, cb);
}

void DgElement::init_parent(int itercount_) {
  itercount = itercount_;
  if (child_count++ == 2) {
    child_count = 0;
    CkCallback cb(CkReductionTarget(DgElement, delete_old), dgElementProxy);;
    int result = 1;
    contribute(sizeof(int), &result, CkReduction::sum_int, cb);
  }
}

void DgElement::delete_old(int nelements) {
  //if (thisIndex == 0 || thisIndex == 100)
  //  CkPrintf("Iteration %i before delete, number of elements %i\n", itercount, nelements);
  if (flag_ == 1 || flag_ == -1) {
    thisProxy[thisIndex].ckDestroy();
  } else {
    CkCallback cb(CkReductionTarget(DgElement, iterate), dgElementProxy);;
    int result = 1;
    contribute(sizeof(int), &result, CkReduction::sum_int, cb);  
  }
}


#include "random_amr_simple.def.h"
