#include "random_amr_simple.h"

#include <charm++.h>
#include <ostream>
#include <vector>

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_DgElement dgElementProxy;

constexpr int initial_number_of_elements = 1;

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
  itercount = 0;
  iterate();
}

void Main::iterate() {
  dgElementProxy.beginInserting();
  dgElementProxy.iterate(itercount++ == 0);
  dgElementProxy.doneInserting();
  CkStartQD(CkCallback(CkIndex_Main::init_new(), mainProxy));
}

void Main::init_new() {
  dgElementProxy.init_new();
  CkStartQD(CkCallback(CkIndex_Main::delete_old(), mainProxy));
}

void Main::delete_old() {
  dgElementProxy.delete_old();
  CkStartQD(CkCallback(CkIndex_Main::iterate(), mainProxy));
}

DgElement::DgElement() {
  itercount = -1;
  flag_ = 0;
  child_count = 0;
}

void DgElement::iterate(int start) {
  if (start)
    itercount++;

  CkAssert(itercount >= 0);

  if (thisIndex == 0 || thisIndex == 100)
    CkPrintf("Iteration %i\n", itercount);

  if (itercount == 10)
    CkExit();

  if (itercount++ % 2 == 0) {
    split();
  } else {
    join();
  }

}

void DgElement::init_new() {
  if (flag_ == 1) {
    thisProxy[100 + 2 * thisIndex].init_child(itercount);
    thisProxy[100 + 2 * thisIndex + 1].init_child(itercount);
  } else if (flag_ == -1) {
    thisProxy[(thisIndex - 100) / 2].init_parent(itercount);
  }
}

void DgElement::split() {
  flag_ = 1;
  thisProxy[100 + 2 * thisIndex].insert();
  thisProxy[100 + 2 * thisIndex + 1].insert();
}

void DgElement::init_child(int itercount_) {
  itercount = itercount_;
}

void DgElement::join() {
  flag_ = -1;
  if (thisIndex % 2 == 0)
    thisProxy[(thisIndex - 100) / 2].insert();
}

void DgElement::init_parent(int itercount_) {
  itercount = itercount_;
  if (child_count++ == 2) {
    child_count = 0;
  }
}

void DgElement::delete_old() {
  if (flag_ == 1 || flag_ == -1) {
    thisProxy[thisIndex].ckDestroy();
  }
}


#include "random_amr_simple.def.h"
