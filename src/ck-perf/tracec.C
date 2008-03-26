#include "charm++.h"
#include "trace-common.h"
#include "trace.h"
#include "tracec.h"

extern "C" {

  void traceMalloc_c(void *where, int size, void **stack, int stackSize) {
    _TRACE_MALLOC(where, size, stack, stackSize);
  }

  void traceFree_c(void *where, int size) {
    _TRACE_FREE(where, size);
  }

}
