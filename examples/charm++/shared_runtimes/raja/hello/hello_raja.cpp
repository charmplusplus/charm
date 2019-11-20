#include "hello.h"
#include "RAJA/RAJA.hpp"

void hello(const uint64_t n, int process) {
  RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0, n), [=] (int i) {
    printf("[Process %d] Hello from i = %d\n", process, i);
  });
}
