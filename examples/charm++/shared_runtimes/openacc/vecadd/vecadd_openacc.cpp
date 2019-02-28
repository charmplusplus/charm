#include "vecadd.h"
#include <cstdlib>

void vecadd(const uint64_t n) {
  double* restrict a = (double*)malloc(n * sizeof(double));
  double* restrict b = (double*)malloc(n * sizeof(double));

#pragma acc data create(a[0:n], b[0:n])
  {
    // Initialize vectors
#pragma acc parallel loop
    for (uint64_t i = 0; i < n; i++) {
      a[i] = 1.0;
      b[i] = 1.0;
    }

#pragma acc parallel loop
    for (uint64_t i = 0; i < n; i++) {
      a[i] += b[i];
    }
  }

  free(a);
  free(b);
}
