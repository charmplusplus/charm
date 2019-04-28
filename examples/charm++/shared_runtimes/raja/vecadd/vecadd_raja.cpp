#include "vecadd.h"
#include "RAJA/RAJA.hpp"
#include <cstdio>
#include <iostream>
#include <typeinfo>
#include <chrono>

#define CORRECT_VALUE 3.0

void vecadd(const uint64_t n, int process, bool use_gpu) {
  double* h_a;
  double* h_b;
  double* d_a;
  double* d_b;
  if (use_gpu) {
    // Vector addition using CUDA
    cudaErrchk(cudaMallocHost((void**)&h_a, n * sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&d_a, n * sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&d_b, n * sizeof(double)));

    auto start = std::chrono::system_clock::now();
    RAJA::forall<RAJA::cuda_exec<256>>(RAJA::RangeSegment(0, n),
      [=] RAJA_DEVICE (int i) {
      d_a[i] = 1.0;
      d_b[i] = 2.0;
    });
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "[Process " << process << "] Vector initialization time on device (CUDA): " <<
      elapsed.count() << std::endl;

    start = std::chrono::system_clock::now();
    RAJA::forall<RAJA::cuda_exec<256>>(RAJA::RangeSegment(0, n),
      [=] RAJA_DEVICE (int i) {
      d_a[i] += d_b[i];
    });
    end = std::chrono::system_clock::now();
    elapsed = end - start;
    std::cout << "[Process " << process << "] Vector addition time on device (CUDA): " <<
      elapsed.count() << std::endl;

    start = std::chrono::system_clock::now();
    cudaErrchk(cudaMemcpy(h_a, d_a, n * sizeof(double), cudaMemcpyDeviceToHost));
    end = std::chrono::system_clock::now();
    elapsed = end - start;
    std::cout << "[Process " << process << "] Time for device -> host data movement: " <<
      elapsed.count() << std::endl;
  }
  else {
    // Vector addition using OpenMP
    h_a = (double*)malloc(n * sizeof(double));
    h_b = (double*)malloc(n * sizeof(double));

    auto start = std::chrono::system_clock::now();
    RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0, n), [=] (int i) {
      h_a[i] = 1.0;
      h_b[i] = 2.0;
    });
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "[Process " << process << "] Vector initialization time on host (OpenMP): " <<
      elapsed.count() << std::endl;

    start = std::chrono::system_clock::now();
    RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0, n), [=] (int i) {
      h_a[i] += h_b[i];
    });
    end = std::chrono::system_clock::now();
    elapsed = end - start;
    std::cout << "[Process " << process << "] Time on host (OpenMP): " <<
      elapsed.count() << std::endl;
  }

  // Validate last element of the vector
  double last_elem = h_a[n-1];
  if (abs(last_elem - CORRECT_VALUE) < 0.000001) {
    std::cout << "[Process " << process << "] Last element validated" << std::endl;
  }
  else {
    std::cout << "[Process " << process << "] Last element NOT validated: it is " <<
      last_elem << ", but should be " << CORRECT_VALUE << std::endl;
  }

  // Free allocated memory
  if (use_gpu) {
    cudaErrchk(cudaFreeHost(h_a));
    cudaErrchk(cudaFree(d_a));
    cudaErrchk(cudaFree(d_b));
  }
  else {
    free(h_a);
    free(h_b);
  }
}
