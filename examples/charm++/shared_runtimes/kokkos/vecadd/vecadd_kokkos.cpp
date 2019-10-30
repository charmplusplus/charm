#include "vecadd.h"
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>
#include <typeinfo>
#include <impl/Kokkos_Timer.hpp>

#define CORRECT_VALUE 3.0

// Column-major layout on GPU for coalesced accesses
typedef Kokkos::View<double*,Kokkos::LayoutLeft,Kokkos::CudaSpace> CudaView;
typedef Kokkos::View<double*,Kokkos::LayoutRight,Kokkos::CudaHostPinnedSpace> HostView;

// Functors
template <typename ViewType>
struct Fill {
  double value;
  ViewType a;

  Fill(const double& val, const ViewType& d_a) : value(val), a(d_a) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    a(i) = value;
  }
};

template <typename ViewType>
struct Compute {
  ViewType a, b;

  Compute(const ViewType& d_a, const ViewType& d_b) : a(d_a), b(d_b) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    a(i) += b(i);
  }
};

void kokkosInit() {
  Kokkos::initialize();
}

void kokkosInit(int device_id) {
  Kokkos::InitArguments args;
  args.device_id = device_id;
  Kokkos::initialize(args);
}

void kokkosFinalize() {
  Kokkos::finalize();
}

void vecadd(const uint64_t n, int process, bool use_gpu) {
#ifdef DEBUG
  std::cout << "[Process " << process << "] " << "Default execution space: " <<
    typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
  std::cout << "[Process " << process << "] " << "Default host execution space: " <<
    typeid(Kokkos::DefaultHostExecutionSpace).name() << std::endl;
#endif

  HostView h_a("Host A", n); // Used for validation with CUDA
  if (use_gpu) {
    // Vector addition using CUDA
    CudaView d_a("Device A", n);
    CudaView d_b("Device B", n);

    Kokkos::Timer timer;

    Kokkos::parallel_for (Kokkos::RangePolicy<Kokkos::Cuda>(0, n), Fill<CudaView>(1.0, d_a));
    Kokkos::parallel_for (Kokkos::RangePolicy<Kokkos::Cuda>(0, n), Fill<CudaView>(2.0, d_b));
    Kokkos::fence();
    std::cout << "[Process " << process << "] Vector initialization time on device (CUDA): " <<
      timer.seconds() << std::endl;

    timer.reset();
    Kokkos::parallel_for (Kokkos::RangePolicy<Kokkos::Cuda>(0, n), Compute<CudaView>(d_a, d_b));
    Kokkos::fence();
    std::cout << "[Process " << process << "] Vector addition time on device (CUDA): " <<
      timer.seconds() << std::endl;

    timer.reset();
    Kokkos::deep_copy(h_a, d_a);
    std::cout << "[Process " << process << "] Time for device -> host data movement: " <<
      timer.seconds() << std::endl;
  }
  else {
    // Vector addition using OpenMP
    HostView h_b("Host B", n);

    Kokkos::Timer timer;

    Kokkos::parallel_for (Kokkos::RangePolicy<Kokkos::OpenMP>(0, n), Fill<HostView>(1.0, h_a));
    Kokkos::parallel_for (Kokkos::RangePolicy<Kokkos::OpenMP>(0, n), Fill<HostView>(2.0, h_b));
    Kokkos::fence();
    std::cout << "[Process " << process << "] Vector initialization time on host (OpenMP): " <<
      timer.seconds() << std::endl;

    timer.reset();
    Kokkos::parallel_for (Kokkos::RangePolicy<Kokkos::OpenMP>(0, n), Compute<HostView>(h_a, h_b));
    Kokkos::fence();
    std::cout << "[Process " << process << "] Time on host (OpenMP): " <<
      timer.seconds() << std::endl;
  }

  // Validate last element of the vector
  uint64_t last_elem = h_a(n-1);
  if (abs(last_elem - CORRECT_VALUE) < 0.000001) {
    std::cout << "[Process " << process << "] Last element validated" << std::endl;
  }
  else {
    std::cout << "[Process " << process << "] Last element NOT validated: it is " <<
      last_elem << ", but should be " << CORRECT_VALUE << std::endl;
  }
}
