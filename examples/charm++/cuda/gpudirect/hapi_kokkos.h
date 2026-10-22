#ifndef HAPI_KOKKOS_H
#define HAPI_KOKKOS_H

// Kokkos on top of the runtime's streams, for the benchmark applications' kernel files.
//
// A chare launches on the stream the runtime handed it (hapiAcquireStream, or a stream it created), and a Kokkos launch
// goes to the stream its execution-space instance wraps, so the instance wrapping a stream is the launch handle. Creating
// an instance from a stream allocates per instance (Kokkos 4.7: scratch flags and space, a lock array of one int per
// resident thread, a pinned host buffer, then a memset), so instances are made once per stream and kept for the life of
// the process. One map for the process: a pooled stream serves chares on different PEs over its life, and with +gpusubmit
// one submitter thread launches for every PE.
//
// Kokkos is initialised on first use, on the device the runtime selected for this process. There is no Kokkos::finalize:
// the runtime tears its streams down at exit, and an instance's finalize fences the stream it wraps.
//
// First launches: Kokkos caches a kernel's function attributes in an unsynchronised static map on the kernel's first
// launch, and the PEs launch the same kernels concurrently, so each launch site runs its first launch under call_once and
// the others wait for it.

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include "hapi_portable.h"

namespace hapikokkos {

#if CMK_HIP
using Space = Kokkos::HIP;
#else
using Space = Kokkos::Cuda;
#endif

struct Instances {
  std::shared_mutex mu;
  std::unordered_map<hapiStream_t, Space> map;
};

inline Instances& instances() {
  static Instances* inst = new Instances;  // never destroyed: see the note on finalize above
  return *inst;
}

inline size_t instanceCount();

inline void ensureInit() {
  static std::once_flag once;
  std::call_once(once, [] {
    int dev = 0;
    if (hapiGetDevice(&dev) != hapiSuccess) dev = 0;
    Kokkos::InitializationSettings s;
    s.set_device_id(dev);
    Kokkos::initialize(s);
    // Registered after Kokkos::initialize and after the runtime's own CUDA initialisation, so it runs before Kokkos's
    // static objects are destroyed and before the CUDA runtime unloads: Kokkos 5 fences its default instance from a
    // static destructor, which throws once the driver is shutting down. The runtime keeps its pooled streams alive to
    // the end, so the instances can be released and Kokkos finalized here.
    std::atexit([] {
      std::fprintf(stderr, "[kokkos] execution-space instances created for %zu streams\n", instanceCount());
      Instances& I = instances();
      std::unique_lock<std::shared_mutex> wl(I.mu);
      I.map.clear();
      Kokkos::finalize();
    });
  });
}

// The execution-space instance wrapping a runtime stream.
inline const Space& space(hapiStream_t stream) {
  Instances& I = instances();
  {
    std::shared_lock<std::shared_mutex> rl(I.mu);
    auto it = I.map.find(stream);
    if (it != I.map.end()) return it->second;
  }
  ensureInit();
  std::unique_lock<std::shared_mutex> wl(I.mu);
  auto it = I.map.find(stream);
  if (it == I.map.end()) it = I.map.emplace(stream, Space(stream)).first;
  return it->second;
}

inline size_t instanceCount() {
  Instances& I = instances();
  std::shared_lock<std::shared_mutex> rl(I.mu);
  return I.map.size();
}

// Run a launch site's first launch under call_once; later launches go straight through.
template <class F>
inline void launch(std::once_flag& once, F&& f) {
  bool first = false;
  std::call_once(once, [&] { first = true; f(); });
  if (!first) f();
}

#ifdef HAPI_KOKKOS_SPLIT_TIMING
// -DHAPI_KOKKOS_SPLIT_TIMING: the pieces of a parallel_for timed separately, summed over the process's threads:
// closure construction (policy validation, team_size_max, scratch bookkeeping) and execute (the kernel launch).
// What parallel_for adds on top (the label string, the profiling hooks) is the launch site's total minus these two.
#include <atomic>
#include <chrono>
inline std::atomic<unsigned long long> split_ctor_ns{0}, split_exec_ns{0}, split_n{0}, split_lookup_ns{0}, split_policy_ns{0};
template <class P, class F>
inline void launchSplit(const P& p, const F& f) {
  const auto t0 = std::chrono::steady_clock::now();
  Kokkos::Impl::ParallelFor<F, P> closure(f, p);
  const auto t1 = std::chrono::steady_clock::now();
  closure.execute();
  const auto t2 = std::chrono::steady_clock::now();
  split_ctor_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(), std::memory_order_relaxed);
  split_exec_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count(), std::memory_order_relaxed);
  split_n.fetch_add(1, std::memory_order_relaxed);
}
inline void splitReport() {
  const unsigned long long n = split_n.load();
  std::fprintf(stderr, "[launch-split] launches=%llu lookup_mean_us=%.2f policy_mean_us=%.2f ctor_mean_us=%.2f exec_mean_us=%.2f\n", n,
               n ? split_lookup_ns.load() / 1e3 / n : 0.0, n ? split_policy_ns.load() / 1e3 / n : 0.0,
               n ? split_ctor_ns.load() / 1e3 / n : 0.0, n ? split_exec_ns.load() / 1e3 / n : 0.0);
}
#endif

}  // namespace hapikokkos

// One static once_flag per launch site.
#define HAPI_KOKKOS_LAUNCH(label, policy, functor)                                           \
  do {                                                                                       \
    static std::once_flag hapi_kokkos_once_;                                                 \
    hapikokkos::launch(hapi_kokkos_once_, [&] { Kokkos::parallel_for(label, policy, functor); }); \
  } while (0)

#endif
