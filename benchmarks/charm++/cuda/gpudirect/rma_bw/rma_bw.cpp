// rma_bw: a one-sided put against a get between two ranks on two hosts, the
// source and the destination each in device or host memory, one operation of
// `size` bytes at a time, `iters` operations. LCI over libfabric, no Charm++:
// it measures the fabric and the PCIe paths the runtime's device channels run
// on. Written 2026-11-07 to pin down why a device-to-device channel put ran at
// 3 to 5 GB/s on Slingshot where the entry-method read ran at 12 to 14 on the
// same NIC (examples/charm++/cuda/gpudirect/moe/README.md, "The read path").
//
//   rma_bw put|get dd|dh|hd|hh <bytes> <iters> [both]
//
// Rank 0 holds the source, rank 1 the destination; `put` is issued by rank 0,
// `get` by rank 1; the data flows 0 -> 1 either way. The memory class is the
// source's then the destination's (`dh` = device source, host destination).
// `both` issues from both ranks at once, each into the other's destination:
// the exchange case. Run one rank per host (srun -N 2 -n 2), the device on
// the NIC's NUMA node.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "lci.hpp"
#ifdef RMA_BW_HIP
#include <hip/hip_runtime.h>
#define devSetDevice hipSetDevice
#define devMalloc hipMalloc
#define devMemset hipMemset
#define devSync hipDeviceSynchronize
#define devSuccess hipSuccess
#else
#include <cuda_runtime.h>
#define devSetDevice cudaSetDevice
#define devMalloc cudaMalloc
#define devMemset cudaMemset
#define devSync cudaDeviceSynchronize
#define devSuccess cudaSuccess
#endif

static void* alloc(bool device, size_t n) {
  void* p = nullptr;
  if (device) {
    if (devMalloc(&p, n) != devSuccess) { fprintf(stderr, "device allocation of %zu bytes failed\n", n); exit(1); }
    devMemset(p, 1, n); devSync();
  } else {
    if (posix_memalign(&p, 4096, n) != 0) exit(1);
    memset(p, 1, n);
  }
  return p;
}

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: rma_bw put|get dd|dh|hd|hh <bytes> <iters> [both]\n"); return 2; }
  const std::string mode = argv[1], mem = argv[2];
  const size_t size = strtoull(argv[3], nullptr, 10);
  const int iters = atoi(argv[4]);
  const bool both = argc > 5 && std::string(argv[5]) == "both";
  if ((mode != "put" && mode != "get") || mem.size() != 2 || iters < 1 || size == 0) {
    fprintf(stderr, "usage: rma_bw put|get dd|dh|hd|hh <bytes> <iters> [both]\n"); return 2;
  }
  const bool srcDev = mem[0] == 'd', dstDev = mem[1] == 'd';
  devSetDevice(0);
  lci::g_runtime_init();
  const int me = lci::get_rank_me(), n = lci::get_rank_n();
  if (n != 2) { if (me == 0) fprintf(stderr, "rma_bw needs exactly 2 ranks\n"); lci::g_runtime_fina(); return 2; }
  // Every rank holds a source and a destination (under `both` each rank is a
  // source for the other); the memory class of each follows its role.
  void* src = alloc(srcDev, size);
  void* dst = alloc(dstDev, size);
  lci::mr_t mr_src = lci::register_memory(src, size);
  lci::mr_t mr_dst = lci::register_memory(dst, size);
  lci::rmr_t mine[2] = { lci::get_rmr(mr_src), lci::get_rmr(mr_dst) };
  lci::rmr_t all[4];
  lci::allgather(mine, all, sizeof(mine));
  const int peer = 1 - me;
  const lci::rmr_t peer_src = all[2 * peer], peer_dst = all[2 * peer + 1];
  lci::comp_t comp = lci::alloc_counter();
  const bool issue = both || (mode == "put" ? me == 0 : me == 1);
  lci::barrier();
  auto t0 = std::chrono::steady_clock::now();
  if (issue) {
    for (int i = 0; i < iters; i++) {
      lci::status_t st;
      do {
        if (mode == "put") st = lci::post_put_x(peer, src, size, comp, 0, peer_dst).mr(mr_src).allow_done(false)();
        else               st = lci::post_get_x(peer, dst, size, comp, 0, peer_src).mr(mr_dst).allow_done(false)();
        lci::progress();
      } while (st.is_retry());
      while (lci::counter_get(comp) < (uint64_t)(i + 1)) lci::progress();
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  lci::barrier();
  const double s = std::chrono::duration<double>(t1 - t0).count();
  if (issue) {
    const double per = s / iters;
    printf("rma_bw rank %d %s %s%s %10zu B x %d: %8.3f ms per op, %6.2f GB/s\n", me, mode.c_str(), mem.c_str(),
           both ? " both" : "", size, iters, per * 1e3, size / per / 1e9);
    fflush(stdout);
  }
  lci::barrier();
  lci::free_comp(&comp);
  lci::deregister_memory(&mr_src); lci::deregister_memory(&mr_dst);
  lci::g_runtime_fina();
  return 0;
}
