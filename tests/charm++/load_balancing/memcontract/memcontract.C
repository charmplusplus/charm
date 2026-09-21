// The LB memory contract (src/ck-ldb/LBMemoryContract.h) on synthetic stats.
//
// A single-process build cannot put two PEs in different processes, on one
// host, or on different GPUs, which is where every interesting case of the
// contract lives. So the stats here are built by hand and the topology is
// substituted: the model, verifier and batch planner run exactly as they do
// inside CentralLB, and an independent checker replays each plan batch by
// batch to confirm it keeps the invariants.

#include "charm++.h"
#include "memcontract.decl.h"
#include "../../../../src/ck-ldb/LBMemoryContract.h"

#include <cstdio>
#include <string>
#include <vector>

#if CMK_CUDA

namespace
{

const size_t MB = 1024 * 1024;

void fail(const char* test, const std::string& detail)
{
  CkPrintf("memcontract test failed: %s: %s\n", test, detail.c_str());
  CkExit(1);
}

void expect(bool condition, const char* test, const std::string& detail)
{
  if (!condition) fail(test, detail);
}

// pesPerProc PEs to a process, procsPerHost processes to a host.
struct Topo : public LBMemoryTopology
{
  int pesPerProc = 1, procsPerHost = 1;
  bool handoff = true;
  int nodeOf(int pe) const override { return pe / pesPerProc; }
  bool samePhysicalNode(int a, int b) const override
  {
    return nodeOf(a) / procsPerHost == nodeOf(b) / procsPerHost;
  }
  bool intraProcessHandoff() const override { return handoff; }
};

struct Pe
{
  uint64_t gpu;
  size_t devFree, poolFree, arena;
  int slots;
};

struct Scenario
{
  BaseLB::LDStats stats;
  Topo topo;
  std::vector<Pe> pes;

  explicit Scenario(int npes) : stats(npes), pes(npes) {}

  void pe(int p, uint64_t gpu, size_t devFree, size_t poolFree, size_t arena, int slots)
  {
    BaseLB::ProcStats& ps = stats.procs[p];
    ps.pe = p;
    ps.available = true;
    ps.gpu_device_id = gpu;
    ps.gpu_mem_remaining = devFree;
    ps.pool_buff_mem_remaining = poolFree;
    ps.gpu_pool_arena_bytes = arena;
    ps.gpu_ipc_slots = slots;
    pes[p] = Pe{gpu, devFree, poolFree, arena, slots};
  }

  void obj(int from, int to, size_t pup, size_t footprint)
  {
    LDObjData od;
    od.migratable = true;
    od.gpuPupSize = pup;
    *(size_t*)od.getUserData(CkpvAccess(_lb_obj_index)) = footprint;
    stats.objData.push_back(od);
    stats.from_proc.push_back(from);
    stats.to_proc.push_back(to);
  }

  int verify()
  {
    std::vector<ContractVerifier::Move> moves;
    for (int i = 0; i < (int)stats.objData.size(); i++)
      if (stats.to_proc[i] != stats.from_proc[i])
        moves.push_back(ContractVerifier::Move{i, stats.from_proc[i], &stats.to_proc[i]});
    return ContractVerifier::verifyAndRepair(&stats, moves, 0.95, &topo);
  }

  int moved() const
  {
    int n = 0;
    for (size_t i = 0; i < stats.objData.size(); i++)
      if (stats.to_proc[i] != stats.from_proc[i]) n++;
    return n;
  }
};

size_t block(size_t b)
{
  if (b == 0) return 0;
  size_t r = 4;
  while (r < b) r <<= 1;
  return r;
}

// The contract re-derived from the stats alone, independently of the header:
// what each device can reach, and then every batch in order against it.
void checkPlan(const char* test, Scenario& sc, const std::vector<int>& batchOf, int nb)
{
  BaseLB::LDStats& st = sc.stats;
  const int P = (int)sc.pes.size();

  std::vector<uint64_t> gpus;
  std::vector<int> devOf(P);
  for (int p = 0; p < P; p++)
  {
    int d = -1;
    for (size_t k = 0; k < gpus.size(); k++)
      if (gpus[k] == sc.pes[p].gpu) d = (int)k;
    if (d < 0) { d = (int)gpus.size(); gpus.push_back(sc.pes[p].gpu); }
    devOf[p] = d;
  }
  const int D = (int)gpus.size();
  std::vector<long long> reach(D, 0), staging(D, 0);
  std::vector<bool> pooled(D, false);
  for (int d = 0; d < D; d++)
  {
    size_t devFree = SIZE_MAX, pool = 0, arena = 0, legacy = 0;
    std::vector<int> procs;
    for (int p = 0; p < P; p++)
    {
      if (devOf[p] != d) continue;
      devFree = std::min(devFree, sc.pes[p].devFree);
      if (legacy == 0) legacy = sc.pes[p].poolFree;
      if (sc.pes[p].arena == 0) continue;
      arena = sc.pes[p].arena;
      const int node = sc.topo.nodeOf(p);
      if (std::find(procs.begin(), procs.end(), node) == procs.end())
      {
        procs.push_back(node);
        pool += sc.pes[p].poolFree;
      }
    }
    pooled[d] = arena > 0;
    // A pooled device reaches only what its arenas have free: growth is not
    // credited (T_g in LBMemoryContract.h).
    reach[d] = (long long)(pooled[d] ? pool : devFree);
    staging[d] = pooled[d] ? reach[d] : (long long)legacy;
  }
  const bool anyPooled = std::find(pooled.begin(), pooled.end(), true) != pooled.end();

  std::vector<long long> avail(D);
  for (int d = 0; d < D; d++)
  {
    avail[d] = (long long)(reach[d] * 0.95);
    // L_g: one migration window of landings per PE of a pooled device, at
    // most a quarter of its reach (LBMigrateWindow.h).
    if (pooled[d])
    {
      int pes = 0;
      for (int p = 0; p < P; p++) if (devOf[p] == d) pes++;
      avail[d] -= (long long)lbLandingReserveBytes((size_t)reach[d], (size_t)pes);
      if (avail[d] < 0) avail[d] = 0;
    }
  }

  for (size_t i = 0; i < st.objData.size(); i++)
    if (st.to_proc[i] != st.from_proc[i] && (batchOf[i] < 0 || batchOf[i] >= nb))
      fail(test, "a move has no batch");

  for (int k = 0; k < nb; k++)
  {
    std::vector<long long> staged(D, 0), landed(D, 0), dep(D, 0), arr(D, 0);
    std::vector<int> ipc(P, 0);
    for (size_t i = 0; i < st.objData.size(); i++)
    {
      const int from = st.from_proc[i], to = st.to_proc[i];
      if (from == to || batchOf[i] != k) continue;
      const int s = devOf[from], d = devOf[to];
      const bool sameProc = sc.topo.nodeOf(from) == sc.topo.nodeOf(to);
      if (sameProc && s == d && sc.topo.handoff) continue;  // handoff: no bytes
      const long long sig =
          (long long)(anyPooled ? block(st.objData[i].gpuPupSize) : st.objData[i].gpuPupSize);
      const long long phi = std::max(
          (long long)*(size_t*)st.objData[i].getUserData(CkpvAccess(_lb_obj_index)), sig);
      staged[s] += sig;
      if (!pooled[d]) landed[d] += sig;
      if (s != d) { dep[s] += phi; arr[d] += phi; }
      if (sig > 0 && !sameProc && sc.topo.samePhysicalNode(from, to)) ipc[from]++;
    }
    for (int d = 0; d < D; d++)
    {
      char buf[256];
      if (pooled[d])
      {
        snprintf(buf, sizeof(buf), "batch %d device %d: staged %lld over A %lld", k, d,
                 staged[d], avail[d]);
        expect(staged[d] <= avail[d], test, buf);
        snprintf(buf, sizeof(buf), "batch %d device %d: peak %lld over A %lld", k, d,
                 staged[d] - dep[d] + arr[d], avail[d]);
        expect(staged[d] - dep[d] + arr[d] <= avail[d], test, buf);
      }
      else
      {
        snprintf(buf, sizeof(buf), "batch %d device %d: staging %lld over S %lld", k, d,
                 staged[d] + landed[d], staging[d]);
        expect(staged[d] + landed[d] <= staging[d], test, buf);
        snprintf(buf, sizeof(buf), "batch %d device %d: net %lld over A %lld", k, d,
                 arr[d] - dep[d], avail[d]);
        expect(arr[d] - dep[d] <= avail[d], test, buf);
      }
      avail[d] -= arr[d] - dep[d];
    }
    for (int p = 0; p < P; p++)
      if (sc.pes[p].slots > 0)
      {
        char buf[128];
        snprintf(buf, sizeof(buf), "batch %d PE %d: %d IPC moves over %d slots", k, p, ipc[p],
                 sc.pes[p].slots);
        expect(ipc[p] <= sc.pes[p].slots, test, buf);
      }
  }
}

int plan(Scenario& sc, std::vector<int>& batchOf, int* refused)
{
  return LBBatchPlanner::plan(&sc.stats, batchOf, refused, 0.95, &sc.topo);
}

// Two processes on one host, each with its own device and plenty of memory:
// only the IPC slots bind.
void testSlotsSplitAWaveThatFitsByBytes()
{
  const char* test = "slot term";
  for (int slots : {64, 0})
  {
    Scenario sc(2);
    sc.topo.procsPerHost = 2;
    sc.pe(0, 0, 0, 10240 * MB, 256 * MB, slots);
    sc.pe(1, 1, 0, 10240 * MB, 256 * MB, slots);
    for (int i = 0; i < 200; i++) sc.obj(0, 1, 1000, 1024);
    expect(sc.verify() == 0, test, "verifier refused a move that fits");
    std::vector<int> batchOf;
    int refused = 0;
    const int nb = plan(sc, batchOf, &refused);
    char buf[96];
    snprintf(buf, sizeof(buf), "%d slot(s): %d batches", slots, nb);
    expect(nb == (slots ? 4 : 1), test, buf);
    expect(refused == 0, test, "planner refused a move");
    checkPlan(test, sc, batchOf, nb);
  }
}

// Different hosts, so no slots: final placement binds, and the verifier cuts
// the one-way wave to what the destination can hold less one payload.
void testInfeasibleOneWayMoves()
{
  const char* test = "infeasible one-way";
  Scenario sc(2);
  sc.pe(0, 0, 0, 100 * MB, 256 * MB, 0);
  sc.pe(1, 1, 0, 100 * MB, 256 * MB, 0);
  for (int i = 0; i < 40; i++) sc.obj(0, 1, 4 * MB, 4 * MB);
  const int refused = sc.verify();
  // The destination packs nothing, so it holds no payload back: H = 0.95 *
  // 100 MB = 95 MB, and 23 arrivals of 4 MB fit.
  char buf[96];
  snprintf(buf, sizeof(buf), "refused %d, kept %d", refused, sc.moved());
  expect(sc.moved() == 23 && refused == 17, test, buf);
  std::vector<int> batchOf;
  int pr = 0;
  const int nb = plan(sc, batchOf, &pr);
  expect(pr == 0, test, "planner refused after the verifier");
  checkPlan(test, sc, batchOf, nb);
}

// Both devices full, objects trade places. Neither arrival fits before its
// partner leaves; the verifier must take the departure credit and the planner
// must stage the swap in pieces rather than refuse it.
void testSwapOnFullDevices()
{
  const char* test = "swap on full devices";
  Scenario sc(2);
  sc.pe(0, 0, 0, 10 * MB, 256 * MB, 0);
  sc.pe(1, 1, 0, 10 * MB, 256 * MB, 0);
  for (int i = 0; i < 3; i++) sc.obj(0, 1, 3 * MB, 8 * MB);
  for (int i = 0; i < 3; i++) sc.obj(1, 0, 3 * MB, 8 * MB);
  char buf[96];
  const int refused = sc.verify();
  snprintf(buf, sizeof(buf), "verifier refused %d move(s) of a swap", refused);
  expect(refused == 0, test, buf);
  std::vector<int> batchOf;
  int pr = 0;
  const int nb = plan(sc, batchOf, &pr);
  snprintf(buf, sizeof(buf), "planner refused %d, %d batches", pr, nb);
  expect(pr == 0 && nb >= 2 && sc.moved() == 6, test, buf);
  checkPlan(test, sc, batchOf, nb);
}

// Two processes share one device. Their moves change no device's final
// footprint, but each stages a payload and lands an arena in the shared pool.
void testProcessesSharingADevice()
{
  const char* test = "processes sharing a device";
  Scenario sc(2);
  sc.topo.procsPerHost = 2;
  sc.pe(0, 7, 0, 8 * MB, 256 * MB, 0);
  sc.pe(1, 7, 0, 8 * MB, 256 * MB, 0);
  for (int i = 0; i < 10; i++) sc.obj(0, 1, 2 * MB, 2 * MB);
  expect(sc.verify() == 0, test, "verifier refused a same-device move");
  std::vector<int> batchOf;
  int pr = 0;
  const int nb = plan(sc, batchOf, &pr);
  // T = 16 MB (two processes' pools), A = 15.2 MB, 2 MB blocks: 7 a batch.
  char buf[64];
  snprintf(buf, sizeof(buf), "%d batches", nb);
  expect(nb == 2 && pr == 0, test, buf);
  checkPlan(test, sc, batchOf, nb);
}

// One process, one device: the chare is handed over, nothing is staged.
void testHandoffMovesNoBytes()
{
  const char* test = "handoff";
  Scenario sc(2);
  sc.topo.pesPerProc = 2;
  sc.pe(0, 0, 0, 1 * MB, 256 * MB, 16);
  sc.pe(1, 0, 0, 1 * MB, 256 * MB, 16);
  for (int i = 0; i < 10; i++) sc.obj(0, 1, 2 * MB, 2 * MB);
  expect(sc.verify() == 0, test, "verifier refused a handoff");
  std::vector<int> batchOf;
  int pr = 0;
  expect(plan(sc, batchOf, &pr) == 1 && pr == 0, test, "a handoff was batched");

  // With the handoff disabled the same moves stage (CHARM_NO_INTRAPROC_MIGRATE).
  sc.topo.handoff = false;
  expect(sc.verify() == 10, test, "a 2 MB payload fit a 1 MB pool");
}

// Without the pool, staging is the +gpulbbuffer region, charged on both ends.
void testSeparateStagingRegion()
{
  const char* test = "separate staging";
  Scenario sc(2);
  sc.pe(0, 0, 1000 * MB, 10 * MB, 0, 0);
  sc.pe(1, 1, 1000 * MB, 10 * MB, 0, 0);
  for (int i = 0; i < 6; i++) sc.obj(0, 1, 4 * MB, 4 * MB);
  expect(sc.verify() == 0, test, "verifier refused a move that fits");
  std::vector<int> batchOf;
  int pr = 0;
  const int nb = plan(sc, batchOf, &pr);
  char buf[64];
  snprintf(buf, sizeof(buf), "%d batches", nb);
  expect(nb == 3 && pr == 0, test, buf);
  checkPlan(test, sc, batchOf, nb);
}

}  // namespace

class Main : public CBase_Main
{
public:
  Main(CkArgMsg* m)
  {
    delete m;
    expect(CkpvAccess(_lb_obj_index) >= 0, "setup", "no footprint slot registered");
    testSlotsSplitAWaveThatFitsByBytes();
    testInfeasibleOneWayMoves();
    testSwapOnFullDevices();
    testProcessesSharingADevice();
    testHandoffMovesNoBytes();
    testSeparateStagingRegion();
    CkPrintf("memcontract test passed\n");
    CkExit(0);
  }
};

#else

class Main : public CBase_Main
{
public:
  Main(CkArgMsg* m)
  {
    delete m;
    CkPrintf("memcontract test skipped: not a CUDA build\n");
    CkExit(0);
  }
};

#endif

#include "memcontract.def.h"
