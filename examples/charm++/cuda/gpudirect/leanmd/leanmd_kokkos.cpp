// Kokkos variant of leanmd.cu. The kernels are the kernels of leanmd.cu as functors over the same raw pointers, launched
// on the chare's stream through the Kokkos execution-space instance that wraps it (hapi_kokkos.h). Every kernel is a
// TeamPolicy of BLOCK_THREADS-thread teams with the league the kernel's grid was, so the launch shapes are the ones
// leanmd.cu uses (a RangePolicy would let Kokkos pick 128-thread blocks for the flat kernels); the shared arrays of the
// reduction kernels are team scratch at level 0, which is shared memory. An A/B against leanmd.cu therefore measures the
// launch path and the functor form and nothing else. The decomposition notes are in leanmd.cu.
#include "defs.h"
#include "leanmd_cuda.h"
#include "hapi_kokkos.h"

#include <cstdio>

#define VDW_A_D  (1.1328e-133)
#define VDW_B_D  (2.23224e-76)
#define POW_TEN     (1.0e-10)
#define POW_TWENTY  (1.0e-20)
#define POW_TEN_POS (1.0e10)

#define BLOCK_THREADS 256

using hapikokkos::Space;
using hapikokkos::space;
// LMD_TEAM_LB=N: __launch_bounds__(BLOCK_THREADS, N) on every kernel, a register-allocation hint the hand-written
// kernels do not carry; a diagnosis knob for the force kernel's code generation.
#ifdef LMD_TEAM_LB
using Team = Kokkos::TeamPolicy<Space, Kokkos::LaunchBounds<BLOCK_THREADS, LMD_TEAM_LB>>;
#else
using Team = Kokkos::TeamPolicy<Space>;
#endif
using Member = Team::member_type;

// LMD_UNROLL=N: an explicit unroll of the force kernels' j-loop. nvcc unrolls the hand-written kernel's loop on its own
// (3 copies of the body in the sm_86 SASS); inside Kokkos's league loop it stopped at 2 and pairForceBoth ran 10-15%
// slower (A40, 2026-10-21: 161.8 vs 140.5 us at 2039 blocks). Unroll 2 brings the kernel back to the hand-written
// kernel's time (141.8 us) at 72 registers; 3 and 4 are no better. LMD_UNROLL=0 compiles without the pragma.
#ifndef LMD_UNROLL
#define LMD_UNROLL 2
#endif
#if LMD_UNROLL > 0
#define LMD_STR(x) #x
#define LMD_XSTR(x) LMD_STR(x)
#define LMD_UNROLL_PRAGMA _Pragma(LMD_XSTR(unroll LMD_UNROLL))
#else
#define LMD_UNROLL_PRAGMA
#endif

// A league of `league` teams of BLOCK_THREADS threads on the stream, with `scratchBytes` of shared memory per team.
static inline Team teamOf(hapiStream_t stream, int league, size_t scratchBytes) {
#ifdef HAPI_KOKKOS_SPLIT_TIMING
  const auto t0 = std::chrono::steady_clock::now();
  const Space& sp = space(stream);
  const auto t1 = std::chrono::steady_clock::now();
  Team p(sp, league, BLOCK_THREADS);
  if (scratchBytes) p.set_scratch_size(0, Kokkos::PerTeam(scratchBytes));
  const auto t2 = std::chrono::steady_clock::now();
  hapikokkos::split_lookup_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(), std::memory_order_relaxed);
  hapikokkos::split_policy_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count(), std::memory_order_relaxed);
  return p;
#else
  Team p(space(stream), league, BLOCK_THREADS);
  if (scratchBytes) p.set_scratch_size(0, Kokkos::PerTeam(scratchBytes));
  return p;
#endif
}

// Blocks of BLOCK_THREADS over n flat indices, as <<<blocks, BLOCK_THREADS>>> was.
static inline int blocksFor(int n) { return (n + BLOCK_THREADS - 1) / BLOCK_THREADS; }

template <class P, class F>
static inline void site(std::once_flag& once, const char* label, const P& p, const F& f) {
#ifdef HAPI_KOKKOS_SPLIT_TIMING
  (void)label;
  hapikokkos::launch(once, [&] { hapikokkos::launchSplit(p, f); });
#else
  hapikokkos::launch(once, [&] { Kokkos::parallel_for(label, p, f); });
#endif
}

struct PairForceKernel {
  const vec3* A; int nA; const vec3* B; int nB; vec3* forceA; vec3 shift; double cutoffSq; int selfInteract;
  double* energyPartial;
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    const int i = m.league_rank();
    if (i >= nA) return;

    double* sx = (double*)m.team_shmem().get_shmem(sizeof(double) * BLOCK_THREADS);
    double* sy = (double*)m.team_shmem().get_shmem(sizeof(double) * BLOCK_THREADS);
    double* sz = (double*)m.team_shmem().get_shmem(sizeof(double) * BLOCK_THREADS);
    double* se = (double*)m.team_shmem().get_shmem(sizeof(double) * BLOCK_THREADS);

    const vec3* __restrict__ Ar = A;
    const vec3* __restrict__ Br = B;
    vec3* __restrict__ fA = forceA;
    double* __restrict__ eP = energyPartial;
    const int t = m.team_rank();
    const int nt = m.team_size();

    const vec3 pos_i = Ar[i];
    double fx = 0.0, fy = 0.0, fz = 0.0, en = 0.0;

    LMD_UNROLL_PRAGMA
    for (int j = t; j < nB; j += nt)
    {
      if (selfInteract && i == j) continue;
      const vec3 pos_j = Br[j];
      const double dx = pos_i.x - (pos_j.x + shift.x);
      const double dy = pos_i.y - (pos_j.y + shift.y);
      const double dz = pos_i.z - (pos_j.z + shift.z);
      double rsqd = dx*dx + dy*dy + dz*dz;
      if (rsqd > 1.0 && rsqd < cutoffSq)
      {
        rsqd *= POW_TWENTY;
        const double rSix    = rsqd * rsqd * rsqd;
        const double rTwelve = rSix * rSix;
        const double f  = (12.0 * VDW_A_D) / rTwelve - (6.0 * VDW_B_D) / rSix;
        const double fr = (f / rsqd) * POW_TEN;
        fx += dx * fr;
        fy += dy * fr;
        fz += dz * fr;
        if (eP != nullptr && (!selfInteract || j > i))
          en += VDW_A_D / rTwelve - VDW_B_D / rSix;
      }
    }

    sx[t] = fx; sy[t] = fy; sz[t] = fz; se[t] = en;
    m.team_barrier();
    for (int s = nt / 2; s > 0; s >>= 1)
    {
      if (t < s)
      {
        sx[t] += sx[t + s];
        sy[t] += sy[t + s];
        sz[t] += sz[t + s];
        se[t] += se[t + s];
      }
      m.team_barrier();
    }
    if (t == 0)
    {
      fA[i].x = sx[0];
      fA[i].y = sy[0];
      fA[i].z = sz[0];
      if (eP != nullptr) eP[i] = se[0];
    }
  }
};

struct PairForceBothKernel {
  const vec3* A; int nA; const vec3* B; int nB; vec3* forceA; vec3* forceB; vec3 shiftB; double cutoffSq;
  double* energyPartial;
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    const int blk = m.league_rank();
    if (blk >= nA + nB) return;

    double* sx = (double*)m.team_shmem().get_shmem(sizeof(double) * BLOCK_THREADS);
    double* sy = (double*)m.team_shmem().get_shmem(sizeof(double) * BLOCK_THREADS);
    double* sz = (double*)m.team_shmem().get_shmem(sizeof(double) * BLOCK_THREADS);
    double* se = (double*)m.team_shmem().get_shmem(sizeof(double) * BLOCK_THREADS);

    const int t = m.team_rank();
    const int nt = m.team_size();

    const bool onA = blk < nA;
    const int i = onA ? blk : blk - nA;
    const vec3* self = onA ? A : B;
    const vec3* other = onA ? B : A;
    const int nOther = onA ? nB : nA;
    const double shx = onA ? shiftB.x : -shiftB.x;
    const double shy = onA ? shiftB.y : -shiftB.y;
    const double shz = onA ? shiftB.z : -shiftB.z;
    const bool wantEnergy = onA && energyPartial != nullptr;

    const vec3 pos_i = self[i];
    double fx = 0.0, fy = 0.0, fz = 0.0, en = 0.0;

    LMD_UNROLL_PRAGMA
    for (int j = t; j < nOther; j += nt)
    {
      const vec3 pos_j = other[j];
      const double dx = pos_i.x - (pos_j.x + shx);
      const double dy = pos_i.y - (pos_j.y + shy);
      const double dz = pos_i.z - (pos_j.z + shz);
      double rsqd = dx*dx + dy*dy + dz*dz;
      if (rsqd > 1.0 && rsqd < cutoffSq)
      {
        rsqd *= POW_TWENTY;
        const double rSix    = rsqd * rsqd * rsqd;
        const double rTwelve = rSix * rSix;
        const double f  = (12.0 * VDW_A_D) / rTwelve - (6.0 * VDW_B_D) / rSix;
        const double fr = (f / rsqd) * POW_TEN;
        fx += dx * fr;
        fy += dy * fr;
        fz += dz * fr;
        if (wantEnergy) en += VDW_A_D / rTwelve - VDW_B_D / rSix;
      }
    }

    sx[t] = fx; sy[t] = fy; sz[t] = fz; se[t] = en;
    m.team_barrier();
    for (int s = nt / 2; s > 0; s >>= 1)
    {
      if (t < s)
      {
        sx[t] += sx[t + s];
        sy[t] += sy[t + s];
        sz[t] += sz[t + s];
        se[t] += se[t + s];
      }
      m.team_barrier();
    }
    if (t == 0)
    {
      vec3* out = onA ? forceA : forceB;
      out[i].x = sx[0];
      out[i].y = sy[0];
      out[i].z = sz[0];
      if (wantEnergy) energyPartial[i] = se[0];
    }
  }
};

struct GatherPositionsKernel {
  const Particle* p; vec3* pos; int n;
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    const Particle* __restrict__ pr = p;
    vec3* __restrict__ out = pos;
    const int i = m.league_rank() * m.team_size() + m.team_rank();
    if (i < n) out[i] = pr[i].pos;
  }
};

struct ReduceDoublesKernel {
  const double* in; int n; double* out;
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    double* s = (double*)m.team_shmem().get_shmem(sizeof(double) * BLOCK_THREADS);
    const double* __restrict__ inr = in;
    const int t = m.team_rank();
    const int nt = m.team_size();
    double acc = 0.0;
    for (int i = t; i < n; i += nt) acc += inr[i];
    s[t] = acc;
    m.team_barrier();
    for (int k = nt / 2; k > 0; k >>= 1)
    {
      if (t < k) s[t] += s[t + k];
      m.team_barrier();
    }
    if (t == 0) *out = s[0];
  }
};

struct ZeroForcesKernel {
  vec3* f; int n;
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    const int i = m.league_rank() * m.team_size() + m.team_rank();
    if (i < n) { f[i].x = 0.0; f[i].y = 0.0; f[i].z = 0.0; }
  }
};

struct SumForcesKernel {
  vec3* dst; const vec3* src; int slots, stride, n;
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    vec3* __restrict__ d = dst;
    const vec3* __restrict__ s = src;
    const int i = m.league_rank() * m.team_size() + m.team_rank();
    if (i >= n) return;
    double x = 0.0, y = 0.0, z = 0.0;
    for (int k = 0; k < slots; k++)
    {
      const vec3 f = s[(size_t)k * stride + i];
      x += f.x; y += f.y; z += f.z;
    }
    d[i].x = x; d[i].y = y; d[i].z = z;
  }
};

struct AccumulateForcesKernel {
  vec3* dst; const vec3* src; int n;
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    vec3* __restrict__ d = dst;
    const vec3* __restrict__ s = src;
    const int i = m.league_rank() * m.team_size() + m.team_rank();
    if (i < n)
    {
      d[i].x += s[i].x;
      d[i].y += s[i].y;
      d[i].z += s[i].z;
    }
  }
};

struct IntegrateKernel {
  Particle* p; const vec3* f; int n; double dtVel, dtPos, maxVelocity;
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    Particle* __restrict__ pr = p;
    const vec3* __restrict__ fr = f;
    const int i = m.league_rank() * m.team_size() + m.team_rank();
    if (i >= n) return;

    const double invMass = 1.0 / pr[i].mass;

    pr[i].acc.x = fr[i].x * invMass;
    pr[i].acc.y = fr[i].y * invMass;
    pr[i].acc.z = fr[i].z * invMass;

    pr[i].vel.x += pr[i].acc.x * dtVel;
    pr[i].vel.y += pr[i].acc.y * dtVel;
    pr[i].vel.z += pr[i].acc.z * dtVel;

    if (pr[i].vel.x >  maxVelocity) pr[i].vel.x =  maxVelocity;
    if (pr[i].vel.x < -maxVelocity) pr[i].vel.x = -maxVelocity;
    if (pr[i].vel.y >  maxVelocity) pr[i].vel.y =  maxVelocity;
    if (pr[i].vel.y < -maxVelocity) pr[i].vel.y = -maxVelocity;
    if (pr[i].vel.z >  maxVelocity) pr[i].vel.z =  maxVelocity;
    if (pr[i].vel.z < -maxVelocity) pr[i].vel.z = -maxVelocity;

    pr[i].pos.x += pr[i].vel.x * dtPos;
    pr[i].pos.y += pr[i].vel.y * dtPos;
    pr[i].pos.z += pr[i].vel.z * dtPos;
  }
};

struct KineticEnergyKernel {
  const Particle* p; int n; double* partial;
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    double* s = (double*)m.team_shmem().get_shmem(sizeof(double) * BLOCK_THREADS);
    const Particle* __restrict__ pr = p;
    double* __restrict__ out = partial;
    const int t = m.team_rank();
    const int nt = m.team_size();
    double acc = 0.0;
    for (int i = m.league_rank() * nt + t; i < n; i += nt * m.league_size())
    {
      const vec3 v = pr[i].vel;
      acc += 0.5 * pr[i].mass * (v.x*v.x + v.y*v.y + v.z*v.z) * POW_TEN_POS;
    }
    s[t] = acc;
    m.team_barrier();
    for (int k = nt / 2; k > 0; k >>= 1)
    {
      if (t < k) s[t] += s[t + k];
      m.team_barrier();
    }
    if (t == 0) out[m.league_rank()] = s[0];
  }
};

KOKKOS_INLINE_FUNCTION int migrateOffset(double p, double origin, double cellSize)
{
  if (p < origin) return -1;
  if (p > origin + cellSize) return 1;
  return 0;
}

KOKKOS_INLINE_FUNCTION double wrapCoord(double v, double origin, double extent)
{
  if (v < origin) v += extent;
  if (v > origin + extent) v -= extent;
  return v;
}

struct BinParticlesKernel {
  const Particle* p; int n; int cellX, cellY, cellZ; double cellSizeX, cellSizeY, cellSizeZ; int dimX, dimY, dimZ;
  Particle* stay; Particle* send; int exchCapacity; int* counts;
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    const Particle* __restrict__ pr = p;
    Particle* __restrict__ st = stay;
    Particle* __restrict__ sd = send;
    int* __restrict__ ct = counts;
    const int i = m.league_rank() * m.team_size() + m.team_rank();
    if (i >= n) return;

    const double ox = cellX * cellSizeX + CELL_ORIGIN_X;
    const double oy = cellY * cellSizeY + CELL_ORIGIN_Y;
    const double oz = cellZ * cellSizeZ + CELL_ORIGIN_Z;

    Particle q = pr[i];

    const int dx = migrateOffset(q.pos.x, ox, cellSizeX);
    const int dy = migrateOffset(q.pos.y, oy, cellSizeY);
    const int dz = migrateOffset(q.pos.z, oz, cellSizeZ);

    const int bucket = (dx + KAWAY_X) * NBRS_Y * NBRS_Z
                     + (dy + KAWAY_Y) * NBRS_Z
                     + (dz + KAWAY_Z);
    const int SELF_BUCKET = KAWAY_X * NBRS_Y * NBRS_Z + KAWAY_Y * NBRS_Z + KAWAY_Z;

    const int slot = Kokkos::atomic_fetch_add(&ct[bucket], 1);

    if (bucket == SELF_BUCKET)
    {
      st[slot] = q;
    }
    else
    {
      q.pos.x = wrapCoord(q.pos.x, CELL_ORIGIN_X, cellSizeX * dimX);
      q.pos.y = wrapCoord(q.pos.y, CELL_ORIGIN_Y, cellSizeY * dimY);
      q.pos.z = wrapCoord(q.pos.z, CELL_ORIGIN_Z, cellSizeZ * dimZ);
      if (slot < exchCapacity) sd[bucket * exchCapacity + slot] = q;
    }
  }
};

struct AppendParticlesKernel {
  Particle* dst; int dstOffset; const Particle* src; int n;
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    Particle* __restrict__ d = dst;
    const Particle* __restrict__ s = src;
    const int i = m.league_rank() * m.team_size() + m.team_rank();
    if (i < n) d[dstOffset + i] = s[i];
  }
};

// ---------------------------------------------------------------------------
// Host-side launchers: the ones of leanmd.cu, through the runtime's submitter exactly as there.
// ---------------------------------------------------------------------------

void hapiSubmitClosure(hapiStream_t stream, void (*fn)(void*), const void* state, size_t bytes);
#ifdef LMD_LAUNCH_TIMING
// -DLMD_LAUNCH_TIMING: as in leanmd.cu, host wall time inside the launch closures, reported at exit.
#include <atomic>
#include <chrono>
static std::atomic<unsigned long long> lmd_launch_ns{0}, lmd_launch_n{0};
static void lmdLaunchReport() {
  static std::atomic<bool> done{false};
  if (done.exchange(true)) return;   // one report per process (the registration below is per closure type)
  unsigned long long ns = lmd_launch_ns.load(), n = lmd_launch_n.load();
  std::fprintf(stderr, "[launch-timing] launches=%llu host_total_ms=%.1f mean_us=%.2f\n", n, ns / 1e6, n ? ns / 1e3 / n : 0.0);
#ifdef HAPI_KOKKOS_SPLIT_TIMING
  hapikokkos::splitReport();
#endif
}
template <class F> static inline void submitLaunch(hapiStream_t stream, F f) {
  static bool registered = (std::atexit(lmdLaunchReport), true); (void)registered;
  const auto t0 = std::chrono::steady_clock::now();
  hapiSubmitClosure(stream, [](void* p) { (*(F*)p)(); }, &f, sizeof(F));
  lmd_launch_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(), std::memory_order_relaxed);
  lmd_launch_n.fetch_add(1, std::memory_order_relaxed);
}
#else
template <class F> static inline void submitLaunch(hapiStream_t stream, F f) {
  hapiSubmitClosure(stream, [](void* p) { (*(F*)p)(); }, &f, sizeof(F));
}
#endif

static const size_t FOUR_ARRAYS = 4 * sizeof(double) * BLOCK_THREADS;
static const size_t ONE_ARRAY = sizeof(double) * BLOCK_THREADS;

void invokePairForce(const vec3* d_A, int nA, const vec3* d_B, int nB,
                     vec3* d_forceA, vec3 shift, double cutoffSq, bool selfInteract,
                     double* d_energyPartial, hapiStream_t stream)
{
  if (nA <= 0 || nB <= 0) return;
  submitLaunch(stream, [=]() {
    static std::once_flag once;
    site(once, "leanmd::pairForce", teamOf(stream, nA, FOUR_ARRAYS),
         PairForceKernel{d_A, nA, d_B, nB, d_forceA, shift, cutoffSq, selfInteract ? 1 : 0, d_energyPartial});
  });
}

void invokePairForceBoth(const vec3* d_A, int nA, const vec3* d_B, int nB,
                         vec3* d_forceA, vec3* d_forceB, vec3 shiftB,
                         double cutoffSq, double* d_energyPartial,
                         hapiStream_t stream)
{
  if (nA <= 0 || nB <= 0) return;
  submitLaunch(stream, [=]() {
    static std::once_flag once;
    site(once, "leanmd::pairForceBoth", teamOf(stream, nA + nB, FOUR_ARRAYS),
         PairForceBothKernel{d_A, nA, d_B, nB, d_forceA, d_forceB, shiftB, cutoffSq, d_energyPartial});
  });
}

void invokeSumForces(vec3* d_dst, const vec3* d_slots, int slots, int stride, int n,
                     hapiStream_t stream)
{
  if (n <= 0 || slots <= 0) return;
  submitLaunch(stream, [=]() {
    static std::once_flag once;
    site(once, "leanmd::sumForces", teamOf(stream, blocksFor(n), 0),
         SumForcesKernel{d_dst, d_slots, slots, stride, n});
  });
}

void invokeGatherPositions(const Particle* d_p, vec3* d_pos, int n,
                           hapiStream_t stream)
{
  if (n <= 0) return;
  submitLaunch(stream, [=]() {
    static std::once_flag once;
    site(once, "leanmd::gatherPositions", teamOf(stream, blocksFor(n), 0),
         GatherPositionsKernel{d_p, d_pos, n});
  });
}

void invokeReduceDoubles(const double* d_in, int n, double* d_out,
                         hapiStream_t stream)
{
  if (n <= 0) return;
  submitLaunch(stream, [=]() {
    static std::once_flag once;
    site(once, "leanmd::reduceDoubles", teamOf(stream, 1, ONE_ARRAY),
         ReduceDoublesKernel{d_in, n, d_out});
  });
}

void invokeZeroForces(vec3* d_f, int n, hapiStream_t stream)
{
  if (n <= 0) return;
  submitLaunch(stream, [=]() {
    static std::once_flag once;
    site(once, "leanmd::zeroForces", teamOf(stream, blocksFor(n), 0), ZeroForcesKernel{d_f, n});
  });
}

void invokeAccumulateForces(vec3* d_dst, const vec3* d_src, int n,
                            hapiStream_t stream)
{
  if (n <= 0) return;
  submitLaunch(stream, [=]() {
    static std::once_flag once;
    site(once, "leanmd::accumulateForces", teamOf(stream, blocksFor(n), 0),
         AccumulateForcesKernel{d_dst, d_src, n});
  });
}

void invokeIntegrate(Particle* d_p, const vec3* d_f, int n, double dtVel,
                     double dtPos, double maxVelocity, hapiStream_t stream)
{
  if (n <= 0) return;
  submitLaunch(stream, [=]() {
    static std::once_flag once;
    site(once, "leanmd::integrate", teamOf(stream, blocksFor(n), 0),
         IntegrateKernel{d_p, d_f, n, dtVel, dtPos, maxVelocity});
  });
}

void invokeKineticEnergy(const Particle* d_p, int n, double* d_partial, int nBlocks,
                         hapiStream_t stream)
{
  if (n <= 0) return;
  submitLaunch(stream, [=]() {
    static std::once_flag once;
    site(once, "leanmd::kineticEnergy", teamOf(stream, nBlocks, ONE_ARRAY),
         KineticEnergyKernel{d_p, n, d_partial});
  });
}

void invokeBinParticles(const Particle* d_p, int n, int cellX, int cellY, int cellZ,
                        double cellSizeX, double cellSizeY, double cellSizeZ,
                        int dimX, int dimY, int dimZ, Particle* d_stay,
                        Particle* d_send, int exchCapacity, int* d_counts,
                        hapiStream_t stream)
{
  if (n <= 0) return;
  submitLaunch(stream, [=]() {
    static std::once_flag once;
    site(once, "leanmd::binParticles", teamOf(stream, blocksFor(n), 0),
         BinParticlesKernel{d_p, n, cellX, cellY, cellZ, cellSizeX, cellSizeY, cellSizeZ,
                            dimX, dimY, dimZ, d_stay, d_send, exchCapacity, d_counts});
  });
}

void invokeAppendParticles(Particle* d_dst, int dstOffset, const Particle* d_src,
                           int n, hapiStream_t stream)
{
  if (n <= 0) return;
  submitLaunch(stream, [=]() {
    static std::once_flag once;
    site(once, "leanmd::appendParticles", teamOf(stream, blocksFor(n), 0),
         AppendParticlesKernel{d_dst, dstOffset, d_src, n});
  });
}
