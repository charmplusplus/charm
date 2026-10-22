// Kokkos variant of sph3d.cu: the kernels of sph3d.cu as functors over the same raw pointers, launched on the patch's
// stream through the Kokkos execution-space instance wrapping it (hapi_kokkos.h). Every kernel is a TeamPolicy of
// teams the size of the original block (BLOCK_1D = 128, the forces block, 1024 for the row table), with the league the
// original grid, so the launch shapes are the ones of sph3d.cu; the shared arrays of the reduction kernels are team
// scratch at level 0 (shared memory); the row table's CUB block scan is a Kokkos team scan. The CUB device scan and
// select (host-side library calls on the stream) stay as they are, like cuBLAS in moe. The comments explaining the
// kernels are in sph3d.cu.
//
// Every read the kernels made through a `const T* __restrict__` parameter goes through SPH_LDG here: nvcc turns such
// parameter reads into read-only-cache loads on its own, but a functor's pointer members (and locals copied from them)
// carry no restrict the compiler honours, and without it the forces kernel ran 12% slower (6 of 17 .nc loads left).
// The functors are host+device functions (KOKKOS_INLINE_FUNCTION), so the device-only intrinsics the kernels use
// (__ldg, __float_as_int) have host fallbacks behind SPH_LDG / SPH_F2I for the host compilation pass, which never
// runs them.
// CUB first: converse.h defines ALIGN_BYTES as a macro, which is also the name
// of a member constant in CUB's alignment traits.
#include "hapi_portable.h"
#if CMK_HIP
#include <hipcub/hipcub.hpp>
namespace cubx = hipcub;
#else
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
namespace cubx = cub;
#endif
#include "hapi.h"
#include "sph3d.h"
#include "hapi_kokkos.h"

#ifdef SPH_PEEK_ERRORS
#define SPH_PEEK() hapiCheck(hapiPeekAtLastError())
#else
#define SPH_PEEK() ((void)0)
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>

static inline void sphSubmitMemset(void* d, int v, size_t n, hapiStream_t s) {
  hapiSubmit(s, [=]() { hapiCheck(hapiMemsetAsync(d, v, n, s)); });
}
static inline void sphSubmitMemcpy(void* dst, const void* src, size_t n, hapiMemcpyKind k, hapiStream_t s) {
  hapiSubmit(s, [=]() { hapiCheck(hapiMemcpyAsync(dst, src, n, k, s)); });
}

#define BLOCK_1D 128
static inline int nblocks(int n) { return (n + BLOCK_1D - 1) / BLOCK_1D; }

using hapikokkos::Space;
using hapikokkos::space;
using Team = Kokkos::TeamPolicy<Space>;
using Member = Team::member_type;

// A league of `league` teams of `team` threads on the stream, with `scratchBytes` of shared memory per team.
static inline Team teamOf(hapiStream_t s, int league, int team, size_t scratchBytes = 0) {
  Team p(space(s), league, team);
  if (scratchBytes) p.set_scratch_size(0, Kokkos::PerTeam(scratchBytes));
  return p;
}
template <class P, class F>
static inline void site(std::once_flag& once, const char* label, const P& p, const F& f) {
  hapikokkos::launch(once, [&] { Kokkos::parallel_for(label, p, f); });
}
// A flat 1-D launch over n indices: Kokkos's block-size deduction gives 128-thread blocks for every one of these
// kernels (the smallest block size at full occupancy, ties to the smaller), BLOCK_1D here, and a RangePolicy launch
// carries none of the team member's per-thread setup, which cost the short kernels 10-45%.
using Range = Kokkos::RangePolicy<Space, Kokkos::IndexType<int>>;
static inline Range rangeOf(hapiStream_t s, int n) { return Range(space(s), 0, n); }

// Device-only intrinsics with host fallbacks (see the note at the top).
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define SPH_LDG(p) __ldg(p)
#define SPH_F2I(x) __float_as_int(x)
#define SPH_F2U(x) __float_as_uint(x)
#else
static inline int SPH_F2I(float x) { int i; std::memcpy(&i, &x, 4); return i; }
static inline unsigned SPH_F2U(float x) { unsigned u; std::memcpy(&u, &x, 4); return u; }
#define SPH_LDG(p) (*(p))
#endif

// desul's native CUDA red.add / red.min for 64-bit integers are instantiated for uint64_t (unsigned long here), not
// for unsigned long long, CUDA's own 64-bit atomic type: Kokkos::atomic_add on an unsigned long long fell to a
// compare-and-swap loop (checkKernel 78x slower: 9.4 ms against 0.12). These go through uint64_t.
KOKKOS_FORCEINLINE_FUNCTION void atomicAddU64(unsigned long long* p, unsigned long long v) {
  Kokkos::atomic_add(reinterpret_cast<uint64_t*>(p), static_cast<uint64_t>(v));
}
KOKKOS_FORCEINLINE_FUNCTION void atomicMinU64(unsigned long long* p, unsigned long long v) {
  Kokkos::atomic_min(reinterpret_cast<uint64_t*>(p), static_cast<uint64_t>(v));
}

// ---------------------------------------------------------------- kernel ----
KOKKOS_FORCEINLINE_FUNCTION RealType wendlandDW(RealType q, RealType inv_h) {
  const RealType t = 1.0f - 0.5f * q;
  const RealType t3 = t * t * t;
  const RealType inv_h2 = inv_h * inv_h;
  return -2.0889086f * inv_h2 * inv_h2 * q * t3;
}

// ------------------------------------------------------------ cell lists ----
KOKKOS_FORCEINLINE_FUNCTION int cellOf(RealType px, RealType py, RealType pz,
    RealType x0, RealType y0, RealType z0, RealType inv_csize,
    int ncx, int ncy, int ncz) {
  int cx = (int)floorf((px - x0) * inv_csize) + 1;
  int cy = (int)floorf((py - y0) * inv_csize) + 1;
  int cz = (int)floorf((pz - z0) * inv_csize) + 1;
  cx = min(max(cx, 0), ncx + 1);
  cy = min(max(cy, 0), ncy + 1);
  cz = min(max(cz, 0), ncz + 1);
  return (cz * (ncy + 2) + cy) * (ncx + 2) + cx;
}

struct CellCountKernel {
  const Particle* parts; int n; RealType x0, y0, z0, inv_csize; int ncx, ncy, ncz; int* counts; int* cell_of;
  KOKKOS_INLINE_FUNCTION void operator()(const int kk_i) const {
    const int i = kk_i;
    if (i >= n) return;
    const int c = cellOf(parts[i].x, parts[i].y, parts[i].z, x0, y0, z0, inv_csize, ncx, ncy, ncz);
    cell_of[i] = c;
    Kokkos::atomic_add(&counts[c], 1);
  }
};

struct CellScatterKernel {
  const int* cell_of; int n; int* cursor; int* cell_parts;
  KOKKOS_INLINE_FUNCTION void operator()(const int kk_i) const {
    const int i = kk_i;
    if (i >= n) return;
    cell_parts[Kokkos::atomic_fetch_add(&cursor[cell_of[i]], 1)] = i;
  }
};

struct LayoutKernel {
  const Particle* parts; const int* cell_parts; int n; float4* pos4; float4* vel4; unsigned char* type8;
  KOKKOS_INLINE_FUNCTION void operator()(const int kk_i) const {
    const Particle* __restrict__ pr = parts;
    const int* __restrict__ cp = cell_parts;
    float4* __restrict__ p4 = pos4;
    float4* __restrict__ v4 = vel4;
    unsigned char* __restrict__ t8 = type8;
    const int k = kk_i;
    if (k >= n) return;
    const Particle* q = &pr[SPH_LDG(&cp[k])];
    p4[k] = make_float4(SPH_LDG(&q->x), SPH_LDG(&q->y), SPH_LDG(&q->z), SPH_LDG(&q->rho));
    v4[k] = make_float4(SPH_LDG(&q->vx), SPH_LDG(&q->vy), SPH_LDG(&q->vz), SPH_LDG(&q->p));
    t8[k] = (unsigned char)SPH_LDG(&q->type);
  }
};

struct GatherKernel {
  const Particle* src; const int* cell_parts; int n; Particle* dst;
  KOKKOS_INLINE_FUNCTION void operator()(const int kk_i) const {
    const int k = kk_i;
    if (k >= n) return;
    dst[k] = src[cell_parts[k]];
  }
};

// --------------------------------------------------------------- physics ----
KOKKOS_FORCEINLINE_FUNCTION RealType taitPressure(RealType rho, RealType rho0, RealType c0) {
  const RealType B = rho0 * c0 * c0 / EOS_GAMMA;
  const RealType r = rho / rho0;
  const RealType r2 = r * r, r4 = r2 * r2;
  const RealType pres = B * (r4 * r2 * r - 1.0f);   // r^7
  return pres < 0.0f ? 0.0f : pres;
}

struct EosKernel {
  Particle* parts; int n; RealType rho0, c0; int* zero; int nzero;
  KOKKOS_INLINE_FUNCTION void operator()(const int kk_i) const {
    const int i = kk_i;
    if (i >= n) return;
    if (i == 0 && zero != nullptr) for (int k = 0; k < nzero; k++) zero[k] = 0;
    RealType rho = parts[i].rho;
    if (parts[i].type == PTYPE_BOUND && rho < rho0) rho = rho0;
    parts[i].p = taitPressure(rho, rho0, c0);
    parts[i].rho = rho;
  }
};

// The interior boundary (sph3d.cu: c_interior_lo); set once by sphSetInteriorLayers.
__constant__ int c_interior_lo = 3;
KOKKOS_FORCEINLINE_FUNCTION int interiorLo() {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return c_interior_lo;
#else
  return 3;
#endif
}
KOKKOS_FORCEINLINE_FUNCTION bool cellIsInterior(int cx, int cy, int cz, int ncx, int ncy, int ncz) {
  const int lo = interiorLo(), hi = lo - 1;
  return cx >= lo && cx <= ncx - hi && cy >= lo && cy <= ncy - hi && cz >= lo && cz <= ncz - hi;
}
void sphSetInteriorLayers(int layers) {
  const int lo = layers + 1;
  hapiCheck(hapiMemcpyToSymbol(c_interior_lo, &lo, sizeof(int)));
}

struct TargetState { RealType x, y, z, vx, vy, vz, rho, p; int type; };
struct PairAccum { RealType adrho, aax, aay, aaz; };

KOKKOS_FORCEINLINE_FUNCTION void accumulatePairs(const TargetState& pi, const float4* __restrict__ pos4,
    const float4* __restrict__ vel4, int beg, int end, RealType h, RealType inv_h, RealType support2,
    RealType eta2h2, RealType inv_rhoi2, RealType mass, RealType c0, PairAccum& acc) {
  for (int k = beg; k < end; k++) {
    const float4 pj = SPH_LDG(&pos4[k]);   // x y z rho
    const RealType dx = pi.x - pj.x, dy = pi.y - pj.y, dz = pi.z - pj.z;
    const RealType r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= support2 || r2 < 1e-16f) continue;
    const float4 vj = SPH_LDG(&vel4[k]);   // vx vy vz p
    const RealType rhoj = pj.w, pj_p = vj.w;
    const RealType rinv = rsqrtf(r2);
    const RealType r = r2 * rinv;
    const RealType gw = wendlandDW(r * inv_h, inv_h) * rinv;   // dW/dr / r
    const RealType gwx = gw * dx, gwy = gw * dy, gwz = gw * dz;
    const RealType inv_rhoj = 1.0f / rhoj;
    const RealType inv_r2e = 1.0f / (r2 + eta2h2);
    const RealType dvx = pi.vx - vj.x, dvy = pi.vy - vj.y, dvz = pi.vz - vj.z;
    acc.adrho += mass * (dvx * gwx + dvy * gwy + dvz * gwz);
    const RealType rdotg = -(dx * gwx + dy * gwy + dz * gwz);
    acc.adrho += 2.0f * DELTA_SPH * h * c0 * (mass * inv_rhoj) * (rhoj - pi.rho) * rdotg * inv_r2e;
    if (pi.type == PTYPE_FLUID) {
      const RealType vdotr = dvx * dx + dvy * dy + dvz * dz;
      RealType visc = 0.0f;
      if (vdotr < 0.0f) {
        const RealType mu = h * vdotr * inv_r2e;
        visc = -VISC_ALPHA * c0 * mu * (2.0f / (pi.rho + rhoj));
      }
      const RealType term = pi.p * inv_rhoi2 + pj_p * inv_rhoj * inv_rhoj + visc;
      acc.aax -= mass * term * gwx;
      acc.aay -= mass * term * gwy;
      acc.aaz -= mass * term * gwz;
    }
  }
}

struct FlagInteriorKernel {
  int n, n_local, ncx, ncy, ncz; const int* cell_parts; const int* cell_of; unsigned char* flags;
  KOKKOS_INLINE_FUNCTION void operator()(const int kk_i) const {
    const int* __restrict__ cp = cell_parts;
    const int* __restrict__ co = cell_of;
    const int k = kk_i;
    if (k >= n) return;
    const int i = SPH_LDG(&cp[k]);
    unsigned char f = 0;
    if (i < n_local) {
      const int c = SPH_LDG(&co[i]);
      f = cellIsInterior(c % (ncx + 2), (c / (ncx + 2)) % (ncy + 2), c / ((ncx + 2) * (ncy + 2)), ncx, ncy, ncz) ? 1 : 0;
    }
    flags[k] = f;
  }
};

struct FlagBoundaryTwoListKernel {
  int n_total, n_local, n0, ncx, ncy, ncz; const int* cell_parts; const int* cell_of; const int* cell_parts2;
  const int* cell_of2; unsigned char* flags;
  KOKKOS_INLINE_FUNCTION void operator()(const int kk_i) const {
    const int* __restrict__ cp = cell_parts;
    const int* __restrict__ co = cell_of;
    const int* __restrict__ cp2 = cell_parts2;
    const int* __restrict__ co2 = cell_of2;
    const int k = kk_i;
    if (k >= n_total) return;
    int i, c;
    if (k < n0) { i = SPH_LDG(&cp[k]); c = SPH_LDG(&co[i]); }
    else { const int k2 = k - n0; const int j2 = SPH_LDG(&cp2[k2]); i = n0 + j2; c = SPH_LDG(&co2[j2]); }
    unsigned char f = 0;
    if (i < n_local)
      f = cellIsInterior(c % (ncx + 2), (c / (ncx + 2)) % (ncy + 2), c / ((ncx + 2) * (ncy + 2)), ncx, ncy, ncz) ? 0 : 1;
    flags[k] = f;
  }
};

KOKKOS_FORCEINLINE_FUNCTION int rowOf(const int* __restrict__ pre, int n, int t) {
  int lo = 0, hi = n;   // pre[lo] <= t < pre[hi]
  while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if (SPH_LDG(&pre[mid]) <= t) lo = mid; else hi = mid; }
  return lo;
}

// The row table: one team of 1024 threads, the exclusive prefix of the items' counts by a team scan (CUB's block
// scan in sph3d.cu), chunk by chunk with a carry.
struct RowTableKernel {
  int pass, ncx, ncy, ncz, layers; const int* cell_off; const int* cell_cnt; const int* cell_off2; const int* cell_cnt2;
  int n0; int* item_beg; int* item_pre; int nitems;
  KOKKOS_FORCEINLINE_FUNCTION int cellId(int cx, int cy, int cz) const { return (cz * (ncy + 2) + cy) * (ncx + 2) + cx; }
  KOKKOS_FORCEINLINE_FUNCTION void item(int j, int& beg, int& cnt) const {
    const int lo = layers + 1;
    beg = 0; cnt = 0;
    if (pass == 1) {
      const int ry = ncy - 2 * layers;
      const int cy = lo + j % ry, cz = lo + j / ry;
      const int ca = cellId(lo, cy, cz), cb = cellId(ncx - layers, cy, cz);
      beg = cell_off[ca]; cnt = cell_off[cb] + cell_cnt[cb] - beg;
    } else {
      const int R1 = 2 * ncy * ncz;
      if (j < R1) {
        const int cz = 1 + j / (2 * ncy), rem = j % (2 * ncy);
        const int seg = rem / ncy, cy = 1 + rem % ncy;
        const bool brow = (cy < lo || cy > ncy - layers || cz < lo || cz > ncz - layers);
        int xa, xb;
        if (brow) { xa = 1; xb = seg ? 0 : ncx; }
        else if (seg == 0) { xa = 1; xb = lo - 1; }
        else { xa = ncx - layers + 1; xb = ncx; }
        if (xb >= xa) { const int ca = cellId(xa, cy, cz), cb = cellId(xb, cy, cz); beg = cell_off[ca]; cnt = cell_off[cb] + cell_cnt[cb] - beg; }
      } else {
        const int r = j - R1;
        const int cy = 1 + r % ncy, cz = 1 + r / ncy;
        const int ca = cellId(1, cy, cz), cb = cellId(ncx, cy, cz);
        beg = n0 + cell_off2[ca]; cnt = cell_off2[cb] + cell_cnt2[cb] - cell_off2[ca];
      }
    }
  }
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    const int nt = m.team_size();
    int carry = 0;
    for (int base = 0; base < nitems; base += nt) {
      int total = 0;
      Kokkos::parallel_scan(Kokkos::TeamThreadRange(m, nt), [&](const int t, int& upd, const bool final) {
        const int j = base + t;
        int beg = 0, cnt = 0;
        if (j < nitems) item(j, beg, cnt);
        if (final && j < nitems) { item_beg[j] = beg; item_pre[j] = carry + upd; }
        upd += cnt;
      }, total);
      m.team_barrier();
      carry += total;
    }
    if (m.team_rank() == 0) item_pre[nitems] = carry;
  }
};

struct ForcesTwoListKernel {
  int n_total, n_local, n0, ncx, ncy, ncz; const int* cell_parts; const int* cell_of; const int* cell_off;
  const int* cell_cnt; const int* cell_parts2; const int* cell_of2; const int* cell_off2; const int* cell_cnt2;
  const float4* pos4; const float4* vel4; const unsigned char* type8; RealType h, mass, c0, gravity;
  RealType* drho; RealType* ax; RealType* ay; RealType* az; const int* sel; const int* d_num;
  const int* item_beg; const int* item_pre; int nitems;
  KOKKOS_INLINE_FUNCTION void operator()(const int kk_i) const {
    const int* __restrict__ cp = cell_parts; const int* __restrict__ co = cell_of;
    const int* __restrict__ coff = cell_off; const int* __restrict__ ccnt = cell_cnt;
    const int* __restrict__ cp2 = cell_parts2; const int* __restrict__ co2 = cell_of2;
    const int* __restrict__ coff2 = cell_off2; const int* __restrict__ ccnt2 = cell_cnt2;
    const float4* __restrict__ p4 = pos4; const float4* __restrict__ v4 = vel4;
    const unsigned char* __restrict__ t8 = type8;
    const int* __restrict__ selr = sel; const int* __restrict__ dnum = d_num;
    const int* __restrict__ ibeg = item_beg; const int* __restrict__ ipre = item_pre;
    const int t = kk_i;
    int k;   // the entry: a local target in a boundary cell
    if (ipre != nullptr) { if (t >= SPH_LDG(&ipre[nitems])) return; const int j = rowOf(ipre, nitems, t); k = SPH_LDG(&ibeg[j]) + (t - SPH_LDG(&ipre[j])); }
    else { if (t >= SPH_LDG(dnum)) return; k = SPH_LDG(&selr[t]); }
    int i, ci;
    if (k < n0) { i = SPH_LDG(&cp[k]); ci = SPH_LDG(&co[i]); }
    else { const int k2 = k - n0; const int j2 = SPH_LDG(&cp2[k2]); i = n0 + j2; ci = SPH_LDG(&co2[j2]); }
    if (i >= n_local) return;
    const int cx = ci % (ncx + 2), cy = (ci / (ncx + 2)) % (ncy + 2), cz = ci / ((ncx + 2) * (ncy + 2));
    TargetState pi;
    { const float4 a = SPH_LDG(&p4[k]), b = SPH_LDG(&v4[k]); pi.x = a.x; pi.y = a.y; pi.z = a.z; pi.rho = a.w;
      pi.vx = b.x; pi.vy = b.y; pi.vz = b.z; pi.p = b.w; pi.type = SPH_LDG(&t8[k]); }
    const RealType inv_h = 1.0f / h;
    const RealType support2 = (KERNEL_SUPPORT * h) * (KERNEL_SUPPORT * h);
    const RealType eta2h2 = ETA2 * h * h;
    const RealType inv_rhoi2 = 1.0f / (pi.rho * pi.rho);
    PairAccum acc = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int oz = -1; oz <= 1; oz++) {
      const int zz = cz + oz; if (zz < 0 || zz > ncz + 1) continue;
      for (int oy = -1; oy <= 1; oy++) {
        const int yy = cy + oy; if (yy < 0 || yy > ncy + 1) continue;
        for (int ox = -1; ox <= 1; ox++) {
          const int xx = cx + ox; if (xx < 0 || xx > ncx + 1) continue;
          const int c = (zz * (ncy + 2) + yy) * (ncx + 2) + xx;
#pragma unroll 1
          for (int l = 0; l < 2; l++) {
            const int beg = l ? n0 + SPH_LDG(&coff2[c]) : SPH_LDG(&coff[c]);
            const int cnt = l ? SPH_LDG(&ccnt2[c]) : SPH_LDG(&ccnt[c]);
            if (cnt > 0) accumulatePairs(pi, p4, v4, beg, beg + cnt, h, inv_h, support2, eta2h2, inv_rhoi2, mass, c0, acc);
          }
        }
      }
    }
    drho[i] = acc.adrho;
    if (pi.type == PTYPE_FLUID) { ax[i] = acc.aax; ay[i] = acc.aay; az[i] = acc.aaz - gravity; }
    else { ax[i] = 0.0f; ay[i] = 0.0f; az[i] = 0.0f; }
  }
};

struct MigrantScatterKernel {
  int n_total, n_local, n0, ncx, ncy, ncz; const int* cell_parts; const int* cell_off; const int* cell_cnt;
  const int* cell_parts2; const int* cell_of2; const float4* pos4; const float4* vel4; const unsigned char* type8;
  RealType h, mass, c0; RealType* drho; RealType* ax; RealType* ay; RealType* az;
  KOKKOS_INLINE_FUNCTION void operator()(const int kk_i) const {
    const int* __restrict__ cp = cell_parts; const int* __restrict__ coff = cell_off; const int* __restrict__ ccnt = cell_cnt;
    const int* __restrict__ cp2 = cell_parts2; const int* __restrict__ co2 = cell_of2;
    const float4* __restrict__ p4 = pos4; const float4* __restrict__ v4 = vel4; const unsigned char* __restrict__ t8 = type8;
    const int e = kk_i;
    if (e >= n_total - n0) return;
    const int j = SPH_LDG(&cp2[e]);
    if (n0 + j >= n_local) return;   // a ghost
    const int ci = SPH_LDG(&co2[j]);
    const int cx = ci % (ncx + 2), cy = (ci / (ncx + 2)) % (ncy + 2), cz = ci / ((ncx + 2) * (ncy + 2));
    const int me = n0 + e;
    const RealType inv_h = 1.0f / h;
    const RealType support2 = (KERNEL_SUPPORT * h) * (KERNEL_SUPPORT * h);
    const RealType eta2h2 = ETA2 * h * h;
    for (int oz = -1; oz <= 1; oz++) {
      const int zz = cz + oz; if (zz < 0 || zz > ncz + 1) continue;
      for (int oy = -1; oy <= 1; oy++) {
        const int yy = cy + oy; if (yy < 0 || yy > ncy + 1) continue;
        for (int ox = -1; ox <= 1; ox++) {
          const int xx = cx + ox; if (xx < 0 || xx > ncx + 1) continue;
          if (!cellIsInterior(xx, yy, zz, ncx, ncy, ncz)) continue;
          const int c = (zz * (ncy + 2) + yy) * (ncx + 2) + xx;
          const int beg = SPH_LDG(&coff[c]), end = beg + SPH_LDG(&ccnt[c]);
          for (int q = beg; q < end; q++) {
            TargetState pi;
            { const float4 a = SPH_LDG(&p4[q]), b = SPH_LDG(&v4[q]); pi.x = a.x; pi.y = a.y; pi.z = a.z; pi.rho = a.w;
              pi.vx = b.x; pi.vy = b.y; pi.vz = b.z; pi.p = b.w; pi.type = SPH_LDG(&t8[q]); }
            const RealType inv_rhoi2 = 1.0f / (pi.rho * pi.rho);
            PairAccum acc = {0.0f, 0.0f, 0.0f, 0.0f};
            accumulatePairs(pi, p4, v4, me, me + 1, h, inv_h, support2, eta2h2, inv_rhoi2, mass, c0, acc);
            if (acc.adrho == 0.0f && acc.aax == 0.0f && acc.aay == 0.0f && acc.aaz == 0.0f) continue;
            const int p = SPH_LDG(&cp[q]);
            Kokkos::atomic_add(&drho[p], acc.adrho);
            if (pi.type == PTYPE_FLUID) { Kokkos::atomic_add(&ax[p], acc.aax); Kokkos::atomic_add(&ay[p], acc.aay); Kokkos::atomic_add(&az[p], acc.aaz); }
          }
        }
      }
    }
  }
};

struct ForcesKernel {
  int n_total, n_local, ncx, ncy, ncz; const int* cell_parts; const int* cell_of; const int* cell_off; const int* cell_cnt;
  const float4* pos4; const float4* vel4; const unsigned char* type8; RealType h, mass, c0, gravity;
  RealType* drho; RealType* ax; RealType* ay; RealType* az; int mode; const int* sel; const int* d_num;
  const int* item_beg; const int* item_pre; int nitems;
  KOKKOS_INLINE_FUNCTION void operator()(const int kk_i) const {
    const int* __restrict__ cp = cell_parts; const int* __restrict__ co = cell_of;
    const int* __restrict__ coff = cell_off; const int* __restrict__ ccnt = cell_cnt;
    const float4* __restrict__ p4 = pos4; const float4* __restrict__ v4 = vel4;
    const unsigned char* __restrict__ t8 = type8;
    const int* __restrict__ selr = sel; const int* __restrict__ dnum = d_num;
    const int* __restrict__ ibeg = item_beg; const int* __restrict__ ipre = item_pre;
    int k = kk_i;
    if (ipre != nullptr) { if (k >= SPH_LDG(&ipre[nitems])) return; const int j = rowOf(ipre, nitems, k); k = SPH_LDG(&ibeg[j]) + (k - SPH_LDG(&ipre[j])); }
    else if (selr != nullptr) { if (k >= SPH_LDG(dnum)) return; k = SPH_LDG(&selr[k]); }
    else if (k >= n_total) return;
    const int i = SPH_LDG(&cp[k]);
    if (i >= n_local) return;
    if (mode != 0 && selr == nullptr && ipre == nullptr) {
      const int c0i = SPH_LDG(&co[i]);
      const int ccx = c0i % (ncx + 2), ccy = (c0i / (ncx + 2)) % (ncy + 2), ccz = c0i / ((ncx + 2) * (ncy + 2));
      const bool inner = cellIsInterior(ccx, ccy, ccz, ncx, ncy, ncz);
      if ((mode == 1) != inner) return;
    }

    struct { RealType x, y, z, vx, vy, vz, rho, p; int type; } pi;
    {
      const float4 a = SPH_LDG(&p4[k]), b = SPH_LDG(&v4[k]);
      pi.x = a.x; pi.y = a.y; pi.z = a.z; pi.rho = a.w;
      pi.vx = b.x; pi.vy = b.y; pi.vz = b.z; pi.p = b.w;
      pi.type = SPH_LDG(&t8[k]);
    }
    const RealType inv_h = 1.0f / h;
    const RealType support2 = (KERNEL_SUPPORT * h) * (KERNEL_SUPPORT * h);
    const RealType eta2h2 = ETA2 * h * h;
    const RealType inv_rhoi2 = 1.0f / (pi.rho * pi.rho);

    RealType adrho = 0.0f, aax = 0.0f, aay = 0.0f, aaz = 0.0f;

    const int ci = SPH_LDG(&co[i]);
    const int cx = ci % (ncx + 2);
    const int cy = (ci / (ncx + 2)) % (ncy + 2);
    const int cz = ci / ((ncx + 2) * (ncy + 2));

    for (int oz = -1; oz <= 1; oz++) {
      const int zz = cz + oz;
      if (zz < 0 || zz > ncz + 1) continue;
      for (int oy = -1; oy <= 1; oy++) {
        const int yy = cy + oy;
        if (yy < 0 || yy > ncy + 1) continue;
        for (int ox = -1; ox <= 1; ox++) {
          const int xx = cx + ox;
          if (xx < 0 || xx > ncx + 1) continue;
          const int c = (zz * (ncy + 2) + yy) * (ncx + 2) + xx;
          const int beg = SPH_LDG(&coff[c]), end = beg + SPH_LDG(&ccnt[c]);
          for (int kk = beg; kk < end; kk++) {
            const float4 pj = SPH_LDG(&p4[kk]);   // x y z rho
            const RealType dx = pi.x - pj.x, dy = pi.y - pj.y, dz = pi.z - pj.z;
            const RealType r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= support2 || r2 < 1e-16f) continue;
            const float4 vj = SPH_LDG(&v4[kk]);   // vx vy vz p
            const RealType rhoj = pj.w, pj_p = vj.w;

            const RealType rinv = rsqrtf(r2);
            const RealType r = r2 * rinv;
            const RealType gw = wendlandDW(r * inv_h, inv_h) * rinv;   // dW/dr / r
            const RealType gwx = gw * dx, gwy = gw * dy, gwz = gw * dz;
            const RealType inv_rhoj = 1.0f / rhoj;
            const RealType inv_r2e = 1.0f / (r2 + eta2h2);

            const RealType dvx = pi.vx - vj.x, dvy = pi.vy - vj.y, dvz = pi.vz - vj.z;

            adrho += mass * (dvx * gwx + dvy * gwy + dvz * gwz);

            const RealType rdotg = -(dx * gwx + dy * gwy + dz * gwz);
            adrho += 2.0f * DELTA_SPH * h * c0 * (mass * inv_rhoj) *
                     (rhoj - pi.rho) * rdotg * inv_r2e;

            if (pi.type == PTYPE_FLUID) {
              const RealType vdotr = dvx * dx + dvy * dy + dvz * dz;
              RealType visc = 0.0f;
              if (vdotr < 0.0f) {
                const RealType mu = h * vdotr * inv_r2e;
                visc = -VISC_ALPHA * c0 * mu * (2.0f / (pi.rho + rhoj));
              }
              const RealType term =
                  pi.p * inv_rhoi2 + pj_p * inv_rhoj * inv_rhoj + visc;
              aax -= mass * term * gwx;
              aay -= mass * term * gwy;
              aaz -= mass * term * gwz;
            }
          }
        }
      }
    }

    drho[i] = adrho;
    if (pi.type == PTYPE_FLUID) {
      ax[i] = aax;
      ay[i] = aay;
      az[i] = aaz - gravity;
    } else {
      ax[i] = 0.0f;
      ay[i] = 0.0f;
      az[i] = 0.0f;
    }
  }
};

KOKKOS_FORCEINLINE_FUNCTION void ecStep(Particle& q, int i, RealType dt, RealType rho0,
    RealType c0, const RealType* drho, const RealType* ax, const RealType* ay,
    const RealType* az) {
  q.rho += dt * drho[i];
  if (q.type == PTYPE_BOUND) {
    if (q.rho < rho0) q.rho = rho0;
  } else {
    q.vx += dt * ax[i];
    q.vy += dt * ay[i];
    q.vz += dt * az[i];
    q.x += dt * q.vx;
    q.y += dt * q.vy;
    q.z += dt * q.vz;
  }
  q.p = taitPressure(q.rho, rho0, c0);
}

KOKKOS_FORCEINLINE_FUNCTION void pcPredict(Particle& q, int i, RealType dt2, RealType rho0,
    RealType c0, const RealType* drho, const RealType* ax, const RealType* ay,
    const RealType* az) {
  q.rho += dt2 * drho[i];
  if (q.type == PTYPE_BOUND) {
    if (q.rho < rho0) q.rho = rho0;
  } else {
    q.x += dt2 * q.vx;
    q.y += dt2 * q.vy;
    q.z += dt2 * q.vz;
    q.vx += dt2 * ax[i];
    q.vy += dt2 * ay[i];
    q.vz += dt2 * az[i];
  }
  q.p = taitPressure(q.rho, rho0, c0);
}
KOKKOS_FORCEINLINE_FUNCTION void pcCorrect(Particle& q, int i, RealType dt, RealType rho0,
    RealType c0, const RealType* drho, const RealType* ax, const RealType* ay,
    const RealType* az) {
  q.rho += dt * drho[i];
  if (q.type == PTYPE_BOUND) {
    if (q.rho < rho0) q.rho = rho0;
  } else {
    const RealType hdt = 0.5f * dt;
    q.x += dt * (q.vx + hdt * ax[i]);
    q.y += dt * (q.vy + hdt * ay[i]);
    q.z += dt * (q.vz + hdt * az[i]);
    q.vx += dt * ax[i];
    q.vy += dt * ay[i];
    q.vz += dt * az[i];
  }
  q.p = taitPressure(q.rho, rho0, c0);
}

KOKKOS_FORCEINLINE_FUNCTION void placeParticle(const Particle& p, RealType x0,
    RealType y0, RealType z0, RealType x1, RealType y1, RealType z1,
    Particle** bufs, Particle* stay, int* counts, int cap_face) {
  int sx = 0, sy = 0, sz = 0;
  if (p.type == PTYPE_FLUID) {
    if (p.x < x0) sx = -1; else if (p.x >= x1) sx = 1;
    if (p.y < y0) sy = -1; else if (p.y >= y1) sy = 1;
    if (p.z < z0) sz = -1; else if (p.z >= z1) sz = 1;
  }
  const int d = dirIndex(sx, sy, sz);
  if (d == STAY) {
    stay[Kokkos::atomic_fetch_add(&counts[STAY], 1)] = p;
    return;
  }
  const int slot = Kokkos::atomic_fetch_add(&counts[d], 1);
  if (slot < dirCap(d, cap_face)) bufs[d][slot] = p;
  else Kokkos::atomic_add(&counts[ERR_COUNTER], 1);
}

struct IntegrateKernel {
  Particle* parts; int n_local; RealType dt, rho0, c0; const RealType* drho; const RealType* ax; const RealType* ay;
  const RealType* az; int* zero; int nzero;
  KOKKOS_INLINE_FUNCTION void operator()(const int kk_i) const {
    const int i = kk_i;
    if (i >= n_local) return;
    if (i == 0 && zero != nullptr) for (int k = 0; k < nzero; k++) zero[k] = 0;
    Particle q = parts[i];
    ecStep(q, i, dt, rho0, c0, drho, ax, ay, az);
    parts[i] = q;
  }
};

struct IntegrateLeaversKernel {
  const Particle* parts; int n_local; RealType dt, rho0, c0; const RealType* drho; const RealType* ax;
  const RealType* ay; const RealType* az; RealType x0, y0, z0, x1, y1, z1; Particle** bufs; Particle* stay; int* counts;
  int cap_face;
  KOKKOS_INLINE_FUNCTION void operator()(const int kk_i) const {
    const int i = kk_i;
    if (i >= n_local) return;
    Particle q = parts[i];
    ecStep(q, i, dt, rho0, c0, drho, ax, ay, az);
    placeParticle(q, x0, y0, z0, x1, y1, z1, bufs, stay, counts, cap_face);
  }
};

struct PredictKernel {
  const Particle* src; Particle* dst; int n_local; RealType dt2, rho0, c0; const RealType* drho; const RealType* ax;
  const RealType* ay; const RealType* az;
  KOKKOS_INLINE_FUNCTION void operator()(const int kk_i) const {
    const int i = kk_i;
    if (i >= n_local) return;
    Particle q = src[i];
    pcPredict(q, i, dt2, rho0, c0, drho, ax, ay, az);
    dst[i] = q;
  }
};

struct CorrectKernel {
  Particle* parts; int n_local; RealType dt, rho0, c0; const RealType* drho; const RealType* ax; const RealType* ay;
  const RealType* az; int* zero; int nzero;
  KOKKOS_INLINE_FUNCTION void operator()(const int kk_i) const {
    const int i = kk_i;
    if (i >= n_local) return;
    if (i == 0 && zero != nullptr) for (int k = 0; k < nzero; k++) zero[k] = 0;
    Particle q = parts[i];
    pcCorrect(q, i, dt, rho0, c0, drho, ax, ay, az);
    parts[i] = q;
  }
};

struct CorrectLeaversKernel {
  const Particle* parts; int n_local; RealType dt, rho0, c0; const RealType* drho; const RealType* ax;
  const RealType* ay; const RealType* az; RealType x0, y0, z0, x1, y1, z1; Particle** bufs; Particle* stay; int* counts;
  int cap_face;
  KOKKOS_INLINE_FUNCTION void operator()(const int kk_i) const {
    const int i = kk_i;
    if (i >= n_local) return;
    Particle q = parts[i];
    pcCorrect(q, i, dt, rho0, c0, drho, ax, ay, az);
    placeParticle(q, x0, y0, z0, x1, y1, z1, bufs, stay, counts, cap_face);
  }
};

// ------------------------------------------------------------- exchanges ----
struct PackHaloKernel {
  const Particle* parts; int n_local; RealType x0, y0, z0, x1, y1, z1, support; Particle** bufs; int** idx; int* counts;
  int cap_face; int valid_mask;
  KOKKOS_INLINE_FUNCTION void operator()(const int kk_i) const {
    const int i = kk_i;
    if (i >= n_local) return;
    const Particle p = parts[i];
    const bool nx0 = (p.x - x0) < support, nx1 = (x1 - p.x) < support;
    const bool ny0 = (p.y - y0) < support, ny1 = (y1 - p.y) < support;
    const bool nz0 = (p.z - z0) < support, nz1 = (z1 - p.z) < support;
    if (!(nx0 | nx1 | ny0 | ny1 | nz0 | nz1)) return;

    // nvcc unrolls this loop by two in the kernel (two copies of each atomic in its SASS) and not inside the Kokkos
    // range loop; without the pragma the functor ran 47% slower (211 vs 144 us).
#pragma unroll 2
    for (int d = 0; d < NUM_DIRS; d++) {
      if (!((valid_mask >> d) & 1)) continue;
      const int sx = dirSX(d), sy = dirSY(d), sz = dirSZ(d);
      const bool okx = sx == 0 || (sx < 0 ? nx0 : nx1);
      const bool oky = sy == 0 || (sy < 0 ? ny0 : ny1);
      const bool okz = sz == 0 || (sz < 0 ? nz0 : nz1);
      if (!(okx && oky && okz)) continue;
      const int slot = Kokkos::atomic_fetch_add(&counts[d], 1);
      if (slot < dirCap(d, cap_face)) { bufs[d][slot] = p; idx[d][slot] = i; }
      else Kokkos::atomic_add(&counts[ERR_COUNTER], 1);
    }
  }
};

struct PackHaloByIndexKernel {
  const Particle* parts; Particle** bufs; const int* idx; const int* halo_cnt; int cap_face; int nslots;
  KOKKOS_INLINE_FUNCTION void operator()(const int kk_i) const {
    const int t = kk_i;
    if (t >= nslots) return;
    int off = 0, d;
    for (d = 0; d < NUM_DIRS; d++) {
      if (d == STAY) continue;
      const int cap = dirCap(d, cap_face);
      if (t < off + cap) break;
      off += cap;
    }
    if (d >= NUM_DIRS) return;
    const int k = t - off;
    if (k >= halo_cnt[d]) return;
    bufs[d][k] = parts[idx[off + k]];
  }
};

struct GatherLeaverGhostsKernel {
  Particle** bufs; const int* counts; Particle* parts; Particle* side; int cap_face; int nslots; int valid_mask;
  KOKKOS_INLINE_FUNCTION void operator()(const int kk_i) const {
    const int t = kk_i;
    if (t >= nslots) return;
    int off = 0, d;
    for (d = 0; d < NUM_DIRS; d++) {
      if (d == STAY) continue;
      const int cap = dirCap(d, cap_face);
      if (t < off + cap) break;
      off += cap;
    }
    if (d >= NUM_DIRS || !((valid_mask >> d) & 1)) return;
    const int k = t - off;
    if (k >= counts[d]) return;
    int prefix = 0;
    for (int e = 0; e < d; e++) if (e != STAY && ((valid_mask >> e) & 1)) prefix += counts[e];
    const Particle q = bufs[d][k];
    parts[counts[STAY] + prefix + k] = q;
    side[prefix + k] = q;
  }
};

struct PackHaloConcurrentKernel {
  const Particle* parts; const int* mig_counts; int valid_mask_mig; RealType x0, y0, z0, x1, y1, z1, support;
  Particle** bufs; int** idx; int* counts; int cap_face; int valid_mask;
  KOKKOS_INLINE_FUNCTION void operator()(const int kk_i) const {
    const int i = kk_i;
    int n = mig_counts[STAY];
    for (int e = 0; e < NUM_DIRS; e++) if (e != STAY && ((valid_mask_mig >> e) & 1)) n += mig_counts[e];
    if (i >= n) return;
    const Particle p = parts[i];
    const bool nx0 = (p.x - x0) < support, nx1 = (x1 - p.x) < support;
    const bool ny0 = (p.y - y0) < support, ny1 = (y1 - p.y) < support;
    const bool nz0 = (p.z - z0) < support, nz1 = (z1 - p.z) < support;
    if (!(nx0 | nx1 | ny0 | ny1 | nz0 | nz1)) return;
    int sx = 0, sy = 0, sz = 0;
    if (p.x < x0) sx = -1; else if (p.x >= x1) sx = 1;
    if (p.y < y0) sy = -1; else if (p.y >= y1) sy = 1;
    if (p.z < z0) sz = -1; else if (p.z >= z1) sz = 1;
    const int owner = dirIndex(sx, sy, sz);
#pragma unroll 2
    for (int d = 0; d < NUM_DIRS; d++) {
      if (!((valid_mask >> d) & 1) || d == owner) continue;
      const int dx = dirSX(d), dy = dirSY(d), dz = dirSZ(d);
      const bool okx = dx == 0 || (dx < 0 ? nx0 : nx1);
      const bool oky = dy == 0 || (dy < 0 ? ny0 : ny1);
      const bool okz = dz == 0 || (dz < 0 ? nz0 : nz1);
      if (!(okx && oky && okz)) continue;
      const int slot = Kokkos::atomic_fetch_add(&counts[d], 1);
      if (slot < dirCap(d, cap_face)) { bufs[d][slot] = p; idx[d][slot] = i; }
      else Kokkos::atomic_add(&counts[ERR_COUNTER], 1);
    }
  }
};

struct MarkLeaversKernel {
  const Particle* parts; int n_local; RealType x0, y0, z0, x1, y1, z1; Particle** bufs; Particle* stay; int* counts;
  int cap_face;
  KOKKOS_INLINE_FUNCTION void operator()(const int kk_i) const {
    const int i = kk_i;
    if (i >= n_local) return;
    placeParticle(parts[i], x0, y0, z0, x1, y1, z1, bufs, stay, counts, cap_face);
  }
};

#define ID_AUDIT_BUCKETS 4096
struct IdBucketKernel {
  const Particle* parts; int n_local; unsigned long long* cnt; unsigned long long* sum; unsigned long long* sum2;
  KOKKOS_INLINE_FUNCTION void operator()(const int kk_i) const {
    const int i = kk_i;
    if (i >= n_local) return;
    const unsigned long long id = (unsigned long long)(unsigned)parts[i].id;
    const int b = (int)(id & (ID_AUDIT_BUCKETS - 1));
    atomicAddU64(&cnt[b], 1ull); atomicAddU64(&sum[b], id); atomicAddU64(&sum2[b], id * id);
  }
};
struct FindIdKernel {
  const Particle* parts; int n; int id; int* out;
  KOKKOS_INLINE_FUNCTION void operator()(const int kk_i) const {
    const int i = kk_i;
    if (i >= n) return;
    if (parts[i].id == id) Kokkos::atomic_min(out, i);
  }
};
void invokeIdBuckets(const Particle* d_parts, int n_local, unsigned long long* d_cnt,
    unsigned long long* d_sum, unsigned long long* d_sum2, hapiStream_t s) {
  sphSubmitMemset(d_cnt, 0, sizeof(unsigned long long) * ID_AUDIT_BUCKETS, s);
  sphSubmitMemset(d_sum, 0, sizeof(unsigned long long) * ID_AUDIT_BUCKETS, s);
  sphSubmitMemset(d_sum2, 0, sizeof(unsigned long long) * ID_AUDIT_BUCKETS, s);
  if (n_local <= 0) return;
  hapiSubmit(s, [=]() {
    static std::once_flag once;
    site(once, "sph3d::idBucket", rangeOf(s, n_local), IdBucketKernel{d_parts, n_local, d_cnt, d_sum, d_sum2});
  });
  SPH_PEEK();
}
void invokeFindId(const Particle* d_parts, int n, int id, int* d_out, hapiStream_t s) {
  sphSubmitMemset(d_out, 0x7f, sizeof(int), s);
  if (n <= 0) return;
  hapiSubmit(s, [=]() {
    static std::once_flag once;
    site(once, "sph3d::findId", rangeOf(s, n), FindIdKernel{d_parts, n, id, d_out});
  });
  SPH_PEEK();
}

// Diagnostics (statsKernel): five shared arrays of BLOCK_1D RealType in team scratch.
struct StatsKernel {
  const Particle* parts; int n_local; RealType* out;
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    RealType* s_rho = (RealType*)m.team_shmem().get_shmem(sizeof(RealType) * BLOCK_1D);
    RealType* s_ke = (RealType*)m.team_shmem().get_shmem(sizeof(RealType) * BLOCK_1D);
    RealType* s_vmax = (RealType*)m.team_shmem().get_shmem(sizeof(RealType) * BLOCK_1D);
    RealType* s_xmax = (RealType*)m.team_shmem().get_shmem(sizeof(RealType) * BLOCK_1D);
    RealType* s_n = (RealType*)m.team_shmem().get_shmem(sizeof(RealType) * BLOCK_1D);
    const int t = m.team_rank();
    const int i = m.league_rank() * m.team_size() + t;

    RealType rho = 0, ke = 0, vmax = 0, xmax = -1e30f, cnt = 0;
    if (i < n_local && parts[i].type == PTYPE_FLUID) {
      const Particle p = parts[i];
      const RealType v2 = p.vx * p.vx + p.vy * p.vy + p.vz * p.vz;
      rho = p.rho;
      ke = 0.5f * v2;
      vmax = sqrtf(v2);
      xmax = p.x;
      cnt = 1;
    }
    s_rho[t] = rho; s_ke[t] = ke; s_vmax[t] = vmax; s_xmax[t] = xmax; s_n[t] = cnt;
    m.team_barrier();
    for (int s = m.team_size() / 2; s > 0; s >>= 1) {
      if (t < s) {
        s_rho[t] += s_rho[t + s];
        s_ke[t] += s_ke[t + s];
        s_n[t] += s_n[t + s];
        s_vmax[t] = fmaxf(s_vmax[t], s_vmax[t + s]);
        s_xmax[t] = fmaxf(s_xmax[t], s_xmax[t + s]);
      }
      m.team_barrier();
    }
    if (t == 0) {
      Kokkos::atomic_add(&out[0], s_rho[0]);
      Kokkos::atomic_add(&out[1], s_ke[0]);
      Kokkos::atomic_add(&out[2], s_n[0]);
      Kokkos::atomic_max((int*)&out[3], SPH_F2I(fmaxf(s_vmax[0], 0.0f)));
      Kokkos::atomic_max((int*)&out[4], SPH_F2I(fmaxf(s_xmax[0], 0.0f)));
    }
  }
};

// ---------------------------------------------------------------- checks ----
KOKKOS_FORCEINLINE_FUNCTION unsigned long long mix64(unsigned long long z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

struct CheckKernel {
  const Particle* parts; int n_local; RealType x0, y0, z0, x1, y1, z1, tol, rho0, c0; unsigned long long* out;
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    unsigned long long* s_id = (unsigned long long*)m.team_shmem().get_shmem(sizeof(unsigned long long) * BLOCK_1D);
    unsigned long long* s_bnd = (unsigned long long*)m.team_shmem().get_shmem(sizeof(unsigned long long) * BLOCK_1D);
    unsigned long long* s_nf = (unsigned long long*)m.team_shmem().get_shmem(sizeof(unsigned long long) * BLOCK_1D);
    unsigned long long* s_nb = (unsigned long long*)m.team_shmem().get_shmem(sizeof(unsigned long long) * BLOCK_1D);
    const int t = m.team_rank();
    const int i = m.league_rank() * m.team_size() + t;

    unsigned long long idh = 0, bndh = 0, nf = 0, nb = 0;
    unsigned long long bad = 0;

    if (i < n_local) {
      const Particle p = parts[i];
      const unsigned long long id = (unsigned long long)(unsigned int)p.id;
      idh = mix64(id + 1);

      if (p.type == PTYPE_BOUND) {
        nb = 1;
        bndh = mix64(id + 1) ^
               mix64(((unsigned long long)SPH_F2U(p.x) << 32) |
                     (unsigned long long)SPH_F2U(p.y)) ^
               mix64(((unsigned long long)SPH_F2U(p.z) << 32) | (id + 7));
      } else if (p.type == PTYPE_FLUID) {
        nf = 1;
      } else {
        bad |= CHK_BAD_TYPE;
      }

      if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z) || !isfinite(p.vx) ||
          !isfinite(p.vy) || !isfinite(p.vz) || !isfinite(p.rho) || !isfinite(p.p)) {
        bad |= CHK_BAD_NAN;
      } else {
        if (p.rho < 0.2f * rho0 || p.rho > 5.0f * rho0) bad |= CHK_BAD_RHO;
        const RealType v2 = p.vx * p.vx + p.vy * p.vy + p.vz * p.vz;
        if (v2 > (0.5f * c0) * (0.5f * c0)) bad |= CHK_BAD_SPEED;
        if (p.x < x0 - tol || p.x >= x1 + tol || p.y < y0 - tol || p.y >= y1 + tol ||
            p.z < z0 - tol || p.z >= z1 + tol)
          bad |= CHK_BAD_OUTSIDE;
      }

      if (bad) {
        Kokkos::atomic_fetch_or(&out[CHK_BAD_MASK], bad);
        atomicAddU64(&out[CHK_BAD_COUNT], 1ull);
      }
    }

    s_id[t] = idh; s_bnd[t] = bndh; s_nf[t] = nf; s_nb[t] = nb;
    m.team_barrier();
    for (int s = m.team_size() / 2; s > 0; s >>= 1) {
      if (t < s) {
        s_id[t] += s_id[t + s];
        s_bnd[t] += s_bnd[t + s];
        s_nf[t] += s_nf[t + s];
        s_nb[t] += s_nb[t + s];
      }
      m.team_barrier();
    }
    if (t == 0) {
      atomicAddU64(&out[CHK_ID_SUM], s_id[0]);
      atomicAddU64(&out[CHK_BND_SUM], s_bnd[0]);
      atomicAddU64(&out[CHK_N_FLUID], s_nf[0]);
      atomicAddU64(&out[CHK_N_BOUND], s_nb[0]);
    }
  }
};

struct GhostCheckKernel {
  const Particle* parts; int n_local; int n_ghost; unsigned long long* out;
  KOKKOS_INLINE_FUNCTION void operator()(const int kk_i) const {
    const int i = kk_i;
    if (i >= n_ghost) return;
    const Particle p = parts[n_local + i];
    if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z) || !isfinite(p.rho) ||
        !isfinite(p.p) || !isfinite(p.vx) || !isfinite(p.vy) || !isfinite(p.vz)) {
      atomicAddU64(&out[6], 1ull);
      atomicMinU64(&out[7], (unsigned long long)i);
    }
  }
};

struct IotaKernel {
  int n; int* d;
  KOKKOS_INLINE_FUNCTION void operator()(const int kk_i) const {
    const int k = kk_i;
    if (k < n) d[k] = k;
  }
};

// ---------------------------------------------------------------- launch ----
size_t sphScanTempBytes(int ncells) {
  size_t bytes = 0;
  int* dummy = NULL;
  hapiCheck(cubx::DeviceScan::ExclusiveSum(NULL, bytes, dummy, dummy, ncells));
  return bytes;
}

void invokeCellBuild(const Particle* d_parts, int n, RealType x0, RealType y0,
    RealType z0, RealType inv_csize, int ncx, int ncy, int ncz, int ncells,
    int* d_cnt, int* d_off, int* d_cursor, int* d_cell_parts, int* d_cell_of,
    void* d_scan_tmp, size_t scan_tmp_bytes, hapiStream_t s) {
  sphSubmitMemset(d_cnt, 0, sizeof(int) * ncells, s);
  if (n > 0)
    hapiSubmit(s, [=]() {
      static std::once_flag once;
      site(once, "sph3d::cellCount", rangeOf(s, n),
           CellCountKernel{d_parts, n, x0, y0, z0, inv_csize, ncx, ncy, ncz, d_cnt, d_cell_of});
    });
  hapiSubmit(s, [=]() {
    size_t bytes = scan_tmp_bytes;
    hapiCheck(cubx::DeviceScan::ExclusiveSum(d_scan_tmp, bytes, d_cnt, d_off, ncells, s));
    hapiCheck(hapiMemcpyAsync(d_cursor, d_off, sizeof(int) * ncells,
        hapiMemcpyDeviceToDevice, s));
  });
  if (n > 0)
    hapiSubmit(s, [=]() {
      static std::once_flag once;
      site(once, "sph3d::cellScatter", rangeOf(s, n),
           CellScatterKernel{d_cell_of, n, d_cursor, d_cell_parts});
    });
  SPH_PEEK();
}

void invokeGather(const Particle* d_src, const int* d_cell_parts, int n,
    Particle* d_dst, hapiStream_t s) {
  if (n <= 0) return;
  hapiSubmit(s, [=]() {
    static std::once_flag once;
    site(once, "sph3d::gather", rangeOf(s, n), GatherKernel{d_src, d_cell_parts, n, d_dst});
  });
  SPH_PEEK();
}

void invokeEOS(Particle* d_parts, int n, RealType rho0, RealType c0,
    int* d_zero, hapiStream_t s) {
  if (n <= 0) return;
  hapiSubmit(s, [=]() {
    static std::once_flag once;
    site(once, "sph3d::eos", rangeOf(s, n), EosKernel{d_parts, n, rho0, c0, d_zero, NUM_COUNTERS});
  });
  SPH_PEEK();
}

void invokeLayout(const Particle* d_parts, const int* d_cell_parts, int n,
    float4* d_pos4, float4* d_vel4, unsigned char* d_type8, hapiStream_t s) {
  if (n <= 0) return;
  hapiSubmit(s, [=]() {
    static std::once_flag once;
    site(once, "sph3d::layout", rangeOf(s, n),
         LayoutKernel{d_parts, d_cell_parts, n, d_pos4, d_vel4, d_type8});
  });
  SPH_PEEK();
}

void invokeForces(int mode, int n_total, int n_local, int ncx, int ncy, int ncz,
    const int* d_cell_parts, const int* d_cell_of, const int* d_off,
    const int* d_cnt, const float4* d_pos4, const float4* d_vel4,
    const unsigned char* d_type8, RealType h, RealType mass, RealType c0,
    RealType gravity, RealType* d_drho, RealType* d_ax, RealType* d_ay,
    RealType* d_az, hapiStream_t s) {
  if (n_local <= 0 || n_total <= 0) return;
  hapiSubmit(s, [=]() {
    static std::once_flag once;
    site(once, "sph3d::forces", rangeOf(s, n_total),
         ForcesKernel{n_total, n_local, ncx, ncy, ncz, d_cell_parts, d_cell_of, d_off, d_cnt, d_pos4, d_vel4, d_type8,
                      h, mass, c0, gravity, d_drho, d_ax, d_ay, d_az, mode, (const int*)nullptr, (const int*)nullptr,
                      (const int*)nullptr, (const int*)nullptr, 0});
  });
  SPH_PEEK();
}

size_t sphSelectTempBytes(int n) {
  size_t bytes = 0;
  const int* it = NULL; unsigned char* f = NULL; int* o = NULL; int* c = NULL;
  hapiCheck(cubx::DeviceSelect::Flagged(NULL, bytes, it, f, o, c, n));
  return bytes;
}
void sphIota(int n, int* d, hapiStream_t s) {
  if (n <= 0) return;
  hapiSubmit(s, [=]() {
    static std::once_flag once;
    site(once, "sph3d::iota", rangeOf(s, n), IotaKernel{n, d});
  });
  SPH_PEEK();
}

void invokeForcesInteriorSel(int n, int ncx, int ncy, int ncz, const int* d_cell_parts, const int* d_cell_of,
    const int* d_off, const int* d_cnt, const float4* d_pos4, const float4* d_vel4, const unsigned char* d_type8,
    RealType h, RealType mass, RealType c0, RealType gravity, RealType* d_drho, RealType* d_ax, RealType* d_ay,
    RealType* d_az, const int* d_iota, unsigned char* d_flags, int* d_sel, int* d_num, void* d_tmp,
    size_t tmp_bytes, hapiStream_t s) {
  if (n <= 0) return;
  hapiSubmit(s, [=]() {
    static std::once_flag o1, o2;
    site(o1, "sph3d::flagInterior", rangeOf(s, n),
         FlagInteriorKernel{n, n, ncx, ncy, ncz, d_cell_parts, d_cell_of, d_flags});
    size_t bytes = tmp_bytes;
    hapiCheck(cubx::DeviceSelect::Flagged(d_tmp, bytes, d_iota, d_flags, d_sel, d_num, n, s));
    site(o2, "sph3d::forcesInteriorSel", rangeOf(s, n),
         ForcesKernel{n, n, ncx, ncy, ncz, d_cell_parts, d_cell_of, d_off, d_cnt, d_pos4, d_vel4, d_type8, h, mass, c0,
                      gravity, d_drho, d_ax, d_ay, d_az, 1, d_sel, d_num, (const int*)nullptr, (const int*)nullptr, 0});
  });
  SPH_PEEK();
}

void invokeForcesInteriorRows(int n, int ncx, int ncy, int ncz, int layers, const int* d_cell_parts,
    const int* d_cell_of, const int* d_off, const int* d_cnt, const float4* d_pos4, const float4* d_vel4,
    const unsigned char* d_type8, RealType h, RealType mass, RealType c0, RealType gravity, RealType* d_drho,
    RealType* d_ax, RealType* d_ay, RealType* d_az, int* d_item_beg, int* d_item_pre, hapiStream_t s) {
  const int nitems = (ncy - 2 * layers) * (ncz - 2 * layers);
  if (n <= 0 || nitems <= 0) return;
  hapiSubmit(s, [=]() {
    static std::once_flag o1, o2;
    site(o1, "sph3d::rowTable1", teamOf(s, 1, 1024),
         RowTableKernel{1, ncx, ncy, ncz, layers, d_off, d_cnt, (const int*)nullptr, (const int*)nullptr, 0,
                        d_item_beg, d_item_pre, nitems});
    site(o2, "sph3d::forcesInteriorRows", rangeOf(s, n),
         ForcesKernel{n, n, ncx, ncy, ncz, d_cell_parts, d_cell_of, d_off, d_cnt, d_pos4, d_vel4, d_type8, h, mass, c0,
                      gravity, d_drho, d_ax, d_ay, d_az, 1, (const int*)nullptr, (const int*)nullptr, d_item_beg,
                      d_item_pre, nitems});
  });
  SPH_PEEK();
}

void invokeForcesTwoListRows(int n_total, int n_local, int n0, int ncx, int ncy, int ncz, int layers,
    const int* d_cell_parts, const int* d_cell_of, const int* d_off, const int* d_cnt,
    const int* d_cell_parts2, const int* d_cell_of2, const int* d_off2, const int* d_cnt2,
    const float4* d_pos4, const float4* d_vel4, const unsigned char* d_type8, RealType h,
    RealType mass, RealType c0, RealType gravity, RealType* d_drho, RealType* d_ax,
    RealType* d_ay, RealType* d_az, int* d_item_beg, int* d_item_pre, hapiStream_t s) {
  const int nitems = 3 * ncy * ncz;
  if (n_total <= 0) return;
  hapiSubmit(s, [=]() {
    static std::once_flag o1, o2;
    site(o1, "sph3d::rowTable2", teamOf(s, 1, 1024),
         RowTableKernel{2, ncx, ncy, ncz, layers, d_off, d_cnt, d_off2, d_cnt2, n0, d_item_beg, d_item_pre, nitems});
    site(o2, "sph3d::forcesTwoListRows", rangeOf(s, n_total),
         ForcesTwoListKernel{n_total, n_local, n0, ncx, ncy, ncz, d_cell_parts, d_cell_of, d_off, d_cnt, d_cell_parts2,
                             d_cell_of2, d_off2, d_cnt2, d_pos4, d_vel4, d_type8, h, mass, c0, gravity, d_drho, d_ax,
                             d_ay, d_az, (const int*)nullptr, (const int*)nullptr, d_item_beg, d_item_pre, nitems});
  });
  SPH_PEEK();
}

void invokeMigrantScatter(int n_total, int n_local, int n0, int ncx, int ncy, int ncz, const int* d_cell_parts,
    const int* d_off, const int* d_cnt, const int* d_cell_parts2, const int* d_cell_of2, const float4* d_pos4,
    const float4* d_vel4, const unsigned char* d_type8, RealType h, RealType mass, RealType c0,
    RealType* d_drho, RealType* d_ax, RealType* d_ay, RealType* d_az, hapiStream_t s) {
  const int nx = n_total - n0;
  if (nx <= 0) return;
  hapiSubmit(s, [=]() {
    static std::once_flag once;
    site(once, "sph3d::migrantScatter", rangeOf(s, nx),
         MigrantScatterKernel{n_total, n_local, n0, ncx, ncy, ncz, d_cell_parts, d_off, d_cnt, d_cell_parts2,
                              d_cell_of2, d_pos4, d_vel4, d_type8, h, mass, c0, d_drho, d_ax, d_ay, d_az});
  });
  SPH_PEEK();
}

void invokeForcesTwoList(int n_total, int n_local, int n0, int ncx, int ncy, int ncz,
    const int* d_cell_parts, const int* d_cell_of, const int* d_off, const int* d_cnt,
    const int* d_cell_parts2, const int* d_cell_of2, const int* d_off2, const int* d_cnt2,
    const float4* d_pos4, const float4* d_vel4, const unsigned char* d_type8, RealType h,
    RealType mass, RealType c0, RealType gravity, RealType* d_drho, RealType* d_ax,
    RealType* d_ay, RealType* d_az, const int* d_iota, unsigned char* d_flags, int* d_sel, int* d_num,
    void* d_tmp, size_t tmp_bytes, hapiStream_t s) {
  if (n_total <= 0) return;
  hapiSubmit(s, [=]() {
    static std::once_flag o1, o2;
    site(o1, "sph3d::flagBoundaryTwoList", rangeOf(s, n_total),
         FlagBoundaryTwoListKernel{n_total, n_local, n0, ncx, ncy, ncz, d_cell_parts, d_cell_of, d_cell_parts2,
                                   d_cell_of2, d_flags});
    size_t bytes = tmp_bytes;
    hapiCheck(cubx::DeviceSelect::Flagged(d_tmp, bytes, d_iota, d_flags, d_sel, d_num, n_total, s));
    site(o2, "sph3d::forcesTwoListSel", rangeOf(s, n_total),
         ForcesTwoListKernel{n_total, n_local, n0, ncx, ncy, ncz, d_cell_parts, d_cell_of, d_off, d_cnt, d_cell_parts2,
                             d_cell_of2, d_off2, d_cnt2, d_pos4, d_vel4, d_type8, h, mass, c0, gravity, d_drho, d_ax,
                             d_ay, d_az, d_sel, d_num, (const int*)nullptr, (const int*)nullptr, 0});
  });
  SPH_PEEK();
}

void invokePredict(const Particle* d_src, Particle* d_dst, int n_local,
    RealType dt2, RealType rho0, RealType c0, const RealType* d_drho,
    const RealType* d_ax, const RealType* d_ay, const RealType* d_az,
    hapiStream_t s) {
  if (n_local <= 0) return;
  hapiSubmit(s, [=]() {
    static std::once_flag once;
    site(once, "sph3d::predict", rangeOf(s, n_local),
         PredictKernel{d_src, d_dst, n_local, dt2, rho0, c0, d_drho, d_ax, d_ay, d_az});
  });
  SPH_PEEK();
}

void invokeCorrect(Particle* d_parts, int n_local, RealType dt, RealType rho0,
    RealType c0, const RealType* d_drho, const RealType* d_ax, const RealType* d_ay,
    const RealType* d_az, int* d_zero, hapiStream_t s) {
  if (n_local <= 0) return;
  hapiSubmit(s, [=]() {
    static std::once_flag once;
    site(once, "sph3d::correct", rangeOf(s, n_local),
         CorrectKernel{d_parts, n_local, dt, rho0, c0, d_drho, d_ax, d_ay, d_az, d_zero, NUM_COUNTERS});
  });
  SPH_PEEK();
}

void invokeIntegrate(Particle* d_parts, int n_local, RealType dt, RealType rho0,
    RealType c0, RealType* d_drho, RealType* d_ax, RealType* d_ay, RealType* d_az,
    int* d_zero, hapiStream_t s) {
  if (n_local <= 0) return;
  hapiSubmit(s, [=]() {
    static std::once_flag once;
    site(once, "sph3d::integrate", rangeOf(s, n_local),
         IntegrateKernel{d_parts, n_local, dt, rho0, c0, d_drho, d_ax, d_ay, d_az, d_zero, NUM_COUNTERS});
  });
  SPH_PEEK();
}

void invokeIntegrateLeavers(bool pc, const Particle* d_parts, int n_local,
    RealType dt, RealType rho0, RealType c0, RealType* d_drho, RealType* d_ax,
    RealType* d_ay, RealType* d_az, RealType x0, RealType y0, RealType z0,
    RealType x1, RealType y1, RealType z1, Particle** d_bufs, Particle* d_stay,
    int* d_counts, int cap_face, hapiStream_t s) {
  sphSubmitMemset(d_counts, 0, sizeof(int) * NUM_COUNTERS, s);
  if (n_local <= 0) return;
  hapiSubmit(s, [=]() {
    static std::once_flag o1, o2;
    if (pc)
      site(o1, "sph3d::correctLeavers", rangeOf(s, n_local),
           CorrectLeaversKernel{d_parts, n_local, dt, rho0, c0, d_drho, d_ax, d_ay, d_az, x0, y0, z0, x1, y1, z1,
                                d_bufs, d_stay, d_counts, cap_face});
    else
      site(o2, "sph3d::integrateLeavers", rangeOf(s, n_local),
           IntegrateLeaversKernel{d_parts, n_local, dt, rho0, c0, d_drho, d_ax, d_ay, d_az, x0, y0, z0, x1, y1, z1,
                                  d_bufs, d_stay, d_counts, cap_face});
  });
  SPH_PEEK();
}

void invokePackHalo(const Particle* d_parts, int n_local, RealType x0,
    RealType y0, RealType z0, RealType x1, RealType y1, RealType z1,
    RealType support, Particle** d_bufs, int** d_idx_bufs, int* d_counts,
    int cap_face, int valid_mask, bool counts_zeroed, hapiStream_t s) {
  if (!(counts_zeroed && n_local > 0)) sphSubmitMemset(d_counts, 0, sizeof(int) * NUM_COUNTERS, s);
  if (n_local > 0)
    hapiSubmit(s, [=]() {
      static std::once_flag once;
      site(once, "sph3d::packHalo", rangeOf(s, n_local),
           PackHaloKernel{d_parts, n_local, x0, y0, z0, x1, y1, z1, support, d_bufs, d_idx_bufs, d_counts, cap_face,
                          valid_mask});
    });
  SPH_PEEK();
}

void invokePackHaloByIndex(const Particle* d_parts, Particle** d_bufs,
    const int* d_idx, const int* d_halo_cnt, int cap_face, int nslots,
    hapiStream_t s) {
  if (nslots <= 0) return;
  hapiSubmit(s, [=]() {
    static std::once_flag once;
    site(once, "sph3d::packHaloByIndex", rangeOf(s, nslots),
         PackHaloByIndexKernel{d_parts, d_bufs, d_idx, d_halo_cnt, cap_face, nslots});
  });
  SPH_PEEK();
}

void invokeGatherLeaverGhosts(Particle** d_bufs, const int* d_counts, Particle* d_parts,
    Particle* d_side, int cap_face, int nslots, int valid_mask, hapiStream_t s) {
  if (nslots <= 0) return;
  hapiSubmit(s, [=]() {
    static std::once_flag once;
    site(once, "sph3d::gatherLeaverGhosts", rangeOf(s, nslots),
         GatherLeaverGhostsKernel{d_bufs, d_counts, d_parts, d_side, cap_face, nslots, valid_mask});
  });
  SPH_PEEK();
}

void invokePackHaloConcurrent(const Particle* d_parts, const int* d_mig_counts,
    int n_max, RealType x0, RealType y0, RealType z0, RealType x1, RealType y1,
    RealType z1, RealType support, Particle** d_bufs, int** d_idx_bufs,
    int* d_counts, int cap_face, int valid_mask, hapiStream_t s) {
  sphSubmitMemset(d_counts, 0, sizeof(int) * NUM_COUNTERS, s);
  if (n_max <= 0) return;
  hapiSubmit(s, [=]() {
    static std::once_flag once;
    site(once, "sph3d::packHaloConcurrent", rangeOf(s, n_max),
         PackHaloConcurrentKernel{d_parts, d_mig_counts, valid_mask, x0, y0, z0, x1, y1, z1, support, d_bufs,
                                  d_idx_bufs, d_counts, cap_face, valid_mask});
  });
  SPH_PEEK();
}

void invokeMarkLeavers(const Particle* d_parts, int n_local, RealType x0,
    RealType y0, RealType z0, RealType x1, RealType y1, RealType z1,
    Particle** d_bufs, Particle* d_stay, int* d_counts, int cap_face,
    bool counts_zeroed, hapiStream_t s) {
  if (!(counts_zeroed && n_local > 0)) sphSubmitMemset(d_counts, 0, sizeof(int) * NUM_COUNTERS, s);
  if (n_local > 0)
    hapiSubmit(s, [=]() {
      static std::once_flag once;
      site(once, "sph3d::markLeavers", rangeOf(s, n_local),
           MarkLeaversKernel{d_parts, n_local, x0, y0, z0, x1, y1, z1, d_bufs, d_stay, d_counts, cap_face});
    });
  SPH_PEEK();
}

void invokeStats(const Particle* d_parts, int n_local, RealType* d_out,
    hapiStream_t s) {
  sphSubmitMemset(d_out, 0, sizeof(RealType) * 8, s);
  if (n_local > 0)
    hapiSubmit(s, [=]() {
      static std::once_flag once;
      site(once, "sph3d::stats", teamOf(s, nblocks(n_local), BLOCK_1D, 5 * sizeof(RealType) * BLOCK_1D),
           StatsKernel{d_parts, n_local, d_out});
    });
  SPH_PEEK();
}

void invokeCheck(const Particle* d_parts, int n_local, RealType x0, RealType y0,
    RealType z0, RealType x1, RealType y1, RealType z1, RealType tol, RealType rho0,
    RealType c0, unsigned long long* d_out, hapiStream_t s, int n_ghost) {
  sphSubmitMemset(d_out, 0, sizeof(unsigned long long) * NUM_CHECKS, s);
  if (n_ghost > 0)
    hapiSubmit(s, [=]() {
      static std::once_flag once;
      hapiCheck(hapiMemsetAsync(d_out + 7, 0xff, sizeof(unsigned long long), s));
      site(once, "sph3d::ghostCheck", rangeOf(s, n_ghost),
           GhostCheckKernel{d_parts, n_local, n_ghost, d_out});
    });
  if (n_local > 0)
    hapiSubmit(s, [=]() {
      static std::once_flag once;
      site(once, "sph3d::check", teamOf(s, nblocks(n_local), BLOCK_1D, 4 * sizeof(unsigned long long) * BLOCK_1D),
           CheckKernel{d_parts, n_local, x0, y0, z0, x1, y1, z1, tol, rho0, c0, d_out});
    });
  SPH_PEEK();
}
