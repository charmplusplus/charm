// CUB first: converse.h defines ALIGN_BYTES as a macro, which is also the name
// of a member constant in CUB's alignment traits.
#include <cub/device/device_scan.cuh>
#include "hapi.h"
#include "sph3d.h"

// See sph2d.cu: the kernels here are the same ones with a third coordinate.
#ifdef SPH_PEEK_ERRORS
#define SPH_PEEK() hapiCheck(cudaPeekAtLastError())
#else
#define SPH_PEEK() ((void)0)
#endif
#include <cstdio>
#include <cstdlib>

static inline void sphSubmitMemset(void* d, int v, size_t n, cudaStream_t s) {
  hapiSubmit(s, [=]() { hapiCheck(cudaMemsetAsync(d, v, n, s)); });
}
static inline void sphSubmitMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind k, cudaStream_t s) {
  hapiSubmit(s, [=]() { hapiCheck(cudaMemcpyAsync(dst, src, n, k, s)); });
}

#define BLOCK_1D 128
static inline int nblocks(int n) { return (n + BLOCK_1D - 1) / BLOCK_1D; }

// ---------------------------------------------------------------- kernel ----
// Wendland C2 in 3D. Support 2h; q = r/h.
//   W(q)   = (21/(16 pi h^3)) (1 - q/2)^4 (2q+1)
//   dW/dr  = -(105/(16 pi h^4)) q (1 - q/2)^3
__device__ __forceinline__ RealType wendlandDW(RealType q, RealType inv_h) {
  const RealType t = 1.0f - 0.5f * q;
  const RealType t3 = t * t * t;
  const RealType inv_h2 = inv_h * inv_h;
  // 105/(16 pi) = 2.0889086
  return -2.0889086f * inv_h2 * inv_h2 * q * t3;
}

// ------------------------------------------------------------ cell lists ----
// Cell size is the kernel support, so a particle's neighbours lie in its own
// cell and the 26 around it. One halo layer of cells on every side so that
// ghosts from neighbouring patches land somewhere valid without clamping
// special cases in the neighbour loop.
__device__ __forceinline__ int cellOf(RealType px, RealType py, RealType pz,
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

// Also records each particle's cell (cell_of): the scatter reuses it, and the
// forces kernel takes its stencil centre from it rather than from the current
// position, so that a cell list reused across a neighbour-list window (-k)
// stays symmetric: i sees j iff j sees i, both judged by where they were binned.
__global__ void cellCountKernel(const Particle* parts, int n, RealType x0,
    RealType y0, RealType z0, RealType inv_csize, int ncx, int ncy, int ncz,
    int* counts, int* cell_of) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int c = cellOf(parts[i].x, parts[i].y, parts[i].z, x0, y0, z0,
      inv_csize, ncx, ncy, ncz);
  cell_of[i] = c;
  atomicAdd(&counts[c], 1);
}

// The exclusive scan of the cell counts is CUB's DeviceScan (invokeCellBuild):
// the single-block scan it replaces took 65 us per call on 13.5k cells, twice
// per patch per step, 12% of all GPU time in the 25M single-GPU profile.

__global__ void cellScatterKernel(const int* cell_of, int n, int* cursor,
    int* cell_parts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  cell_parts[atomicAdd(&cursor[cell_of[i]], 1)] = i;
}

// The neighbour data of every particle (locals and ghosts) in cell order, as
// two float4 streams: pos4 = (x, y, z, rho), vel4 = (vx, vy, vz, p). The forces
// kernel walks a cell as a contiguous run of these -- two 16-byte loads per
// neighbour, no index indirection -- instead of 40-byte Particle structs
// reached through cell_parts. Rebuilt every evaluation (positions change);
// the cell list it follows may be a window's stale one (sph3d.h).
__global__ void layoutKernel(const Particle* __restrict__ parts,
    const int* __restrict__ cell_parts, int n, float4* __restrict__ pos4,
    float4* __restrict__ vel4, unsigned char* __restrict__ type8) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  const Particle p = parts[cell_parts[k]];
  pos4[k] = make_float4(p.x, p.y, p.z, p.rho);
  vel4[k] = make_float4(p.vx, p.vy, p.vz, p.p);
  type8[k] = (unsigned char)p.type;
}

// Gather particles in cell order: dst[k] = src[cell_parts[k]]. Run on the
// locals every step so that the forces kernel walks neighbours that are
// contiguous in memory. Without it the compaction's atomicAdd placement
// scrambles the lattice order a little more every step and the neighbour
// reads lose locality steadily: 303 -> 961 ms/step over 300 steps at 98M
// particles (job 22727057), with clocks pinned. GPUSPH re-sorts by cell hash
// every neighbour-list rebuild for the same reason.
__global__ void gatherKernel(const Particle* src, const int* cell_parts, int n,
    Particle* dst) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  dst[k] = src[cell_parts[k]];
}

// --------------------------------------------------------------- physics ----
// Tait equation of state, no tension (see sph2d.cu). Pressure is set wherever
// the density changes -- the integrators below, and once at init -- so a ghost
// arrives with its pressure set by its owner and no pass precedes the pack.
__device__ __forceinline__ RealType taitPressure(RealType rho, RealType rho0, RealType c0) {
  const RealType B = rho0 * c0 * c0 / EOS_GAMMA;
  const RealType r = rho / rho0;
  const RealType r2 = r * r, r4 = r2 * r2;
  const RealType pres = B * (r4 * r2 * r - 1.0f);   // r^7
  return pres < 0.0f ? 0.0f : pres;
}

// Init only: the lattice arrives without pressures.
__global__ void eosKernel(Particle* parts, int n, RealType rho0, RealType c0,
    int* zero, int nzero) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (i == 0 && zero != nullptr) for (int k = 0; k < nzero; k++) zero[k] = 0;
  RealType rho = parts[i].rho;
  if (parts[i].type == PTYPE_BOUND && rho < rho0) rho = rho0;
  parts[i].p = taitPressure(rho, rho0, c0);
  parts[i].rho = rho;
}

// One neighbour pass: continuity (with delta-SPH diffusion) and momentum
// (pressure + Monaghan artificial viscosity). Local particles only; ghosts
// are sources, never sinks.
// One thread per SORTED position k, locals only: a warp then holds 32
// particles of one cell and reads each neighbour cell once through L1. With
// the threads in array order instead, every thread streams its own 27 cells
// -- 2x to 5x slower, and worse as the compaction scrambles the array (the
// sort of the locals used to buy this coherence for the whole array; it is
// no longer needed). Results go back by the original index.
__global__ void forcesKernel(int n_total, int n_local, int ncx,
    int ncy, int ncz, const int* __restrict__ cell_parts,
    const int* __restrict__ cell_of, const int* __restrict__ cell_off,
    const int* __restrict__ cell_cnt, const float4* __restrict__ pos4,
    const float4* __restrict__ vel4, const unsigned char* __restrict__ type8,
    RealType h, RealType mass, RealType c0, RealType gravity, RealType* drho,
    RealType* ax, RealType* ay, RealType* az) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n_total) return;
  const int i = cell_parts[k];
  if (i >= n_local) return;   // a ghost: a source, never a sink

  struct { RealType x, y, z, vx, vy, vz, rho, p; int type; } pi;
  {
    const float4 a = pos4[k], b = vel4[k];
    pi.x = a.x; pi.y = a.y; pi.z = a.z; pi.rho = a.w;
    pi.vx = b.x; pi.vy = b.y; pi.vz = b.z; pi.p = b.w;
    pi.type = type8[k];
  }
  const RealType inv_h = 1.0f / h;
  const RealType support2 = (KERNEL_SUPPORT * h) * (KERNEL_SUPPORT * h);
  const RealType eta2h2 = ETA2 * h * h;
  const RealType inv_rhoi2 = 1.0f / (pi.rho * pi.rho);

  RealType adrho = 0.0f, aax = 0.0f, aay = 0.0f, aaz = 0.0f;

  // The cell this particle was binned into when the list was built (see
  // cellCountKernel), not the cell its current position falls in.
  const int ci = cell_of[i];
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
        const int beg = cell_off[c], end = beg + cell_cnt[c];
        for (int k = beg; k < end; k++) {
          // The particle itself is the pair at zero distance, skipped below
          // with everything closer than 1e-8 h.
          const float4 pj = __ldg(&pos4[k]);   // x y z rho
          const RealType dx = pi.x - pj.x, dy = pi.y - pj.y, dz = pi.z - pj.z;
          const RealType r2 = dx * dx + dy * dy + dz * dz;
          if (r2 >= support2 || r2 < 1e-16f) continue;
          const float4 vj = __ldg(&vel4[k]);   // vx vy vz p
          const RealType rhoj = pj.w, pj_p = vj.w;

          // One reciprocal square root per pair; the eight divisions this
          // replaced were a measurable share of the kernel.
          const RealType rinv = rsqrtf(r2);
          const RealType r = r2 * rinv;
          const RealType gw = wendlandDW(r * inv_h, inv_h) * rinv;   // dW/dr / r
          const RealType gwx = gw * dx, gwy = gw * dy, gwz = gw * dz;
          const RealType inv_rhoj = 1.0f / rhoj;
          const RealType inv_r2e = 1.0f / (r2 + eta2h2);

          const RealType dvx = pi.vx - vj.x, dvy = pi.vy - vj.y, dvz = pi.vz - vj.z;

          // continuity
          adrho += mass * (dvx * gwx + dvy * gwy + dvz * gwz);

          // delta-SPH density diffusion; sign as explained in sph2d.cu
          const RealType rdotg = -(dx * gwx + dy * gwy + dz * gwz);   // (r_j - r_i).gradW_i
          adrho += 2.0f * DELTA_SPH * h * c0 * (mass * inv_rhoj) *
                   (rhoj - pi.rho) * rdotg * inv_r2e;

          if (pi.type == PTYPE_FLUID) {
            const RealType vdotr = dvx * dx + dvy * dy + dvz * dz;
            RealType visc = 0.0f;
            if (vdotr < 0.0f) {   // only for approaching pairs
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

// Euler-Cromer for one particle: velocity first, then position from the new
// velocity; boundary particles integrate density only. Pressure from the new
// density.
__device__ __forceinline__ void ecStep(Particle& q, int i, RealType dt, RealType rho0,
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

// GPUSPH's predictor-corrector (euler_kernel.def there). Predictor, a half
// step from the n-state:
//   x* = x + v dt/2,  v* = v + a dt/2,  rho* = rho + drho dt/2.
// Corrector, the n-state advanced with the forces at the predicted state:
//   vc = v + a* dt/2;  x1 = x + vc dt;  v1 = v + a* dt;  rho1 = rho + drho* dt.
__device__ __forceinline__ void pcPredict(Particle& q, int i, RealType dt2, RealType rho0,
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
__device__ __forceinline__ void pcCorrect(Particle& q, int i, RealType dt, RealType rho0,
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

// Where an integrated particle goes at a migration: into the compacted "stay"
// buffer, or the slot of the neighbour it crossed into. Boundary particles
// are fixed, so they always stay.
__device__ __forceinline__ void placeParticle(const Particle& p, RealType x0,
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
    stay[atomicAdd(&counts[STAY], 1)] = p;
    return;
  }
  const int slot = atomicAdd(&counts[d], 1);
  if (slot < dirCap(d, cap_face)) bufs[d][slot] = p;
  else atomicAdd(&counts[ERR_COUNTER], 1);
}

// In place (a step inside a neighbour-list window, or a reporting step whose
// statistics read the array before the compaction). zero/nzero: see sph2d.cu
// (a counter array the next kernel on this stream accumulates into, cleared
// here by thread 0).
__global__ void integrateKernel(Particle* parts, int n_local, RealType dt,
    RealType rho0, RealType c0, const RealType* drho, const RealType* ax,
    const RealType* ay, const RealType* az, int* zero, int nzero) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_local) return;
  if (i == 0 && zero != nullptr) for (int k = 0; k < nzero; k++) zero[k] = 0;
  Particle q = parts[i];
  ecStep(q, i, dt, rho0, c0, drho, ax, ay, az);
  parts[i] = q;
}

// Integrate and compact in one pass: the particle is read once and written
// once, to where it belongs after the step. counts must be zero on entry.
__global__ void integrateLeaversKernel(const Particle* parts, int n_local,
    RealType dt, RealType rho0, RealType c0, const RealType* drho,
    const RealType* ax, const RealType* ay, const RealType* az,
    RealType x0, RealType y0, RealType z0, RealType x1, RealType y1, RealType z1,
    Particle** bufs, Particle* stay, int* counts, int cap_face) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_local) return;
  Particle q = parts[i];
  ecStep(q, i, dt, rho0, c0, drho, ax, ay, az);
  placeParticle(q, x0, y0, z0, x1, y1, z1, bufs, stay, counts, cap_face);
}

__global__ void predictKernel(const Particle* src, Particle* dst, int n_local,
    RealType dt2, RealType rho0, RealType c0, const RealType* drho,
    const RealType* ax, const RealType* ay, const RealType* az) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_local) return;
  Particle q = src[i];
  pcPredict(q, i, dt2, rho0, c0, drho, ax, ay, az);
  dst[i] = q;
}

__global__ void correctKernel(Particle* parts, int n_local, RealType dt,
    RealType rho0, RealType c0, const RealType* drho, const RealType* ax,
    const RealType* ay, const RealType* az, int* zero, int nzero) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_local) return;
  if (i == 0 && zero != nullptr) for (int k = 0; k < nzero; k++) zero[k] = 0;
  Particle q = parts[i];
  pcCorrect(q, i, dt, rho0, c0, drho, ax, ay, az);
  parts[i] = q;
}

__global__ void correctLeaversKernel(const Particle* parts, int n_local,
    RealType dt, RealType rho0, RealType c0, const RealType* drho,
    const RealType* ax, const RealType* ay, const RealType* az,
    RealType x0, RealType y0, RealType z0, RealType x1, RealType y1, RealType z1,
    Particle** bufs, Particle* stay, int* counts, int cap_face) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_local) return;
  Particle q = parts[i];
  pcCorrect(q, i, dt, rho0, c0, drho, ax, ay, az);
  placeParticle(q, x0, y0, z0, x1, y1, z1, bufs, stay, counts, cap_face);
}

// ------------------------------------------------------------- exchanges ----
// Copy every local particle within one support radius of a patch face into
// that neighbour's halo buffer. A particle near a corner goes to seven
// neighbours (three faces, three edges, one corner): the direction loop tests
// each component independently.
// Also writes each sent particle's local index into the per-direction index
// lists (idx), which the by-index pack below replays for the rest of the
// neighbour-list window and for the predictor-corrector's second exchange.
__global__ void packHaloKernel(const Particle* parts, int n_local,
    RealType x0, RealType y0, RealType z0, RealType x1, RealType y1, RealType z1,
    RealType support, Particle** bufs, int** idx, int* counts, int cap_face,
    int valid_mask) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_local) return;
  const Particle p = parts[i];
  // near[axis][0] = near the low face, near[axis][1] = near the high face
  const bool nx0 = (p.x - x0) < support, nx1 = (x1 - p.x) < support;
  const bool ny0 = (p.y - y0) < support, ny1 = (y1 - p.y) < support;
  const bool nz0 = (p.z - z0) < support, nz1 = (z1 - p.z) < support;
  // Interior particles -- most of them -- go to nobody.
  if (!(nx0 | nx1 | ny0 | ny1 | nz0 | nz1)) return;

  for (int d = 0; d < NUM_DIRS; d++) {
    // Only directions with a partner: a tank wall is three layers thick and
    // sits right at the face towards the outside, and counting it for a
    // direction that is never sent overflowed the slot (job 22732376).
    if (!((valid_mask >> d) & 1)) continue;
    const int sx = dirSX(d), sy = dirSY(d), sz = dirSZ(d);
    const bool okx = sx == 0 || (sx < 0 ? nx0 : nx1);
    const bool oky = sy == 0 || (sy < 0 ? ny0 : ny1);
    const bool okz = sz == 0 || (sz < 0 ? nz0 : nz1);
    if (!(okx && oky && okz)) continue;
    const int slot = atomicAdd(&counts[d], 1);
    if (slot < dirCap(d, cap_face)) { bufs[d][slot] = p; idx[d][slot] = i; }
    else atomicAdd(&counts[ERR_COUNTER], 1);
  }
}

// Re-send the particles the rebuild chose, in the same order: one thread per
// exchange slot, the slot's direction found by walking the per-direction
// capacities (the same layout dirOffset() uses on the host). One launch covers
// all 26 directions.
__global__ void packHaloByIndexKernel(const Particle* parts, Particle** bufs,
    const int* idx, const int* halo_cnt, int cap_face, int nslots) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
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

// After an in-place integration, move out anything that left the box and
// compact what stays (placeParticle). Used on the reporting steps; elsewhere
// the integration and this are one kernel.
__global__ void markLeaversKernel(const Particle* parts, int n_local,
    RealType x0, RealType y0, RealType z0, RealType x1, RealType y1, RealType z1,
    Particle** bufs, Particle* stay, int* counts, int cap_face) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_local) return;
  placeParticle(parts[i], x0, y0, z0, x1, y1, z1, bufs, stay, counts, cap_face);
}

// Diagnostics: fluid count, summed density, kinetic energy, max speed and the
// leading edge of the surge front along x.
__global__ void statsKernel(const Particle* parts, int n_local, RealType* out) {
  __shared__ RealType s_rho[BLOCK_1D], s_ke[BLOCK_1D], s_vmax[BLOCK_1D],
      s_xmax[BLOCK_1D], s_n[BLOCK_1D];
  const int t = threadIdx.x;
  const int i = blockIdx.x * blockDim.x + t;

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
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (t < s) {
      s_rho[t] += s_rho[t + s];
      s_ke[t] += s_ke[t + s];
      s_n[t] += s_n[t + s];
      s_vmax[t] = fmaxf(s_vmax[t], s_vmax[t + s]);
      s_xmax[t] = fmaxf(s_xmax[t], s_xmax[t + s]);
    }
    __syncthreads();
  }
  if (t == 0) {
    atomicAdd(&out[0], s_rho[0]);
    atomicAdd(&out[1], s_ke[0]);
    atomicAdd(&out[2], s_n[0]);
    atomicMax((int*)&out[3], __float_as_int(fmaxf(s_vmax[0], 0.0f)));
    atomicMax((int*)&out[4], __float_as_int(fmaxf(s_xmax[0], 0.0f)));
  }
}

// ---------------------------------------------------------------- checks ----
__device__ __forceinline__ unsigned long long mix64(unsigned long long z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Over the LOCAL particles only, before the physics of the step.
// tol: how far outside its box a local particle may sit. Zero right after a
// migration; one support inside a neighbour-list window, where a particle is
// integrated by its old owner until the next migration (a wrong-neighbour
// delivery is a whole patch away and is still caught).
__global__ void checkKernel(const Particle* parts, int n_local, RealType x0,
    RealType y0, RealType z0, RealType x1, RealType y1, RealType z1,
    RealType tol, RealType rho0, RealType c0, unsigned long long* out) {
  __shared__ unsigned long long s_id[BLOCK_1D], s_bnd[BLOCK_1D],
      s_nf[BLOCK_1D], s_nb[BLOCK_1D];
  const int t = threadIdx.x;
  const int i = blockIdx.x * blockDim.x + t;

  unsigned long long idh = 0, bndh = 0, nf = 0, nb = 0;
  unsigned long long bad = 0;

  if (i < n_local) {
    const Particle p = parts[i];
    const unsigned long long id = (unsigned long long)(unsigned int)p.id;
    idh = mix64(id + 1);

    if (p.type == PTYPE_BOUND) {
      nb = 1;
      // Identity AND position: a boundary particle is fixed for the whole run.
      bndh = mix64(id + 1) ^
             mix64(((unsigned long long)__float_as_uint(p.x) << 32) |
                   (unsigned long long)__float_as_uint(p.y)) ^
             mix64(((unsigned long long)__float_as_uint(p.z) << 32) | (id + 7));
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
      atomicOr(&out[CHK_BAD_MASK], bad);
      atomicAdd(&out[CHK_BAD_COUNT], 1ull);
    }
  }

  s_id[t] = idh; s_bnd[t] = bndh; s_nf[t] = nf; s_nb[t] = nb;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (t < s) {
      s_id[t] += s_id[t + s];
      s_bnd[t] += s_bnd[t + s];
      s_nf[t] += s_nf[t + s];
      s_nb[t] += s_nb[t + s];
    }
    __syncthreads();
  }
  if (t == 0) {
    atomicAdd(&out[CHK_ID_SUM], s_id[0]);
    atomicAdd(&out[CHK_BND_SUM], s_bnd[0]);
    atomicAdd(&out[CHK_N_FLUID], s_nf[0]);
    atomicAdd(&out[CHK_N_BOUND], s_nb[0]);
  }
}

__global__ void ghostCheckKernel(const Particle* parts, int n_local, int n_ghost,
    unsigned long long* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_ghost) return;
  const Particle p = parts[n_local + i];
  if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z) || !isfinite(p.rho) ||
      !isfinite(p.p) || !isfinite(p.vx) || !isfinite(p.vy) || !isfinite(p.vz)) {
    atomicAdd(&out[6], 1ull);
    atomicMin(&out[7], (unsigned long long)i);
  }
}

// ---------------------------------------------------------------- launch ----
// Temporary storage CUB's scan needs for ncells counts; allocated once per
// patch (allocDevice) and reused by every build.
size_t sphScanTempBytes(int ncells) {
  size_t bytes = 0;
  int* dummy = NULL;
  hapiCheck(cub::DeviceScan::ExclusiveSum(NULL, bytes, dummy, dummy, ncells));
  return bytes;
}

void invokeCellBuild(const Particle* d_parts, int n, RealType x0, RealType y0,
    RealType z0, RealType inv_csize, int ncx, int ncy, int ncz, int ncells,
    int* d_cnt, int* d_off, int* d_cursor, int* d_cell_parts, int* d_cell_of,
    void* d_scan_tmp, size_t scan_tmp_bytes, cudaStream_t s) {
  sphSubmitMemset(d_cnt, 0, sizeof(int) * ncells, s);
  if (n > 0)
    hapiSubmit(s, [=]() {
      cellCountKernel<<<nblocks(n), BLOCK_1D, 0, s>>>(d_parts, n, x0, y0, z0,
          inv_csize, ncx, ncy, ncz, d_cnt, d_cell_of);
    });
  hapiSubmit(s, [=]() {
    size_t bytes = scan_tmp_bytes;
    hapiCheck(cub::DeviceScan::ExclusiveSum(d_scan_tmp, bytes, d_cnt, d_off, ncells, s));
    // The scatter's cursor starts at each cell's offset.
    hapiCheck(cudaMemcpyAsync(d_cursor, d_off, sizeof(int) * ncells,
        cudaMemcpyDeviceToDevice, s));
  });
  if (n > 0)
    hapiSubmit(s, [=]() {
      cellScatterKernel<<<nblocks(n), BLOCK_1D, 0, s>>>(d_cell_of, n, d_cursor,
          d_cell_parts);
    });
  SPH_PEEK();
}

void invokeGather(const Particle* d_src, const int* d_cell_parts, int n,
    Particle* d_dst, cudaStream_t s) {
  if (n <= 0) return;
  hapiSubmit(s, [=]() {
    gatherKernel<<<nblocks(n), BLOCK_1D, 0, s>>>(d_src, d_cell_parts, n, d_dst);
  });
  SPH_PEEK();
}

void invokeEOS(Particle* d_parts, int n, RealType rho0, RealType c0,
    int* d_zero, cudaStream_t s) {
  if (n <= 0) return;
  hapiSubmit(s, [=]() {
    eosKernel<<<nblocks(n), BLOCK_1D, 0, s>>>(d_parts, n, rho0, c0, d_zero, NUM_COUNTERS);
  });
  SPH_PEEK();
}

void invokeLayout(const Particle* d_parts, const int* d_cell_parts, int n,
    float4* d_pos4, float4* d_vel4, unsigned char* d_type8, cudaStream_t s) {
  if (n <= 0) return;
  hapiSubmit(s, [=]() {
    layoutKernel<<<nblocks(n), BLOCK_1D, 0, s>>>(d_parts, d_cell_parts, n, d_pos4,
        d_vel4, d_type8);
  });
  SPH_PEEK();
}

void invokeForces(int n_total, int n_local, int ncx, int ncy, int ncz,
    const int* d_cell_parts, const int* d_cell_of, const int* d_off,
    const int* d_cnt, const float4* d_pos4, const float4* d_vel4,
    const unsigned char* d_type8, RealType h, RealType mass, RealType c0,
    RealType gravity, RealType* d_drho, RealType* d_ax, RealType* d_ay,
    RealType* d_az, cudaStream_t s) {
  if (n_local <= 0 || n_total <= 0) return;
  // SPH_FORCES_BLOCK: the forces kernel's block size (default 128).
  static const int fb = [] { const char* e = getenv("SPH_FORCES_BLOCK"); return e ? atoi(e) : BLOCK_1D; }();
  hapiSubmit(s, [=]() {
    forcesKernel<<<(n_total + fb - 1) / fb, fb, 0, s>>>(n_total, n_local, ncx, ncy,
        ncz, d_cell_parts, d_cell_of, d_off, d_cnt, d_pos4, d_vel4, d_type8, h, mass,
        c0, gravity, d_drho, d_ax, d_ay, d_az);
  });
  SPH_PEEK();
}

void invokePredict(const Particle* d_src, Particle* d_dst, int n_local,
    RealType dt2, RealType rho0, RealType c0, const RealType* d_drho,
    const RealType* d_ax, const RealType* d_ay, const RealType* d_az,
    cudaStream_t s) {
  if (n_local <= 0) return;
  hapiSubmit(s, [=]() {
    predictKernel<<<nblocks(n_local), BLOCK_1D, 0, s>>>(d_src, d_dst, n_local, dt2,
        rho0, c0, d_drho, d_ax, d_ay, d_az);
  });
  SPH_PEEK();
}

void invokeCorrect(Particle* d_parts, int n_local, RealType dt, RealType rho0,
    RealType c0, const RealType* d_drho, const RealType* d_ax, const RealType* d_ay,
    const RealType* d_az, int* d_zero, cudaStream_t s) {
  if (n_local <= 0) return;
  hapiSubmit(s, [=]() {
    correctKernel<<<nblocks(n_local), BLOCK_1D, 0, s>>>(d_parts, n_local, dt, rho0,
        c0, d_drho, d_ax, d_ay, d_az, d_zero, NUM_COUNTERS);
  });
  SPH_PEEK();
}

void invokeIntegrate(Particle* d_parts, int n_local, RealType dt, RealType rho0,
    RealType c0, RealType* d_drho, RealType* d_ax, RealType* d_ay, RealType* d_az,
    int* d_zero, cudaStream_t s) {
  if (n_local <= 0) return;
  hapiSubmit(s, [=]() {
    integrateKernel<<<nblocks(n_local), BLOCK_1D, 0, s>>>(d_parts, n_local, dt,
        rho0, c0, d_drho, d_ax, d_ay, d_az, d_zero, NUM_COUNTERS);
  });
  SPH_PEEK();
}

// Integrate (Euler-Cromer or the corrector, by pc) and compact in one pass.
// Clears the counters itself.
void invokeIntegrateLeavers(bool pc, const Particle* d_parts, int n_local,
    RealType dt, RealType rho0, RealType c0, RealType* d_drho, RealType* d_ax,
    RealType* d_ay, RealType* d_az, RealType x0, RealType y0, RealType z0,
    RealType x1, RealType y1, RealType z1, Particle** d_bufs, Particle* d_stay,
    int* d_counts, int cap_face, cudaStream_t s) {
  sphSubmitMemset(d_counts, 0, sizeof(int) * NUM_COUNTERS, s);
  if (n_local <= 0) return;
  hapiSubmit(s, [=]() {
    if (pc)
      correctLeaversKernel<<<nblocks(n_local), BLOCK_1D, 0, s>>>(d_parts, n_local,
          dt, rho0, c0, d_drho, d_ax, d_ay, d_az, x0, y0, z0, x1, y1, z1, d_bufs,
          d_stay, d_counts, cap_face);
    else
      integrateLeaversKernel<<<nblocks(n_local), BLOCK_1D, 0, s>>>(d_parts, n_local,
          dt, rho0, c0, d_drho, d_ax, d_ay, d_az, x0, y0, z0, x1, y1, z1, d_bufs,
          d_stay, d_counts, cap_face);
  });
  SPH_PEEK();
}


void invokePackHalo(const Particle* d_parts, int n_local, RealType x0,
    RealType y0, RealType z0, RealType x1, RealType y1, RealType z1,
    RealType support, Particle** d_bufs, int** d_idx_bufs, int* d_counts,
    int cap_face, int valid_mask, bool counts_zeroed, cudaStream_t s) {
  if (!(counts_zeroed && n_local > 0)) sphSubmitMemset(d_counts, 0, sizeof(int) * NUM_COUNTERS, s);
  if (n_local > 0)
    hapiSubmit(s, [=]() {
      packHaloKernel<<<nblocks(n_local), BLOCK_1D, 0, s>>>(d_parts, n_local, x0,
          y0, z0, x1, y1, z1, support, d_bufs, d_idx_bufs, d_counts, cap_face,
          valid_mask);
    });
  SPH_PEEK();
}

void invokePackHaloByIndex(const Particle* d_parts, Particle** d_bufs,
    const int* d_idx, const int* d_halo_cnt, int cap_face, int nslots,
    cudaStream_t s) {
  if (nslots <= 0) return;
  hapiSubmit(s, [=]() {
    packHaloByIndexKernel<<<nblocks(nslots), BLOCK_1D, 0, s>>>(d_parts, d_bufs,
        d_idx, d_halo_cnt, cap_face, nslots);
  });
  SPH_PEEK();
}

void invokeMarkLeavers(const Particle* d_parts, int n_local, RealType x0,
    RealType y0, RealType z0, RealType x1, RealType y1, RealType z1,
    Particle** d_bufs, Particle* d_stay, int* d_counts, int cap_face,
    bool counts_zeroed, cudaStream_t s) {
  if (!(counts_zeroed && n_local > 0)) sphSubmitMemset(d_counts, 0, sizeof(int) * NUM_COUNTERS, s);
  if (n_local > 0)
    hapiSubmit(s, [=]() {
      markLeaversKernel<<<nblocks(n_local), BLOCK_1D, 0, s>>>(d_parts, n_local,
          x0, y0, z0, x1, y1, z1, d_bufs, d_stay, d_counts, cap_face);
    });
  SPH_PEEK();
}

void invokeStats(const Particle* d_parts, int n_local, RealType* d_out,
    cudaStream_t s) {
  sphSubmitMemset(d_out, 0, sizeof(RealType) * 8, s);
  if (n_local > 0)
    hapiSubmit(s, [=]() {
      statsKernel<<<nblocks(n_local), BLOCK_1D, 0, s>>>(d_parts, n_local, d_out);
    });
  SPH_PEEK();
}

void invokeCheck(const Particle* d_parts, int n_local, RealType x0, RealType y0,
    RealType z0, RealType x1, RealType y1, RealType z1, RealType tol, RealType rho0,
    RealType c0, unsigned long long* d_out, cudaStream_t s, int n_ghost) {
  sphSubmitMemset(d_out, 0, sizeof(unsigned long long) * NUM_CHECKS, s);
  if (n_ghost > 0)
    hapiSubmit(s, [=]() {
      hapiCheck(cudaMemsetAsync(d_out + 7, 0xff, sizeof(unsigned long long), s));
      ghostCheckKernel<<<nblocks(n_ghost), BLOCK_1D, 0, s>>>(d_parts, n_local, n_ghost, d_out);
    });
  if (n_local > 0)
    hapiSubmit(s, [=]() {
      checkKernel<<<nblocks(n_local), BLOCK_1D, 0, s>>>(d_parts, n_local, x0, y0, z0,
          x1, y1, z1, tol, rho0, c0, d_out);
    });
  SPH_PEEK();
}
