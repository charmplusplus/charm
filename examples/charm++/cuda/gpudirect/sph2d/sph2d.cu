#include "hapi.h"
#include "sph2d.h"
#include <cstdio>

#define BLOCK_1D 128
static inline int nblocks(int n) { return (n + BLOCK_1D - 1) / BLOCK_1D; }

// ---------------------------------------------------------------- kernel ----
// Wendland C2 in 2D. Support 2h; q = r/h.
//   W(q)   = (7/(4 pi h^2)) (1 - q/2)^4 (2q+1)
//   dW/dr  = -(35/(4 pi h^3)) q (1 - q/2)^3
__device__ __forceinline__ RealType wendlandDW(RealType q, RealType inv_h) {
  const RealType t = 1.0f - 0.5f * q;
  const RealType t3 = t * t * t;
  // 35/(4 pi) = 2.785211504
  return -2.785211504f * inv_h * inv_h * inv_h * q * t3;
}

// ------------------------------------------------------------ cell lists ----
// Cell size is the kernel support, so a particle's neighbours lie in its own
// cell and the eight around it. The grid carries one halo layer of cells so
// that ghost particles received from neighbouring patches land somewhere
// valid without any clamping special case in the neighbour loop.
__device__ __forceinline__ int cellOf(RealType px, RealType py, RealType x0,
    RealType y0, RealType inv_csize, int ncx, int ncy) {
  int cx = (int)floorf((px - x0) * inv_csize) + 1;
  int cy = (int)floorf((py - y0) * inv_csize) + 1;
  cx = min(max(cx, 0), ncx + 1);
  cy = min(max(cy, 0), ncy + 1);
  return cy * (ncx + 2) + cx;
}

__global__ void cellCountKernel(const Particle* parts, int n, RealType x0,
    RealType y0, RealType inv_csize, int ncx, int ncy, int* counts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  atomicAdd(&counts[cellOf(parts[i].x, parts[i].y, x0, y0, inv_csize, ncx, ncy)], 1);
}

// Exclusive scan of the cell counts, one block. The cell count is small (a
// patch is tens of cells on a side), so a single block looping over chunks is
// both simpler and quicker than a multi-kernel scan.
__global__ void scanCellsKernel(const int* counts, int* offsets, int ncells) {
  extern __shared__ int tmp[];
  int running = 0;
  for (int base = 0; base < ncells; base += blockDim.x) {
    const int i = base + threadIdx.x;
    const int v = (i < ncells) ? counts[i] : 0;
    tmp[threadIdx.x] = v;
    __syncthreads();
    for (int off = 1; off < blockDim.x; off <<= 1) {
      const int t = (threadIdx.x >= off) ? tmp[threadIdx.x - off] : 0;
      __syncthreads();
      tmp[threadIdx.x] += t;
      __syncthreads();
    }
    if (i < ncells) offsets[i] = running + tmp[threadIdx.x] - v;  // exclusive
    __syncthreads();
    running += tmp[blockDim.x - 1];
    __syncthreads();
  }
}

__global__ void cellScatterKernel(const Particle* parts, int n, RealType x0,
    RealType y0, RealType inv_csize, int ncx, int ncy, int* cursor,
    int* cell_parts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int c = cellOf(parts[i].x, parts[i].y, x0, y0, inv_csize, ncx, ncy);
  cell_parts[atomicAdd(&cursor[c], 1)] = i;
}

// --------------------------------------------------------------- physics ----
// Tait equation of state. Run before the halo exchange so a ghost arrives with
// its pressure already set by its owner -- the receiver cannot compute it,
// having no claim on the ghost's density history.
__global__ void eosKernel(Particle* parts, int n, RealType rho0, RealType c0) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  RealType rho = parts[i].rho;
  // A boundary particle is not allowed to fall below the reference density:
  // without this the wall develops negative pressure and sucks the fluid into
  // it, which is the classic dynamic-boundary failure.
  if (parts[i].type == PTYPE_BOUND && rho < rho0) rho = rho0;
  const RealType B = rho0 * c0 * c0 / EOS_GAMMA;
  const RealType r = rho / rho0;
  const RealType r2 = r * r, r4 = r2 * r2;
  RealType pres = B * (r4 * r2 * r - 1.0f);   // r^7
  // No tension. A free-surface particle has only half its neighbourhood, so
  // continuity density drifts below rho0 there, Tait returns a negative
  // pressure, and the surface particles pull on each other instead of pushing.
  // That is the tensile instability: it is invisible for hundreds of steps and
  // then clumps the surface and blows the run up. Clamping at zero is the
  // standard weakly-compressible remedy and costs nothing in the bulk, where
  // the pressure is positive anyway.
  if (pres < 0.0f) pres = 0.0f;
  parts[i].p = pres;
  parts[i].rho = rho;
}

// One neighbour pass: continuity (with delta-SPH diffusion) and momentum
// (pressure + Monaghan artificial viscosity). Evaluated for local particles
// only; ghosts are sources, never sinks.
__global__ void forcesKernel(const Particle* parts, int n_local, RealType x0,
    RealType y0, RealType inv_csize, int ncx, int ncy, const int* cell_off,
    const int* cell_cnt, const int* cell_parts, RealType h, RealType mass,
    RealType c0, RealType gravity, RealType* drho, RealType* ax, RealType* ay) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_local) return;

  const Particle pi = parts[i];
  const RealType inv_h = 1.0f / h;
  const RealType support2 = (KERNEL_SUPPORT * h) * (KERNEL_SUPPORT * h);
  const RealType eta2h2 = ETA2 * h * h;
  const RealType inv_rhoi2 = 1.0f / (pi.rho * pi.rho);

  RealType adrho = 0.0f, aax = 0.0f, aay = 0.0f;

  const int cx = min(max((int)floorf((pi.x - x0) * inv_csize) + 1, 0), ncx + 1);
  const int cy = min(max((int)floorf((pi.y - y0) * inv_csize) + 1, 0), ncy + 1);

  for (int oy = -1; oy <= 1; oy++) {
    const int yy = cy + oy;
    if (yy < 0 || yy > ncy + 1) continue;
    for (int ox = -1; ox <= 1; ox++) {
      const int xx = cx + ox;
      if (xx < 0 || xx > ncx + 1) continue;
      const int c = yy * (ncx + 2) + xx;
      const int beg = cell_off[c], end = beg + cell_cnt[c];
      for (int k = beg; k < end; k++) {
        const int j = cell_parts[k];
        if (j == i) continue;
        const Particle pj = parts[j];
        const RealType dx = pi.x - pj.x, dy = pi.y - pj.y;
        const RealType r2 = dx * dx + dy * dy;
        if (r2 >= support2 || r2 < 1e-16f) continue;

        const RealType r = sqrtf(r2);
        const RealType dwdr = wendlandDW(r * inv_h, inv_h);
        const RealType gwx = dwdr * dx / r, gwy = dwdr * dy / r;

        const RealType dvx = pi.vx - pj.vx, dvy = pi.vy - pj.vy;

        // continuity
        adrho += mass * (dvx * gwx + dvy * gwy);

        // delta-SPH density diffusion (Molteni & Colagrossi, Marrone 2011):
        // damps the acoustic noise weakly-compressible SPH otherwise
        // accumulates in the pressure field.
        //
        // The sign matters and is easy to get backwards. The diffusion is
        //   D_i = 2 delta h c0 sum (rho_j - rho_i) (r_j - r_i).gradW_i m_j/rho_j
        // over |r|^2, and note (r_j - r_i), not the (r_i - r_j) that dx/dy
        // hold here. gradW_i is antiparallel to (r_i - r_j), so
        // (r_i - r_j).gradW_i = r*W'(r) is NEGATIVE; using it directly makes a
        // denser neighbour pull rho_i down instead of up, which is
        // anti-diffusion and blows the run up within a hundred steps.
        const RealType rdotg = -(dx * gwx + dy * gwy);   // (r_j - r_i).gradW_i
        adrho += 2.0f * DELTA_SPH * h * c0 * (mass / pj.rho) *
                 (pj.rho - pi.rho) * rdotg / (r2 + eta2h2);

        if (pi.type == PTYPE_FLUID) {
          const RealType vdotr = dvx * dx + dvy * dy;
          RealType visc = 0.0f;
          if (vdotr < 0.0f) {   // only for approaching pairs
            const RealType mu = h * vdotr / (r2 + eta2h2);
            visc = -VISC_ALPHA * c0 * mu / (0.5f * (pi.rho + pj.rho));
          }
          const RealType term =
              pi.p * inv_rhoi2 + pj.p / (pj.rho * pj.rho) + visc;
          aax -= mass * term * gwx;
          aay -= mass * term * gwy;
        }
      }
    }
  }

  drho[i] = adrho;
  if (pi.type == PTYPE_FLUID) {
    ax[i] = aax;
    ay[i] = aay - gravity;
  } else {
    ax[i] = 0.0f;
    ay[i] = 0.0f;
  }
}

// Euler-Cromer: velocity first, then position from the new velocity. Boundary
// particles integrate density only -- they are what holds the tank together.
__global__ void integrateKernel(Particle* parts, int n_local, RealType dt,
    RealType rho0, RealType* drho, RealType* ax, RealType* ay) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_local) return;
  parts[i].rho += dt * drho[i];
  if (parts[i].type == PTYPE_BOUND) {
    if (parts[i].rho < rho0) parts[i].rho = rho0;
    return;
  }
  parts[i].vx += dt * ax[i];
  parts[i].vy += dt * ay[i];
  parts[i].x += dt * parts[i].vx;
  parts[i].y += dt * parts[i].vy;
}

// ------------------------------------------------------------- exchanges ----
// Copy every local particle within one support radius of a patch edge into
// that neighbour's halo buffer. A particle near a corner goes to three
// neighbours, which is why the tests below are independent rather than an
// else-if chain.
__global__ void packHaloKernel(const Particle* parts, int n_local, RealType x0,
    RealType y0, RealType x1, RealType y1, RealType support, Particle** bufs,
    int* counts, int cap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_local) return;
  const Particle p = parts[i];
  const bool nearL = (p.x - x0) < support;
  const bool nearR = (x1 - p.x) < support;
  const bool nearB = (p.y - y0) < support;
  const bool nearT = (y1 - p.y) < support;

  bool want[NUM_DIRS];
  want[LEFT] = nearL;    want[RIGHT] = nearR;
  want[TOP] = nearT;     want[BOTTOM] = nearB;
  want[TL] = nearL && nearT;  want[TR] = nearR && nearT;
  want[BL] = nearL && nearB;  want[BR] = nearR && nearB;

  for (int d = 0; d < NUM_DIRS; d++) {
    if (!want[d]) continue;
    const int slot = atomicAdd(&counts[d], 1);
    if (slot < cap) bufs[d][slot] = p;
    else atomicAdd(&counts[ERR_COUNTER], 1);
  }
}

// After integration, move out anything that left the rectangle and compact
// what stays. Boundary particles are fixed, so they always stay.
__global__ void markLeaversKernel(const Particle* parts, int n_local,
    RealType x0, RealType y0, RealType x1, RealType y1, Particle** bufs,
    Particle* stay, int* counts, int cap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_local) return;
  const Particle p = parts[i];

  int sx = 0, sy = 0;
  if (p.type == PTYPE_FLUID) {
    if (p.x < x0) sx = -1; else if (p.x >= x1) sx = 1;
    if (p.y < y0) sy = -1; else if (p.y >= y1) sy = 1;
  }

  if (sx == 0 && sy == 0) {
    stay[atomicAdd(&counts[STAY], 1)] = p;
    return;
  }

  int d;
  if (sx == -1 && sy == 0)      d = LEFT;
  else if (sx == 1 && sy == 0)  d = RIGHT;
  else if (sx == 0 && sy == 1)  d = TOP;
  else if (sx == 0 && sy == -1) d = BOTTOM;
  else if (sx == -1 && sy == 1) d = TL;
  else if (sx == 1 && sy == 1)  d = TR;
  else if (sx == -1)            d = BL;
  else                          d = BR;

  const int slot = atomicAdd(&counts[d], 1);
  if (slot < cap) bufs[d][slot] = p;
  else atomicAdd(&counts[ERR_COUNTER], 1);
}

// Diagnostics: fluid count, summed density, kinetic energy, max speed and the
// leading edge of the surge front -- the last is the quantity a dam break is
// actually validated on.
__global__ void statsKernel(const Particle* parts, int n_local, RealType* out) {
  __shared__ RealType s_rho[BLOCK_1D], s_ke[BLOCK_1D], s_vmax[BLOCK_1D],
      s_xmax[BLOCK_1D], s_n[BLOCK_1D];
  const int t = threadIdx.x;
  const int i = blockIdx.x * blockDim.x + t;

  RealType rho = 0, ke = 0, vmax = 0, xmax = -1e30f, cnt = 0;
  if (i < n_local && parts[i].type == PTYPE_FLUID) {
    const Particle p = parts[i];
    const RealType v2 = p.vx * p.vx + p.vy * p.vy;
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
    // atomicMax has no float overload; these are positive so the bit pattern
    // of a non-negative float orders the same as its integer reinterpretation.
    atomicMax((int*)&out[3], __float_as_int(fmaxf(s_vmax[0], 0.0f)));
    atomicMax((int*)&out[4], __float_as_int(fmaxf(s_xmax[0], 0.0f)));
  }
}

// ---------------------------------------------------------------- launch ----
void invokeCellBuild(const Particle* d_parts, int n, RealType x0, RealType y0,
    RealType inv_csize, int ncx, int ncy, int ncells, int* d_cnt, int* d_off,
    int* d_cursor, int* d_cell_parts, cudaStream_t s) {
  hapiCheck(cudaMemsetAsync(d_cnt, 0, sizeof(int) * ncells, s));
  if (n > 0)
    cellCountKernel<<<nblocks(n), BLOCK_1D, 0, s>>>(d_parts, n, x0, y0,
        inv_csize, ncx, ncy, d_cnt);
  scanCellsKernel<<<1, 1024, sizeof(int) * 1024, s>>>(d_cnt, d_off, ncells);
  hapiCheck(cudaMemcpyAsync(d_cursor, d_off, sizeof(int) * ncells,
      cudaMemcpyDeviceToDevice, s));
  if (n > 0)
    cellScatterKernel<<<nblocks(n), BLOCK_1D, 0, s>>>(d_parts, n, x0, y0,
        inv_csize, ncx, ncy, d_cursor, d_cell_parts);
  hapiCheck(cudaPeekAtLastError());
}

void invokeEOS(Particle* d_parts, int n, RealType rho0, RealType c0,
    cudaStream_t s) {
  if (n <= 0) return;
  eosKernel<<<nblocks(n), BLOCK_1D, 0, s>>>(d_parts, n, rho0, c0);
  hapiCheck(cudaPeekAtLastError());
}

void invokeForces(const Particle* d_parts, int n_local, RealType x0, RealType y0,
    RealType inv_csize, int ncx, int ncy, const int* d_off, const int* d_cnt,
    const int* d_cell_parts, RealType h, RealType mass, RealType c0,
    RealType gravity, RealType* d_drho, RealType* d_ax, RealType* d_ay,
    cudaStream_t s) {
  if (n_local <= 0) return;
  forcesKernel<<<nblocks(n_local), BLOCK_1D, 0, s>>>(d_parts, n_local, x0, y0,
      inv_csize, ncx, ncy, d_off, d_cnt, d_cell_parts, h, mass, c0, gravity,
      d_drho, d_ax, d_ay);
  hapiCheck(cudaPeekAtLastError());
}

void invokeIntegrate(Particle* d_parts, int n_local, RealType dt, RealType rho0,
    RealType* d_drho, RealType* d_ax, RealType* d_ay, cudaStream_t s) {
  if (n_local <= 0) return;
  integrateKernel<<<nblocks(n_local), BLOCK_1D, 0, s>>>(d_parts, n_local, dt,
      rho0, d_drho, d_ax, d_ay);
  hapiCheck(cudaPeekAtLastError());
}

void invokePackHalo(const Particle* d_parts, int n_local, RealType x0,
    RealType y0, RealType x1, RealType y1, RealType support, Particle** d_bufs,
    int* d_counts, int cap, cudaStream_t s) {
  hapiCheck(cudaMemsetAsync(d_counts, 0, sizeof(int) * NUM_COUNTERS, s));
  if (n_local > 0)
    packHaloKernel<<<nblocks(n_local), BLOCK_1D, 0, s>>>(d_parts, n_local, x0,
        y0, x1, y1, support, d_bufs, d_counts, cap);
  hapiCheck(cudaPeekAtLastError());
}

void invokeMarkLeavers(const Particle* d_parts, int n_local, RealType x0,
    RealType y0, RealType x1, RealType y1, Particle** d_bufs, Particle* d_stay,
    int* d_counts, int cap, cudaStream_t s) {
  hapiCheck(cudaMemsetAsync(d_counts, 0, sizeof(int) * NUM_COUNTERS, s));
  if (n_local > 0)
    markLeaversKernel<<<nblocks(n_local), BLOCK_1D, 0, s>>>(d_parts, n_local,
        x0, y0, x1, y1, d_bufs, d_stay, d_counts, cap);
  hapiCheck(cudaPeekAtLastError());
}

void invokeStats(const Particle* d_parts, int n_local, RealType* d_out,
    cudaStream_t s) {
  hapiCheck(cudaMemsetAsync(d_out, 0, sizeof(RealType) * 8, s));
  if (n_local > 0)
    statsKernel<<<nblocks(n_local), BLOCK_1D, 0, s>>>(d_parts, n_local, d_out);
  hapiCheck(cudaPeekAtLastError());
}
