#include "defs.h"
#include "leanmd_cuda.h"

#include <cstdio>

// ---------------------------------------------------------------------------
// Lennard-Jones force kernels.
//
// Ported from calcPairForces / calcInternalForces in physics.h, with the same
// unit system: positions are in Angstroms, the r^-6 / r^-12 terms are evaluated
// after scaling r^2 by 1e-20, and the resulting force is scaled by 1e-10. Those
// constants are compile-time here rather than pow() calls per invocation.
//
// DECOMPOSITION: one block per atom i, threads striding over j, then a shared
// -memory reduction to a single force vector. The obvious alternative -- one
// thread per atom -- yields only nAtoms/256 blocks, four at 1000 atoms, which
// would leave an A40's 84 SMs almost entirely idle and reproduce exactly the
// narrow-kernel problem this port exists to avoid. Block-per-atom gives 1000
// blocks at 1000 atoms, and each block still evaluates a full row of the pair
// matrix, so the total work is the same 1M pair evaluations.
//
// Newton's third law is deliberately NOT exploited. Using f_ji = -f_ij would
// require atomics into the other cell's force array, and the contention on a
// 1000-row matrix costs more than recomputing. Instead the reciprocal forces come
// from a second launch with the two cells swapped: 2x the arithmetic, zero atomics,
// and every write is to a location only one block owns.
// ---------------------------------------------------------------------------

#define VDW_A_D  (1.1328e-133)
#define VDW_B_D  (2.23224e-76)
#define POW_TEN     (1.0e-10)
#define POW_TWENTY  (1.0e-20)
// updateProperties reports kinetic energy in milliJoules by scaling by 10^10.
// The pair potential is already in those units, so both have to agree or the
// conservation test compares apples to oranges.
#define POW_TEN_POS (1.0e10)

#define BLOCK_THREADS 256

// Accumulate the force on one atom of A from every atom of B.
//
// POSITIONS ONLY. The kernels never read mass, velocity or acceleration, and a
// full Particle is 80 bytes against a vec3's 24. Since these arrays are what
// crosses the wire every step, shipping Particles would triple communication
// volume in a benchmark whose entire purpose is to exercise a communication-aware
// balancer. The Cell keeps its Particles device resident and gathers a positions
// array to send.
//
// shift displaces every B position, carrying the periodic wraparound that
// calcPairForces applied to the position array before looping. Applying it here
// keeps the position arrays untouched, which matters because they are device
// resident and read concurrently by several Computes -- mutating one for a
// neighbour's benefit would corrupt every other reader. calcPairForces shifted A
// by +diff, which is the same separation as shifting B by -diff, so callers pass
// the negation of the host code's diff; the reciprocal launch negates it again.
//
// selfInteract skips i == j, for the case where A and B are the same cell.
//
// energyPartial, when non-null, receives this block's share of the pair
// potential. Callers pass it only on the two steps that need energy, and only on
// the first of a pair's two launches, so each unordered pair is counted once.
__global__ void pairForceKernel(const vec3* __restrict__ A, int nA,
                                const vec3* __restrict__ B, int nB,
                                vec3* __restrict__ forceA,
                                vec3 shift, double cutoffSq, int selfInteract,
                                double* __restrict__ energyPartial)
{
  const int i = blockIdx.x;
  if (i >= nA) return;

  __shared__ double sx[BLOCK_THREADS];
  __shared__ double sy[BLOCK_THREADS];
  __shared__ double sz[BLOCK_THREADS];
  __shared__ double se[BLOCK_THREADS];

  const vec3 pos_i = A[i];

  double fx = 0.0, fy = 0.0, fz = 0.0, en = 0.0;

  for (int j = threadIdx.x; j < nB; j += blockDim.x)
  {
    if (selfInteract && i == j) continue;

    const vec3 pos_j = B[j];

    const double dx = pos_i.x - (pos_j.x + shift.x);
    const double dy = pos_i.y - (pos_j.y + shift.y);
    const double dz = pos_i.z - (pos_j.z + shift.z);

    double rsqd = dx*dx + dy*dy + dz*dz;

    // Same guard as the host version: rsqd > 1 rejects coincident/overlapping
    // atoms whose r^-12 term would blow up, cutoffSq is the interaction range.
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

      // calcInternalForces ran j from i+1, visiting each intra-cell pair once;
      // this kernel visits both orderings because it needs the force on each
      // atom separately, so the energy has to reject one of them. For a
      // two-cell pair every (i,j) is seen exactly once, hence no filter.
      if (energyPartial != nullptr && (!selfInteract || j > i))
        en += VDW_A_D / rTwelve - VDW_B_D / rSix;
    }
  }

  sx[threadIdx.x] = fx;
  sy[threadIdx.x] = fy;
  sz[threadIdx.x] = fz;
  se[threadIdx.x] = en;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (threadIdx.x < s)
    {
      sx[threadIdx.x] += sx[threadIdx.x + s];
      sy[threadIdx.x] += sy[threadIdx.x + s];
      sz[threadIdx.x] += sz[threadIdx.x + s];
      se[threadIdx.x] += se[threadIdx.x + s];
    }
    __syncthreads();
  }

  // One block owns forceA[i] outright, so this is a plain store, not an atomic.
  if (threadIdx.x == 0)
  {
    forceA[i].x += sx[0];
    forceA[i].y += sy[0];
    forceA[i].z += sz[0];
    if (energyPartial != nullptr) energyPartial[i] = se[0];
  }
}

// Copy the position field out of the device-resident Particle array. This is
// what a Cell sends to its Computes -- see the payload note on pairForceKernel.
__global__ void gatherPositionsKernel(const Particle* __restrict__ p,
                                      vec3* __restrict__ pos, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) pos[i] = p[i].pos;
}

// Sum an array of doubles to a single device scalar, one block, grid stride.
// Used to finish both the pair-potential and kinetic-energy partials so only 8
// bytes ever cross to the host.
__global__ void reduceDoublesKernel(const double* __restrict__ in, int n,
                                    double* __restrict__ out)
{
  __shared__ double s[BLOCK_THREADS];
  double acc = 0.0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) acc += in[i];
  s[threadIdx.x] = acc;
  __syncthreads();
  for (int k = blockDim.x / 2; k > 0; k >>= 1)
  {
    if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
    __syncthreads();
  }
  if (threadIdx.x == 0) *out = s[0];
}

// Zero a force array between steps.
__global__ void zeroForcesKernel(vec3* f, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { f[i].x = 0.0; f[i].y = 0.0; f[i].z = 0.0; }
}

// Sum one incoming force contribution into a cell's accumulator.
//
// This is what replaces the section reduction of the original. A Charm++ reduction
// operates on host buffers, so keeping it would have forced every force array back
// through host memory every step -- the opposite of device residency. Instead each
// Compute ships its force array device-to-device and the owning Cell folds it in
// here.
__global__ void accumulateForcesKernel(vec3* __restrict__ dst,
                                       const vec3* __restrict__ src, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    dst[i].x += src[i].x;
    dst[i].y += src[i].y;
    dst[i].z += src[i].z;
  }
}

// Leapfrog integration, matching Cell::updateProperties. Runs on the device so the
// particle array is never read back to host.
//
// TWO deltas, not one. updateProperties advances velocity by
// DEFAULT_DELTA * 10^-20 (acceleration is in m/s^2, velocity in A/fs) but advances
// position by plain DEFAULT_DELTA. Collapsing them into a single dt -- which is
// what this kernel did in its first draft -- rescales the dynamics by 10^20 and
// the simulation ceases to conserve anything.
__global__ void integrateKernel(Particle* __restrict__ p, const vec3* __restrict__ f,
                                int n, double dtVel, double dtPos,
                                double maxVelocity)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const double invMass = 1.0 / p[i].mass;

  p[i].acc.x = f[i].x * invMass;
  p[i].acc.y = f[i].y * invMass;
  p[i].acc.z = f[i].z * invMass;

  p[i].vel.x += p[i].acc.x * dtVel;
  p[i].vel.y += p[i].acc.y * dtVel;
  p[i].vel.z += p[i].acc.z * dtVel;

  // Clamp to the same terminal velocity the host integrator enforces, which keeps
  // an atom from crossing more than one cell in a step.
  if (p[i].vel.x >  maxVelocity) p[i].vel.x =  maxVelocity;
  if (p[i].vel.x < -maxVelocity) p[i].vel.x = -maxVelocity;
  if (p[i].vel.y >  maxVelocity) p[i].vel.y =  maxVelocity;
  if (p[i].vel.y < -maxVelocity) p[i].vel.y = -maxVelocity;
  if (p[i].vel.z >  maxVelocity) p[i].vel.z =  maxVelocity;
  if (p[i].vel.z < -maxVelocity) p[i].vel.z = -maxVelocity;

  p[i].pos.x += p[i].vel.x * dtPos;
  p[i].pos.y += p[i].vel.y * dtPos;
  p[i].pos.z += p[i].vel.z * dtPos;
}

// Kinetic energy per atom, for the conservation check. Reduced on device; only the
// scalar crosses to host. The 10^10 matches updateProperties -- it puts the result
// in the same milliJoules as the pair potential, without which the two halves of
// the conservation test are in different units.
__global__ void kineticEnergyKernel(const Particle* __restrict__ p, int n,
                                    double* __restrict__ partial)
{
  __shared__ double s[BLOCK_THREADS];
  double acc = 0.0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x)
  {
    const vec3 v = p[i].vel;
    acc += 0.5 * p[i].mass * (v.x*v.x + v.y*v.y + v.z*v.z) * POW_TEN_POS;
  }
  s[threadIdx.x] = acc;
  __syncthreads();
  for (int k = blockDim.x / 2; k > 0; k >>= 1)
  {
    if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
    __syncthreads();
  }
  if (threadIdx.x == 0) partial[blockIdx.x] = s[0];
}

// ---------------------------------------------------------------------------
// Particle migration, on device.
//
// Cell::migrateParticles sorted the particle vector on the host every
// MIGRATE_STEPCOUNT steps. Keeping that would mean a full D2H of the primary
// simulation state and an H2D of what is left -- the single largest violation of
// device residency in the application. So the classify, wrap and compact all
// happen here, and the outgoing buckets are shipped device-to-device.
// ---------------------------------------------------------------------------

// Which neighbour a particle belongs to now, as a (dx,dy,dz) in [-1,1].
//
// migrateToCell also produced +/-2, for a particle that crossed two cells in one
// migration interval. That cannot happen here -- MAX_VELOCITY is 0.1 A/fs and
// DEFAULT_DELTA is 1 fs, so 20 steps move an atom at most 2 A against a 30 A cell
// -- and the host code would have indexed past the end of its 27-entry vector if
// it ever did. Clamping keeps an unreachable case from becoming a memory stomp.
__device__ inline int migrateOffset(double p, double origin, double cellSize)
{
  if (p < origin) return -1;
  if (p > origin + cellSize) return 1;
  return 0;
}

__device__ inline double wrapCoord(double v, double origin, double extent)
{
  if (v < origin) v += extent;
  if (v > origin + extent) v -= extent;
  return v;
}

__global__ void binParticlesKernel(const Particle* __restrict__ p, int n,
                                   int cellX, int cellY, int cellZ,
                                   double cellSizeX, double cellSizeY,
                                   double cellSizeZ,
                                   int dimX, int dimY, int dimZ,
                                   Particle* __restrict__ stay,
                                   Particle* __restrict__ send,
                                   int exchCapacity,
                                   int* __restrict__ counts)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const double ox = cellX * cellSizeX + CELL_ORIGIN_X;
  const double oy = cellY * cellSizeY + CELL_ORIGIN_Y;
  const double oz = cellZ * cellSizeZ + CELL_ORIGIN_Z;

  Particle q = p[i];

  const int dx = migrateOffset(q.pos.x, ox, cellSizeX);
  const int dy = migrateOffset(q.pos.y, oy, cellSizeY);
  const int dz = migrateOffset(q.pos.z, oz, cellSizeZ);

  const int bucket = (dx + KAWAY_X) * NBRS_Y * NBRS_Z
                   + (dy + KAWAY_Y) * NBRS_Z
                   + (dz + KAWAY_Z);

  // SELF_BUCKET is the (0,0,0) offset -- the particle stays put.
  const int SELF_BUCKET = KAWAY_X * NBRS_Y * NBRS_Z + KAWAY_Y * NBRS_Z + KAWAY_Z;

  const int slot = atomicAdd(&counts[bucket], 1);

  if (bucket == SELF_BUCKET)
  {
    stay[slot] = q;
  }
  else
  {
    // wrapAround, applied only to the copy that is leaving.
    q.pos.x = wrapCoord(q.pos.x, CELL_ORIGIN_X, cellSizeX * dimX);
    q.pos.y = wrapCoord(q.pos.y, CELL_ORIGIN_Y, cellSizeY * dimY);
    q.pos.z = wrapCoord(q.pos.z, CELL_ORIGIN_Z, cellSizeZ * dimZ);
    // Overflow is reported by the host from the count; dropping the write keeps
    // it from corrupting the neighbouring bucket in the meantime.
    if (slot < exchCapacity) send[bucket * exchCapacity + slot] = q;
  }
}

// Copy an arriving bucket onto the tail of the particle array.
__global__ void appendParticlesKernel(Particle* __restrict__ dst, int dstOffset,
                                      const Particle* __restrict__ src, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[dstOffset + i] = src[i];
}

// ---------------------------------------------------------------------------
// Host-side launchers. Everything takes an explicit stream so callers can overlap
// and so completion can be signalled through hapiAddCallback rather than a
// synchronous wait.
// ---------------------------------------------------------------------------

void invokePairForce(const vec3* d_A, int nA, const vec3* d_B, int nB,
                     vec3* d_forceA, vec3 shift, double cutoffSq, bool selfInteract,
                     double* d_energyPartial, cudaStream_t stream)
{
  if (nA <= 0 || nB <= 0) return;
  // One block per atom of A -- see the decomposition note at the top.
  pairForceKernel<<<nA, BLOCK_THREADS, 0, stream>>>(
      d_A, nA, d_B, nB, d_forceA, shift, cutoffSq, selfInteract ? 1 : 0,
      d_energyPartial);
}

void invokeGatherPositions(const Particle* d_p, vec3* d_pos, int n,
                           cudaStream_t stream)
{
  if (n <= 0) return;
  const int blocks = (n + BLOCK_THREADS - 1) / BLOCK_THREADS;
  gatherPositionsKernel<<<blocks, BLOCK_THREADS, 0, stream>>>(d_p, d_pos, n);
}

void invokeReduceDoubles(const double* d_in, int n, double* d_out,
                         cudaStream_t stream)
{
  if (n <= 0) return;
  reduceDoublesKernel<<<1, BLOCK_THREADS, 0, stream>>>(d_in, n, d_out);
}

void invokeZeroForces(vec3* d_f, int n, cudaStream_t stream)
{
  if (n <= 0) return;
  const int blocks = (n + BLOCK_THREADS - 1) / BLOCK_THREADS;
  zeroForcesKernel<<<blocks, BLOCK_THREADS, 0, stream>>>(d_f, n);
}

void invokeAccumulateForces(vec3* d_dst, const vec3* d_src, int n,
                            cudaStream_t stream)
{
  if (n <= 0) return;
  const int blocks = (n + BLOCK_THREADS - 1) / BLOCK_THREADS;
  accumulateForcesKernel<<<blocks, BLOCK_THREADS, 0, stream>>>(d_dst, d_src, n);
}

void invokeIntegrate(Particle* d_p, const vec3* d_f, int n, double dtVel,
                     double dtPos, double maxVelocity, cudaStream_t stream)
{
  if (n <= 0) return;
  const int blocks = (n + BLOCK_THREADS - 1) / BLOCK_THREADS;
  integrateKernel<<<blocks, BLOCK_THREADS, 0, stream>>>(
      d_p, d_f, n, dtVel, dtPos, maxVelocity);
}

void invokeKineticEnergy(const Particle* d_p, int n, double* d_partial, int nBlocks,
                         cudaStream_t stream)
{
  if (n <= 0) return;
  kineticEnergyKernel<<<nBlocks, BLOCK_THREADS, 0, stream>>>(d_p, n, d_partial);
}

void invokeBinParticles(const Particle* d_p, int n, int cellX, int cellY, int cellZ,
                        double cellSizeX, double cellSizeY, double cellSizeZ,
                        int dimX, int dimY, int dimZ, Particle* d_stay,
                        Particle* d_send, int exchCapacity, int* d_counts,
                        cudaStream_t stream)
{
  if (n <= 0) return;
  const int blocks = (n + BLOCK_THREADS - 1) / BLOCK_THREADS;
  binParticlesKernel<<<blocks, BLOCK_THREADS, 0, stream>>>(
      d_p, n, cellX, cellY, cellZ, cellSizeX, cellSizeY, cellSizeZ,
      dimX, dimY, dimZ, d_stay, d_send, exchCapacity, d_counts);
}

void invokeAppendParticles(Particle* d_dst, int dstOffset, const Particle* d_src,
                           int n, cudaStream_t stream)
{
  if (n <= 0) return;
  const int blocks = (n + BLOCK_THREADS - 1) / BLOCK_THREADS;
  appendParticlesKernel<<<blocks, BLOCK_THREADS, 0, stream>>>(
      d_dst, dstOffset, d_src, n);
}
