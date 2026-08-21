#ifndef __LEANMD_CUDA_H__
#define __LEANMD_CUDA_H__

// Host-visible interface to the device kernels in leanmd.cu.
//
// Included by both the .cu (which sees the real CUDA types) and the Charm++ .cc
// files, so it must not pull in anything nvcc-only beyond cuda_runtime.h.
//
// The design constraint behind all of these: particle and force arrays are DEVICE
// RESIDENT for the lifetime of a Cell. They are allocated once on the GPU, read and
// written only by kernels, exchanged between chares by device-to-device zerocopy,
// and never staged through host memory. The only things that cross to the host are
// scalars -- a kinetic energy, a particle count -- and the migration payload when a
// chare actually moves.

#include <cuda_runtime.h>

struct vec3;
struct Particle;

// Forces on every atom of A due to every atom of B.
//
// Takes positions, not Particles: 24 bytes per atom on the wire instead of 80,
// and the kernel reads nothing else. Cells keep Particles device resident and
// gather positions with invokeGatherPositions before sending.
//
// shift displaces B's positions to carry periodic wraparound, so the position
// arrays themselves are never mutated on a neighbour's behalf -- they belong to
// their owning Cell and are read concurrently by several Computes. It is the
// negation of the diff that calcPairForces applied to A; the reciprocal launch,
// with the two cells swapped, negates it again.
//
// selfInteract skips i == j, for a cell interacting with itself.
//
// d_energyPartial, when non-null, receives nA per-block pair-potential terms to
// be finished by invokeReduceDoubles. Pass it on exactly one of a pair's two
// launches, or every interaction is counted twice.
//
// Accumulates into d_forceA rather than overwriting, so a Compute handling several
// cell pairs can fold them into one buffer.
void invokePairForce(const vec3* d_A, int nA, const vec3* d_B, int nB,
                     vec3* d_forceA, vec3 shift, double cutoffSq, bool selfInteract,
                     double* d_energyPartial, cudaStream_t stream);

void invokeZeroForces(vec3* d_f, int n, cudaStream_t stream);

// Extract positions from the device-resident Particle array, for sending.
void invokeGatherPositions(const Particle* d_p, vec3* d_pos, int n,
                           cudaStream_t stream);

// Finish a partials array into one device scalar, so only 8 bytes go to host.
void invokeReduceDoubles(const double* d_in, int n, double* d_out,
                         cudaStream_t stream);

// Fold one received force contribution into a Cell's accumulator. Replaces the
// original's section reduction, which was host-side and would have forced every
// force array back through host memory each step.
void invokeAccumulateForces(vec3* d_dst, const vec3* d_src, int n,
                            cudaStream_t stream);

// Leapfrog update, on device, so the particle array is never read back.
//
// dtVel and dtPos are genuinely different: updateProperties advances velocity by
// DEFAULT_DELTA * 1e-20 and position by plain DEFAULT_DELTA, because acceleration
// is in m/s^2 while velocity is in A/fs.
void invokeIntegrate(Particle* d_p, const vec3* d_f, int n, double dtVel,
                     double dtPos, double maxVelocity, cudaStream_t stream);

// Per-block partial kinetic energies; finish with invokeReduceDoubles.
void invokeKineticEnergy(const Particle* d_p, int n, double* d_partial, int nBlocks,
                         cudaStream_t stream);

// Classify, wrap and compact particles for migration, replacing the host-side sort
// in Cell::migrateParticles. Stayers land in d_stay, movers in bucket-major order
// in d_send; d_counts (NUM_NEIGHBORS ints, zeroed by the caller) holds the per
// bucket totals and is the only thing the host reads back.
void invokeBinParticles(const Particle* d_p, int n, int cellX, int cellY, int cellZ,
                        double cellSizeX, double cellSizeY, double cellSizeZ,
                        int dimX, int dimY, int dimZ, Particle* d_stay,
                        Particle* d_send, int exchCapacity, int* d_counts,
                        cudaStream_t stream);

// Append an arriving bucket to the tail of a Cell's particle array.
void invokeAppendParticles(Particle* d_dst, int dstOffset, const Particle* d_src,
                           int n, cudaStream_t stream);

#endif
