#ifndef __CUDA_GPUDIRECT_SPH3D_H_
#define __CUDA_GPUDIRECT_SPH3D_H_

// Weakly-compressible SPH dam break in THREE dimensions, decomposed over a 3D
// chare array. A port of sph2d (see sph2d/sph2d.h for the rationale and the
// correctness-check design, which are unchanged); the differences are the
// third coordinate, 26 exchange partners instead of 8, 3D cell lists and the
// 3D Wendland kernel. z is vertical (gravity acts along -z), y is the tank's
// depth, x the direction the water column collapses along.
//
// Formulation (all standard WCSPH, see Monaghan 2005, Colagrossi 2003):
//   kernel      Wendland C2, support 2h
//   density     continuity equation dRho/dt = sum m_j v_ij . grad W  (+ delta-SPH)
//   pressure    Tait, p = (rho0 c0^2/gamma)((rho/rho0)^gamma - 1)
//   viscosity   Monaghan artificial viscosity
//   boundary    dynamic boundary particles (density evolves, position fixed)

typedef float RealType;

// 40 bytes; every field is 4 bytes wide, which is what lets a device pup ship
// the array as RealType. type is an int rather than a bitfield so the struct
// stays trivially copyable for device zerocopy.
struct Particle {
  RealType x, y, z;     // position
  RealType vx, vy, vz;  // velocity
  RealType rho;         // density (evolved)
  RealType p;           // pressure (derived from rho each step)
  int type;             // PTYPE_FLUID or PTYPE_BOUND
  int id;               // global lattice id, unique and fixed for the run
};

// Advertisement bits, carried on every migration message (see sph3d.C).
#define ADV_ACTIVE 1
#define ADV_FLUID  2
#define ADV_ANY    1

#define PTYPE_FLUID 0
#define PTYPE_BOUND 1

// Directions. A partner is an offset (sx,sy,sz) in {-1,0,1}^3, indexed as
// (sx+1)*9 + (sy+1)*3 + (sz+1); 13 is (0,0,0), which no partner is, so that
// slot doubles as the "stays here" counter of the compaction. Flipping every
// component (the receiver's view of the sender's direction) is 26 - d.
#define NUM_DIRS 27
#define STAY 13

// d_counts layout: [0..26] outgoing counts per direction (13 = staying),
// [27] error flag (a particle moved further than one patch in one step, or a
// per-direction buffer overflowed).
#define NUM_COUNTERS 28
#define ERR_COUNTER 27

#ifdef __CUDACC__
#define SPH_HD __host__ __device__
#else
#define SPH_HD
#endif
SPH_HD inline int dirIndex(int sx, int sy, int sz) { return (sx + 1) * 9 + (sy + 1) * 3 + (sz + 1); }
SPH_HD inline int dirSX(int d) { return d / 9 - 1; }
SPH_HD inline int dirSY(int d) { return (d / 3) % 3 - 1; }
SPH_HD inline int dirSZ(int d) { return d % 3 - 1; }
SPH_HD inline int dirNonzero(int d) { return (dirSX(d) != 0) + (dirSY(d) != 0) + (dirSZ(d) != 0); }
SPH_HD inline int flipDir(int d) { return 26 - d; }
// Per-direction exchange slot. What crosses a face is a slab one support
// thick, what crosses an edge is a bar support x support, a corner a cube:
// each step down is a factor of (support / patch side), i.e. tens. Giving
// every one of the 26 slots the face size would multiply the exchange memory
// by 27; faces get the full slot, edges a quarter, corners a sixteenth. The
// pack kernels check against this and raise ERR_COUNTER on overflow.
SPH_HD inline int dirCap(int d, int cap_face) { return cap_face >> (2 * (dirNonzero(d) - 1)); }

// Wendland C2 in 3D, support radius 2h.
//   W(q)  = (21/(16 pi h^3)) (1 - q/2)^4 (2q + 1),    q = r/h, 0 <= q <= 2
//   W'(r) = -(105/(16 pi h^4)) q (1 - q/2)^3
#define KERNEL_SUPPORT 2.0f

// Tait equation of state exponent.
#define EOS_GAMMA 7.0f

// Monaghan artificial viscosity coefficient, and the delta-SPH density
// diffusion coefficient.
#define VISC_ALPHA 0.05f
#define DELTA_SPH 0.1f

// Softening in the mu_ij denominator, in units of h^2.
#define ETA2 0.01f

// ---------------------------------------------------------------- checks ----
// Exact, order-invariant 64-bit integer checks; see sph2d.h for the design.
#define CHK_ID_SUM     0
#define CHK_BND_SUM    1
#define CHK_N_FLUID    2
#define CHK_N_BOUND    3
#define CHK_BAD_MASK   4
#define CHK_BAD_COUNT  5
#define NUM_CHECKS     8

#define CHK_BAD_NAN      0x01ull
#define CHK_BAD_RHO      0x02ull
#define CHK_BAD_SPEED    0x04ull
#define CHK_BAD_TYPE     0x08ull
#define CHK_BAD_OUTSIDE  0x10ull

#if !defined(__CUDACC__)
#include "pup.h"
PUPbytes(Particle)
#endif

#endif // __CUDA_GPUDIRECT_SPH3D_H_

// ------------------------------------------------- integrator and windows ----
// Two options make the cost structure match GPUSPH's (see sph3d.C):
//
//   -p   predictor-corrector, GPUSPH's integrator: two force evaluations per
//        step with a halo exchange of the predicted state in between.
//          x* = x + v dt/2,  v* = v + a dt/2,  rho* = rho + drho dt/2
//          x1 = x + (v + a* dt/2) dt,  v1 = v + a* dt,  rho1 = rho + drho* dt
//        The default is Euler-Cromer, one evaluation per step.
//
//   -k N neighbour-list window: the cell list, the halo MEMBERSHIP and the
//        migration happen every N steps (GPUSPH rebuilds every 10). Inside a
//        window the particle order is fixed, every halo exchange re-sends the
//        same particles by index into the same ghost slots, forces walk the
//        cells a particle was binned into at the rebuild, and a particle that
//        has drifted out of its patch is integrated by its old owner until the
//        next migration. Like GPUSPH (nlexpansionfactor 1), a pair that was
//        further than one cell apart at the rebuild and comes within 2h before
//        the next one is missed; -m adds a Verlet margin (in h) to the cell
//        size and the halo depth to close that gap at the price of more
//        candidate pairs.
