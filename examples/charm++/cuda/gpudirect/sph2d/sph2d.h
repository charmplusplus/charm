#ifndef __CUDA_GPUDIRECT_SPH2D_H_
#define __CUDA_GPUDIRECT_SPH2D_H_

// Weakly-compressible SPH dam break, decomposed over a 2D chare array.
//
// The point of this example is load imbalance that is physical rather than
// imposed. A particle method has no per-cell floor: an empty region of the tank
// costs exactly nothing, because there are no particles there to interact. The
// water column starts in one corner and collapses across the tank, so the work
// is concentrated where the fluid is and that region MOVES -- which is what a
// balancer has to track, and what a static or synthetic imbalance cannot test.
//
// Formulation (all standard WCSPH, see Monaghan 2005, Colagrossi 2003):
//   kernel      Wendland C2, support 2h
//   density     continuity equation dRho/dt = sum m_j v_ij . grad W  (+ delta-SPH)
//   pressure    Tait, p = (rho0 c0^2/gamma)((rho/rho0)^gamma - 1)
//   viscosity   Monaghan artificial viscosity
//   boundary    dynamic boundary particles (density evolves, position fixed)
//
// Continuity rather than summation density is deliberate: rho travels with the
// particle, so a step needs ONE halo exchange rather than two (positions, then
// densities of the same ghosts).

typedef float RealType;

// 32 bytes, so a halo/migration message is a clean multiple of the transfer
// width. type is an int rather than a bitfield so the struct stays trivially
// copyable for device zerocopy.
struct Particle {
  RealType x, y;        // position
  RealType vx, vy;      // velocity
  RealType rho;         // density (evolved)
  RealType p;           // pressure (derived from rho each step)
  int type;             // PTYPE_FLUID or PTYPE_BOUND
  int id;               // global lattice id, unique and fixed for the run
};

#define PTYPE_FLUID 0
#define PTYPE_BOUND 1

// Directions, tagged from the receiver's perspective -- the side of the
// receiving patch the message arrived on. Same convention as pic2d.
enum Dir { LEFT = 0, RIGHT, TOP, BOTTOM, TL, TR, BL, BR, NUM_DIRS = 8, STAY = 8 };

// d_counts layout: [0..7] outgoing counts per direction, [8] staying,
// [9] error flag (a particle moved further than one patch in one step).
#define NUM_COUNTERS 10
#define ERR_COUNTER 9

// Wendland C2 in 2D, support radius 2h.
//   W(q)  = (7/(4 pi h^2)) (1 - q/2)^4 (2q + 1),      q = r/h, 0 <= q <= 2
//   W'(r) = -(35/(4 pi h^3)) q (1 - q/2)^3
#define KERNEL_SUPPORT 2.0f

// Tait equation of state exponent.
#define EOS_GAMMA 7.0f

// Monaghan artificial viscosity coefficient, and the delta-SPH density
// diffusion coefficient. 0.01-0.1 and 0.1 respectively are the usual choices.
#define VISC_ALPHA 0.05f
#define DELTA_SPH 0.1f

// Softening in the mu_ij denominator, in units of h^2.
#define ETA2 0.01f


// ---------------------------------------------------------------- checks ----
// Correctness checking. Ordering in this application is NOT reproducible: the
// compaction, the halo pack and the cell scatter all place particles with
// atomicAdd, so the neighbour sums are summed in a different order every run
// and the float state diverges in the low bits. A float checksum could
// therefore never be compared between two runs, let alone between no-LB and
// LB. So every check below is built out of quantities that are exact and
// INDEPENDENT OF ORDER -- 64-bit integers under wrapping addition, which is
// associative and commutative, so the global value does not depend on how the
// particles are spread over patches, on the order they arrive in, or on the
// order the reduction combines them. That is what makes the same number
// comparable across no-LB, sync LB and async LB.
//
//   id sum        sum of a 64-bit mix of every particle's id. Constant while
//                 no particle leaves the domain: catches a particle lost or
//                 duplicated by the halo/migration path, and any corruption
//                 of the id field itself.
//   boundary sum  the same over the boundary particles' ids AND positions.
//                 A boundary particle never moves, so this is a constant of
//                 the whole run -- and boundary particles are interleaved with
//                 fluid throughout the local array, so it is a direct test
//                 that a migration's device-to-device pup moved the bits
//                 unchanged.
//   counts        fluid and boundary particle counts, separately, so a
//                 particle that changes type is caught even though the total
//                 is unchanged.
//   validity      per-particle: finite, density and speed in range, type
//                 known, and inside the owning patch's rectangle. The last
//                 one tests the exchange's routing: a particle handed to the
//                 wrong neighbour lands outside that neighbour's rectangle.
#define CHK_ID_SUM     0
#define CHK_BND_SUM    1
#define CHK_N_FLUID    2
#define CHK_N_BOUND    3
#define CHK_BAD_MASK   4
#define CHK_BAD_COUNT  5
#define NUM_CHECKS     8

#define CHK_BAD_NAN      0x01ull   // non-finite position, velocity, rho or p
#define CHK_BAD_RHO      0x02ull   // density far outside the weakly-compressible band
#define CHK_BAD_SPEED    0x04ull   // speed past Mach 0.5, i.e. the run has blown up
#define CHK_BAD_TYPE     0x08ull   // type field is neither FLUID nor BOUND
#define CHK_BAD_OUTSIDE  0x10ull   // local particle outside its own patch rectangle

#if !defined(__CUDACC__)
#include "pup.h"
PUPbytes(Particle)
#endif

#endif // __CUDA_GPUDIRECT_SPH2D_H_
