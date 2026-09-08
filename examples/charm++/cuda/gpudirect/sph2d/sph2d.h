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
  int pad;
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

#if !defined(__CUDACC__)
#include "pup.h"
PUPbytes(Particle)
#endif

#endif // __CUDA_GPUDIRECT_SPH2D_H_
