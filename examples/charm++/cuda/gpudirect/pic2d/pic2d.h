#ifndef __CUDA_GPUDIRECT_PIC2D_H_
#define __CUDA_GPUDIRECT_PIC2D_H_

typedef float RealType;

struct Particle {
  RealType x, y, vx, vy;
};

// Directions for ghost/particle exchange, always tagged from the receiver's
// perspective (the side of the receiving patch the message arrived on).
// The first four (face neighbors) are used by the phi exchange; all eight
// are used for charge ghosts, E-field ghosts and particle exchange.
enum Dir { LEFT = 0, RIGHT, TOP, BOTTOM, TL, TR, BL, BR, NUM_DIRS = 8, STAY = 8 };

// d_counts layout: [0..7] outgoing particle counts per direction,
// [8] particles staying in the patch, [9] error flag (a particle moved
// further than one patch in a single step)
#define NUM_COUNTERS 10
#define ERR_COUNTER 9

// Halo depth of the field arrays, in cells, and hence how many Jacobi sweeps
// can run between phi exchanges (temporal blocking).
//
// The Poisson solve was one 4-way ghost exchange per sweep: with -j 8 that is
// nine round trips per timestep, each gating a single ~4us kernel over a 128^2
// patch. Measured on Delta A40s the device was busy 1-2% of the time and the
// step was ~90% halo latency, so the sweeps are effectively free and the
// exchanges are everything.
//
// With a halo of H the patch can take H sweeps before it needs fresh
// boundaries: sweep s updates cells out to margin (H-1-s), each sweep
// consuming one layer of the halo it was given.
//
// Bounded by the *rho* halo, not this one: the update reads rho at every cell
// it writes, and the charge exchange fills one ghost layer. H=2 is therefore
// the most that works without widening that exchange too -- which is the next
// step if more is wanted.
#define PHI_HALO 1

// Reduces to ((block_width+2)*(y)+(x)) at PHI_HALO 1, so interior indices are
// unchanged and only the array grows. Negative x/y reach the outer halo.
#define IDX(x,y) ((block_width+2*PHI_HALO)*((y)+PHI_HALO-1)+((x)+PHI_HALO-1))

// Corner block a diagonal neighbour must supply, per side. A margin-m sweep
// writes the halo out to layer m and reads one layer beyond, and near a corner
// that read lands in the diagonal neighbour's territory -- which the 4-way
// cross exchange never covers. Zero at PHI_HALO 1, where nothing reads the
// halo at all.
// H deep, not H-1. Sweep 1 runs at margin H-1 and writes the whole square,
// including its corner (2-H, 2-H), whose stencil reaches (1-H, 2-H) -- layer H
// in x. Later sweeps then read the corner cells earlier sweeps wrote, so the
// exchange has to supply an H x H block per diagonal.
#define PHI_CORNER_D ((PHI_HALO > 1) ? PHI_HALO : 0)
// Charge must be valid as deep as the deepest sweep writes, one layer less
// than phi, which the sweep reads one beyond where it writes.
#define RHO_HALO_D   (PHI_HALO - 1)
#define PHI_CORNER   (PHI_CORNER_D * PHI_CORNER_D)
// 4 directions when no corner data is needed, 8 once it is.
#define PHI_DIRS     ((PHI_HALO > 1) ? NUM_DIRS : 4)

// Cells per row/column including both halos.
#define FIELD_W (block_width + 2*PHI_HALO)
#define FIELD_H (block_height + 2*PHI_HALO)

#if !defined(__CUDACC__)
#include "pup.h"
PUPbytes(Particle)
#endif

#endif // __CUDA_GPUDIRECT_PIC2D_H_
