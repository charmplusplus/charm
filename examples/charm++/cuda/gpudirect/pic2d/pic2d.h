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

#define IDX(x,y) ((block_width+2)*(y)+(x))

#if !defined(__CUDACC__)
#include "pup.h"
PUPbytes(Particle)
#endif

#endif // __CUDA_GPUDIRECT_PIC2D_H_
