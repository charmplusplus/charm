#ifndef __DEFAULTS_H__
#define __DEFAULTS_H__

#define DEFAULT_THETA 0.5 
#define DEFAULT_DT 0
#define DEFAULT_DTIME 0.025
#define DEFAULT_EPS 0.05
#define DEFAULT_TOL 1.0


#define DEFAULT_PPC 1000

// Particles per bucket. The GPU kernel gives one block of GRAV_BLOCK_SIZE
// threads to a bucket and one thread to each of its particles, so a bucket of
// 10 leaves 118 of 128 threads idle. The tree granularity is a free parameter
// -- it trades interaction-list length against approximation error -- so each
// build gets the value that suits its force loop.
#ifdef GPU_GRAVITY
#define DEFAULT_PPB 128
#else
#define DEFAULT_PPB 10
#endif
#define DEFAULT_KILLAT 10
#define DEFAULT_CHUNK_DEPTH 3
#define DEFAULT_YIELD_PERIOD 5
#define DEFAULT_TREE_PIECES_PER_PROC 8

// Load balancing is off unless -lbperiod is given, so a run with no extra
// arguments behaves as it did before.
#define DEFAULT_FIRST_LB_ITERATION 2
#define DEFAULT_LB_PERIOD 0

// The split barrier is opt-in: it needs +LBAsync on the command line to do
// anything, and the unsplit path is what every existing run used.
#define DEFAULT_ASYNC_LB 0

// Instrument the two iterations before each balancing iteration. The same
// switch gates CUPTI tracing, which is the expensive half, so a window is what
// keeps a load-balanced run from paying for tracing it never reads. 0 restores
// continuous instrumentation.
#define DEFAULT_LB_WINDOW 2

// 1M sources is 16 MB of pinned staging per tree piece, and only reached by a
// tree piece holding an unusually large share of the PE's particles.
#define DEFAULT_GPU_FLUSH_LIMIT (1 << 20)
#endif
