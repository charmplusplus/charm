
#ifndef __DEFS__
#define __DEFS__

#include <cstdlib>
#include "pup.h"

#define HYDROGEN_MASS           (1.67 * pow( 10.0,-24)) // in g
#define VDW_A                   (1.1328 * pow(10.0, -133)) // in (g m^2/s^2) m^12
#define VDW_B                   (2.23224 * pow(10.0, -76)) // in (g m^2/s^2) m^6

#define ENERGY_VAR              (1.0 * pow(10.0,-5))

//average of next two should be what you want as your atom density
//this should comply with the PERDIM parameter; for KAWAY 1 1 1, the maximum number
//of particles can be 10*10*10 = 1000 - 10 comes from PERDIM parameter, which is
//currently set to be 10, using a GAP of 3; as KWAYness increases, the maximum
//number of particles decreases - for 2 1 1, it is 500, for 2 2 1 it is 250; you
//can set them to have lower values but not higher; alternatively a host of
//paramters including PTP_CUT_OFF, PERDIM, GAP can be set to suitable values to
// L0 granularity decision (LEANMD_PORT_PLAN.html): these must be large enough that
// one Compute's pair kernel can occupy the device, or this benchmark reproduces
// pic2d's occupancy pathology -- kernels too narrow to fill the SMs, SM-normalised
// GPU load near zero, and an elasticity of GPU work to step time around 0.3.
//
// A Compute evaluates N_A x N_B pairs. On an A40 (84 SMs):
//
//     atoms/cell   pairs      blocks(256)   SM occupancy
//        100        10 K           40          7.9%     <- stock minimum
//        250        62 K          245         48.6%     <- stock maximum
//       1000         1 M        3,907        100.0%
//
// So the stock configuration launches 40-block kernels; pic2d's Jacobi was 64.
// Memory is not the constraint -- 1000 atoms is only ~78 KB of device state per
// cell -- so the answer is simply to go wider.
//
// The lattice caps how many atoms fit: PERDIM^3 sites of spacing GAP must fit in a
// cell of side (PTP_CUT_OFF + CELL_MARGIN)/KAWAY. At KAWAY 2,2,1 that cap is 250,
// which is exactly why the stock values are 100-250. Dropping to KAWAY 1,1,1 gives
// a cell of side 30 = PERDIM(10) x GAP(3), so 10^3 = 1000 sites.
//
// L1 granularity (measured 14 Sep 2026 on gpub023, 4 A40 / 32 PEs): the L0
// decision above fixed the kernel width but not what binds the step. A sweep at
// 64, 128, 192 and 256 cells per device held noLB at a flat 0.54-0.56 ms per
// cell and the device share at 32-41% of the interval, while the launch (driver)
// share climbed to 51%. Adding cells adds objects, and an object costs its
// messages and its launches whatever its atom count, so the step never becomes
// device-bound; the balancer's win decayed from 23% at 128 cells/GPU to nothing
// at 256, because LeanMD's imbalance is in work PER object (which a balancer can
// move) while the cost that grows with size is per object (which it cannot).
//
// Device time per Compute goes as N_A x N_B, so the lever is atoms per cell, not
// cells. GAP stays 3: the LJ minimum for these constants is at
// r = (2A/B)^(1/6) = 3.18, so anything tighter starts every atom inside the
// r^-12 wall. PERDIM x GAP must equal the cell side, so holding more atoms means
// a wider cell, and CELL_MARGIN is the way to widen it without touching the
// physics: side 42 = PERDIM(14) x GAP(3) gives 14^3 = 2744 sites, 2.7x the atoms
// and ~6x the pair work per Compute, which should put the device at 70-80% of
// the step. PTP_CUT_OFF stays 26, so the interaction physics is unchanged and a
// cell wider than the cutoff only makes the 1-away decomposition more
// conservative -- fewer, fatter kernels, which is the trade this benchmark wants.
//
// The stock granularity, for comparison runs: CELL_MARGIN 4, PERDIM 10,
// densities 200 / 800 / 1000.
#define PARTICLES_PER_CELL_START        2200
#define PARTICLES_PER_CELL_END          2744

// Atom count for the sparse cells of the imbalanced profiles below. A Compute
// evaluates N_A x N_B pairs, so 550 against 2744 is a 25x spread in work between
// the lightest and heaviest pair -- far more than any measurement noise.
#define PARTICLES_PER_CELL_LIGHT        550

// Where the load imbalance comes from.
//
// The stock benchmark is uniform by construction and cannot show a load balancer
// doing anything useful: measured GPU work per device came out even to 1.5%, and
// DiffusionLB correctly moved 96 objects, then 19, then 8, then none. Two things
// make it uniform, and both have to be switched for an imbalance to survive:
//
//   -density     the atom count per cell. The stock ramp is 800 -> 1000 across the
//                linearised cell index, a 1.56x spread in pair work, which is real
//                but small.
//   -computemap  where Computes are placed. Cell::createComputes inserts each one
//                round-robin over every PE, which averages *any* density profile
//                away -- each PE ends up with a sample of the whole box. Placing a
//                Compute on the PE of one of its cells keeps the spatial structure,
//                so a density profile becomes a PE (and device) imbalance.
//
// CellMap blocks cells along the linearised index z + y*Z + x*Y*Z, so x is the axis
// that maps onto whole processes: a gradient or a dense half-box in x lands as a
// gradient across the four devices rather than being smeared within each.
enum DensityMode { DENSITY_UNIFORM = 0, DENSITY_GRADIENT, DENSITY_CLUMP };
enum ComputeMapMode { COMPUTEMAP_RR = 0, COMPUTEMAP_LOCAL };

#define DEFAULT_DELTA           1	// in femtoseconds

#define DEFAULT_FIRST_LDB       20
#define DEFAULT_LDB_PERIOD      20
// Steps a chare keeps running between AtSyncStart() and AtSyncWait() under
// -lbasync. This is the whole overlap window: the strategy runs during it, and
// nothing migrates until the park at the end. 0 reproduces the unsplit barrier.
#define DEFAULT_LB_LAG          1
// Steps before a balancing step that LB instrumentation is left on for. Only
// these are measured, so this is the window the balancer actually decides from
// (Compute::updateInstrumentation).
#define DEFAULT_FT_PERIOD       100000

// 1-away decomposition: larger cells, so each Compute's pair kernel is wide enough
// to fill the device (see PARTICLES_PER_CELL_START above). Also cuts neighbours from
// 5x5x3 = 75 to 3x3x3 = 27, so 14 Computes per Cell instead of 38 -- fewer, fatter
// kernels, which is the trade this benchmark needs.
#define KAWAY_X                 1
#define KAWAY_Y                 1
#define KAWAY_Z                 1
#define NBRS_X	                (2*KAWAY_X+1)
#define NBRS_Y                  (2*KAWAY_Y+1)
#define NBRS_Z                  (2*KAWAY_Z+1)
#define NUM_NEIGHBORS           (NBRS_X * NBRS_Y * NBRS_Z)

#define CELLARRAY_DIM_X         3
#define CELLARRAY_DIM_Y         3
#define CELLARRAY_DIM_Z         3
#define PTP_CUT_OFF             26 // cut off for atom to atom interactions
#define CELL_MARGIN             16 // cell side - cutoff; 16 makes the cell 42 = PERDIM x GAP
#define CELL_SIZE_X             (PTP_CUT_OFF + CELL_MARGIN)/KAWAY_X
#define CELL_SIZE_Y             (PTP_CUT_OFF + CELL_MARGIN)/KAWAY_Y
#define CELL_SIZE_Z             (PTP_CUT_OFF + CELL_MARGIN)/KAWAY_Z

//variables to control initial uniform placement of atoms;
//atoms should not be too close at startup for a stable system;  
//PERDIM * GAP should be less than (PTPCUTOFF+CELL_MARGIN);
//max particles per cell should not be greater thatn PERDIM^3 for 1 AWAY;
#define PERDIM                  14
#define GAP                     3 

#define CELL_ORIGIN_X           0
#define CELL_ORIGIN_Y	        0
#define CELL_ORIGIN_Z	        0

#define MIGRATE_STEPCOUNT	        20
#define DEFAULT_FINALSTEPCOUNT	        1001
#define MAX_VELOCITY		        .1  //in A/fs

#define WRAP_X(a)		(((a) + cellArrayDimX) % cellArrayDimX)
#define WRAP_Y(a)		(((a) + cellArrayDimY) % cellArrayDimY)
#define WRAP_Z(a)		(((a) + cellArrayDimZ) % cellArrayDimZ)

// vec3 and dot() are used by both the host integrator and the device force kernels,
// so every method has to be callable from both. Under nvcc that means annotating
// them; under the host compiler the macro vanishes.
#ifdef __CUDACC__
#define LEANMD_HD __host__ __device__
#else
#define LEANMD_HD
#endif

struct vec3 {
  double x, y, z;

  LEANMD_HD vec3(double d = 0.0) : x(d), y(d), z(d) { }
  LEANMD_HD vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) { }

  LEANMD_HD inline vec3& operator += (const vec3 &rhs) {
    x += rhs.x; y += rhs.y; z += rhs.z;
    return *this;
  }
  LEANMD_HD inline vec3& operator -= (const vec3 &rhs) {
    return *this += (rhs * -1.0);
  }
  LEANMD_HD inline vec3 operator* (const double d) const {
    return vec3(d*x, d*y, d*z);
  }
  LEANMD_HD inline vec3 operator- (const vec3& rhs) const {
    return vec3(x - rhs.x, y - rhs.y, z - rhs.z);
  }
};
LEANMD_HD inline double dot(const vec3& a, const vec3& b) {
  return a.x*b.x + a.y*b.y + a.z*b.z;
}
PUPbytes(vec3)

//class for keeping track of the properties for a particle
struct Particle {
  double mass;
  //   Position, acceleration, velocity
  vec3 pos,acc,vel;
};
PUPbytes(Particle);

#include "leanmd.decl.h"

extern /* readonly */ CProxy_Main mainProxy;
extern /* readonly */ CProxy_Cell cellArray;
extern /* readonly */ CProxy_Compute computeArray;

extern /* readonly */ int cellArrayDimX;
extern /* readonly */ int cellArrayDimY;
extern /* readonly */ int cellArrayDimZ;
extern /* readonly */ int finalStepCount;
// Needed outside the SDAG so Compute::sendForces can tell whether this step
// ends in AtSync, and therefore whether its force buffers must be held until
// the receiving cells have pulled them.
extern /* readonly */ int firstLdbStep;
extern /* readonly */ int ldbPeriod;
extern /* readonly */ int checkptStrategy;
extern /* readonly */ std::string logs;
extern /* readonly */ int densityMode;      // DensityMode
extern /* readonly */ int computeMapMode;   // ComputeMapMode
extern /* readonly */ int maxCellParts;     // atoms in the fullest cell
extern /* readonly */ int densityReportFreq; // -densityreport: steps between reports, 0 = off
// -lbasync: split AtSync into AtSyncStart/AtSyncWait so the balancing step
// overlaps the simulation. Needs +LBAsync on the command line to mean anything.
extern /* readonly */ int asyncLb;
extern /* readonly */ int lbLag;  // -lblag: steps between the two halves

// LEANMD_ATOM_PCT=<1..100> scales every cell's atom count (default 100): a
// smaller problem on the SAME cells and computes. The object count, the
// communication graph and the imbalance ratio are unchanged; only the work per
// object shrinks (a Compute's pair work goes as the square).
inline int atomScalePct() {
  static const int pct = [] {
    const char* e = getenv("LEANMD_ATOM_PCT");
    const int p = e ? atoi(e) : 100;
    return (p >= 1 && p <= 100) ? p : 100;
  }();
  return pct;
}

// Atoms in cell (x,y,z) under the selected density profile, before scaling.
inline int cellParticleCountFull(int x, int y, int z) {
  const int nCells = cellArrayDimX * cellArrayDimY * cellArrayDimZ;
  switch (densityMode) {
    case DENSITY_GRADIENT: {
      // Linear in x, the axis CellMap blocks along.
      const int span = PARTICLES_PER_CELL_END - PARTICLES_PER_CELL_LIGHT;
      const int denom = (cellArrayDimX > 1) ? (cellArrayDimX - 1) : 1;
      return PARTICLES_PER_CELL_LIGHT + (x * span) / denom;
    }
    case DENSITY_CLUMP:
      // Dense half-box: the low half of x is full, the rest is sparse.
      return (x < (cellArrayDimX + 1) / 2) ? PARTICLES_PER_CELL_END
                                           : PARTICLES_PER_CELL_LIGHT;
    default: {
      // Stock: a mild ramp across the linearised cell index.
      const int myid = z + cellArrayDimZ * (y + x * cellArrayDimY);
      return PARTICLES_PER_CELL_START +
             (myid * (PARTICLES_PER_CELL_END - PARTICLES_PER_CELL_START)) / nCells;
    }
  }
}

// Atoms in cell (x,y,z): the profile's count, scaled by LEANMD_ATOM_PCT.
inline int cellParticleCount(int x, int y, int z) {
  const int n = cellParticleCountFull(x, y, z) * atomScalePct() / 100;
  return n > 0 ? n : 1;
}
#endif
