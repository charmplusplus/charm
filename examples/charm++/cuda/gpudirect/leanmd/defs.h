
#ifndef __DEFS__
#define __DEFS__

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
#define PARTICLES_PER_CELL_START        800
#define PARTICLES_PER_CELL_END          1000

#define DEFAULT_DELTA           1	// in femtoseconds

#define DEFAULT_FIRST_LDB       20
#define DEFAULT_LDB_PERIOD      20
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
#define CELL_MARGIN             4  // constant diff between cutoff and cell size
#define CELL_SIZE_X             (PTP_CUT_OFF + CELL_MARGIN)/KAWAY_X
#define CELL_SIZE_Y             (PTP_CUT_OFF + CELL_MARGIN)/KAWAY_Y
#define CELL_SIZE_Z             (PTP_CUT_OFF + CELL_MARGIN)/KAWAY_Z

//variables to control initial uniform placement of atoms;
//atoms should not be too close at startup for a stable system;  
//PERDIM * GAP should be less than (PTPCUTOFF+CELL_MARGIN);
//max particles per cell should not be greater thatn PERDIM^3 for 1 AWAY;
#define PERDIM                  10
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
extern /* readonly */ int checkptStrategy;
extern /* readonly */ std::string logs;
#endif
