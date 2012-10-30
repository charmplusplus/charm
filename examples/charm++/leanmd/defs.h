
#ifndef __DEFS__
#define __DEFS__

#include "pup.h"

#define HYDROGEN_MASS         (1.67 * pow( 10.0,-24))
#define VDW_A			            (1.60694452 * pow(10.0, -134))
#define VDW_B			            (1.031093844 * pow(10.0, -77))

#define ENERGY_VAR  		      (1.0 * pow(10.0,-5))
//average of next two should be what you want as you atom density
#define PARTICLES_PER_CELL_START  300
#define PARTICLES_PER_CELL_END  	300


#define DEFAULT_DELTA         1	// in femtoseconds

#define DEFAULT_FIRST_LDB     20
#define DEFAULT_LDB_PERIOD    20
#define DEFAULT_FT_PERIOD     100000

#define KAWAY_X               1
#define KAWAY_Y               1
#define KAWAY_Z               1
#define NBRS_X	              (2*KAWAY_X+1)
#define NBRS_Y		            (2*KAWAY_Y+1)
#define NBRS_Z		            (2*KAWAY_Z+1)
#define NUM_NEIGHBORS	        (NBRS_X * NBRS_Y * NBRS_Z)

#define CELLARRAY_DIM_X	      3
#define CELLARRAY_DIM_Y	      3
#define CELLARRAY_DIM_Z	      3
#define PTP_CUT_OFF		        12 // cut off for atom to atom interactions
#define CELL_MARGIN		        4  // constant diff between cutoff and cell size
#define CELL_SIZE_X		        (PTP_CUT_OFF + CELL_MARGIN)/KAWAY_X
#define CELL_SIZE_Y		        (PTP_CUT_OFF + CELL_MARGIN)/KAWAY_Y
#define CELL_SIZE_Z		        (PTP_CUT_OFF + CELL_MARGIN)/KAWAY_Z
#define CELL_ORIGIN_X		      0
#define CELL_ORIGIN_Y		      0
#define CELL_ORIGIN_Z		      0

#define MIGRATE_STEPCOUNT	    20
#define DEFAULT_FINALSTEPCOUNT	1001
#define MAX_VELOCITY		      30.0

#define WRAP_X(a)		(((a)+cellArrayDimX)%cellArrayDimX)
#define WRAP_Y(a)		(((a)+cellArrayDimY)%cellArrayDimY)
#define WRAP_Z(a)		(((a)+cellArrayDimZ)%cellArrayDimZ)

struct vec3 {
  double x, y, z;

  vec3(double d = 0.0) : x(d), y(d), z(d) { }
  vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) { }

  inline vec3& operator += (const vec3 &rhs) {
    x += rhs.x; y += rhs.y; z += rhs.z;
    return *this;
  }
  inline vec3& operator -= (const vec3 &rhs) {
    return *this += (rhs * -1.0);
  }
  inline vec3 operator* (const double d) const {
    return vec3(d*x, d*y, d*z);
  }
  inline vec3 operator- (const vec3& rhs) const {
    return vec3(x - rhs.x, y - rhs.y, z - rhs.z);
  }
};
inline double dot(const vec3& a, const vec3& b) {
  return a.x*b.x + a.y*b.y + a.z*b.z;
}
PUPbytes(vec3)

//class for keeping track of the properties for a particle
struct Particle {
  double mass;
  //   Position, acceleration, velocity
  vec3 pos,acc,vel,force;

  // Function for pupping properties
  void pup(PUP::er &p) {
    p | mass;
    p | pos;
    p | acc;
    p | vel;
	p | force;
  }
};
#endif
