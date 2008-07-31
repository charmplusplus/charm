/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/** \file common.h
 *  Author: Abhinav S Bhatele
 *  Date Created: July 1st, 2008
 *
 */

#ifndef __COMMON_H__
#define __COMMON_H__

#include "pup.h"

#define DEFAULT_MASS 1
#define DEFAULT_DELTA 0.005
#define DEFAULT_PARTICLES 2000
#define DEFAULT_M 5
#define DEFAULT_N 5
#define DEFAULT_L 1
#define DEFAULT_RADIUS 5
#define DEFAULT_FINALSTEPCOUNT 10
#define MAX_VELOCITY 30.0

// Class for keeping track of the properties for a particle
class Particle{
  public:
    double mass;	// mass of the particle
    double x;		// position in x axis
    double y;		// position in y axis
    double fx;		// total forces on x axis
    double fy;		// total forces on y axis
    double ax;		// acceleration on x axis
    double ay;		// acceleration on y axis
    double vx;		// velocity on x axis
    double vy;		// velocity on y axis
    int id;

    // Default constructor
    Particle() {
      fx = fy = 0.0;
    }

    // Function for pupping properties
    void pup(PUP::er &p) {
      p | mass;
      p | x;
      p | y;
      p | fx;
      p | fy;
      p | ax;
      p | ay;
      p | vx;
      p | vy;
      p | id;
    }
};

class Color {
  public:
    unsigned char R, G, B;

    // Generate a unique color for each index from 0 to total-1
    Color(int index){
      int total = 8;
      if(index % total == 0) {
	R = 255;
	G = 100;
	B = 100;
      } else if(index % total == 1) {
	R = 100;
	G = 255;
	B = 100;
      } else if(index % total == 2) {
	R = 100;
	G = 100;
	B = 255;
      } else if(index % total == 3) {
	R = 100;
	G = 255;
	255;
      } else if(index % total == 4) {
	R = 100;
	G = 255;
	B = 255;
      } else if(index % total == 5) {
	R = 255;
	G = 255;
	B = 100;
      } else if(index % total == 6) {
	R = 255;
	G = 100;
	B = 255;
      } else {
	R = 170;
	G = 170;
	B = 170;
      }
    }	
};

#endif
