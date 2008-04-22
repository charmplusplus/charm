/*
 University of Illinois at Urbana-Champaign
 Department of Computer Science
 Parallel Programming Lab
 2008
 Authors: Kumaresh Pattabiraman and Esteban Meneses
*/

#include "charm++.h"
#include "time.h"

#define DEFAULT_MASS 1
#define DEFAULT_DELTA 0.1

// Class for keeping track of the properties for a particle
class Particle{
	public:
		double mass;							// mass of the particle
		double x;									// position in x axis
		double y;									// position in y axis
		double fx;								// total forces on x axis
		double fy;								// total forces on y axis
		double ax;								// acceleration on x axis
		double ay;								// acceleration on y axis
		double vx;								// velocity on x axis
		double vy;								// velocity on y axis

	// Default constructor
	Particle(){
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
		}

};
