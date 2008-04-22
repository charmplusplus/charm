/*
 University of Illinois at Urbana-Champaign
 Department of Computer Science
 Parallel Programming Lab
 2008
 Authors: Kumaresh Pattabiraman and Esteban Meneses
*/

#include "main.h"
#include "main.decl.h"

#define DEBUG 1

#define A 2.0
#define B 1.0

#define DEFAULT_PARTICLES 350
#define DEFAULT_M 5
#define DEFAULT_N 7
#define DEFAULT_L 10
#define DEFAULT_RADIUS 10
#define DEFAULT_FINALSTEPCOUNT 2

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_Cell cellArray;
/* readonly */ CProxy_Interaction interactionArray;

/* readonly */ int numParts;
/* readonly */ int m; // Number of Chare Rows
/* readonly */ int n; // Number of Chare Columns
/* readonly */ int L; 
/* readonly */ double radius;
/* readonly */ int finalStepCount; 

// Class representing a cell in the grid. We consider each cell as a square of LxL units
class Cell : public CBase_Cell {
  private:
    CkVec<Particle> particles;
		CkVec<Particle> incomingParticles;
    int forceCount; 																	// to count the returns from interactions
    int stepCount;  																	// to count the number of steps, and decide when to stop
		int updateCount;
		bool updateFlag;
		bool incomingFlag;

		void updateProperties();													// updates properties after receiving forces from interactions
		void checkNextStep();															// checks whether to continue with next step
		void print();																			// prints all its particles

  public:
    Cell();
    Cell(CkMigrateMessage *msg);
    ~Cell();

    void start();
    void updateParticles(CkVec<Particle>&);
    void updateForces(CkVec<Particle>&);
    void stepDone();
};

// Class representing the interaction agents between a couple of cells
class Interaction : public CBase_Interaction {
  private:
    int cellCount;  																	// to count the number of interact() calls
    CkVec<Particle> bufferedParticles;
    int bufferedX;
 		int bufferedY;

		void interact(CkVec<Particle> &first, CkVec<Particle> &second);
		void interact(Particle &first, Particle &second);

  public:
    Interaction();
    Interaction(CkMigrateMessage *msg);

    void interact(CkVec<Particle> particles, int i, int j);


};
    
// Main class
class Main : public CBase_Main {

  private:
    int checkInCount; 																// Count to terminate

    #ifdef DEBUG
      int interactionCount;
    #endif

  public:

    /// Constructors ///
    Main(CkArgMsg* msg);
    Main(CkMigrateMessage* msg);

    void checkIn();
};

// Entry point of Charm++ application
Main::Main(CkArgMsg* msg) {
	int i, j, k, l;  

  numParts = DEFAULT_PARTICLES;
  m = DEFAULT_M;
  n = DEFAULT_N;
	L = DEFAULT_L;
  radius = DEFAULT_RADIUS;
  finalStepCount = DEFAULT_FINALSTEPCOUNT;

  delete msg;
  checkInCount = 0;

  #ifdef DEBUG
    interactionCount=0;
  #endif

  mainProxy = thisProxy;

	// initializing the cell 2D array
  cellArray = CProxy_Cell::ckNew(m,n);

	// initializing the interaction 4D array
  interactionArray = CProxy_Interaction::ckNew();
  
  for (int x = 0; x < m ; x++ ) {
    for (int y = 0; y < n; y++ ) {

      //Processor Round Robin needed /* FIXME */
 
      #ifdef DEBUG
        //CkPrintf("INITIAL:( %d, %d) ( %d , %d )\n", x,y,x,y);
        interactionCount++;
      #endif

      // self interaction
      interactionArray( x, y, x, y ).insert( /* processor number */0 );

      // (x,y) and (x+1,y) pair
      (x == m-1) ? (i=(x+1)%m, k=x) : (i=x, k=x+1);
      #ifdef DEBUG
        //CkPrintf("INITIAL:( %d, %d) ( %d , %d )\n", i,y,k,y);
        interactionCount++;
      #endif
      interactionArray( i, y, k, y ).insert( /* processor number */0 );

      // (x,y) and (x,y+1) pair
      (y == n-1) ? (j=(y+1)%n, l=y) : (j=y, l=y+1);
      #ifdef DEBUG
        //CkPrintf("INITIAL:( %d, %d) ( %d , %d )\n", x,j,x,l);
        interactionCount++;
      #endif

			
      interactionArray( x, j, x, l ).insert( /* processor number */0 );

      // (x,y) and (x+1,y+1) pair, Irrespective of y /* UNDERSTAND */
      (x == m-1) ? ( i=(x+1)%m, k=x, j=(y+1)%n, l=y ) : (i=x, k=x+1, j=y, l=(y+1)%n );
      #ifdef DEBUG
        //CkPrintf("INITIAL:( %d, %d) ( %d , %d )\n", i,j,k,l);
        interactionCount++;
      #endif
      interactionArray( i, j, k, l ).insert( /* processor number */0 );

      // (x,y) and (x-1,y+1) pair /* UNDERSTAND */
      (x == 0) ? ( i=x, k=(x-1+m)%m, j=y, l=(y+1)%n ) : (i=x-1, k=x, j=(y+1)%n, l=y );
      #ifdef DEBUG
        //CkPrintf("INITIAL:( %d, %d) ( %d , %d )\n", i,j,k,l);
        interactionCount++;
      #endif
      interactionArray( i, j, k, l ).insert( /* processor number */0 );

    }
  }

  interactionArray.doneInserting();

  #ifdef DEBUG
    CkPrintf("Interaction Count: %d\n", interactionCount);
  #endif

  cellArray.start();
}

// Constructor needed for chare object migration
Main::Main(CkMigrateMessage* msg) { }

void Main::checkIn() {

  checkInCount ++;
  if( checkInCount >= m*n)
    CkExit();

}

// Default constructor
Cell::Cell() {
	int i;

	// starting random generator
	srand48(time(NULL));
  
	/* Particle initialization */
	// initializing a number of particles
	for(i = 0; i < numParts / (m * n); i++){
		particles.push_back(Particle());
		particles[i].x = drand48() * L + thisIndex.x * L;
    particles[i].y = drand48() * L + thisIndex.y * L;
	}	

  forceCount = 0;
  stepCount = 0;
	updateFlag = false;
	incomingFlag = false;

}

// Constructor needed for chare object migration (ignore for now)
Cell::Cell(CkMigrateMessage *msg) { }                                         
Cell::~Cell() {
  /* FIXME */ // Deallocate particle lists
}

// Function to start interaction among particles in neighboring cells as well as its own particles
void Cell::start() {

  int x = thisIndex.x;
  int y = thisIndex.y;

  int i, j, k, l;

  #ifdef DEBUG
    //CkPrintf("START:( %d, %d) ( %d , %d )\n", x,y,x,y);
  #endif
  
  // self interaction
  interactionArray( x, y, x, y).interact(particles, x, y);

  // interaction with (x-1, y-1)
  (x == 0) ? ( i=x, k=(x-1+m)%m, j=y, l=(y-1+n)%n ) : (i=x-1, k=x, j=(y-1+n)%n, l=y);
  interactionArray( i, j, k, l ).interact(particles, x, y);

  // interaction with (x-1, y)
  (x == 0) ? (i=x, k=(x-1+m)%m) : (i=x-1, k=x);
  interactionArray( i, y, k, y).interact(particles, x, y);

  // interaction with (x-1, y+1)
  (x == 0) ? ( i=x, k=(x-1+m)%m, j=y, l=(y+1)%n ) : (i=x-1, k=x, j=(y+1)%n, l=y);
  interactionArray( i, j, k, l ).interact(particles, x, y);


  // interaction with (x, y-1)
  (y == 0) ? (j=y, l=(y-1+n)%n) : (j=(y-1+n)%n, l=y);
  interactionArray( x, j, x, l ).interact(particles, x, y);

  // interaction with (x, y+1)
  (y == n-1) ? (j=(y+1)%n, l=y) : (j=y, l=y+1);// compute
  interactionArray( x, j, x, l ).interact(particles, x, y);


  // interaction with (x+1, y-1)
  (x == m-1) ? ( i=(x+1)%m, k=x, j=(y-1+n)%n, l=y ) : (i=x, k=x+1, j=y, l=(y-1+n)%n );
  interactionArray( i, j, k, l ).interact(particles, x, y);

  // interaction with (x+1, y)
  (x == m-1) ? (i=(x+1)%m, k=x) : (i=x, k=x+1);
  interactionArray( i, y, k, y).interact(particles, x, y);

  // interaction with (x+1, y+1)
  (x == m-1) ? ( i=(x+1)%m, k=x, j=(y+1)%n, l=y ) : (i=x, k=x+1, j=y, l=(y+1)%n );
  interactionArray( i, j, k, l ).interact(particles, x, y);

}

// Function to update forces coming from a neighbor interaction chare
void Cell::updateForces(CkVec<Particle> &updates) {
	int i, x ,y;
	CkVec<Particle> outgoing;

	// incrementing the counter for receiving updates
  forceCount++;

	// updating force information
	for(i = 0; i < updates.length(); i++){
		particles[i].fx += updates[i].fx;
		particles[i].fy += updates[i].fy;
	}

	// if all forces are received, then it must recompute particles location
  if( forceCount >= 9) {
    
		CkPrintf("Forces done!\n");

    // Received all it's forces from the interactions.
    stepCount++;
    forceCount = 0;
    
    // Update properties on own particles
		updateProperties();
		updateFlag = true;

		// Sending particles to neighboring cells
		x = thisIndex.x;
		y = thisIndex.y;
		// particles sent to (x-1,y-1)		
		outgoing.removeAll();
		i = 0;
		while(i < particles.length()){
			if(particles[i].x < x*L && particles[i].y < y*L){
				outgoing.push_back(particles[i]);
				particles.remove(i);
			}else
				i++;
		}
		cellArray((x-1+m)%m,(y-1+n)%n).updateParticles(outgoing);
		      
		// particles sent to (x-1,y)		
		outgoing.removeAll();
		i = 0;
		while(i < particles.length()){
			if(particles[i].x < x*L && particles[i].y <= y*(L+1)){
				outgoing.push_back(particles[i]);
				particles.remove(i);
			}else
				i++;
		}
		cellArray((x-1+m)%m,y).updateParticles(outgoing);

		// particles sent to (x-1,y+1)
		outgoing.removeAll();
		i = 0;
		while(i < particles.length()){
			if(particles[i].x < x*L && particles[i].y > y*(L+1)){
				outgoing.push_back(particles[i]);
				particles.remove(i);
			}else
				i++;
		}
		cellArray((x-1+m)%m,(y+1)%n).updateParticles(outgoing);

		// particles sent to (x+1,y-1)
		outgoing.removeAll();
		i = 0;
		while(i < particles.length()){
			if(particles[i].x > x*(L+1) && particles[i].y < y*L){
				outgoing.push_back(particles[i]);
				particles.remove(i);
			}else
				i++;
		}
		cellArray((x+1)%m,(y-1+n)%n).updateParticles(outgoing);

		// particles sent to (x+1,y)
		outgoing.removeAll();
		i = 0;
		while(i < particles.length()){
			if(particles[i].x > x*(L+1) && particles[i].y <= y*(L+1)){
				outgoing.push_back(particles[i]);
				particles.remove(i);
			}else
				i++;
		}
		cellArray((x+1)%m,y).updateParticles(outgoing);

		// particles sent to (x+1,y+1)
		outgoing.removeAll();
		i = 0;
		while(i < particles.length()){
			if(particles[i].x > x*(L+1) && particles[i].y > y*(L+1)){
				outgoing.push_back(particles[i]);
				particles.remove(i);
			}else
				i++;
		}
		cellArray((x+1)%m,(y+1)%n).updateParticles(outgoing);

		// particles sent to (x,y-1)
		outgoing.removeAll();
		i = 0;
		while(i < particles.length()){
			if(particles[i].y < y*L){
				outgoing.push_back(particles[i]);
				particles.remove(i);
			}else
				i++;
		}
		cellArray(x,(y-1+n)%n).updateParticles(outgoing);

		// particles sent to (x,y+1)
		outgoing.removeAll();
		i = 0;
		while(i < particles.length()){
			if(particles[i].y > (y+1)*L){
				outgoing.push_back(particles[i]);
				particles.remove(i);
			}else
				i++;
		}
		cellArray(x,(y+1)%n).updateParticles(outgoing);

    #ifdef DEBUG
      CkPrintf("STEP: %d DONE:( %d , %d )\n", stepCount, thisIndex.x, thisIndex.y);
    #endif

		// checking whether to proceed with next step
		checkNextStep();

  }
    
}

// Prints all particle set
void Cell::print(){
	int i;
	CkPrintf("*******************************************************************\n");
	CkPrintf("Cell (%d,%d)\n",thisIndex.x,thisIndex.y);
	CkPrintf("Part     x     y\n");
	for(i = 0; i < particles.length(); i++){
		CkPrintf("%-5d %7.4f %7.4f \n",i,particles[i].x,particles[i].y);
	}
	CkPrintf("*******************************************************************\n");
}

// Function that checks whether it must start the following step or wait until other messages are received
void Cell::checkNextStep(){
	int i;

	if(updateFlag && incomingFlag){

		// resetting flags
		updateFlag = false;
		incomingFlag = false;

		// adding new elements
		for(i = 0; i < incomingParticles.length(); i++){
			particles.push_back(incomingParticles[i]);
		}
		incomingParticles.removeAll();

		// checking for next step
		if(stepCount >= finalStepCount) {
			print();
			mainProxy.checkIn();
	  } else {
	    thisProxy( thisIndex.x, thisIndex.y ).start();
	  }
	}

}

// Function that receives a set of particles and updates the forces of them into the local set
void Cell::updateParticles(CkVec<Particle> &updates) {
  updateCount++;

  for( int i=0; i < updates.length(); i++) {
    incomingParticles.push_back(updates[i]);
  }

	// if all the incoming particle updates have been received, we must check whether to proceed with next step
	if(updateCount >= 8 ) {
		updateCount = 0;
		incomingFlag = true;
		checkNextStep();
	}

}

// Function to update properties (i.e. acceleration, velocity and position) in particles
void Cell::updateProperties(){
	int i;
	
	for(i = 0; i < particles.length(); i++){

		// applying kinetic equations
		particles[i].ax = particles[i].fx / DEFAULT_MASS;
		particles[i].ay = particles[i].fy / DEFAULT_MASS;
		particles[i].vx = particles[i].vx + particles[i].ax * DEFAULT_DELTA;
		particles[i].vy = particles[i].vy + particles[i].ay * DEFAULT_DELTA;
		particles[i].x = particles[i].x + particles[i].vx * DEFAULT_DELTA;
		particles[i].y = particles[i].y + particles[i].vy * DEFAULT_DELTA;
		particles[i].fx = 0.0;
		particles[i].fy = 0.0;

		// checking boundary conditions
		if(particles[i].x < 0.0) particles[i].x = 0.0;
		if(particles[i].y < 0.0) particles[i].y = 0.0;
		if(particles[i].x > L*m) particles[i].x = L*m;
		if(particles[i].y > L*n) particles[i].y = L*n;

	}

}


// Default constructor
Interaction::Interaction() {
  cellCount = 0;

  /* FIXME */
    bufferedX = 0;
    bufferedY = 0;
}

Interaction::Interaction(CkMigrateMessage *msg) { }
  

// Function to receive vector of particles
void Interaction::interact(CkVec<Particle> particles, int x, int y ) {

  if(cellCount == 0) {

    // self interaction check
    if( thisIndex.x == thisIndex.z && thisIndex.y == thisIndex.w ) {
      //CkPrintf("SELF: ( %d , %d )\n", thisIndex.x, thisIndex.y );
      cellCount = 0;
			interact(particles,particles);
      cellArray( x, y).updateForces(particles);
    } else {
			 bufferedX = x;
    	 bufferedY = y;
			 bufferedParticles = particles; /* CHECKME */
		}

  }

  cellCount++;

	// if both particle sets are received, compute interaction
  if(cellCount >= 2) {

    //CkPrintf("PAIR:( %d , %d )  ( %d , %d ) \n", bufferedX, bufferedY, x, y );
    cellCount = 0;

		interact(bufferedParticles,particles);

    cellArray(bufferedX, bufferedY).updateForces(bufferedParticles);
    cellArray(x, y).updateForces(particles);

  }
}

// Function to compute all the interactions between pairs of particles in two sets
void Interaction::interact(CkVec<Particle> &first, CkVec<Particle> &second){
	int i, j;
	for(i = 0; i < first.length(); i++)
		for(j = 0; j < second.length(); j++)
			interact(first[i], second[j]);
}

// Function for computing interaction among two particles
// There is an extra test for interaction of identical particles, in which case there is no effect
void Interaction::interact(Particle &first, Particle &second){
	float rx,ry,rz,r,fx,fy,fz,f;

	// computing base values
	rx = first.x - second.x;
	ry = first.y - second.y;
	r = sqrt(rx*rx + ry*ry);

	if(r == 0.0 || r >= DEFAULT_RADIUS)
		return;

	f = A / pow(r,12) - B / pow(r,6);
	fx = f * rx / r;
	fy = f * ry / r;

	// updating particle properties
	second.fx += fx;
	second.fy += fy;
	first.fx += -fx;
	first.fy += -fy;

}

#include "main.def.h"
