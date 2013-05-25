/*
 University of Illinois at Urbana-Champaign
 Department of Computer Science
 Parallel Programming Lab
 2008
*/

#include "liveViz.h"
#include "common.h"
#include "cell.decl.h"
#include "cell.h"
#include "main.decl.h"
#include "time.h"
#include "rand48_replacement.h"

extern /* readonly */ CProxy_Main mainProxy;
extern /* readonly */ CProxy_Cell cellArray;
extern /* readonly */ CProxy_Interaction interactionArray;

extern /* readonly */ int numParts;
extern /* readonly */ int m; // Number of Chare Rows
extern /* readonly */ int n; // Number of Chare Columns
extern /* readonly */ int L; 
extern /* readonly */ double radius;
extern /* readonly */ int finalStepCount; 

double A = 2.0; // Force Calculation parameter 1
double B = 1.0; // Force Calculation parameter 2

// Default constructor
Cell::Cell() {
	int i;
 
	// starting random generator
	srand48( thisIndex.x * 1000 + thisIndex.y +time(NULL));

	/* Particle initialization */
	for(i = 0; i < numParts / (m * n); i++){
	    particles.push_back(Particle());

	    particles[i].x = drand48() * L + thisIndex.x * L;
	    particles[i].y = drand48() * L + thisIndex.y * L;
	    particles[i].vx = (drand48() - 0.5) * .2 * MAX_VELOCITY;
	    particles[i].vy = (drand48() - 0.5) * .2 * MAX_VELOCITY;
	    particles[i].id = (thisIndex.x*m + thisIndex.y) * numParts / (m*n)  + i;
/*
	    particles[i].x = 0.0 + thisIndex.x * L;
	    particles[i].y = (float) i* 0.9 * L + thisIndex.y * L;
	    particles[i].vx = (drand48() - 0.5) * .2 * MAX_VELOCITY;
	    particles[i].vy = 0.0;
	    particles[i].id = (thisIndex.x*m + thisIndex.y) * numParts / (m*n)  + i;
*/
	}	

  updateCount = 0;
  forceCount = 0;
  stepCount = 0;
	updateFlag = false;
	incomingFlag = false;
  incomingParticles.resize(0);

}

// Constructor for chare object migration
Cell::Cell(CkMigrateMessage *msg) { }  
                                       
Cell::~Cell() {}

// Function to start interaction among particles in neighboring cells as well as its own particles
void Cell::start() {

  int x = thisIndex.x;
  int y = thisIndex.y;
  int i, j, k, l;
  
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
  (y == 0) ? (j=y, l=(y-1+n)%n) : (j=y-1, l=y);
  interactionArray( x, j, x, l ).interact(particles, x, y);

  // interaction with (x, y+1)
  (y == n-1) ? (j=(y+1)%n, l=y) : (j=y, l=y+1);// compute
  interactionArray( x, j, x, l ).interact(particles, x, y);

  // interaction with (x+1, y-1)
  (x == m-1) ? ( i=0, k=x, j=(y-1+n)%n, l=y ) : (i=x, k=x+1, j=y, l=(y-1+n)%n );
  interactionArray( i, j, k, l ).interact(particles, x, y);

  // interaction with (x+1, y)
  (x == m-1) ? (i=0, k=x) : (i=x, k=x+1);
  interactionArray( i, y, k, y).interact(particles, x, y);

  // interaction with (x+1, y+1)
  (x == m-1) ? ( i=0, k=x, j=(y+1)%n, l=y ) : (i=x, k=x+1, j=y, l=(y+1)%n );
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
    // Received all it's forces from the interactions.
    forceCount = 0;
    
    // Update properties on own particles
		updateProperties();

		// Sending particles to neighboring cells
		x = thisIndex.x;
		y = thisIndex.y;

		// particles sent to (x-1,y-1)		
		outgoing.removeAll();
		i = 0;
		while(i < particles.length()){
			if(particles[i].x < x*L && particles[i].y < y*L){
				outgoing.push_back(wrapAround(particles[i]));
				particles.remove(i);
			}else
				i++;
		}
		cellArray((x-1+m)%m,(y-1+n)%n).updateParticles(outgoing);
		      
		// particles sent to (x-1,y)		
		outgoing.removeAll();
		i = 0;
		while(i < particles.length()){
			if(particles[i].x < x*L && particles[i].y <= (y+1)*L){
				outgoing.push_back(wrapAround(particles[i]));
				particles.remove(i);
			}else
				i++;
		}
		cellArray((x-1+m)%m,y).updateParticles(outgoing);

		// particles sent to (x-1,y+1)
		outgoing.removeAll();
		i = 0;
		while(i < particles.length()){
			if(particles[i].x < x*L && particles[i].y > (y+1)*L){
				outgoing.push_back(wrapAround(particles[i]));
				particles.remove(i);
			}else
				i++;
		}
		cellArray((x-1+m)%m,(y+1)%n).updateParticles(outgoing);

		// particles sent to (x+1,y-1)
		outgoing.removeAll();
		i = 0;
		while(i < particles.length()){
			if(particles[i].x > (x+1)*L && particles[i].y < y*L){
				outgoing.push_back(wrapAround(particles[i]));
				particles.remove(i);
			}else
				i++;
		}
		cellArray((x+1)%m,(y-1+n)%n).updateParticles(outgoing);

		// particles sent to (x+1,y)
		outgoing.removeAll();
		i = 0;
		while(i < particles.length()){
			if(particles[i].x > (x+1)*L && particles[i].y <= (y+1)*L){
				outgoing.push_back(wrapAround(particles[i]));
				particles.remove(i);
			}else
				i++;
		}
		cellArray((x+1)%m,y).updateParticles(outgoing);

		// particles sent to (x+1,y+1)
		outgoing.removeAll();
		i = 0;
		while(i < particles.length()){
			if(particles[i].x > (x+1)*L && particles[i].y > (y+1)*L){
				outgoing.push_back(wrapAround(particles[i]));
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
				outgoing.push_back(wrapAround(particles[i]));
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
				outgoing.push_back(wrapAround(particles[i]));
				particles.remove(i);
			}else
				i++;
		}
		cellArray(x,(y+1)%n).updateParticles(outgoing);
    outgoing.removeAll();

		updateFlag = true;
		
    // checking whether to proceed with next step
		checkNextStep();

  }
    
}

// Function that checks whether it must start the following step or wait until other messages are received
void Cell::checkNextStep(){
	int i;

	if(updateFlag && incomingFlag){

		// resetting flags
		updateFlag = false;
		incomingFlag = false;
    stepCount++;

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
  double xDisp, yDisp;
	
	for(i = 0; i < particles.length(); i++){

		// applying kinetic equations
		particles[i].ax = particles[i].fx / DEFAULT_MASS;
		particles[i].ay = particles[i].fy / DEFAULT_MASS;
		particles[i].vx = particles[i].vx + particles[i].ax * DEFAULT_DELTA;
		particles[i].vy = particles[i].vy + particles[i].ay * DEFAULT_DELTA;

    limitVelocity( particles[i] );

		particles[i].x = particles[i].x + particles[i].vx * DEFAULT_DELTA;
		particles[i].y = particles[i].y + particles[i].vy * DEFAULT_DELTA;

		particles[i].fx = 0.0;
		particles[i].fy = 0.0;

	}

}

void Cell::limitVelocity(Particle &p) {

    //if( fabs(p.vx * DEFAULT_DELTA) > DEFAULT_RADIUS ) {
    if( fabs( p.vx ) > MAX_VELOCITY ) {
      //if( p.vx * DEFAULT_DELTA < 0.0 )
      if( p.vx < 0.0 )
        p.vx = -MAX_VELOCITY;
      else
        p.vx = MAX_VELOCITY;
      
    }

    //if( fabs(p.vy * DEFAULT_DELTA) > DEFAULT_RADIUS ) {
    if( fabs(p.vy) > MAX_VELOCITY ) {

      //if( p.vy * DEFAULT_DELTA < 0.0 )
      if( p.vy < 0.0 )
        p.vy = -MAX_VELOCITY;
      else
        p.vy = MAX_VELOCITY;
    }
}

Particle& Cell::wrapAround(Particle &p) {

		if(p.x < 0.0) p.x += L*m;
		if(p.y < 0.0) p.y += L*n;
		if(p.x > L*m) p.x -= L*m;
		if(p.y > L*n) p.y -= L*n;

    return p;
}

// Helper function to help with LiveViz
void color_pixel(unsigned char*buf,int width, int height, int xpos,int ypos,unsigned char R,unsigned char G,unsigned char B){
  if(xpos>=0 && xpos<width && ypos>=0 && ypos<height){
    buf[3*(ypos*width+xpos)+0] = R; 
    buf[3*(ypos*width+xpos)+1] = G; 
    buf[3*(ypos*width+xpos)+2] = B; 
  }
}
    
// Each chare provides its particle data to LiveViz
void Cell::requestNextFrame(liveVizRequestMsg *lvmsg) {
  // These specify the desired total image size requested by the client viewer
  int wdes = lvmsg->req.wid;
  int hdes = lvmsg->req.ht;
   
  int myWidthPx = wdes / m;
  int myHeightPx = hdes / n;
  int sx=thisIndex.x*myWidthPx;
  int sy=thisIndex.y*myHeightPx; 

  // set the output pixel values for rectangle
  // Each component is a char which can have 256 possible values
  unsigned char *intensity= new unsigned char[3*myWidthPx*myHeightPx];
  for(int i=0;i<myHeightPx;++i){
    for(int j=0;j<myWidthPx;++j){
        		
      // black background
      color_pixel(intensity,myWidthPx,myHeightPx,j,i,0,0,0);

    } 
  }

  for (int i=0; i < particles.length(); i++ ){
    
    int xpos = (int)((particles[i].x /(double) (L*m)) * wdes) - sx;
    int ypos = (int)((particles[i].y /(double) (L*n)) * hdes) - sy;

    Color c(particles[i].id);
    color_pixel(intensity,myWidthPx,myHeightPx,xpos+1,ypos,c.R,c.B,c.G);
    color_pixel(intensity,myWidthPx,myHeightPx,xpos-1,ypos,c.R,c.B,c.G);
    color_pixel(intensity,myWidthPx,myHeightPx,xpos,ypos+1,c.R,c.B,c.G);
    color_pixel(intensity,myWidthPx,myHeightPx,xpos,ypos-1,c.R,c.B,c.G);
    color_pixel(intensity,myWidthPx,myHeightPx,xpos,ypos,c.R,c.B,c.G);
  }
        
  for(int i=0;i<myHeightPx;++i){
    for(int j=0;j<myWidthPx;++j){
      
    // Draw red lines
    if(i==0 || j==0){
      color_pixel(intensity,myWidthPx,myHeightPx,j,i,128,0,0);
    }
    }
  }
        
  liveVizDeposit(lvmsg, sx,sy, myWidthPx,myHeightPx, intensity, this, max_image_data);
  delete[] intensity;

}

// Prints all particles 
void Cell::print(){
#ifdef PRINT
	int i;
	CkPrintf("*****************************************************\n");
	CkPrintf("Cell (%d,%d)\n",thisIndex.x,thisIndex.y);
	//CkPrintf("Part     x     y\n");
	for(i = 0; i < particles.length(); i++){
		CkPrintf("Cell (%d,%d) %-5d %7.4f %7.4f \n",thisIndex.x,thisIndex.y,i,particles[i].x,particles[i].y);
	}
	CkPrintf("*****************************************************\n");
#endif
}

/* ------------------ Interaction Methods --------------------- */

// Interaction - Default constructor
Interaction::Interaction() {
  cellCount = 0;
  bufferedX = 0;
  bufferedY = 0;
}

Interaction::Interaction(CkMigrateMessage *msg) { }
  
// Function to receive vector of particles
void Interaction::interact(CkVec<Particle> particles, int x, int y ) {

  int i;

  // self interaction check
  if( thisIndex.x == thisIndex.z && thisIndex.y == thisIndex.w ) {
		interact(particles,particles);
    cellArray( x, y).updateForces(particles);
  } else {
    if(cellCount == 0) {

	      bufferedX = x;
    	  bufferedY = y;
        bufferedParticles = particles;
        cellCount++;

		} else if(cellCount == 1) {
    
	    // if both particle sets are received, compute interaction
      cellCount = 0;
		  interact(bufferedParticles,particles);
      cellArray(bufferedX, bufferedY).updateForces(bufferedParticles);
      cellArray(x, y).updateForces(particles);

    }

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

  // We include 0.000001 to ensure that r doesn't tend to zero in the force calculation
	//if(r < 0.000001 || r >= DEFAULT_RADIUS)
	if(r < 0.000001 || r >= L)
		return;

	f = A / pow(r,12) - B / pow(r,6);
	fx = f * rx / r;
	fy = f * ry / r;

	// updating particle properties
	second.fx -= fx;
	second.fy -= fy;
	first.fx += fx;
	first.fy += fy;

}

#include "cell.def.h"
