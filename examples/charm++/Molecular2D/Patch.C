/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/** \file Patch.C
 *  Author: Abhinav S Bhatele
 *  Date Created: July 1st, 2008
 *
 */

#include "time.h"
#include "common.h"
#ifdef RUN_LIVEVIZ
  #include "liveViz.h"
#endif
#include "Patch.decl.h"
#include "Patch.h"
#include "Compute.h"

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_Patch patchArray;
/* readonly */ CProxy_Compute computeArray;

/* readonly */ int numParts;
/* readonly */ int patchArrayDimX;	// Number of Chare Rows
/* readonly */ int patchArrayDimY;	// Number of Chare Columns
/* readonly */ int patchSize; 
/* readonly */ double radius;
/* readonly */ int finalStepCount; 

double A = 2.0;			// Force Calculation parameter 1
double B = 1.0;			// Force Calculation parameter 2

// Entry point of Charm++ application
Main::Main(CkArgMsg* msg) {

  CkPrintf("\nLENNARD JONES MOLECULAR DYNAMICS RUNNING ...\n");

  numParts = DEFAULT_PARTICLES;
  patchArrayDimX = PATCHARRAY_DIM_X;
  patchArrayDimY = PATCHARRAY_DIM_Y;
  patchSize = PATCH_SIZE;
  radius = DEFAULT_RADIUS;
  finalStepCount = DEFAULT_FINALSTEPCOUNT;

  delete msg;
  mainProxy = thisProxy;

  // initializing the cell 2D array
  patchArray = CProxy_Patch::ckNew(patchArrayDimX, patchArrayDimY);
  CkPrintf("%d PATCHES CREATED\n", patchArrayDimX*patchArrayDimY);

  // initializing the interaction 4D array
  computeArray = CProxy_Compute::ckNew();
 
  for (int x=0; x<patchArrayDimX; x++) {
    for (int y=0; y<patchArrayDimY; y++) {
      patchArray(x, y).createComputes();
    }
  }

}

// Constructor for chare object migration
Main::Main(CkMigrateMessage* msg) { }

void Main::allDone() {
  CkPrintf("SIMULATION COMPLETE.\n\n");
  CkExit();
}

void Main::computeCreationDone() {
  computeArray.doneInserting();
  CkPrintf("%d COMPUTES CREATED\n", 5*patchArrayDimX*patchArrayDimY);

  // setup liveviz
  // CkCallback c(CkIndex_Patch::requestNextFrame(0),patchArray);
  // liveVizConfig cfg(liveVizConfig::pix_color,true);
  // liveVizInit(cfg,patchArray,c);

  // sleep(1);
  patchArray.start();
}

// Default constructor
Patch::Patch() {
  int i;

  // starting random generator
  srand48( thisIndex.x * 1000 + thisIndex.y +time(NULL));

  // Particle initialization
  for(i=0; i < numParts/(patchArrayDimX*patchArrayDimY); i++) {
    particles.push_back(Particle());

    particles[i].x = drand48() * patchSize + thisIndex.x * patchSize;
    particles[i].y = drand48() * patchSize + thisIndex.y * patchSize;
    particles[i].vx = (drand48() - 0.5) * .2 * MAX_VELOCITY;
    particles[i].vy = (drand48() - 0.5) * .2 * MAX_VELOCITY;
    particles[i].id = (thisIndex.x*patchArrayDimX + thisIndex.y) * numParts / (patchArrayDimX*patchArrayDimY)  + i;
  }	

  updateCount = 0;
  forceCount = 0;
  stepCount = 0;
  updateFlag = false;
  incomingFlag = false;
  incomingParticles.resize(0);
}

// Constructor for chare object migration
Patch::Patch(CkMigrateMessage *msg) { }  
                                       
Patch::~Patch() {}

void Patch::createComputes() {
  int i, j, k, l, num;  
  
  int x = thisIndex.x;
  int y = thisIndex.y;

  // For Round Robin insertion
  int numPes = CkNumPes();
  int currPE = CkMyPe();
 
  num = 0;

  /*  The computes X are inserted by a given patch:
   *
   *	^  X  X  X
   *	|  0  X  X
   *	y  0  0  0
   *	   x ---->
   *
   */

  // interaction with (x-1, y-1)
  (x==0) ? (i=x, j=y, k=WRAP_X(x-1), l=WRAP_Y(y-1)) : (i=x-1, j=WRAP_Y(y-1), k=x, l=y);
  computesList[num][0] = i; computesList[num][1] = j; computesList[num][2] = k; computesList[num][3] = l; num++;

  // interaction with (x-1, y)
  (x==0) ? (i=x, j=y, k=WRAP_X(x-1), l=y) : (i=x-1, j=y, k=x, l=y);
  computesList[num][0] = i; computesList[num][1] = j; computesList[num][2] = k; computesList[num][3] = l; num++;
  
  // interaction with (x-1, y+1)
  (x==0) ? (i=x, j=y, k=WRAP_X(x-1), l=WRAP_Y(y+1)) : (i=x-1, j=WRAP_Y(y+1), k=x, l=y);
  computesList[num][0] = i; computesList[num][1] = j; computesList[num][2] = k; computesList[num][3] = l; num++;
  computeArray(i, j, k, l).insert((currPE++) % numPes);

  // interaction with (x, y-1)
  (y==0) ? (i=x, j=y, k=x, l=WRAP_Y(y-1)) : (i=x, j=y-1, k=x, l=y);
  computesList[num][0] = i; computesList[num][1] = j; computesList[num][2] = k; computesList[num][3] = l; num++;

  // interaction with (x, y) -- SELF
  i=x; j=y; k=x; l=y;
  computesList[num][0] = i; computesList[num][1] = j; computesList[num][2] = k; computesList[num][3] = l; num++;
  computeArray(i, j, k, l).insert((currPE++) % numPes);

  // interaction with (x, y+1)
  (y==patchArrayDimY-1) ? (i=x, j=0, k=x, l=y) : (i=x, j=y, k=x, l=y+1);
  computesList[num][0] = i; computesList[num][1] = j; computesList[num][2] = k; computesList[num][3] = l; num++;
  computeArray(i, j, k, l).insert((currPE++) % numPes);

  // interaction with (x+1, y-1)
  (x==patchArrayDimX-1) ? (i=0, j=WRAP_Y(y-1), k=x, l=y) : (i=x, j=y, k=x+1, l=WRAP_Y(y-1));
  computesList[num][0] = i; computesList[num][1] = j; computesList[num][2] = k; computesList[num][3] = l; num++;

  // interaction with (x+1, y)
  (x==patchArrayDimX-1) ? (i=0, j=y, k=x, l=y) : (i=x, j=y, k=x+1, l=y);
  computesList[num][0] = i; computesList[num][1] = j; computesList[num][2] = k; computesList[num][3] = l; num++;
  computeArray(i, j, k, l).insert((currPE++) % numPes);

  // interaction with (x+1, y+1)
  (x==patchArrayDimX-1) ? (i=0, j=WRAP_Y(y+1), k=x, l=y ) : (i=x, j=y, k=x+1, l=WRAP_Y(y+1));
  computesList[num][0] = i; computesList[num][1] = j; computesList[num][2] = k; computesList[num][3] = l; num++;
  computeArray(i, j, k, l).insert((currPE++) % numPes);

  contribute(0, 0, CkReduction::concat, CkCallback(CkIndex_Main::computeCreationDone(), mainProxy));
}

// Function to start interaction among particles in neighboring cells as well as its own particles
void Patch::start() {

  int x = thisIndex.x;
  int y = thisIndex.y;
  int i, j, k, l;
 
  for(int num=0; num<NUM_NEIGHBORS; num++) {
    i = computesList[num][0];
    j = computesList[num][1];
    k = computesList[num][2];
    l = computesList[num][3];
    computeArray(i, j, k, l).interact(particles, x, y);
  }

}

// Function to update forces coming from a neighbor interaction chare
void Patch::updateForces(CkVec<Particle> &updates) {
  int i, x, y, x1, y1;
  CkVec<Particle> outgoing[NUM_NEIGHBORS];

  // incrementing the counter for receiving updates
  forceCount++;

  // updating force information
  for(i = 0; i < updates.length(); i++){
    particles[i].fx += updates[i].fx;
    particles[i].fy += updates[i].fy;
  }

  // if all forces are received, then it must recompute particles location
  if( forceCount == NUM_NEIGHBORS) {
    // Received all it's forces from the interactions.
    forceCount = 0;
  
    // Update properties on own particles
    updateProperties();

    // sending particles to neighboring cells
    x = thisIndex.x;
    y = thisIndex.y;

    for(i=0; i<particles.length(); i++) {
      migrateToPatch(particles[i], x1, y1);
      if(x1 !=0 || y1!=0) {
	outgoing[(x1+1)*3+(y1+1)].push_back(wrapAround(particles[i]));
	particles.remove(i);
      }
    }
   
    for(i=0; i<NUM_NEIGHBORS; i++)
      patchArray(WRAP_X(x + i/3 - 2), WRAP_Y(y + i%3 -2)).updateParticles(outgoing[i]);

    updateFlag = true;
	      
    // checking whether to proceed with next step
    checkNextStep();
  }
  
}

void Patch::migrateToPatch(Particle p, int &px, int &py) {
  int x = thisIndex.x * patchSize;
  int y = thisIndex.y * patchSize;

  if (p.x < x) px = -1;
  else if (p.x > x+patchSize) px = 1;
  else px = 0;

  if (p.y < y) py = -1;
  else if (p.y > y+patchSize) py = 1;
  else py = 0;
}

// Function that checks whether it must start the following step or wait until other messages are received
void Patch::checkNextStep(){
  int i;
  if(updateFlag && incomingFlag) {
    if(thisIndex.x==0 && thisIndex.y==0 && stepCount%10==0)
      CkPrintf("Step %d %f\n", stepCount, CmiWallTimer());

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
      contribute(0, 0, CkReduction::concat, CkCallback(CkIndex_Main::allDone(), mainProxy)); 
    } else {
      thisProxy(thisIndex.x, thisIndex.y).start();
    }
  }
}

// Function that receives a set of particles and updates the 
// forces of them into the local set
void Patch::updateParticles(CkVec<Particle> &updates) {
  updateCount++;

  for( int i=0; i < updates.length(); i++) {
    incomingParticles.push_back(updates[i]);
  }

  // if all the incoming particle updates have been received, we must check 
  // whether to proceed with next step
  if(updateCount == NUM_NEIGHBORS-1 ) {
    updateCount = 0;
    incomingFlag = true;
    checkNextStep();
  }
}

// Function to update properties (i.e. acceleration, velocity and position) in particles
void Patch::updateProperties() {
  int i;
  double xDisp, yDisp;
	
  for(i = 0; i < particles.length(); i++) {
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

void Patch::limitVelocity(Particle &p) {
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

Particle& Patch::wrapAround(Particle &p) {
  if(p.x < 0.0) p.x += patchSize*patchArrayDimX;
  if(p.y < 0.0) p.y += patchSize*patchArrayDimY;
  if(p.x > patchSize*patchArrayDimX) p.x -= patchSize*patchArrayDimX;
  if(p.y > patchSize*patchArrayDimY) p.y -= patchSize*patchArrayDimY;

  return p;
}

// Helper function to help with LiveViz
void color_pixel(unsigned char*buf,int width, int height, int xpos,int ypos,
                             unsigned char R,unsigned char G,unsigned char B) {
  if(xpos>=0 && xpos<width && ypos>=0 && ypos<height) {
    buf[3*(ypos*width+xpos)+0] = R; 
    buf[3*(ypos*width+xpos)+1] = G; 
    buf[3*(ypos*width+xpos)+2] = B; 
  }
}
    
#ifdef RUN_LIVEVIZ
// Each chare provides its particle data to LiveViz
void Patch::requestNextFrame(liveVizRequestMsg *lvmsg) {
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
  for(int i=0;i<myHeightPx;++i) {
    for(int j=0;j<myWidthPx;++j) {
      // black background
      color_pixel(intensity,myWidthPx,myHeightPx,j,i,0,0,0);
    } 
  }

  for (int i=0; i < particles.length(); i++ ) {
    int xpos = (int)((particles[i].x /(double) (patchSize*m)) * wdes) - sx;
    int ypos = (int)((particles[i].y /(double) (patchSize*n)) * hdes) - sy;

    Color c(particles[i].id);
    color_pixel(intensity,myWidthPx,myHeightPx,xpos+1,ypos,c.R,c.B,c.G);
    color_pixel(intensity,myWidthPx,myHeightPx,xpos-1,ypos,c.R,c.B,c.G);
    color_pixel(intensity,myWidthPx,myHeightPx,xpos,ypos+1,c.R,c.B,c.G);
    color_pixel(intensity,myWidthPx,myHeightPx,xpos,ypos-1,c.R,c.B,c.G);
    color_pixel(intensity,myWidthPx,myHeightPx,xpos,ypos,c.R,c.B,c.G);
  }
        
  for(int i=0;i<myHeightPx;++i) {
    for(int j=0;j<myWidthPx;++j) {
      // Draw red lines
      if(i==0 || j==0) {
	color_pixel(intensity,myWidthPx,myHeightPx,j,i,128,0,0);
      }
    }
  }
        
  liveVizDeposit(lvmsg, sx,sy, myWidthPx,myHeightPx, intensity, this, max_image_data);
  delete[] intensity;
}
#endif

// Prints all particles 
void Patch::print(){
#ifdef PRINT
  int i;
  CkPrintf("*****************************************************\n");
  CkPrintf("Patch (%d,%d)\n",thisIndex.x,thisIndex.y);

  // CkPrintf("Part     x     y\n");
  for(i=0; i < particles.length(); i++)
    CkPrintf("Patch (%d,%d) %-5d %7.4f %7.4f \n", thisIndex.x, thisIndex.y, i,
					      particles[i].x, particles[i].y);
  CkPrintf("*****************************************************\n");
#endif
}

#include "Patch.def.h"
