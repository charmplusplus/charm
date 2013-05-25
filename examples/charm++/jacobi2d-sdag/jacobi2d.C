/** \file jacobi2d.C
 *  Author: Eric Bohm and Abhinav S Bhatele
 *
 *  This is jacobi3d-sdag cut down to 2d and fixed to be a correct
 *  implementation of the finite difference method by Eric Bohm.
 *
 *  Date Created: Dec 7th, * 2010
 *
 */

#include "jacobi2d.decl.h"

// See README for documentation

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int arrayDimX;
/*readonly*/ int arrayDimY;
/*readonly*/ int blockDimX;
/*readonly*/ int blockDimY;

// specify the number of worker chares in each dimension
/*readonly*/ int num_chare_x;
/*readonly*/ int num_chare_y;

/*readonly*/ int maxiterations;

static unsigned long next = 1;



#define index(a, b)	( (b)*(blockDimX+2) + (a) )

#define MAX_ITER		100
#define WARM_ITER		2
#define LEFT			1
#define RIGHT			2
#define TOP			3
#define BOTTOM			4
#define DIVIDEBY5       	0.2
const double THRESHHOLD =  0.001;

double startTime;
double endTime;

/** \class Main
 *
 */
class Main : public CBase_Main {
public:
  CProxy_Jacobi array;
  int iterations;
  double maxdifference;
  Main_SDAG_CODE;
  Main(CkArgMsg* m) {
    if ( (m->argc < 3) || (m->argc > 6)) {
      CkPrintf("%s [array_size] [block_size]\n", m->argv[0]);
      CkPrintf("OR %s [array_size] [block_size] maxiterations\n", m->argv[0]);
      CkPrintf("OR %s [array_size_X] [array_size_Y] [block_size_X] [block_size_Y] \n", m->argv[0]);
      CkPrintf("OR %s [array_size_X] [array_size_Y] [block_size_X] [block_size_Y] maxiterations\n", m->argv[0]);
      CkAbort("Abort");
    }

    // set iteration counter to zero
    iterations = 0;
    // store the main proxy
    mainProxy = thisProxy;
	
    if(m->argc <= 4) {
      arrayDimX = arrayDimY = atoi(m->argv[1]);
      blockDimX = blockDimY = atoi(m->argv[2]);
    }
    else if (m->argc >= 5) {
      arrayDimX = atoi(m->argv[1]);
      arrayDimY = atoi(m->argv[2]);
      blockDimX = atoi(m->argv[3]); 
      blockDimY = atoi(m->argv[4]); 
    }
    maxiterations=MAX_ITER;
    if(m->argc==4)
      maxiterations=atoi(m->argv[3]); 
    if(m->argc==6)
      maxiterations=atoi(m->argv[5]); 
      
    if (arrayDimX < blockDimX || arrayDimX % blockDimX != 0)
      CkAbort("array_size_X % block_size_X != 0!");
    if (arrayDimY < blockDimY || arrayDimY % blockDimY != 0)
      CkAbort("array_size_Y % block_size_Y != 0!");

    num_chare_x = arrayDimX / blockDimX;
    num_chare_y = arrayDimY / blockDimY;

    // print info
    CkPrintf("\nSTENCIL COMPUTATION WITH NO BARRIERS\n");
    CkPrintf("Running Jacobi on %d processors with (%d, %d) chares\n", CkNumPes(), num_chare_x, num_chare_y);
    CkPrintf("Array Dimensions: %d %d\n", arrayDimX, arrayDimY);
    CkPrintf("Block Dimensions: %d %d\n", blockDimX, blockDimY);
    CkPrintf("max iterations %d\n", maxiterations);
    CkPrintf("Threshhold %.10g\n", THRESHHOLD);
      
    // NOTE: boundary conditions must be set based on values
      
    // make proxy and populate array in one call
    array = CProxy_Jacobi::ckNew(num_chare_x, num_chare_y);

    // initiate computation
    thisProxy.run();
  }

  void done(bool success) {
      if(success)
	CkPrintf("Difference %.10g Satisfied Threshhold %.10g in %d Iterations\n", maxdifference,THRESHHOLD,iterations);
      else
	CkPrintf("Completed %d Iterations , Difference %lf fails threshhold\n", iterations,maxdifference);
      endTime = CkWallTimer();
      CkPrintf("Time elapsed per iteration: %f\n", (endTime - startTime)/(maxiterations-1-WARM_ITER));
      CkExit();
  }
};

/** \class Jacobi
 *
 */

class Jacobi: public CBase_Jacobi {
  Jacobi_SDAG_CODE

  public:
  double *temperature;
  double *new_temperature;
  int imsg;
  int iterations;
  int numExpected;
  int istart,ifinish,jstart,jfinish;
  double maxdifference;
  bool leftBound, rightBound, topBound, bottomBound;
  // Constructor, initialize values
  Jacobi() {
    usesAtSync=true;

    int i, j;
    // allocate a two dimensional array
    temperature = new double[(blockDimX+2) * (blockDimY+2)];
    new_temperature = new double[(blockDimX+2) * (blockDimY+2)];

    for(i=0; i<blockDimX+2; ++i) {
      for(j=0; j<blockDimY+2; ++j) {
	temperature[index(i, j)] = 0.;
      } 
    }
    imsg=0;
    iterations = 0;
    numExpected=0;
    maxdifference=0.;
    // determine border conditions
    leftBound=rightBound=topBound=bottomBound=false;
    istart=jstart=1;
    ifinish=blockDimX+1;
    jfinish=blockDimY+1;

    if(thisIndex.x==0)
      {
	leftBound=true;
	istart++;
      }
    else
      numExpected++;

    if(thisIndex.x==num_chare_x-1)
      {
	rightBound=true;
	ifinish--;
      }
    else
      numExpected++;

    if(thisIndex.y==0)
      {
	topBound=true;
	jstart++;
      }
    else
      numExpected++;

    if(thisIndex.y==num_chare_y-1)
      {
	bottomBound=true;
	jfinish--;
      }
    else
      numExpected++;
    constrainBC();
  }

  void pup(PUP::er &p)
  {
    CBase_Jacobi::pup(p);
    __sdag_pup(p);
    p|imsg;
    p|iterations;
    p|numExpected;
    p|maxdifference;
    p|istart; p|ifinish; p|jstart; p|jfinish;
    p|leftBound; p|rightBound; p|topBound; p|bottomBound;
    
    size_t size = (blockDimX+2) * (blockDimY+2);
    if (p.isUnpacking()) {
      temperature = new double[size];
      new_temperature = new double[size];
    }
    p(temperature, size);
    p(new_temperature, size);
  }

  Jacobi(CkMigrateMessage* m) { }

  ~Jacobi() { 
    delete [] temperature; 
    delete [] new_temperature; 
  }

  // Send ghost faces to the six neighbors
  void begin_iteration(void) {
    AtSync();
    if (thisIndex.x == 0 && thisIndex.y == 0) {
      CkPrintf("Start of iteration %d\n", iterations);
    }
    iterations++;

    if(!leftBound)
      {
	double *leftGhost =  new double[blockDimY];
	for(int j=0; j<blockDimY; ++j) 
	  leftGhost[j] = temperature[index(1, j+1)];
	thisProxy(thisIndex.x-1, thisIndex.y)
	  .receiveGhosts(iterations, RIGHT, blockDimY, leftGhost);
	delete [] leftGhost;
      }
    if(!rightBound)
      {
	double *rightGhost =  new double[blockDimY];
	for(int j=0; j<blockDimY; ++j) 
	  rightGhost[j] = temperature[index(blockDimX, j+1)];
        thisProxy(thisIndex.x+1, thisIndex.y)
	  .receiveGhosts(iterations, LEFT, blockDimY, rightGhost);
	delete [] rightGhost;
      }
    if(!topBound)
      {
	double *topGhost =  new double[blockDimX];
	for(int i=0; i<blockDimX; ++i) 
	  topGhost[i] = temperature[index(i+1, 1)];
	thisProxy(thisIndex.x, thisIndex.y-1)
	  .receiveGhosts(iterations, BOTTOM, blockDimX, topGhost);
	delete [] topGhost;
      }
    if(!bottomBound)
      {
	double *bottomGhost =  new double[blockDimX];
	for(int i=0; i<blockDimX; ++i) 
	  bottomGhost[i] = temperature[index(i+1, blockDimY)];
	thisProxy(thisIndex.x, thisIndex.y+1)
	  .receiveGhosts(iterations, TOP, blockDimX, bottomGhost);
	delete [] bottomGhost;
      }
  }

  void processGhosts(int dir, int size, double gh[]) {
    switch(dir) {
    case LEFT:
      for(int j=0; j<size; ++j) {
	temperature[index(0, j+1)] = gh[j];
      }
      break;
    case RIGHT:
      for(int j=0; j<size; ++j) {
	temperature[index(blockDimX+1, j+1)] = gh[j];
      }
      break;
    case TOP:
      for(int i=0; i<size; ++i) {
	temperature[index(i+1, 0)] = gh[i];
      }
      break;
    case BOTTOM:
      for(int i=0; i<size; ++i) {
	temperature[index(i+1, blockDimY+1)] = gh[i];
      }
      break;
    default:
      CkAbort("ERROR\n");
    }
  }


  void check_and_compute(CkCallback cb, int numSteps) {
    compute_kernel();

    double *tmp;
    tmp = temperature;
    temperature = new_temperature;
    new_temperature = tmp;
    constrainBC();

    contribute(sizeof(double), &maxdifference, CkReduction::max_double, cb);
    if (numSteps > 1)
      doStep(cb, numSteps-1);
  }


  // When all neighbor values have been received, 
  // we update our values and proceed
  void compute_kernel() {
    double temperatureIth=0.;
    double difference=0.;
    maxdifference=0.;
#pragma unroll    
    for(int i=istart; i<ifinish; ++i) {
      for(int j=jstart; j<jfinish; ++j) {
	// calculate discrete mean value property 5 pt stencil
	temperatureIth=(temperature[index(i, j)] 
			+ temperature[index(i-1, j)] 
			+  temperature[index(i+1, j)]
			+  temperature[index(i, j-1)]
			+  temperature[index(i, j+1)]) * 0.2;

	// update relative error
	difference=temperatureIth - temperature[index(i, j)];
	// fix sign without fabs overhead
	if(difference<0) difference*=-1.0; 
	maxdifference=(maxdifference>difference) ? maxdifference : difference;
	new_temperature[index(i, j)] = temperatureIth;
      }
    }
  }

  // Enforce some boundary conditions
  void constrainBC() {
    if(topBound)
      for(int i=0; i<blockDimX+2; ++i)
	temperature[index(i, 1)] = 1.0;
    if(leftBound)
      for(int j=0; j<blockDimY+2; ++j)
	temperature[index(1, j)] = 1.0;

    if(bottomBound)
      for(int i=0; i<blockDimX+2; ++i)
	temperature[index(i, blockDimY)] = 0.;
    if(rightBound)
      for(int j=0; j<blockDimY+2; ++j)
	temperature[index(blockDimX, j)] = 0.;

  }
  // for debugging
 void dumpMatrix(double *matrix)
  {
    CkPrintf("[%d,%d]\n",thisIndex.x, thisIndex.y);
    for(int i=0; i<blockDimX+2;++i)
      {
	for(int j=0; j<blockDimY+2;++j)
	  {
	    CkPrintf("%0.3lf ",matrix[index(i,j)]);
	  }
	CkPrintf("\n");
      }
  }
};


#include "jacobi2d.def.h"
