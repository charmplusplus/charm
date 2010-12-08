/** \file jacobi2d.C
 *  Author: Eric Bohm and Abhinav S Bhatele
 *  This is Abhinav's jacobi3d-sdag cut down to 2d by Eric Bohm
 *  Date Created: Dec 7th, 2010
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

/*readonly*/ int globalBarrier;

static unsigned long next = 1;

int myrand(int numpes) {
  next = next * 1103515245 + 12345;
  return((unsigned)(next/65536) % numpes);
}

// We want to wrap entries around, and because mod operator % 
// sometimes misbehaves on negative values. -1 maps to the highest value.
#define wrap_x(a)	(((a)+num_chare_x)%num_chare_x)
#define wrap_y(a)	(((a)+num_chare_y)%num_chare_y)


#define index(a, b)	( (a)*(blockDimY+2) + (b) )

#define MAX_ITER		26
#define WARM_ITER		5
#define LEFT			1
#define RIGHT			2
#define TOP			3
#define BOTTOM			4
#define DIVIDEBY5       	0.2

double startTime;
double endTime;

/** \class Main
 *
 */
class Main : public CBase_Main {
  public:
    CProxy_Jacobi array;
    int iterations;

    Main(CkArgMsg* m) {
      if ( (m->argc != 3) && (m->argc != 5) ) {
        CkPrintf("%s [array_size] [block_size]\n", m->argv[0]);
        CkPrintf("OR %s [array_size_X] [array_size_Y] [block_size_X] [block_size_Y] \n", m->argv[0]);
        CkAbort("Abort");
      }

      // set iteration counter to zero
      iterations = 0;

      // store the main proxy
      mainProxy = thisProxy;
	
      if(m->argc == 3) {
	arrayDimX = arrayDimY = atoi(m->argv[1]);
        blockDimX = blockDimY  = atoi(m->argv[2]); 
      }
      else if (m->argc == 5) {
        arrayDimX = atoi(m->argv[1]);
	arrayDimY = atoi(m->argv[2]);
        blockDimX = atoi(m->argv[3]); 
	blockDimY = atoi(m->argv[4]); 
      }

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
      
      // make proxy and populate array in one call
      array = CProxy_Jacobi::ckNew(num_chare_x, num_chare_y);
      
      // initiate computation
      array.doStep();
    }

    // Each worker reports back to here when it completes an iteration
    void report() {
      iterations++;
      if (iterations <= WARM_ITER) {
	if (iterations == WARM_ITER)
	  startTime = CmiWallTimer();
	array.doStep();
      }
      else {
	CkPrintf("Completed %d iterations\n", MAX_ITER-1);
	endTime = CmiWallTimer();
	CkPrintf("Time elapsed per iteration: %f\n", (endTime - startTime)/(MAX_ITER-1-WARM_ITER));
        CkExit();
      }
    }
};

/** \class Jacobi
 *
 */

class Jacobi: public CBase_Jacobi {
  Jacobi_SDAG_CODE

  public:
    int iterations;
    int imsg;

    double *temperature;
    double *new_temperature;

    // Constructor, initialize values
    Jacobi() {
      __sdag_init();
      usesAtSync=CmiTrue;

      int i, j;
      // allocate a two dimensional array
      temperature = new double[(blockDimX+2) * (blockDimY+2)];
      new_temperature = new double[(blockDimX+2) * (blockDimY+2)];

      for(i=0; i<blockDimX+2; ++i) {
	for(j=0; j<blockDimY+2; ++j) {
	    temperature[index(i, j)] = 0.0;
	} 
      }

      iterations = 0;
      imsg = 0;
      constrainBC();
    }

  void pup(PUP::er &p)
  {
    CBase_Jacobi::pup(p);
    __sdag_pup(p);
    p|iterations;
    p|imsg;

    size_t size = (blockDimX+2) * (blockDimY+2);
    if (p.isUnpacking()) {
	temperature = new double[size];
	new_temperature = new double[size];
      }
    p(temperature, size);
    p(new_temperature, size);
  }

  Jacobi(CkMigrateMessage* m) {__sdag_init();}

    ~Jacobi() { 
      delete [] temperature; 
      delete [] new_temperature; 
    }

    // Send ghost faces to the six neighbors
    void begin_iteration(void) {
      AtSync();
      if (thisIndex.x == 0 && thisIndex.y == 0) {
          CkPrintf("Start of iteration %d\n", iterations);
          //BgPrintf("BgPrint> Start of iteration at %f\n");
      }
      iterations++;

      // Copy different faces into messages
      double *leftGhost =  new double[blockDimY];
      double *rightGhost =  new double[blockDimY];
      double *topGhost =  new double[blockDimX];
      double *bottomGhost =  new double[blockDimX];

      for(int j=0; j<blockDimY; ++j) {
	  leftGhost[j] = temperature[index(1, j+1)];
	  rightGhost[j] = temperature[index(blockDimX, j+1)];
      }

      for(int i=0; i<blockDimX; ++i) {
	  topGhost[i] = temperature[index(i+1, 1)];
	  bottomGhost[i] = temperature[index(i+1, blockDimY)];
      }
      // TODO for the inner dimension we can do this in one memcopy

      // Send my left face
      thisProxy(wrap_x(thisIndex.x-1), thisIndex.y)
	  .receiveGhosts(iterations, RIGHT, blockDimY, leftGhost);
      // Send my right face
      thisProxy(wrap_x(thisIndex.x+1), thisIndex.y)
	  .receiveGhosts(iterations, LEFT, blockDimY, rightGhost);
      // Send my top face
      thisProxy(thisIndex.x, wrap_y(thisIndex.y-1))
	  .receiveGhosts(iterations, BOTTOM, blockDimX, topGhost);
      // Send my bottom face
      thisProxy(thisIndex.x, wrap_y(thisIndex.y+1))
	  .receiveGhosts(iterations, TOP, blockDimX, bottomGhost);
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


    void check_and_compute() {
      compute_kernel();

      // calculate error
      // not being done right now since we are doing a fixed no. of iterations

      double *tmp;
      tmp = temperature;
      temperature = new_temperature;
      new_temperature = tmp;

      constrainBC();

      if (iterations <= WARM_ITER || iterations >= MAX_ITER)
	contribute(0, 0, CkReduction::concat, CkCallback(CkIndex_Main::report(), mainProxy));
      else
	doStep();
    }

    // Check to see if we have received all neighbor values yet
    // If all neighbor values have been received, we update our values and proceed
    void compute_kernel() {
#pragma unroll    
      for(int i=1; i<blockDimX+1; ++i) {
	for(int j=1; j<blockDimY+1; ++j) {
	    // update my value based on the surrounding values
	    new_temperature[index(i, j)] = (temperature[index(i-1, j)] 
					    +  temperature[index(i+1, j)]
					    +  temperature[index(i, j-1)]
					    +  temperature[index(i, j+1)]
					    +  temperature[index(i, j)] ) * DIVIDEBY5;
	}
      }
    }

    // Enforce some boundary conditions
    void constrainBC() {
      // Heat left and top  of each chare's block
      for(int i=1; i<blockDimX+1; ++i)
	  temperature[index(i, 1)] = 255.0;
      for(int j=1; j<blockDimY+1; ++j)
	  temperature[index(1, j)] = 255.0;
    }

};


#include "jacobi2d.def.h"
