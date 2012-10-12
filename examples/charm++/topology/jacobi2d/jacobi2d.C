/** \file jacobi2d.C
 *  Author: Abhinav S Bhatele
 *  Date Created: October 24th, 2007
 *
 *
 *    ***********  ^
 *    *		*  |
 *    *		*  |
 *    *		*  X
 *    *		*  |
 *    *		*  |
 *    ***********  ~
 *    <--- Y --->
 *
 *    X: blockDimX, arrayDimX --> wrap_x
 *    Y: blockDimY, arrayDimY --> wrap_y
 */

#include "jacobi2d.decl.h"
#include "TopoManager.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int blockDimX;
/*readonly*/ int blockDimY;
/*readonly*/ int arrayDimX;
/*readonly*/ int arrayDimY;

// specify the number of worker chares in each dimension
/*readonly*/ int num_chare_x;
/*readonly*/ int num_chare_y;

// We want to wrap entries around, and because mod operator % 
// sometimes misbehaves on negative values. -1 maps to the highest value.
#define wrap_x(a)  (((a)+num_chare_x)%num_chare_x)
#define wrap_y(a)  (((a)+num_chare_y)%num_chare_y)

#define MAX_ITER	25
#define LEFT		1
#define RIGHT		2
#define TOP		3
#define BOTTOM		4

#define USE_TOPOMAP	0

double startTime;
double endTime;

class Main : public CBase_Main
{
  public:
    CProxy_Jacobi array;
    int iterations;

    Main(CkArgMsg* m) {
      if ( (m->argc != 3) && (m->argc != 5) ) {
        CkPrintf("%s [array_size] [block_size]\n", m->argv[0]);
        CkPrintf("%s [array_size_X] [array_size_Y] [block_size_X] [block_size_Y]\n", m->argv[0]);
        CkAbort("Abort");
      }

      if(m->argc == 3) {
        arrayDimY = arrayDimX = atoi(m->argv[1]);
        blockDimY = blockDimX = atoi(m->argv[2]);
      }
      else {
        arrayDimX = atoi(m->argv[1]);
        arrayDimY = atoi(m->argv[2]);
        blockDimX = atoi(m->argv[3]);
        blockDimY = atoi(m->argv[4]);
      }
      if (arrayDimX < blockDimX || arrayDimX % blockDimX != 0)
        CkAbort("array_size_X % block_size_X != 0!");
      if (arrayDimY < blockDimY || arrayDimY % blockDimY != 0)
        CkAbort("array_size_Y % block_size_Y != 0!");

      // store the main proxy
      mainProxy = thisProxy;

      num_chare_x = arrayDimX / blockDimX;
      num_chare_y = arrayDimY / blockDimY;

      // print info
      CkPrintf("Running Jacobi on %d processors with (%d, %d) elements\n", CkNumPes(), num_chare_x, num_chare_y);
      CkPrintf("Array Dimensions: %d %d\n", arrayDimX, arrayDimY);
      CkPrintf("Block Dimensions: %d %d\n", blockDimX, blockDimY);

      // Create new array of worker chares
#if USE_TOPOMAP
      CkPrintf("Topology Mapping is being done ... \n");
      CProxy_JacobiMap map = CProxy_JacobiMap::ckNew(num_chare_x, num_chare_y);
      CkArrayOptions opts(num_chare_x, num_chare_y);
      opts.setMap(map);
      array = CProxy_Jacobi::ckNew(opts);
#else
      array = CProxy_Jacobi::ckNew(num_chare_x, num_chare_y);
#endif

      //Start the computation
      iterations = 0;
      array.begin_iteration();
    }

    // Each worker reports back to here when it completes an iteration
    void report(CkReductionMsg *msg) {
      iterations++;
      if(iterations == 5)
	startTime = CkWallTimer();
      double error = *((double *)msg->getData());

      if (error > 0.001 && iterations < MAX_ITER) {
	array.begin_iteration();
      } else {
	CkPrintf("Completed %d iterations\n", iterations);
        endTime = CkWallTimer();
        CkPrintf("Time elapsed per iteration: %f\n", (endTime - startTime)/(MAX_ITER-5));
        CkExit();
      }
    }

};

class Jacobi: public CBase_Jacobi {
  public:
    int msgs;

    double **temperature;
    double **new_temperature;

    // Constructor, initialize values
    Jacobi() {
      int i,j;
      // allocate two dimensional arrays
      temperature = new double*[blockDimX+2];
      new_temperature = new double*[blockDimX+2];
      for (i=0; i<blockDimX+2; i++) {
	temperature[i] = new double[blockDimY+2];
	new_temperature[i] = new double[blockDimY+2];
      }
      for(i=0;i<blockDimX+2; i++) {
	for(j=0;j<blockDimY+2; j++) {
	  temperature[i][j] = 0.5;
	  new_temperature[i][j] = 0.5;
	}
      }

      msgs = 0;
      constrainBC();
    }

    Jacobi(CkMigrateMessage* m) {}

    ~Jacobi() { 
      for (int i=0; i<blockDimX+2; i++) {
        delete [] temperature[i];
        delete [] new_temperature[i];
      }
      delete [] temperature; 
      delete [] new_temperature; 
    }

    // Perform one iteration of work
    void begin_iteration(void) {
      // Copy left column and right column into temporary arrays
      double *left_edge = new double[blockDimX];
      double *right_edge = new double[blockDimX];

      for(int i=0; i<blockDimX; i++){
	  left_edge[i] = temperature[i+1][1];
	  right_edge[i] = temperature[i+1][blockDimY];
      }

      int x = thisIndex.x;
      int y = thisIndex.y;

      // Send my left edge
      thisProxy(x, wrap_y(y-1)).receiveGhosts(RIGHT, blockDimX, left_edge);
      // Send my right edge
      thisProxy(x, wrap_y(y+1)).receiveGhosts(LEFT, blockDimX, right_edge);
      // Send my top edge
      thisProxy(wrap_x(x-1), y).receiveGhosts(BOTTOM, blockDimY, &temperature[1][1]);
      // Send my bottom edge
      thisProxy(wrap_x(x+1), y).receiveGhosts(TOP, blockDimY, &temperature[blockDimX][1]);

      delete [] right_edge;
      delete [] left_edge;
    }

    void receiveGhosts(int dir, int size, double gh[]) {
      int i, j;
      switch(dir) {
	case LEFT:
	  for(i=0; i<size; i++)
            temperature[i+1][0] = gh[i];
	  break;
	case RIGHT:
	  for(i=0; i<size; i++)
            temperature[i+1][blockDimY+1] = gh[i];
	  break;
	case TOP:
	  for(j=0; j<size; j++)
            temperature[0][j+1] = gh[j];
	  break;
	case BOTTOM:
	  for(j=0; j<size; j++)
            temperature[blockDimX+1][j+1] = gh[j];
          break;
      }
      check_and_compute();
    }

    void check_and_compute() {
      double error = 0.0, max_error = 0.0;
      msgs++;

      if (msgs == 4) {
	msgs = 0;

	compute_kernel();	

	for(int i=1; i<blockDimX+1; i++) {
	  for(int j=1; j<blockDimY+1; j++) {
	    error = fabs(new_temperature[i][j] - temperature[i][j]);
	    if(error > max_error) {
	      max_error = error;
	    }
	  }
	}

	double **tmp;
	tmp = temperature;
	temperature = new_temperature;
	new_temperature = tmp;

	constrainBC();

	// if(thisIndex.x == 0 && thisIndex.y == 0) CkPrintf("Iteration %f %f %f\n", max_error, temperature[1][1], temperature[1][2]);
	 
	contribute(sizeof(double), &max_error, CkReduction::max_double,
	      CkCallback(CkIndex_Main::report(NULL), mainProxy));
      }
    }

    // Check to see if we have received all neighbor values yet
    // If all neighbor values have been received, we update our values and proceed
    void compute_kernel()
    {
      for(int i=1; i<blockDimX+1; i++) {
	for(int j=1; j<blockDimY+1; j++) {
	  // update my value based on the surrounding values
	  new_temperature[i][j] = (temperature[i-1][j]+temperature[i+1][j]+temperature[i][j-1]+temperature[i][j+1]+temperature[i][j]) * 0.2;
	}
      }
    }

    // Enforce some boundary conditions
    void constrainBC()
    {
     if(thisIndex.y == 0 && thisIndex.x < num_chare_x/2) {
	for(int i=1; i<=blockDimX; i++)
	  temperature[i][1] = 1.0;
      }

      if(thisIndex.x == num_chare_x-1 && thisIndex.y >= num_chare_y/2) {
	for(int j=1; j<=blockDimY; j++)
	  temperature[blockDimX][j] = 0.0;
      }
    }

};

/** \class JacobiMap
 *
 */

class JacobiMap : public CkArrayMap {
  public:
    int X, Y;
    int **mapping;

    JacobiMap(int x, int y) {
      X = x; Y = y;
      int i, j;
      mapping = new int*[x];
      for (i=0; i<x; i++)
        mapping[i] = new int[y];

      TopoManager tmgr;
      // we are assuming that the no. of chares in each dimension is a 
      // multiple of the torus dimension
      int dimX = tmgr.getDimNX();
      int dimY = tmgr.getDimNY();
      int dimZ = tmgr.getDimNZ();
      int dimT = tmgr.getDimNT();

      int numCharesPerZ = y/dimZ;
      int numCharesPerPeX = x / dimX;
      int numCharesPerPeY = numCharesPerZ / dimY;

      if(dimT < 2) {    // one core per node
      if(CkMyPe()==0) CkPrintf("%d %d %d %d : %d %d %d \n", dimX, dimY, dimZ, dimT, numCharesPerPeX, numCharesPerPeY, numCharesPerZ);
      for(int i=0; i<dimX; i++)
        for(int j=0; j<dimY; j++)
          for(int k=0; k<dimZ; k++)
	    for(int ci=i*numCharesPerPeX; ci<(i+1)*numCharesPerPeX; ci++)
	      for(int cj=j*numCharesPerPeY+k*numCharesPerZ; cj<(j+1)*numCharesPerPeY+k*numCharesPerZ; cj++)
		mapping[ci][cj] = tmgr.coordinatesToRank(i, j, k);
      }
    }

    ~JacobiMap() {
      for (int i=0; i<X; i++)
        delete [] mapping[i];
      delete [] mapping;
    }

    int procNum(int, const CkArrayIndex &idx) {
      int *index = (int *)idx.data();
      return mapping[index[0]][index[1]];
    }
};

#include "jacobi2d.def.h"
