/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/** \file jacobi2d.C
 *  Author: Abhinav S Bhatele
 *  Date Created: October 24th, 2007
 *
 *  This does a topological placement for a 2d jacobi.
 *  This jacobi is different from the one in ../../jacobi2d-iter in
 *  the sense that it does not use barriers
 *
 *
 *    ***********  ^
 *    *		*  |
 *    *		*  |
 *    *		*  Y
 *    *		*  |
 *    *		*  |
 *    ***********  ~
 *    <--- X --->
 *
 *    X: blockDimX, arrayDimX --> wrap_x
 *    Y: blockDimY, arrayDimY --> wrap_y
 */

#include "jacobi2d.decl.h"
#include "TopoManager.h"

// See README for documentation

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int blockDimX;
/*readonly*/ int blockDimY;
/*readonly*/ int arrayDimX;
/*readonly*/ int arrayDimY;

// specify the number of worker chares in each dimension
/*readonly*/ int num_chare_x;
/*readonly*/ int num_chare_y;

#define USE_TOPOMAP	1
// We want to wrap entries around, and because mod operator % 
// sometimes misbehaves on negative values. -1 maps to the highest value.
#define wrap_x(a)  (((a)+num_chare_x)%num_chare_x)
#define wrap_y(a)  (((a)+num_chare_y)%num_chare_y)

#define MAX_ITER	1000
#define LEFT		1
#define RIGHT		2
#define TOP		3
#define BOTTOM		4

double startTime;
double endTime;

/** \class Main
 *
 */

class Main : public CBase_Main
{
  public:
    int recieve_count;
    CProxy_Jacobi array;
    int num_chares;
    int iterations;

    Main(CkArgMsg* m) {
      if ( (m->argc != 3) && (m->argc != 5) ) {
        CkPrintf("%s [array_size] [block_size]\n", m->argv[0]);
        CkPrintf("%s [array_size_X] [array_size_Y] [block_size_X] [block_size_Y]\n", m->argv[0]);
        CkAbort("Abort");
      }

      // set iteration counter to zero
      iterations=0;

      // store the main proxy
      mainProxy = thisProxy;

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

      TopoManager tmgr;
      CkArray *jarr = array.ckLocalBranch();
      int jmap[num_chare_x][num_chare_y];

      int hops=0, p;
      for(int i=0; i<num_chare_x; i++)
        for(int j=0; j<num_chare_y; j++)
            jmap[i][j] = jarr->procNum(CkArrayIndex2D(i, j));

      for(int i=0; i<num_chare_x; i++)
        for(int j=0; j<num_chare_y; j++) {
          int p = jmap[i][j];
          hops += tmgr.getHopsBetweenRanks(p, jmap[wrap_x(i+1)][j]);
          hops += tmgr.getHopsBetweenRanks(p, jmap[wrap_x(i-1)][j]);
          hops += tmgr.getHopsBetweenRanks(p, jmap[i][wrap_y(j+1)]);
          hops += tmgr.getHopsBetweenRanks(p, jmap[i][wrap_y(j-1)]);
        }
      CkPrintf("Total Hops: %d\n", hops);

      // save the total number of worker chares we have in this simulation
      num_chares = num_chare_x * num_chare_y;

      //Start the computation
      recieve_count = 0;
      startTime = CmiWallTimer();
      array.begin_iteration();
    }

    // Each worker reports back to here when it completes an iteration
    void report(int x, int y) {
      recieve_count++;
      if (num_chares == recieve_count) {
	endTime = CmiWallTimer();
        CkPrintf("Time elapsed: %f\n", endTime - startTime);
        CkExit();
      }
    }
};

class Jacobi: public CBase_Jacobi {
  public:
    int arrived_left;
    int arrived_right;
    int arrived_top;
    int arrived_bottom;
    int iterations;
    int msgs;

    double **temperature;

    // Constructor, initialize values
    Jacobi() {
        int i,j;
        // allocate two dimensional array
        temperature = new double*[blockDimX+2];
        for (i=0; i<blockDimX+2; i++)
          temperature[i] = new double[blockDimY+2];
        for(i=0;i<blockDimX+2;++i){
            for(j=0;j<blockDimY+2;++j){
                temperature[i][j] = 0.0;
            }
        }
	iterations = 0;
	arrived_left = 0;
	arrived_right = 0;
	arrived_top = 0;
	arrived_bottom = 0;
	msgs = 0;
        BC();
    }

    // Enforce some boundary conditions
    void BC(){
        // Heat left and top edges of each chare's block
	for(int i=1;i<blockDimX+1;++i)
            temperature[i][1] = 255.0;
        for(int j=1;j<blockDimY+1;++j)
            temperature[1][j] = 255.0;
    }

    // a necessary function which we ignore now
    // if we were to use load balancing and migration
    // this function might become useful
    Jacobi(CkMigrateMessage* m) {}

    ~Jacobi() { 
      for (int i=0; i<blockDimX+2; i++)
        delete [] temperature[i];
      delete [] temperature; 
    }

    // Perform one iteration of work
    // The first step is to send the local state to the neighbors
    void begin_iteration(void) {

        // Copy left column and right column into temporary arrays
        double *left_edge = new double[blockDimY];
        double *right_edge = new double[blockDimY];

        for(int j=0;j<blockDimY;++j){
            left_edge[j] = temperature[1][j+1];
            right_edge[j] = temperature[blockDimX][j+1];
        }

        // Send my left edge
        thisProxy(wrap_x(thisIndex.x-1), thisIndex.y).ghostsFromRight(blockDimY, left_edge);
	// Send my right edge
        thisProxy(wrap_x(thisIndex.x+1), thisIndex.y).ghostsFromLeft(blockDimY, right_edge);
	// Send my top edge
        thisProxy(thisIndex.x, wrap_y(thisIndex.y-1)).ghostsFromBottom(blockDimX, &temperature[1][1]);
	// Send my bottom edge
        thisProxy(thisIndex.x, wrap_y(thisIndex.y+1)).ghostsFromTop(blockDimX, &temperature[blockDimY][1]);

        delete [] right_edge;
        delete [] left_edge;
    }

    void ghostsFromRight(int width, double ghost_values[]) {
        for(int j=0;j<width;++j){
            temperature[blockDimX+1][j+1] = ghost_values[j];
        }
        check_and_compute(RIGHT);
    }

    void ghostsFromLeft(int width, double ghost_values[]) {
        for(int j=0;j<width;++j){
            temperature[0][j+1] = ghost_values[j];
        }
        check_and_compute(LEFT);
    }

    void ghostsFromBottom(int width, double ghost_values[]) {
        for(int i=0;i<width;++i){
            temperature[i+1][blockDimY+1] = ghost_values[i];
        }
        check_and_compute(BOTTOM);
    }

    void ghostsFromTop(int width, double ghost_values[]) {
        for(int i=0;i<width;++i){
            temperature[i+1][0] = ghost_values[i];
        }
        check_and_compute(TOP);
    }

    void check_and_compute(int direction) {
        switch(direction) {
	  case LEFT:
	    arrived_left++;
	    msgs++;
	    break;
	  case RIGHT:
	    arrived_right++;
	    msgs++;
	    break;
	  case TOP:
	    arrived_top++;
	    msgs++;
	    break;
	  case BOTTOM:
	    arrived_bottom++;
	    msgs++;
	    break;
	}
        if (arrived_left >=1 && arrived_right >=1 && arrived_top >=1 && arrived_bottom >=1) {
	  arrived_left--;
	  arrived_right--;
	  arrived_top--;
	  arrived_bottom--;
          compute();
	  iterations++;
          if (iterations == MAX_ITER) {
            if(thisIndex.x==0 && thisIndex.y==0) CkPrintf("Completed %d iterations\n", iterations);
            //CkPrintf("INDEX %d %d MSGS %d\n", thisIndex.x, thisIndex.y, msgs);
            mainProxy.report(thisIndex.x, thisIndex.y);
            // CkExit();
          } else {
            //if(thisIndex.x==0 && thisIndex.y==0) CkPrintf("Starting new iteration %d.\n", iterations);
            // Call begin_iteration on all worker chares in array
            begin_iteration();
          } 
        }
    }

    // Check to see if we have received all neighbor values yet
    // If all neighbor values have been received, we update our values and proceed
    void compute() {
        // We must create a new array for these values because we don't want to 
        // update any of the values in temperature[][] array until using them first.
        // Other schemes could be used to accomplish this same problem. We just put
        // the new values in a temporary array and write them to temperature[][] 
        // after all of the new values are computed.
        double new_temperature[blockDimY+2][blockDimX+2];
    
        for(int i=1;i<blockDimX+1;++i) {
            for(int j=1;j<blockDimY+1;++j) {
                // update my value based on the surrounding values
                new_temperature[i][j] = (temperature[i-1][j]+temperature[i+1][j]+temperature[i][j-1]+temperature[i][j+1]+temperature[i][j]) * 0.2;

            }
        }

        for(int i=0;i<blockDimX+2;++i)
            for(int j=0;j<blockDimY+2;++j)
                temperature[i][j] = new_temperature[i][j];

        // Enforce the boundary conditions again
        BC();

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
