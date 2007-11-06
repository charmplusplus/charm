/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/** \file jacobi3d.C
 *  Author: Abhinav S Bhatele
 *  Date Created: October 24th, 2007
 *
 *  This does a topological placement for a 3d jacobi.
 *
 *	
 *	      *****************
 *	   *		   *  *
 *   ^	*****************     *
 *   |	*		*     *
 *   |	*		*     *
 *   |	*		*     *
 *   Y	*		*     *
 *   |	*		*     *
 *   |	*		*     *
 *   |	*		*  * 
 *   ~	*****************    Z
 *	<------ X ------> 
 *
 *   X: left, right --> wrap_x
 *   Y: top, bottom --> wrap_y
 *   Z: front, back --> wrap_z
 */

#include "jacobi3d.decl.h"
#include "TopoManager.h"

// See README for documentation

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int blockDimX;
/*readonly*/ int blockDimY;
/*readonly*/ int blockDimZ;
/*readonly*/ int arrayDimX;
/*readonly*/ int arrayDimY;
/*readonly*/ int arrayDimZ;

// specify the number of worker chares in each dimension
/*readonly*/ int num_chare_x;
/*readonly*/ int num_chare_y;
/*readonly*/ int num_chare_z;

#define USE_TOPOMAP	0
// We want to wrap entries around, and because mod operator % 
// sometimes misbehaves on negative values. -1 maps to the highest value.
#define wrap_x(a)  (((a)+num_chare_x)%num_chare_x)
#define wrap_y(a)  (((a)+num_chare_y)%num_chare_y)
#define wrap_z(a)  (((a)+num_chare_z)%num_chare_z)

#define MAX_ITER	1000
#define LEFT		1
#define RIGHT		2
#define TOP		3
#define BOTTOM		4
#define FRONT		5
#define BACK		6

double startTime;
double endTime;

/** \class Main
 *
 */

class Main : public CBase_Main {
  public:
    int recieve_count;
    CProxy_Jacobi array;
    int num_chares;
    int iterations;

    Main(CkArgMsg* m) {
      if ( (m->argc != 3) && (m->argc != 7) ) {
        CkPrintf("%s [array_size] [block_size]\n", m->argv[0]);
        CkPrintf("OR %s [array_size_X] [array_size_Y] [array_size_Z] [block_size_X] [block_size_Y] [block_size_Z]\n", m->argv[0]);
        CkAbort("Abort");
      }

      // set iteration counter to zero
      iterations=0;

      // store the main proxy
      mainProxy = thisProxy;
	
      if(m->argc == 3) {
	arrayDimX = arrayDimY = arrayDimZ = atoi(m->argv[1]);
        blockDimX = blockDimY = blockDimZ = atoi(m->argv[2]); 
      }
      else {
        arrayDimX = atoi(m->argv[1]);
	arrayDimY = atoi(m->argv[2]);
	arrayDimZ = atoi(m->argv[3]);
        blockDimX = atoi(m->argv[4]); 
	blockDimY = atoi(m->argv[5]); 
	blockDimZ = atoi(m->argv[6]);
      }

      if (arrayDimX < blockDimX || arrayDimX % blockDimX != 0)
        CkAbort("array_size_X % block_size_X != 0!");
      if (arrayDimY < blockDimY || arrayDimY % blockDimY != 0)
        CkAbort("array_size_Y % block_size_Y != 0!");
      if (arrayDimZ < blockDimZ || arrayDimZ % blockDimZ != 0)
        CkAbort("array_size_Z % block_size_Z != 0!");

      num_chare_x = arrayDimX / blockDimX;
      num_chare_y = arrayDimY / blockDimY;
      num_chare_z = arrayDimZ / blockDimZ;

      // print info
      CkPrintf("Running Jacobi on %d processors with (%d, %d, %d) chares\n", CkNumPes(), num_chare_x, num_chare_y, num_chare_z);
      CkPrintf("Array Dimensions: %d %d %d\n", arrayDimX, arrayDimY, arrayDimZ);
      CkPrintf("Block Dimensions: %d %d %d\n", blockDimX, blockDimY, blockDimZ);

      // Create new array of worker chares
#if USE_TOPOMAP
      CkPrintf("Topology Mapping is being done ... \n");
      CProxy_JacobiMap map = CProxy_JacobiMap::ckNew(num_chare_x, num_chare_y, num_chare_z);
      CkArrayOptions opts(num_chare_x, num_chare_y, num_chare_z);
      opts.setMap(map);
      array = CProxy_Jacobi::ckNew(opts);
#else
      array = CProxy_Jacobi::ckNew(num_chare_x, num_chare_y, num_chare_z);
#endif

      // save the total number of worker chares we have in this simulation
      num_chares = num_chare_x * num_chare_y * num_chare_z;

      //Start the computation
      recieve_count = 0;
      startTime = CmiWallTimer();
      array.begin_iteration();
    }

    // Each worker reports back to here when it completes an iteration
    void report(int x, int y, int z) {
      recieve_count++;
      if (num_chares == recieve_count) {
	endTime = CmiWallTimer();
	CkPrintf("Time elapsed: %f\n", endTime - startTime);
        CkExit();
      }
    }
};

/** \class Jacobi
 *
 */

class Jacobi: public CBase_Jacobi {
  public:
    int arrived_left;
    int arrived_right;
    int arrived_top;
    int arrived_bottom;
    int arrived_front;
    int arrived_back;
    int iterations;
    int msgs;

    double ***temperature;

    // Constructor, initialize values
    Jacobi() {
        int i, j, k;
        // allocate a three dimensional array
        temperature = new double**[blockDimX+2];
        for (i=0; i<blockDimX+2; i++) {
          temperature[i] = new double*[blockDimY+2];
	  for(j=0; j<blockDimY+2; j++)
	    temperature[i][j] = new double[blockDimZ+2];
	}
        for(i=0; i<blockDimX+2; ++i) {
          for(j=0; j<blockDimY+2; ++j) {
            for(k=0; k<blockDimZ+2; ++k) {
              temperature[i][j][k] = 0.0;
            }
          } 
	}
	iterations = 0;
	arrived_left = 0;
	arrived_right = 0;
	arrived_top = 0;
	arrived_bottom = 0;
        arrived_front = 0;
        arrived_back = 0;
	msgs = 0;
        BC();
    }

    // Enforce some boundary conditions
    void BC() {
        // Heat left, top and front faces of each chare's block
	for(int i=1; i<blockDimX+1; ++i)
	  for(int k=1; k<blockDimZ+1; ++k)
            temperature[i][1][k] = 255.0;
        for(int j=1; j<blockDimY+1; ++j)
	  for(int k=1; k<blockDimZ+1; ++k)
            temperature[1][j][k] = 255.0;
	for(int i=1; i<blockDimX+1; ++i)
          for(int j=1; j<blockDimY+1; ++j)
            temperature[i][j][1] = 255.0;
    }

    // a necessary function which we ignore now
    // if we were to use load balancing and migration
    // this function might become useful
    Jacobi(CkMigrateMessage* m) {}

    ~Jacobi() { 
      for (int i=0; i<blockDimX+2; i++) {
	for(int j=0; j<blockDimY+2; j++)
	  delete [] temperature[i][j];
        delete [] temperature[i];
      }
      delete [] temperature; 
    }

    // Perform one iteration of work
    // The first step is to send the local state to the neighbors
    void begin_iteration(void) {

        // Copy different faces into temporary arrays
        double *left_face = new double[blockDimY*blockDimZ];
        double *right_face = new double[blockDimY*blockDimZ];
        double *top_face = new double[blockDimX*blockDimZ];
        double *bottom_face = new double[blockDimX*blockDimZ];
        double *front_face = new double[blockDimX*blockDimY];
        double *back_face = new double[blockDimX*blockDimY];

        for(int j=0; j<blockDimY; ++j) 
	  for(int k=0; k<blockDimZ; ++k) {
            left_face[k*blockDimY+j] = temperature[1][j+1][k+1];
            right_face[k*blockDimY+j] = temperature[blockDimX][j+1][k+1];
        }

	for(int i=0; i<blockDimX; ++i) 
	  for(int k=0; k<blockDimZ; ++k) {
            top_face[k*blockDimX+i] = temperature[i+1][1][k+1];
            bottom_face[k*blockDimX+i] = temperature[i+1][blockDimY][k+1];
        }

	for(int i=0; i<blockDimX; ++i) 
          for(int j=0; j<blockDimY; ++j) {
            front_face[j*blockDimX+i] = temperature[i+1][j+1][1];
            back_face[j*blockDimX+i] = temperature[i+1][j+1][blockDimZ];
        }

        // Send my left face
        thisProxy(wrap_x(thisIndex.x-1), thisIndex.y, thisIndex.z).ghostsFromRight(blockDimY, blockDimZ, left_face);
	// Send my right face
        thisProxy(wrap_x(thisIndex.x+1), thisIndex.y, thisIndex.z).ghostsFromLeft(blockDimY, blockDimZ, right_face);
	// Send my top face
        thisProxy(thisIndex.x, wrap_y(thisIndex.y-1), thisIndex.z).ghostsFromBottom(blockDimX, blockDimZ, top_face);
	// Send my bottom face
        thisProxy(thisIndex.x, wrap_y(thisIndex.y+1), thisIndex.z).ghostsFromTop(blockDimX, blockDimZ, bottom_face);
        // Send my front face
        thisProxy(thisIndex.x, thisIndex.y, wrap_z(thisIndex.z-1)).ghostsFromFront(blockDimX, blockDimY, back_face);
        // Send my back face
        thisProxy(thisIndex.x, thisIndex.y, wrap_z(thisIndex.z-1)).ghostsFromBack(blockDimX, blockDimY, front_face);

        delete [] back_face;
	delete [] front_face;
	delete [] bottom_face;
	delete [] top_face; 
        delete [] right_face;
        delete [] left_face;
    }

    void ghostsFromRight(int height, int width, double ghost_values[]) {
        for(int j=0; j<height; ++j) 
	  for(int k=0; k<width; ++k) {
            temperature[blockDimX][j+1][k+1] = ghost_values[k*height+j];
        }
        check_and_compute(RIGHT);
    }

    void ghostsFromLeft(int height, int width, double ghost_values[]) {
        for(int j=0; j<height; ++j) 
	  for(int k=0; k<width; ++k) {
            temperature[1][j+1][k+1] = ghost_values[k*height+j];
        }
        check_and_compute(LEFT);
    }

    void ghostsFromBottom(int height, int width, double ghost_values[]) {
	for(int i=0; i<height; ++i) 
	  for(int k=0; k<width; ++k) {
            temperature[i+1][blockDimY][k+1] = ghost_values[k*height+i];
        }
        check_and_compute(BOTTOM);
    }

    void ghostsFromTop(int height, int width, double ghost_values[]) {
	for(int i=0; i<height; ++i) 
	  for(int k=0; k<width; ++k) {
            temperature[i+1][1][k+1] = ghost_values[k*height+i];
        }
        check_and_compute(TOP);
    }

    void ghostsFromFront(int height, int width, double ghost_values[]) {
	for(int i=0; i<height; ++i) 
          for(int j=0; j<width; ++j) {
            temperature[i+1][j+1][blockDimZ] = ghost_values[j*height+i];
        }
        check_and_compute(FRONT);
    }

    void ghostsFromBack(int height, int width, double ghost_values[]) {
	for(int i=0; i<height; ++i) 
          for(int j=0; j<width; ++j) {
            temperature[i+1][j+1][1] = ghost_values[j*height+i];
        }
        check_and_compute(BACK);
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
	  case FRONT:
	    arrived_front++;
	    msgs++;
	    break;
	  case BACK:
	    arrived_back++;
	    msgs++;
	    break;
	}
        if (arrived_left >=1 && arrived_right >=1 && arrived_top >=1 && arrived_bottom >=1 && arrived_front >= 1 && arrived_back >= 1) {
	  arrived_left--;
	  arrived_right--;
	  arrived_top--;
	  arrived_bottom--;
	  arrived_front--;
	  arrived_back--;
          compute();
	  iterations++;
          if (iterations == MAX_ITER) {
            if(thisIndex.x==0 && thisIndex.y==0 && thisIndex.z==0) CkPrintf("Completed %d iterations\n", iterations);
            mainProxy.report(thisIndex.x, thisIndex.y, thisIndex.z);
            // CkExit();
          } else {
            // if(thisIndex.x==0 && thisIndex.y==0 && thisIndex.z==0) CkPrintf("%d ... \n", iterations);
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
        double new_temperature[blockDimX+2][blockDimY+2][blockDimZ+2];
    
        for(int i=1; i<blockDimX+1; ++i) {
          for(int j=1; j<blockDimY+1; ++j) {
            for(int k=1; k<blockDimZ+1; ++k) {
              // update my value based on the surrounding values
              new_temperature[i][j][k] = (temperature[i-1][j][k] + temperature[i+1][j][k] + temperature[i][j-1][k] + temperature[i][j+1][k] + temperature[i][j][k-1] + temperature[i][j][k+1] + temperature[i][j][k]) / 7.0;
            }
	  }
        }

        for(int i=0;i<blockDimX+2;++i)
          for(int j=0;j<blockDimY+2;++j)
            for(int k=1; k<blockDimZ+1; ++k)
                temperature[i][j][k] = new_temperature[i][j][k];

        // Enforce the boundary conditions again
        BC();
    }

};

/** \class
 *
 */

class JacobiMap : public CkArrayMap {
  public:
    int X, Y, Z;
    int ***mapping;

    JacobiMap(int x, int y, int z) {
      X = x; Y = y; Z = z;
      int i, j, k;
      mapping = new int**[x];
      for (i=0; i<x; i++) {
        mapping[i] = new int*[y];
	for(j=0; j<y; j++)
	  mapping[i][j] = new int[z];
      }

      TopoManager tmgr;
#if 0	// naive mapping
      for (i=0; i<x; i++)
	for(j=0; j<y; j++)
	  for(k=0; k<z; k++)
	    mapping[i][j][k] = tmgr.coordinatesToRank(i, j, k);
#else
      // we are assuming that the no. of chares in each dimension is a 
      // multiple of the torus dimension
      int dimX = tmgr.getDimX();
      int dimY = tmgr.getDimY();
      int dimZ = tmgr.getDimZ();
      
      int numCharesPerPeX = x / dimX;
      int numCharesPerPeY = y / dimY;
      int numCharesPerPeZ = z / dimZ;

      for(int i=0; i<dimX; i++)
	for(int j=0; j<dimY; j++)
	  for(int k=0; k<dimZ; k++)
	    for(int ci=i*numCharesPerPeX; ci<(i+1)*numCharesPerPeX; ci++)
	      for(int cj=j*numCharesPerPeY; cj<(j+1)*numCharesPerPeY; cj++)
		for(int ck=k*numCharesPerPeZ; ck<(k+1)*numCharesPerPeZ; ck++) {
		  mapping[ci][cj][ck] = tmgr.coordinatesToRank(i, j, k);
		}

#endif
    }

    ~JacobiMap() { 
      for (int i=0; i<X; i++) {
	for(int j=0; j<Y; j++)
	  delete [] mapping[i][j];
        delete [] mapping[i];
      }
      delete [] mapping;
    }

    int procNum(int, const CkArrayIndex &idx) {
      int *index = (int *)idx.data();
      return mapping[index[0]][index[1]][index[2]]; 
    }
};

#include "jacobi3d.def.h"
