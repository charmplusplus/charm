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
/*readonly*/ int arrayDimX;
/*readonly*/ int arrayDimY;
/*readonly*/ int arrayDimZ;
/*readonly*/ int blockDimX;
/*readonly*/ int blockDimY;
/*readonly*/ int blockDimZ;
/*readonly*/ int torusDimX;
/*readonly*/ int torusDimY;
/*readonly*/ int torusDimZ;

// specify the number of worker chares in each dimension
/*readonly*/ int num_chare_x;
/*readonly*/ int num_chare_y;
/*readonly*/ int num_chare_z;

static unsigned long next = 1;

int myrand(int numpes) {
  next = next * 1103515245 + 12345;
  return((unsigned)(next/65536) % numpes);
}

#define SMPWAYX			1
#define SMPWAYY			2
#define SMPWAYZ			2
#define USE_TOPOMAP		0
#define USE_RRMAP		0
#define USE_BLOCKMAP		1
#define USE_BLOCK_RNDMAP	0
#define USE_BLOCK_RRMAP		1
#define USE_SMPMAP		0
#define USE_3D_ARRAYS		0

// We want to wrap entries around, and because mod operator % 
// sometimes misbehaves on negative values. -1 maps to the highest value.
#define wrap_x(a)	(((a)+num_chare_x)%num_chare_x)
#define wrap_y(a)	(((a)+num_chare_y)%num_chare_y)
#define wrap_z(a)	(((a)+num_chare_z)%num_chare_z)

#if USE_3D_ARRAYS
#define index(a, b, c)	a][b][c	
#else
#define index(a, b, c)	( (a)*(blockDimY+2)*(blockDimZ+2) + (b)*(blockDimZ+2) + (c) )
#endif

#define MAX_ITER		21
#define LEFT			1
#define RIGHT			2
#define TOP			3
#define BOTTOM			4
#define FRONT			5
#define BACK			6
#define DIVIDEBY7       	0.14285714285714285714

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
      if ( (m->argc != 3) && (m->argc != 7) && (m->argc != 10) ) {
        CkPrintf("%s [array_size] [block_size]\n", m->argv[0]);
        CkPrintf("OR %s [array_size_X] [array_size_Y] [array_size_Z] [block_size_X] [block_size_Y] [block_size_Z]\n", m->argv[0]);
        CkAbort("Abort");
      }

      // set iteration counter to zero
      iterations = 0;

      // store the main proxy
      mainProxy = thisProxy;
	
      if(m->argc == 3) {
	arrayDimX = arrayDimY = arrayDimZ = atoi(m->argv[1]);
        blockDimX = blockDimY = blockDimZ = atoi(m->argv[2]); 
      }
      else if (m->argc == 7) {
        arrayDimX = atoi(m->argv[1]);
	arrayDimY = atoi(m->argv[2]);
	arrayDimZ = atoi(m->argv[3]);
        blockDimX = atoi(m->argv[4]); 
	blockDimY = atoi(m->argv[5]); 
	blockDimZ = atoi(m->argv[6]);
      } else {
        arrayDimX = atoi(m->argv[1]);
	arrayDimY = atoi(m->argv[2]);
	arrayDimZ = atoi(m->argv[3]);
        blockDimX = atoi(m->argv[4]); 
	blockDimY = atoi(m->argv[5]); 
	blockDimZ = atoi(m->argv[6]);
	torusDimX = atoi(m->argv[7]);
	torusDimY = atoi(m->argv[8]);
	torusDimZ = atoi(m->argv[9]);
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
#if USE_TOPOMAP || USE_RRMAP || USE_BLOCKMAP || USE_BLOCK_RRMAP || USE_BLOCK_RNDMAP || USE_SMPMAP
       CkPrintf("Topology Mapping is being done ... %d %d %d %d\n", USE_TOPOMAP, USE_RRMAP, USE_BLOCKMAP, USE_SMPMAP);
      CProxy_JacobiMap map = CProxy_JacobiMap::ckNew(num_chare_x, num_chare_y, num_chare_z, torusDimX, torusDimY, torusDimZ);
      CkArrayOptions opts(num_chare_x, num_chare_y, num_chare_z);
      opts.setMap(map);
      array = CProxy_Jacobi::ckNew(opts);
#else
      array = CProxy_Jacobi::ckNew(num_chare_x, num_chare_y, num_chare_z);
#endif

      TopoManager tmgr;
      CkArray *jarr = array.ckLocalBranch();
      int jmap[num_chare_x][num_chare_y][num_chare_z];

      int hops=0, p;
      for(int i=0; i<num_chare_x; i++)
	for(int j=0; j<num_chare_y; j++)
	  for(int k=0; k<num_chare_z; k++) {
	    jmap[i][j][k] = jarr->procNum(CkArrayIndex3D(i, j, k));
	  }

      for(int i=0; i<num_chare_x; i++)
	for(int j=0; j<num_chare_y; j++)
	  for(int k=0; k<num_chare_z; k++) {
	    p = jmap[i][j][k];
	    hops += tmgr.getHopsBetweenRanks(p, jmap[wrap_x(i+1)][j][k]);
	    hops += tmgr.getHopsBetweenRanks(p, jmap[wrap_x(i-1)][j][k]);
	    hops += tmgr.getHopsBetweenRanks(p, jmap[i][wrap_y(j+1)][k]);
	    hops += tmgr.getHopsBetweenRanks(p, jmap[i][wrap_y(j-1)][k]);
	    hops += tmgr.getHopsBetweenRanks(p, jmap[i][j][wrap_z(k+1)]);
	    hops += tmgr.getHopsBetweenRanks(p, jmap[i][j][wrap_z(k-1)]);
	  }
      CkPrintf("Total Hops: %d\n", hops);

      //Start the computation
      array.begin_iteration();
    }

    // Each worker reports back to here when it completes an iteration
    void report() {
      if (iterations == 0) {
	iterations++;
	startTime = CkWallTimer();
	array.begin_iteration();
      }
      else {
	endTime = CkWallTimer();
	CkPrintf("TIME : %f\n", (endTime - startTime)/(MAX_ITER-1));
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

#if USE_3D_ARRAYS
    double ***temperature;
    double ***new_temperature;
#else
    double *temperature;
    double *new_temperature;
#endif

    // Constructor, initialize values
    Jacobi() {
        int i, j, k;
        // allocate a three dimensional array
#if USE_3D_ARRAYS
	temperature = new double**[blockDimX+2];
	new_temperature = new double**[blockDimX+2];
        for (i=0; i<blockDimX+2; i++) {
          temperature[i] = new double*[blockDimY+2];
          new_temperature[i] = new double*[blockDimY+2];
          for(j=0; j<blockDimY+2; j++) {
            temperature[i][j] = new double[blockDimZ+2];
            new_temperature[i][j] = new double[blockDimZ+2];
	  }
	}
#else
        temperature = new double[(blockDimX+2) * (blockDimY+2) * (blockDimZ+2)];
        new_temperature = new double[(blockDimX+2) * (blockDimY+2) * (blockDimZ+2)];
#endif

        for(i=0; i<blockDimX+2; ++i) {
          for(j=0; j<blockDimY+2; ++j) {
            for(k=0; k<blockDimZ+2; ++k) {
              temperature[index(i, j, k)] = 0.0;
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
            temperature[index(i, 1, k)] = 255.0;
        for(int j=1; j<blockDimY+1; ++j)
	  for(int k=1; k<blockDimZ+1; ++k)
            temperature[index(1, j, k)] = 255.0;
	for(int i=1; i<blockDimX+1; ++i)
          for(int j=1; j<blockDimY+1; ++j)
            temperature[index(i, j, 1)] = 255.0;
    }

    // a necessary function which we ignore now
    // if we were to use load balancing and migration
    // this function might become useful
    Jacobi(CkMigrateMessage* m) {}

    ~Jacobi() { 
#if USE_3D_ARRAYS
      for (int i=0; i<blockDimX+2; i++) {
        for(int j=0; j<blockDimY+2; j++) {
          delete [] temperature[i][j];
          delete [] new_temperature[i][j];
	}
        delete [] temperature[i];
        delete [] new_temperature[i];
      }
      delete [] temperature; 
      delete [] new_temperature; 
#else
      delete [] temperature; 
      delete [] new_temperature; 
#endif
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
            left_face[k*blockDimY+j] = temperature[index(1, j+1, k+1)];
            right_face[k*blockDimY+j] = temperature[index(blockDimX, j+1, k+1)];
        }

	for(int i=0; i<blockDimX; ++i) 
	  for(int k=0; k<blockDimZ; ++k) {
            top_face[k*blockDimX+i] = temperature[index(i+1, 1, k+1)];
            bottom_face[k*blockDimX+i] = temperature[index(i+1, blockDimY, k+1)];
        }

	for(int i=0; i<blockDimX; ++i) 
          for(int j=0; j<blockDimY; ++j) {
            front_face[j*blockDimX+i] = temperature[index(i+1, j+1, 1)];
            back_face[j*blockDimX+i] = temperature[index(i+1, j+1, blockDimZ)];
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
        thisProxy(thisIndex.x, thisIndex.y, wrap_z(thisIndex.z+1)).ghostsFromBack(blockDimX, blockDimY, front_face);

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
            temperature[index(blockDimX+1, j+1, k+1)] = ghost_values[k*height+j];
        }
        check_and_compute(RIGHT);
    }

    void ghostsFromLeft(int height, int width, double ghost_values[]) {
        for(int j=0; j<height; ++j) 
	  for(int k=0; k<width; ++k) {
            temperature[index(0, j+1, k+1)] = ghost_values[k*height+j];
        }
        check_and_compute(LEFT);
    }

    void ghostsFromBottom(int height, int width, double ghost_values[]) {
	for(int i=0; i<height; ++i) 
	  for(int k=0; k<width; ++k) {
            temperature[index(i+1, blockDimY+1, k+1)] = ghost_values[k*height+i];
        }
        check_and_compute(BOTTOM);
    }

    void ghostsFromTop(int height, int width, double ghost_values[]) {
	for(int i=0; i<height; ++i) 
	  for(int k=0; k<width; ++k) {
            temperature[index(i+1, 0, k+1)] = ghost_values[k*height+i];
        }
        check_and_compute(TOP);
    }

    void ghostsFromFront(int height, int width, double ghost_values[]) {
	for(int i=0; i<height; ++i) 
          for(int j=0; j<width; ++j) {
            temperature[index(i+1, j+1, blockDimZ+1)] = ghost_values[j*height+i];
        }
        check_and_compute(FRONT);
    }

    void ghostsFromBack(int height, int width, double ghost_values[]) {
	for(int i=0; i<height; ++i) 
          for(int j=0; j<width; ++j) {
            temperature[index(i+1, j+1, 0)] = ghost_values[j*height+i];
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
	  if (iterations == 1 || iterations == MAX_ITER)
	    contribute(0, 0, CkReduction::concat, CkCallback(CkIndex_Main::report(), mainProxy));
          else
            begin_iteration();
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

#pragma unroll    
        for(int i=1; i<blockDimX+1; ++i) {
          for(int j=1; j<blockDimY+1; ++j) {
            for(int k=1; k<blockDimZ+1; ++k) {
              // update my value based on the surrounding values
              new_temperature[index(i, j, k)] = (temperature[index(i-1, j, k)] 
					      +  temperature[index(i+1, j, k)]
					      +  temperature[index(i, j-1, k)]
					      +  temperature[index(i, j+1, k)]
					      +  temperature[index(i, j, k-1)]
					      +  temperature[index(i, j, k+1)]
					      +  temperature[index(i, j, k)] ) * DIVIDEBY7;
            }
	  }
        }

#pragma unroll
        for(int i=0;i<blockDimX+2;++i)
          for(int j=0;j<blockDimY+2;++j)
            for(int k=1; k<blockDimZ+1; ++k)
                temperature[index(i, j, k)] = new_temperature[index(i, j, k)];

        // Enforce the boundary conditions again
        BC();
    }

};

/** \class JacobiMap
 *
 */

class JacobiMap : public CkArrayMap {
  public:
    int X, Y, Z;
    int *mapping;

    JacobiMap(int x, int y, int z, int tx, int ty, int tz) {
      X = x; Y = y; Z = z;
      mapping = new int[X*Y*Z];

      // we are assuming that the no. of chares in each dimension is a 
      // multiple of the torus dimension

      TopoManager tmgr;
      int dimX, dimY, dimZ, dimT;

#if USE_TOPOMAP
      dimX = tmgr.getDimNX();
      dimY = tmgr.getDimNY();
      dimZ = tmgr.getDimNZ();
      dimT = tmgr.getDimNT();
#elif USE_BLOCKMAP
      dimX = tx;
      dimY = ty;
      dimZ = tz;
      dimT = 1;
#endif

#if USE_RRMAP
      if(CkMyPe()==0) CkPrintf("%d %d %d %d : %d %d %d\n", x, y, z, numCharesPerPe, numCharesPerPeX, numCharesPerPeY, numCharesPerPeZ); 
      int pe = 0;
      for(int i=0; i<x; i++)
	for(int j=0; j<y; j++)
	  for(int k=0; k<z; k++) {
	    if(pe == CkNumPes()) {
	      pe = 0;
	    }
	    mapping[i*Y*Z + j*Z + k] = pe;
	    pe++;
	  }
#elif USE_SMPMAP
      if(CkMyPe()==0) CkPrintf("%d %d %d %d : %d %d %d\n", x, y, z, numCharesPerPe, numCharesPerPeX, numCharesPerPeY, numCharesPerPeZ); 
      int pe = -1;
      x /= (numCharesPerPeX*SMPWAYX);
      y /= (numCharesPerPeY*SMPWAYY);
      z /= (numCharesPerPeZ*SMPWAYZ);

      for(int i=0; i<x; i++)
	for(int j=0; j<y; j++)
	  for(int k=0; k<z; k++)
	    for(int bi=i*SMPWAYX; bi<(i+1)*SMPWAYX; bi++)
	      for(int bj=j*SMPWAYY; bj<(j+1)*SMPWAYY; bj++)
		for(int bk=k*SMPWAYZ; bk<(k+1)*SMPWAYZ; bk++) {
		  pe++;
		  for(int ci=bi*numCharesPerPeX; ci<(bi+1)*numCharesPerPeX; ci++)
		    for(int cj=bj*numCharesPerPeY; cj<(bj+1)*numCharesPerPeY; cj++)
		      for(int ck=bk*numCharesPerPeZ; ck<(bk+1)*numCharesPerPeZ; ck++) {
			//if(CkMyPe()==0) CkPrintf("%d %d %d %d %d %d [%d]\n", i, j, k, ci, cj, ck, pe);
			mapping[ci*Y*Z + cj*Z + ck] = pe;
		      }
		}
#else

      // we are assuming that the no. of chares in each dimension is a 
      // multiple of the torus dimension
      int numCharesPerPe = X*Y*Z/CkNumPes();

      int numCharesPerPeX = X / dimX;
      int numCharesPerPeY = Y / dimY;
      int numCharesPerPeZ = Z / dimZ;
      int pe = 0, pes = CkNumPes();

#if USE_BLOCK_RNDMAP
      int used[pes];
      for(int i=0; i<pes; i++)
	used[i] = 0;
#endif

      if(dimT < 2) {	// one core per node
	if(CkMyPe()==0) CkPrintf("%d %d %d %d : %d %d %d \n", dimX, dimY, dimZ, dimT, numCharesPerPeX, numCharesPerPeY, numCharesPerPeZ); 
	for(int i=0; i<dimX; i++)
	  for(int j=0; j<dimY; j++)
	    for(int k=0; k<dimZ; k++)
	    {
#if USE_BLOCK_RNDMAP
	      pe = myrand(pes); 
	      while(used[pe]!=0) {
		pe = myrand(pes); 
	      }
	      used[pe] = 1;
#endif

	      for(int ci=i*numCharesPerPeX; ci<(i+1)*numCharesPerPeX; ci++)
		for(int cj=j*numCharesPerPeY; cj<(j+1)*numCharesPerPeY; cj++)
		  for(int ck=k*numCharesPerPeZ; ck<(k+1)*numCharesPerPeZ; ck++) {
#if USE_TOPOMAP
		    mapping[ci*Y*Z + cj*Z + ck] = tmgr.coordinatesToRank(i, j, k);
#elif USE_BLOCKMAP
		    mapping[ci*Y*Z + cj*Z + ck] = i + j*dimX + k*dimX*dimY;
  #if USE_BLOCK_RNDMAP
		    mapping[ci*Y*Z + cj*Z + ck] = pe;
  #elif USE_BLOCK_RRMAP
		    mapping[ci*Y*Z + cj*Z + ck] = pe;
  #endif
#endif
		  }
#if USE_BLOCK_RRMAP
	      pe++;
#endif
	    }
      } else {		// multiple cores per node
      // In this case, we split the chares in the X dimension among the
      // cores on the same node. The strange thing I figured out is that
      // doing this in the Z dimension is not as good.
      numCharesPerPeX /= dimT;
      if(CkMyPe()==0) CkPrintf("%d %d %d : %d %d %d %d : %d %d %d \n", x, y, z, dimX, dimY, dimZ, dimT, numCharesPerPeX, numCharesPerPeY, numCharesPerPeZ); 
      for(int i=0; i<dimX; i++)
	for(int j=0; j<dimY; j++)
	  for(int k=0; k<dimZ; k++)
	    for(int l=0; l<dimT; l++)
	      for(int ci=(dimT*i+l)*numCharesPerPeX; ci<(dimT*i+l+1)*numCharesPerPeX; ci++)
	        for(int cj=j*numCharesPerPeY; cj<(j+1)*numCharesPerPeY; cj++)
		  for(int ck=k*numCharesPerPeZ; ck<(k+1)*numCharesPerPeZ; ck++) {
		    mapping[ci*Y*Z + cj*Z + ck] = tmgr.coordinatesToRank(i, j, k, l);
		  }
      } // end of if
#endif

      if(CkMyPe() == 0) CkPrintf("Map generated ... \n");
    }

    ~JacobiMap() { 
      delete [] mapping;
    }

    int procNum(int, const CkArrayIndex &idx) {
      int *index = (int *)idx.data();
      return mapping[index[0]*Y*Z + index[1]*Z + index[2]]; 
    }
};

#include "jacobi3d.def.h"
