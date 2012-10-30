/** \file jacobi3d.C
 *  Author: Abhinav S Bhatele
 *  Date Created: June 01st, 2009
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

// specify the number of worker chares in each dimension
/*readonly*/ int num_chare_x;
/*readonly*/ int num_chare_y;
/*readonly*/ int num_chare_z;

static unsigned long next = 1;

int myrand(int numpes) {
  next = next * 1103515245 + 12345;
  return((unsigned)(next/65536) % numpes);
}

// We want to wrap entries around, and because mod operator % 
// sometimes misbehaves on negative values. -1 maps to the highest value.
#define wrap_x(a)	(((a)+num_chare_x)%num_chare_x)
#define wrap_y(a)	(((a)+num_chare_y)%num_chare_y)
#define wrap_z(a)	(((a)+num_chare_z)%num_chare_z)

#define USE_3D_ARRAYS		0
#if USE_3D_ARRAYS
#define index(a, b, c)	a][b][c	
#else
#define index(a, b, c)	( (a)*(blockDimY+2)*(blockDimZ+2) + (b)*(blockDimZ+2) + (c) )
#endif

#define MAX_ITER		26
#define WARM_ITER		5
#define LEFT			1
#define RIGHT			2
#define TOP			3
#define BOTTOM			4
#define FRONT			5
#define BACK			6
#define DIVIDEBY7       	0.14285714285714285714

double startTime;
double endTime;

/** \class ghostMsg
 *
 */
class ghostMsg: public CMessage_ghostMsg {
  public:
    int dir;
    int height;
    int width;
    double* gh;

    ghostMsg(int _d, int _h, int _w) : dir(_d), height(_h), width(_w) {
    }
};

/** \class Main
 *
 */
class Main : public CBase_Main {
  public:
    CProxy_Jacobi array;
    int iterations;

    Main(CkArgMsg* m) {
      if ( (m->argc != 3) && (m->argc != 7) ) {
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
      CkPrintf("\nSTENCIL COMPUTATION WITH NO BARRIERS\n");
      CkPrintf("Running Jacobi on %d processors with (%d, %d, %d) chares\n", CkNumPes(), num_chare_x, num_chare_y, num_chare_z);
      CkPrintf("Array Dimensions: %d %d %d\n", arrayDimX, arrayDimY, arrayDimZ);
      CkPrintf("Block Dimensions: %d %d %d\n", blockDimX, blockDimY, blockDimZ);

      // Create new array of worker chares
#if USE_TOPOMAP
      CProxy_JacobiMap map = CProxy_JacobiMap::ckNew(num_chare_x, num_chare_y, num_chare_z);
      CkPrintf("Topology Mapping is being done ... \n");
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
      imsg = 0;
      constrainBC();
    }

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

    // Send ghost faces to the six neighbors
    void begin_iteration(void) {
      if (thisIndex.x == 0 && thisIndex.y == 0 && thisIndex.z == 0) {
        CkPrintf("Start of iteration %d\n", iterations);
#if CMK_BIGSIM_CHARM
        BgPrintf("BgPrint> Start of iteration at %f\n");
#endif
      }
      iterations++;

      // Copy different faces into messages
      ghostMsg *leftMsg = new (blockDimY*blockDimZ) ghostMsg(RIGHT, blockDimY, blockDimZ);
      ghostMsg *rightMsg = new (blockDimY*blockDimZ) ghostMsg(LEFT, blockDimY, blockDimZ);
      ghostMsg *topMsg = new (blockDimX*blockDimZ) ghostMsg(BOTTOM, blockDimX, blockDimZ);
      ghostMsg *bottomMsg = new (blockDimX*blockDimZ) ghostMsg(TOP, blockDimX, blockDimZ);
      ghostMsg *frontMsg = new (blockDimX*blockDimY) ghostMsg(BACK, blockDimX, blockDimY);
      ghostMsg *backMsg = new (blockDimX*blockDimY) ghostMsg(FRONT, blockDimX, blockDimY);

      CkSetRefNum(leftMsg, iterations);
      CkSetRefNum(rightMsg, iterations);
      CkSetRefNum(topMsg, iterations);
      CkSetRefNum(bottomMsg, iterations);
      CkSetRefNum(frontMsg, iterations);
      CkSetRefNum(backMsg, iterations);

      for(int j=0; j<blockDimY; ++j) 
	for(int k=0; k<blockDimZ; ++k) {
	  leftMsg->gh[k*blockDimY+j] = temperature[index(1, j+1, k+1)];
	  rightMsg->gh[k*blockDimY+j] = temperature[index(blockDimX, j+1, k+1)];
	}

      for(int i=0; i<blockDimX; ++i) 
	for(int k=0; k<blockDimZ; ++k) {
	  topMsg->gh[k*blockDimX+i] = temperature[index(i+1, 1, k+1)];
	  bottomMsg->gh[k*blockDimX+i] = temperature[index(i+1, blockDimY, k+1)];
	}

      for(int i=0; i<blockDimX; ++i) 
	for(int j=0; j<blockDimY; ++j) {
	  frontMsg->gh[j*blockDimX+i] = temperature[index(i+1, j+1, 1)];
	  backMsg->gh[j*blockDimX+i] = temperature[index(i+1, j+1, blockDimZ)];
	}

      // Send my left face
      thisProxy(wrap_x(thisIndex.x-1), thisIndex.y, thisIndex.z).receiveGhosts(leftMsg);
      // Send my right face
      thisProxy(wrap_x(thisIndex.x+1), thisIndex.y, thisIndex.z).receiveGhosts(rightMsg);
      // Send my bottom face
      thisProxy(thisIndex.x, wrap_y(thisIndex.y-1), thisIndex.z).receiveGhosts(bottomMsg);
      // Send my top face
      thisProxy(thisIndex.x, wrap_y(thisIndex.y+1), thisIndex.z).receiveGhosts(topMsg);
      // Send my front face
      thisProxy(thisIndex.x, thisIndex.y, wrap_z(thisIndex.z-1)).receiveGhosts(frontMsg);
      // Send my back face
      thisProxy(thisIndex.x, thisIndex.y, wrap_z(thisIndex.z+1)).receiveGhosts(backMsg);
    }

    void processGhosts(ghostMsg *gmsg) {
      int height = gmsg->height;
      int width = gmsg->width;

      switch(gmsg->dir) {
	case LEFT:
	  for(int j=0; j<height; ++j) 
	    for(int k=0; k<width; ++k) {
	      temperature[index(0, j+1, k+1)] = gmsg->gh[k*height+j];
	    }
	  break;
	case RIGHT:
	  for(int j=0; j<height; ++j) 
	    for(int k=0; k<width; ++k) {
	      temperature[index(blockDimX+1, j+1, k+1)] = gmsg->gh[k*height+j];
	    }
	  break;
	case BOTTOM:
	  for(int i=0; i<height; ++i) 
	    for(int k=0; k<width; ++k) {
	      temperature[index(i+1, 0, k+1)] = gmsg->gh[k*height+i];
	    }
	  break;
	case TOP:
	  for(int i=0; i<height; ++i) 
	    for(int k=0; k<width; ++k) {
	      temperature[index(i+1, blockDimY+1, k+1)] = gmsg->gh[k*height+i];
	    }
	  break;
	case FRONT:
	  for(int i=0; i<height; ++i) 
	    for(int j=0; j<width; ++j) {
	      temperature[index(i+1, j+1, 0)] = gmsg->gh[j*height+i];
	    }
	  break;
	case BACK:
	  for(int i=0; i<height; ++i) 
	    for(int j=0; j<width; ++j) {
	      temperature[index(i+1, j+1, blockDimZ+1)] = gmsg->gh[j*height+i];
	    }
	  break;
	default:
          CkAbort("ERROR\n");
      }

      delete gmsg;
    }


    void check_and_compute() {
      compute_kernel();

      // calculate error
      // not being done right now since we are doing a fixed no. of iterations

#if USE_3D_ARRAYS
      double ***tmp;
#else
      double *tmp;
#endif
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
    }

    // Enforce some boundary conditions
    void constrainBC() {
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

};

/** \class JacobiMap
 *
 */

class JacobiMap : public CkArrayMap {
  public:
    int X, Y, Z;
    int *mapping;

    JacobiMap(int x, int y, int z) {
      X = x; Y = y; Z = z;
      mapping = new int[X*Y*Z];

      // we are assuming that the no. of chares in each dimension is a 
      // multiple of the torus dimension

      TopoManager tmgr;
      int dimNX, dimNY, dimNZ, dimNT;

      dimNX = tmgr.getDimNX();
      dimNY = tmgr.getDimNY();
      dimNZ = tmgr.getDimNZ();
      dimNT = tmgr.getDimNT();

      // we are assuming that the no. of chares in each dimension is a 
      // multiple of the torus dimension
      int numCharesPerPe = X*Y*Z/CkNumPes();

      int numCharesPerPeX = X / dimNX;
      int numCharesPerPeY = Y / dimNY;
      int numCharesPerPeZ = Z / dimNZ;
      int pe = 0, pes = CkNumPes();

#if USE_BLOCK_RNDMAP
      int used[pes];
      for(int i=0; i<pes; i++)
	used[i] = 0;
#endif

      if(dimNT < 2) {	// one core per node
	if(CkMyPe()==0) CkPrintf("%d %d %d %d : %d %d %d \n", dimNX, dimNY, dimNZ, dimNT, numCharesPerPeX, numCharesPerPeY, numCharesPerPeZ); 
	for(int i=0; i<dimNX; i++)
	  for(int j=0; j<dimNY; j++)
	    for(int k=0; k<dimNZ; k++)
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
#elif USE_BLOCK_RNDMAP
		    mapping[ci*Y*Z + cj*Z + ck] = pe;
#endif
		  }
	    }
      } else {		// multiple cores per node
	// In this case, we split the chares in the X dimension among the
	// cores on the same node. The strange thing I figured out is that
	// doing this in the Z dimension is not as good.
	numCharesPerPeX /= dimNT;
	if(CkMyPe()==0) CkPrintf("%d %d %d %d : %d %d %d \n", dimNX, dimNY, dimNZ, dimNT, numCharesPerPeX, numCharesPerPeY, numCharesPerPeZ);

	for(int i=0; i<dimNX; i++)
	  for(int j=0; j<dimNY; j++)
	    for(int k=0; k<dimNZ; k++)
	      for(int l=0; l<dimNT; l++)
		for(int ci=(dimNT*i+l)*numCharesPerPeX; ci<(dimNT*i+l+1)*numCharesPerPeX; ci++)
		  for(int cj=j*numCharesPerPeY; cj<(j+1)*numCharesPerPeY; cj++)
		    for(int ck=k*numCharesPerPeZ; ck<(k+1)*numCharesPerPeZ; ck++) {
		      mapping[ci*Y*Z + cj*Z + ck] = tmgr.coordinatesToRank(i, j, k, l);
		    }
      } // end of if

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
