
#include "gaussSeidel3d.decl.h"
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

/*readonly*/ int globalBarrier;

static unsigned long next = 1;

int myrand(int numpes) {
  next = next * 1103515245 + 12345;
  return((unsigned)(next/65536) % numpes);
}


#define USE_3D_ARRAYS		0
#if USE_3D_ARRAYS
#define index(a, b, c)	a][b][c	
#else
#define index(a, b, c)	(a*(blockDimY+2)*(blockDimZ+2) + b*(blockDimZ+2) + c)
#endif

#define MAX_ITER		10
#define LEFT			1
#define RIGHT			2
#define TOP			3
#define BOTTOM			4
#define FRONT			5
#define BACK			6
#define DIVIDEBY7       	0.14285714285714285714


char * dirstring(int dir){
  switch(dir){
  case LEFT:
    return "LEFT";
  case RIGHT:
    return "RIGHT";
  case TOP:
    return "TOP";
  case BOTTOM:
    return "BOTTOM";
  case FRONT:
    return "FRONT";
  case BACK:
    return "BACK";
  }
}


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
  CProxy_GaussSeidel array;
  int iterations;

  Main(CkArgMsg* m) {
    if ( (m->argc != 3) && (m->argc != 7) ) {
      CkPrintf("%s [array_size] [block_size]\n", m->argv[0]);
      CkPrintf("OR %s [array_size_X] [array_size_Y] [array_size_Z] [block_size_X] [block_size_Y] [block_size_Z]\n", m->argv[0]);
      CkAbort("Abort");
    }

    traceRegisterUserEvent("Begin Iteration ***", 1000);

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
    CkPrintf("\nSTENCIL Gauss Seidel COMPUTATION WITH NO BARRIERS\n");
    CkPrintf("Running GaussSeidel on %d processors with (%d, %d, %d) chares\n", CkNumPes(), num_chare_x, num_chare_y, num_chare_z);
    CkPrintf("Array Dimensions: %d %d %d\n", arrayDimX, arrayDimY, arrayDimZ);
    CkPrintf("Block Dimensions: %d %d %d\n", blockDimX, blockDimY, blockDimZ);

    // Create new array of worker chares
    array = CProxy_GaussSeidel::ckNew(num_chare_x, num_chare_y, num_chare_z); 

    startTime = CkWallTimer();

    //Start the computation
    array.doStep();
  }

  // Each worker reports back to here when it completes an iteration
  void report() {
    iterations++;
    endTime = CkWallTimer();
    CkPrintf("Average time for each of the first %d iteration: %f\n", iterations ,(endTime - startTime)/(iterations));
    if(iterations >= MAX_ITER){
      CkExit();
    } else {
      array.doStep();	  
    }
  }
};

/** \class GaussSeidel
 *
 */

class GaussSeidel: public CBase_GaussSeidel {
  GaussSeidel_SDAG_CODE

  public:
  int iterations;
  int imsg;

#if USE_3D_ARRAYS
  double ***temperature;
#else
  double *temperature;
#endif

  // Constructor, initialize values
  GaussSeidel() {

    int i, j, k;
    // allocate a three dimensional array
#if USE_3D_ARRAYS
    temperature = new double**[blockDimX+2];
    for (i=0; i<blockDimX+2; i++) {
      temperature[i] = new double*[blockDimY+2];
      for(j=0; j<blockDimY+2; j++) {
	temperature[i][j] = new double[blockDimZ+2];
      }
    }
#else
    temperature = new double[(blockDimX+2) * (blockDimY+2) * (blockDimZ+2)];
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

  GaussSeidel(CkMigrateMessage* m) {}

  ~GaussSeidel() { 
#if USE_3D_ARRAYS
    for (int i=0; i<blockDimX+2; i++) {
      for(int j=0; j<blockDimY+2; j++) {
	delete [] temperature[i][j];
      }
      delete [] temperature[i];
    }
    delete [] temperature; 
#else
    delete [] temperature; 
#endif
  }

  // Send ghost faces to three of the neighbors containing the values computed during the previous step
  void begin_iteration(void) {
    //    CkPrintf("Elem %d,%d,%d Start of iteration %d\n", thisIndex.x, thisIndex.y, thisIndex.z, iterations);
    if(thisIndex.x == 0 and thisIndex.y == 0 and thisIndex.z == 0)
      traceUserEvent(1000);

    // Copy different faces into messages
      
      
    // Send my left face
    if(thisIndex.x > 0){
      ghostMsg *leftMsg = new (blockDimY*blockDimZ) ghostMsg(RIGHT, blockDimY, blockDimZ);
      CkSetRefNum(leftMsg, iterations);
      for(int j=0; j<blockDimY; ++j) 
	for(int k=0; k<blockDimZ; ++k) 
	  leftMsg->gh[k*blockDimY+j] = temperature[index(1, j+1, k+1)];
      thisProxy(thisIndex.x-1, thisIndex.y, thisIndex.z).receiveGhostsPrevX(leftMsg);
    }
      
    // Send my top face
    if(thisIndex.y > 0){
      ghostMsg *topMsg = new (blockDimX*blockDimZ) ghostMsg(BOTTOM, blockDimX, blockDimZ);
      CkSetRefNum(topMsg, iterations);
      for(int i=0; i<blockDimX; ++i) 
	for(int k=0; k<blockDimZ; ++k)
	  topMsg->gh[k*blockDimX+i] = temperature[index(i+1, 1, k+1)];
      thisProxy(thisIndex.x, thisIndex.y-1, thisIndex.z).receiveGhostsPrevY(topMsg);
    }
      
    // Send my front face
    if(thisIndex.z > 0){
      ghostMsg *frontMsg = new (blockDimX*blockDimY) ghostMsg(BACK, blockDimX, blockDimY);
      CkSetRefNum(frontMsg, iterations);
      for(int i=0; i<blockDimX; ++i) 
	for(int j=0; j<blockDimY; ++j) 
	  frontMsg->gh[j*blockDimX+i] = temperature[index(i+1, j+1, 1)];
      thisProxy(thisIndex.x, thisIndex.y, thisIndex.z-1).receiveGhostsPrevZ(frontMsg);
    }
      
      
  }
  
  
  
  void processGhosts(ghostMsg *gmsg) {
    //    CkPrintf("Elem %d,%d,%d processGhosts dir=%s\n", thisIndex.x, thisIndex.y, thisIndex.z, dirstring(gmsg->dir) );

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
    case TOP:
      for(int i=0; i<height; ++i) 
	for(int k=0; k<width; ++k) {
	  temperature[index(i+1, 0, k+1)] = gmsg->gh[k*height+i];
	}
      break;
    case BOTTOM:
      for(int i=0; i<height; ++i) 
	for(int k=0; k<width; ++k) {
	  temperature[index(i+1, blockDimY+1, k+1)] = gmsg->gh[k*height+i];
	}
      break;
    case FRONT:
      for(int i=0; i<height; ++i) 
	for(int j=0; j<width; ++j) {
	  temperature[index(i+1, j+1, blockDimZ+1)] = gmsg->gh[j*height+i];
	}
      break;
    case BACK:
      for(int i=0; i<height; ++i) 
	for(int j=0; j<width; ++j) {
	  temperature[index(i+1, j+1, 0)] = gmsg->gh[j*height+i];
	}
      break;
    default:
      CkAbort("ERROR\n");
    }

    delete gmsg;
  }


  void compute() {
    compute_kernel();
    constrainBC();


    // Send my right face
    if(thisIndex.x < num_chare_x-1){
      ghostMsg *rightMsg = new (blockDimY*blockDimZ) ghostMsg(LEFT, blockDimY, blockDimZ);
      CkSetRefNum(rightMsg, iterations);
      for(int j=0; j<blockDimY; ++j) 
	for(int k=0; k<blockDimZ; ++k) {
	  rightMsg->gh[k*blockDimY+j] = temperature[index(blockDimX, j+1, k+1)];
	}
      thisProxy(thisIndex.x+1, thisIndex.y, thisIndex.z).receiveGhostsCurrentX(rightMsg);
      //CkPrintf("Elem %d,%d,%d AFTER KERNEL sending to x+1 iter=%d\n", thisIndex.x, thisIndex.y, thisIndex.z, iterations);
    }
            
    // Send my bottom face
    if(thisIndex.y < num_chare_y-1){
      ghostMsg *bottomMsg = new (blockDimX*blockDimZ) ghostMsg(TOP, blockDimX, blockDimZ);
      CkSetRefNum(bottomMsg, iterations);
      for(int i=0; i<blockDimX; ++i) 
	for(int k=0; k<blockDimZ; ++k) {
	  bottomMsg->gh[k*blockDimX+i] = temperature[index(i+1, blockDimY, k+1)];
	}
      thisProxy(thisIndex.x, thisIndex.y+1, thisIndex.z).receiveGhostsCurrentY(bottomMsg);
      //CkPrintf("Elem %d,%d,%d AFTER KERNEL sending to y+1 iter=%d\n", thisIndex.x, thisIndex.y, thisIndex.z, iterations);
    }
      
    // Send my back face
    if(thisIndex.z < num_chare_z-1){
      ghostMsg *backMsg = new (blockDimX*blockDimY) ghostMsg(FRONT, blockDimX, blockDimY);
      CkSetRefNum(backMsg, iterations);
      for(int i=0; i<blockDimX; ++i) 
	for(int j=0; j<blockDimY; ++j) {
	  backMsg->gh[j*blockDimX+i] = temperature[index(i+1, j+1, blockDimZ)];
	}
      thisProxy(thisIndex.x, thisIndex.y, thisIndex.z+1).receiveGhostsCurrentZ(backMsg);
      //CkPrintf("Elem %d,%d,%d AFTER KERNEL sending to z+1 iter=%d\n", thisIndex.x, thisIndex.y, thisIndex.z, iterations);
    }


    iterations++;

    contribute(0, 0, CkReduction::concat, CkCallback(CkIndex_Main::report(), mainProxy));
  }


  void compute_kernel() {
#pragma unroll    
    for(int i=1; i<blockDimX+1; ++i) {
      for(int j=1; j<blockDimY+1; ++j) {
	for(int k=1; k<blockDimZ+1; ++k) {
	  // update my value based on the surrounding values
	  temperature[index(i, j, k)] = (temperature[index(i-1, j, k)] 
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


#include "gaussSeidel3d.def.h"
