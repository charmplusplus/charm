#define MAX_ITER	200	
#define LEFT			1
#define RIGHT			2
#define TOP			3
#define BOTTOM			4
#define FRONT			5
#define BACK			6
#define DIVIDEBY7       	0.14285714285714285714
#define DELTA       	        0.01

#include "jacobi3d.decl.h"

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
/*readonly*/ int max_iter;

#define wrapX(a)	(((a)+num_chare_x)%num_chare_x)
#define wrapY(a)	(((a)+num_chare_y)%num_chare_y)
#define wrapZ(a)	(((a)+num_chare_z)%num_chare_z)

#define index(a,b,c)	((a)+(b)*(blockDimX+2)+(c)*(blockDimX+2)*(blockDimY+2))

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
    if ( (m->argc != 3) && (m->argc != 7) && (m->argc != 4) && (m->argc != 8)) {
      CkPrintf("%s [array_size] [block_size]\n", m->argv[0]);
      CkPrintf("OR %s [array_size_X] [array_size_Y] [array_size_Z] [block_size_X] [block_size_Y] [block_size_Z]\n", m->argv[0]);
      CkAbort("Abort");
    }

    // set iteration counter to zero
    iterations = 0;
    max_iter = MAX_ITER;
    // store the main proxy
    mainProxy = thisProxy;
	
    if(m->argc <5 ) {
      arrayDimX = arrayDimY = arrayDimZ = atoi(m->argv[1]);
      blockDimX = blockDimY = blockDimZ = atoi(m->argv[2]); 
      if(m->argc == 4)
          max_iter =  atoi(m->argv[3]);
    }
    else if (m->argc <9) {
      arrayDimX = atoi(m->argv[1]);
      arrayDimY = atoi(m->argv[2]);
      arrayDimZ = atoi(m->argv[3]);
      blockDimX = atoi(m->argv[4]); 
      blockDimY = atoi(m->argv[5]); 
      blockDimZ = atoi(m->argv[6]);
      if(m->argc == 8)
          max_iter =  atoi(m->argv[7]);
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
    array = CProxy_Jacobi::ckNew(num_chare_x, num_chare_y, num_chare_z);

    //Start the computation
    array.run();
    startTime = CkWallTimer();
  }

  void commonExit(int iter)
  {
    endTime = CkWallTimer();
    CkPrintf("Time elapsed per iteration: %f total time %f \n", (endTime - startTime) / iter, (endTime-startTime));
    CkExit();

  }
  void doneConverge(int done_iters) {
      CkPrintf(" finished due to convergence %d \n", done_iters); 
      commonExit(done_iters);
  }
  void doneIter(double error)
  {
      CkPrintf(" finished due to maximum iterations %d with error  %f \n", max_iter, error); 
      commonExit(max_iter);
  }
};

/** \class Jacobi
 *
 */

class Jacobi: public CBase_Jacobi {
  Jacobi_SDAG_CODE

public:
  int iterations;
  int neighbors;
  int remoteCount;
  double error;
  double *temperature;
  double *new_temperature;
  bool converged;

  // Constructor, initialize values
  Jacobi() {
    converged = false;
    neighbors = 6;
    if(thisIndex.x == 0) 
        neighbors--;
    if( thisIndex.x== num_chare_x-1)
        neighbors--;
    if(thisIndex.y == 0) 
        neighbors--;
    if( thisIndex.y== num_chare_y-1)
        neighbors--;
    if(thisIndex.z == 0) 
        neighbors--;
    if( thisIndex.z== num_chare_z-1)
          neighbors--;

    // allocate a three dimensional array
    temperature = new double[(blockDimX+2) * (blockDimY+2) * (blockDimZ+2)];
    new_temperature = new double[(blockDimX+2) * (blockDimY+2) * (blockDimZ+2)];

    for(int k=0; k<blockDimZ+2; ++k)
      for(int j=0; j<blockDimY+2; ++j)
        for(int i=0; i<blockDimX+2; ++i)
          new_temperature[index(i, j, k)] = temperature[index(i, j, k)] = 0.0;
    //print();
    iterations = 0;
    constrainBC();
    //print();
  }

  void pup(PUP::er &p)
  {
    CBase_Jacobi::pup(p);
    __sdag_pup(p);
    p|iterations;
    p|neighbors;

    size_t size = (blockDimX+2) * (blockDimY+2) * (blockDimZ+2);
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
    // Copy different faces into messages
    double *leftGhost =  new double[blockDimY*blockDimZ];
    double *rightGhost =  new double[blockDimY*blockDimZ];
    double *topGhost =  new double[blockDimX*blockDimZ];
    double *bottomGhost =  new double[blockDimX*blockDimZ];
    double *frontGhost =  new double[blockDimX*blockDimY];
    double *backGhost =  new double[blockDimX*blockDimY];
    for(int k=0; k<blockDimZ; ++k)
      for(int j=0; j<blockDimY; ++j) {
        leftGhost[k*blockDimY+j] = temperature[index(1, j+1, k+1)];
        rightGhost[k*blockDimY+j] = temperature[index(blockDimX, j+1, k+1)];
      }

    for(int k=0; k<blockDimZ; ++k)
      for(int i=0; i<blockDimX; ++i) {
        topGhost[k*blockDimX+i] = temperature[index(i+1, 1, k+1)];
        bottomGhost[k*blockDimX+i] = temperature[index(i+1, blockDimY, k+1)];
      }

    for(int j=0; j<blockDimY; ++j)
      for(int i=0; i<blockDimX; ++i) {
        frontGhost[j*blockDimX+i] = temperature[index(i+1, j+1, 1)];
        backGhost[j*blockDimX+i] = temperature[index(i+1, j+1, blockDimZ)];
      }

    int x = thisIndex.x, y = thisIndex.y, z = thisIndex.z;
    if(thisIndex.x>0)
        thisProxy(wrapX(x-1),y,z).updateGhosts(iterations, RIGHT,  blockDimY, blockDimZ, rightGhost);
    if(thisIndex.x<num_chare_x-1)
        thisProxy(wrapX(x+1),y,z).updateGhosts(iterations, LEFT,   blockDimY, blockDimZ, leftGhost);
    if(thisIndex.y>0)
        thisProxy(x,wrapY(y-1),z).updateGhosts(iterations, TOP,    blockDimX, blockDimZ, topGhost);
    if(thisIndex.y<num_chare_y-1)
        thisProxy(x,wrapY(y+1),z).updateGhosts(iterations, BOTTOM, blockDimX, blockDimZ, bottomGhost);
    if(thisIndex.z>0)
        thisProxy(x,y,wrapZ(z-1)).updateGhosts(iterations, BACK,   blockDimX, blockDimY, backGhost);
    if(thisIndex.z<num_chare_z-1)
        thisProxy(x,y,wrapZ(z+1)).updateGhosts(iterations, FRONT,  blockDimX, blockDimY, frontGhost);

    delete [] leftGhost;
    delete [] rightGhost;
    delete [] bottomGhost;
    delete [] topGhost;
    delete [] frontGhost;
    delete [] backGhost;
  }

  void updateBoundary(int dir, int height, int width, double* gh) {
    switch(dir) {
    case LEFT:
      for(int k=0; k<width; ++k)
        for(int j=0; j<height; ++j) {
          temperature[index(0, j+1, k+1)] = gh[k*height+j];
        }
      break;
    case RIGHT:
      for(int k=0; k<width; ++k)
        for(int j=0; j<height; ++j) {
          temperature[index(blockDimX+1, j+1, k+1)] = gh[k*height+j];
        }
      break;
    case BOTTOM:
      for(int k=0; k<width; ++k)
        for(int i=0; i<height; ++i) {
          temperature[index(i+1, 0, k+1)] = gh[k*height+i];
        }
      break;
    case TOP:
      for(int k=0; k<width; ++k)
        for(int i=0; i<height; ++i) {
          temperature[index(i+1, blockDimY+1, k+1)] = gh[k*height+i];
        }
      break;
    case FRONT:
      for(int j=0; j<width; ++j)
        for(int i=0; i<height; ++i) {
          temperature[index(i+1, j+1, 0)] = gh[j*height+i];
        }
      break;
    case BACK:
      for(int j=0; j<width; ++j)
        for(int i=0; i<height; ++i) {
          temperature[index(i+1, j+1, blockDimZ+1)] = gh[j*height+i];
        }
      break;
    default:
      CkAbort("ERROR\n");
    }
  }

  // Check to see if we have received all neighbor values yet
  // If all neighbor values have been received, we update our values and proceed
  double computeKernel() {
    double error = 0.0, max_error = 0.0;
    for(int k=1; k<blockDimZ+1; ++k)
      for(int j=1; j<blockDimY+1; ++j)
        for(int i=1; i<blockDimX+1; ++i) {
          // update my value based on the surrounding values
          new_temperature[index(i, j, k)] = (temperature[index(i-1, j, k)] 
                                             +  temperature[index(i+1, j, k)]
                                             +  temperature[index(i, j-1, k)]
                                             +  temperature[index(i, j+1, k)]
                                             +  temperature[index(i, j, k-1)]
                                             +  temperature[index(i, j, k+1)]
                                             +  temperature[index(i, j, k)] ) * DIVIDEBY7;
          error = fabs(new_temperature[index(i,j,k)] - temperature[index(i,j,k)]);
          if (error > max_error) {
            max_error = error;
          }
        } // end for
    
    double *tmp;
    tmp = temperature;
    temperature = new_temperature;
    new_temperature = tmp;

    //constrainBC();

    return max_error;
  }

  void print()
  {

    for(int k=1; k<blockDimZ+2; ++k)
      for(int j=1; j<blockDimY+2; ++j)
        for(int i=1; i<blockDimX+2; ++i)
          CkPrintf(" -%d:%d:%d %f ", k,j,i, temperature[index(k, j, i)]);
    CkPrintf("--------------------------------\n");
  }
  // Enforce some boundary conditions
  void constrainBC() {
    // // Heat right, left
    if(thisIndex.x == 0 )
        for(int j=0; j<blockDimY+2; ++j)
            for(int k=0; k<blockDimZ+2; ++k)
            {   
                new_temperature[index(0, j, k)] = temperature[index(0, j, k)] = 255.0;
            }
    if(thisIndex.y == 0 )
        for(int j=0; j<blockDimX+2; ++j)
            for(int k=0; k<blockDimZ+2; ++k)
            {   
                new_temperature[index(j,0, k)]  = temperature[index(j,0, k)] = 255.0;
            }
    if(thisIndex.z == 0 )
        for(int j=0; j<blockDimX+2; ++j)
            for(int k=0; k<blockDimY+2; ++k)
            {   
                new_temperature[index(j, k, 0)] = temperature[index(j, k, 0)] = 255.0;
            }

  }
};

#include "jacobi3d.def.h"
