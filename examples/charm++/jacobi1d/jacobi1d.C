/** \file jacobi1d.C
 *  Author: Abhinav S Bhatele
 *  Date Created: July 16th, 2009
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
 *    Y: arrayDimY --> wrap_y
 * 
 *  This example does a 1D decomposition of a 2D data array
 */

#include "jacobi1d.decl.h"
#include "TopoManager.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int blockDimX;
/*readonly*/ int arrayDimX;
/*readonly*/ int arrayDimY;

// specify the number of worker chares in each dimension
/*readonly*/ int num_chares;

// We want to wrap entries around, and because mod operator % 
// sometimes misbehaves on negative values. -1 maps to the highest value.
#define wrap_y(a)  (((a)+num_chares)%num_chares)

#define MAX_ITER	26
#define WARM_ITER	5
#define TOP		1
#define BOTTOM		2

double startTime;
double endTime;

class Main : public CBase_Main
{
  public:
    CProxy_Jacobi array;
    int iterations;

    Main(CkArgMsg* m) {
      if (m->argc != 4) {
        CkPrintf("%s [array_size_X] [array_size_Y] [num_chares]\n", m->argv[0]);
        CkAbort("Abort");
      }

      CkPrintf("\nSTENCIL COMPUTATION WITH NO BARRIERS\n");
      arrayDimX = atoi(m->argv[1]);
      arrayDimY = atoi(m->argv[2]);
      num_chares = atoi(m->argv[3]);
      if (arrayDimX < num_chares || arrayDimX % num_chares != 0)
        CkAbort("array_size_X % num_chares != 0!");
      blockDimX = arrayDimX / num_chares;

      // store the main proxy
      mainProxy = thisProxy;

      // print info
      CkPrintf("Running Jacobi on %d processors with (%d) elements\n", CkNumPes(), num_chares);
      CkPrintf("Array Dimensions: %d %d\n", arrayDimX, arrayDimY);
      CkPrintf("Block Dimensions: %d %d\n", blockDimX, arrayDimY);

      // Create new array of worker chares
      array = CProxy_Jacobi::ckNew(num_chares);

      //Start the computation
      iterations = 0;
      array.begin_iteration();
    }

    // Each worker reports back to here when it completes an iteration
    void report(CkReductionMsg *msg) {
      iterations++;
      if(iterations == WARM_ITER)
	startTime = CkWallTimer();
      double error = *((double *)msg->getData());

      if(iterations < MAX_ITER) {
	CkPrintf("Start of iteration %d\n", iterations);
	array.begin_iteration();
      } else {
	CkPrintf("Completed %d iterations\n", MAX_ITER-1);
        endTime = CkWallTimer();
        CkPrintf("Time elapsed per iteration: %f\n", (endTime - startTime)/(MAX_ITER-1-WARM_ITER));
        CkExit();
      }
    }

};

class Jacobi: public CBase_Jacobi {
  public:
    int arrived_top;
    int arrived_bottom;

    double **temperature;
    double **new_temperature;
    void *sendLogs[4];
    void *ackLogs[5];
    int iterations;

    // Constructor, initialize values
    Jacobi() {
      int i,j;
      // allocate two dimensional arrays
      temperature = new double*[blockDimX+2];
      new_temperature = new double*[blockDimX+2];
      for (i=0; i<blockDimX+2; i++) {
	temperature[i] = new double[arrayDimY];
	new_temperature[i] = new double[arrayDimY];
      }
      for(i=0;i<blockDimX+2; i++) {
	for(j=0;j<arrayDimY; j++) {
	  temperature[i][j] = 0.5;
	  new_temperature[i][j] = 0.5;
	}
      }

      arrived_top = 0;
      arrived_bottom = 0;
      iterations = 0;
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
      iterations++;
	
      // Send my top edge
      thisProxy(wrap_y(thisIndex)).receiveGhosts(BOTTOM, arrayDimY, &temperature[1][1]);
      // Send my bottom edge
      thisProxy(wrap_y(thisIndex)).receiveGhosts(TOP, arrayDimY, &temperature[blockDimX][1]);
    }

    void receiveGhosts(int dir, int size, double gh[]) {
      int i, j;

      switch(dir) {
	case TOP:
	  arrived_top++;
	  for(j=0; j<size; j++)
            temperature[0][j+1] = gh[j];
	  break;
	case BOTTOM:
	  arrived_bottom++;
	  for(j=0; j<size; j++)
            temperature[blockDimX+1][j+1] = gh[j];
          break;
	default:
	  CkAbort("ERROR\n");
      }
      check_and_compute();
    }

    void check_and_compute() {
      double error = 0.0, max_error = 0.0;

      if (arrived_top >=1 && arrived_bottom >= 1) {
	arrived_top--;
	arrived_bottom--;

	compute_kernel();	

	for(int i=1; i<blockDimX+1; i++) {
	  for(int j=0; j<arrayDimY; j++) {
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

	contribute(sizeof(double), &max_error, CkReduction::max_double,
	      CkCallback(CkIndex_Main::report(NULL), mainProxy));
      }
    }

    // Check to see if we have received all neighbor values yet
    // If all neighbor values have been received, we update our values and proceed
    void compute_kernel()
    {
      for(int i=1; i<blockDimX+1; i++) {
	for(int j=0; j<arrayDimY; j++) {
	  // update my value based on the surrounding values
	  new_temperature[i][j] = (temperature[i-1][j]+temperature[i+1][j]+temperature[i][j-1]+temperature[i][j+1]+temperature[i][j]) * 0.2;
	}
      }
    }

    // Enforce some boundary conditions
    void constrainBC()
    {
     if(thisIndex <= num_chares/2) {
	for(int i=1; i<=blockDimX; i++)
	  temperature[i][1] = 1.0;
      }

      if(thisIndex == num_chares-1) {
	for(int j=arrayDimY/2; j<arrayDimY; j++)
	  temperature[blockDimX][j] = 0.0;
      }
    }

};

#include "jacobi1d.def.h"
