/** \file jacobi1d.C
 *  Author: Harshitha Menon, Yanhua Sun
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
 *  This example does a 1D decomposition of a 2D data array
 */

#include "jacobi1d.decl.h"
#include "TopoManager.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int blockDimX;
/*readonly*/ int arrayDimX;
/*readonly*/ int arrayDimY;

// specify the number of worker chares 
/*readonly*/ int numChares;
/*readonly*/ int maxiterations;

#define MAX_ITER		100
#define TOP		1
#define BOTTOM		2
const double THRESHOLD =  0.004;

class Main : public CBase_Main
{
  double startTime;
  double endTime;

public:
  CProxy_Jacobi array;

  Main(CkArgMsg* m) {
    if (m->argc!=4 && m->argc!=5 ) {
      CkPrintf("%s [array_size_X] [array_size_Y] [numChares]\n", m->argv[0]);
      CkPrintf("OR %s [array_size_X] [array_size_Y] [numChares] maxiterations\n", m->argv[0]);
      CkAbort("Abort");
    }

    CkPrintf("\nSTENCIL COMPUTATION WITH NO BARRIERS\n");
    arrayDimX = atoi(m->argv[1]);
    arrayDimY = atoi(m->argv[2]);
    numChares = atoi(m->argv[3]);
    if (arrayDimX < numChares || arrayDimX % numChares != 0)
      CkAbort("array_size_X %% numChares != 0!");
    blockDimX = arrayDimX / numChares;

    maxiterations = MAX_ITER;
    if(m->argc==5)
      maxiterations = atoi(m->argv[4]); 
    // store the main proxy
    mainProxy = thisProxy;

    // print info
    CkPrintf("Running Jacobi on %d processors with (%d) elements\n", CkNumPes(), numChares);
    CkPrintf("Array Dimensions: %d %d\n", arrayDimX, arrayDimY);
    CkPrintf("Block Dimensions: %d %d\n", arrayDimX, blockDimX);

    // Create new array of worker chares
    array = CProxy_Jacobi::ckNew(numChares);

    //Start the computation
    startTime = CkWallTimer();
    array.run();
  }

  void done(int totalIter) {
    if(totalIter >= maxiterations)
      CkPrintf("Finish due to max iterations %d, total time %.3f seconds. \n", totalIter, CkWallTimer()-startTime); 
    else
      CkPrintf("Finish due to convergence, iterations %d, total time %.3f seconds. \n", totalIter, CkWallTimer()-startTime); 
    CkExit();
  }
};

class Jacobi: public CBase_Jacobi {
  Jacobi_SDAG_CODE

  using array2d = std::vector<std::vector<double>>;
  using array1d = std::vector<double>;

public:

    array2d temperature;
    array2d new_temperature;
    int imsg;
    int iterations;
    int neighbors;
    double max_error;
    int converged;
    bool topBound, bottomBound;
    int istart, ifinish;

    // Constructor, initialize values
    Jacobi()
    : temperature(blockDimX + 2, array1d(arrayDimY, 0.0))
    , new_temperature(blockDimX + 2, array1d(arrayDimY, 0.0))
    , imsg(0)
    , iterations(0)
    , neighbors(0)
    , max_error(0.0)
    , converged(0)
    , topBound(false)
    , bottomBound(false)
    {
      if(thisIndex == 0) {
        topBound = true;
        istart = 2;
      }
      else {
        neighbors++;
        istart = 1;
      }

      if(thisIndex == numChares -1){
        bottomBound = true;
        ifinish = blockDimX;
      }
      else {
        neighbors++;
        ifinish = blockDimX+1;
      }

      constrainBC();
    }

    Jacobi(CkMigrateMessage* m) {}

    void pup(PUP::er &p)
    {
      p|temperature;
      p|new_temperature;
      p|imsg;
      p|iterations;
      p|neighbors;
      p|max_error;
      p|converged;
      p|topBound;
      p|bottomBound;
      p|istart;
      p|ifinish;
    }

    void check_and_compute() {
      double error = 0.0;
      array2d tmp;

      max_error = 0.0;
      for(int i=istart; i<ifinish; i++) {
        for(int j=1; j<arrayDimY-1; j++) {
          // update my value based on the surrounding values
          new_temperature[i][j] = (temperature[i-1][j]+temperature[i+1][j]+temperature[i][j-1]+temperature[i][j+1]+temperature[i][j]) * 0.2;
          error = fabs(new_temperature[i][j] - temperature[i][j]);
          if(error > max_error) {
            max_error = error;
          }
        }
      }

      tmp = temperature;
      temperature = new_temperature;
      new_temperature = tmp;
    }

    // Enforce some boundary conditions
    void constrainBC() {
      int i;
      //top boundary 
      if(topBound) {
        for(i=0;i<arrayDimY; i++) {
          temperature[1][i] = 1.0;
          new_temperature[1][i] = 1.0;
        }
      }
      //left, right boundary
      for(i=0;i<blockDimX+2; i++) 
      {
        temperature[i][0] = 1.0;
        new_temperature[i][0] = 1.0;
        temperature[i][arrayDimY-1] = 1.0;
        new_temperature[i][arrayDimY-1] = 1.0;
      }
      //bottom boundary
      if(bottomBound) {
        for(i=0;i<arrayDimY; i++) {
          temperature[blockDimX][i] = 1.0;
          new_temperature[blockDimX][i] = 1.0;
        }
      }
    } 
    // for debugging
    void dumpMatrix(array2d matrix)
    {
      CkPrintf("\n\n[%d]\n",thisIndex);
      for(int i=0; i<blockDimX+2;++i)
      {
        for(int j=0; j<arrayDimY;++j)
        {
          CkPrintf("%0.6lf ",matrix[i][j]);
        }
        CkPrintf("\n");
      }
    }

};

#include "jacobi1d.def.h"
