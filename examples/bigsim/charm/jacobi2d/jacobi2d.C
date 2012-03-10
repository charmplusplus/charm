/** \file jacobi2d.C
 *  Author: Abhinav S Bhatele
 *  Date Created: March 09th, 2009
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

/*readonly*/ int globalBarrier;
/*readonly*/ int localBarrier;

// We want to wrap entries around, and because mod operator % 
// sometimes misbehaves on negative values. -1 maps to the highest value.
#define wrap_x(a)  (((a)+num_chare_x)%num_chare_x)
#define wrap_y(a)  (((a)+num_chare_y)%num_chare_y)

#define MAX_ITER	26
#define WARM_ITER	5
#define LEFT		1
#define RIGHT		2
#define TOP		3
#define BOTTOM		4

double startTime;
double endTime;

class Main : public CBase_Main
{
  public:
    CProxy_Jacobi array;
    int iterations;

    Main(CkArgMsg* m) {
      if ( (m->argc < 4) || (m->argc > 7) ) {
        CkPrintf("%s [array_size] [block_size] +[no]barrier [+[no]local]\n", m->argv[0]);
        CkPrintf("%s [array_size_X] [array_size_Y] [block_size_X] [block_size_Y] +[no]barrier [+[no]local]\n", m->argv[0]);
        CkAbort("Abort");
      }

      if(m->argc < 6) {
        arrayDimY = arrayDimX = atoi(m->argv[1]);
        blockDimY = blockDimX = atoi(m->argv[2]);
	if(strcasecmp(m->argv[3], "+nobarrier") == 0) {
	  globalBarrier = 0;
	  if(strcasecmp(m->argv[4], "+nolocal") == 0)
	    localBarrier = 0;
	  else
	    localBarrier = 1;
	}
	else
	  globalBarrier = 1;
	if(globalBarrier == 0) CkPrintf("\nSTENCIL COMPUTATION WITH NO BARRIERS\n");
      }
      else {
        arrayDimX = atoi(m->argv[1]);
        arrayDimY = atoi(m->argv[2]);
        blockDimX = atoi(m->argv[3]);
        blockDimY = atoi(m->argv[4]);
	if(strcasecmp(m->argv[5], "+nobarrier") == 0) {
	  globalBarrier = 0;
	  if(strcasecmp(m->argv[6], "+nolocal") == 0)
	    localBarrier = 0;
	  else
	    localBarrier = 1;
	}
	else
	  globalBarrier = 1;
	if(globalBarrier == 0 && localBarrier == 0)
	  CkPrintf("\nSTENCIL COMPUTATION WITH NO BARRIERS\n");
	else
	  CkPrintf("\nSTENCIL COMPUTATION WITH LOCAL BARRIERS\n");
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
      CProxy_JacobiMap map = CProxy_JacobiMap::ckNew(num_chare_x, num_chare_y);
      CkPrintf("Topology Mapping is being done ... \n");
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
      if(iterations == WARM_ITER)
	startTime = CmiWallTimer();
      double error = *((double *)msg->getData());

      if((globalBarrier == 1 && iterations < MAX_ITER) || (globalBarrier == 0 && iterations <= WARM_ITER)) {
	if(iterations > WARM_ITER) CkPrintf("Start of iteration %d\n", iterations);
	BgPrintf("BgPrint> Start of iteration at %f\n");
	array.begin_iteration();
      } else {
	CkPrintf("Completed %d iterations\n", MAX_ITER-1);
        endTime = CmiWallTimer();
        CkPrintf("Time elapsed per iteration: %f\n", (endTime - startTime)/(MAX_ITER-1-WARM_ITER));
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
    int readyToSend;

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
	temperature[i] = new double[blockDimY+2];
	new_temperature[i] = new double[blockDimY+2];
      }
      for(i=0;i<blockDimX+2; i++) {
	for(j=0;j<blockDimY+2; j++) {
	  temperature[i][j] = 0.5;
	  new_temperature[i][j] = 0.5;
	}
      }

      arrived_left = 0;
      arrived_right = 0;
      arrived_top = 0;
      arrived_bottom = 0;
      readyToSend = 5;
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
      if(localBarrier == 1 && iterations > WARM_ITER) {
	_TRACE_BG_TLINE_END(&ackLogs[readyToSend]);
	readyToSend++;
      }

      if(readyToSend == 5) {
	if(thisIndex.x == 0 && thisIndex.y == 0  && (globalBarrier == 0 && iterations > WARM_ITER)) {
	  CkPrintf("Start of iteration %d\n", iterations);
	  BgPrintf("BgPrint> Start of iteration at %f\n");
	}

	if(localBarrier == 1 && iterations > WARM_ITER) {
	  void *curLog = NULL;
	  _TRACE_BG_END_EXECUTE(1);
	  _TRACE_BG_BEGIN_EXECUTE_NOMSG("start next iteration", &curLog);
	  for(int i=0; i<5; i++)
	    _TRACE_BG_ADD_BACKWARD_DEP(ackLogs[i]);
	}
	if(localBarrier == 1 && iterations >= WARM_ITER)  readyToSend = 0;
	iterations++;

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
    }

    void receiveGhosts(int dir, int size, double gh[]) {
      int i, j;
      _TRACE_BG_TLINE_END(&sendLogs[dir-1]);

      switch(dir) {
	case LEFT:
	  arrived_left++;
	  for(i=0; i<size; i++)
            temperature[i+1][0] = gh[i];
	  break;
	case RIGHT:
	  arrived_right++;
	  for(i=0; i<size; i++)
            temperature[i+1][blockDimY+1] = gh[i];
	  break;
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
      void *curLog = NULL;

      if (arrived_left >=1 && arrived_right >=1 && arrived_top >=1 && arrived_bottom >= 1) {
	arrived_left--;
	arrived_right--;
	arrived_top--;
	arrived_bottom--;

	_TRACE_BG_END_EXECUTE(1);
	_TRACE_BG_BEGIN_EXECUTE_NOMSG("start computation", &curLog);
	for(int i=0; i<4; i++)
	  _TRACE_BG_ADD_BACKWARD_DEP(sendLogs[i]);

	if(localBarrier == 1 && iterations > WARM_ITER) {
	  int x = thisIndex.x;
	  int y = thisIndex.y;
	  thisProxy(x, wrap_y(y-1)).begin_iteration();
	  thisProxy(x, wrap_y(y+1)).begin_iteration();
	  thisProxy(wrap_x(x-1), y).begin_iteration();
	  thisProxy(wrap_x(x+1), y).begin_iteration();
	}

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

	if(globalBarrier == 1 || (globalBarrier==0 && (iterations <= WARM_ITER || iterations >= MAX_ITER))) {
	  contribute(sizeof(double), &max_error, CkReduction::max_double,
	      CkCallback(CkIndex_Main::report(NULL), mainProxy));
	} else {
	  begin_iteration();
	}
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
      int dimNX = tmgr.getDimNX();
      int dimNY = tmgr.getDimNY();
      int dimNZ = tmgr.getDimNZ();
      int dimNT = tmgr.getDimNT();

      int numCharesPerPeX = x / dimNX;
      int numCharesPerPeY = numCharesPerPeX / dimNY;
      int numCharesPerPeZ = y / dimNZ;
      int numCharesPerPeT = numCharesPerPeZ / dimNT;

      if(CkMyPe()==0) CkPrintf("%d %d %d %d : %d %d %d %d\n", dimNX, dimNY, dimNZ, dimNT, numCharesPerPeX, numCharesPerPeY, numCharesPerPeZ, numCharesPerPeT);

      for(int i=0; i<dimNX; i++)
	for(int j=0; j<dimNY; j++)
	  for(int k=0; k<dimNZ; k++)
	    for(int l=0; l<dimNT; l++)
	      for(int ci = i*numCharesPerPeX+j*numCharesPerPeY; ci < i*numCharesPerPeX+(j+1)*numCharesPerPeY; ci++)
		for(int cj = k*numCharesPerPeZ+l*numCharesPerPeT; cj < k*numCharesPerPeZ+(l+1)*numCharesPerPeT; cj++)
		  mapping[ci][cj] = tmgr.coordinatesToRank(i, j, k, l);

      /*if(CkMyPe() == 0) {
	for(int ci=0; ci<x; ci++) {
	  for(int cj=0; cj<y; cj++) {
	    CkPrintf("%d ", mapping[ci][cj]);
	  }
	  CkPrintf("\n");
	}
      }*/

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
