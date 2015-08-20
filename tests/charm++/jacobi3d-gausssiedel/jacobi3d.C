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
#include "jacobi3d.h"
#include "TopoManager.h"

#include "CkLoopAPI.h"
CProxy_FuncCkLoop ckLoopProxy;
#ifdef JACOBI_OPENMP
#include <omp.h>
#endif

//#define PRINT_DEBUG 1
#define  LOW_VALUE 0 
#define  HIGH_VALUE 255 
//#define JACOBI  1 

/*readonly*/ int GAUSS_ITER;  
/*readonly*/ double THRESHOLD;
/*readonly*/ int threadNums;

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int arrayDimX;
/*readonly*/ int arrayDimY;
/*readonly*/ int blockDimX;
/*readonly*/ int arrayDimZ;
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

// We want to wrap entries around, and because mod operator % 
// sometimes misbehaves on negative values. -1 maps to the highest value.
#define wrap_x(a)	(((a)+num_chare_x)%num_chare_x)
#define wrap_y(a)	(((a)+num_chare_y)%num_chare_y)
#define wrap_z(a)	(((a)+num_chare_z)%num_chare_z)

#define index(a, b, c)	( (a)*(blockDimY+2)*(blockDimZ+2) + (b)*(blockDimZ+2) + (c) )

#define  START_ITER     10
#define   END_ITER      20
#define PRINT_FREQ     100 
#define CKP_FREQ		100
#define MAX_ITER	 10	
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
    CProxy_TraceControl _traceControl;
    int iterations;
    
    void processCommandlines(int argc, char** argv)
    {
        GAUSS_ITER = 1;
        THRESHOLD = 0.01;
        threadNums = 1;
        for (int i=0; i<argc; i++) {
            if (argv[i][0]=='-') {
                switch (argv[i][1]) {
                case 'X':
                    arrayDimX = atoi(argv[++i]);
                    break;
                case 'Y':
                    arrayDimY = atoi(argv[++i]);
                    break;
                case 'Z':
                    arrayDimZ = atoi(argv[++i]);
                    break;
                case 'x':
                    blockDimX = atoi(argv[++i]);
                    break;
                case 'y': 
                    blockDimY = atoi(argv[++i]);
                    break;
                case 'z': 
                    blockDimZ = atoi(argv[++i]);
                    break;
                case 'g':
                    GAUSS_ITER =  atoi(argv[++i]);
                    break;
                case 't':
                   THRESHOLD = atof(argv[++i]);
                   break;
                case 'r':
                   threadNums = atoi(argv[++i]);
                   break;
                }   
            }       
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
        CkPrintf("Running Jacobi on %d processors with (%d, %d, %d) chares threshold=%f\n", CkNumPes(), num_chare_x, num_chare_y, num_chare_z, THRESHOLD);
        CkPrintf("Array Dimensions: %d %d %d\n", arrayDimX, arrayDimY, arrayDimZ);
        CkPrintf("Block Dimensions: %d %d %d\n", blockDimX, blockDimY, blockDimZ);
    } 
    Main(CkArgMsg* m) {
      iterations = 0;
      mainProxy = thisProxy;
      processCommandlines(m->argc, m->argv);
      delete m;
      // Create new array of worker chares
#if USE_TOPOMAP
      CProxy_JacobiMap map = CProxy_JacobiMap::ckNew(num_chare_x, num_chare_y, num_chare_z);
      CkPrintf("Topology Mapping is being done ... \n");
      CkArrayOptions opts(num_chare_x, num_chare_y, num_chare_z);
      opts.setMap(map);
      array = CProxy_Jacobi::ckNew(opts);
#else
#if CKLOOP
      CProxy_JacobiNodeMap map = CProxy_JacobiNodeMap::ckNew();
      CkArrayOptions opts(num_chare_x, num_chare_y, num_chare_z);
      opts.setMap(map);
      array = CProxy_Jacobi::ckNew(opts);
#else
      array = CProxy_Jacobi::ckNew(num_chare_x, num_chare_y, num_chare_z);
#endif
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

      _traceControl = CProxy_TraceControl::ckNew();
#if CKLOOP
      ckLoopProxy = CkLoop_Init(CkMyNodeSize());
#endif
      start();
    }

    void start() {
      array.doStep();
    }

    void converge(double maxDiff)
    {
        iterations++;
        if(iterations == START_ITER)
            _traceControl.startTrace();
        if(iterations == END_ITER)
            _traceControl.endTrace();

        if(iterations == WARM_ITER)
            startTime = CmiWallTimer();                
        //if(maxDiff <= THRESHOLD || iterations > 20)
        if(maxDiff <= THRESHOLD )
        {
            endTime = CmiWallTimer();
            CkPrintf("Completed:<x,y,z>chares  msgbytes iterations, times, timeperstep(ms)\n benchmark[%d:%d] [ %d %d %d ] \t\t %d \t\t %d \t\t %d\t\t %.3f \t\t %.3f \n", CkNumPes(), CkNumNodes(), num_chare_x, num_chare_y, num_chare_z, num_chare_x *num_chare_y * num_chare_z, sizeof(double) * blockDimX*blockDimY, iterations*GAUSS_ITER, endTime - startTime, (endTime - startTime)/(iterations-WARM_ITER)/GAUSS_ITER*1000);
            //array.print();
            CkExit();
        }else
        {
            if(iterations % PRINT_FREQ== 0)
                CkPrintf("iteration %d  diff=%e\n", iterations, maxDiff); 
			array.doStep();
        }
    }
    // Each worker reports back to here when it completes an iteration
	void report() {
        iterations += CKP_FREQ;
        if (iterations < MAX_ITER) {
#ifdef CMK_MEM_CHECKPOINT
            CkCallback cb (CkIndex_Jacobi::doStep(), array);
            CkStartMemCheckpoint(cb);
#else
			array.doStep();
#endif
        } else {
            CkPrintf("Completed %d iterations\n", MAX_ITER-1);
            endTime = CmiWallTimer();
            CkPrintf("Time elapsed per iteration: %f\n", (endTime - startTime)/(MAX_ITER-1-WARM_ITER));
            CkExit();
        }
            CkExit();
    }
};

/** \class Jacobi
 *
 */

extern "C" void doCalc(int first,int last, void *result, int paramNum, void * param)
{
#if JACOBI
    int i;
    double maxDiff=0;
    double tmp, update=0;
    maxDiff = 0;

    Jacobi  *obj = (Jacobi*)param;
    int x_c = arrayDimX/2/blockDimX;
    int y_c = arrayDimY/2/blockDimY;
    int z_c = arrayDimZ/2/blockDimZ;
    int x_p = (arrayDimX/2)%blockDimX+1;
    int y_p = (arrayDimY/2)%blockDimY+1;
    int z_p = (arrayDimZ/2)%blockDimZ+1;
    //CkPrintf(" ckloop on %d  first last %d %d \n", CkMyPe(), first, last);  
    for(i=first; i<=last; ++i) {
          for(int j=1; j<blockDimY+1; ++j) {
              for(int k=1; k<blockDimZ+1; ++k) {
                  if(obj->thisIndex_x == x_c && y_c == obj->thisIndex_y && z_c == obj->thisIndex_z && i==x_p && j==y_p && k==z_p)
                      continue;
                  update = 0;//temperature[index(i, j, k)];
                  update += obj->temperature[index(i+1, j, k)];
                  update += obj->temperature[index(i-1, j, k)];
                  update += obj->temperature[index(i, j+1, k)];
                  update += obj->temperature[index(i, j-1, k)];
                  update += obj->temperature[index(i, j, k+1)];
                  update += obj->temperature[index(i, j, k-1)];
                  update /= 6;
                 
                  double diff = obj->temperature[index(i, j, k)] - update;
                  if(diff<0) diff = -1*diff;
                  if(diff>maxDiff) maxDiff = diff;
                  obj->new_temperature[index(i, j, k)] = update;
              }
          }
        }
        *((double*)result) = maxDiff;
#else
#endif 
}
extern "C" void doUpdate(int first,int last, void *result, int paramNum, void *param){
#if JACOBI
        int i;
        Jacobi  *obj = (Jacobi*)param;
        int x_c = arrayDimX/2/blockDimX;
        int y_c = arrayDimY/2/blockDimY;
        int z_c = arrayDimZ/2/blockDimZ;
        int x_p = (arrayDimX/2)%blockDimX+1;
        int y_p = (arrayDimY/2)%blockDimY+1;
        int z_p = (arrayDimZ/2)%blockDimZ+1;
        //CkPrintf("\n {%d:%d:%d  %d:%d:%d  %d:%d:%d }", obj->thisIndex_x, obj->thisIndex_y, obj->thisIndex_z, x_c, y_c, z_c, x_p, y_p, z_p);
        for(i=first; i<=last; ++i) {
            for(int j=1; j<blockDimY+1; ++j) {
                for(int k=1; k<blockDimZ+1; ++k) {
                    if(!(obj->thisIndex_x == x_c && y_c == obj->thisIndex_y && z_c == obj->thisIndex_z && i==x_p && j==y_p && k==z_p))
                        obj->temperature[index(i, j, k)] = obj->new_temperature[index(i, j, k)];
                    //CkPrintf("[ %d:%d:%d  .%2f ]", i, j, k, obj->temperature[index(i, j, k)]); 
                }
            }
        }
       //CkPrintf("\n\n");
#else
#endif
}


Jacobi::Jacobi() {
      // This call is an anachronism - the need to call __sdag_init() has been
      // removed. We still call it here to test backward compatibility.
      __sdag_init();

      int i, j, k;
      // allocate a three dimensional array
      temperature = new double[(blockDimX+2) * (blockDimY+2) * (blockDimZ+2)];
#if  JACOBI
      new_temperature = new double[(blockDimX+2) * (blockDimY+2) * (blockDimZ+2)];
#endif
      for(i=0; i<blockDimX+2; ++i) {
          for(j=0; j<blockDimY+2; ++j) {
              for(k=0; k<blockDimZ+2; ++k) {
                  temperature[index(i, j, k)] = 1.0;
#if  JACOBI
                  new_temperature[index(i, j, k)] = 1.0;
#endif
              }
          } 
      }
      thisIndex_x = thisIndex.x;
      thisIndex_y = thisIndex.y;
      thisIndex_z = thisIndex.z;
      iterations = 0;
      imsg = 0;
      constrainBC();

      usesAtSync = true;
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
       // CkPrintf("neighbor = %d \n", neighbors);
    }
void Jacobi::pup(PUP::er &p){
		// pupping properties of this class
		p | iterations;
		p | imsg;
		// if unpacking, allocate the memory space
		if(p.isUnpacking()){
			temperature = new double[(blockDimX+2) * (blockDimY+2) * (blockDimZ+2)];
		}
		// pupping the arrays
		p((char *)temperature, (blockDimX+2) * (blockDimY+2) * (blockDimZ+2) * sizeof(double));
	}
    // Send ghost faces to the six neighbors
    void Jacobi::begin_iteration(void) {
      iterations++;

      // Copy different faces into messages
       ghostMsg *leftMsg ;
       ghostMsg *rightMsg; 
       ghostMsg *topMsg ;
       ghostMsg *bottomMsg;
       ghostMsg *frontMsg ;
       ghostMsg *backMsg ;
     
       
      if(thisIndex.x-1 >= 0)
      {
          leftMsg = new (blockDimY*blockDimZ) ghostMsg(RIGHT, blockDimY, blockDimZ);
          for(int j=0; j<blockDimY; ++j) 
              for(int k=0; k<blockDimZ; ++k) {
                  leftMsg->gh[k*blockDimY+j] = temperature[index(1, j+1, k+1)];
              }
          CkSetRefNum(leftMsg, iterations);
          thisProxy(wrap_x(thisIndex.x-1), thisIndex.y, thisIndex.z).receiveGhosts(leftMsg);
      }

      if(thisIndex.x+1 <num_chare_x) 
      {
          rightMsg = new (blockDimY*blockDimZ) ghostMsg(LEFT, blockDimY, blockDimZ);
          for(int j=0; j<blockDimY; ++j) 
              for(int k=0; k<blockDimZ; ++k) {
                  rightMsg->gh[k*blockDimY+j] = temperature[index(blockDimX, j+1, k+1)];}
          CkSetRefNum(rightMsg, iterations);
          thisProxy(wrap_x(thisIndex.x+1), thisIndex.y, thisIndex.z).receiveGhosts(rightMsg);
      }

      if(thisIndex.y-1 >= 0)
      {
          topMsg = new (blockDimX*blockDimZ) ghostMsg(BOTTOM, blockDimX, blockDimZ);
          for(int i=0; i<blockDimX; ++i) 
              for(int k=0; k<blockDimZ; ++k) {
                  topMsg->gh[k*blockDimX+i] = temperature[index(i+1, 1, k+1)];
              }

          CkSetRefNum(topMsg, iterations);
          thisProxy(thisIndex.x, wrap_y(thisIndex.y-1), thisIndex.z).receiveGhosts(topMsg);
      }

      if(thisIndex.y+1 <num_chare_y)
      {
          bottomMsg = new (blockDimX*blockDimZ) ghostMsg(TOP, blockDimX, blockDimZ);
          for(int i=0; i<blockDimX; ++i) 
          for(int k=0; k<blockDimZ; ++k) {
              bottomMsg->gh[k*blockDimX+i] = temperature[index(i+1, blockDimY, k+1)];
          }
          CkSetRefNum(bottomMsg, iterations);
          thisProxy(thisIndex.x, wrap_y(thisIndex.y+1), thisIndex.z).receiveGhosts(bottomMsg);
      }

      if(thisIndex.z-1 >= 0)
      {
          frontMsg = new (blockDimX*blockDimY) ghostMsg(BACK, blockDimX, blockDimY);
          for(int i=0; i<blockDimX; ++i) 
              for(int j=0; j<blockDimY; ++j) {
                  frontMsg->gh[j*blockDimX+i] = temperature[index(i+1, j+1, 1)];
              }
          CkSetRefNum(frontMsg, iterations);
          thisProxy(thisIndex.x, thisIndex.y, wrap_z(thisIndex.z-1)).receiveGhosts(frontMsg);
      }
      
      if(thisIndex.z+1 <num_chare_z)
      {
          backMsg = new (blockDimX*blockDimY) ghostMsg(FRONT, blockDimX, blockDimY);
          for(int i=0; i<blockDimX; ++i) 
              for(int j=0; j<blockDimY; ++j) {
                  backMsg->gh[j*blockDimX+i] = temperature[index(i+1, j+1, blockDimZ)];
              }
          CkSetRefNum(backMsg, iterations);
          thisProxy(thisIndex.x, thisIndex.y, wrap_z(thisIndex.z+1)).receiveGhosts(backMsg);
      }
    }

    void Jacobi::processGhosts(ghostMsg *gmsg) {
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
                  temperature[index(i+1, j+1,0)] = gmsg->gh[j*height+i];
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


	void Jacobi::check_and_compute() {
        double maxDiff;
#if CKLOOP
        CkLoop_Parallelize(doCalc, 1,  (void*)this, 4*CkMyNodeSize(), 1, blockDimX, 1, &maxDiff, CKLOOP_DOUBLE_MAX);
        CkLoop_Parallelize(doUpdate, 1,  (void*)this, CkMyNodeSize(), 1, blockDimX, 1);
#else
		maxDiff = compute_kernel();
#endif
		// calculate error
		// not being done right now since we are doing a fixed no. of iterations
        //CkPrintf(" iteration %d [%d,%d,%d] max=%e\n", iterations, thisIndex.x, thisIndex.y, thisIndex.z, maxDiff);
		//constrainBC();
		if (iterations % CKP_FREQ == 0 || iterations > MAX_ITER){
#ifdef CMK_MEM_CHECKPOINT
			contribute(0, 0, CkReduction::concat, CkCallback(CkIndex_Main::report(), mainProxy));
#elif CMK_MESSAGE_LOGGING
			if(iterations > MAX_ITER)
				contribute(0, 0, CkReduction::concat, CkCallback(CkIndex_Main::report(), mainProxy));
			else
				AtSync();
#else
            CkCallback cb(CkReductionTarget(Main, converge), mainProxy);
            contribute(sizeof(double), &maxDiff, CkReduction::max_double, cb);
#endif
        } else {
            CkCallback cb(CkReductionTarget(Main, converge), mainProxy);
            contribute(sizeof(double), &maxDiff, CkReduction::max_double, cb);
			//doStep();
        }
    }

	void Jacobi::ResumeFromSync(){
		doStep();
	}


#if JACOBI
    double Jacobi::compute_kernel() {     //Gauss-Siedal compute
        int i;

        int x_c = arrayDimX/2/blockDimX;
        int y_c = arrayDimY/2/blockDimY;
        int z_c = arrayDimZ/2/blockDimZ;
        int x_p = (arrayDimX/2)%blockDimX+1;
        int y_p = (arrayDimY/2)%blockDimY+1;
        int z_p = (arrayDimZ/2)%blockDimZ+1;
#ifdef JACOBI_OPENMP
        double *maxDiffSub = new double[threadNums];
        for(i=0; i<threadNums; i++)
            maxDiffSub[i] = 0;
#endif
        double maxDiff=0;
  //#pragma omp parallel for shared(temperature, new_temperature, maxDiff) 
  #pragma omp parallel  
        {  
        double t1 = CkWallTimer();
        #pragma omp for 
        for(i=1; i<blockDimX+1; ++i) {
#ifdef JACOBI_OPENMP
            int tid = omp_get_thread_num();
          //printf("[%d] did  %d iteration out of %d \n", omp_get_thread_num(), i, blockDimX+1); 
#endif
          for(int j=1; j<blockDimY+1; ++j) {
              for(int k=1; k<blockDimZ+1; ++k) {
        
                  if(thisIndex.x == x_c && y_c == thisIndex.y && z_c == thisIndex.z && i==x_p && j==y_p && k==z_p)
                      continue;
                  double update = 0;//temperature[index(i, j, k)];
                  update += temperature[index(i+1, j, k)];
                  update += temperature[index(i-1, j, k)];
                  update += temperature[index(i, j+1, k)];
                  update += temperature[index(i, j-1, k)];
                  update += temperature[index(i, j, k+1)];
                  update += temperature[index(i, j, k-1)];
                  update /= 6;
                 
                  double diff = temperature[index(i, j, k)] - update;
                  if(diff<0) diff = -1*diff;
#ifdef JACOBI_OPENMP
                  if(diff>maxDiffSub[tid]) maxDiffSub[tid] = diff;
#else
                  if(diff>maxDiff) maxDiff = diff;
#endif
                  new_temperature[index(i, j, k)] = update;
              }
          }
        }
        //printf(" timecost [%d out of %d ]  %f\n", omp_get_thread_num(), omp_get_num_threads(), (CkWallTimer()-t1)*1e6);
      }
#ifdef JACOBI_OPENMP
        maxDiff= maxDiffSub[0];
        for(i=1; i<threadNums; i++)
            if(maxDiff < maxDiffSub[i]) maxDiff = maxDiffSub[i];
#endif
  #pragma omp parallel  
        {  
        #pragma omp for 
        for(i=1; i<blockDimX+1; ++i) {
            for(int j=1; j<blockDimY+1; ++j) {
                for(int k=1; k<blockDimZ+1; ++k) {
                  if(thisIndex.x == x_c && y_c == thisIndex.y && z_c == thisIndex.z && i==x_p && j==y_p && k==z_p)
                      continue;
                    temperature[index(i, j, k)] = new_temperature[index(i, j, k)];
                }
            }
        }
      }
#if     PRINT_DEBUG
        CkPrintf("\n {%d:%d:%d  %d:%d:%d  %d:%d:%d %f}", thisIndex_x, thisIndex_y, thisIndex_z, x_c, y_c, z_c, x_p, y_p, z_p, maxDiff);
        for(i=1; i<blockDimX+1; ++i) {
          for(int j=1; j<blockDimY+1; ++j) {
              for(int k=1; k<blockDimZ+1; ++k) {
                  CkPrintf("([%d:%d:%d]: %.2f )", i, j, k, temperature[index(i, j, k)]); 
              }
          }
        }
        CkPrintf("\n\n");
#endif
        return maxDiff;
    }

#else

    // Check to see if we have received all neighbor values yet
    // If all neighbor values have been received, we update our values and proceed
    double Jacobi::compute_kernel() {     //Gauss-Siedal compute
        int i=1;
        double maxDiff=0;
        double tmp, update=0;
        int count=1;
        int gg;

        int x_c = arrayDimX/2/blockDimX;
        int y_c = arrayDimY/2/blockDimY;
        int z_c = arrayDimZ/2/blockDimZ;
        int x_p = (arrayDimX/2)%blockDimX+1;
        int y_p = (arrayDimY/2)%blockDimY+1;
        int z_p = (arrayDimZ/2)%blockDimZ+1;
            //CkPrintf("\n\n START %d [%d:%d:%d] \n", iterations, thisIndex.x, thisIndex.y, thisIndex.z);
  #pragma omp parallel for schedule(dynamic) 
        for(gg=0; gg<GAUSS_ITER; gg++){
            maxDiff = 0;
            for(i=1; i<blockDimX+1; ++i) {
          //printf("[%d] did  %d iteration out of %d \n", omp_get_thread_num(), i, blockDimX+1); 
          for(int j=1; j<blockDimY+1; ++j) {
              for(int k=1; k<blockDimZ+1; ++k) {
                  if(thisIndex.x == x_c && y_c == thisIndex.y && z_c == thisIndex.z && i==x_p && j==y_p && k==z_p)
                      continue;
                  update = 0;//temperature[index(i, j, k)];
                  update += temperature[index(i+1, j, k)];
                  update += temperature[index(i-1, j, k)];
                  update += temperature[index(i, j+1, k)];
                  update += temperature[index(i, j-1, k)];
                  update += temperature[index(i, j, k+1)];
                  update += temperature[index(i, j, k-1)];
                  update /= 6;
                 
                  double diff = temperature[index(i, j, k)] - update;
                  if(diff<0) diff = -1*diff;
                  if(diff>maxDiff) maxDiff = diff;
                  temperature[index(i, j, k)] = update;
              }
          }
        }
        }
#if     PRINT_DEBUG
        CkPrintf("[%d:%d:%d] iteration=%d max:%.3f ==== ", thisIndex.x, thisIndex.y, thisIndex.z, iterations, maxDiff); 
        for(i=0; i<blockDimX+2; ++i) {
          for(int j=0; j<blockDimY+2; ++j) {
              for(int k=0; k<blockDimZ+2; ++k) {
                  CkPrintf("([%d:%d:%d]: %.2f )", i, j, k, temperature[index(i, j, k)]); 
              }
          }
        }
        CkPrintf("\n\n");
#endif
        return maxDiff;
    }
#endif
    // Enforce some boundary conditions
    void Jacobi::constrainBC() {
      
        // Heat right, left
        if(thisIndex.x == 0 || thisIndex.x == num_chare_x-1)
            for(int j=0; j<blockDimY+2; ++j)
                for(int k=0; k<blockDimZ+2; ++k)
                {   
                    if(thisIndex.x == 0)
                        temperature[index(0, j, k)] = LOW_VALUE;
                    else
                        temperature[index(blockDimX+1, j, k)] = LOW_VALUE;
                }
        if(thisIndex.y == 0 || thisIndex.y == num_chare_y-1)
            for(int j=0; j<blockDimX+2; ++j)
                for(int k=0; k<blockDimZ+2; ++k)
                {   
                    if(thisIndex.y == 0)
                        temperature[index(j,0, k)] = LOW_VALUE;
                    else
                        temperature[index(j, blockDimY+1, k)] = LOW_VALUE;
                }
        if(thisIndex.z == 0 || thisIndex.z == num_chare_z-1)
            for(int j=0; j<blockDimX+2; ++j)
                for(int k=0; k<blockDimY+2; ++k)
                {   
                    if(thisIndex.z == 0)
                        temperature[index(j, k, 0)] = LOW_VALUE;
                    else
                        temperature[index(j, k, blockDimZ+1)] = LOW_VALUE;
                }
        int x_c = arrayDimX/2/blockDimX;
        int y_c = arrayDimY/2/blockDimY;
        int z_c = arrayDimZ/2/blockDimZ;
        int x_p = (arrayDimX/2)%blockDimX+1;
        int y_p = (arrayDimY/2)%blockDimY+1;
        int z_p = (arrayDimZ/2)%blockDimZ+1;
        if(thisIndex.x == x_c && y_c == thisIndex.y && z_c == thisIndex.z)
        {
            temperature[index(x_p, y_p, z_p)] = HIGH_VALUE;
        }

    }

    void Jacobi::print()
    {
        CkPrintf(" print %d:%d:%d ", thisIndex.x, thisIndex.y, thisIndex.z); 
          //printf("[%d] did  %d iteration out of %d \n", omp_get_thread_num(), i, blockDimX+1); 
          for(int i=1; i<blockDimX+1;++i){CkPrintf("==\n");
              for(int j=1;j<blockDimY+1;++j)
              for(int k=1; k<blockDimZ+1; ++k) {
                  CkPrintf(" %e ",  temperature[index(i, j, k)]) ;
              }
              CkPrintf("------\n-----\n");
          }
              contribute(0, 0, CkReduction::concat, CkCallback(CkIndex_Main::report(), mainProxy));
    }
/** \class JacobiMap
 *
 */

class JacobiNodeMap : public CkArrayMap {

public:
    JacobiNodeMap() {}
    ~JacobiNodeMap() { 
    }

    int procNum(int, const CkArrayIndex &idx) {
      int *index = (int *)idx.data();
      return (CkMyNodeSize() * (index[0]*num_chare_x*num_chare_y + index[1]*num_chare_y + index[2]))%CkNumPes(); 
    }

};
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

class OmpInitializer : public CBase_OmpInitializer{
public:
	OmpInitializer(int numThreads) {
#ifdef JACOBI_OPENMP
          if (numThreads < 1) {
            numThreads = 1; 
          }
          //omp_set_num_threads(numThreads);
          if(CkMyPe() == 0)
          {
#pragma omp parallel
              { if(omp_get_thread_num() == 0 )
                  CkPrintf("Computation loop will be parallelized"
               " with %d OpenMP threads\n", omp_get_num_threads());
              }
          }
#endif
          mainProxy.start();
	}
};

class TraceControl : public Group 
{
public:
    TraceControl() { omp_set_num_threads(threadNums);}

    void startTrace() { traceBegin(); }
    
    void endTrace() { traceEnd(); }
};

#include "jacobi3d.def.h"
