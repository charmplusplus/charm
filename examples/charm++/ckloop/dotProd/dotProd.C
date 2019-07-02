/** Authors: Vivek Kale and Karthik Senthil **/
/** Description: Charm++ + CkLoopHybrid code to compute dot product of two vectors whose sizes can be given as input by the user. **/
/** Guide to compiling and running this code:

 To compile the code, you can type:

 $(CHARMC) -language charm++ -o dotProd dotProd.o -module CkLoop


 To run the code on 1 physical node with 4 logical PEs per node, you can type:

./charmrun ./dotProd 1 +p4 ++ppn 4 ++local


 To run the code with a vector size of 800 and static fraction of .7, you can type:

 ./charmrun +p4 dotProd 800 2 0.7 5 ++ppn 2 ++local


 You can also look at the Makefile given to try other experiments.


**/
#include "dotProd.decl.h"
#include <cstdlib>
#include "charm++.h"
#include <math.h>

#include "CkLoopAPI.h"

//#define OMP_HYBRID - uncomment this for runs with OMP_HYBRID

#ifdef OMP_HYBRID
#include <omp.h>
#endif

#ifdef PAPI_PROFILING
#include <papi.h>
#endif

#define MAX_ITERS 1000000 // TODO: The constant MAX_ITERS ought to be a command-line parameter, but leave for now.
#define DEFAULT_SIZE 512 // Size of vectors, i.e., global size of the array.

#define DEFAULT_NUM_ELEMS 4 // Default number of chares per node.
#define DEFAULT_STATIC_FRACTION 0.5 // Size of vectors, i.e., global size of the array.
#define DEFAULT_CHUNK_SIZE 1 // Size of vectors, i.e., global size of the array.
#define DEFAULT_OP_NUM 0
#define DEFAULT_NUM_ITERS 110

#define DEFAULT_NUM_THREADS 4

#define SIZE 512 // Size of vectors, i.e., global size of the array.
int numElemsPerNode; // This is the number of chares per node, i.e., process.
int probSize; //Global problem size.
double staticFraction; // Fraction of loop iterations to be scheduled statically.
int chunkSize; // Number of iterations in a task for static scheduling portion and dynamic scheduling portion of hybrid static/dynamic scheduling.
int opNum;
int numIters;
int numThreads;

class Main : public CBase_Main {
public:
  CProxy_Elem vChunks;
  int iter = DEFAULT_NUM_ITERS;
  int warm_up = 10;
  double t1;
  Main(CkArgMsg* msg) {
    probSize = msg->argc > 1 ?  atoi(msg->argv[1]) : DEFAULT_SIZE;
    numElemsPerNode = msg->argc > 2 ? atoi(msg->argv[2]) : DEFAULT_NUM_ELEMS;
    if (probSize % numElemsPerNode != 0)
      CkAbort("Error! probSize not divisible by number of chares per node.\n");
    // if (numElemsPerNode < CkNumNodes())  // shouldn't need an error check involving numElemsPerNode here.
    staticFraction = msg->argc > 3 ?  atof(msg->argv[3]) : DEFAULT_STATIC_FRACTION;
    chunkSize = msg->argc > 4 ?  atoi(msg->argv[4]) : DEFAULT_CHUNK_SIZE;
    opNum = msg->argc > 5 ?  atoi(msg->argv[5]) : DEFAULT_OP_NUM;
    iter = msg->argc > 6 ?  atoi(msg->argv[6]) : DEFAULT_NUM_ITERS;
    delete msg;
    CkPrintf("Running dot product code with %d iterations.\n", iter);
#ifdef OMP_HYBRID
    omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel
    {
      numThreads = omp_get_num_threads();
      if(numThreads <= 1)
  {
    printf("Number of threads is %d. Resetting a number greater than 1. \n", numThreads);
    omp_set_num_threads(DEFAULT_NUM_THREADS);
  }
      printf("Number of threads running on is: %d .\n", numThreads);
    }
#else
    CkLoop_Init(-1); // Run without shared memory of Charm.
#endif
    CProxy_rank0BlockMap myMap = CProxy_rank0BlockMap::ckNew();
    CkArrayOptions opts(numElemsPerNode*CkNumNodes());
    opts.setMap(myMap);
    vChunks = CProxy_Elem::ckNew(thisProxy, opts);
    int ind2 = CkIndex_Main::initializeStructures();
    CkStartQD(ind2, &thishandle); // Quiescience detection.
  }
  void doTests(){vChunks.doDotProduct();}
  void initializeStructures(){vChunks.doInitVectors(); int ind = CkIndex_Main::doTests(); CkStartQD(ind, &thishandle); } // Quiescience detection.
  void printResult(float result) {
    iter--;
    warm_up--;
    if (warm_up == 0) {
      t1 = CkWallTimer();
      CkPrintf("[main] Started wallclock timer\n");
    }
    if(iter == 0)
      {
  double t2 = CkWallTimer();
  //TODO: add print of number of iterations here.
  CkPrintf("dotP: probSize = %lld \t \t charesPerNode = %d \t staticFraction = %f \t chunkSize=%d \t time = %f \n", probSize, numElemsPerNode, staticFraction, chunkSize, t2-t1);
  CkPrintf("dotP: result = %f \n", result);
/*
  float errorPercent = fabs( 100.0* (result - 6.00*probSize*CkNumNodes())/result);
        if ( errorPercent > 0.1)
          CkPrintf("result %f is wrong. shd be: %f\n", result, (6.0*probSize*CkNumNodes()));
*/
  CkExit();
      }
    else
      vChunks.doDotProduct();
  }
};

extern "C" void initVectors(int start, int end, void * result, int numParams, void* params)
{
  float** z = (float**) params;
  float* v1 = z[0]; float* v2 = z[1];
  for(int i=start; i<end; i++)
    {
      v1[i] = drand48() * 1000.0;
      v2[i] = drand48() * 1000.0;
    }
}

extern "C" void dotP_chunked(int start, int end, void* result, int numParams, void* params)
{
  float** z = (float**) params;
  float* v1 = z[0];  float* v2 = z[1];
  float x = 0.0;
  //CkPrintf("DOTP:DEBUG: Executing chunk %d: %d on PE %d \n", start, end, CkMyPe());
  // Make this easily handle sparse vectors: One way to do this is to use hash tables to represent
  // non-zeros in sparse vectors. A more complicated example will add Sparse Matrix Vector Multiplication
  // with CSR format to handle sparsity of the matrix.
  for(int i=start; i<end; i++)
    x += v1[i]*v2[i];
  * ((double*)result) = x;
  // printf("DOTP:DEBUG: result = %f \t x = %f. \n", *((float*) result), x);
}

class Elem : public CBase_Elem {
public:
  float* a;
  float* b;
  CProxy_Main mainProxy;
  Elem(CProxy_Main _mainProxy) {
    int value = thisIndex;
    // CkPrintf("DEBUG: Elem constructor %d on PE: %d \n", thisIndex, CkMyPe());
    mainProxy = _mainProxy;
    a = new float[probSize/numElemsPerNode]; // Each array on a node is of the size total problem size divided by number of chares per PE.
    b = new float[probSize/numElemsPerNode];
    #ifdef OMP_HYBRID

    #else
    #endif
  }

  void doInitVectors()
  {
    float* params[2]; // Used for parameters to be passed to initVectors.
    params[0] = a;
    params[1] = b;
    int numberOfChunks= (probSize/numElemsPerNode)/chunkSize;
    srand48(time(NULL));
#ifdef OMP_HYBRID
    float r =0.0;
#pragma omp parallel
    {
      printf("OpenMP implementation: Initializing vectors: The number of threads is %d\n", omp_get_num_threads());
#pragma omp for
      for(int i=0; i < (int) (ceil) (probSize/numElemsPerNode); i++)
  {
    a[i] = drand48() * 1000.0;
    b[i] = drand48() * 1000.0;
  }
    }
#else
    CkLoop_ParallelizeHybrid(1.0, initVectors, 2, params, numberOfChunks, 0, (probSize/numElemsPerNode), 1); // Use static scheduling for initializing array.
#endif
  }

  void doDotProduct()
  {
    float r = 0.0;
    int numberOfChunks= (probSize/numElemsPerNode)/chunkSize;
#ifdef OMP_HYBRID
#pragma omp parallel
    {

      if(omp_get_thread_num() == 0)
  printf("OpenMP implementation: Doing computation: The number of threads is %d\n", omp_get_num_threads());
#pragma omp for nowait reduction(+:r)
      for(int i=0; i < (int) (ceil((staticFraction*(probSize/numElemsPerNode)))); i++)
  r = r + a[i]*b[i];
#pragma omp for schedule(dynamic,chunkSize) reduction(+:r) // The clause reduction is a hint to LLVM OpenMP to optimize the code for reduct\ion operation.
      for(int i=(int) ((floor) (staticFraction*(probSize/numElemsPerNode))); i< (int) (probSize/numElemsPerNode); i++)
  r = r + a[i]*b[i];
    }
#else
    float* params[2]; // Used for parameters to be passed to dotP_chunked.
    params[0] = a;
    params[1] = b;
    // CkPrintf("numberOfChunks = %d \n", numberOfChunks);
    CkLoop_ParallelizeHybrid(staticFraction, dotP_chunked, 2, params, numberOfChunks, 0, (probSize/numElemsPerNode), 1, &r, CKLOOP_FLOAT_SUM);
    // CkPrintf("End CkLoop_Parallelize function.\n");
#endif
    CkCallback cb(CkReductionTarget(Main, printResult), mainProxy);
    contribute(sizeof(float), &r, CkReduction::sum_float, cb);
    // CkPrintf("End contribute with r = %f\n.", r);
  }

  Elem(CkMigrateMessage*){}
  float dotP(){
    float x = 0;
    for(int i=0; i<probSize/numElemsPerNode; i++)
      {
  x += a[i]*b[i];
      }
    return x;
  }
};

class rank0BlockMap : public CkArrayMap
{
public:
  rank0BlockMap(void) {}
  rank0BlockMap(CkMigrateMessage *m){}
  int registerArray(CkArrayIndex& numElements,CkArrayID aid) {
    return 0;
  }
  // Assign chares to rank 0 of each process.
  int procNum(int /*arrayHdl*/, const CkArrayIndex &idx) {
    int elem=*(int *)idx.data();
    int charesPerNode = numElemsPerNode;
    int nodeNum = (elem/(charesPerNode));
    int numPEsPerNode = CkNumPes()/CkNumNodes();
    int penum = nodeNum*numPEsPerNode;
    //CkPrintf("DEBUG: procNum: Assigning elem index %d to %d\n", elem, penum);
    return penum;
  }
};
#include "dotProd.def.h"
