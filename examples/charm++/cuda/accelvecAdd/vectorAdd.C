#include "vectorAdd.decl.h"
#include "vectorAdd.h"
#include <time.h>
#define DEBUG
#undef DEBUG


CProxy_Main mainProxy;
int vectorSize;
Main::Main(CkArgMsg *m) {
  mainProxy = thisProxy;

  if (m->argc < 3) {
    numChares = 2;
    vectorSize = 8;
  }
  else{
    numChares = atoi(m->argv[1]);
    vectorSize = atoi(m->argv[2]);
  }
  delete m;
  workers = CProxy_Workers::ckNew(numChares);

  startTime = CkWallTimer();

  workers.beginWork(vectorSize);
}

void Main::finishWork(CkReductionMsg *m) {
  delete m;
  CkPrintf("Elapsed time: %f s\n", CkWallTimer() - startTime);
  CkExit();
}

Workers::Workers() {
  A = new float[vectorSize];
  B = new float[vectorSize];
  C = new float[vectorSize];
  srand(time(NULL));
  randomInit(A, vectorSize);
  randomInit(B, vectorSize);
}

Workers::~Workers() {
  delete [] A;
  delete [] B;
  delete [] C;
}

Workers::Workers(CkMigrateMessage *msg) { }

void Workers::complete() {
#ifdef DEBUG
  CkPrintf("[%d] A\n", thisIndex);
  for (int i=0; i<vectorSize; i++) {
    CkPrintf("%.2f ", A[i]);
  }
  CkPrintf("\n");

  CkPrintf("[%d] B\n", thisIndex);
  for (int i=0; i<vectorSize; i++) {
    CkPrintf("%.2f ", B[i]);
  }
  CkPrintf("\n");

  CkPrintf("[%d] C\n", thisIndex);
  for (int i=0; i<vectorSize; i++) {
    CkPrintf("%.2f ", C[i]);
    }
  CkPrintf("\n");
#endif
  double *c_temp= new double[vectorSize];
/*  CkPrintf("[%d] C-gold\n", thisIndex);
*/

  for (int j=0; j<vectorSize; j++) {
    c_temp[j] = A[j] + B[j];
  }
  for (int j=0; j<vectorSize; j++) {
    if(c_temp[j]!=C[j])
      CkPrintf("Error at index %d\n",j);
  }


  contribute(CkCallback(CkIndex_Main::finishWork(NULL), mainProxy));
}

void randomInit(float* data, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = rand() / (float)RAND_MAX;
  }
}

#include "vectorAdd.def.h"
