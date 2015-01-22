#include "vectorAdd.decl.h"
#include "vectorAdd.h"
#include <time.h>
#define DEBUG

extern void createWorkRequest(int vectorSize, float *A, float *B, float **C, int myIndex, void *cb);

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

  workers.beginWork();
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


void Workers::beginWork() {
  CkCallback *cb;
  CkArrayIndex1D myIndex = CkArrayIndex1D(thisIndex);
  cb = new CkCallback(CkIndex_Workers::complete(), myIndex, thisArrayID);
  createWorkRequest(vectorSize, A, B, &C, thisIndex, (void *) cb);
}

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

  CkPrintf("[%d] C-gold\n", thisIndex);
  for (int j=0; j<vectorSize; j++) {
    C[j] = A[j] + B[j];
    CkPrintf("%.2f ", C[j]);
  }
  CkPrintf("\n");


#endif

  contribute(CkCallback(CkIndex_Main::finishWork(NULL), mainProxy));
}

void randomInit(float* data, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = rand() / (float)RAND_MAX;
  }
}

#include "vectorAdd.def.h"
