#include "overlapTest.decl.h"
#include "overlapTest.h"

#define DEBUG

extern void cudaMatMul(int matrixSize, ElementType *A, ElementType *B, ElementType *C); 
CProxy_Main mainProxy; 
int matrixSize;

Main::Main(CkArgMsg *m) {
  mainProxy = thisProxy; 

  if (m->argc >= 2) {
    numChares = atoi(m->argv[1]); 
  }
  if (m->argc == 3) {
    matrixSize = atoi(m->argv[2]); 
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
  int size = matrixSize * matrixSize; 
  A = new ElementType[size];
  B = new ElementType[size];
  C = new ElementType[size]; 
  
  randomInit(A, size); 
  randomInit(B, size); 
}

Workers::~Workers() {
  delete [] A; 
  delete [] B; 
  delete [] C; 
}

Workers::Workers(CkMigrateMessage *msg) { } 

void Workers::beginWork() {
  cudaMatMul(matrixSize, A, B, C);  
#ifdef DEBUG
  CkPrintf("[%d] A\n", thisIndex); 
  for (int i=0; i<matrixSize; i++) {
    CkPrintf("[%d] ", thisIndex);
    for (int j=0; j<matrixSize; j++) {
      CkPrintf("%.2f ", A[i*matrixSize+j]); 
    }
    CkPrintf("\n");
  }
  CkPrintf("[%d] B\n", thisIndex); 
  for (int i=0; i<matrixSize; i++) {
    CkPrintf("[%d] ", thisIndex);
    for (int j=0; j<matrixSize; j++) {
      CkPrintf("%.2f ", B[i*matrixSize+j]); 
    }
    CkPrintf("\n");
  }
  CkPrintf("[%d] C\n", thisIndex); 
  for (int i=0; i<matrixSize; i++) {
    CkPrintf("[%d] ", thisIndex);
    for (int j=0; j<matrixSize; j++) {
      CkPrintf("%.2f ", C[i*matrixSize+j]); 
    }
    CkPrintf("\n");
  }
  CkPrintf("[%d] C-gold\n", thisIndex);
  for (int i=0; i<matrixSize; i++) {
    CkPrintf("[%d] ", thisIndex);
    for (int j=0; j<matrixSize; j++) {
      C[i*matrixSize + j] = 0; 
      for (int k=0; k<matrixSize; k++) {
	C[i*matrixSize + j] += A[i*matrixSize +k] * B[k * matrixSize + j];
      }
      CkPrintf("%.2f ", C[i*matrixSize+j]); 
    }
    CkPrintf("\n");
  }

#endif

  contribute(CkCallback(CkIndex_Main::finishWork(NULL), mainProxy));
}

void randomInit(ElementType* data, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = rand() / (ElementType)RAND_MAX;
  }
}

#include "overlapTest.def.h"
