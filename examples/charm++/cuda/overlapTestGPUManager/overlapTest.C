#include "overlapTest.decl.h"
#include "overlapTest.h"

#define DEBUG

extern void cudaMatMul(int matrixSize, ElementType *A, ElementType *B, ElementType *C, int myIndex, void *cb,int useCublas);
extern void hostMemorySetup(int matrixSize, ElementType **h_A, ElementType **h_B, ElementType **h_C, void *cb); 
extern void hostMemoryCleanup(ElementType *h_A, ElementType *h_B, ElementType *h_C);

CProxy_Main mainProxy; 
int matrixSize;
int useCublas=0;
Main::Main(CkArgMsg *m) {
  mainProxy = thisProxy;

  numChares = atoi(m->argv[1]);
  matrixSize = atoi(m->argv[2]);

  if (m->argc == 4) {
    useCublas = atoi(m->argv[3]);
  }
  delete m;
  workers = CProxy_Workers::ckNew(numChares); 

  startTime = CkWallTimer(); 
    
  workers.setupBuffers(); 
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
  hostMemoryCleanup(h_A, h_B, h_C);
}

Workers::Workers(CkMigrateMessage *msg) { } 

void Workers::setupBuffers() {
  CkArrayIndex1D myIndex = CkArrayIndex1D(thisIndex);
  CkCallback *cb = new CkCallback(CkIndex_Workers::beginWork(), myIndex, thisArrayID);
  hostMemorySetup(matrixSize, &h_A, &h_B, &h_C, (void *) cb); 
}

void Workers::beginWork() {
  CkCallback *cb;
  CkArrayIndex1D myIndex = CkArrayIndex1D(thisIndex);
  cb = new CkCallback(CkIndex_Workers::complete(), myIndex, thisArrayID); 
  int size = matrixSize * matrixSize * sizeof(ElementType);
  memcpy(h_A, A, size);
  memcpy(h_B, B, size); 
  cudaMatMul(matrixSize, h_A, h_B, h_C, thisIndex, (void *) cb,useCublas);
}

void Workers::complete() {
  int size = matrixSize * matrixSize * sizeof(ElementType); 
  memcpy(C, h_C, size); 
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
      if(useCublas)
        CkPrintf("%.2f ", C[j*matrixSize+i]);
      else
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
