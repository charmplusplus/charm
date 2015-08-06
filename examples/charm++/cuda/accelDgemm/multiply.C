#include "multiply.decl.h"
#include "multiply.h"
#include <time.h>
#define DEBUG


CProxy_Main mainProxy;
int matrixSize;
Main::Main(CkArgMsg *m) {
  mainProxy = thisProxy;

  if (m->argc < 3) {
    numChares = 2;
    matrixSize = 8;
  }
  else{
    numChares = atoi(m->argv[1]);
    matrixSize = atoi(m->argv[2]);
  }
  delete m;
  workers = CProxy_Workers::ckNew(numChares);

  startTime = CkWallTimer();

  workers.beginWork(matrixSize);
}

void Main::finishWork(CkReductionMsg *m) {
  delete m;
  CkPrintf("Elapsed time: %f s\n", CkWallTimer() - startTime);
  CkExit();
}

Workers::Workers() {
  int size = matrixSize * matrixSize;
  A = new float[size];
  B = new float[size];
  C = new float[size];
  srand(time(NULL));
  randomInit(A, size);
  randomInit(B, size);
}

Workers::~Workers() {
  delete [] A;
  delete [] B;
  delete [] C;
}

Workers::Workers(CkMigrateMessage *msg) { }



void Workers::complete() {

  int size = matrixSize * matrixSize * sizeof(float);
#if 0
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
#endif
  float *verify= new float[matrixSize*matrixSize]; 
  //CkPrintf("[%d] C-gold\n", thisIndex);
  for (int i=0; i<matrixSize; i++) {
   // CkPrintf("[%d] ", thisIndex);
    for (int j=0; j<matrixSize; j++) {
      verify[i*matrixSize + j] = 0;  
      for (int k=0; k<matrixSize; k++) {
        verify[i*matrixSize + j]+= A[i*matrixSize +k] * B[k * matrixSize + j];
     }   
     if(verify[i*matrixSize+j]-C[i*matrixSize+j]!=0.00){
     
      CkPrintf("Error at index %d %d\n", i,j);
    }   
    CkPrintf("\n");
  }

}
  contribute(CkCallback(CkIndex_Main::finishWork(NULL), mainProxy));
}


void randomInit(float* data, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = rand() / (float)RAND_MAX;
  }
}

#include "multiply.def.h"
