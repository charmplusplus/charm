#include <stdio.h>

#include "main.decl.h"
#include "main.h"
#include "jacobi.decl.h"
#include "jacobi_shared.h"


Main::Main(CkArgMsg *msg) {

  // Print some header information for the user
  CkPrintf(" ----- 2D Jacobi for Cell -----\n");
  CkPrintf("   Matrix : [ %d x %d ]\n", NUM_ROWS * NUM_CHARES, NUM_COLS * NUM_CHARES);
  CkPrintf("   Chare Matrix : [ %d x %d ]\n", NUM_CHARES, NUM_CHARES);
  CkPrintf("   Per Chare Matrix : [ %d x %d ]\n", NUM_ROWS, NUM_COLS);
  unsigned int memNeeded = ((NUM_ROWS + 2) * (NUM_COLS + 2)) * sizeof(float) * 2;
  CkPrintf("   Per Work Request Memory : %d (0x%08x) bytes\n", memNeeded, memNeeded);

  // Init the member variables
  iterationCount = 0;

  // Set the mainProxy readonly
  mainProxy = thisProxy;

  // Create the Jacobi array
  jArray = CProxy_Jacobi::ckNew(NUM_CHARES * NUM_CHARES);

  // Register a reduction callback with the array
  CkCallback *cb = new CkCallback(CkIndex_Main::maxErrorReductionClient(NULL), mainProxy);
  jArray.ckSetReductionClient(cb);

  // Tell the jArray to start the first iteration
  iterationCount++;
  jArray.startIteration();
  #if DISPLAY_MATRIX != 0
    CkPrintf("Starting Iteration %d...\n", iterationCount);
  #endif
}

void Main::maxErrorReductionClient(CkReductionMsg *msg) {

  float maxError = *((float*)(msg->getData()));

  #if DISPLAY_MAX_ERROR_FREQ > 0
    if (iterationCount == 1 || (iterationCount % DISPLAY_MAX_ERROR_FREQ) == 0)
      CkPrintf("Iteration %d Finished... maxError = %f...\n", iterationCount, maxError);
  #endif

  if (maxError <= MAX_ERROR) {
    CkPrintf("final maxError = %f\n", maxError);
    CkPrintf("final iterationCount = %d\n", iterationCount);
    CkExit();
  } else {
    iterationCount++;
    jArray.startIteration();
    #if DISPLAY_MATRIX != 0
      CkPrintf("Starting Iteration %d...\n", iterationCount);
    #endif
  }

}


#include "main.def.h"
