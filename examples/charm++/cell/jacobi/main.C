#include <stdio.h>

#include "main.decl.h"
#include "main.h"
#include "jacobi.decl.h"
#include "jacobi_shared.h"


Main::Main(CkArgMsg *msg) {

  // Print some header information for the user
  CkPrintf(" ----- 2D Jacobi for Cell -----\n");

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

  if (iterationCount == 1 || (iterationCount % 10) == 0)
    CkPrintf("Iteration %d Finished... maxError = %f...\n", iterationCount, maxError);

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
