#include <stdio.h>

#include "main.decl.h"
#include "main.h"
#include "jacobi.decl.h"
#include "jacobi_config.h"


/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_Jacobi jacobiProxy;


Main::Main(CkArgMsg *msg) {

  // Print some header information for the user
  CkPrintf(" ----- 2D Jacobi for Cell -----\n");
  CkPrintf("   Matrix : [ %d x %d ]\n", NUM_ROWS * NUM_CHARES, NUM_COLS * NUM_CHARES);
  CkPrintf("   Chare Matrix : [ %d x %d ]\n", NUM_CHARES, NUM_CHARES);
  CkPrintf("   Per Chare Matrix : [ %d x %d ]\n", NUM_ROWS, NUM_COLS);
  unsigned int memNeeded = ((NUM_ROWS + 2) * (NUM_COLS + 2)) * sizeof(float) * 2;
  CkPrintf("   Per Work Request Memory : %d (0x%08x) bytes\n", memNeeded, memNeeded);

  // Display some information on the configuration of the run
  CkPrintf("Config:\n");
  CkPrintf("  USE_REDUCTION = %d\n", USE_REDUCTION);
  CkPrintf("  USE_MESSAGES = %d\n", USE_MESSAGES);
  CkPrintf("  CHARE_MAPPING_TO_PES__STRIPE = %d\n", CHARE_MAPPING_TO_PES__STRIPE);

  // Init the member variables
  iterationCount = 0;

  // Initialize the partialMaxError and checkInCount arrays
  for (int i = 0; i < REPORT_MAX_ERROR_BUFFER_DEPTH; i++) {
    partialMaxError[i] = 0.0f;
    checkInCount[i] = 0;
  }

  // Set the mainProxy readonly
  mainProxy = thisProxy;

  // Create the Jacobi array
  #if CHARE_MAPPING_TO_PES__STRIPE != 0

    CkPrintf("Using Chare striping...\n");

    register float numPEs_f = (float)(CkNumPes());
    jArray = CProxy_Jacobi::ckNew();
    for (int i = 0; i < (NUM_CHARES * NUM_CHARES); i++) {
      register int pe = (int)(((float)i / (float)(NUM_CHARES * NUM_CHARES)) * numPEs_f);
      jArray(i).insert(pe);
    }
    jArray.doneInserting();

  #else

    CkPrintf("Using default Chare placement...\n");

    jArray = CProxy_Jacobi::ckNew(NUM_CHARES * NUM_CHARES);

  #endif
  jacobiProxy = jArray;

  #if USE_REDUCTION != 0
    // Register a reduction callback with the array
    CkCallback *cb = new CkCallback(CkIndex_Main::maxErrorReductionClient(NULL), mainProxy);
    jArray.ckSetReductionClient(cb);
  #endif

  // Clear the createdCheckIn count
  createdCheckIn_count = 0;

  // STATS
  reportMaxError_resendCount = 0;
}

void Main::createdCheckIn() {

  createdCheckIn_count++;
  if (createdCheckIn_count >= (NUM_CHARES * NUM_CHARES)) {

    // Start timing
    startTime = CkWallTimer();

    // Tell the jArray to start the first iteration
    jArray.startIteration();
    #if DISPLAY_MATRIX != 0
      CkPrintf("Starting Iteration %d...\n", iterationCount);
    #endif
  }
}

void Main::maxErrorReductionClient(CkReductionMsg *msg) {

  float maxError = *((float*)(msg->getData()));

  #if DISPLAY_MAX_ERROR_FREQ > 0
    if (iterationCount == 0 || (iterationCount % DISPLAY_MAX_ERROR_FREQ) == 0) {
      CkPrintf("Iteration %d Finished... maxError = %f...\n", iterationCount, maxError);
      fflush(NULL);
    }
  #endif

  if (maxError <= MAX_ERROR) {

    // Stop timing
    endTime = CkWallTimer();

    CkPrintf("final maxError = %.12f\n", maxError);
    CkPrintf("final iterationCount = %d\n", iterationCount);
    CkPrintf("Time: %lfs\n", endTime - startTime);

    CkExit();

  } else {
    iterationCount++;
  }

}


void Main::reportMaxError(float val, int iter) {

  // Calculate the iteration count offset of this value and the current iteration
  //   the main chare is working on
  register int iterDelta = iter - iterationCount;
  if (iterDelta < 0) {
    CkPrintf("iterDelta = %d, iter = %d, iterationCount = %d\n",
             iterDelta, iter, iterationCount
	    );
    CkAbort("ERROR: iterDelta < 0 in Main::reportMaxError... later...\n");
  }

  // Check to see if this partial max error does not go into the buffer of max error
  //   values... if so, resend to self... if it does, combine with the appropriate value
  if (iterDelta >= REPORT_MAX_ERROR_BUFFER_DEPTH) {

    // STATS
    reportMaxError_resendCount++;

    thisProxy.reportMaxError(val, iter);
    return;
  }

  // Combine the value with the appropriate partial max error
  partialMaxError[iterDelta] = (partialMaxError[iterDelta] < val)
                                 ? (val)
                                 : (partialMaxError[iterDelta]);
  checkInCount[iterDelta]++;

  // Check to see if one or more iterations have completed
  while (checkInCount[0] >= (NUM_CHARES * NUM_CHARES)) {

    //CkPrintf("Iteration %d Finished... maxError = %f...\n",
    //         iterationCount, partialMaxError[0]);
    #if DISPLAY_MAX_ERROR_FREQ > 0
      if (iterationCount == 0 || (iterationCount % DISPLAY_MAX_ERROR_FREQ) == 0)
        CkPrintf("Iteration %d Finished... maxError = %f...\n", iterationCount, partialMaxError[0]);
    #endif

    if (partialMaxError[0] <= MAX_ERROR) {

      // Stop timing
      endTime = CkWallTimer();

      CkPrintf("final maxError = %.12f\n", partialMaxError[0]);
      CkPrintf("final iterationCount = %d\n", iterationCount);
      CkPrintf("Time: %lfs\n", endTime - startTime);

      // STATS
      CkPrintf("reportMaxError_resendCount = %d\n", reportMaxError_resendCount);

      CkExit();

    } else {

      iterationCount++;

      // Advance the elements in the paritial max error buffer
      for (int i = 1; i < REPORT_MAX_ERROR_BUFFER_DEPTH; i++) {
        partialMaxError[i-1] = partialMaxError[i];
        checkInCount[i-1] = checkInCount[i];
      }
      partialMaxError[REPORT_MAX_ERROR_BUFFER_DEPTH - 1] = 0.0f;
      checkInCount[REPORT_MAX_ERROR_BUFFER_DEPTH - 1] = 0;
    }

  } // end while

}


#include "main.def.h"
