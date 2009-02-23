#include <stdlib.h>
#include <time.h>

#include "main.h"
#include "md_config.h"


// Read-Onlys
CProxy_Main mainProxy;
CProxy_Patch patchArrayProxy;
CProxy_SelfCompute selfComputeArrayProxy;
CProxy_PairCompute pairComputeArrayProxy;
int numPatchesX;
int numPatchesY;
int numPatchesZ;


void Main::parseCommandLine(int argc, char** argv) {

  // Verify the parameters
  if (argc <= 0) return;
  if (argv == NULL) return;

  // Set default parameters for the application
  numParticlesPerPatch = DEFAULT_NUM_PARTICLES_PER_PATCH;
  numPatchesX = DEFAULT_NUM_PATCHES_X;
  numPatchesY = DEFAULT_NUM_PATCHES_Y;
  numPatchesZ = DEFAULT_NUM_PATCHES_Z;
  numStepsRemaining = DEFAULT_NUM_STEPS;

  // Parse the command line
  if (argc > 1) { numParticlesPerPatch = atoi(argv[1]); }
  if (argc > 4) {
    numPatchesX = atoi(argv[2]);
    numPatchesY = atoi(argv[3]);
    numPatchesZ = atoi(argv[4]);
  }
  if (argc > 5) { numStepsRemaining = atoi(argv[5]); }

  // Check application parameters
  if (numParticlesPerPatch <= 0) { CkAbort("numParticlesPerPatch must be greater than zero"); }
  if ((numParticlesPerPatch % 4) != 0) { CkAbort("numParticlesPerPatch must be a multiple of 4"); }
  if (numPatchesX <= 0) { CkAbort("Invalid number of patches in X dimension"); }
  if (numPatchesY <= 0) { CkAbort("Invalid number of patches in Y dimension"); }
  if (numPatchesZ <= 0) { CkAbort("Invalid number of patches in Z dimension"); }
  if (numStepsRemaining <= 0) { CkAbort("Invalid number of steps"); }
}


Main::Main(CkArgMsg* msg) {

  // Parse the command line (sets application parameters)
  parseCommandLine(msg->argc, msg->argv);
  delete msg;

  // Display a header that displays info about the run
  CkPrintf("MD Simulation\n"
           "  Patch Grid: x:%d by y:%d by z:%d\n"
           "  NumParticlesPerPatch: %d\n"
           "  Simulation Steps: %d\n",
           numPatchesX, numPatchesY, numPatchesZ,
           numParticlesPerPatch,
           numStepsRemaining
	  );

  // Spread a proxy to this main chare object to all processors via a read-only
  mainProxy = thisProxy;

  // Create the patch array
  patchArrayProxy = CProxy_Patch::ckNew(numParticlesPerPatch, numPatchesX, numPatchesY, numPatchesZ);

  // Create the self compute array
  selfComputeArrayProxy = CProxy_SelfCompute::ckNew(numParticlesPerPatch, numPatchesX, numPatchesY, numPatchesZ);

  // Create the pair compute array
  pairComputeArrayProxy = CProxy_PairCompute::ckNew();
  const int numPatches = numPatchesX * numPatchesY * numPatchesZ;
  for (int p0 = 0; p0 < numPatches; p0++) {
    for (int p1 = p0 + 1; p1 < numPatches; p1++) {
      pairComputeArrayProxy(p0, p1).insert(numParticlesPerPatch);
    }
  }
  pairComputeArrayProxy.doneInserting();
}


Main::~Main() {
}


void Main::proxyCheckIn() {

  static int numCheckedIn = 0;
  const int numPatches = numPatchesX * numPatchesY * numPatchesZ;
  const int numToCheckIn = (2 * numPatches) + ((numPatches * (numPatches - 1)) / 2);

  // Count this caller and check to see if everyone has called
  numCheckedIn++;
  if (numCheckedIn >= numToCheckIn) {
    numCheckedIn = 0;

    // Start timing
    simStartTime = CkWallTimer();

    // One step for main (patches do many steps)
    const int numStepsToDo = (numStepsRemaining > STEPS_PER_PRINT) ? (STEPS_PER_PRINT) : (numStepsRemaining);
    patchArrayProxy.startIterations(numStepsToDo);
    CkPrintf("Main::patchCheckIn() - Starting Simulation (%d steps remaining)...\n", numStepsRemaining);
    numStepsRemaining -= numStepsToDo - 1;
  }
}


void Main::patchCheckIn() {

  static int numCheckedIn = 0;
  const int numToCheckIn = (numPatchesX * numPatchesY * numPatchesZ);

  // Count this caller and check to see if everyone has called
  numCheckedIn++;
  if (numCheckedIn >= numToCheckIn) {
    numCheckedIn = 0;

    // Check to see if there is another step, if so start it, otherwise exit
    numStepsRemaining--;
    CkPrintf("Main::patchCheckIn() - Simulation (%d steps remaining)...\n", numStepsRemaining);
    if (numStepsRemaining <= 0) {

      // Stop timing and display elapsed time
      double simStopTime = CkWallTimer();
      CkPrintf("Elapsed Time: %lf sec\n", simStopTime - simStartTime);

      // The simulation has completed, so exit
      CkExit();

    } else {

      // Begin another set of iterations
      const int numStepsToDo = (numStepsRemaining > STEPS_PER_PRINT) ? (STEPS_PER_PRINT) : (numStepsRemaining);
      patchArrayProxy.startIterations(numStepsToDo);
      numStepsRemaining -= numStepsToDo - 1;
    }
  }
}


void seedRand() {

  // Initialize the seed for this processor to a function of the time and the PE index
  srand(time(NULL) + (754027 + CkMyPe()));
}


#include "main.def.h"
