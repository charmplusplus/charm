#include <stdlib.h>
#include <time.h>

#include "main.h"
#include "md_config.h"


// Read-Only Variables
CProxy_Main mainProxy;
CProxy_Patch patchArrayProxy;
CProxy_SelfCompute selfComputeArrayProxy;
CProxy_PairCompute pairComputeArrayProxy;
int numPatchesX;
int numPatchesY;
int numPatchesZ;


// DMK - DEBUG
#if COUNT_FLOPS != 0
  unsigned long long int globalFlopCount;
#endif


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
           "  Simulation Steps: %d\n"
           #if USE_PROXY_PATCHES != 0
           "  Proxy Patches Enabled\n"
           #endif
           #if USE_ARRAY_SECTIONS != 0
           "  Array Sections Enabled\n"
           #endif
           "  StepPerPrint: %d\n",
           numPatchesX, numPatchesY, numPatchesZ,
           numParticlesPerPatch,
           numStepsRemaining,
           STEPS_PER_PRINT
	  );
  if(sizeof(MD_FLOAT)==4)
    CkPrintf("Single Precision\n");
  else  if(sizeof(MD_FLOAT)==8)
    CkPrintf("Double Precision\n");
  else 
    CkPrintf("Precision %d bytes\n",sizeof(MD_FLOAT));
  // DMK - DEBUG
  #if ENABLE_USER_EVENTS != 0
    traceRegisterUserEvent("Patch::forceCheckIn_callback()", PROJ_USER_EVENT_PATCH_FORCECHECKIN_CALLBACK);
    traceRegisterUserEvent("Patch::integrate_callback()", PROJ_USER_EVENT_PATCH_INTEGRATE_CALLBACK);
    traceRegisterUserEvent("SelfCompute::doCalc_callback()", PROJ_USER_EVENT_SELFCOMPUTE_DOCALC_CALLBACK);
    traceRegisterUserEvent("PairCompute::doCalc_callback()", PROJ_USER_EVENT_PAIRCOMPUTE_DOCALC_CALLBACK);
    traceRegisterUserEvent("SelfCompute::doCalc() - Work", PROJ_USER_EVENT_SELFCOMPUTE_DOCALC_WORK);
    traceRegisterUserEvent("PairCompute::doCalc() - Work", PROJ_USER_EVENT_PAIRCOMPUTE_DOCALC_WORK);
    traceRegisterUserEvent("CmiMachineProgressImpl", PROJ_USER_EVENT_MACHINEPROGRESS);
  #endif

  // DMK - DEBUG
  #if COUNT_FLOPS != 0
    globalFlopCount = 0;
    if (CkNumPes() != 1) {
      CkPrintf("ERROR: When COUNT_FLOPS is enabled, only a single processor should be used... Exiting...\n");
      CkExit();
    }
  #endif

  // Spread a proxy to this main chare object to all processors via a read-only
  mainProxy = thisProxy;

  // Create the patch array
  patchArrayProxy = CProxy_Patch::ckNew(numPatchesX, numPatchesY, numPatchesZ);

  // Create the self compute array
  selfComputeArrayProxy = CProxy_SelfCompute::ckNew(numPatchesX, numPatchesY, numPatchesZ);

  // Create the pair compute array
  #if ENABLE_STATIC_LOAD_BALANCING != 0
    // NOTE : For now, this code has to be manually changed to match the nodelist file since there is no way to
    //   pass this information into the program at runtime.  In the future, this is something the runtime system
    //   take care of in the ideal case.
    int numPEs = CkNumPes();
    #define W_X86    ( 10)  //  10
    #define W_BLADE  (125)  // 100
    #define W_PS3    ( 96)  //  75
    // NOTE: The peWeights should match the hetero nodelist file being used
    //int peWeights[13] = { W_X86, W_BLADE, W_PS3, W_BLADE, W_PS3, W_BLADE, W_PS3, W_BLADE, W_PS3, W_BLADE, W_BLADE, W_BLADE, W_BLADE };
    int peWeights[13] = { W_BLADE, W_PS3, W_BLADE, W_PS3, W_BLADE, W_PS3, W_BLADE, W_PS3, W_BLADE, W_BLADE, W_BLADE, W_BLADE, W_BLADE };
    //int peWeights[14] = { W_X86, W_X86, W_BLADE, W_PS3, W_BLADE, W_PS3, W_BLADE, W_PS3, W_BLADE, W_PS3, W_BLADE, W_BLADE, W_BLADE, W_BLADE };
    int peStats[13] = { 0 };
    CkAssert(numPEs <= 13);
    int rValLimit  = 0;
    for (int i = 0; i < numPEs; i++) { rValLimit += peWeights[i]; }
  #endif
  pairComputeArrayProxy = CProxy_PairCompute::ckNew();
  const int numPatches = numPatchesX * numPatchesY * numPatchesZ;
  for (int p0 = 0; p0 < numPatches; p0++) {
    for (int p1 = p0 + 1; p1 < numPatches; p1++) {
      #if ENABLE_STATIC_LOAD_BALANCING != 0
        int pe = 0;
        int rVal = rand() % rValLimit;
        for (int i = 0; i < numPEs; i++) { if (rVal < peWeights[i]) { pe = i; break; } rVal -= peWeights[i]; }
        pairComputeArrayProxy(p0, p1).insert(pe);
        peStats[pe]++;
      #else
        pairComputeArrayProxy(p0, p1).insert();
      #endif
    }
  }
  pairComputeArrayProxy.doneInserting();
  #if ENABLE_STATIC_LOAD_BALANCING != 0
    int numPairComputes = 0;
    for (int i = 0; i < numPEs; i++) { numPairComputes += peStats[i]; }
    for (int i = 0; i < numPEs; i++) {
      CkPrintf("[STATS] :: peStats[%d] = %6d (%5.2f%%)\n", i, peStats[i], ((float)peStats[i]) / ((float)numPairComputes) * 10zero);
    }
  #endif

  // Start initialization (NOTE: Patch will initiate proxy patches directly if proxy patches are being used)
  selfComputeArrayProxy.init(numParticlesPerPatch);
  pairComputeArrayProxy.init(numParticlesPerPatch);

  #if USE_PROXY_PATCHES != 0
    for (int x = 0; x < numPatchesX; x++) {
      for (int y = 0; y < numPatchesY; y++) {
        for (int z = 0; z < numPatchesZ; z++) {
          int patchIndex = PATCH_XYZ_TO_I(x, y, z);
          CProxy_ProxyPatch proxyPatchProxy = CProxy_ProxyPatch::ckNew(patchIndex);
          proxyPatchProxy.init(numParticlesPerPatch);
          patchArrayProxy(x, y, z).init(numParticlesPerPatch, proxyPatchProxy);
	}
      }
    }
  #else
    patchArrayProxy.init(numParticlesPerPatch);
  #endif
}


Main::~Main() {
}


void Main::initCheckIn() {

  static int numCheckedIn = 0;
  const int numPatches = numPatchesX * numPatchesY * numPatchesZ;
  int numToCheckIn = (2 * numPatches) + ((numPatches * (numPatches - 1)) / 2);
  #if USE_PROXY_PATCHES != 0
    numToCheckIn += (numPatches * CkNumPes());
  #endif

  // Count this caller and check to see if everyone has called
  numCheckedIn++;
  if (numCheckedIn >= numToCheckIn) {
    numCheckedIn = 0;

    // Start timing
    simPrevTime = simStartTime = CkWallTimer();

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
    double curTime = CkWallTimer();
    CkPrintf("Main::patchCheckIn() - Simulation (%d steps remaining)... (deltaTime: %lf sec)...\n",
             numStepsRemaining, curTime - simPrevTime
            );
    simPrevTime = curTime;
    if (numStepsRemaining <= 0) {

      // Stop timing and display elapsed time
      double simStopTime = curTime;
      CkPrintf("Elapsed Time: %lf sec\n", simStopTime - simStartTime);

      // DMK - DEBUG
      #if COUNT_FLOPS != 0
        CkPrintf("Global Flop Count: %llu flops (%llu GFlops)\n", globalFlopCount, globalFlopCount / 1000000000);
      #endif

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
