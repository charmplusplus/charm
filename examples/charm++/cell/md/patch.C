#include "patch.h"
#include "main.h"


void Patch::randomizeParticles() {

  // Fill in a box with electrons that initially have no velocity
  for (int i = 0; i < numParticles; i++) {
    particleX[i] = (randf() * SIM_BOX_SIDE_LEN) - (SIM_BOX_SIDE_LEN / 2.0f);
    particleY[i] = (randf() * SIM_BOX_SIDE_LEN) - (SIM_BOX_SIDE_LEN / 2.0f);
    particleZ[i] = (randf() * SIM_BOX_SIDE_LEN) - (SIM_BOX_SIDE_LEN / 2.0f);
    particleQ[i] = ELECTRON_CHARGE;
    particleM[i] = ELECTRON_MASS;
    velocityX[i] = 0.0f;
    velocityY[i] = 0.0f;
    velocityZ[i] = 0.0f;
  }
}


float Patch::randf() {
  return (((float)(rand() % 1483727)) / (1483727.0f));
}


Patch::Patch(int numParticles) {

  // Allocate memory for the particles
  this->numParticles = numParticles;
  particleX = (float*)(malloc_aligned(numParticles * sizeof(float), 128));
  particleY = (float*)(malloc_aligned(numParticles * sizeof(float), 128));
  particleZ = (float*)(malloc_aligned(numParticles * sizeof(float), 128));
  particleQ = (float*)(malloc_aligned(numParticles * sizeof(float), 128));
  particleM = (float*)(malloc_aligned(numParticles * sizeof(float), 128));
  forceSumX = (float*)(malloc_aligned(numParticles * sizeof(float), 128));
  forceSumY = (float*)(malloc_aligned(numParticles * sizeof(float), 128));
  forceSumZ = (float*)(malloc_aligned(numParticles * sizeof(float), 128));
  velocityX = (float*)(malloc_aligned(numParticles * sizeof(float), 128));
  velocityY = (float*)(malloc_aligned(numParticles * sizeof(float), 128));
  velocityZ = (float*)(malloc_aligned(numParticles * sizeof(float), 128));

  // Initialize the particles
  randomizeParticles();

  // Check in with the main proxy so it "knows" that this patch is ready to start simulation
  mainProxy.proxyCheckIn();
}


Patch::Patch(CkMigrateMessage* msg) {
  CkAbort("Patch::Patch(CkMigrateMessage* msg) not implemented yet");
}


Patch::~Patch() {
  if (particleX != NULL) { free_aligned(particleX); particleX = NULL; }
  if (particleY != NULL) { free_aligned(particleY); particleY = NULL; }
  if (particleZ != NULL) { free_aligned(particleZ); particleZ = NULL; }
  if (particleQ != NULL) { free_aligned(particleQ); particleQ = NULL; }
  if (particleM != NULL) { free_aligned(particleM); particleM = NULL; }
  if (forceSumX != NULL) { free_aligned(forceSumX); forceSumX = NULL; }
  if (forceSumY != NULL) { free_aligned(forceSumY); forceSumY = NULL; }
  if (forceSumZ != NULL) { free_aligned(forceSumZ); forceSumZ = NULL; }
  if (velocityX != NULL) { free_aligned(velocityX); velocityX = NULL; }
  if (velocityY != NULL) { free_aligned(velocityY); velocityY = NULL; }
  if (velocityZ != NULL) { free_aligned(velocityZ); velocityZ = NULL; }
  numParticles = 0;
}


void Patch::startIteration() { startIteration_common(1); }
void Patch::startIterations(int numIters) { startIteration_common(numIters); }
void Patch::startIteration_common(int numIters) {

  // Set the number of remaining time stops
  CkAssert(numIters > 0);
  remainingIterations = numIters;

  // Reset the number of expected computes that will check in
  //   NOTE: pair compute for every other patch, self compute for this patch
  remainingForceCheckIns = numPatchesX * numPatchesY * numPatchesZ;

  // Clear the force sum arrays
  register vec4f* fsx = (vec4f*)forceSumX;
  register vec4f* fsy = (vec4f*)forceSumY;
  register vec4f* fsz = (vec4f*)forceSumZ;
  const vec4f zero_vec = vspread4f(0.0f);
  register const int numParticles_vec = numParticles / (sizeof(vec4f) * sizeof(float));
  for (int i = 0; i < numParticles_vec; i++) {
    fsx[i] = zero_vec;
    fsy[i] = zero_vec;
    fsz[i] = zero_vec;
  }

  // DMK - DEBUG
  NetworkProgress

  // Send particle data to self computes
  selfComputeArrayProxy(thisIndex.x, thisIndex.y, thisIndex.z).doCalc(numParticles, particleX, particleY, particleZ, particleQ);

  // Send particle data to pair computes
  const int index = PATCH_XYZ_TO_I(thisIndex.x, thisIndex.y, thisIndex.z);
  for (int i = 0; i < index; i++) {

    // DMK - DEBUG
    NetworkProgress

    pairComputeArrayProxy(i, index).patchData(numParticles, particleX, particleY, particleZ, particleQ, 1);
  }
  const int numPatches = numPatchesX * numPatchesY * numPatchesZ;
  for (int i = index + 1; i < numPatches; i++) {

    // DMK - DEBUG
    NetworkProgress

    pairComputeArrayProxy(index, i).patchData(numParticles, particleX, particleY, particleZ, particleQ, 0);
  }

  // DMK - DEBUG
  NetworkProgress
}


void Patch::forceCheckIn_callback() {

  // DMK - DEBUG
  #if ENABLE_USER_EVENTS != 0
    double __start_time__ = CmiWallTimer();
  #endif

  // DMK - DEBUG
  NetworkProgress

  // Decrement the counter containing the number of remaining computes that need to report forces
  //   back to this patch.  Once all computes have checked in, send a message to accelerated
  //   'integrate' entry method.
  remainingForceCheckIns--;
  if (remainingForceCheckIns <= 0) {
    thisProxy(thisIndex.x, thisIndex.y, thisIndex.z).integrate();
  }

  // DMK - DEBUG
  NetworkProgress

  // DMK - DEBUG
  #if ENABLE_USER_EVENTS != 0
    double __end_time__ = CmiWallTimer();
    traceUserBracketEvent(PROJ_USER_EVENT_PATCH_FORCECHECKIN_CALLBACK, __start_time__, __end_time__);
  #endif
}

void Patch::integrate_callback() {

  // DMK - DEBUG
  #if ENABLE_USER_EVENTS != 0
    double __start_time__ = CmiWallTimer();
  #endif

  // DMK - DEBUG
  NetworkProgress

  // Decrement the counter containing the number of remaining iterations.  If there are
  //   more iterations, do another one, otherwise, check in with main
  remainingIterations--;
  if (remainingIterations > 0) {
    startIteration_common(remainingIterations);
  } else {
    mainProxy.patchCheckIn();
  }

  // DMK - DEBUG
  NetworkProgress

  // DMK - DEBUG
  #if ENABLE_USER_EVENTS != 0
    double __end_time__ = CmiWallTimer();
    traceUserBracketEvent(PROJ_USER_EVENT_PATCH_INTEGRATE_CALLBACK, __start_time__, __end_time__);
  #endif
}


#include "patch.def.h"
