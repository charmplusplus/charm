#include "patch.h"
#include "selfCompute.h"
#include "pairCompute.h"
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


Patch::Patch() {

  particleX = particleY = particleZ = NULL;
  particleQ = particleM = NULL;
  forceSumX = forceSumY = forceSumZ = NULL;
  velocityX = velocityY = velocityZ = NULL;
  numParticles = -1;
}


Patch::Patch(CkMigrateMessage* msg) {
  CkAbort("Patch::Patch(CkMigrateMessage* msg) not implemented yet");
}


Patch::~Patch() {
  if (particleX != NULL) { CmiFreeAligned(particleX); particleX = NULL; }
  if (particleY != NULL) { CmiFreeAligned(particleY); particleY = NULL; }
  if (particleZ != NULL) { CmiFreeAligned(particleZ); particleZ = NULL; }
  if (particleQ != NULL) { CmiFreeAligned(particleQ); particleQ = NULL; }
  if (particleM != NULL) { CmiFreeAligned(particleM); particleM = NULL; }
  if (forceSumX != NULL) { CmiFreeAligned(forceSumX); forceSumX = NULL; }
  if (forceSumY != NULL) { CmiFreeAligned(forceSumY); forceSumY = NULL; }
  if (forceSumZ != NULL) { CmiFreeAligned(forceSumZ); forceSumZ = NULL; }
  if (velocityX != NULL) { CmiFreeAligned(velocityX); velocityX = NULL; }
  if (velocityY != NULL) { CmiFreeAligned(velocityY); velocityY = NULL; }
  if (velocityZ != NULL) { CmiFreeAligned(velocityZ); velocityZ = NULL; }
  numParticles = 0;
}


void Patch::init(int numParticles) {

  // Also tell the proxy patches to initialize
  #if USE_PROXY_PATCHES != 0
  {
    const int patchIndex = PATCH_XYZ_TO_I(thisIndex.x, thisIndex.y, thisIndex.z);
    proxyPatchProxy = CProxy_ProxyPatch::ckNew(patchIndex);
    proxyPatchProxy.init(numParticles);
  }
  #endif

  // Allocate memory for the particles
  this->numParticles = numParticles;
  particleX = (float*)(CmiMallocAligned(numParticles * sizeof(float), 128));
  particleY = (float*)(CmiMallocAligned(numParticles * sizeof(float), 128));
  particleZ = (float*)(CmiMallocAligned(numParticles * sizeof(float), 128));
  particleQ = (float*)(CmiMallocAligned(numParticles * sizeof(float), 128));
  particleM = (float*)(CmiMallocAligned(numParticles * sizeof(float), 128));
  forceSumX = (float*)(CmiMallocAligned(numParticles * sizeof(float), 128));
  forceSumY = (float*)(CmiMallocAligned(numParticles * sizeof(float), 128));
  forceSumZ = (float*)(CmiMallocAligned(numParticles * sizeof(float), 128));
  velocityX = (float*)(CmiMallocAligned(numParticles * sizeof(float), 128));
  velocityY = (float*)(CmiMallocAligned(numParticles * sizeof(float), 128));
  velocityZ = (float*)(CmiMallocAligned(numParticles * sizeof(float), 128));

  // Initialize the particles
  randomizeParticles();

  // Create an array section for the pair computes this patch interacts with
  #if USE_ARRAY_SECTIONS != 0

    // Enumerate the pair computes that this patch interacts with
    const int patchIndex = PATCH_XYZ_TO_I(thisIndex.x, thisIndex.y, thisIndex.z);
    const int numPatches = numPatchesX * numPatchesY * numPatchesZ;

    // Lower section
    CkVec<CkArrayIndex2D> pairIndexes_lower;
    for (int i = 0; i < patchIndex; i++) {
      pairIndexes_lower.push_back(CkArrayIndex2D(i, patchIndex));
    }
    pairComputeArraySection_lower = CProxySection_PairCompute::ckNew(pairComputeArrayProxy, pairIndexes_lower.getVec(), pairIndexes_lower.size());

    CkVec<CkArrayIndex2D> pairIndexes_upper;
    for (int i = patchIndex + 1; i < numPatches; i++) {
      pairIndexes_upper.push_back(CkArrayIndex2D(patchIndex, i));
    }
    pairComputeArraySection_upper = CProxySection_PairCompute::ckNew(pairComputeArrayProxy, pairIndexes_upper.getVec(), pairIndexes_upper.size());

  #endif

  // Check in with the main proxy
  mainProxy.initCheckIn();
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

  // Send particle data to pair computes

  #if USE_PROXY_PATCHES != 0

    proxyPatchProxy.patchData(numParticles, particleX, particleY, particleZ, particleQ);

  #elif USE_ARRAY_SECTIONS != 0

    // Send particle data to self computes
    selfComputeArrayProxy(thisIndex.x, thisIndex.y, thisIndex.z).patchData(numParticles, particleX, particleY, particleZ, particleQ);

    pairComputeArraySection_lower.patchData(numParticles, particleX, particleY, particleZ, particleQ, 1);
    pairComputeArraySection_upper.patchData(numParticles, particleX, particleY, particleZ, particleQ, 0);

  #else

    // Send particle data to self computes
    selfComputeArrayProxy(thisIndex.x, thisIndex.y, thisIndex.z).patchData(numParticles, particleX, particleY, particleZ, particleQ);

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

  #endif
}


void Patch::forceCheckIn(int numParticles, float* forceX, float* forceY, float* forceZ) {
  forceCheckIn(numParticles, forceX, forceY, forceZ, 1);
}
void Patch::forceCheckIn(int numParticles, float* forceX, float* forceY, float* forceZ, int numForceCheckIns) {

  // Accumulate the force data
  #if 0
    register vec4f* fsx = (vec4f*)forceSumX;
    register vec4f* fsy = (vec4f*)forceSumY;
    register vec4f* fsz = (vec4f*)forceSumZ;
    register vec4f* fx = (vec4f*)forceX;
    register vec4f* fy = (vec4f*)forceY;
    register vec4f* fz = (vec4f*)forceZ;
    register const int numParticles_vec = numParticles / (sizeof(vec4f) * sizeof(float));
    register int i;
    for (i = 0; i < numParticles_vec; i++) {
      fsx[i] = vadd4f(fsx[i], fx[i]);
      fsy[i] = vadd4f(fsy[i], fy[i]);
      fsz[i] = vadd4f(fsz[i], fz[i]);
    }
  #else
    for (int i = 0; i < numParticles; i++) {
      forceSumX[i] += forceX[i];
      forceSumY[i] += forceY[i];
      forceSumZ[i] += forceZ[i];
    }
  #endif

  // Count the incoming forced data and integrate if all force data has arrived
  remainingForceCheckIns -= numForceCheckIns;
  if (remainingForceCheckIns <= 0) {
    thisProxy(thisIndex.x, thisIndex.y, thisIndex.z).integrate();
  }
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


ProxyPatch::ProxyPatch(int patchIndex) {
  this->patchIndex = patchIndex;
  numParticles = -1;
  particleX = particleY = particleZ = particleQ = NULL;
  forceSumX = forceSumY = forceSumZ = NULL;
}


ProxyPatch::ProxyPatch(CkMigrateMessage *msg) {
  CkAbort("ProxyPatch::ProxyPatch(CkMigrateMessage *msg) not implemented yet");
}


ProxyPatch::~ProxyPatch() {
  if (particleX != NULL) { CmiFreeAligned(particleX); particleX = NULL; }
  if (particleY != NULL) { CmiFreeAligned(particleY); particleY = NULL; }
  if (particleZ != NULL) { CmiFreeAligned(particleZ); particleZ = NULL; }
  if (particleQ != NULL) { CmiFreeAligned(particleQ); particleQ = NULL; }
  if (forceSumX != NULL) { CmiFreeAligned(forceSumX); forceSumX = NULL; }
  if (forceSumY != NULL) { CmiFreeAligned(forceSumY); forceSumY = NULL; }
  if (forceSumZ != NULL) { CmiFreeAligned(forceSumZ); forceSumZ = NULL; }
  numParticles = -1;
}


void ProxyPatch::init(int numParticles) {

  // Allocate memory for the particles
  this->numParticles = numParticles;
  particleX = (float*)(CmiMallocAligned(numParticles * sizeof(float), 128));
  particleY = (float*)(CmiMallocAligned(numParticles * sizeof(float), 128));
  particleZ = (float*)(CmiMallocAligned(numParticles * sizeof(float), 128));
  particleQ = (float*)(CmiMallocAligned(numParticles * sizeof(float), 128));
  forceSumX = (float*)(CmiMallocAligned(numParticles * sizeof(float), 128));
  forceSumY = (float*)(CmiMallocAligned(numParticles * sizeof(float), 128));
  forceSumZ = (float*)(CmiMallocAligned(numParticles * sizeof(float), 128));

  // Check in with the main proxy
  mainProxy.initCheckIn();
}


void ProxyPatch::patchData(int numParticles, float* particleX, float* particleY, float* particleZ, float* particleQ) {

  // Copy in the updated particle data
  memcpy(this->particleX, particleX, numParticles * sizeof(float));
  memcpy(this->particleY, particleY, numParticles * sizeof(float));
  memcpy(this->particleZ, particleZ, numParticles * sizeof(float));
  memcpy(this->particleQ, particleQ, numParticles * sizeof(float));

  // Clear out the force arrays
  memset(this->forceSumX, 0, numParticles * sizeof(float));
  memset(this->forceSumY, 0, numParticles * sizeof(float));
  memset(this->forceSumZ, 0, numParticles * sizeof(float));

  // Reset the patch checkin counters
  checkInCount = 0;

  // Call patchData on the local self compute
  const int patchX = PATCH_I_TO_X(patchIndex);
  const int patchY = PATCH_I_TO_Y(patchIndex);
  const int patchZ = PATCH_I_TO_Z(patchIndex);
  SelfCompute* localSelfCompute = selfComputeArrayProxy(patchX, patchY, patchZ).ckLocal();
  if (localSelfCompute != NULL) {
    localSelfCompute->patchData(numParticles, this->particleX, this->particleY, this->particleZ, this->particleQ, thisProxy);
    checkInCount++;
  }

  // Call patchData on all local pair computes
  const int myPe = CkMyPe();
  for (int i = 0; i < patchIndex; i++) {
    PairCompute* localPairCompute = pairComputeArrayProxy(i, patchIndex).ckLocal();
    if (localPairCompute != NULL) {
      localPairCompute->patchData(numParticles, this->particleX, this->particleY, this->particleZ, this->particleQ, 1, thisProxy);
      checkInCount++;
    }
  }
  const int numPatches = numPatchesX * numPatchesY * numPatchesZ;
  for (int i = patchIndex + 1; i < numPatches; i++) {
    PairCompute* localPairCompute = pairComputeArrayProxy(patchIndex, i).ckLocal();
    if (localPairCompute != NULL) {
      localPairCompute->patchData(numParticles, this->particleX, this->particleY, this->particleZ, this->particleQ, 0, thisProxy);
      checkInCount++;
    }
  }

  numToCheckIn = checkInCount;
}


void ProxyPatch::forceCheckIn(int numParticles, float* forceX, float* forceY, float* forceZ) {

  // Accumulate the force data
  #if 0
    register vec4f* forceX_vec = (vec4f*)forceX;
    register vec4f* forceY_vec = (vec4f*)forceY;
    register vec4f* forceZ_vec = (vec4f*)forceZ;
    register vec4f* forceSumX_vec = (vec4f*)forceSumX;
    register vec4f* forceSumY_vec = (vec4f*)forceSumY;
    register vec4f* forceSumZ_vec = (vec4f*)forceSumZ;
    const int numParticles_vec = numParticles / (sizeof(vec4f) / sizeof(float));
    for (int i = 0; i < numParticles_vec; i++) {
      forceSumX_vec[i] += forceX_vec[i];
      forceSumY_vec[i] += forceY_vec[i];
      forceSumZ_vec[i] += forceZ_vec[i];
    }
  #else
    for (int i = 0; i < numParticles; i++) {
      forceSumX[i] += forceX[i];
      forceSumY[i] += forceY[i];
      forceSumZ[i] += forceZ[i];
    }
  #endif

  // Once all computes this proxy called have contributed forces, send the data back to the patch itself
  checkInCount--;
  if (checkInCount <= 0) {
    const int patchX = PATCH_I_TO_X(patchIndex);
    const int patchY = PATCH_I_TO_Y(patchIndex);
    const int patchZ = PATCH_I_TO_Z(patchIndex);
    patchArrayProxy(patchX, patchY, patchZ).forceCheckIn(numParticles, forceSumX, forceSumY, forceSumZ, numToCheckIn);
  }
}


#include "patch.def.h"
