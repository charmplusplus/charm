#include "patch.h"
#include "selfCompute.h"
#include "pairCompute.h"
#include "main.h"


void Patch::randomizeParticles() {

  // Fill in a box with electrons that initially have no velocity
  for (int i = 0; i < numParticles; i++) {
    particleX[i] = (randf() * SIM_BOX_SIDE_LEN) - (SIM_BOX_SIDE_LEN / two);
    particleY[i] = (randf() * SIM_BOX_SIDE_LEN) - (SIM_BOX_SIDE_LEN / two);
    particleZ[i] = (randf() * SIM_BOX_SIDE_LEN) - (SIM_BOX_SIDE_LEN / two);
    particleQ[i] = ELECTRON_CHARGE;
    particleM[i] = ELECTRON_MASS;
    velocityX[i] = zero;
    velocityY[i] = zero;
    velocityZ[i] = zero;

    // DMK - DEBUG
    #if DUMP_INITIAL_PARTICLE_DATA != 0
      CkPrintf("[INFO] :: Patch[%02d,%02d,%02d]::randomizeParticles() - particle[%04d] = { "
               "px:%+6e, py:%+6e, pz:%+6e, q:%+6e, m:%+6e, vx:%+6e, vy:%+6e, vz:%+6e }\n",
               thisIndex.x, thisIndex.y, thisIndex.z, i,
               particleX[i], particleY[i], particleZ[i], particleQ[i], particleM[i],
               velocityX[i], velocityY[i], velocityZ[i]
              );
    #endif
      //EJB DEBUG
    #if SANITY_CHECK
      CkAssert(finite(particleX[i]));
      CkAssert(finite(particleY[i]));
      CkAssert(finite(particleZ[i]));
      CkAssert(finite(velocityX[i]));
      CkAssert(finite(velocityY[i]));
      CkAssert(finite(velocityZ[i]));
    #endif 

  }
}


MD_FLOAT Patch::randf() {
  const int mask = 0x7FFFFFFF;
  return (((MD_FLOAT)(rand() % mask)) / ((MD_FLOAT)(mask)));

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
  init_common(numParticles);
}

void Patch::init(int numParticles, CProxy_ProxyPatch proxyPatchProxy) {
  #if USE_PROXY_PATCHES != 0
    this->proxyPatchProxy = proxyPatchProxy;
  #endif
  init_common(numParticles);
}

void Patch::init_common(int numParticles) {

  // Allocate memory for the particles
  this->numParticles = numParticles;
  particleX = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  particleY = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  particleZ = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  particleQ = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  particleM = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  forceSumX = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  forceSumY = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  forceSumZ = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  velocityX = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  velocityY = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  velocityZ = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));

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

    // Upper section
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
  #if 1
    memset(forceSumX, 0, sizeof(MD_FLOAT) * numParticles);
    memset(forceSumY, 0, sizeof(MD_FLOAT) * numParticles);
    memset(forceSumZ, 0, sizeof(MD_FLOAT) * numParticles);
  #else
    register MD_VEC* fsx = (MD_VEC*)forceSumX;
    register MD_VEC* fsy = (MD_VEC*)forceSumY;
    register MD_VEC* fsz = (MD_VEC*)forceSumZ;
    const MD_VEC zero_vec = vspread_MDF(zero);
    register const int numParticles_vec = numParticles / myvec_numElems;
    for (int i = 0; i < numParticles_vec; i++) {
      fsx[i] = zero_vec;
      fsy[i] = zero_vec;
      fsz[i] = zero_vec;
    }
  #endif

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


void Patch::forceCheckIn(int numParticles, MD_FLOAT* forceX, MD_FLOAT* forceY, MD_FLOAT* forceZ) {
  forceCheckIn(numParticles, forceX, forceY, forceZ, 1);
}
void Patch::forceCheckIn(int numParticles, MD_FLOAT* forceX, MD_FLOAT* forceY, MD_FLOAT* forceZ, int numForceCheckIns) {

  // Accumulate the force data
  #if 0
    register MD_VEC* fsx = (MD_VEC*)forceSumX;
    register MD_VEC* fsy = (MD_VEC*)forceSumY;
    register MD_VEC* fsz = (MD_VEC*)forceSumZ;
    register MD_VEC* fx = (MD_VEC*)forceX;
    register MD_VEC* fy = (MD_VEC*)forceY;
    register MD_VEC* fz = (MD_VEC*)forceZ;
    register const int numParticles_vec = numParticles / myvec_numElems;
    register int i;
    for (i = 0; i < numParticles_vec; i++) {
      fsx[i] = vadd_MDF(fsx[i], fx[i]);
      fsy[i] = vadd_MDF(fsy[i], fy[i]);
      fsz[i] = vadd_MDF(fsz[i], fz[i]);
    }
  #else
    for (int i = 0; i < numParticles; i++) {
#if SANITY_CHECK
      CkAssert(finite(forceX[i]));
      CkAssert(finite(forceY[i]));
      CkAssert(finite(forceZ[i]));
#endif 
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
    double __start_time__ = CkWallTimer();
  #endif

  // DMK - DEBUG
  #if COUNT_FLOPS != 0
    globalFlopCount += localFlopCount;
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
    double __end_time__ = CkWallTimer();
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
  particleX = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  particleY = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  particleZ = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  particleQ = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  forceSumX = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  forceSumY = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  forceSumZ = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));

  // Check in with the main proxy
  mainProxy.initCheckIn();
}


void ProxyPatch::patchData(int numParticles, MD_FLOAT* particleX, MD_FLOAT* particleY, MD_FLOAT* particleZ, MD_FLOAT* particleQ) {

  // Copy in the updated particle data
  memcpy(this->particleX, particleX, numParticles * sizeof(MD_FLOAT));
  memcpy(this->particleY, particleY, numParticles * sizeof(MD_FLOAT));
  memcpy(this->particleZ, particleZ, numParticles * sizeof(MD_FLOAT));
  memcpy(this->particleQ, particleQ, numParticles * sizeof(MD_FLOAT));

  // Clear out the force arrays
  memset(this->forceSumX, 0, numParticles * sizeof(MD_FLOAT));
  memset(this->forceSumY, 0, numParticles * sizeof(MD_FLOAT));
  memset(this->forceSumZ, 0, numParticles * sizeof(MD_FLOAT));

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


void ProxyPatch::forceCheckIn(int numParticles, MD_FLOAT* forceX, MD_FLOAT* forceY, MD_FLOAT* forceZ) {

  // Accumulate the force data
  #if USE_PROXY_PATCHES != 0  // Calls will be local and pointers will be aligned, so take advantage and vectorize the code
    register MD_VEC* forceX_vec = (MD_VEC*)forceX;
    register MD_VEC* forceY_vec = (MD_VEC*)forceY;
    register MD_VEC* forceZ_vec = (MD_VEC*)forceZ;
    register MD_VEC* forceSumX_vec = (MD_VEC*)forceSumX;
    register MD_VEC* forceSumY_vec = (MD_VEC*)forceSumY;
    register MD_VEC* forceSumZ_vec = (MD_VEC*)forceSumZ;
    const int numParticles_vec = numParticles / myvec_numElems;
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
