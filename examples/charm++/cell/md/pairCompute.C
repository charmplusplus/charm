#include "pairCompute.h"
#include "patch.h"
#include "main.h"


PairCompute::PairCompute() {
  particleX[0] = particleX[1] = NULL;
  particleY[0] = particleY[1] = NULL;
  particleZ[0] = particleZ[1] = NULL;
  particleQ[0] = particleQ[1] = NULL;
  forceX[0] = forceX[1] = NULL;
  forceY[0] = forceY[1] = NULL;
  forceZ[0] = forceZ[1] = NULL;
  numParticles = -1;
}


PairCompute::PairCompute(CkMigrateMessage* msg) {
  CkAbort("PairCompute::PairCompute(CkMigrateMessage* msg) not implemented yet");
}


PairCompute::~PairCompute() {

  #if USE_PROXY_PATCHES == 0
    if (particleX[0] != NULL) { CmiFreeAligned(particleX[0]); particleX[0] = NULL; }
    if (particleX[1] != NULL) { CmiFreeAligned(particleX[1]); particleX[1] = NULL; }
    if (particleY[0] != NULL) { CmiFreeAligned(particleY[0]); particleY[0] = NULL; }
    if (particleY[1] != NULL) { CmiFreeAligned(particleY[1]); particleY[1] = NULL; }
    if (particleZ[0] != NULL) { CmiFreeAligned(particleZ[0]); particleZ[0] = NULL; }
    if (particleZ[1] != NULL) { CmiFreeAligned(particleZ[1]); particleZ[1] = NULL; }
    if (particleQ[0] != NULL) { CmiFreeAligned(particleQ[0]); particleQ[0] = NULL; }
    if (particleQ[1] != NULL) { CmiFreeAligned(particleQ[1]); particleQ[1] = NULL; }
  #endif
  if (forceX[0] != NULL) { CmiFreeAligned(forceX[0]); forceX[0] = NULL; }
  if (forceX[1] != NULL) { CmiFreeAligned(forceX[1]); forceX[1] = NULL; }
  if (forceY[0] != NULL) { CmiFreeAligned(forceY[0]); forceY[0] = NULL; }
  if (forceY[1] != NULL) { CmiFreeAligned(forceY[1]); forceY[1] = NULL; }
  if (forceZ[0] != NULL) { CmiFreeAligned(forceZ[0]); forceZ[0] = NULL; }
  if (forceZ[1] != NULL) { CmiFreeAligned(forceZ[1]); forceZ[1] = NULL; }
  numParticles = -1;
}


void PairCompute::init(int numParticlesPerPatch) {

  // Initialize the arrays
  numParticles = numParticlesPerPatch;
  #if USE_PROXY_PATCHES == 0
    particleX[0] = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
    particleX[1] = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
    particleY[0] = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
    particleY[1] = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
    particleZ[0] = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
    particleZ[1] = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
    particleQ[0] = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
    particleQ[1] = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  #endif
  forceX[0] = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  forceX[1] = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  forceY[0] = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  forceY[1] = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  forceZ[0] = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  forceZ[1] = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  patchDataCount = 0;

  // Check in with the main chare
  mainProxy.initCheckIn();
}


void PairCompute::patchData(int numParticles, MD_FLOAT* particleX, MD_FLOAT* particleY, MD_FLOAT* particleZ, MD_FLOAT* particleQ, int fromPatch, CProxy_ProxyPatch proxyPatchProxy) {
  #if USE_PROXY_PATCHES != 0
    this->proxyPatchProxy[fromPatch] = proxyPatchProxy;
  #endif
  patchData(numParticles, particleX, particleY, particleZ, particleQ, fromPatch);
}

void PairCompute::patchData(int numParticles, MD_FLOAT* particleX, MD_FLOAT* particleY, MD_FLOAT* particleZ, MD_FLOAT* particleQ, int fromPatch) {

  // Copy the data from the parameters
  #if USE_PROXY_PATCHES != 0
    this->particleX[fromPatch] = particleX;
    this->particleY[fromPatch] = particleY;
    this->particleZ[fromPatch] = particleZ;
    this->particleQ[fromPatch] = particleQ;
  #else
    memcpy(this->particleX[fromPatch], particleX, numParticles * sizeof(MD_FLOAT));
    memcpy(this->particleY[fromPatch], particleY, numParticles * sizeof(MD_FLOAT));
    memcpy(this->particleZ[fromPatch], particleZ, numParticles * sizeof(MD_FLOAT));
    memcpy(this->particleQ[fromPatch], particleQ, numParticles * sizeof(MD_FLOAT));
  #endif

  // Increment the patch count and initiate the calculation of both patches have
  //   sent their data to this compute
  patchDataCount++;
  if (patchDataCount >= 2) {

    // DMK - DEBUG - Call doCalc function locally instead of sending a message
    #if CMK_CELL != 0 && 0
      CkIndex_PairCompute::_accelCall_spe_doCalc_void(this->numParticles,
                                                      this->thisIndex.x, this->thisIndex.y,
                                                      this->particleX[0], this->particleX[1],
                                                      this->particleY[0], this->particleY[1],
                                                      this->particleZ[0], this->particleZ[1],
                                                      this->particleQ[0], this->particleQ[1],
                                                      this->forceX[0], this->forceX[0],
                                                      this->forceY[0], this->forceY[0],
                                                      this->forceZ[0], this->forceZ[0],
                                                      this->localFlopCount,
                                                      this
                                                     );
    #else
      thisProxy(thisIndex.x, thisIndex.y).doCalc();  // Send message to self to do calculation
    #endif

    patchDataCount = 0;
  }
}


void PairCompute::doCalc_callback() {

  // DMK - DEBUG
  #if ENABLE_USER_EVENTS != 0
    double __start_time__ = CkWallTimer();
  #endif

  // DMK - DEBUG
  #if COUNT_FLOPS != 0
    globalFlopCount += localFlopCount;
  #endif

  #if USE_PROXY_PATCHES != 0

    (proxyPatchProxy[0].ckLocalBranch())->forceCheckIn(numParticles, forceX[0], forceY[0], forceZ[0]);
    (proxyPatchProxy[1].ckLocalBranch())->forceCheckIn(numParticles, forceX[1], forceY[1], forceZ[1]);

  #else

    // DMK - DEBUG
    NetworkProgress;

    // Calculate the index of patch 0 and send force data back to it
    int p0Index = thisIndex.x;
    int p0IndexX = PATCH_I_TO_X(p0Index);
    int p0IndexY = PATCH_I_TO_Y(p0Index);
    int p0IndexZ = PATCH_I_TO_Z(p0Index);
    patchArrayProxy(p0IndexX, p0IndexY, p0IndexZ).forceCheckIn(numParticles, forceX[0], forceY[0], forceZ[0]);

    // DMK - DEBUG
    NetworkProgress;

    // Calculate the index of patch 1 and send force data back to it
    int p1Index = thisIndex.y;
    int p1IndexX = PATCH_I_TO_X(p1Index);
    int p1IndexY = PATCH_I_TO_Y(p1Index);
    int p1IndexZ = PATCH_I_TO_Z(p1Index);
    patchArrayProxy(p1IndexX, p1IndexY, p1IndexZ).forceCheckIn(numParticles, forceX[1], forceY[1], forceZ[1]);

    // DMK - DEBUG
    NetworkProgress;

  #endif

  // DMK - DEBUG
  #if ENABLE_USER_EVENTS != 0
    double __end_time__ = CkWallTimer();
    traceUserBracketEvent(PROJ_USER_EVENT_PAIRCOMPUTE_DOCALC_CALLBACK, __start_time__, __end_time__);
  #endif
}


#include "pairCompute.def.h"
