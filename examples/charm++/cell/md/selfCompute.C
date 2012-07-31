#include "selfCompute.h"
#include "main.h"


SelfCompute::SelfCompute() {
  numParticles = -1;
  particleX = NULL;
  particleY = NULL;
  particleZ = NULL;
  particleQ = NULL;
  forceX = NULL;
  forceY = NULL;
  forceZ = NULL;
}


SelfCompute::SelfCompute(CkMigrateMessage* msg) {
  CkAbort("SelfCompute::SelfCompute(CkMigrateMessage *msg) not implemented yet");
}


SelfCompute::~SelfCompute() {
  #if USE_PROXY_PATCHES == 0
    if (particleX != NULL) { CmiFreeAligned(particleX); particleX = NULL; }
    if (particleY != NULL) { CmiFreeAligned(particleY); particleY = NULL; }
    if (particleZ != NULL) { CmiFreeAligned(particleZ); particleZ = NULL; }
    if (particleQ != NULL) { CmiFreeAligned(particleQ); particleQ = NULL; }
  #endif
  if (forceX != NULL) { CmiFreeAligned(forceX); forceX = NULL; }
  if (forceY != NULL) { CmiFreeAligned(forceY); forceY = NULL; }
  if (forceZ != NULL) { CmiFreeAligned(forceZ); forceZ = NULL; }
  numParticles = -1;
}


void SelfCompute::init(int numParticlesPerPatch) {

  // Allocate buffers for force data
  numParticles = numParticlesPerPatch;
  #if USE_PROXY_PATCHES == 0
    particleX = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
    particleY = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
    particleZ = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
    particleQ = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  #endif
  forceX = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  forceY = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));
  forceZ = (MD_FLOAT*)(CmiMallocAligned(numParticles * sizeof(MD_FLOAT), 128));

  // Check in with the main chare
  mainProxy.initCheckIn();
}


void SelfCompute::patchData(int numParticles, MD_FLOAT* particleX, MD_FLOAT* particleY, MD_FLOAT* particleZ, MD_FLOAT* particleQ, CProxy_ProxyPatch proxyPatchProxy) {
  #if USE_PROXY_PATCHES != 0
    this->proxyPatchProxy = proxyPatchProxy;
  #endif
  patchData(numParticles, particleX, particleY, particleZ, particleQ);
}

void SelfCompute::patchData(int numParticles, MD_FLOAT* particleX, MD_FLOAT* particleY, MD_FLOAT* particleZ, MD_FLOAT* particleQ) {

  // Copy the data from the parameters
  #if USE_PROXY_PATCHES != 0
    this->particleX = particleX;
    this->particleY = particleY;
    this->particleZ = particleZ;
    this->particleQ = particleQ;
  #else
    memcpy(this->particleX, particleX, numParticles * sizeof(MD_FLOAT));
    memcpy(this->particleY, particleY, numParticles * sizeof(MD_FLOAT));
    memcpy(this->particleZ, particleZ, numParticles * sizeof(MD_FLOAT));
    memcpy(this->particleQ, particleQ, numParticles * sizeof(MD_FLOAT));
  #endif

  // Initiate the calculation for this compute
  thisProxy(thisIndex.x, thisIndex.y, thisIndex.z).doCalc();
}


void SelfCompute::doCalc_callback() {

  // DMK - DEBUG
  #if ENABLE_USER_EVENTS != 0
    double __start_time__ = CkWallTimer();
  #endif

  // DMK - DEBUG
  #if COUNT_FLOPS != 0
    globalFlopCount += localFlopCount;
  #endif

  // DMK - DEBUG
  NetworkProgress;

  #if USE_PROXY_PATCHES != 0
    proxyPatchProxy.ckLocalBranch()->forceCheckIn(numParticles, forceX, forceY, forceZ);
  #else
    patchArrayProxy(thisIndex.x, thisIndex.y, thisIndex.z).forceCheckIn(numParticles, forceX, forceY, forceZ);
  #endif

  // DMK - DEBUG
  NetworkProgress;

  // DMK - DEBUG
  #if ENABLE_USER_EVENTS != 0
    double __end_time__ = CkWallTimer();
    traceUserBracketEvent(PROJ_USER_EVENT_SELFCOMPUTE_DOCALC_CALLBACK, __start_time__, __end_time__);
  #endif
}


#include "selfCompute.def.h"
