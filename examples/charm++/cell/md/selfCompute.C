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
    if (particleX != NULL) { free_aligned(particleX); particleX = NULL; }
    if (particleY != NULL) { free_aligned(particleY); particleY = NULL; }
    if (particleZ != NULL) { free_aligned(particleZ); particleZ = NULL; }
    if (particleQ != NULL) { free_aligned(particleQ); particleQ = NULL; }
  #endif
  if (forceX != NULL) { free_aligned(forceX); forceX = NULL; }
  if (forceY != NULL) { free_aligned(forceY); forceY = NULL; }
  if (forceZ != NULL) { free_aligned(forceZ); forceZ = NULL; }
  numParticles = -1;
}


void SelfCompute::init(int numParticlesPerPatch) {

  // Allocate buffers for force data
  numParticles = numParticlesPerPatch;
  #if USE_PROXY_PATCHES == 0
    particleX = (float*)(malloc_aligned(numParticles * sizeof(float), 128));
    particleY = (float*)(malloc_aligned(numParticles * sizeof(float), 128));
    particleZ = (float*)(malloc_aligned(numParticles * sizeof(float), 128));
    particleQ = (float*)(malloc_aligned(numParticles * sizeof(float), 128));
  #endif
  forceX = (float*)(malloc_aligned(numParticles * sizeof(float), 128));
  forceY = (float*)(malloc_aligned(numParticles * sizeof(float), 128));
  forceZ = (float*)(malloc_aligned(numParticles * sizeof(float), 128));

  // Check in with the main chare
  mainProxy.initCheckIn();
}


void SelfCompute::patchData(int numParticles, float* particleX, float* particleY, float* particleZ, float* particleQ, CProxy_ProxyPatch proxyPatchProxy) {
  #if USE_PROXY_PATCHES != 0
    this->proxyPatchProxy = proxyPatchProxy;
  #endif
  patchData(numParticles, particleX, particleY, particleZ, particleQ);
}

void SelfCompute::patchData(int numParticles, float* particleX, float* particleY, float* particleZ, float* particleQ) {

  // Copy the data from the parameters
  #if USE_PROXY_PATCHES != 0
    this->particleX = particleX;
    this->particleY = particleY;
    this->particleZ = particleZ;
    this->particleQ = particleQ;
  #else
    memcpy(this->particleX, particleX, numParticles * sizeof(float));
    memcpy(this->particleY, particleY, numParticles * sizeof(float));
    memcpy(this->particleZ, particleZ, numParticles * sizeof(float));
    memcpy(this->particleQ, particleQ, numParticles * sizeof(float));
  #endif

  // Initiate the calculation for this compute
  thisProxy(thisIndex.x, thisIndex.y, thisIndex.z).doCalc();
}


void SelfCompute::doCalc_callback() {

  // DMK - DEBUG
  #if ENABLE_USER_EVENTS != 0
    double __start_time__ = CmiWallTimer();
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
    double __end_time__ = CmiWallTimer();
    traceUserBracketEvent(PROJ_USER_EVENT_SELFCOMPUTE_DOCALC_CALLBACK, __start_time__, __end_time__);
  #endif
}


#include "selfCompute.def.h"
