#include "selfCompute.h"
#include "main.h"


SelfCompute::SelfCompute(int numParticlesPerPatch) {

  // Allocate buffers for force data
  numParticles = numParticlesPerPatch;
  forceX = (float*)(malloc_aligned(numParticles * sizeof(float), 128));
  forceY = (float*)(malloc_aligned(numParticles * sizeof(float), 128));
  forceZ = (float*)(malloc_aligned(numParticles * sizeof(float), 128));

  // Check in with the main chare indicating that this object is ready for the simulation to start
  mainProxy.proxyCheckIn();
}


SelfCompute::SelfCompute(CkMigrateMessage* msg) {
  CkAbort("SelfCompute::SelfCompute(CkMigrateMessage *msg) not implemented yet");
}


SelfCompute::~SelfCompute() {
  if (forceX != NULL) { free_aligned(forceX); forceX = NULL; }
  if (forceY != NULL) { free_aligned(forceY); forceY = NULL; }
  if (forceZ != NULL) { free_aligned(forceZ); forceZ = NULL; }
  numParticles = 0;
}


void SelfCompute::doCalc_callback() {
  patchArrayProxy(thisIndex.x, thisIndex.y, thisIndex.z).forceCheckIn(numParticles, forceX, forceY, forceZ);
}


#include "selfCompute.def.h"
