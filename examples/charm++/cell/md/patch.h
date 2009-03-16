#ifndef __PATCH_H__
#define __PATCH_H__


#include "md_config.h"
#include "patch.decl.h"
#include "pairCompute.decl.h"


class Patch : public CBase_Patch {

  // Declare CkIndex_Patch as a friend so accelerated entry methods can access
  //   the member variables of the object they execute on
  friend class CkIndex_Patch;

  private:

    /// Member Variables ///
    int remainingForceCheckIns;
    int remainingIterations;

    int numParticles;
    float* particleX;  // x-coordinate
    float* particleY;  // y-coordinate
    float* particleZ;  // z-coordinate
    float* particleQ;  // charge
    float* particleM;  // mass
    float* velocityX;  // velocity x-component
    float* velocityY;  // velocity y-component
    float* velocityZ;  // velocity z-component
    float* forceSumX;  // Buffer to sum of force x components from computes
    float* forceSumY;  // Buffer to sum of force y components from computes
    float* forceSumZ;  // Buffer to sum of force z components from computes

    #if USE_ARRAY_SECTIONS != 0
      CProxySection_PairCompute pairComputeArraySection_lower;
      CProxySection_PairCompute pairComputeArraySection_upper;
    #endif

    #if USE_PROXY_PATCHES != 0
      CProxy_ProxyPatch proxyPatchProxy;
      int* proxyPatchPresentFlags;
    #endif

    /// Member Functions ///
    void randomizeParticles();
    float randf();
    void startIteration_common(int numIters);

  public:

    /// Constructor(s) \ Destructor ///
    Patch();
    Patch(CkMigrateMessage* msg);
    ~Patch();

    /// Entry Methods ///
    void init(int numParticles);
    void startIteration();
    void startIterations(int numIters);
    void forceCheckIn(int numParticles, float* forceX, float* forceY, float* forceZ);
    void forceCheckIn(int numParticles, float* forceX, float* forceY, float* forceZ, int numForceCheckIns);
    void integrate_callback();

};


class ProxyPatch : public CBase_ProxyPatch {

  // Declare CkIndex_ProxyPatch as a friend so accelerated entry methods can access
  //   the member variables of the object they execute on
  friend class CkIndex_ProxyPatch;

  private:

    /// Member Variables ///
    int numParticles;
    float* particleX;
    float* particleY;
    float* particleZ;
    float* particleQ;
    float* forceSumX;
    float* forceSumY;
    float* forceSumZ;

    int patchIndex;

    int checkInCount;
    int numToCheckIn;

  public:

    /// Constructor(s) \ Destructor ///
    ProxyPatch(int proxyIndex);
    ProxyPatch(CkMigrateMessage *msg);
    ~ProxyPatch();

    /// Entry Methods ///
    void init(int numParticles);
    void patchData(int numParticles, float* particleX, float* particleY, float* particleZ, float* particleQ);
    void forceCheckIn(int numParticles, float* forceX, float* forceY, float* forceZ);

};

#endif //__PATCH_H__
