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
    MD_FLOAT* particleX;  // x-coordinate
    MD_FLOAT* particleY;  // y-coordinate
    MD_FLOAT* particleZ;  // z-coordinate
    MD_FLOAT* particleQ;  // charge
    MD_FLOAT* particleM;  // mass
    MD_FLOAT* velocityX;  // velocity x-component
    MD_FLOAT* velocityY;  // velocity y-component
    MD_FLOAT* velocityZ;  // velocity z-component
    MD_FLOAT* forceSumX;  // Buffer to sum of force x components from computes
    MD_FLOAT* forceSumY;  // Buffer to sum of force y components from computes
    MD_FLOAT* forceSumZ;  // Buffer to sum of force z components from computes

    #if USE_ARRAY_SECTIONS != 0
      CProxySection_PairCompute pairComputeArraySection_lower;
      CProxySection_PairCompute pairComputeArraySection_upper;
    #endif

    #if USE_PROXY_PATCHES != 0
      CProxy_ProxyPatch proxyPatchProxy;
      int* proxyPatchPresentFlags;
    #endif


    // DMK - DEBUG
    unsigned int localFlopCount;


    /// Member Functions ///
    void randomizeParticles();
    MD_FLOAT randf();
    void startIteration_common(int numIters);

  public:

    /// Constructor(s) \ Destructor ///
    Patch();
    Patch(CkMigrateMessage* msg);
    ~Patch();

    /// Entry Methods ///
    void init(int numParticles);
    void init(int numParticles, CProxy_ProxyPatch proxyPatchProxy);
    void init_common(int numParticles);
    void startIteration();
    void startIterations(int numIters);
    void forceCheckIn(int numParticles, MD_FLOAT* forceX, MD_FLOAT* forceY, MD_FLOAT* forceZ);
    void forceCheckIn(int numParticles, MD_FLOAT* forceX, MD_FLOAT* forceY, MD_FLOAT* forceZ, int numForceCheckIns);
    void integrate_callback();

};


class ProxyPatch : public CBase_ProxyPatch {

  // Declare CkIndex_ProxyPatch as a friend so accelerated entry methods can access
  //   the member variables of the object they execute on
  friend class CkIndex_ProxyPatch;

  private:

    /// Member Variables ///
    int numParticles;
    MD_FLOAT* particleX;
    MD_FLOAT* particleY;
    MD_FLOAT* particleZ;
    MD_FLOAT* particleQ;
    MD_FLOAT* forceSumX;
    MD_FLOAT* forceSumY;
    MD_FLOAT* forceSumZ;

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
    void patchData(int numParticles, MD_FLOAT* particleX, MD_FLOAT* particleY, MD_FLOAT* particleZ, MD_FLOAT* particleQ);
    void forceCheckIn(int numParticles, MD_FLOAT* forceX, MD_FLOAT* forceY, MD_FLOAT* forceZ);

};

#endif //__PATCH_H__
