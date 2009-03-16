#ifndef __PAIR_COMPUTE_H__
#define __PAIR_COMPUTE_H__


#include "md_config.h"
#include "pairCompute.decl.h"
#include "patch.decl.h"


class PairCompute : public CBase_PairCompute {

  // Declare CkIndex_PairCompute as a friend so accelerated entry methods can access
  //   the member variables of the object they execute on
  friend class CkIndex_PairCompute;

  private:

    int numParticles;
    int patchDataCount;

    /// Particle Buffers ///
    float* particleX[2];
    float* particleY[2];
    float* particleZ[2];
    float* particleQ[2];

    /// Force Buffers ///
    float* forceX[2];
    float* forceY[2];
    float* forceZ[2];

    #if USE_PROXY_PATCHES != 0
      CProxy_ProxyPatch proxyPatchProxy[2];
    #endif

  public:

    /// Constructor(s) \ Destructor ///
    PairCompute();
    PairCompute(CkMigrateMessage *msg);
    ~PairCompute();

    /// Entry Methods ///
    void init(int numParticlesPerPatch);
    void patchData(int numParticles, float* particleX, float* particleY, float* particleZ, float* particleQ, int fromPatch, CProxy_ProxyPatch proxyPatchProxy);
    void patchData(int numParticles, float* particleX, float* particleY, float* particleZ, float* particleQ, int fromPatch);
    void doCalc_callback();

};


#endif //__PAIR_COMPUTE_H__
