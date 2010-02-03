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
    MD_FLOAT* particleX[2];
    MD_FLOAT* particleY[2];
    MD_FLOAT* particleZ[2];
    MD_FLOAT* particleQ[2];

    /// Force Buffers ///
    MD_FLOAT* forceX[2];
    MD_FLOAT* forceY[2];
    MD_FLOAT* forceZ[2];

    #if USE_PROXY_PATCHES != 0
      CProxy_ProxyPatch proxyPatchProxy[2];
    #endif

    // DMK - DEBUG
    unsigned int localFlopCount;

  public:

    /// Constructor(s) \ Destructor ///
    PairCompute();
    PairCompute(CkMigrateMessage *msg);
    ~PairCompute();

    /// Entry Methods ///
    void init(int numParticlesPerPatch);
    void patchData(int numParticles, MD_FLOAT* particleX, MD_FLOAT* particleY, MD_FLOAT* particleZ, MD_FLOAT* particleQ, int fromPatch, CProxy_ProxyPatch proxyPatchProxy);
    void patchData(int numParticles, MD_FLOAT* particleX, MD_FLOAT* particleY, MD_FLOAT* particleZ, MD_FLOAT* particleQ, int fromPatch);
    void doCalc_callback();

};


#endif //__PAIR_COMPUTE_H__
