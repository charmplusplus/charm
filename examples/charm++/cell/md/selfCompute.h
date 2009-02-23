#ifndef __SELF_COMPUTE_H__
#define __SELF_COMPUTE_H__


#include "selfCompute.decl.h"
#include "md_config.h"


class SelfCompute : public CBase_SelfCompute {

  // Declare CkIndex_SelfCompute as a friend so accelerated entry methods can access
  //   the member variables of the object they execute on
  friend class CkIndex_SelfCompute;

  private:

    /// Force Buffers ///
    int numParticles;
    float* forceX;
    float* forceY;
    float* forceZ;

  public:

    /// Constructor(s) \ Destructor ///
    SelfCompute(int numParticlesPerPatch);
    SelfCompute(CkMigrateMessage* msg);
    ~SelfCompute();

    /// Entry Methods ///
    void doCalc_callback();

};


#endif //__SELF_COMPUTE_H__
