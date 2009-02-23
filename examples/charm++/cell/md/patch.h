#ifndef __PATCH_H__
#define __PATCH_H__


#include "patch.decl.h"
#include "md_config.h"


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

    /// Member Functions ///
    void randomizeParticles();
    float randf();
    void startIteration_common(int numIters);

  public:

    /// Constructor(s) \ Destructor ///
    Patch(int numParticles);
    Patch(CkMigrateMessage* msg);
    ~Patch();

    /// Entry Methods ///
    void startIteration();
    void startIterations(int numIters);
    void forceCheckIn_callback();
    void integrate_callback();

};


#endif //__PATCH_H__
