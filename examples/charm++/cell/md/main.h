#ifndef __MAIN_H__
#define __MAIN_H__


#include "md_config.h"
#include "main.decl.h"


// Read-Only Variables
extern CProxy_Main mainProxy;
extern CProxy_Patch patchArrayProxy;
extern CProxy_SelfCompute selfComputeArrayProxy;
extern CProxy_PairCompute pairComputeArrayProxy;
extern int numPatchesX;
extern int numPatchesY;
extern int numPatchesZ;


// DMK - DEBUG
#if COUNT_FLOPS != 0
  extern unsigned long long int globalFlopCount;
#endif


class Main : public CBase_Main {

  private:

    /// Application Parameters ///
    int numParticlesPerPatch;

    /// Member Variables ///
    int numStepsRemaining;
    double simStartTime;
    double simPrevTime;

    /// Member Functions ///
    void parseCommandLine(int argc, char** argv);

  public:

    /// Constructor(s) \ Destructor ///
    Main(CkArgMsg* msg);
    ~Main();

    /// Entry Methods ///
    void initCheckIn();
    void patchCheckIn();

};


#endif //__MAIN_H__
