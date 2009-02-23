#ifndef __MAIN_H__
#define __MAIN_H__


#include "main.decl.h"
#include "md_config.h"


// DMK - DEBUG
#if ((!(defined(CMK_CELL))) || (CMK_CELL == 0))
  void* malloc_aligned(int size, int align) { return malloc(size); }
  void free_aligned(void* ptr) { free(ptr); }
#endif


// Read-Onlys
extern CProxy_Main mainProxy;
extern CProxy_Patch patchArrayProxy;
extern CProxy_SelfCompute selfComputeArrayProxy;
extern CProxy_PairCompute pairComputeArrayProxy;
extern int numPatchesX;
extern int numPatchesY;
extern int numPatchesZ;


class Main : public CBase_Main {

  private:

    /// Application Parameters ///
    int numParticlesPerPatch;

    /// Member Variables ///
    int numStepsRemaining;
    double simStartTime;

    /// Member Functions ///
    void parseCommandLine(int argc, char** argv);

  public:

    /// Constructor(s) \ Destructor ///
    Main(CkArgMsg* msg);
    ~Main();

    /// Entry Methods ///
    void proxyCheckIn();
    void patchCheckIn();

};


#endif //__MAIN_H__
