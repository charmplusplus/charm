#ifndef __MAIN_H__
#define __MAIN_H__


#include "main.decl.h"
#include "md_config.h"


// DMK - DEBUG - Until a general Charm++ API for aligned memory buffers is defined,
//   simply use malloc_aligned and free_align.  These functions are defined for
//   net-linux-cell builds, but not general net-linux builds.  Declare them here
//   if this is not a net-linux-cell build (for now, in this example program, it
//   is not important if they do not actually align on non net-linux-cell builds).
#if ((!(defined(CMK_CELL))) || (CMK_CELL == 0))
  void* malloc_aligned(int size, int align) { return malloc(size); }
  void free_aligned(void* ptr) { free(ptr); }
#endif


// Read-Only Variables
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
