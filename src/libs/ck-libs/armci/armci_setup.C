#include "armci_impl.h"

// This is the way to adapt a library's preferred start interface with the
// one provided by TCharm (eg. argc,argv vs void).
extern "C" void armciLibStart(void) {
  int argc=CkGetArgc();
  char **argv=CkGetArgv();
  armciStart(argc,argv);
}

// Default startup routine (can be overridden by user's own)
// This will be registered with TCharm's startup routine
// in the Node initialization function.
static void ArmciDefaultSetup(void) {
  // Create the base threads on TCharm using user-defined start routine.
  TCHARM_Create(TCHARM_Get_num_chunks(), armciLibStart);
}

// Bind the virtual processors in the armci library to TCharm's.
// This is called by the user's thread when it starts up.
CDECL int ARMCI_Init(void) {
  if (TCHARM_Element()==0) {
    CkArrayID _tc_aid;
    CkArrayOptions opt = TCHARM_Attach_start(&_tc_aid, NULL);
    CkArrayID aid = CProxy_ArmciVirtualProcessor::ckNew(_tc_aid, opt);
    CProxy_ArmciVirtualProcessor vpProxy = CProxy_ArmciVirtualProcessor(aid);
    // FIXME: should do reductions to element 0 of the array, not some bizarre malloc'd thing.
    CkArrayID *clientAid = new CkArrayID;
    *clientAid = aid;
    vpProxy.setReductionClient(mallocClient, (void *)clientAid);
  }
  
  ArmciVirtualProcessor *vp=(ArmciVirtualProcessor *)
  	TCharm::get()->semaGet(ARMCI_TCHARM_SEMAID);
  return 0;
}


CtvDeclare(ArmciVirtualProcessor *, _armci_ptr);

// Node initialization (made by initcall of the module armci)
void armciNodeInit(void) {
  CtvInitialize(ArmciVirtualProcessor, _armci_ptr);
  CtvAccess(_armci_ptr) = NULL;

  // Register the library's default startup routine to TCharm
  TCHARM_Set_fallback_setup(ArmciDefaultSetup);
};

#include "armci.def.h"
