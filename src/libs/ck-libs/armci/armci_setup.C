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
  TCharmCreate(TCharmGetNumChunks(), armciLibStart);
  // Attach the array of TCShmemThreads to the corresponding TCharm threads.
  ARMCI_Attach();
}

// Bind the virtual processors in the armci library to TCharm's.
// This can be called by default using ArmciDefaultSetup or directly
// invoked by the user of the armci library if the user chooses to use
// his/her own startup routine and/or employ multi-module programs.
void ARMCI_Attach(void) {
  CkArrayID _tc_aid;
  CkArrayOptions opt = TCharmAttachStart(&_tc_aid, NULL);
  CkArrayID aid = CProxy_ArmciVirtualProcessor::ckNew(_tc_aid, opt);
  CProxy_ArmciVirtualProcessor vpProxy = CProxy_ArmciVirtualProcessor(aid);
  CkArrayID *clientAid = new CkArrayID;
  *clientAid = aid;
  vpProxy.setReductionClient(mallocClient, (void *)clientAid);
  TCharmAttachFinish(aid);
  armci_nproc = TCharmNumElements();
}


CtvDeclare(ArmciVirtualProcessor *, _armci_ptr);

// Node initialization (made by initcall of the module armci)
void armciNodeInit(void) {
  CtvInitialize(ArmciVirtualProcessor, _armci_ptr);
  CtvAccess(_armci_ptr) = NULL;

  // Register the library's default startup routine to TCharm
  TCharmSetFallbackSetup(ArmciDefaultSetup);
};

#include "armci.def.h"
