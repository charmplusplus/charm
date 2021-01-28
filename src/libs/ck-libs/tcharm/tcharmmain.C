/*
Threaded Charm++ "Framework Framework" Startup:
This file controls the startup process when writing
to a tcharm-based language, like AMPI or the FEM framework.

Note that you could also start a bunch of AMPI threads
from a regular Charm program, which would use its own
main instead of this one.

Orion Sky Lawlor, olawlor@acm.org, 2003/6/20
 */
#include "tcharm_impl.h"
#include "tcharm.h"
#include "tcharmmain.decl.h"

//Tiny simple main chare
class TCharmMain : public Chare {
public:
  static void nodeInit(void) {
    TCHARM_User_node_setup();
    FTN_NAME(TCHARM_USER_NODE_SETUP,tcharm_user_node_setup)();
    _registerCommandLineOpt("+tcharm_nomig");
    _registerCommandLineOpt("+tcharm_nothread");
    _registerCommandLineOpt("+tcharm_nothreads");
    _registerCommandLineOpt("+tcharm_trace");
    _registerCommandLineOpt("+tcharm_stacksize");
    _registerCommandLineOpt("-vp");
    _registerCommandLineOpt("+vp");
    _registerCommandLineOpt("+tcharm_getarg");
  }
  
  TCharmMain(CkArgMsg *msg) {
    delete msg;
    
    TCHARM_Set_exit(); // Exit when done running these threads.
    
    /*Call user-overridable Fortran setup.
      If not overridden, this will call the overridable C setup,
      which unless overridden will call the library "fallback" setup,
      which usually starts a bunch of TCharm threads running
      something like "MPI_Main" (AMPI) or "driver" (FEM).
    */
    FTN_NAME(TCHARM_USER_SETUP,tcharm_user_setup)();
  }
};

#include "tcharmmain.def.h"
