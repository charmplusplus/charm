/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "trace.h"
#include <stdlib.h>

CpvDeclare(Trace*, _trace);

extern "C"
void traceInit(char**argv)
{
  traceCommonInit(argv,0);
  CpvInitialize(Trace*, _trace);
  CpvAccess(_trace) = 0;
}

extern "C" void traceBeginIdle(void) {}
extern "C" void traceEndIdle(void) {}
extern "C" void traceResume(void) {}
extern "C" void traceSuspend(void) {}
extern "C" void traceAwaken(CthThread t) {}
extern "C" void traceUserEvent(int) {}
extern "C" int  traceRegisterUserEvent(const char*) {return 0;}
extern "C" void traceClearEps(void) {}
extern "C" void traceClose(void) {}

extern "C" void CkSummary_MarkEvent(int) {}
extern "C" void CkSummary_StartPhase(int) {}
