/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "trace.h"

CpvDeclare(Trace*, _trace);
CpvDeclare(int, traceOn);

extern "C"
void traceInit(int*, char**)
{
  CpvInitialize(Trace*, _trace);
  CpvAccess(_trace) = 0;
  CpvInitialize(int,traceOn);
  CpvAccess(traceOn) = 0;
}

extern "C" void traceBeginIdle(void) {}
extern "C" void traceEndIdle(void) {}
extern "C" void traceResume(void) {}
extern "C" void traceSuspend(void) {}
extern "C" void traceAwaken(void) {}
extern "C" void traceUserEvent(int) {}
extern "C" int  traceRegisterUserEvent(const char*) {return 0;}
extern "C" void traceClearEps(void) {}
extern "C" void traceClose(void) {}
