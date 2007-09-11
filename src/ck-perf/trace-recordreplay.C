/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkPerf
*/
/*@{*/

#include "charm++.h"
#include "trace-recordreplay.h"
#include "signal.h"

#define DEBUGF(x)  // CmiPrintf x

#define VER   4.0

#define INVALIDEP     -2

CkpvStaticDeclare(TraceRecordReplay*, _trace);

/**
  For each TraceFoo module, _createTraceFoo() must be defined.
  This function is called in _createTraces() generated in moduleInit.C
*/
void _createTracerecordreplay(char **argv)
{
  DEBUGF(("%d createTraceRecordReplay\n", CkMyPe()));
  CkpvInitialize(TraceRecordReplay*, _trace);
  CkpvAccess(_trace) = new  TraceRecordReplay(argv);
  CkpvAccess(_traces)->addTrace(CkpvAccess(_trace));
}

typedef void (*sigfunc)(int);
CkpvStaticDeclare(sigfunc, segfault_sig);

void segfault_signal(int sig) {
  printf("Segfault handler reached!\n");
  signal(SIGSEGV, CkpvAccess(segfault_sig));
}

TraceRecordReplay::TraceRecordReplay(char **argv):curevent(1)
{
  //CkpvAccess(segfault_sig) = signal(SIGSEGV, segfault_signal);
}

void TraceRecordReplay::beginExecute(envelope *e)
{
  // no message means thread execution
  if (e==NULL) {
  }
  else {
    e->setEvent(curevent++);
  }  
}


void TraceRecordReplay::creation(envelope *e, int ep, int num)
{
  e->setEvent(curevent++);
}


/*@}*/
