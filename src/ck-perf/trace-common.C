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

#include <stdlib.h>
#include "trace.h"
#include "trace-common.h"
#include "stdlib.h"

#define DEBUGF(x)          // CmiPrintf x

#define LogBufSize      10000

#ifdef CMK_OPTIMIZE
static int warned = 0;
#define OPTIMIZE_WARNING if (!warned) { warned=1;  CmiPrintf("\n\n!!!! Warning: tracing not available with CMK_OPTIMIZE!\n");  return;  }
#else
#define OPTIMIZE_WARNING /*empty*/
#endif

CkpvDeclare(TraceArray*, _traces);

CkpvDeclare(double, traceInitTime);
CpvDeclare(int, traceOn);
#if CMK_TRACE_IN_CHARM
CkpvDeclare(int, traceOnPe);
#endif
CkpvDeclare(int, CtrLogBufSize);
CkpvDeclare(char*, traceRoot);

/// decide parameters from command line
static void traceCommonInit(char **argv)
{
  DEBUGF(("[%d] in traceCommonInit.\n", CkMyPe()));
  CkpvInitialize(double, traceInitTime);
  CkpvAccess(traceInitTime) = TRACE_TIMER();
  CpvInitialize(int, traceOn);
  CkpvInitialize(int, CtrLogBufSize);
  CkpvInitialize(char*, traceRoot);
  CpvAccess(traceOn) = 0;
#if CMK_TRACE_IN_CHARM
  CkpvInitialize(int, traceOnPe);
  CkpvAccess(traceOnPe) = 1;
#endif
  CkpvAccess(CtrLogBufSize) = LogBufSize;
  CmiGetArgInt(argv,"+logsize",&CkpvAccess(CtrLogBufSize));
  char *root;
  if (CmiGetArgString(argv, "+trace-root", &root)) {
    int i;
    for (i=strlen(argv[0])-1; i>=0; i--) if (argv[0][i] == '/') break;
    i++;
    CkpvAccess(traceRoot) = (char *)malloc(strlen(argv[0]+i) + strlen(root) + 2);    _MEMCHECK(CkpvAccess(traceRoot));
    strcpy(CkpvAccess(traceRoot), root);
    strcat(CkpvAccess(traceRoot), "/");
    strcat(CkpvAccess(traceRoot), argv[0]+i);
  }
  else {
    CkpvAccess(traceRoot) = (char *) malloc(strlen(argv[0])+1);
    _MEMCHECK(CkpvAccess(traceRoot));
    strcpy(CkpvAccess(traceRoot), argv[0]);
  }
}

/*Install the beginIdle/endIdle condition handlers.*/
extern "C" void traceBegin(void) {
  OPTIMIZE_WARNING
  DEBUGF(("[%d] traceBegin called with %d\n", CkMyPe(), CpvAccess(traceOn)));
  if (CpvAccess(traceOn)==1) return;
  CkpvAccess(_traces)->traceBegin();
  CpvAccess(traceOn) = 1;
}

/*Cancel the beginIdle/endIdle condition handlers.*/
extern "C" void traceEnd(void) {
  OPTIMIZE_WARNING
  if (CpvAccess(traceOn)==0) return;
  CkpvAccess(_traces)->traceEnd();
  CpvAccess(traceOn) = 0;
}

/// defined in moduleInit.C
void _createTraces(char **argv);

/**
    traceInit: 		called at Converse level
    traceCharmInit:	called at Charm++ level
*/
/// initialize trace framework, also create the trace module(s).
static inline void _traceInit(char **argv) 
{
  CkpvInitialize(TraceArray *, _traces);
  CkpvAccess(_traces) = new TraceArray;

  // common init
  traceCommonInit(argv);

  // in moduleInit.C
  _createTraces(argv);

  if (CkpvAccess(_traces)->length() && !CmiGetArgFlag(argv,"+traceoff"))
    traceBegin();
}

/// Converse version
extern "C" void traceInit(char **argv) 
{
#if ! CMK_TRACE_IN_CHARM
  _traceInit(argv);
#endif
}

/// Charm++ version
extern "C" void traceCharmInit(char **argv) 
{
#if CMK_TRACE_IN_CHARM
  _traceInit(argv);
#endif
}

// CMK_OPTIMIZE is already guarded in convcore.c
extern "C"
void traceMessageRecv(char *msg, int pe)
{
#if ! CMK_TRACE_IN_CHARM
  CkpvAccessOther(_traces, CmiRankOf(pe))->messageRecv(msg, pe);
#endif
}

// CMK_OPTIMIZE is already guarded in convcore.c
extern "C"
void traceResume(void)
{
#if ! CMK_TRACE_IN_CHARM
    CkpvAccess(_traces)->beginExecute(0);
#endif
}

extern "C"
void traceSuspend(void)
{
#if ! CMK_TRACE_IN_CHARM
  CkpvAccess(_traces)->endExecute();
#endif
}

extern "C"
void traceAwaken(CthThread t)
{
#if ! CMK_TRACE_IN_CHARM
  CkpvAccess(_traces)->creation(0);
#endif
}

extern "C"
void traceUserEvent(int e)
{
#ifndef CMK_OPTIMIZE
  CkpvAccess(_traces)->userEvent(e);
#endif
}

extern "C"
int traceRegisterUserEvent(const char*x)
{
#ifndef CMK_OPTIMIZE
  return CkpvAccess(_traces)->traceRegisterUserEvent(x);
#else
  return 0;
#endif
}

extern "C"
void traceClearEps(void)
{
  OPTIMIZE_WARNING
  CkpvAccess(_traces)->traceClearEps();
}

extern "C"
void traceWriteSts(void)
{
  OPTIMIZE_WARNING
  CkpvAccess(_traces)->traceWriteSts();
}

/**
    traceClose: 	this function is called at Converse
    traceCharmClose:	called at Charm++ level
*/
extern "C"
void traceClose(void)
{
#if ! CMK_BLUEGENE_CHARM
  OPTIMIZE_WARNING
  CkpvAccess(_traces)->traceClose();
#endif
}

extern "C"
void traceCharmClose(void)
{
#if CMK_BLUEGENE_CHARM
  OPTIMIZE_WARNING
  CkpvAccess(_traces)->traceClose();
#endif
}

#if 0
// helper functions
int CkIsCharmMessage(char *msg)
{
//CmiPrintf("getMsgtype: %d %d %d %d %d\n", ((envelope *)msg)->getMsgtype(), CmiGetHandler(msg), CmiGetXHandler(msg), _charmHandlerIdx, index_skipCldHandler);
  if ((CmiGetHandler(msg) == _charmHandlerIdx) &&
         (CmiGetHandlerFunction(msg) == (CmiHandler)_processHandler))
    return 1;
  if (CmiGetXHandler(msg) == _charmHandlerIdx) return 1;
  return 0;
}
#endif

/*@}*/
