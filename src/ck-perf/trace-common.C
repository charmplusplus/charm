/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <stdlib.h>
#include "trace.h"
#include "stdlib.h"

#define DEBUGF(x)          // CmiPrintf x

#define LogBufSize      10000

#ifdef CMK_OPTIMIZE
static int warned = 0;
#define OPTIMIZE_WARNING if (!warned) { warned=1;  CmiPrintf("\n\n!!!! Warning: tracing not available with CMK_OPTIMIZE!\n");  return;  }
#else
#define OPTIMIZE_WARNING /*empty*/
#endif

CpvDeclare(TraceArray*, _traces);

CpvDeclare(int, traceOn);
CpvDeclare(int, CtrLogBufSize);
CpvDeclare(char*, traceRoot);

extern "C" 
void traceCommonInit(char **argv)
{
  int i;
  DEBUGF(("[%d] in traceCommonInit.\n", CkMyPe()));
  CpvInitialize(int, traceOn);
  CpvInitialize(int, CtrLogBufSize);
  CpvInitialize(char*, traceRoot);
  CpvAccess(traceOn) = 0;
  CpvAccess(CtrLogBufSize) = LogBufSize;
  CmiGetArgInt(argv,"+logsize",&CpvAccess(CtrLogBufSize));
  char *root;
  if (CmiGetArgString(argv, "+trace-root", &root)) {
    int i;
    for (i=strlen(argv[0])-1; i>=0; i--) if (argv[0][i] == '/') break;
    i++;
    CpvAccess(traceRoot) = (char *)malloc(strlen(argv[0]+i) + strlen(root) + 2);    _MEMCHECK(CpvAccess(traceRoot));
    strcpy(CpvAccess(traceRoot), root);
    strcat(CpvAccess(traceRoot), "/");
    strcat(CpvAccess(traceRoot), argv[0]+i);
  }
  else {
    CpvAccess(traceRoot) = (char *) malloc(strlen(argv[0])+1);
    _MEMCHECK(CpvAccess(traceRoot));
    strcpy(CpvAccess(traceRoot), argv[0]);
  }
}

/*Install the beginIdle/endIdle condition handlers.*/
extern "C" void traceBegin(void) {
  OPTIMIZE_WARNING
  DEBUGF(("[%d] traceBegin called with %d\n", CkMyPe(), CpvAccess(traceOn)));
  if (CpvAccess(traceOn)==1) return;
  CpvAccess(_traces)->traceBegin();
  CpvAccess(traceOn) = 1;
}

/*Cancel the beginIdle/endIdle condition handlers.*/
extern "C" void traceEnd(void) {
  OPTIMIZE_WARNING
  if (CpvAccess(traceOn)==0) return;
  CpvAccess(_traces)->traceEnd();
  CpvAccess(traceOn) = 0;
}

void _createTraces();

extern "C" void traceInit(char **argv) 
{
  CpvInitialize(TraceArray *, _traces);
  CpvAccess(_traces) = new TraceArray;
  // in moduleInit.C
  _createTraces();

  traceCommonInit(argv);
  CpvAccess(_traces)->traceInit(argv);
  
  if (CpvAccess(_traces)->length() && !CmiGetArgFlag(argv,"+traceoff"))
    traceBegin();
}

extern "C"
void traceResume(void)
{
  CpvAccess(_traces)->beginExecute(0);
}

extern "C"
void traceSuspend(void)
{
  CpvAccess(_traces)->endExecute();
}

extern "C"
void traceAwaken(CthThread t)
{
  CpvAccess(_traces)->creation(0);
}

extern "C"
void traceUserEvent(int e)
{
  OPTIMIZE_WARNING
  CpvAccess(_traces)->userEvent(e);
}

extern "C"
int traceRegisterUserEvent(const char*x)
{
  return CpvAccess(_traces)->traceRegisterUserEvent(x);
}

extern "C"
void traceClearEps(void)
{
  OPTIMIZE_WARNING
  CpvAccess(_traces)->traceClearEps();
}

extern "C"
void traceWriteSts(void)
{
  OPTIMIZE_WARNING
  CpvAccess(_traces)->traceWriteSts();
}

extern "C"
void traceClose(void)
{
  OPTIMIZE_WARNING
  CpvAccess(_traces)->traceClose();
}
