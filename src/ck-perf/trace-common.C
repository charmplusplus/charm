/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <stdlib.h>
#include "trace.h"
#include "stdlib.h"

#if CMK_OPTIMIZE
static int warned = 0;
#define OPTIMIZE_WARNING if (!warned) { warned=1;  CmiPrintf("\n\n!!!! Warning: tracing not available with CMK_OPTIMIZE!\n");  return;  }
#else
#define OPTIMIZE_WARNING /*empty*/
#endif

CpvDeclare(int, traceOn);

typedef struct {
  int cancel_beginIdle;
  int cancel_endIdle;
} trace_common_calls;
CpvStaticDeclare(trace_common_calls,tracecommon);

extern "C" 
void traceCommonInit(char **argv,int enabled)
{
  CpvInitialize(trace_common_calls,tracecommon);
  CpvInitialize(int, traceOn);
  CpvAccess(traceOn) = 0;
  if (enabled && !CmiGetArgFlag(argv,"+traceoff"))
    traceBegin();
}

/*Install the beginIdle/endIdle condition handlers.*/
extern "C" void traceBegin(void) {
  OPTIMIZE_WARNING
  if (CpvAccess(traceOn)==1) return;
  CpvAccess(tracecommon).cancel_beginIdle=
    CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,traceBeginIdle,0);
  CpvAccess(tracecommon).cancel_endIdle=
    CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_BUSY,traceEndIdle,0);
  CpvAccess(traceOn) = 1;
}

/*Cancel the beginIdle/endIdle condition handlers.*/
extern "C" void traceEnd(void) {
  OPTIMIZE_WARNING
  if (CpvAccess(traceOn)==0) return;
  CcdCancelCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,
	  CpvAccess(tracecommon).cancel_beginIdle);
  CcdCancelCallOnConditionKeep(CcdPROCESSOR_BEGIN_BUSY,
	  CpvAccess(tracecommon).cancel_endIdle);
  CpvAccess(traceOn) = 0;
}




