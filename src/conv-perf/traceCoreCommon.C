
#include "traceCore.h"
#include "traceCoreAPI.h"
#include "traceCoreCommon.h"
#include "charmProjections.h"
//#include "ampiProjections.h"
#include "converse.h"


/* Trace Module Constants (Default Values) */
#define TRACE_CORE_BUFFER_SIZE 10

/* Trace Storage and associated Structure */
extern "C" {
CpvDeclare(int, _traceCoreOn);
}
CpvDeclare(double, _traceCoreInitTime);
CpvDeclare(char*, _traceCoreRoot);
CpvDeclare(int, _traceCoreBufferSize);
CpvDeclare(TraceCore*, _traceCore);

/* Trace Timer */
#define  TRACE_CORE_TIMER   CmiWallTimer
inline double TraceCoreTimer() { return TRACE_CORE_TIMER() - CpvAccess(_traceCoreInitTime); }

/*****************************************************************/
/* Tracing API 
 * Implementation of functions declared in traceCoreCommon.h 
 *****************************************************************/
/* Initialize TraceCore Module */
//TODO decide parameters from command line
//TODO - trace-common.C
extern "C" void initTraceCore(char** argv)
{
  /*CpvInitialize(int, _traceCoreOn);
  	CpvAccess(_traceCoreOn) = 0;*/

  CpvInitialize(char*, _traceCoreRoot);
  	CpvAccess(_traceCoreRoot) = (char *) malloc(strlen(argv[0])+1);
  	_MEMCHECK(CpvAccess(_traceCoreRoot));
  	strcpy(CpvAccess(_traceCoreRoot), argv[0]);

  CpvInitialize(int, _traceCoreBufferSize);
	CpvAccess(_traceCoreBufferSize) = TRACE_CORE_BUFFER_SIZE;

  CpvInitialize(double, _traceCoreInitTime);
  	CpvAccess(_traceCoreInitTime) = TRACE_CORE_TIMER();

  CpvInitialize(TraceCore*, _traceCore);
  	CpvAccess(_traceCore) = new TraceCore(argv);
 // initCharmProjections();
 // initAmpiProjections();
}

/* End Core Trace Module */
//TODO - trace-common.C
extern "C" void closeTraceCore() {
	//closeAmpiProjections();
	delete CpvAccess(_traceCore);
}

/* Resume Core Trace Module */
//TODO - trace-common.C
extern "C" void resumeTraceCore() {}

/* Suspend Core Trace Module */
//TODO - trace-common.C
extern "C" void suspendTraceCore() {}

/*Install the beginIdle/endIdle condition handlers.*/
//TODO - trace-common.C
extern "C" void beginTraceCore(void) {}

/*Cancel the beginIdle/endIdle condition handlers.*/
//TODO - trace-common.C
extern "C" void endTraceCore(void) {}

/*****************************************************************/
/* Tracing API 
 * Implementation of functions declared in traceCoreAPI.h 
 *****************************************************************/
extern "C" void RegisterLanguage(int lID, const char* ln)
{ LOGCONDITIONAL(CpvAccess(_traceCore)->RegisterLanguage(lID, ln)); }

extern "C" void RegisterEvent(int lID, int eID)
{ LOGCONDITIONAL(CpvAccess(_traceCore)->RegisterEvent(lID, eID)); }

/* These Log routines will segfault if called under ! CMK_TRACE_ENABLED;
   the solution is to surround their callers with LOGCONDITIONAL. */
extern "C" void LogEvent(int lID, int eID)
{ CpvAccess(_traceCore)->LogEvent(lID, eID); }

extern "C" void LogEvent1(int lID, int eID, int iLen, const int* iData)
{ CpvAccess(_traceCore)->LogEvent(lID, eID, iLen, iData); }

extern "C" void LogEvent2(int lID, int eID, int sLen, const char* sData)
{ CpvAccess(_traceCore)->LogEvent(lID, eID, sLen, sData); }

extern "C" void LogEvent3(int lID, int eID, int iLen, const int* iData, int sLen, const char* sData)
{ CpvAccess(_traceCore)->LogEvent(lID, eID, iLen, iData, sLen, sData); }

extern "C" void LogEvent4(int lID, int eID, int iLen, const int* iData, double t)
{ CpvAccess(_traceCore)->LogEvent(lID, eID, iLen, iData,t); }

