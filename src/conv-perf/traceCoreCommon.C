
#include "converse.h"
#include "traceCore.h"
#include "traceCoreCommon.h"

/* Trace Module Constants (Default Values) */
#define TRACE_CORE_BUFFER_SIZE 10000

/* Trace Storage and associated Structure */
CpvDeclare(int, _traceCoreOn);
CpvDeclare(double, _traceCoreInitTime);
CpvDeclare(char*, _traceCoreRoot);
CpvDeclare(int, _traceCoreBufferSize);
CpvDeclare(TraceCore*, _traceCore);

/* Initialize TraceCore Module */
//TODO decide parameters from command line
extern "C" void initTraceCore(char** argv)
{
  CpvInitialize(int, _traceCoreOn);
  	CpvAccess(_traceCoreOn) = 0;

  CpvInitialize(char*, _traceCoreRoot);
  	CpvAccess(_traceCoreRoot) = (char *) malloc(strlen(argv[0])+1);
  	_MEMCHECK(CpvAccess(_traceCoreRoot));
  	strcpy(CpvAccess(_traceCoreRoot), argv[0]);

  CpvInitialize(int, _traceCoreBufferSize);
	CpvAccess(_traceCoreBufferSize) = TRACE_CORE_BUFFER_SIZE;

  CpvInitialize(double, _traceCoreInitTime);
  	CpvAccess(_traceCoreInitTime) = TRACE_CORE_TIMER();

  CpvInitialize(TraceCore*, _traceCore);
  	CpvAccess(_traceCore) = new TraceCore();
}

/* End Core Trace Module */
//TODO
extern "C" void closeTraceCore() {}

/* Resume Core Trace Module */
//TODO
extern "C" void resumeTraceCore() {}

/* Tracing API */
extern "C" void RegisterLanguage(int lID)
{ CpvAccess(_traceCore)->RegisterLanguage(lID); }

extern "C" void RegisterEvent(int lID, int eID)
{ CpvAccess(_traceCore)->RegisterEvent(lID, eID); }

extern "C" void LogEvent(int lID, int eID)
{ CpvAccess(_traceCore)->LogEvent(lID, eID); }

extern "C" void LogEvent1(int lID, int eID, int iLen, int* iData)
{ CpvAccess(_traceCore)->LogEvent(lID, eID, iLen, iData); }

extern "C" void LogEvent2(int lID, int eID, int sLen, char* sData)
{ CpvAccess(_traceCore)->LogEvent(lID, eID, sLen, sData); }

extern "C" void LogEvent3(int lID, int eID, int iLen, int* iData, int sLen, char* sData)
{ CpvAccess(_traceCore)->LogEvent(lID, eID, iLen, iData, sLen, sData); }



