
#ifndef __TRACE_CORE_COMMON_H__
#define __TRACE_CORE_COMMON_H__

#include "converse.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Trace Storage and associated Structure */
CpvExtern(int, _traceCoreOn);
CpvExtern(double, _traceCoreInitTime);
CpvExtern(char*, _traceCoreRoot);
CpvExtern(int, _traceCoreBufferSize);

/* Trace Timer */
#define  TRACE_CORE_TIMER   CmiWallTimer
inline double TraceCoreTimer() { return TRACE_CORE_TIMER() - CpvAccess(_traceCoreInitTime); }

/* Initialize Core Trace Module */
void initTraceCore(char** argv);

/* End Core Trace Module */
void closeTraceCore();

/* Resume Core Trace Module */
void resumeTraceCore();

/* Tracing API */
void RegisterLanguage(int lID);
void RegisterEvent(int lID, int eID);
void LogEvent(int lID, int eID);
void LogEvent1(int lID, int eID, int iLen, int* iData);
void LogEvent2(int lID, int eID, int sLen, char* sData);
void LogEvent3(int lID, int eID, int iLen, int* iData, int sLen, char* sData);

#ifdef __cplusplus
}
#endif

#endif
