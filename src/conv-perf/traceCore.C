
#include <stdio.h>
#include <stdlib.h>

#include "converse.h"
#include "traceCore.h"
#include "traceCoreCommon.h"

/* Class TraceCore Definition */
//TODO: currently these are dummy definitions
void TraceCore::RegisterLanguage(int lID)
{
	CmiPrintf("registering language (%d)\n", lID);
}	

void TraceCore::RegisterEvent(int lID, int eID)
{
	CmiPrintf("registering event (%d, %d)\n", lID, eID);
}	

//NOTE: only for compatibility with incomplete converse instrumentation
void TraceCore::LogEvent(int lID, int eID)
{
	CmiPrintf("logging event (%d, %d)\n", lID, eID);
}

void TraceCore::LogEvent(int lID, int eID, int iLen, int* iData)
{ LogEvent(int lID, int eID, int iLen, int* iData, 0, NULL); }

void TraceCore::LogEvent(int lID, int eID, int sLen, char* sData);
{ LogEvent(int lID, int eID, 0, NULL, int sLen, int* sData); }

void TraceCore::LogEvent(int lID, int eID, int iLen, int* iData, int sLen, char* sData)
{
	CmiPrintf("logging event (%d, %d)\n", lID, eID);
}

