
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

void TraceCore::LogEvent(int lID, int eID) 
{
	CmiPrintf("logging event (%d, %d)\n", lID, eID);
}

