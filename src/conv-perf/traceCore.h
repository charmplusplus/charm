#ifndef __TRACE_CORE_H__
#define __TRACE_CORE_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "converse.h"

/* Prototype Declarations */
class TraceCore; 
class TraceLogger; 
class LogEntry;
class LogPool;

/* Class Declarations */
class TraceCore 
{
  public:
	//TODO
	/*
	RegisterLanguage(LanuageID)	
	RegisterEvent(LanguageID, EventID, EventDataPrototype)
	LogEvent(LanguageID, EventID, EventData)
	*/
	void RegisterLanguage(int lID);
	void RegisterEvent(int lID, int eID);
	void LogEvent(int lID, int eID);
	void LogEvent(int lID, int eID, int iLen, int* iData);
	void LogEvent(int lID, int eID, int sLen, char* sData);
	void LogEvent(int lID, int eID, int iLen, int* iData, int sLen, char* sData);
};

//TODO
class TraceLogger 
{

};

//TODO
class LogEntry
{

};

//TODO
class LogPool
{

};

#endif
