
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include "converse.h"
#include "traceCore.h"
#include "traceCoreCommon.h"

#include "converseEvents.h"	//TODO: remove this hack for REGISTER_CONVESE
#include "charmEvents.h"	//TODO: remove this hack for REGISTER_CHARM

CpvExtern(int, _traceCoreOn);
CpvExtern(double, _traceCoreInitTime);
CpvExtern(char*, _traceCoreRoot);
CpvExtern(int, _traceCoreBufferSize);
CpvExtern(TraceCore*, _traceCore);

/* Trace Timer */
#define  TRACE_CORE_TIMER   CmiWallTimer
inline double TraceCoreTimer() { return TRACE_CORE_TIMER() - CpvAccess(_traceCoreInitTime); }

/***************** Class TraceCore Definition *****************/
TraceCore::TraceCore(char** argv)
{
	int binary = CmiGetArgFlag(argv,"+binary-trace");
	traceLogger = new TraceLogger(CpvAccess(_traceCoreRoot), binary);

	REGISTER_CONVERSE
	REGISTER_CHARM
}

TraceCore::~TraceCore()
{ if(traceLogger) delete traceLogger; }

void TraceCore::RegisterLanguage(int lID, char* ln)
{ traceLogger->RegisterLanguage(lID, ln); }	

//TODO: currently these are dummy definitions
void TraceCore::RegisterEvent(int lID, int eID)
{ CmiPrintf("registering event (%d, %d)\n", lID, eID); }	

//TODO: only for compatibility with incomplete converse instrumentation
void TraceCore::LogEvent(int lID, int eID)
{ LogEvent(lID, eID, 0, NULL, 0, NULL); }

void TraceCore::LogEvent(int lID, int eID, int iLen, int* iData)
{ LogEvent(lID, eID, iLen, iData, 0, NULL); }

void TraceCore::LogEvent(int lID, int eID, int sLen, char* sData)
{ LogEvent(lID, eID, 0, NULL, sLen, sData); }

void TraceCore::LogEvent(int lID, int eID, int iLen, int* iData, int sLen, char* sData)
{
	CmiPrintf("lID: %d, eID: %d", lID, eID);
	if(iData != NULL) {
		CmiPrintf(" iData: ");
		for(int i=0; i<iLen; i++) { CmiPrintf("%d ", iData[i]); }
	}
	if(sData != NULL) {
		CmiPrintf("sData: %s", sData);
	}
	CmiPrintf("\n");


	traceLogger->add(lID, eID, TraceCoreTimer(), iLen, iData, sLen, sData); 
}

/***************** Class TraceEntry Definition *****************/
TraceEntry::TraceEntry(TraceEntry& te)
{
	languageID = te.languageID;
	eventID    = te.eventID;
	timestamp  = te.timestamp;
	eLen	   = te.eLen;
	entity	   = te.entity;
	iLen	   = te.iLen;
	iData	   = te.iData;
	sLen	   = te.sLen;
	sData	   = te.sData;
}

TraceEntry::~TraceEntry()
{
	if(entity) delete [] entity;
	if(iData)  delete [] iData;
	if(sData)  delete [] sData;
}

void TraceEntry::write(FILE* fp, int prevLID, int prevSeek, int nextLID, int nextSeek)
{
	//NOTE: no need to write languageID to file
	if(prevLID == 0 && nextLID ==0)
		fprintf(fp, "%d %f %d %d", eventID, timestamp, 0, 0); 
	else if(prevLID == 0 && nextLID !=0)
		fprintf(fp, "%d %f %d %d %d", eventID, timestamp, 0, nextLID, nextSeek); 
	else if(prevLID != 0 && nextLID ==0)
		fprintf(fp, "%d %f %d %d %d", eventID, timestamp, prevLID, prevSeek, 0); 
	else // if(prevLID != 0 && nextLID !=0)
		fprintf(fp, "%d %f %d %d %d %d", eventID, timestamp, prevLID, prevSeek, nextLID, nextSeek);

	fprintf(fp, " %d", eLen);
	if(eLen != 0) {
	  for(int i=0; i<eLen; i++) fprintf(fp, " %d", entity[i]);  
	}

	fprintf(fp, " %d", iLen);
	if(iLen != 0) {
	  for(int i=0; i<iLen; i++) fprintf(fp, " %d", iData[i]);  
	}

	if(sLen !=0) fprintf(fp, " %s\n", sData);
	else fprintf(fp, "\n");

	// free memory 
	if(entity) delete [] entity;
	if(iData)  delete [] iData;
	if(sData)  delete [] sData;
}

/***************** Class TraceLogger Definition *****************/
TraceLogger::TraceLogger(char* program, int b): 
	numLangs(1), numEntries(0), lastWriteFlag(0), prevLID(0), prevSeek(0) 
{
  binary = b;
  pool = new TraceEntry[CpvAccess(_traceCoreBufferSize)];
  poolSize = CpvAccess(_traceCoreBufferSize);

  pgm = new char[strlen(program)];
  sprintf(pgm, "%s", program);
  openLogFiles();
  closeLogFiles();
}

TraceLogger::~TraceLogger() 
{
  if(binary)
  { lastWriteFlag = 1; writeBinary(); }
  else
  { lastWriteFlag = 1; write(); }
  delete [] pool;
  delete [] fName;
}

void TraceLogger::RegisterLanguage(int lID, char* ln)
{
	numLangs++;

	lName[lID] = new char[strlen(ln)];
	sprintf(lName[lID], "%s", ln);

	char pestr[10]; sprintf(pestr, "%d", CmiMyPe());
	fName[lID] = new char[strlen(pgm)+1+strlen(pestr)+1+strlen(ln)+strlen(".log")];
	sprintf(fName[lID], "%s.%s.%s.log", pgm, pestr, ln);

	FILE* fp = NULL;
	do
  	{
    	fp = fopen(fName[lID], "w");
  	} while (!fp && (errno == EINTR || errno == EMFILE));
  	if(!fp) {
    	CmiAbort("Cannot open Projector Trace File for writing ... \n");
  	}
  	if(!binary) {
    	fprintf(fp, "PROJECTOR-RECORD: %s.%s\n", pestr, lName[lID]);
  	}
  	fclose(fp);
}

void TraceLogger::write(void)
{
 	openLogFiles();

	int currLID=0, nextLID=0;
	int pLID=0, nLID=0;
	int currSeek=0, nextSeek=0;
	int i;
  	for(i=0; i<numEntries-1; i++) {
		currLID  = pool[i].languageID;
		FILE* fp = fptrs[currLID];
		currSeek = ftell(fp); 
		nextLID  = pool[i+1].languageID;
		nextSeek = ftell(fptrs[nextLID]);

		pLID = ((prevLID==currLID)?0:prevLID);
		nLID = ((nextLID==currLID)?0:nextLID);
		pool[i].write(fp, pLID, prevSeek, nLID, nextSeek);

		prevSeek = currSeek; prevLID = currLID; 
  	}
	if(lastWriteFlag==1) {
		currLID  = pool[i].languageID;
		FILE* fp = fptrs[currLID];
		currSeek = ftell(fp); 
		nextLID  = nextSeek = 0;

		pLID = ((prevLID==currLID)?0:prevLID);
		nLID = ((nextLID==currLID)?0:nextLID);
		pool[i].write(fp, pLID, prevSeek, nLID, nextSeek);
	}

  	closeLogFiles();
}

//TODO
void TraceLogger::writeBinary(void) {};
//TODO
void TraceLogger::writeSts(void) {};

void TraceLogger::add(int lID, int eID, double timestamp, int iLen, int* iData, int sLen, char* sData)
{
  new (&pool[numEntries++]) TraceEntry(lID, eID, timestamp, iLen, iData, sLen, sData); 
  if(poolSize==numEntries) {
    double writeTime = TraceCoreTimer();

    if(binary) writeBinary(); 
	else 	   write();

	// move the last entry of pool to first position
    new (&pool[0]) TraceEntry(pool[numEntries-1]); 
    numEntries = 1;
	//TODO
    //new (&pool[numEntries++]) TraceEntry(0, BEGIN_INTERRUPT, writeTime);
    //new (&pool[numEntries++]) TraceEntry(0, END_INTERRUPT, TraceCoreTimer());
  }
}

void TraceLogger::openLogFiles()
{
  for(int i=1; i<numLangs; i++) {
	FILE* fp = NULL;
	do
  	{
    	fp = fopen(fName[i], "a");
  	} while (!fp && (errno == EINTR || errno == EMFILE));
  	if(!fp) {
    	CmiAbort("Cannot open Projector Trace File for writing ... \n");
  	}
	fptrs[i] = fp;
  }
}

void TraceLogger::closeLogFiles()
{
  for(int i=1; i<numLangs; i++)
	fclose(fptrs[i]);
}

