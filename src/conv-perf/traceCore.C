
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include "converse.h"
#include "traceCore.h"
#include "traceCoreCommon.h"

/***************** Class TraceCore Definition *****************/
//TODO: currently these are dummy definitions

TraceCore::TraceCore(char** argv)
{
	int binary = CmiGetArgFlag(argv,"+binary-trace");
	logPool = new LogPool(CpvAccess(_traceCoreRoot), binary);
}

TraceCore::~TraceCore()
{
	if(logPool) delete logPool;
}

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


	logPool->add(lID, eID, TraceCoreTimer(), iLen, iData, sLen, sData); 
}

/***************** Class LogEntry Definition *****************/
LogEntry::~LogEntry()
{
	if(entity) delete [] entity;
	if(iData)  delete [] iData;
	if(sData)  delete [] sData;
}

void LogEntry::write(FILE* fp, int prevLID, int prevSeek, int nextLID, int nextSeek)
{
	if(prevLID == 0 && nextLID ==0)
		fprintf(fp, "%d %d %f %d %d", languageID, eventID, timestamp, 0, 0); 
	else if(prevLID == 0 && nextLID !=0)
		fprintf(fp, "%d %d %f %d %d %d", languageID, eventID, timestamp, 0, nextLID, nextSeek); 
	else if(prevLID != 0 && nextLID ==0)
		fprintf(fp, "%d %d %f %d %d %d", languageID, eventID, timestamp, prevLID, prevSeek, 0); 
	else // if(prevLID != 0 && nextLID !=0)
		fprintf(fp, "%d %d %f %d %d %d %d", languageID, eventID, timestamp, prevLID, prevSeek, nextLID, nextSeek);

	fprintf(fp, " %d", eLen);
	if(eLen != 0) {
	  for(int i=0; i<eLen; i++) fprintf(fp, " %d", entity[i]);  
	}

	fprintf(fp, " %d", iLen);
	if(iLen != 0) {
	  for(int i=0; i<iLen; i++) fprintf(fp, " %d", iData[i]);  
	}

	if(sLen !=0) fprintf(fp, " %s", sData);
}

/***************** Class LogPool Definition *****************/
LogPool::LogPool(char* program, int b): numLangs(0)
{
  binary = b;
  pool = new LogEntry[CpvAccess(_traceCoreBufferSize)];
  numEntries = 0;
  poolSize = CpvAccess(_traceCoreBufferSize);

  pgm = new char[strlen(program)];
  sprintf(pgm, "%s", program);
  openLogFiles();
  closeLogFiles();
}

LogPool::~LogPool() 
{
  if(binary) writeBinary();
  else       write();
  delete [] pool;
  delete [] fName;
}

void LogPool::RegisterLanguage(int lID, char* ln)
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
    	fp = fopen(fName[lID], "w+");
  	} while (!fp && (errno == EINTR || errno == EMFILE));
  	if(!fp) {
    	CmiAbort("Cannot open Projector Trace File for writing ... \n");
  	}
  	if(!binary) {
    	fprintf(fp, "PROJECTOR-RECORD: %s.%s\n", pestr, lName[lID]);
  	}
  	fclose(fp);
}

//TODO: incomplete - how to take care of first and last entry's prev and next fields
void LogPool::write(void)
{
 	openLogFiles();

	int currLID=0, prevLID=0, nextLID=0;
	int pLID =0, nLID =0;
	int currSeek=0, prevSeek=0, nextSeek=0;
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
	//pool[i].write(fptrs[pool[i].languageID], prevLID, prevSeek, 0, 0);	//NOTE: this is wrong ??
	//TODO how to write the last entry, we donot know about next entry ??

  	closeLogFiles();
}

//TODO
void LogPool::writeBinary(void) {};
void LogPool::writeSts(void) {};

void LogPool::add(int lID, int eID, double timestamp, int iLen, int* iData, int sLen, char* sData)
{
  new (&pool[numEntries++])
    LogEntry(lID, eID, timestamp, iLen, iData, sLen, sData); 
  if(poolSize==numEntries) {
    double writeTime = TraceCoreTimer();

    if(binary) writeBinary(); 
	else 	   write();

    numEntries = 0;
	//TODO
    //new (&pool[numEntries++]) LogEntry(0, BEGIN_INTERRUPT, writeTime);
    //new (&pool[numEntries++]) LogEntry(0, END_INTERRUPT, TraceCoreTimer());
  }
}

void LogPool::openLogFiles()
{
  for(int i=0; i<numLangs; i++) {
	FILE* fp = NULL;
	do
  	{
    	fp = fopen(fName[i], "w+");
  	} while (!fp && (errno == EINTR || errno == EMFILE));
  	if(!fp) {
    	CmiAbort("Cannot open Projector Trace File for writing ... \n");
  	}
	fptrs[i] = fp;
  }
}

void LogPool::closeLogFiles()
{
  for(int i=0; i<numLangs; i++)
	fclose(fptrs[i]);
}

