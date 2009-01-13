
#ifndef __TRACE_CORE_H__
#define __TRACE_CORE_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "converse.h"

#define	MAX_NUM_LANGUAGES  32			//NOTE: fixed temporarily



/* Prototype Declarations */
class TraceCore;
class TraceLogger;
class TraceEntry;

CpvCExtern(int, _traceCoreOn);
/*** structure of events ***/

struct TraceCoreEvent {
	int eID;
	struct TraceCoreEvent *next;
};

/* Class Declarations */
class TraceCore
{
  private:
	TraceLogger* traceLogger;
	void startPtc();
	void closePtc();
	FILE *fpPtc;	// File pointer for the ptc file
	struct TraceCoreEvent  *eventLists[MAX_NUM_LANGUAGES];
	int maxlID;
	int maxeID[MAX_NUM_LANGUAGES];
	int numLangs;
	int numEvents[MAX_NUM_LANGUAGES];
	int lIDList[MAX_NUM_LANGUAGES];
	char *lNames[MAX_NUM_LANGUAGES];
	int traceCoreOn;
  public:
	TraceCore(char** argv);
	~TraceCore();

	//TODO: some of these methods are for temporary use only
	void RegisterLanguage(int lID);
	void RegisterLanguage(int lID, const char* lName);
	void RegisterEvent(int lID, int eID);
	void LogEvent(int lID, int eID);
	void LogEvent(int lID, int eID, int iLen, const int* iData);
	void LogEvent(int lID, int eID, int iLen, const int* iData,double t);
	void LogEvent(int lID, int eID, int sLen, const char* sData);
	void LogEvent(int lID, int eID, int iLen, const int* iData, int sLen, const char* sData);

};

class TraceEntry
{
  public:
	int    languageID;
	int    eventID;
	double timestamp;
	int    eLen;
	int*   entity;
	int    iLen;
	int*   iData;
	int    sLen;
	char*  sData;

	TraceEntry() {}
	TraceEntry(int lID, int eID, double ts, int el, int* e,
			 int il, int* i, int sl, char* s):
			 languageID(lID), eventID(eID), timestamp(ts),
			 eLen(el), entity(e), iLen(il), iData(i), sLen(sl), sData(s) {}
	TraceEntry(int lID, int eID, double ts,
			 int il, int* i, int sl, char* s):
			 languageID(lID), eventID(eID), timestamp(ts),
			 eLen(0), entity(NULL), iLen(il), iData(i), sLen(sl), sData(s) {}
	TraceEntry(TraceEntry& te);
	~TraceEntry();

    void *operator new(size_t s) {void*ret=malloc(s);_MEMCHECK(ret);return ret;}
    void *operator new(size_t, void *ptr) { return ptr; }
    void operator delete(void *ptr) { free(ptr); }
#if defined(WIN32) || CMK_MULTIPLE_DELETE
    void operator delete(void *, void *) { }
#endif

	void write(FILE* fp, int prevLID, int prevSeek, int nextLID, int nextSeek);
};

class TraceLogger
{
  private:
    int poolSize;
    int numEntries;
    TraceEntry *pool;
    TraceEntry *buffer;

	int   numLangs;
 char *lName[MAX_NUM_LANGUAGES];		// Language Name
    char *fName[MAX_NUM_LANGUAGES];		// File name
    FILE *fptrs[MAX_NUM_LANGUAGES];		// File pointer

    int   binary;

	int lastWriteFlag;		// set when writing log to file at end of program
	int prevLID, prevSeek;	// for writing logs to file
	int isWriting;

  public:
    TraceLogger(char* program, int b);
    ~TraceLogger();

    void RegisterLanguage(int lID, const char* ln);

    void write(void);
    void writeBinary(void);
	void writeSts(void);

    void add(int lID, int eID, double timestamp, int iLen, int* iData, int sLen, char* sData);
    void initlogfiles();


  private:
	void openLogFiles();
	void closeLogFiles();
	void verifyFptrs();
	void flushLogFiles();

	char* pgm;
};

#endif
