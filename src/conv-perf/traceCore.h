
#ifndef __TRACE_CORE_H__
#define __TRACE_CORE_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "converse.h"

/* Prototype Declarations */
class TraceCore; 
class TraceLogger; 
class TraceEntry;

/* Class Declarations */
class TraceCore 
{
  private:
	TraceLogger* traceLogger;

  public:
	TraceCore(char** argv);
	~TraceCore();

	//TODO: some of these methods are for temporary use only
	void RegisterLanguage(int lID);	
	void RegisterLanguage(int lID, char* lName);
	void RegisterEvent(int lID, int eID);
	void LogEvent(int lID, int eID);
	void LogEvent(int lID, int eID, int iLen, int* iData);
	void LogEvent(int lID, int eID, int sLen, char* sData);
	void LogEvent(int lID, int eID, int iLen, int* iData, int sLen, char* sData);
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
#ifdef WIN32
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
	
#define	MAX_NUM_LANGUAGES  10			//NOTE: fixed temporarily

	int   numLangs;
	char *lName[MAX_NUM_LANGUAGES];		// Language Name 
    char *fName[MAX_NUM_LANGUAGES];		// File name
    FILE *fptrs[MAX_NUM_LANGUAGES];		// File pointer
    int   binary;

	int lastWriteFlag;		// set when writing log to file at end of program
	int prevLID, prevSeek;	// for writing logs to file

  public:
    TraceLogger(char* program, int b);
    ~TraceLogger();

	void RegisterLanguage(int lID, char* ln);

    void write(void);
    void writeBinary(void);
	void writeSts(void);

    void add(int lID, int eID, double timestamp, int iLen, int* iData, int sLen, char* sData);

  private:
	void openLogFiles();
	void closeLogFiles();

	char* pgm;
};

#endif
