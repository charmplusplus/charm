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
  private:
	LogPool* logPool;

  public:
	TraceCore(char** argv);
	~TraceCore();

	void RegisterLanguage(int lID);	//TODO temporary
	void RegisterLanguage(int lID, char* lName);
	void RegisterEvent(int lID, int eID);
	void LogEvent(int lID, int eID);
	void LogEvent(int lID, int eID, int iLen, int* iData);
	void LogEvent(int lID, int eID, int sLen, char* sData);
	void LogEvent(int lID, int eID, int iLen, int* iData, int sLen, char* sData);
};

//TODO: probably not required at this point. Since we are writing only to file
class TraceLogger {};

class LogEntry
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

	LogEntry() {}
	LogEntry(int lID, int eID, double ts, int el, int* e, 
			 int il, int* i, int sl, char* s): 
			 languageID(lID), eventID(eID), timestamp(ts), 
			 eLen(el), entity(e), iLen(il), iData(i), sLen(sl), sData(s) {}
	LogEntry(int lID, int eID, double ts,
			 int il, int* i, int sl, char* s): 
			 languageID(lID), eventID(eID), timestamp(ts), 
			 eLen(0), entity(NULL), iLen(il), iData(i), sLen(sl), sData(s) {}
	~LogEntry();

    void *operator new(size_t s) {void*ret=malloc(s);_MEMCHECK(ret);return ret;}
    void *operator new(size_t, void *ptr) { return ptr; }
    void operator delete(void *ptr) { free(ptr); }
#ifdef WIN32
    void operator delete(void *, void *) { }
#endif

	void write(FILE* fp, int prevLID, int prevSeek, int nextLID, int nextSeek);
};

class LogPool
{
  private:
    int poolSize;
    int numEntries;
    LogEntry *pool;
	
#define	MAX_NUM_LANGUAGES  10			//NOTE: fixed temporarily

	int   numLangs;
	char *lName[MAX_NUM_LANGUAGES];		// Language Name 
    char *fName[MAX_NUM_LANGUAGES];		// File name
    FILE *fptrs[MAX_NUM_LANGUAGES];		// File pointer
    int   binary;

  public:
    LogPool(char* program, int b);
    ~LogPool();

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
