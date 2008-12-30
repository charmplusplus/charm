#ifndef BIGSIM_OOO_H
#define BIGSIM_OOO_H

#include "blue_impl.h"

extern int bgUseOutOfCore;

/* This variable indicates the max memory all cores can occupy in a physical processor */
extern double bgOOCMaxMemSize;
/* System should always keep LEASTMEMRATE*bgOOCMaxMemSize memory for remaining simulations */
#define LEASTMEMRATE 0.01

/* Declarations related with out-of-core scheduling */
#define MOSTRECENTACCESSED 1000

struct threadInMemEntry
{
    threadInfo *thd;
    //int threadID; //global thread ID; -1 means this entry is empty

    /*
     * parameter deciding which thread to be put to disk.
     * Currently, this simulates LRU policy.
     * When some thread (say A) is brought into memory, this parameter for A is set
     * to MOSTRECENTACCESSED and other threads' will be decreased by 1   
     */
    int useFreq;
    //may be used later
    double thdSize;

    threadInMemEntry *nextEntry;

    threadInMemEntry(){
        thd = NULL;
        //useFreq = MOSTRECENTACCESSED; 
        useFreq = -1;

        thdSize = 0.0;
        nextEntry = NULL;
    }

    void initSelf(){
        thd = NULL;
        //useFreq = MOSTRECENTACCESSED; 
        useFreq = -1;

        thdSize = 0.0;
        nextEntry = NULL;
    }
    /*threadInMemEntry(){
        thd = NULL;
        useFreq = -1;
    }*/
};
//threadInMemEntry;

extern int TBLCAPACITY;

extern threadInMemEntry *tblThreadInMemHead;
//extern threadInMemEntry *tblThreadInMemTail;
//extern int tblThreadInMemActualSize;

//Initialize the table that keeps which threads are in the memory
//The table size is initially allocated to the half size of TBLCAPACITY
void initTblThreadInMem();

void deInitTblThreadInMem();

//if found return the entry of this thread, otherwise return NULL
threadInMemEntry *checkThreadInCore(/*int threadID*/threadInfo *thd);

/*
 * Gives one empty entry for the thread which is going to be put into memory.
 * This includes following actions:
 * 1. Check tblThreadInMem for an empty entry
 * 2. If none is found, evict one entry based on useFreq and then take that thread out of memory
 */
threadInMemEntry *giveEmptyThdInMemEntry();

threadInMemEntry *findLeastUsedThdInMemEntry();
threadInMemEntry *detachLeastUsedThdInMemEntry();

/* Set the thread indicated by threadID to the most recently accessed one */
void updateThdInMemTable(/*int threadID*/threadInfo *thd);

/* printing tblThdInMem */
void printTblThdInMem();


void bgOutOfCoreSchedule(threadInfo *thd);
//The following two functions are now embeded into the class workThreadInfo
//void bringThdIntoMem(/*int threadID*/threadInfo *thd);
//void takeThdOutofMem(/*int threadID*/threadInfo *thd);

//functions related with getting memory usage information
#define MEMINFOFILE "/proc/meminfo"
//The /proc/meminfo file is in the following format:
//MemTotal:  <size>kB
//MemFree:   <size>kB

//the unit is MB
double bgGetSysTotalMemSize();

double bgGetSysFreeMemSize();

//Get the process memory usage information from the /proc/<PID>/statm file
//The file is arranged in the format: <program size> <resident set size> ...
//The second filed means how many physical pages this process occupies.
//So to get the physical mmeory usage of this process, we also need to know
//the memory page size which could be obtained from getpagesize() sys call.

extern int bgMemPageSize;
extern char bgMemStsFile[25];

double bgGetProcessMemUsage();

#endif
