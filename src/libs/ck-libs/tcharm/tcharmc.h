/*
User-callable C API for TCharm library
Orion Sky Lawlor, olawlor@acm.org, 11/20/2001
*/
#ifndef __TCHARM_H
#define __TCHARM_H

#include "pup_c.h"

#ifdef __cplusplus
extern "C" {
#endif


/*User callbacks: you define these functions*/
void TCharmUserNodeSetup(void);
void TCharmUserSetup(void);


/**** Routines you can call from UserNodeSetup: ****/

/*Register readonly global variables-- these will be broadcast
after init and available in driver.*/
typedef void (*TCpupReadonlyGlobal)(pup_er p);
void TCharmReadonlyGlobals(TCpupReadonlyGlobal fn);

/**** Routines you can call from UserSetup: ****/

/*Set the size of the thread stack*/
void TCharmSetStackSize(int newStackSize);

/*Create a new array of threads, which will be bound to by subsequent libraries*/
typedef void (*TCharmThreadStartFn)(void);
void TCharmCreate(int nThreads,TCharmThreadStartFn threadFn);

/*As above, but pass along (arbitrary) data to thread*/
typedef void (*TCharmThreadDataStartFn)(void *threadData);
void TCharmCreateData(int nThreads,TCharmThreadDataStartFn threadFn,
		  void *threadData,int threadDataLen);

/*Get the unconsumed command-line arguments (C only; no Fortran)*/
char **TCharmArgv(void);
int TCharmArgc(void);

/*Get the number of chunks we expect based on the command line*/
int TCharmGetNumChunks(void);


/**** Routines you can call from the thread (driver) ****/
int TCharmElement(void);
int TCharmNumElements(void);

typedef void (*TCharmPupFn)(pup_er p,void *data);
int TCharmRegister(void *data,TCharmPupFn pfn);
void *TCharmGetUserdata(int id);
void TCharmMigrate(void);
void TCharmDone(void);


#ifdef __cplusplus
};
#endif
#endif /*def(thisHeader)*/

