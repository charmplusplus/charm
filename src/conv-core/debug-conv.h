/*
 Interface to Converse portion of parallel debugger.
 Moved here from converse.h 4/10/2001 by Orion Sky Lawlor, olawlor@acm.org
 */
#ifndef __CMK_DEBUG_CONV_H
#define __CMK_DEBUG_CONV_H

#include "conv-ccs.h"

#ifdef __cplusplus
extern "C" {
#endif

CpvExtern(void *,debugQueue);

void CpdInit(void); 
void CpdFreeze(void);  
void CpdUnFreeze(void);
void CpdFreezeModeScheduler(void);
void Cpd_CmiHandleMessage(void *msg);

void CpdInitializeObjectTable();
void CpdInitializeHandlerArray();
void CpdInitializeBreakPoints();

    /*Note: these global symbol names are incredibly stupid.*/
#define MAX_NUM_HANDLERS 1000
typedef char* (*hndlrIDFunction)(char *);
typedef hndlrIDFunction handlerType[MAX_NUM_HANDLERS][2];
void handlerArrayRegister(int, hndlrIDFunction, hndlrIDFunction);

typedef int (*indirectionFunction)(char *);
typedef indirectionFunction indirectionType[MAX_NUM_HANDLERS];

typedef char* (*symbolTableFunction)(void);
typedef symbolTableFunction symbolTableType[MAX_NUM_HANDLERS];

void symbolTableFnArrayRegister(int hndlrID, int noOfBreakPoints,
				symbolTableFunction f, indirectionFunction g);
char* getSymbolTableInfo();
int isBreakPoint(char *msg);
int isEntryPoint(char *msg);
void setBreakPoints(char *);
char *getBreakPoints();

char* getObjectList();
char* getObjectContents(int);

void msgListCache();
void msgListCleanup();

char* genericViewMsgFunction(char *msg, int type);
char* getMsgListSched();
char* getMsgListPCQueue();
char* getMsgListFIFO();
char* getMsgListDebug();
char* getMsgContentsSched(int index);
char* getMsgContentsPCQueue(int index);
char* getMsgContentsFIFO(int index);
char* getMsgContentsDebug(int index);

#ifdef __cplusplus
};
#endif

#endif
