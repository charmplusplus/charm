/*
  Blue.h -- header file to include for bluegene emulator program
  First written by Gengbin Zheng, gzheng@uiuc.edu on 2/20/2001
*/ 

#ifndef BLUEGENE_H
#define BLUEGENE_H

#include "converse.h"

#ifdef __cplusplus
extern "C" {
#endif

/* API data structures */
	/* define size of a work which helps communication threads schedule */
typedef enum WorkType {LARGE_WORK, SMALL_WORK} WorkType;

	/* user handler function prototype */
typedef void (*BgHandler) (char *);

/* API functions */
	/* get a bluegene node coordinate */
void BgGetXYZ(int *x, int *y, int *z);
	/* get and set blue gene cube size */
	/* set functions can only be called in user's BgGlobalInit code */
void BgGetSize(int *sx, int *sy, int *sz);
void BgSetSize(int sx, int sy, int sz);

	/* get and set number of worker and communication thread */
int  BgGetNumWorkThread();
void BgSetNumWorkThread(int num);
int  BgGetNumCommThread();
void BgSetNumCommThread(int num);

int  BgGetThreadID();

	/* get and set user-defined node private data */
char *BgGetNodeData();
void BgSetNodeData(char *data);

	/* register user defined function and get a handler, 
	   should only be called in BgGlobalInit() */
int  BgRegisterHandler(BgHandler h);

double BgGetTime();

	/* send packet functions */
void BgSendLocalPacket(int threadID, int handlerID, WorkType type, int numbytes, char* data);
void BgSendNonLocalPacket(int x, int y, int z, int threadID, int handlerID, WorkType type, int numbytes, char* data);
void BgSendPacket(int x, int y, int z, int threadID, int handlerID, WorkType type, int numbytes, char* data);

	/** shutdown the simulator */
void BgShutdown();

/* user defined functions called by bluegene */
	/*  called exactly once per process, used to check argc/argv,
	    setup bluegene simulation size, number of communication/worker 
	    threads and register user handler functions */
extern void BgEmulatorInit(int argc, char **argv);
	/* called every bluegene node to start the first job */
extern void BgNodeStart(int argc, char **argv);

#if defined(__cplusplus)
}
#endif

#endif
