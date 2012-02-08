/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _CONV_TRACE_H
#define _CONV_TRACE_H

#include "converse.h"

/* 
 * These functions are called from Converse, and should be provided C binding
 * by the tracing strategies.
 */

void traceInit(char **argv);
void traceCharmInit(char **argv);	/* init trace module in ck */
void traceMessageRecv(char *msg, int pe);
void traceBeginIdle(void);
void traceEndIdle(void);
void traceResume(CmiObjId *);
void traceSuspend(void);
void traceAwaken(CthThread t);
void traceUserEvent(int);
void traceUserBracketEvent(int, double, double);
void traceUserSuppliedData(int);
void traceUserSuppliedBracketedNote(char *note, int eventID, double bt, double et);
void traceUserSuppliedNote(char*);
void traceMemoryUsage();
int  traceRegisterUserEvent(const char*, int e
#ifdef __cplusplus
=-1
#endif
);

#if CMK_SMP_TRACE_COMMTHREAD
int  traceBeginCommOp(char *msg);
void traceEndCommOp(char *msg);
void traceSendMsgComm(char *msg);
#endif
void traceChangeLastTimestamp(double ts);

/* Support for machine layers to register their user events to projections */
void registerMachineUserEventsFunction(void (*eventRegistrationFunc)());

int traceRegisterFunction(const char*, int idx
#ifdef __cplusplus
=-999
#endif
);
void traceBeginFuncIndexProj(int, char* file, int);
void traceEndFuncIndexProj(int);

void traceClose(void);
void traceCharmClose(void);          /* close trace in ck */
void traceBegin(void);
void traceEnd(void);
void traceWriteSts(void);
void traceFlushLog(void);

#if CMK_TRACE_ENABLED
CpvExtern(int, traceOn);
#define traceIsOn()  (CpvAccess(traceOn))
#else 
#define traceIsOn()  0
#endif

int  traceAvailable();

/* Comm thread tracing */
#if CMK_SMP_TRACE_COMMTHREAD
#define  TRACE_COMM_CREATION(time, msg)   \
                    if (traceBeginCommOp(msg)) {   \
                      traceChangeLastTimestamp(time);    \
                      traceSendMsgComm(msg);   \
                      traceEndCommOp(msg);    \
                    }
#define  TRACE_COMM_RECV(time, msg)   \
                    if (traceBeginCommOp(msg)) {   \
                      traceChangeLastTimestamp(time);    \
                      traceEndCommOp(msg);    \
                    }
#else
#define TRACE_COMM_CREATION(time, msg)
#define TRACE_COMM_RECV(time, msg)
#endif

#endif
