#ifndef _CONV_TRACE_H
#define _CONV_TRACE_H

#include "converse.h"

/* 
 * These functions are called from Converse, and should be provided C binding
 * by the tracing strategies.
 */

void traceInit(int* argc, char **argv);
void traceBeginIdle(void);
void traceEndIdle(void);
void traceResume(void);
void traceSuspend(void);
void traceAwaken(void);
void traceUserEvent(int);
int  traceRegisterUserEvent(const char*);
void traceClose(void);

#ifndef CMK_OPTIMIZE
CpvExtern(int, traceOn);
#endif
CpvExtern(CthThread, curThread);

#endif
