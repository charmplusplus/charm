/**
 Dummy versions of the Charm++/C++ routines in trace-common.C,
 used when you're *only* linking with Converse.

 FIXME: this file should be eliminated via a pluggable
 tracing architecture (projector?).
 
 Orion Sky Lawlor, olawlor@acm.org, 2003/3/27
*/
#include <stdlib.h>
#include "conv-trace.h"

CpvDeclare(int, traceOn); /* For threads.c */
CpvExtern(int, _traceCoreOn);   /* For cursed projector core */
int _threadEP=-123; /* for charmProjections.C */

void traceInit(char **argv) {
  CpvInitialize(int, traceOn);
  CpvInitialize(int, _traceCoreOn); 
  CpvAccess(traceOn)=0;
  CpvAccess(_traceCoreOn)=0; 
  initTraceCore(argv);
}
void traceMessageRecv(char *msg, int pe) {}
void traceBeginIdle(void) {}
void traceEndIdle(void) {}
void traceResume(CmiObjId *t) {}
void traceSuspend(void) {}
void traceAwaken(CthThread t) {}
void traceUserEvent(int i) {}
void traceUserBracketEvent(int a, double b, double c) {}
int traceRegisterUserEvent(const char* e, int f) { return -1; }
void traceClose(void) {}
void traceBegin(void) {}
void traceEnd(void) {}
void traceWriteSts(void) {}
int  traceAvailable() {return 0;}
