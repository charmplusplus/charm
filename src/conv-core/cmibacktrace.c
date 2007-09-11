/*
Stack Tracing, for debugging.
  
This routine lets you inspect the run-time stack
for the names of the routines that have been called up 
to this point.  It's useful for things like CkAbort
that are called when stuff has gone really wrong and 
you'd like to know what led up to this point.
These routines even seem to work from signal handlers, 
although I wouldn't count on that.

This file is intended to be #included whole by the autoconf
script and conv-core.c.

Orion Sky Lawlor, olawlor@acm.org, 8/20/2002
*/

#if CMK_USE_BACKTRACE
#  include <execinfo.h> /* for backtrace (GNU glibc header) */

/* Extract the function-return pointers listed in the stack
   up to this depth.
 */
void CmiBacktraceRecord(void **retPtrs,int nSkip,int *nLevels) {
	int i;
#define max_stack 64 /* trace back at most this many levels of the stack */
	void *stackPtrs[max_stack];
	nSkip++; /* don't trace this routine */
	*nLevels=backtrace(stackPtrs,nSkip+*nLevels)-nSkip;
	for (i=0;i<*nLevels;i++)
		retPtrs[i]=stackPtrs[nSkip+i];
}

/* Meant to be used for large stack traces, avoids copy */
void CmiBacktraceRecordHuge(void **retPtrs,int *nLevels) {
  *nLevels=backtrace(retPtrs,*nLevels);
}

/* Look up the names of these function pointers */
char **CmiBacktraceLookup(void **srcPtrs,int nLevels) {
	return backtrace_symbols(srcPtrs,nLevels);
}

#else /*Backtrace not available-- use do-nothing version*/
void CmiBacktraceRecord(void **retPtrs,int nSkip,int *nLevels) {
	*nLevels=0;
}

void CmiBacktraceRecordHuge(void **retPtrs,int nSkip,int *nLevels) {
    *nLevels=0;
}

/* Look up the names of these function pointers */
char **CmiBacktraceLookup(void **srcPtrs,int nLevels) {
	return NULL;
}
#endif
