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

char **CmiBacktrace(int *nStackLevels) {
#define max_stack 64 /* trace back at most this many levels of the stack */
	void *stackPtrs[max_stack];
        *nStackLevels=backtrace(stackPtrs,max_stack);
        return backtrace_symbols(stackPtrs,*nStackLevels);
}

#else /*Backtrace not available-- use do-nothing verion*/
char **CmiBacktrace(int *nStackLevels) {
	*nStackLevels=0;
	return NULL;
}
#endif
