/**
 * Serial version of common Charm++ routines:
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h> /*<- for va_start & friends */

extern "C" void CmiAbort(const char *why) {
	fprintf(stderr,"Fatal error> %s\n",why);
	abort();
}

extern "C" void CmiPrintf(const char *fmt, ...) {
        va_list p; va_start(p, fmt);
        vfprintf(stdout,fmt,p);
        va_end(p);
}
extern "C" void CmiError(const char *fmt, ...) {
        va_list p; va_start(p, fmt);
        vfprintf(stderr,fmt,p);
        va_end(p);
}

