
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h> /*<- for va_start & friends */
#include <string.h> /*<- for strlen */

/* CkPrintf/CkAbort support: */
extern "C" void CmiOutOfMemory(int nbytes) {
       fprintf(stderr,"Fatal error> %d\n",nbytes);
       abort();
}

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

extern "C" void *CmiTmpAlloc(int size) {return malloc(size);}
extern "C" void CmiTmpFree(void *p) {free(p);}
extern "C" void __cmi_assert(const char *errmsg) { CmiAbort(errmsg);}

extern "C" void *CmiAlloc(int size) { return malloc(size); }
extern "C" void CmiFree(void *blk) { return free(blk); }

