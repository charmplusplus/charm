#ifndef __TRACE_MEMORY_H__
#define __TRACE_MEMORY_H__

#include "charm++.h"
#include "trace.h"
#include "trace-common.h"
#include <errno.h>

/** A representant of a memory operation */

class MemEntry {
 private:
  int type;
  void *where;
  int size;
  
 public:  
  MemEntry();
  void write(FILE *fp);
  void set(int t, void *w, int s=0) {
    type = t;
    where = w;
    size = s;
  }
};

/**
   class to trace all memory related events. Currently works only in conjunction
   with "-memory charmdebug".
*/
class TraceMemory : public Trace {
 private:
  int firstTime;
  int logBufSize;
  int usedBuffer;
  MemEntry *logBuffer;
  void checkFlush();
  void flush();
 public:
  TraceMemory(char **argv);
  
  void traceClose();
  void malloc(void *where, int size);
  void free(void *where);
};

#endif
