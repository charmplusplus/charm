#ifndef __TRACE_MEMORY_H__
#define __TRACE_MEMORY_H__

#include "charm++.h"
#include "trace.h"
#include "trace-common.h"
#include <errno.h>

/** A representant of a memory operation */

class MemEntry {
  friend class TraceMemory;
 private:
  int type;
  void *where;
  int size;
  int stackSize;
  
 public:  
  MemEntry();
  void write(FILE *fp);
  void set(int t, void *w, int s) {
    type = t;
    where = w;
    size = s;
    stackSize = 0;
  }
  void setStack(int ss, void **s) {
    stackSize = ss;
    memcpy(this+1, s, ss*sizeof(void*));
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
  bool recordStack;
  char *logBuffer;
  bool traceDisabled;

  void checkFlush(int add);
  void flush();
 public:
  TraceMemory(char **argv);
  
  void traceBegin();
  void traceClose();
  void malloc(void *where, int size, void **stack, int stackSize);
  void free(void *where, int size);
};

#endif
