#include "trace-memory.h"

#define DefaultBufferSize  1000000

#define DEBUGF(x) // CmiPrintf x

CkpvStaticDeclare(TraceMemory*, _trace);

/**
  For each TraceFoo module, _createTraceFoo() must be defined.
  This function is called in _createTraces() generated in moduleInit.C
*/
void _createTracememory(char **argv)
{
  DEBUGF(("%d createTraceMemory\n", CkMyPe()));
  CkpvInitialize(TraceMemory*, _trace);
  CkpvAccess(_trace) = new TraceMemory(argv);
  CkpvAccess(_traces)->addTrace(CkpvAccess(_trace));
}

MemEntry::MemEntry() : type(0), where(0), size(0) { }

void MemEntry::write(FILE *fp) {
  fprintf(fp, "%d %p", type, where);
  if (type == MEMORY_MALLOC) fprintf(fp, " %d", size);
  fprintf(fp, "\n");
}

TraceMemory::TraceMemory(char **argv) {
  usedBuffer = 0;
  firstTime = 1;
  logBufSize = DefaultBufferSize;
  if (CmiGetArgIntDesc(argv,"+memlogsize",&logBufSize, 
		       "Log entries to buffer per I/O")) {
    if (CkMyPe() == 0) {
      CmiPrintf("Trace: logsize: %d\n", logBufSize);
    }
  }
  logBuffer = new MemEntry[logBufSize];
}

inline void TraceMemory::checkFlush() {
  if (usedBuffer == logBufSize) {
    flush();
  }
}

inline void TraceMemory::flush() {
  char *mode;
  if (firstTime) mode = "w";
  else mode = "a";
  firstTime = 0;
  // flushing the logs
  char fname[1024];
  sprintf(fname, "memoryLog_%d", CkMyPe());
  FILE *fp;
  do {
    fp = fopen(fname, mode);
  } while (!fp && (errno == EINTR || errno == EMFILE));
  if (!fp) {
    CmiAbort("Cannot open file for Memory log writing\n");
  }
  for (int i=0; i<usedBuffer; ++i) logBuffer[i].write(fp);
  fclose(fp);
  usedBuffer = 0;
}

void TraceMemory::traceClose() {
  flush();
}

void TraceMemory::malloc(void *where, int size) {
  logBuffer[usedBuffer++].set(MEMORY_MALLOC, where, size);
  checkFlush();
}

void TraceMemory::free(void *where) {
  logBuffer[usedBuffer++].set(MEMORY_FREE, where);
  checkFlush();
}
