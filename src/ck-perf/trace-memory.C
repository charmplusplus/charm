#include "trace-memory.h"

#define DefaultBufferSize  10000

#define DEBUGF(x) // CmiPrintf x

CkpvStaticDeclare(TraceMemory*, _trace);
extern "C" void memory_trace_all_existing_mallocs();
extern "C" int get_memory_allocated_user_total();

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
  /* Since we started after the beginning of the program, we missed a bunch of
   * allocations. We cannot record what was allocated and then deleted, but we
   * can still record all the memory that is still allocated.
   */
}

MemEntry::MemEntry() : type(0), where(0), size(0), stackSize(0) { }

void MemEntry::write(FILE *fp) {
  if (type == BEGIN_TRACE) {
    fprintf(fp, "%d %d\n", type, size);
    return;
  }
  fprintf(fp, "%d %p %d", type, where, size);
  if (type == MEMORY_MALLOC) {
    fprintf(fp, " %d", stackSize);
    void **stack = (void**)(this+1);
    for (int i=stackSize-1; i>=0; --i) {
      fprintf(fp, " %p", stack[i]);
    }
  }
  fprintf(fp, "\n");
}

TraceMemory::TraceMemory(char **argv) {
  usedBuffer = 0;
  firstTime = 1;
  traceDisabled = false;
  logBufSize = DefaultBufferSize;
  if (CmiGetArgIntDesc(argv,"+memlogsize",&logBufSize, 
		       "Log buffer size (in kB)")) {
    if (CkMyPe() == 0) {
      CmiPrintf("Trace: logsize: %d kB\n", logBufSize);
    }
  }
  recordStack = false;
  if (CmiGetArgFlagDesc(argv,"+recordStack",
               "Record stack trace for malloc")) {
    recordStack = true;
  }
  logBufSize *= 1024;
  logBuffer = (char *) ::malloc(logBufSize);
}

inline void TraceMemory::checkFlush(int increment) {
  if (usedBuffer+increment >= logBufSize) {
    flush();
  }
}

inline void TraceMemory::flush() {
  traceDisabled = true;
  //CmiPrintf("[%d] TraceMemory::flush %d\n",CmiMyPe(),usedBuffer);
  const char *mode;
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
  //fprintf(fp, "begin flush\n");
  for (int i=0; i<usedBuffer; i += sizeof(MemEntry) + ((MemEntry*)&logBuffer[i])->stackSize*sizeof(void*)) {
    ((MemEntry*)&logBuffer[i])->write(fp);
  }
  //fprintf(fp, "end flush\n");
  fclose(fp);
  usedBuffer = 0;
  traceDisabled = false;
}

void TraceMemory::traceClose() {
  flush();
}

void TraceMemory::traceBegin() {
  int increment = sizeof(MemEntry);
  checkFlush(increment);
  ((MemEntry*)&logBuffer[usedBuffer])->set(BEGIN_TRACE, 0, get_memory_allocated_user_total());
  usedBuffer += increment;
}

void TraceMemory::malloc(void *where, int size, void **stack, int stackSize) {
  if (!traceDisabled) {
    int increment = sizeof(MemEntry) + (recordStack ? stackSize*sizeof(void*) : 0);
    checkFlush(increment);
    ((MemEntry*)&logBuffer[usedBuffer])->set(MEMORY_MALLOC, where, size);
    if (recordStack) ((MemEntry*)&logBuffer[usedBuffer])->setStack(stackSize, stack);
    usedBuffer += increment;
    //CmiPrintf("[%d] TraceMemory::malloc  %d  (%p)\n",CmiMyPe(),usedBuffer,where);
  }
}

void TraceMemory::free(void *where, int size) {
  if (!traceDisabled) {
    int increment = sizeof(MemEntry);
    checkFlush(increment);
    ((MemEntry*)&logBuffer[usedBuffer])->set(MEMORY_FREE, where, size);
    usedBuffer += increment;
    //CmiPrintf("[%d] TraceMemory::free    %d  (%p)\n",CmiMyPe(),usedBuffer,where);
  }
}
