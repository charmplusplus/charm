#include "trace-projections.h"

CpvDeclare(Trace*, _trace);
CpvDeclare(int, traceOn);
CpvDeclare(int, CtrLogBufSize);
CpvStaticDeclare(LogPool*, _logPool);

extern "C" 
void traceInit(int* argc, char **argv)
{
  CpvInitialize(Trace*, _trace);
  CpvInitialize(LogPool*, _logPool);
  CpvInitialize(int, traceOn);
  CpvInitialize(int, CtrLogBufSize);
  CpvAccess(traceOn) = 1;
  // find +logsize and +traceoff here in argv
  // and initialize the vars properly.
}

extern "C"
void traceBeginIdle(void)
{
}

extern "C"
void traceEndIdle(void)
{
}

extern "C"
void traceResume(void)
{
}

extern "C"
void traceSuspend(void)
{
}

extern "C"
void traceAwaken(void)
{
}

extern "C"
void traceUserEvent(int)
{
}

extern "C"
int traceRegisterUserEvent(const char*)
{
  return 0;
}

extern "C"
void traceClose(void)
{
}

void LogEntry::write(FILE* fp)
{
  fprintf(fp, "%d ", type);

  switch (type) {
    case USER_EVENT:
      fprintf(fp, "%d %u %d %d", mIdx, (int) (time*1.0e6), event, pe);
      break;

    case BEGIN_IDLE:
    case END_IDLE:
    case BEGIN_PACK:
    case END_PACK:
    case BEGIN_UNPACK:
    case END_UNPACK:
      fprintf(fp, "%u %d", (int) (time*1.0e6), pe);
      break;

    case CREATION:
    case BEGIN_PROCESSING:
    case END_PROCESSING:
      fprintf(fp, "%d %d %u %d %d", mIdx, eIdx, (int) (time*1.0e6), event, pe);
      break;

    case ENQUEUE:
    case DEQUEUE:
      fprintf(fp, "%d %u %d %d", mIdx, (int) (time*1.0e6), event, pe);
      break;

    case BEGIN_INTERRUPT:
    case END_INTERRUPT:
      fprintf(fp, "%u %d %d", (int) (time*1.0e6), event, pe);
      break;

    case BEGIN_COMPUTATION:
    case END_COMPUTATION:
    fprintf(fp, "%u", (int) (time*1.0e6));
      break;

    default:
      CkPrintf("***Internal Error*** Wierd Event %d.\n", type);
      break;
  }
}
