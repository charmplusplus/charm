#include "conv-taskQ.h"
#if CMK_SMP && CMK_TASKQUEUE
extern "C" void StealTask() {
#if CMK_TRACE_ENABLED
  double _start = CmiWallTimer();
#endif
  int random_rank = CrnRand() % (CmiMyNodeSize()-1);
  if (random_rank >= CmiMyRank())
    ++random_rank;
#if CMK_TRACE_ENABLED
  char s[10];
  sprintf( s, "%d", random_rank );
  traceUserSuppliedBracketedNote(s, TASKQ_QUEUE_STEAL_EVENTID, _start, CmiWallTimer());
#endif
  void* msg = TaskQueueSteal((TaskQueue)CpvAccessOther(CsdTaskQueue, random_rank));
  if (msg != NULL) {
    TaskQueuePush((TaskQueue)CpvAccess(CsdTaskQueue), msg);
  }
#if CMK_TRACE_ENABLED
  traceUserSuppliedBracketedNote(s, TASKQ_STEAL_EVENTID, _start, CmiWallTimer());
#endif
}

static void TaskStealBeginIdle(void *dummy) {
  if (CmiMyNodeSize() > 1)
    StealTask();
}

extern "C" void CmiTaskQueueInit() {
  if(CmiMyNodeSize() > 1) {
    CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,
        (CcdVoidFn) TaskStealBeginIdle, NULL);

    CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,
        (CcdVoidFn) TaskStealBeginIdle, NULL);
  }
#if CMK_TRACE_ENABLED
  traceRegisterUserEvent("taskq create work", TASKQ_CREATE_EVENTID);
  traceRegisterUserEvent("taskq work", TASKQ_WORK_EVENTID);
  traceRegisterUserEvent("taskq steal", TASKQ_STEAL_EVENTID);
  traceRegisterUserEvent("taskq from queue steal", TASKQ_QUEUE_STEAL_EVENTID);
#endif
}
#endif
