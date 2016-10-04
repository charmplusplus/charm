#include "charm++.h"
#include "envelope.h"
#include "queueing.h"

#if CMK_SMP && CMK_TASKQUEUE
#include "taskqueue.h"
#include "cktaskQ.h"

void StealTask() {
  double _start = CmiWallTimer();
  int random_pe = CrnRand() % CkMyNodeSize();

  while (random_pe == CkMyPe()) {
    random_pe = CrnRand() % CkMyNodeSize();
  }
#if CMK_TRACE_ENABLED
  char s[10];
  sprintf( s, "%d", random_pe );
  traceUserSuppliedBracketedNote(s, TASKQ_QUEUE_STEAL_EVENTID, _start, CmiWallTimer());
#endif
  void* msg = TaskQueueSteal((TaskQueue)CpvAccessOther(CsdTaskQueue, random_pe));
  if (msg != NULL) {
    TaskQueuePush((TaskQueue)CpvAccess(CsdTaskQueue), msg);
  }
#if CMK_TRACE_ENABLED
  traceUserSuppliedBracketedNote(s, TASKQ_STEAL_EVENTID, _start, CmiWallTimer());
#endif
}

static void TaskStealBeginIdle(void *dummy) {
  StealTask();
}

void _taskqInit() {

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
