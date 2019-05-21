#ifndef _CK_TASKQ_H_
#define _CK_TASKQ_H_
#include "converse.h"
#include "taskqueue.h"

#if CMK_TRACE_ENABLED
#include "conv-trace.h"
#define TASKQ_CREATE_EVENTID 145
#define TASKQ_WORK_EVENTID 147
#define TASKQ_STEAL_EVENTID 149
#define TASKQ_QUEUE_STEAL_EVENTID 151
#endif
#ifdef __cplusplus
extern "C" {
#endif
void StealTask();
void CmiTaskQueueInit();
#ifdef __cplusplus
}
#endif
#endif
