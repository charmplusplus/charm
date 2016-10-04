#ifndef _CK_TASKQ_H_
#define _CK_TASKQ_H_

#include "cklists.h"
#include "taskqueue.h"

#if CMK_TRACE_ENABLED
#define TASKQ_CREATE_EVENTID 145
#define TASKQ_WORK_EVENTID 147
#define TASKQ_STEAL_EVENTID 149
#define TASKQ_QUEUE_STEAL_EVENTID 151
#endif

#endif
