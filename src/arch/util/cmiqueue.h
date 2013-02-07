#ifndef _CMI_QUEUE_DECL_H
#define _CMI_QUEUE_DECL_H

#if SPECIFIC_PCQUEUE && CMK_SMP
#define CMIQueue LRTSQueue 
#define CMIQueuePush    LRTSQueuePush
#define CMIQueueCreate  LRTSQueueCreate
#define CMIQueuePop     LRTSQueuePop
#define CMIQueueEmpty   LRTSQueueEmpty
#else
#define CMIQueue PCQueue
#define CMIQueuePush    PCQueuePush
#define CMIQueueCreate  PCQueueCreate
#define CMIQueuePop     PCQueuePop
#define CMIQueueEmpty   PCQueueEmpty
#endif

#endif //_CMI_QUEUE_DECL_H
