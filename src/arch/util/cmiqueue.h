#if SPECIFIC_PCQUEUE
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
