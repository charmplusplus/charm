#ifndef _CKTASKQUEUE_H
#define _CKTASKQUEUE_H
#define TaskQueueSize 1024
//Uncomment for debug print statements
#define TaskQueueDebug(...) //CmiPrintf(__VA_ARGS__)
// This taskqueue implementation refers to the work-stealing queue of Cilk (THE protocol).
// New tasks are pushed into the tail of this queue and the tasks are popped at the tail of this queue by the same thread. Thieves(other threads trying to steal) steal a task at the head of this queue.
// So, synchronization is needed only when there is only one task in the queue because thieves and victim can try to obtain the same task.

typedef struct TaskQueueStruct {
  int head; // This pointer indicates the first task in the queue
  int tail; // The tail indicates the array element next to the last available task in the queue. So, if head == tail, the queue is empty
  void *data[TaskQueueSize];
} *TaskQueue;

inline static TaskQueue TaskQueueCreate() {
  TaskQueue t = (TaskQueue)malloc(sizeof(struct TaskQueueStruct));
  t->head = 0;
  t->tail = 0;
  return t;
}

inline static void TaskQueuePush(TaskQueue Q, void *data) {
  Q->data[ Q->tail % TaskQueueSize] = data;
  CmiMemoryWriteFence();
  Q->tail +=1;
}

inline static void* TaskQueuePop(TaskQueue Q) { // Pop happens in the same worker thread which pushed the task before.
  TaskQueueDebug("[%d] TaskQueuePop head %d tail %d\n", CmiMyPe(), Q->head, Q->tail);
  int t = Q->tail - 1;
  Q->tail = t;
  CmiMemoryWriteFence();
  int h = Q->head;
  if (t > h) { // This means there are more than two tasks in the queue, so it is safe to pop a task from the queue.
    TaskQueueDebug("[%d] returning valid data\n", CmiMyPe());
    return Q->data[t % TaskQueueSize];
  }

  if (t < h) { // The taskqueue is empty and the last task has been stolen by a thief.
    Q->tail = h;
    return NULL;
  }
  // From now on, we should handle the situation where there is only one task so thieves and victim can try to obtain this task simultaneously.
  Q->tail = h + 1;
  if (!__sync_bool_compare_and_swap(&(Q->head), h, h+1)) // Check whether the last task has already stolen.
    return NULL;
  return Q->data[t % TaskQueueSize];
}

inline static void* TaskQueueSteal(TaskQueue Q) {
  int h, t;
  void *task;
  while (1) {
    h = Q->head;
    t = Q->tail;
    if (h >= t) // The queue is empty or the last element has been stolen by other thieves or popped by the victim.
      return NULL;
    if(!__sync_bool_compare_and_swap(&(Q->head), h, h+1)) // Check whether the task this thief is trying to steal is still in the queue and not stolen by the other thieves.
      continue;
    return Q->data[h % TaskQueueSize];
  }
}

#endif
