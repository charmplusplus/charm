/*
 * wrqueue.cu
 *
 * by Lukasz Wesolowski
 * 04.12.2008
 *
 * a simple FIFO queue for GPU work requests
 *
 */

#include "cuda-hybrid-api.h"
#include "wrqueue.h"
#include "stdio.h"

#ifdef GPU_WRQ_VERBOSE
extern int CmiMyPe();
#endif

extern void QdCreate(int n); 
extern void QdProcess(int n); 

void initWRqueue(workRequestQueue **qptr) {

  (*qptr) = (workRequestQueue*) malloc(sizeof(workRequestQueue));  

  (*qptr)->head = 0; 
  (*qptr)->tail = -1;
  (*qptr)->size = 0; 
  (*qptr)->capacity = QUEUE_SIZE_INIT; 
  (*qptr)->requests = (workRequest *) 
    malloc(QUEUE_SIZE_INIT * sizeof(workRequest)); 

}

void enqueue(workRequestQueue *q, workRequest *wr) {
  if (q->size == q->capacity) {
    workRequest *newArray; 
    int newSize; 
    int tailendIndex;/* the starting index for the second part of the array */

    /* queue is out of space: create a new queue that is a factor
       QUEUE_EXPANSION_FACTOR larger */

    newSize = q->capacity * QUEUE_EXPANSION_FACTOR;
    newArray = (workRequest *) malloc(newSize * sizeof(workRequest));

    /* copy requests to the new array */
    memcpy(newArray, &q->requests[q->head], 
	   (q->capacity - q->head) * sizeof(workRequest));

    /* if head index is not 0, there are additional work requests to
       be copied from the beginning of the array */
    if (q->head != 0) {
      tailendIndex = q->capacity - q->head; 
      memcpy(&newArray[tailendIndex], q->requests, 
	     q->head * sizeof(workRequest)); 
    }

    /* free the old queue's memory */
    
    free(q->requests); 

    /* update bookkeeping variables in the expanded queue */
    q->tail = q->size - 1; 
    q->capacity *= QUEUE_EXPANSION_FACTOR;
    q->head = 0;
    
    /* reassign the pointer to the new queue */
    q->requests = newArray;
  }

  q->tail++; 
  if (q->tail == q->capacity) {
    q->tail = 0; 
  }

  memcpy(&q->requests[q->tail], wr, sizeof(workRequest));

  q->requests[q->tail].state = QUEUED; 

  q->size++; 

  QdCreate(1);

#ifdef GPU_WRQ_VERBOSE
  printf("(%d) ENQ size: %d\n", CmiMyPe(), q->size);
#endif
}

void setWRCallback(workRequest *wr, void *cb) {
    wr->callbackFn = cb;
}

void dequeue(workRequestQueue *q) {
  q->head++; 
  if (q->head == q->capacity) {
    q->head = 0; 
  }
  q->size--; 
#ifdef GPU_WRQ_VERBOSE
  printf("(%d) DEQ size: %d\n", CmiMyPe(), q->size);
#endif

  QdProcess(1);

}

int deleteWRqueue(workRequestQueue *q) {
  if (q->size != 0) {
    return -1; 
  }
  else {
    free(q->requests); 
    return 0; 
  }
}

workRequest * firstElement(workRequestQueue *q) {
  if (q->size == 0) {
    return NULL; 
  }
  else {
    return &q->requests[q->head]; 
  }
}

workRequest * secondElement(workRequestQueue *q) {
  if (q->size < 2) {
    return NULL; 
  }
  else {
    if (q->head == (q->capacity-1)) {
      return &q->requests[0];
    }
    else {
      return &q->requests[q->head+1]; 
    }
  }
}

workRequest * thirdElement(workRequestQueue *q) {
  if (q->size < 3) {
    return NULL; 
  }
  else {
    int wrIndex = q->head+2;

    if (wrIndex >= q->capacity) {
      wrIndex -= q->capacity; 
    }

    return &q->requests[wrIndex]; 
  }
}

int isEmpty(workRequestQueue *q) {
  if (q->size == 0) {
    return 1; 
  }
  else {
    return 0; 
  }
}
