/*
 * wrqueue.cu
 *
 * by Lukasz Wesolowski
 * 04.12.2008
 *
 * a simple FIFO queue for GPU work requests
 *
 */

#include "wrqueue.h"

void initWRqueue(workRequestQueue **qptr) {

  (*qptr) = (workRequestQueue*) malloc(sizeof(workRequestQueue));  

  (*qptr)->head = 0; 
  (*qptr)->tail = -1;
  (*qptr)->size = 0; 
  (*qptr)->capacity = QUEUE_SIZE_INIT; 

  (*qptr)->requests = (workRequest *) malloc(QUEUE_SIZE_INIT * sizeof(workRequest)); 

}

void enqueue(workRequestQueue *q, workRequest *wr) {
  workRequest *newArray; 
  int newSize; 
  int tailendIndex;  /* the starting index for the second part of the array */

  if (q->size == q->capacity) {

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
    q->tail = q->size; 
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
}

void dequeue(workRequestQueue *q) {
  q->head++; 
  if (q->head == q->capacity) {
    q->head = 0; 
  }
  q->size--; 
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

workRequest * head(workRequestQueue *q) {
    return &q->requests[q->head]; 
}

int isEmpty(workRequestQueue *q) {
  if (q->size == 0) {
    return 1; 
  }
  else {
    return 0; 
  }
}
