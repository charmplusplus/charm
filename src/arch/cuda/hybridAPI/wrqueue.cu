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

void init_wrqueue(workRequestQueue *q) {

  q = (workRequestQueue*) malloc(sizeof(workRequestQueue));  

  q->head = -1; 
  q->tail = -1;
  q->size = 0; 
  q->capacity = QUEUE_SIZE_INIT; 

  q->requests = (workRequest *) malloc(QUEUE_SIZE_INIT * sizeof(workRequest)); 

}

void enqueue(workRequestQueue *q, workRequest *wr) {
  workRequest *newArray; 
  int newSize; 
  int tailendIndex;  /* index for the second part of the array in the new array */

  if (q->size == q->capacity) {

    /* queue is out of space: create a new queue with
       QUEUE_EXPANSION_SIZE more slots */

    newSize = q->capacity + QUEUE_EXPANSION_SIZE;
    newArray = (workRequest *) malloc(newSize * sizeof(workRequest));

    /* copy requests to the new array */
    memcpy(newArray, &q->requests[q->head], 
	   (q->capacity - q->head) * sizeof(workRequest));

    /* if head index is not 0, there are additional work requests to
       be copied from the beginning of the array */
    if (q->head != 0) {
      tailendIndex = q->capacity - q->head; 
      memcpy(&newArray[tailendIndex], q->requests, q->head); 
    }

    /* update bookkeeping variables in the expanded queue */
    q->tail = q->size; 
    q->capacity += QUEUE_EXPANSION_SIZE;
    q->head = 0;
    
    /* free the old queue's memory */
    free(q->requests); 

    /* reassign the pointer to the new queue */
    q->requests = newArray;
  }

  q->tail++; 
  if (q->tail == q->capacity) {
    q->tail = 0; 
  }

  memcpy(&q->requests[q->tail], wr, sizeof(workRequest));
  free(wr); 

  q->size++; 
}

void dequeue(workRequestQueue *q) {
  q->head++; 
  if (q->head == q->capacity) {
    q->head = 0; 
  }
  q->size--; 
}

int delete_wrqueue(workRequestQueue *q) {
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
