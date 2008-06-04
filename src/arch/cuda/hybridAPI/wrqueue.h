/*
 * wrqueue.h
 *
 * by Lukasz Wesolowski
 * 04.12.2008
 *
 * a simple FIFO queue for GPU work requests
 *
 */

#ifndef __WR_QUEUE_H__
#define __WR_QUEUE_H__

#include "wr.h"

/* number of work requests the queue is initialized to handle */
#define QUEUE_SIZE_INIT 10

/* if the queue is filled, it will be expanded by this many additional 
   units */ 
#define QUEUE_EXPANSION_SIZE 10

/* initWRqueue
 *
 * allocate memory for the queue and initialize bookkeeping variables
 *
 */
void initWRqueue(workRequestQueue **qptr); 

/* dequeue
 *
 * delete the head entry in the queue
 * assumes memory buffers have previously been freed or will be reused
 *
 */
void dequeue(workRequestQueue *q); 

/* deleteWRqueue
 *
 * if queue is nonempty, return -1  
 * if queue is empty, delete the queue, freeing dynamically allocated 
 * memory, and return 0
 *
 *
 */
int deleteWRqueue(workRequestQueue *q); 

/* head
 * 
 * returns the first element in the queue 
 *
 */
workRequest * head(workRequestQueue *q);

/*
 * isEmpty
 *
 * returns:
 * 1 if queue has no pending requests stored
 * 0 otherwise
 */
int isEmpty(workRequestQueue *q);

#endif
