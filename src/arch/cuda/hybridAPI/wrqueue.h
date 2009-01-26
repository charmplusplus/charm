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
#define QUEUE_SIZE_INIT 100

/* if the queue is filled, it will be expanded by this factor */ 
#define QUEUE_EXPANSION_FACTOR 2

/* work request states */
#define QUEUED           0   /* work request waiting in queue */
#define TRANSFERRING_IN  1   /* data is being transferred to the GPU */
#define READY            2   /* ready for kernel execution */
#define EXECUTING        3   /* kernel is executing */
#define TRANSFERRING_OUT 4   /* data is being transferred from the GPU */

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
 * or NULL if the queue is empty
 *
 */
workRequest * firstElement(workRequestQueue *q);

/* second
 * 
 * returns the second element in the queue or NULL if the queue has
 * fewer than 2 elements
 *
 */
workRequest * secondElement(workRequestQueue *q); 

/* third
 * 
 * returns the third element in the queue or NULL if the queue has
 * fewer than 3 elements
 *
 */
workRequest * thirdElement(workRequestQueue *q); 

/*
 * isEmpty
 *
 * returns:
 * 1 if queue has no pending requests stored
 * 0 otherwise
 */
int isEmpty(workRequestQueue *q);

#endif
