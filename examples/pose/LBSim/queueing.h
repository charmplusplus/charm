#ifndef _QUEUEING_H_
#define _QUEUEING_H_

//#include <stdio.h> 
//#include <stdlib.h>
//#include <malloc.h>

typedef struct node
{
    long int data;          /* Store the data*/
    struct node *next;  /* Pointer to the next node */
} NODE;

typedef struct queue
{
    NODE *front;        /* Front of the queue */
    NODE *rear;         /* Back of the queue */
} QUEUE;

int myinitialise(QUEUE *queue);
int myenqueue(QUEUE *queue, long int key);
int mydequeue(QUEUE *queue);
int myisempty(QUEUE *queue);
int myqlength(QUEUE *queue);
#endif
