
#include <stdio.h>
#include "converse.h"

typedef struct fifo_queue {
  void **block;
  unsigned int size;
  unsigned int pull;
  unsigned int push;
  unsigned int fill;
} FIFO_QUEUE;


#define BLK_LEN 512

void *FIFO_Create()
{
  FIFO_QUEUE *queue;
  queue = (FIFO_QUEUE *)malloc(sizeof(FIFO_QUEUE));
  queue->block = (void **) malloc(BLK_LEN * sizeof(void *));
  queue->size = BLK_LEN;
  queue->push = queue->pull = 0;
  queue->fill = 0;
  return (void *)queue;
}

int FIFO_Fill(queue)
     FIFO_QUEUE *queue;
{
  return queue->fill;
}

int FIFO_Empty(queue)
     FIFO_QUEUE *queue;
{
  return (queue->fill == 0) ? 1 : 0;
}

void FIFO_Expand(queue)
     FIFO_QUEUE *queue;
{
  int newsize; void **newblock; int rest;
  int    pull  = queue->pull;
  int    size  = queue->size;
  void **block = queue->block;
  newsize = size * 3;
  newblock = (void**)malloc(newsize * sizeof(void *));
  rest = size - pull;
  memcpy(newblock, block + pull, rest * sizeof(void *));
  memcpy(newblock + rest, block, pull * sizeof(void *));
  free(block);
  queue->block = newblock;
  queue->size = newsize;
  queue->pull = 0;
  queue->push = size;
  queue->fill = size;
}

void FIFO_EnQueue(queue, elt)
     FIFO_QUEUE *queue;
     void       *elt;
{
  if (queue->fill == queue->size) FIFO_Expand(queue);
  queue->block[queue->push] = elt;
  queue->push = ((queue->push + 1) % queue->size);
  queue->fill++;
}

void FIFO_EnQueue_Front(queue, elt)
     FIFO_QUEUE *queue;
     void *elt;
{
  if (queue->fill == queue->size) FIFO_Expand(queue);
  queue->pull = ((queue->pull + queue->size - 1) % queue->size);
  queue->block[queue->pull] = elt;
  queue->fill++;
}

void *FIFO_Peek(queue)
     FIFO_QUEUE *queue;
{
  if (queue->fill == 0) return 0;
  return queue->block[queue->pull];
}

void FIFO_Pop(queue)
     FIFO_QUEUE *queue;
{
  if (queue->fill) {
    queue->pull = (queue->pull+1) % queue->size;
    queue->fill--;
  }
}

void FIFO_DeQueue(queue, element)
     FIFO_QUEUE     *queue;
     void      **element;
{
  if (queue->fill) {
    *element = queue->block[queue->pull];
    queue->pull = (queue->pull+1) % queue->size;
    queue->fill--;
  } else *element = 0;
}

FIFO_Destroy(queue)
     FIFO_QUEUE *queue;
{
  if (!FIFO_Empty(queue)) {
    CmiError("Tried to FIFO_Destroy a non-empty queue.\n");
    exit(1);
  }
  free(queue->block);
  free(queue);
}
