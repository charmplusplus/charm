#include <stdlib.h>
#include "converse.h"
#include "fifo.h"


FIFO_QUEUE *FIFO_Create(void)
{
  FIFO_QUEUE *queue;
  queue = (FIFO_QUEUE *)malloc(sizeof(FIFO_QUEUE));
  queue->block = (void **) malloc(_FIFO_BLK_LEN * sizeof(void *));
  queue->size = _FIFO_BLK_LEN;
  queue->push = queue->pull = 0;
  queue->fill = 0;
  return queue;
}

int FIFO_Fill(FIFO_QUEUE *queue)
{
  return queue->fill;
}

int FIFO_Empty(FIFO_QUEUE *queue)
{
  return !(queue->fill);
}

static void FIFO_Expand(FIFO_QUEUE *queue)
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

void FIFO_EnQueue(FIFO_QUEUE *queue, void *elt)
{
  if (queue->fill == queue->size) FIFO_Expand(queue);
  queue->block[queue->push] = elt;
  queue->push = ((queue->push + 1) % queue->size);
  queue->fill++;
}

void FIFO_EnQueue_Front(FIFO_QUEUE *queue, void *elt)
{
  if (queue->fill == queue->size) FIFO_Expand(queue);
  queue->pull = ((queue->pull + queue->size - 1) % queue->size);
  queue->block[queue->pull] = elt;
  queue->fill++;
}

void *FIFO_Peek(FIFO_QUEUE *queue)
{
  if (queue->fill == 0) return 0;
  return queue->block[queue->pull];
}

void FIFO_Pop(FIFO_QUEUE *queue)
{
  if (queue->fill) {
    queue->pull = (queue->pull+1) % queue->size;
    queue->fill--;
  }
}

void FIFO_DeQueue(FIFO_QUEUE *queue, void **element)
{
  *element = 0;
  if (queue->fill) {
    *element = queue->block[queue->pull];
    queue->pull = (queue->pull+1) % queue->size;
    queue->fill--;
  }
}


/* This assumes the the caller has not allocated
   memory for element 
*/
void FIFO_Enumerate(FIFO_QUEUE *queue, void ***element)
{
  int i = 0;
  int num = queue->fill;
  int pull = queue->pull;
  *element = 0;
  if(num == 0)
    return;
  *element = (void **)malloc(num * sizeof(void *));
  while(num > 0){
    (*element)[i++] = queue->block[pull];
    pull = (pull + 1) % queue->size;
    num--;
  }
}

void FIFO_Destroy(FIFO_QUEUE *queue)
{
  if (!FIFO_Empty(queue)) {
    CmiError("Tried to FIFO_Destroy a non-empty queue.\n");
    exit(1);
  }
  free(queue->block);
  free(queue);
}
