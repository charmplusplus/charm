
#include <stdio.h>
#include "converse.h"

typedef struct fifo_queue {
        void **block;
        int block_len;
        int first;
        int avail;
        int length;
} FIFO_QUEUE;


#define BLK_LEN 512

void           *
FIFO_Create()
{
	FIFO_QUEUE     *queue;
	queue = (FIFO_QUEUE *)malloc(sizeof(FIFO_QUEUE));
	queue->block = (void **) malloc(BLK_LEN * sizeof(void *));
	queue->block_len = BLK_LEN;
	queue->first = queue->avail = 0;
	queue->length = 0;

	return (void *) queue;
}


FIFO_Empty(queue)
FIFO_QUEUE     *queue;
{
	return (queue->length == 0) ? 1 : 0;
}


FIFO_EnQueue(queue, element)
FIFO_QUEUE     *queue;
void           *element;
{

	if (!queue->length)	/* Queue is empty */
	{
		queue->block[queue->avail] = element;
		queue->first = queue->avail;
		queue->avail = (queue->avail + 1) % queue->block_len;
		queue->length++;
	}
	else
	{
		if (queue->avail == queue->first)	/* Queue is full */
		{
			void          **blk = queue->block;
			int             i, j;
			queue->block = (void **) malloc(sizeof(void *) * (queue->
							     block_len) *3);
			for (i = queue->first, j = 0; i < queue->block_len; i++, j++)
				queue->block[j] = blk[i];
			for (i = 0; i < queue->avail; i++, j++)
				queue->block[j] = blk[i];
			queue->block[j++] = element;
			queue->block_len *= 3;
			queue->first = 0;
			queue->avail = j;
			queue->length++;
			free(blk);
		}
		else
		{
			queue->block[queue->avail] = element;
			queue->avail = (queue->avail + 1) % queue->block_len;
			queue->length++;
		}
	}

}


FIFO_DeQueue(queue, element)
FIFO_QUEUE     *queue;
void      **element;
{
	*element = NULL;
	if (queue->length)
	{
		*element = queue->block[queue->first++];
		queue->first %= queue->block_len;
		queue->length--;
	}
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
