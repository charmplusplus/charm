#ifndef FIFO_H
#define FIFO_H

typedef struct fifo_queue {
  void **block;
  unsigned int size;
  unsigned int pull;
  unsigned int push;
  unsigned int fill;
} FIFO_QUEUE;

#define _FIFO_BLK_LEN 512

#ifdef __cplusplus
extern "C" {
#endif

FIFO_QUEUE *FIFO_Create(void);
int FIFO_Fill(FIFO_QUEUE *);
int FIFO_Empty(FIFO_QUEUE *);
void FIFO_EnQueue(FIFO_QUEUE *queue, void *elt);
void FIFO_EnQueue_Front(FIFO_QUEUE *queue, void *elt);
void *FIFO_Peek(FIFO_QUEUE *queue);
void FIFO_Pop(FIFO_QUEUE *queue);
void FIFO_DeQueue(FIFO_QUEUE *queue, void **element);
void FIFO_Enumerate(FIFO_QUEUE *queue, void ***element);
void FIFO_Destroy(FIFO_QUEUE *queue);

#ifdef __cplusplus
}
#endif

#endif
