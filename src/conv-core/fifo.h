#ifndef FIFO_H
#define FIFO_H

typedef struct fifo_queue {
  void **block;
  unsigned int size;
  unsigned int pull;
  unsigned int push;
  unsigned int fill;
} FIFO_QUEUE;


#define BLK_LEN 512

#endif
