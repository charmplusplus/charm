#ifndef _FIFOINT_H
#define  _FIFOINT_H

typedef struct {
  int size;
  int max;
  int head;
  int tail;
  int *vector;
} IntQueue;


#endif
