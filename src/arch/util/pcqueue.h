
#ifndef __PCQUEUE__
#define __PCQUEUE__

/*****************************************************************************
 *
 * Producer-Consumer Queues
 *
 * This queue implementation enables a producer and a consumer to
 * communicate via a queue.  The queues are optimized for this situation,
 * they don't require any operating system locks (they do require 32-bit
 * reads and writes to be atomic.)  Cautions: there can only be one
 * producer, and one consumer.  These queues cannot store null pointers.
 *
 ****************************************************************************/

#define PCQueueSize 0x100

typedef struct CircQueueStruct
{
  struct CircQueueStruct *next;
  int push;
  int pull;
  char *data[PCQueueSize];
}
*CircQueue;

typedef struct PCQueueStruct
{
  CircQueue head;
  CircQueue tail;
}
*PCQueue;

/* static CircQueue Cmi_freelist_circqueuestruct = 0;
   static int freeCount = 0; */

#define FreeCircQueueStruct(dg) {\
  CircQueue d;\
  CmiMemLock();\
  d=(dg);\
  d->next = Cmi_freelist_circqueuestruct;\
  Cmi_freelist_circqueuestruct = d;\
  freeCount++;\
  CmiMemUnlock();\
}

#define MallocCircQueueStruct(dg) {\
  CircQueue d;\
  CmiMemLock();\
  d = Cmi_freelist_circqueuestruct;\
  if (d==(CircQueue)0){\
    d = ((CircQueue)calloc(1, sizeof(struct CircQueueStruct)));\
  }\
  else{\
    freeCount--;\
    Cmi_freelist_circqueuestruct = d->next;\
    }\
  dg = d;\
  CmiMemUnlock();\
}


PCQueue PCQueueCreate(void)
{
  CircQueue circ;
  PCQueue Q;

  /* MallocCircQueueStruct(circ); */
  circ = (CircQueue)calloc(1, sizeof(struct CircQueueStruct));

  Q = (PCQueue)malloc(sizeof(struct PCQueueStruct));
  _MEMCHECK(Q);
  Q->head = circ;
  Q->tail = circ;
  return Q;
}

int PCQueueEmpty(PCQueue Q)
{
  CircQueue circ = Q->head;
  char *data = circ->data[circ->pull];
  return (data == 0);
}

char *PCQueuePop(PCQueue Q)
{
  CircQueue circ; int pull; char *data;

    circ = Q->head;
    pull = circ->pull;
    data = circ->data[pull];
    if (data) {
      circ->pull = (pull + 1);
      circ->data[pull] = 0;
      if (pull == PCQueueSize - 1) { /* just pulled the data from the last slot
                                     of this buffer */
        Q->head = circ-> next; /* next buffer must exist, because "Push"  */
	
	/* FreeCircQueueStruct(circ); */
        free(circ);
	
	/* links in the next buffer *before* filling */
                               /* in the last slot. See below. */
      }
      return data;
    }
    else { /* queue seems to be empty. The producer may be adding something
              to it, but its ok to report queue is empty. */
      return 0;
    }
}

void PCQueuePush(PCQueue Q, char *data)
{
  CircQueue circ, circ1; int push;
  
  circ1 = Q->tail;
  push = circ1->push;
  if (push == (PCQueueSize -1)) { /* last slot is about to be filled */
    /* this way, the next buffer is linked in before data is filled in 
       in the last slot of this buffer */

    circ = (CircQueue)calloc(1, sizeof(struct CircQueueStruct));
    /* MallocCircQueueStruct(circ); */

    Q->tail->next = circ;
    Q->tail = circ;
  }
  circ1->data[push] = data;
  circ1->push = (push + 1);
}



#endif
