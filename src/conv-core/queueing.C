#include "queueing.h"
#include <converse.h>
#include <string.h>

#if CMK_USE_STL_MSGQ
#include "msgq.h"
typedef CMK_MSG_PRIO_TYPE prio_t;
#endif


/** @defgroup CharmScheduler 
    @brief The portion of Charm++ responsible for scheduling the execution 
    of Charm++ entry methods

    CqsEnqueueGeneral() is the main function that is responsible for enqueueing 
    messages. It will store the messages in one of three queues based on the 
    specified priorities or strategies. The Charm++ message queue is really three 
    queues, one for positive priorities, one for zero priorities, and one for 
    negative priorities. The positive and negative priorty queues are actually heaps.


    The charm++ messages are only scheduled after the \ref
    ConverseScheduler "converse message queues" have been emptied:

    - CsdScheduleForever() is the commonly used Converse scheduler loop
       - CsdNextMessage()
          - First processes converse message from a  \ref ConverseScheduler "converse queue" if one exists using CdsFifo_Dequeue() 
          - Then if no converse message was found, call CqsDequeue() which gets a charm++ message to execute if one exists


    @file
    Implementation of queuing data structure functions.
    @ingroup CharmScheduler
    
    @addtogroup CharmScheduler
    @{
*/

#if 1
/* I don't see why we need CmiAlloc here, allocated from pinned memory
 * can be expensive */
#define CmiAlloc   malloc
#define CmiFree    free
#endif

/** A memory limit threshold for adaptively scheduling */
int schedAdaptMemThresholdMB;


/** Initialize a deq */
static void CqsDeqInit(_deq d)
{
  d->bgn  = d->space;
  d->end  = d->space+4;
  d->head = d->space;
  d->tail = d->space;
}

/** Double the size of a deq */
static void CqsDeqExpand(_deq d)
{
  int rsize = (d->end - d->head);
  int lsize = (d->head - d->bgn);
  int oldsize = (d->end - d->bgn);
  int newsize = (oldsize << 1);
  void **ovec = d->bgn;
  void **nvec = (void **)CmiAlloc(newsize * sizeof(void *));
  memcpy(nvec, d->head, rsize * sizeof(void *));
  memcpy(nvec+rsize, d->bgn, lsize * sizeof(void *));
  d->bgn = nvec;
  d->end = nvec + newsize;
  d->head = nvec;
  d->tail = nvec + oldsize;
  if (ovec != d->space) CmiFree(ovec);
}

/** Insert a data pointer at the tail of a deq */
void CqsDeqEnqueueFifo(_deq d, void *data)
{
  void **tail = d->tail;
  *tail = data;
  tail++;
  if (tail == d->end) tail = d->bgn;
  d->tail = tail;
  if (tail == d->head) CqsDeqExpand(d);
}

/** Insert a data pointer at the head of a deq */
void CqsDeqEnqueueLifo(_deq d, void *data)
{
  void **head = d->head;
  if (head == d->bgn) head = d->end;
  head--;
  *head = data;
  d->head = head;
  if (head == d->tail) CqsDeqExpand(d);
}

/** Remove a data pointer from the head of a deq */
void *CqsDeqDequeue(_deq d)
{
  void **head;
  void **tail;
  void *data;
  head = d->head;
  tail = d->tail;
  if (head == tail) return 0;
  data = *head;
  head++;
  if (head == d->end) head = d->bgn;
  d->head = head;
  return data;
}

/** Initialize a Priority Queue */
static void CqsPrioqInit(_prioq pq)
{
  int i;
  pq->heapsize = 100;
  pq->heapnext = 1;
  pq->hash_key_size = PRIOQ_TABSIZE;
  pq->hash_entry_size = 0;
  pq->heap = (_prioqelt *)CmiAlloc(100 * sizeof(_prioqelt));
  pq->hashtab = (_prioqelt *)CmiAlloc(pq->hash_key_size * sizeof(_prioqelt));
  for (i=0; i<pq->hash_key_size; i++) pq->hashtab[i]=0;
}

#if CMK_C_INLINE
inline
#endif
/** Double the size of a Priority Queue's heap */
static void CqsPrioqExpand(_prioq pq)
{
  int oldsize = pq->heapsize;
  int newsize = oldsize * 2;
  _prioqelt *oheap = pq->heap;
  _prioqelt *nheap = (_prioqelt *)CmiAlloc(newsize*sizeof(_prioqelt));
  memcpy(nheap, oheap, oldsize * sizeof(_prioqelt));
  pq->heap = nheap;
  pq->heapsize = newsize;
  CmiFree(oheap);
}
#ifndef FASTQ
/** Double the size of a Priority Queue's hash table */
void CqsPrioqRehash(_prioq pq)
{
  int oldHsize = pq->hash_key_size;
  int newHsize = oldHsize * 2;
  unsigned int hashval;
  _prioqelt pe, pe1, pe2;
  int i,j;

  _prioqelt *ohashtab = pq->hashtab;
  _prioqelt *nhashtab = (_prioqelt *)CmiAlloc(newHsize*sizeof(_prioqelt));

  pq->hash_key_size = newHsize;

  for(i=0; i<newHsize; i++)
    nhashtab[i] = 0;

  for(i=0; i<oldHsize; i++) {
    for(pe=ohashtab[i]; pe; ) {
      pe2 = pe->ht_next;
      hashval = pe->pri.bits;
      for (j=0; j<pe->pri.ints; j++) hashval ^= pe->pri.data[j];
      hashval = (hashval&0x7FFFFFFF)%newHsize;

      pe1=nhashtab[hashval];
      pe->ht_next = pe1;
      pe->ht_handle = (nhashtab+hashval);
      if (pe1) pe1->ht_handle = &(pe->ht_next);
      nhashtab[hashval]=pe;
      pe = pe2;
    }
  }
  pq->hashtab = nhashtab;
  pq->hash_key_size = newHsize;
  CmiFree(ohashtab);
}
#endif

int CqsPrioGT_(unsigned int ints1, unsigned int *data1, unsigned int ints2, unsigned int *data2)
{
  unsigned int val1;
  unsigned int val2;
  while (1) {
    if (ints1==0) return 0;
    if (ints2==0) return 1;
    val1 = *data1++;
    val2 = *data2++;
    if (val1 < val2) return 0;
    if (val1 > val2) return 1;
    ints1--;
    ints2--;
  }
}

/**
 * Compare two priorities (treated as unsigned).
 * @return 1 if prio1 > prio2
 * @return ? if prio1 == prio2
 * @return 0 if prio1 < prio2
 */
int CqsPrioGT(_prio prio1, _prio prio2)
{
#ifndef FASTQ
  unsigned int ints1 = prio1->ints;
  unsigned int ints2 = prio2->ints;
#endif
  unsigned int *data1 = prio1->data;
  unsigned int *data2 = prio2->data;
#ifndef FASTQ
  unsigned int val1;
  unsigned int val2;
#endif
  while (1) {
#ifndef FASTQ
    if (ints1==0) return 0;
    if (ints2==0) return 1;
#else
    if (prio1->ints==0) return 0;
    if (prio2->ints==0) return 1;
#endif
#ifndef FASTQ
    val1 = *data1++;
    val2 = *data2++;
    if (val1 < val2) return 0;
    if (val1 > val2) return 1;
    ints1--;
    ints2--;
#else
    if(*data1++ < *data2++) return 0;
    if(*data1++ > *data2++) return 1;
    (prio1->ints)--;
    (prio2->ints)--;
#endif
  }
}

/** Find or create a bucket in the hash table for the specified priority. */
_deq CqsPrioqGetDeq(_prioq pq, unsigned int priobits, unsigned int *priodata)
{
  unsigned int prioints = (priobits+CINTBITS-1)/CINTBITS;
  unsigned int hashval, i;
  int heappos; 
  _prioqelt *heap, pe, next, parent;
  _prio pri;
  int mem_cmp_res;
  unsigned int pri_bits_cmp;
  static int cnt_nilesh=0;

#ifdef FASTQ
  /*  printf("Hi I'm here %d\n",cnt_nilesh++); */
#endif
  /* Scan for priority in hash-table, and return it if present */
  hashval = priobits;
  for (i=0; i<prioints; i++) hashval ^= priodata[i];
  hashval = (hashval&0x7FFFFFFF)%PRIOQ_TABSIZE;
#ifndef FASTQ
  for (pe=pq->hashtab[hashval]; pe; pe=pe->ht_next)
    if (priobits == pe->pri.bits)
      if (memcmp(priodata, pe->pri.data, sizeof(int)*prioints)==0)
	return &(pe->data);
#else
  parent=NULL;
  for(pe=pq->hashtab[hashval]; pe; )
  {
    parent=pe;
    pri_bits_cmp=pe->pri.bits;
    mem_cmp_res=memcmp(priodata,pe->pri.data,sizeof(int)*prioints);
    if(priobits == pri_bits_cmp && mem_cmp_res==0)
      return &(pe->data);
    else if(priobits > pri_bits_cmp || (priobits == pri_bits_cmp && mem_cmp_res>0))
    {
      pe=pe->ht_right;
    }
    else 
    {
      pe=pe->ht_left;
    }
  }
#endif
  
  /* If not present, allocate a bucket for specified priority */
  pe = (_prioqelt)CmiAlloc(sizeof(struct prioqelt_struct)+((prioints-1)*sizeof(int)));
  pe->pri.bits = priobits;
  pe->pri.ints = prioints;
  memcpy(pe->pri.data, priodata, (prioints*sizeof(int)));
  CqsDeqInit(&(pe->data));
  pri=&(pe->pri);

  /* Insert bucket into hash-table */
  next = pq->hashtab[hashval];
#ifndef FASTQ
  pe->ht_next = next;
  pe->ht_handle = (pq->hashtab+hashval);
  if (next) next->ht_handle = &(pe->ht_next);
  pq->hashtab[hashval] = pe;
#else
  pe->ht_parent = parent;
  pe->ht_left = NULL;
  pe->ht_right = NULL;
  if(priobits > pri_bits_cmp || (priobits == pri_bits_cmp && mem_cmp_res>0))
  {
    if(parent) {
      parent->ht_right = pe;
      pe->ht_handle = &(parent->ht_right);
    }
    else {
      pe->ht_handle = (pq->hashtab+hashval);
      pq->hashtab[hashval] = pe;
    }
    /*    pe->ht_handle = &(pe); */
  }
  else
  {
    if(parent) {
      parent->ht_left = pe;
      pe->ht_handle = &(parent->ht_left);
    }
    else {
      pe->ht_handle = (pq->hashtab+hashval);
      pq->hashtab[hashval] = pe;
    }
    /*    pe->ht_handle = &(pe); */
  }
  if(!next)
    pq->hashtab[hashval] = pe;
#endif
  pq->hash_entry_size++;
#ifndef FASTQ
  if(pq->hash_entry_size > 2*pq->hash_key_size)
    CqsPrioqRehash(pq);
#endif  
  /* Insert bucket into heap */
  heappos = pq->heapnext++;
  if (heappos == pq->heapsize) CqsPrioqExpand(pq);
  heap = pq->heap;
  while (heappos > 1) {
    int parentpos = (heappos >> 1);
    _prioqelt parent = heap[parentpos];
    if (CqsPrioGT(pri, &(parent->pri))) break;
    heap[heappos] = parent; heappos=parentpos;
  }
  heap[heappos] = pe;

#ifdef FASTQ
  /*  printf("Hi I'm here222\n"); */
#endif
  
  return &(pe->data);
}

/** Dequeue an entry */
void *CqsPrioqDequeue(_prioq pq)
{
  _prio pri;
  _prioqelt pe, old; void *data;
  int heappos, heapnext;
  _prioqelt *heap = pq->heap;
  int left_child;
  _prioqelt temp1_ht_right, temp1_ht_left, temp1_ht_parent;
  _prioqelt *temp1_ht_handle;
  static int cnt_nilesh1=0;

#ifdef FASTQ
  /*  printf("Hi I'm here too!! %d\n",cnt_nilesh1++); */
#endif
  if (pq->heapnext==1) return 0;
  pe = heap[1];
  data = CqsDeqDequeue(&(pe->data));
  if (pe->data.head == pe->data.tail) {
    /* Unlink prio-bucket from hash-table */
#ifndef FASTQ
    _prioqelt next = pe->ht_next;
    _prioqelt *handle = pe->ht_handle;
    if (next) next->ht_handle = handle;
    *handle = next;
    old=pe;
#else
    old=pe;
    prioqelt *handle;
    if(pe->ht_parent)
    { 
      if(pe->ht_parent->ht_left==pe) left_child=1;
      else left_child=0;
    }
    else
      {  /* it is the root in the hashtable entry, so its ht_handle should be used by whoever is the new root */
      handle = pe->ht_handle;
    }
    
    if(!pe->ht_left && !pe->ht_right)
    {
      if(pe->ht_parent) {
	if(left_child) pe->ht_parent->ht_left=NULL;
	else pe->ht_parent->ht_right=NULL;
      }
      else {
	*handle = NULL;
      }
    }
    else if(!pe->ht_right)
    {
      /*if the node does not have a right subtree, its left subtree root is the new child of its parent */
      pe->ht_left->ht_parent=pe->ht_parent;
      if(pe->ht_parent)
      {
	if(left_child) {
	  pe->ht_parent->ht_left = pe->ht_left;
	  pe->ht_left->ht_handle = &(pe->ht_parent->ht_left);
	}
	else {
	  pe->ht_parent->ht_right = pe->ht_left;
	  pe->ht_left->ht_handle = &(pe->ht_parent->ht_right);
	}
      }
      else {
	pe->ht_left->ht_handle = handle;
	*handle = pe->ht_left;
      }
    }
    else if(!pe->ht_left)
    {
      /*if the node does not have a left subtree, its right subtree root is the new child of its parent */
      pe->ht_right->ht_parent=pe->ht_parent;
      /*pe->ht_right->ht_left=pe->ht_left; */
      if(pe->ht_parent)
      {
	if(left_child) {
	  pe->ht_parent->ht_left = pe->ht_right;
	  pe->ht_right->ht_handle = &(pe->ht_parent->ht_left);
	}
	else {
	  pe->ht_parent->ht_right = pe->ht_right;
	  pe->ht_right->ht_handle = &(pe->ht_parent->ht_right);
	}
      }
      else {
	pe->ht_right->ht_handle = handle;
	*handle = pe->ht_right;
      }
    }
    else if(!pe->ht_right->ht_left)
    {
      pe->ht_right->ht_parent=pe->ht_parent;
      if(pe->ht_parent)
      {
	if(left_child) {
	  pe->ht_parent->ht_left = pe->ht_right;
	  pe->ht_right->ht_handle = &(pe->ht_parent->ht_left);
	}
	else {
	  pe->ht_parent->ht_right = pe->ht_right;
	  pe->ht_right->ht_handle = &(pe->ht_parent->ht_right);
	}
      }
      else {
	pe->ht_right->ht_handle = handle;
	*handle = pe->ht_right;
      }
      if(pe->ht_left) {
	pe->ht_right->ht_left = pe->ht_left;
	pe->ht_left->ht_parent = pe->ht_right;
	pe->ht_left->ht_handle = &(pe->ht_right->ht_left);
      }
    }
    else
    {
      /*if it has both subtrees, swap it with its successor */
      for(pe=pe->ht_right; pe; )
      {
	if(pe->ht_left) pe=pe->ht_left;
	else  /*found the sucessor */
	  { /*take care of the connections */
	  if(old->ht_parent)
	  {
	    if(left_child) {
	      old->ht_parent->ht_left = pe;
	      pe->ht_handle = &(old->ht_parent->ht_left);
	    }
	    else {
	      old->ht_parent->ht_right = pe;
	      pe->ht_handle = &(old->ht_parent->ht_right);
	    }
	  }
	  else {
	    pe->ht_handle = handle;
	    *handle = pe;
	  }
	  temp1_ht_right = pe->ht_right;
	  temp1_ht_left = pe->ht_left;
	  temp1_ht_parent = pe->ht_parent;
	  temp1_ht_handle = pe->ht_handle;

	  pe->ht_parent = old->ht_parent;
	  pe->ht_left = old->ht_left;
	  pe->ht_right = old->ht_right;
	  if(pe->ht_left) {
	    pe->ht_left->ht_parent = pe;
	    pe->ht_right->ht_handle = &(pe->ht_right);
	  }
	  if(pe->ht_right) {
	    pe->ht_right->ht_parent = pe;
	    pe->ht_right->ht_handle = &(pe->ht_right);
	  }
	  temp1_ht_parent->ht_left = temp1_ht_right;
	  if(temp1_ht_right) {
	    temp1_ht_right->ht_handle = &(temp1_ht_parent->ht_left);
	    temp1_ht_right->ht_parent = temp1_ht_parent;
	  }
	  break;
	}
      }
    }
#endif
    pq->hash_entry_size--;
    
    /* Restore the heap */
    heapnext = (--pq->heapnext);
    pe = heap[heapnext];
    pri = &(pe->pri);
    heappos = 1;
    while (1) {
      int childpos1, childpos2, childpos;
      _prioqelt ch1, ch2, child;
      childpos1 = heappos<<1;
      if (childpos1>=heapnext) break;
      childpos2 = childpos1+1;
      if (childpos2>=heapnext)
	{ childpos=childpos1; child=heap[childpos1]; }
      else {
	ch1 = heap[childpos1];
	ch2 = heap[childpos2];
	if (CqsPrioGT(&(ch1->pri), &(ch2->pri)))
	     {childpos=childpos2; child=ch2;}
	else {childpos=childpos1; child=ch1;}
      }
      if (CqsPrioGT(&(child->pri), pri)) break;
      heap[heappos]=child; heappos=childpos;
    }
    heap[heappos]=pe;
    
    /* Free prio-bucket */
    if (old->data.bgn != old->data.space) CmiFree(old->data.bgn);
    CmiFree(old);
  }
  return data;
}

Queue CqsCreate(void)
{
  Queue q = (Queue)CmiAlloc(sizeof(struct Queue_struct));
  q->length = 0;
  q->maxlen = 0;
#ifdef FASTQ
  /*  printf("\nIN fastq"); */
#endif
  CqsDeqInit(&(q->zeroprio));
  CqsPrioqInit(&(q->negprioq));
  CqsPrioqInit(&(q->posprioq));
#if CMK_USE_STL_MSGQ
  q->stlQ = (void*) new conv::msgQ<prio_t>;
#endif
  return q;
}

void CqsDelete(Queue q)
{
  CmiFree(q->negprioq.heap);
  CmiFree(q->posprioq.heap);
#if CMK_USE_STL_MSGQ
  if (q->stlQ != NULL) delete (conv::msgQ<prio_t>*)(q->stlQ);
#endif
  CmiFree(q);
}

unsigned int CqsMaxLength(Queue q)
{
  return q->maxlen;
}

#if CMK_USE_STL_MSGQ

unsigned int CqsLength(Queue q)
{ return ( (conv::msgQ<prio_t>*)(q->stlQ) )->size(); }

int CqsEmpty(Queue q)
{ return ( (conv::msgQ<prio_t>*)(q->stlQ) )->empty(); }

void CqsEnqueueGeneral(Queue q, void *data, int strategy, int priobits,unsigned int *prioptr)
{
    bool isFifo = (strategy == CQS_QUEUEING_FIFO  ||
                   strategy == CQS_QUEUEING_IFIFO ||
                   strategy == CQS_QUEUEING_BFIFO ||
                   strategy == CQS_QUEUEING_LFIFO);
    if (priobits >= sizeof(int)*8 && strategy != CQS_QUEUEING_FIFO && strategy != CQS_QUEUEING_LIFO)
        ( (conv::msgQ<prio_t>*)(q->stlQ) )->enq( data, prioptr[0], isFifo);
    else
        ( (conv::msgQ<prio_t>*)(q->stlQ) )->enq( data, 0, isFifo);
}

void CqsEnqueueFifo(Queue q, void *data)
{ ( (conv::msgQ<prio_t>*)(q->stlQ) )->enq(data); }

void CqsEnqueueLifo(Queue q, void *data)
{ ( (conv::msgQ<prio_t>*)(q->stlQ) )->enq(data, 0, false); }

void CqsEnqueue(Queue q, void *data)
{ ( (conv::msgQ<prio_t>*)(q->stlQ) )->enq(data); }

void CqsDequeue(Queue q, void **resp)
{ *resp = (void*) ( (conv::msgQ<prio_t>*)(q->stlQ) )->deq(); }

#else

unsigned int CqsLength(Queue q)
{
  return q->length;
}

int CqsEmpty(Queue q)
{
  return (q->length == 0);
}

void CqsEnqueueGeneral(Queue q, void *data, int strategy, 
           int priobits,unsigned int *prioptr)
{
  _deq d; int iprio;
  CmiInt8 lprio0, lprio;
  switch (strategy) {
  case CQS_QUEUEING_FIFO: 
    CqsDeqEnqueueFifo(&(q->zeroprio), data); 
    break;
  case CQS_QUEUEING_LIFO: 
    CqsDeqEnqueueLifo(&(q->zeroprio), data); 
    break;
  case CQS_QUEUEING_IFIFO:
    iprio=prioptr[0]+(1U<<(CINTBITS-1));
    if ((int)iprio<0)
      d=CqsPrioqGetDeq(&(q->posprioq), CINTBITS, (unsigned int*)&iprio);
    else d=CqsPrioqGetDeq(&(q->negprioq), CINTBITS, (unsigned int*)&iprio);
    CqsDeqEnqueueFifo(d, data);
    break;
  case CQS_QUEUEING_ILIFO:
    iprio=prioptr[0]+(1U<<(CINTBITS-1));
    if ((int)iprio<0)
      d=CqsPrioqGetDeq(&(q->posprioq), CINTBITS, (unsigned int*)&iprio);
    else d=CqsPrioqGetDeq(&(q->negprioq), CINTBITS, (unsigned int*)&iprio);
    CqsDeqEnqueueLifo(d, data);
    break;
  case CQS_QUEUEING_BFIFO:
    if (priobits&&(((int)(prioptr[0]))<0))
       d=CqsPrioqGetDeq(&(q->posprioq), priobits, prioptr);
    else d=CqsPrioqGetDeq(&(q->negprioq), priobits, prioptr);
    CqsDeqEnqueueFifo(d, data);
    break;
  case CQS_QUEUEING_BLIFO:
    if (priobits&&(((int)(prioptr[0]))<0))
       d=CqsPrioqGetDeq(&(q->posprioq), priobits, prioptr);
    else d=CqsPrioqGetDeq(&(q->negprioq), priobits, prioptr);
    CqsDeqEnqueueLifo(d, data);
    break;

    /* The following two cases have a 64bit integer as priority value. Therefore,
     * we can cast the address of the CmiInt8 lprio to an "unsigned int" pointer
     * when passing it to CqsPrioqGetDeq. The endianness is taken care explicitly.
     */
  case CQS_QUEUEING_LFIFO:     
    CmiAssert(priobits == CLONGBITS);
    lprio0 =((CmiInt8 *)prioptr)[0];
    lprio0 += (1ULL<<(CLONGBITS-1));
    if (CmiEndianness() == 0) {           /* little-endian */
      lprio =(((CmiUInt4 *)&lprio0)[0]*1LL)<<CINTBITS | ((CmiUInt4 *)&lprio0)[1];
    }
    else {                /* little-endian */
      lprio = lprio0;
    }
    if (lprio0<0)
        d=CqsPrioqGetDeq(&(q->posprioq), priobits, (unsigned int *)&lprio);
    else
        d=CqsPrioqGetDeq(&(q->negprioq), priobits, (unsigned int *)&lprio);
    CqsDeqEnqueueFifo(d, data);
    break;
  case CQS_QUEUEING_LLIFO:
    lprio0 =((CmiInt8 *)prioptr)[0];
    lprio0 += (1ULL<<(CLONGBITS-1));
    if (CmiEndianness() == 0) {           /* little-endian happen to compare least significant part first */
      lprio =(((CmiUInt4 *)&lprio0)[0]*1LL)<<CINTBITS | ((CmiUInt4 *)&lprio0)[1];
    }
    else {                /* little-endian */
      lprio = lprio0;
    }
    if (lprio0<0)
        d=CqsPrioqGetDeq(&(q->posprioq), priobits, (unsigned int *)&lprio);
    else
        d=CqsPrioqGetDeq(&(q->negprioq), priobits, (unsigned int *)&lprio);
    CqsDeqEnqueueLifo(d, data);
    break;
  default:
    CmiAbort("CqsEnqueueGeneral: invalid queueing strategy.\n");
  }
  q->length++; if (q->length>q->maxlen) q->maxlen=q->length;
}

void CqsEnqueueFifo(Queue q, void *data)
{
  CqsDeqEnqueueFifo(&(q->zeroprio), data);
  q->length++; if (q->length>q->maxlen) q->maxlen=q->length;
}

void CqsEnqueueLifo(Queue q, void *data)
{
  CqsDeqEnqueueLifo(&(q->zeroprio), data);
  q->length++; if (q->length>q->maxlen) q->maxlen=q->length;
}

void CqsEnqueue(Queue q, void *data)
{
  CqsDeqEnqueueFifo(&(q->zeroprio), data);
  q->length++; if (q->length>q->maxlen) q->maxlen=q->length;
}

void CqsDequeue(Queue q, void **resp)
{
#ifdef ADAPT_SCHED_MEM
    /* Added by Isaac for testing purposes: */
    if((q->length > 1) && (CmiMemoryUsage() > schedAdaptMemThresholdMB*1024*1024) ){
	/* CqsIncreasePriorityForEntryMethod(q, 153); */
	CqsIncreasePriorityForMemCriticalEntries(q); 
    }
#endif
    
  if (q->length==0) 
    { *resp = 0; return; }
  if (q->negprioq.heapnext>1)
    { *resp = CqsPrioqDequeue(&(q->negprioq)); q->length--; return; }
  if (q->zeroprio.head != q->zeroprio.tail)
    { *resp = CqsDeqDequeue(&(q->zeroprio)); q->length--; return; }
  if (q->posprioq.heapnext>1)
    { *resp = CqsPrioqDequeue(&(q->posprioq)); q->length--; return; }
  *resp = 0; return;
}

#endif // CMK_USE_STL_MSGQ
static struct prio_struct kprio_zero = { 0, 0, {0} };
static struct prio_struct kprio_max  = { 32, 1, {((unsigned int)(-1))} };

_prio CqsGetPriority(Queue q)
{
  if (q->negprioq.heapnext>1) return &(q->negprioq.heap[1]->pri);
  if (q->zeroprio.head != q->zeroprio.tail) { return &kprio_zero; }
  if (q->posprioq.heapnext>1) return &(q->posprioq.heap[1]->pri);
  return &kprio_max;
}


/* prio CqsGetSecondPriority(q) */
/* Queue q; */
/* { */
/*   return CqsGetPriority(q); */
/* } */


/** Produce an array containing all the entries in a deq
    @return a newly allocated array filled with copies of the (void*) elements in the deq. 
    @param [in] q a deq
    @param [out] num the number of pointers in the returned array
*/
void** CqsEnumerateDeq(_deq q, int *num){
  void **head, **tail;
  void **result;
  int count = 0;
  int i;

  head = q->head;
  tail = q->tail;

  while(head != tail){
    count++;
    head++;
    if(head == q->end)
      head = q->bgn;
  }

  result = (void **)CmiAlloc(count * sizeof(void *));
  i = 0;
  head = q->head;
  tail = q->tail;
  while(head != tail){
    result[i] = *head;
    i++;
    head++;
    if(head == q->end)
      head = q->bgn;
  }
  *num = count;
  return(result);
}

/** Produce an array containing all the entries in a prioq
    @return a newly allocated array filled with copies of the (void*) elements in the prioq. 
    @param [in] q a deq
    @param [out] num the number of pointers in the returned array
*/
void** CqsEnumeratePrioq(_prioq q, int *num){
  void **head, **tail;
  void **result;
  int i,j;
  int count = 0;
  _prioqelt pe;

  for(i = 1; i < q->heapnext; i++){
    pe = (q->heap)[i];
    head = pe->data.head;
    tail = pe->data.tail;
    while(head != tail){
      count++;
      head++;
      if(head == (pe->data).end)
	head = (pe->data).bgn;
    }
  }

  result = (void **)CmiAlloc((count) * sizeof(void *));
  *num = count;
  
  j = 0;
  for(i = 1; i < q->heapnext; i++){
    pe = (q->heap)[i];
    head = pe->data.head;
    tail = pe->data.tail;
    while(head != tail){
      result[j] = *head;
      j++;
      head++;
      if(head ==(pe->data).end)
	head = (pe->data).bgn; 
    }
  }

  return result;
}

#if CMK_USE_STL_MSGQ
void CqsEnumerateQueue(Queue q, void ***resp){
  conv::msgQ<prio_t> *stlQ = (conv::msgQ<prio_t>*) q->stlQ;
  *resp = (void **)CmiAlloc(stlQ->size() * sizeof(conv::msg_t*));
  stlQ->enumerate(*resp, *resp + stlQ->size());
}

#else
void CqsEnumerateQueue(Queue q, void ***resp){
  void **result;
  int num;
  int i,j;

  *resp = (void **)CmiAlloc(q->length * sizeof(void *));
  j = 0;

  result = CqsEnumeratePrioq(&(q->negprioq), &num);
  for(i = 0; i < num; i++){
    (*resp)[j] = result[i];
    j++;
  }
  CmiFree(result);
  
  result = CqsEnumerateDeq(&(q->zeroprio), &num);
  for(i = 0; i < num; i++){
    (*resp)[j] = result[i];
    j++;
  }
  CmiFree(result);

  result = CqsEnumeratePrioq(&(q->posprioq), &num);
  for(i = 0; i < num; i++){
    (*resp)[j] = result[i];
    j++;
  }
  CmiFree(result);
}
#endif

/**
   Remove first occurence of a specified entry from the deq  by
   setting the entry to NULL.

   The size of the deq will not change, it will now just contain an
   entry for a NULL pointer.

   @return number of entries that were replaced with NULL
*/
int CqsRemoveSpecificDeq(_deq q, const void *msgPtr){
  void **head, **tail;

  head = q->head;
  tail = q->tail;

  while(head != tail){
    if(*head == msgPtr){
      /*    CmiPrintf("Replacing %p in deq with NULL\n", msgPtr); */
      /*     *head = NULL;  */
      return 1;
    }
    head++;
    if(head == q->end)
      head = q->bgn;
  }
  return 0;
}

/**
   Remove first occurence of a specified entry from the prioq by
   setting the entry to NULL.

   The size of the prioq will not change, it will now just contain an
   entry for a NULL pointer.

   @return number of entries that were replaced with NULL
*/
int CqsRemoveSpecificPrioq(_prioq q, const void *msgPtr){
  void **head, **tail;
  void **result;
  int i;
  _prioqelt pe;

  for(i = 1; i < q->heapnext; i++){
    pe = (q->heap)[i];
    head = pe->data.head;
    tail = pe->data.tail;
    while(head != tail){
      if(*head == msgPtr){
	/*	CmiPrintf("Replacing %p in prioq with NULL\n", msgPtr); */
	*head = NULL;
	return 1;
      }     
      head++;
      if(head == (pe->data).end)
	head = (pe->data).bgn;
    }
  } 
  return 0;
}

void CqsRemoveSpecific(Queue q, const void *msgPtr){
  if( CqsRemoveSpecificPrioq(&(q->negprioq), msgPtr) == 0 )
    if( CqsRemoveSpecificDeq(&(q->zeroprio), msgPtr) == 0 )  
      if(CqsRemoveSpecificPrioq(&(q->posprioq), msgPtr) == 0){
	CmiPrintf("Didn't remove the specified entry because it was not found\n");
      }  
}

#ifdef ADAPT_SCHED_MEM
int numMemCriticalEntries=0;
int *memCriticalEntries=NULL;
#endif

/** @} */
