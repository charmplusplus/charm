/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <converse.h>
#include <string.h>
#include "queueing.h"

static void CqsDeqInit(d)
deq d;
{
  d->bgn  = d->space;
  d->end  = d->space+4;
  d->head = d->space;
  d->tail = d->space;
}

static void CqsDeqExpand(d)
deq d;
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

void CqsDeqEnqueueFifo(d, data)
deq d; void *data;
{
  void **tail = d->tail;
  *tail = data;
  tail++;
  if (tail == d->end) tail = d->bgn;
  d->tail = tail;
  if (tail == d->head) CqsDeqExpand(d);
}

void CqsDeqEnqueueLifo(d, data)
deq d; void *data;
{
  void **head = d->head;
  if (head == d->bgn) head = d->end;
  head--;
  *head = data;
  d->head = head;
  if (head == d->tail) CqsDeqExpand(d);
}

void *CqsDeqDequeue(d)
deq d;
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

static void CqsPrioqInit(pq)
prioq pq;
{
  int i;
  pq->heapsize = 100;
  pq->heapnext = 1;
  pq->hash_key_size = PRIOQ_TABSIZE;
  pq->hash_entry_size = 0;
  pq->heap = (prioqelt *)CmiAlloc(100 * sizeof(prioqelt));
  pq->hashtab = (prioqelt *)CmiAlloc(pq->hash_key_size * sizeof(prioqelt));
  for (i=0; i<pq->hash_key_size; i++) pq->hashtab[i]=0;
}

#if CMK_C_INLINE
inline
#endif
static void CqsPrioqExpand(pq)
prioq pq;
{
  int oldsize = pq->heapsize;
  int newsize = oldsize * 2;
  prioqelt *oheap = pq->heap;
  prioqelt *nheap = (prioqelt *)CmiAlloc(newsize*sizeof(prioqelt));
  memcpy(nheap, oheap, oldsize * sizeof(prioqelt));
  pq->heap = nheap;
  pq->heapsize = newsize;
  CmiFree(oheap);
}
#ifndef FASTQ
void CqsPrioqRehash(pq)
     prioq pq;
{
  int oldHsize = pq->hash_key_size;
  int newHsize = oldHsize * 2;
  unsigned int hashval;
  prioqelt pe, pe1, pe2;
  int i,j;

  prioqelt *ohashtab = pq->hashtab;
  prioqelt *nhashtab = (prioqelt *)CmiAlloc(newHsize*sizeof(prioqelt));

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
/*
 * This routine compares priorities. It returns:
 * 
 * 1 if prio1 > prio2
 * ? if prio1 == prio2
 * 0 if prio1 < prio2
 *
 * where prios are treated as unsigned
 */

int CqsPrioGT(prio1, prio2)
prio prio1;
prio prio2;
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

deq CqsPrioqGetDeq(pq, priobits, priodata)
prioq pq;
unsigned int priobits, *priodata;
{
  unsigned int prioints = (priobits+CINTBITS-1)/CINTBITS;
  unsigned int hashval, i;
  int heappos; 
  prioqelt *heap, pe, next, parent;
  prio pri;
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
  pe = (prioqelt)CmiAlloc(sizeof(struct prioqelt_struct)+((prioints-1)*sizeof(int)));
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
    prioqelt parent = heap[parentpos];
    if (CqsPrioGT(pri, &(parent->pri))) break;
    heap[heappos] = parent; heappos=parentpos;
  }
  heap[heappos] = pe;

#ifdef FASTQ
  /*  printf("Hi I'm here222\n"); */
#endif
  
  return &(pe->data);
}

void *CqsPrioqDequeue(pq)
prioq pq;
{
  prio pri;
  prioqelt pe, old; void *data;
  int heappos, heapnext;
  prioqelt *heap = pq->heap;
  int left_child;
  prioqelt temp1_ht_right, temp1_ht_left, temp1_ht_parent;
  prioqelt *temp1_ht_handle;
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
    prioqelt next = pe->ht_next;
    prioqelt *handle = pe->ht_handle;
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
      prioqelt ch1, ch2, child;
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
  return q;
}

void CqsDelete(Queue q)
{
  CmiFree(q->negprioq.heap);
  CmiFree(q->posprioq.heap);
  CmiFree(q);
}

unsigned int CqsLength(Queue q)
{
  return q->length;
}

unsigned int CqsMaxLength(Queue q)
{
  return q->maxlen;
}

int CqsEmpty(Queue q)
{
  return (q->length == 0);
}

void CqsEnqueueGeneral(Queue q, void *data, int strategy, 
           int priobits,unsigned int *prioptr)
{
  deq d; int iprio;
  CmiInt8 *lprio;
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
      d=CqsPrioqGetDeq(&(q->posprioq), CINTBITS, &iprio);
    else d=CqsPrioqGetDeq(&(q->negprioq), CINTBITS, &iprio);
    CqsDeqEnqueueFifo(d, data);
    break;
  case CQS_QUEUEING_ILIFO:
    iprio=prioptr[0]+(1U<<(CINTBITS-1));
    if ((int)iprio<0)
      d=CqsPrioqGetDeq(&(q->posprioq), CINTBITS, &iprio);
    else d=CqsPrioqGetDeq(&(q->negprioq), CINTBITS, &iprio);
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

  case CQS_QUEUEING_LFIFO:     
    /* allow signed priority queueing on 64 bit integers */
    lprio =(CmiInt8 *)prioptr;
    if (*lprio<0)
      {
	d=CqsPrioqGetDeq(&(q->negprioq), priobits, prioptr);
      }
    else
      {
	d=CqsPrioqGetDeq(&(q->posprioq), priobits, prioptr);
      }
    CqsDeqEnqueueFifo(d, data);
    break;
  case CQS_QUEUEING_LLIFO:
    lprio =(CmiInt8 *)prioptr;
    if (*lprio<0)
      d=CqsPrioqGetDeq(&(q->negprioq), priobits, prioptr);
    else
      d=CqsPrioqGetDeq(&(q->posprioq), priobits, prioptr);
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

static struct prio_struct kprio_zero = { 0, 0, {0} };
static struct prio_struct kprio_max  = { 32, 1, {((unsigned int)(-1))} };

prio CqsGetPriority(q)
Queue q;
{
  if (q->negprioq.heapnext>1) return &(q->negprioq.heap[1]->pri);
  if (q->zeroprio.head != q->zeroprio.tail) { return &kprio_zero; }
  if (q->posprioq.heapnext>1) return &(q->posprioq.heap[1]->pri);
  return &kprio_max;
}

prio CqsGetSecondPriority(q)
Queue q;
{
  return CqsGetPriority(q);
}

void** CqsEnumerateDeq(deq q, int *num){
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

void** CqsEnumeratePrioq(prioq q, int *num){
  void **head, **tail;
  void **result;
  int i,j;
  int count = 0;
  prioqelt pe;

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

  result = (void **)CmiAlloc(count * sizeof(void *));
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

