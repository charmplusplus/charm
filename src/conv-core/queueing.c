#include <converse.h>
#include "queueing.h"

void CqsDeqInit(d)
deq d;
{
  d->bgn  = d->space;
  d->end  = d->space+4;
  d->head = d->space;
  d->tail = d->space;
}

void CqsDeqExpand(d)
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

void CqsPrioqInit(pq)
prioq pq;
{
  int i;
  pq->heapsize = 100;
  pq->heapnext = 1;
  pq->heap = (prioqelt *)CmiAlloc(100 * sizeof(prioqelt));
  for (i=0; i<PRIOQ_TABSIZE; i++) pq->hashtab[i]=0;
}

void CqsPrioqExpand(pq)
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

/*
 * This routine compares priorities. It returns:
 *
 * 1 if prio1 > prio2
 * ? if prio1 == prio2
 * 0 if prio1 < prio2
 *
 */

int CqsPrioGT(prio1, prio2)
prio prio1;
prio prio2;
{
  unsigned int ints1 = prio1->ints;
  unsigned int ints2 = prio2->ints;
  unsigned int *data1 = prio1->data;
  unsigned int *data2 = prio2->data;
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

deq CqsPrioqGetDeq(pq, priobits, priodata)
prioq pq;
unsigned int priobits, *priodata;
{
  unsigned int prioints = (priobits+CINTBITS-1)/CINTBITS;
  unsigned int hashval;
  int heappos, i, j; 
  prioqelt *heap, pe, next;
  prio pri;

  /* Scan for priority in hash-table, and return it if present */
  hashval = priobits;
  for (i=0; i<prioints; i++) hashval ^= priodata[i];
  hashval = (hashval&0x7FFFFFFF)%PRIOQ_TABSIZE;
  for (pe=pq->hashtab[hashval]; pe; pe=pe->ht_next)
    if (priobits == pe->pri.bits)
      if (memcmp(priodata, pe->pri.data, sizeof(int)*prioints)==0)
	return &(pe->data);
  
  /* If not present, allocate a bucket for specified priority */
  pe = (prioqelt)CmiAlloc(sizeof(struct prioqelt_struct)+((prioints-1)*sizeof(int)));
  pe->pri.bits = priobits;
  pe->pri.ints = prioints;
  memcpy(pe->pri.data, priodata, (prioints*sizeof(int)));
  CqsDeqInit(&(pe->data));
  pri=&(pe->pri);

  /* Insert bucket into hash-table */
  next = pq->hashtab[hashval];
  pe->ht_next = next;
  pe->ht_handle = (pq->hashtab+hashval);
  if (next) next->ht_handle = &(pe->ht_next);
  pq->hashtab[hashval] = pe;
  
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
  
  return &(pe->data);
}

void *CqsPrioqDequeue(pq)
prioq pq;
{
  prio pri;
  prioqelt pe, old; void *data;
  int heappos, heapnext;
  prioqelt *heap = pq->heap;

  if (pq->heapnext==1) return 0;
  pe = heap[1];
  data = CqsDeqDequeue(&(pe->data));
  if (pe->data.head == pe->data.tail) {
    /* Unlink prio-bucket from hash-table */
    prioqelt next = pe->ht_next;
    prioqelt *handle = pe->ht_handle;
    if (next) next->ht_handle = handle;
    *handle = next;
    old=pe;
    
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

Queue CqsCreate()
{
  Queue q = (Queue)CmiAlloc(sizeof(struct Queue_struct));
  q->length = 0;
  q->maxlen = 0;
  CqsDeqInit(&(q->zeroprio));
  CqsPrioqInit(&(q->negprioq));
  CqsPrioqInit(&(q->posprioq));
  return q;
}

unsigned int CqsLength(q)
Queue q;
{
  return q->length;
}

unsigned int CqsMaxLength(q)
Queue q;
{
  return q->maxlen;
}

int CqsEmpty(q)
Queue q;
{
  return (q->length == 0);
}

void CqsEnqueueGeneral(q, data, strategy, priobits, prioptr)
Queue q; void *data; unsigned int strategy, priobits, *prioptr;
{
  deq d; int iprio;
  switch (strategy) {
  case CQS_QUEUEING_FIFO: 
    CqsDeqEnqueueFifo(&(q->zeroprio), data); 
    break;
  case CQS_QUEUEING_LIFO: 
    CqsDeqEnqueueLifo(&(q->zeroprio), data); 
    break;
  case CQS_QUEUEING_IFIFO:
    iprio=prioptr[0]+(1<<(CINTBITS-1));
    if ((int)iprio<0)
      d=CqsPrioqGetDeq(&(q->posprioq), CINTBITS, &iprio);
    else d=CqsPrioqGetDeq(&(q->negprioq), CINTBITS, &iprio);
    CqsDeqEnqueueFifo(d, data);
    break;
  case CQS_QUEUEING_ILIFO:
    iprio=prioptr[0]+(1<<(CINTBITS-1));
    if ((int)iprio<0)
      d=CqsPrioqGetDeq(&(q->posprioq), CINTBITS, &iprio);
    else d=CqsPrioqGetDeq(&(q->negprioq), CINTBITS, &iprio);
    CqsDeqEnqueueLifo(d, data);
    break;
  case CQS_QUEUEING_BFIFO:
    if (priobits&&(((int)(prioptr[1]))<0))
       d=CqsPrioqGetDeq(&(q->posprioq), priobits, prioptr);
    else d=CqsPrioqGetDeq(&(q->negprioq), priobits, prioptr);
    CqsDeqEnqueueFifo(d, data);
    break;
  case CQS_QUEUEING_BLIFO:
    if (priobits&&(((int)(prioptr[1]))<0))
       d=CqsPrioqGetDeq(&(q->posprioq), priobits, prioptr);
    else d=CqsPrioqGetDeq(&(q->negprioq), priobits, prioptr);
    CqsDeqEnqueueLifo(d, data);
    break;
  default:
    CmiError("CqsEnqueueGeneral: invalid queueing strategy.\n");
    exit(1);
  }
  q->length++; if (q->length>q->maxlen) q->maxlen=q->length;
}

void CqsEnqueueFifo(q, data)
Queue q; void *data;
{
  CqsDeqEnqueueFifo(&(q->zeroprio), data);
  q->length++; if (q->length>q->maxlen) q->maxlen=q->length;
}

void CqsEnqueueLifo(q, data)
Queue q; void *data;
{
  CqsDeqEnqueueLifo(&(q->zeroprio), data);
  q->length++; if (q->length>q->maxlen) q->maxlen=q->length;
}

void CqsEnqueue(q, data)
Queue q; void *data;
{
  CqsDeqEnqueueFifo(&(q->zeroprio), data);
  q->length++; if (q->length>q->maxlen) q->maxlen=q->length;
}

void CqsDequeue(q, resp)
Queue q;
void **resp;
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

