/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include "converse.h"

typedef struct _ccd_callback {
  CcdVoidFn fn;
  void *arg;
} ccd_callback;

typedef struct _ccd_cblist_elem {
  ccd_callback cb;
  int next;
  int prev;
} ccd_cblist_elem;

typedef struct _ccd_cblist {
  unsigned int maxlen;
  unsigned int len;
  int first, last;
  int first_free;
  ccd_cblist_elem *elems;
} ccd_cblist;

/* initializes a callback list to the maximum length of ml.
 */
static void init_cblist(ccd_cblist *l, unsigned int ml)
{
  int i;
  l->elems = (ccd_cblist_elem*) malloc(ml*sizeof(ccd_cblist_elem));
  _MEMCHECK(l->elems);
  for(i=0;i<ml;i++) {
    l->elems[i].next = i+1;
    l->elems[i].prev = i-1;
  }
  l->elems[ml-1].next = -1;
  l->len = 0;
  l->maxlen = ml;
  l->first = l->last = -1;
  l->first_free = 0;
}

/* expand the callback list to a max length of ml
 */
static void expand_cblist(ccd_cblist *l, unsigned int ml)
{
  ccd_cblist_elem *old_elems = l->elems;
  int i = 0;
  int idx;
  l->elems = (ccd_cblist_elem*) malloc(ml*sizeof(ccd_cblist_elem));
  _MEMCHECK(l->elems);
  for(i=0;i<(l->len);i++)
    l->elems[i] = old_elems[idx];
  free(old_elems);
  for(i=l->len;i<ml;i++) {
    l->elems[i].next = i+1;
    l->elems[i].prev = i-1;
  }
  l->elems[ml-1].next = -1;
  l->elems[l->len].prev = -1;
  l->maxlen = ml;
  l->first_free = l->len;
}

/* remove element referred to by given list index idx.
 */
static void remove_elem(ccd_cblist *l, int idx)
{
  ccd_cblist_elem *e = l->elems;
  /* remove lidx from the busy list */
  if(e[idx].next != (-1))
    e[e[idx].next].prev = e[idx].prev;
  if(e[idx].prev != (-1))
    e[e[idx].prev].next = e[idx].next;
  if(idx==(l->first)) 
    l->first = e[idx].next;
  if(idx==(l->last)) 
    l->last = e[idx].prev;
  /* put lidx in the free list */
  e[idx].prev = -1;
  e[idx].next = l->first_free;
  if(e[idx].next != (-1))
    e[e[idx].next].prev = idx;
  l->first_free = idx;
  l->len--;
}

/* remove n elements from the beginning of the list.
 */
static void remove_n_elems(ccd_cblist *l, int n)
{
  int i;
  if(n==0 || (l->len < n))
    return;
  for(i=0;i<n;i++) {
    remove_elem(l, l->first);
  }
}

/* append callback to the given cblist, and return the index.
 */
static int append_elem(ccd_cblist *l, CcdVoidFn fn, void *arg)
{
  register int idx;
  register ccd_cblist_elem *e;
  if(l->len == l->maxlen)
    expand_cblist(l, l->maxlen*2);
  idx = l->first_free;
  e = l->elems;
  l->first_free = e[idx].next;
  e[idx].next = -1;
  e[idx].prev = l->last;
  if(l->first == (-1))
    l->first = idx;
  if(l->last != (-1))
    e[l->last].next = idx;
  l->last = idx;
  e[idx].cb.fn = fn;
  e[idx].cb.arg = arg;
  l->len++;
  return idx;
}

/* call functions on the cblist. functions that are added after the call 
 * cblist is started (e.g. callbacks registered from other callbacks) are 
 * ignored. callbacks are kept in the list even after they are called.
 * Note: it is illegal to cancel callbacks from within ccd callbacks.
 */
static void call_cblist_keep(ccd_cblist *l)
{
  int i, len = l->len, idx;
  for(i=0, idx=l->first;i<len;i++) {
    (*(l->elems[idx].cb.fn))(l->elems[idx].cb.arg);
    idx = l->elems[idx].next;
  }
}

/* call functions on the cblist. functions that are added after the call 
 * cblist is started (e.g. callbacks registered from other callbacks) are 
 * ignored. callbacks are removed from the list after they are called.
 * Note: it is illegal to cancel callbacks from within ccd callbacks.
 */
static void call_cblist_remove(ccd_cblist *l)
{
  int i, len = l->len, idx;
  for(i=0, idx=l->first;i<len;i++) {
    (*(l->elems[idx].cb.fn))(l->elems[idx].cb.arg);
    idx = l->elems[idx].next;
  }
  remove_n_elems(l,len);
}

#define CBLIST_INIT_LEN   8
#define MAXNUMCONDS       512

typedef struct {
  ccd_cblist condcb[MAXNUMCONDS];
  ccd_cblist condcb_keep[MAXNUMCONDS];
} ccd_cond_callbacks;

CpvStaticDeclare(ccd_cond_callbacks, conds);   

typedef struct {
	int nSkip;/*Number of opportunities to skip*/
	double lastCheck;/*Time of last check*/
	ccd_cblist periodic;
	ccd_cblist keep;
} ccd_periodic_callbacks;

CpvStaticDeclare(ccd_periodic_callbacks, pcb);
CpvDeclare(int, _ccd_numchecks);

#define MAXTIMERHEAPENTRIES       512

typedef struct {
    double time;
    ccd_callback cb;
} ccd_heap_elem;


/* Note : The heap is only stored in elements ccd_heap[0] to 
 * ccd_heap[ccd_heaplen]
 */

CpvStaticDeclare(ccd_heap_elem*, ccd_heap); 
CpvStaticDeclare(int, ccd_heaplen);

static void ccd_heap_swap(int index1, int index2)
{
  ccd_heap_elem *h = CpvAccess(ccd_heap);
  ccd_heap_elem temp;
  
  temp = h[index1];
  h[index1] = h[index2];
  h[index2] = temp;
}

static void ccd_heap_insert(double t, CcdVoidFn fnp, void *arg)
{
  int child, parent;
  ccd_heap_elem *h = CpvAccess(ccd_heap);
  
  if(CpvAccess(ccd_heaplen) > MAXTIMERHEAPENTRIES) {
    CmiAbort("Heap overflow (InsertInHeap), exiting...\n");
  } else {
    ccd_heap_elem *e = &(h[++CpvAccess(ccd_heaplen)]);
    e->time = t;
    e->cb.fn = fnp;
    e->cb.arg = arg;
    child  = CpvAccess(ccd_heaplen);    
    parent = child / 2;
    while((parent>0) && (h[child].time<h[parent].time)) {
	    ccd_heap_swap(child, parent);
	    child  = parent;
	    parent = parent / 2;
    }
  }
}

/* remove the top of the heap
 */
static void ccd_heap_remove(void)
{
  int parent,child;
  ccd_heap_elem *h = CpvAccess(ccd_heap);
  
  parent = 1;
  if(CpvAccess(ccd_heaplen)>0) {
    /* put deleted value at end of heap */
    ccd_heap_swap(1,CpvAccess(ccd_heaplen)); 
    CpvAccess(ccd_heaplen)--;
    if(CpvAccess(ccd_heaplen)) {
      /* if any left, then bubble up values */
	    child = 2 * parent;
	    while(child <= CpvAccess(ccd_heaplen)) {
	      if(((child + 1) <= CpvAccess(ccd_heaplen))  &&
		       (h[child].time > h[child+1].time))
                child++; /* use the smaller of the two */
	      if(h[parent].time <= h[child].time) 
		      break;
	      ccd_heap_swap(parent,child);
	      parent  = child;      /* go down the tree one more step */
	      child  = 2 * child;
      }
    }
  } 
}

/* If any of the CallFnAfter functions can now be called, call them 
 */
static void ccd_heap_update(double ctime)
{
  ccd_heap_elem *h = CpvAccess(ccd_heap);
  while ((CpvAccess(ccd_heaplen) > 0) && (h[1].time < ctime)) {
      (*(h[1].cb.fn))(h[1].cb.arg);
      ccd_heap_remove();
  }
}

void CcdModuleInit(void)
{
   int i;

   CpvInitialize(ccd_heap_elem*, ccd_heap);
   CpvInitialize(ccd_cond_callbacks, conds);
   CpvInitialize(ccd_periodic_callbacks, pcb);
   CpvInitialize(int, ccd_heaplen);
   CpvInitialize(int, _ccd_numchecks);

   CpvAccess(ccd_heap) = 
     (ccd_heap_elem*) malloc(sizeof(ccd_heap_elem)*(MAXTIMERHEAPENTRIES + 1));
   _MEMCHECK(CpvAccess(ccd_heap));
   CpvAccess(ccd_heaplen) = 0;
   for(i=0;i<MAXNUMCONDS;i++) {
     init_cblist(&(CpvAccess(conds).condcb[i]), CBLIST_INIT_LEN);
     init_cblist(&(CpvAccess(conds).condcb_keep[i]), CBLIST_INIT_LEN);
   }
   CpvAccess(_ccd_numchecks) = 10;
   CpvAccess(pcb).nSkip = 10;
   CpvAccess(pcb).lastCheck = CmiWallTimer();
   init_cblist(&(CpvAccess(pcb).periodic), CBLIST_INIT_LEN);
   init_cblist(&(CpvAccess(pcb).keep), CBLIST_INIT_LEN);
}



/* Add a function that will be called when a particular condition is raised
 */
int CcdCallOnCondition(int condnum, CcdVoidFn fnp, void *arg)
{
  return append_elem(&(CpvAccess(conds).condcb[condnum]), fnp, arg);
} 

int CcdCallOnConditionKeep(int condnum, CcdVoidFn fnp, void *arg)
{
  return append_elem(&(CpvAccess(conds).condcb_keep[condnum]), fnp, arg);
} 

void CcdCancelCallOnCondition(int condnum, int idx)
{
  remove_elem(&(CpvAccess(conds).condcb[condnum]), idx);
}

void CcdCancelCallOnConditionKeep(int condnum, int idx)
{
  remove_elem(&(CpvAccess(conds).condcb_keep[condnum]), idx);
}

/* Add a function that will be called during next call to PeriodicChecks
 */
int CcdPeriodicCall(CcdVoidFn fnp, void *arg)
{
  return append_elem(&(CpvAccess(pcb).periodic), fnp, arg);
}

int CcdPeriodicCallKeep(CcdVoidFn fnp, void *arg)
{
  return append_elem(&(CpvAccess(pcb).keep), fnp, arg);
}

void CcdCancelPeriodicCall(int idx)
{
  remove_elem(&(CpvAccess(pcb).periodic), idx);
}

void CcdCancelPeriodicCallKeep(int idx)
{
  remove_elem(&(CpvAccess(pcb).keep), idx);
}

/* Call the function with the provided argument after a minimum delay of deltaT
 */
void CcdCallFnAfter(CcdVoidFn fnp, void *arg, unsigned int deltaT)
{
  double ctime  = CmiWallTimer();
  double tcall = ctime + (double)deltaT/1000.0;
  ccd_heap_insert(tcall, fnp, arg);
} 

/* Call all the functions that are waiting for this condition to be raised
 */
void CcdRaiseCondition(int condnum)
{
  call_cblist_remove(&(CpvAccess(conds).condcb[condnum]));
  call_cblist_keep(&(CpvAccess(conds).condcb_keep[condnum]));
}

/* call functions to be called periodically, and also the time-indexed
 * functions if their time has arrived
 */
void CcdCallBacks(void)
{
  ccd_periodic_callbacks *o=&CpvAccess(pcb);
  
  /* Figure out how many times to skip Ccd processing */
  double currTime = CmiWallTimer();
  double elapsed = currTime - o->lastCheck;
  if (elapsed>0) /* Try to wait about 5 ms between time checks */
     o->nSkip = (int)(5.0e-3*o->nSkip/elapsed);
  else
    o->nSkip *= 2;
  CpvAccess(_ccd_numchecks) = o->nSkip;
  o->lastCheck=currTime;
  
  ccd_heap_update(currTime);
    
  call_cblist_remove(&(o->periodic));
  call_cblist_keep(&(o->keep));
} 

