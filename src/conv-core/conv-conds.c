#include <stdio.h>

#include "converse.h"

static void InsertInHeap(double Time, CcdVoidFn fnp, void *arg);
static void RemoveFromHeap(int index);
static void SwapHeapEntries(int index1, int index2);

typedef struct fn_arg {
  CcdVoidFn fn;
  void *arg;
  struct fn_arg *next;
} FN_ARG;

/* We have a fixed number of these elements .. */
typedef struct {
  FN_ARG *fn_arg_list;
} CONDS;

typedef struct {
    double timeVal;     /* the actual time value we sort on           */
    CcdVoidFn fn; 
    void *arg; 
} HeapIndexType;

#define MAXTIMERHEAPENTRIES       512
#define MAXCONDCHKARRAYELTS       512

/** Note : The heap is only stored in elements 
    timerHeap[0] to timerHeap[numHeapEntries] */

CpvStaticDeclare(HeapIndexType*, timerHeap); 
CpvStaticDeclare(CONDS*, CondArr);   
CpvStaticDeclare(FN_ARG*, PeriodicCalls);
CpvStaticDeclare(int, numHeapEntries);

CpvDeclare(int, CcdNumChecks);

extern double CmiWallTimer(void);

void CcdModuleInit(void)
{
   int i;

   CpvInitialize(HeapIndexType*, timerHeap);
   CpvInitialize(CONDS*, CondArr);
   CpvInitialize(FN_ARG*, PeriodicCalls);
   CpvInitialize(int, numHeapEntries);
   CpvInitialize(int, CcdNumChecks);

   CpvAccess(timerHeap) = 
     (HeapIndexType*) malloc(sizeof(HeapIndexType)*(MAXTIMERHEAPENTRIES + 1));
   CpvAccess(CondArr) = (CONDS*) malloc(sizeof(CONDS)*(MAXCONDCHKARRAYELTS));
   CpvAccess(CcdNumChecks) = 0;
   CpvAccess(numHeapEntries) = 0;
   CpvAccess(PeriodicCalls) = (FN_ARG *) 0;
   for(i=0; i<MAXCONDCHKARRAYELTS; i++)
     CpvAccess(CondArr)[i].fn_arg_list = 0;
}



/*****************************************************************************
  Add a function that will be called when a particular condition is raised
 *****************************************************************************/
void CcdCallOnCondition(int condnum,CcdVoidFn fnp,void *arg)
{
  FN_ARG *newEntry = (FN_ARG *) malloc(sizeof(FN_ARG)); 
  newEntry->fn = fnp;  
  newEntry->arg  = arg;
  newEntry->next = CpvAccess(CondArr)[condnum].fn_arg_list;
  CpvAccess(CondArr)[condnum].fn_arg_list =  newEntry;
} 

/*****************************************************************************
  Add a function that will be called during next call to PeriodicChecks
 *****************************************************************************/
void CcdPeriodicallyCall(CcdVoidFn fnp, void *arg)
{
  FN_ARG *temp = (FN_ARG *) malloc(sizeof(FN_ARG)); 
  temp->fn = fnp;
  temp->arg = arg;
  temp->next = CpvAccess(PeriodicCalls);
  CpvAccess(PeriodicCalls) = temp;
  CpvAccess(CcdNumChecks)++;
}

/*****************************************************************************
  Call all the functions that are waiting for this condition to be raised
 *****************************************************************************/
void CcdRaiseCondition(int condNum)
{
  FN_ARG *temp, *del;
  temp = CpvAccess(CondArr)[condNum].fn_arg_list;
  CpvAccess(CondArr)[condNum].fn_arg_list = 0;
  while(temp) {
    (*(temp->fn))(temp->arg);
    del = temp;
    temp = temp->next;
    free(del);
  }
}

/*****************************************************************************
  Call the function with the provided argument after a minimum delay of deltaT
 *****************************************************************************/
void CcdCallFnAfter(CcdVoidFn fnp, void *arg, unsigned int deltaT)
{
  double tPrime, currT;
  currT  = CmiWallTimer();                /* get current time */
  tPrime = currT + (double)deltaT/1000.0; /* add delta to determine what time
					    to actually execute fn */
  InsertInHeap(tPrime, fnp, arg); /* insert into tmr hp */
} 

/*****************************************************************************
  If any of the CallFnAfter functions can now be called, call them 
  ****************************************************************************/
void CcdCallBacks()
{
  double currTime;
  int index;
  int i,j;
  FN_ARG *temp, *next;
  
  if ( CpvAccess(numHeapEntries) > 0 ) {
    currTime = CmiWallTimer();
    while ((CpvAccess(numHeapEntries) > 0) && 
           CpvAccess(timerHeap)[1].timeVal < currTime)
    {
      (*(CpvAccess(timerHeap)[1].fn))(CpvAccess(timerHeap)[1].arg);
      RemoveFromHeap(1);
    }
  }

  temp = CpvAccess(PeriodicCalls); 
  CpvAccess(PeriodicCalls) = 0 ;
  for(; temp; temp = next) {
    CpvAccess(CcdNumChecks)--;
    (*(temp->fn))(temp->arg);
    next = temp->next ;
    free(temp) ;
  }
} 

/*****************************************************************************
  These are internal functions
  ****************************************************************************/

static void InsertInHeap(double theTime, CcdVoidFn fnp, void *arg)
{
  int child, parent;
  
  if(CpvAccess(numHeapEntries) > MAXTIMERHEAPENTRIES) 
    {
      CmiPrintf("Heap overflow (InsertInHeap), exiting...\n");
      exit(1);
    }
  else 
    {
      CpvAccess(CcdNumChecks)++;
      CpvAccess(numHeapEntries)++;
      CpvAccess(timerHeap)[CpvAccess(numHeapEntries)].timeVal    = theTime;
      CpvAccess(timerHeap)[CpvAccess(numHeapEntries)].fn = fnp;
      CpvAccess(timerHeap)[CpvAccess(numHeapEntries)].arg = arg;
      child  = CpvAccess(numHeapEntries);    
      parent = child / 2;
      while((parent>0) && (CpvAccess(timerHeap)[child].timeVal<CpvAccess(timerHeap)[parent].timeVal))
	{
	  SwapHeapEntries(child,parent);
	  child  = parent;
	  parent = parent / 2;
        }
    }
} 

static void RemoveFromHeap(int index)
{
  int parent,child;
  
  parent = index;
  if(!CpvAccess(numHeapEntries) || (index != 1)) 
    {
      CmiPrintf("Internal inconsistency (RemoveFromHeap), exiting ...\n");
      exit(1);
    } 
  else 
    {
      CpvAccess(timerHeap)[index].arg = 0;
      SwapHeapEntries(index,CpvAccess(numHeapEntries)); /* put deleted value at end 
						of heap */
      CpvAccess(numHeapEntries)--;
      CpvAccess(CcdNumChecks)--;
      if(CpvAccess(numHeapEntries)) 
	{             /* if any left, then bubble up values */
	  child = 2 * parent;
	  while(child <= CpvAccess(numHeapEntries)) 
	    {
	      if(((child + 1) <= CpvAccess(numHeapEntries))  &&
		 (CpvAccess(timerHeap)[child].timeVal > CpvAccess(timerHeap)[child + 1].timeVal))
                child++;              /* use the smaller of the two */
	      if(CpvAccess(timerHeap)[parent].timeVal <= CpvAccess(timerHeap)[child].timeVal) 
		break;
	      SwapHeapEntries(parent,child);
	      parent  = child;      /* go down the tree one more step */
	      child  = 2 * child;
            }
        }
    } 
} 

static void SwapHeapEntries(int index1, int index2)
{
  HeapIndexType temp;
  
  temp              = CpvAccess(timerHeap)[index1];
  CpvAccess(timerHeap)[index1] = CpvAccess(timerHeap)[index2];
  CpvAccess(timerHeap)[index2] = temp;
} 





