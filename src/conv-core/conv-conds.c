/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile$
 *	$Author$	$Locker$		$State$
 *	$Revision$	$Date$
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 * $Log$
 * Revision 2.6  1995-07-10 07:03:07  narain
 * Made the timer field double
 *
 * Revision 2.5  1995/07/07  14:42:49  gursoy
 * fixed a bug (initialized PeriodicCalls to NULL)
 *
 * Revision 2.4  1995/06/26  19:46:14  sanjeev
 * removed CkExit
 *
 * Revision 2.3  1995/06/19  17:45:44  sanjeev
 * bug in PeriodicChecks
 *
 * Revision 2.2  1995/06/19  16:36:04  sanjeev
 * Integrated TimerChecks and PeriodicChecks
 *
 * Revision 2.1  1995/06/18  21:56:02  sanjeev
 * *** empty log message ***
 *
 * Revision 2.0  1995/06/18  21:25:44  sanjeev
 * *** empty log message ***
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";


#include <stdio.h>

#include "converse.h"
#include "conv-mach.h"
#include "conv-conds.h"

/**** these variables are local to this file but are persistant *****/

/** Note : The heap is only stored in elements 
                  timerHeap[0] to timerHeap[numHeapEntries] */



CpvStaticDeclare(HeapIndexType*, timerHeap); 
CpvStaticDeclare(CONDS*, CondArr);   
CpvStaticDeclare(FN_ARG*, PeriodicCalls);
CpvStaticDeclare(int, numHeapEntries);

CpvDeclare(int, CcdNumChecks);

extern double CmiTimer();

void conv_condsModuleInit()
{
   int i;

   CpvInitialize(HeapIndexType*, timerHeap);
   CpvInitialize(CONDS*, CondArr);
   CpvInitialize(FN_ARG*, PeriodicCalls);
   CpvInitialize(int, numHeapEntries);
   CpvInitialize(int, CcdNumChecks);

   CpvAccess(timerHeap) = (HeapIndexType*) CmiAlloc(sizeof(HeapIndexType)*(MAXTIMERHEAPENTRIES + 1));
   CpvAccess(CondArr) = (CONDS*) CmiAlloc(sizeof(CONDS)*(MAXCONDCHKARRAYELTS));
  
   CpvAccess(CcdNumChecks) = 0;
   CpvAccess(numHeapEntries) = 0;
   CpvAccess(PeriodicCalls) = (FN_ARG *) NULL;
   for(i = 0; i < MAXCONDCHKARRAYELTS; i++) {
     CpvAccess(CondArr)[i].fn_arg_list = NULL;
   }
}



/*****************************************************************************
  Add a function that will be called when a particular condition is raised
 *****************************************************************************/
void CcdCallOnCondition(int condnum, FN_PTR fnp, void *arg)
{
  FN_ARG *newEntry;
  
  if((newEntry = (FN_ARG *) CmiAlloc(sizeof(FN_ARG))) == NULL) 
    {
      CmiError("CallOnCondition: Allocation Failed");
      return;
    }
  else 
    {
      newEntry->fn = fnp;  
      newEntry->arg  = arg;
      newEntry->next = CpvAccess(CondArr)[condnum].fn_arg_list;
      CpvAccess(CondArr)[condnum].fn_arg_list =  newEntry;
    }
} 

/*****************************************************************************
  Add a function that will be called during each call to PeriodicChecks
 *****************************************************************************/
void CcdPeriodicallyCall(FN_PTR fnp, void *arg)
{
  FN_ARG *temp;
  
  if((temp = (FN_ARG *) CmiAlloc(sizeof(FN_ARG))) == NULL) 
    {
      CmiError("PeriodicallyCall: Allocation failed");
      return;
    }
  temp->fn = fnp;
  temp->arg = arg;
  
  temp->next = CpvAccess(PeriodicCalls);
  CpvAccess(PeriodicCalls) = temp;
  CpvAccess(CcdNumChecks)++;
}

/*****************************************************************************
  Call all the functions that are waiting for this condition to be raised
 *****************************************************************************/
void CcdRaiseCondition(condNum)
     int condNum;
{
  FN_ARG *temp, *del;
  temp = CpvAccess(CondArr)[condNum].fn_arg_list;
  while(temp)
    {
      (*(temp->fn))(temp->arg); /* Any freeing of the argument structure should
				   be done here */
      del = temp;
      temp = temp->next;
      free(del);
    }
  CpvAccess(CondArr)[condNum].fn_arg_list = NULL;
}

/*****************************************************************************
  Call the function with the provided argument after a minimum delay of deltaT
 *****************************************************************************/
void CcdCallFnAfter(FN_PTR fnp, void *arg, unsigned int deltaT)
{
  double tPrime, currT;
  currT  = CmiTimer();                    /* get current time */
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
  
/* This was formerly TimerChecks() */
  if ( CpvAccess(numHeapEntries) > 0 ) {
    currTime = CmiTimer();
  
    while ((CpvAccess(numHeapEntries) > 0) && CpvAccess(timerHeap)[1].timeVal < currTime)
    {
      (*(CpvAccess(timerHeap)[1].fn))(CpvAccess(timerHeap)[1].arg);
      RemoveFromHeap(1);
    }
  }

/* This was formerly PeriodicChecks() */
/* Call the functions that have been added to the list of periodic functions */

  temp = CpvAccess(PeriodicCalls); 
  CpvAccess(PeriodicCalls) = NULL ;

  for(; temp; temp = next) {
    CpvAccess(CcdNumChecks)--;

    (*(temp->fn))(temp->arg);

    next = temp->next ;
    CmiFree(temp) ;
  }
} 

/*****************************************************************************
  These are internal functions
  ****************************************************************************/

static void InsertInHeap(double theTime, FN_PTR fnp, void *arg)
{
  int child, parent;
  
  if(CpvAccess(numHeapEntries) > MAXTIMERHEAPENTRIES) 
    {
      CmiPrintf("Heap overflow (InsertInHeap), exiting...\n");
      exit();
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
      exit();
    } 
  else 
    {
      CpvAccess(timerHeap)[index].arg = NULL;
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





