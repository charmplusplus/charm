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
 * Revision 2.1  1995-06-08 17:07:12  gursoy
 * Cpv macro changes done
 *
 * Revision 1.3  1995/04/23  00:46:38  sanjeev
 * changed STATIC to static
 *
 * Revision 1.2  1995/04/13  20:53:29  sanjeev
 * Changed Mc to Cmi
 *
 * Revision 1.1  1994/11/03  17:39:05  brunner
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";
#include "const.h"
#include "chare.h"
#include "globals.h"
#include "condsend.h"

/***************************************************************************
 
 This file contains functions pertaining to conditional sends. The functions
 in this file are:
***************************************************************************/

/********************** the user may call these functions *********************/
void SendMsgOnCondition(); /* call param function. If 1, send msg          */
void CallBocOnCondition(); /* call boc function. It handles situation         */
void SendMsgAfter();       /* put time in heap, TimerChecks will send msg     */
void CallBocAfter();       /* ditto, TimerChecks will call boc functin        */

/********************* these are private to the CK ***************************/
void SendMsgIfCondArises();
void CallBocIfCondArises();
void RaiseCondition();
int  NoDelayedMsgs();
void PeriodicChecks();     /* performs checks on various conditions           */
void TimerChecks();        /* check list of times against current, act acdngly*/
void CondSendInit();       /* initialize the data structures */

/********************** These are private to this file ************************/
void InsertInHeap();       /* self-explanatory                                */
void RemoveFromHeap();     /* ditto                                           */
void SwapHeapEntries();    /* ditto                                           */
unsigned int LowestTime();/* returns the lowest time in the heap            */


/**** these variables are local to this file but are persistant *****/

typedef HeapIndexType *HeapIndexType_;
typedef CondArrayEltType *CondArrayEltType_;
typedef IfCondArisesArrayEltType *IfCondArisesArrayEltType_;

CpvStaticDeclare(int, numRaiseCondArryElts); /* init to # of system cnds */
CpvStaticDeclare(HeapIndexType_,           timerHeap);
CpvStaticDeclare(CondArrayEltType_,      condChkArray);
CpvStaticDeclare(IfCondArisesArrayEltType_, ifCondArisesArray);




void condsendModuleInit()
{
     CpvInitialize(int, numRaiseCondArryElts);
     CpvInitialize(HeapIndexType_, timerHeap);
     CpvInitialize(CondArrayEltType_, condChkArray);
     CpvInitialize(IfCondArisesArrayEltType_, ifCondArisesArray);
}




/***************************************************************************
Author: Wayne Fenton
Date:   1/20/90

Info:  In this function we just create data structure to hold our information
       and add it to the  condChkArray. Later, in PeriodicChecks, it will
       check each condition, and if true, then it will send the corresponding
       message. 
***************************************************************************/
void SendMsgOnCondition(cond_fn, entry, msgToSend, size, pChareID)
FUNCTION_PTR   cond_fn;
int            entry;
void           *msgToSend;
int            size;
ChareIDType    *pChareID;
{
ChareDataEntry *newEntry;

if((newEntry = (ChareDataEntry *) CmiAlloc(sizeof(ChareDataEntry))) == NULL) {
    CkMemError(newEntry);
    return;
    }
else {
    newEntry->cond_fn   = cond_fn;   /* fill up the data record */
    newEntry->entry     = entry;
    newEntry->msg       = msgToSend;
    newEntry->size      = size;
    newEntry->chareID   = *pChareID;
    CpvAccess(condChkArray)[CpvAccess(numCondChkArryElts)].theData     = (void *) newEntry;
    CpvAccess(condChkArray)[CpvAccess(numCondChkArryElts)++].bocOrChare = ITSACHARE;
    }
} 


/***************************************************************************
Author: Wayne Fenton
Date:   1/20/90

Info:   just like SendMsgOnCondition, execpt that we are dealing with a 
        boc function call.
***************************************************************************/
void CallBocOnCondition(fn_ptr, bocNum)
FUNCTION_PTR fn_ptr;
int bocNum;
{
BocDataEntry *newEntry;

if((newEntry = (BocDataEntry *) CmiAlloc(sizeof(BocDataEntry))) == NULL) {
    CkMemError(newEntry);
    return;
    }
else {
    newEntry->bocNum   = bocNum;
    newEntry->fn_ptr   = fn_ptr;;

    CpvAccess(condChkArray)[CpvAccess(numCondChkArryElts)].theData     = (void *) newEntry;
    CpvAccess(condChkArray)[CpvAccess(numCondChkArryElts)++].bocOrChare = ITSABOC;
    }
}


/***************************************************************************
Author: Wayne Fenton
Date:   1/20/90

Info: We  just place the information into the linked list in the
      ifCondArisesArry that corresponds to the condID (ie this value
      is just an index into the array. Later, if the condition arises,
      then a call to RaiseCondition will set a flag in the array elts.
      Then, when PeriodicChecks is executed, it will check to see if
      the flag is set on any of the array elts, and if so, it will send
      a message (or call a boc function if CallBocIfCondArises was called)
      for each element on the linked list.
***************************************************************************/
void SendMsgIfCondArises(condNum,ep,msgToSend,size,pChareID)
int         condNum;
int         ep;
void        *msgToSend;
int         size;
ChareIDType *pChareID;
{ 
ChareDataEntry *newEntry;
LinkRec        *theLink, *listPtr,*lockStpPtr;

if((newEntry = (ChareDataEntry *) CmiAlloc(sizeof(ChareDataEntry))) == NULL) {
    CkMemError(newEntry);
    return;
    }
if((theLink = (LinkRec *) CmiAlloc(sizeof(LinkRec))) == NULL) {
    CkMemError(theLink);
    return;
    }
newEntry->entry     = ep;
newEntry->msg       = msgToSend;
newEntry->size      = size;
newEntry->chareID   = *pChareID;        /* fill up our data structure   */

theLink->theData    = (void *) newEntry;
theLink->bocOrChare = ITSACHARE;
theLink->next       = NULL;
listPtr = CpvAccess(ifCondArisesArray)[condNum].dataListPtr; /* put entry into list */
if(listPtr == NULL)
    CpvAccess(ifCondArisesArray)[condNum].dataListPtr = theLink;
else {
    lockStpPtr = listPtr->next;
    while(lockStpPtr != NULL) {
        listPtr = lockStpPtr;
        lockStpPtr = lockStpPtr->next;
        }
    listPtr->next  = theLink;
    }
} 


/***************************************************************************
Author: Wayne Fenton
Date:   1/20/90

Info:  The same as SendMsgIfCondArises (see above description) except that
       we would later like a boc function to be called.
***************************************************************************/
void CallBocIfCondArises(condNum, fn_ptr, bocNum)
int          condNum;
FUNCTION_PTR fn_ptr;
int          bocNum;
{
BocDataEntry *newEntry;
LinkRec      *theLink,*listPtr,*lockStpPtr;

if((newEntry = (BocDataEntry *) CmiAlloc(sizeof(BocDataEntry))) == NULL) {
    CkMemError(newEntry);
    return;
    }
if((theLink = (LinkRec *) CmiAlloc(sizeof(LinkRec))) == NULL) {
    CkMemError(theLink);
    return;
    }
newEntry->fn_ptr    = fn_ptr;               /* fill in the new data structure */
newEntry->bocNum    = bocNum;

theLink->theData    = (void *) newEntry;    /* set up the linked list info */
theLink->next       = NULL;
theLink->bocOrChare = ITSABOC;
listPtr = CpvAccess(ifCondArisesArray)[condNum].dataListPtr; /* put entry into list */
if(listPtr == NULL)
    CpvAccess(ifCondArisesArray)[condNum].dataListPtr = theLink;
else {
    lockStpPtr = listPtr->next;
    while(lockStpPtr != NULL) {
        listPtr = lockStpPtr;
        lockStpPtr = lockStpPtr->next;
        }
    listPtr->next  = theLink;
    }
} 


/***************************************************************************
Author: Wayne Fenton
Date:   1/20/90
***************************************************************************/
void RaiseCondition(condNum)
int condNum;
{
CpvAccess(ifCondArisesArray)[condNum].isCondRaised = 1;
}


/***************************************************************************
Author: Wayne Fenton
Date:   1/20/90

Info: This function sets up a data structure and enters it into a heap of time
      values so that after a pre-determined amount of time passes (ie the 
      amount specified in the deltaT parameter, a msg will be sent. At this
      point we just fill the data structure, add deltaT to the current time
      and put this value into the heap. Later, the routine TimerChecks will
      compare the then current time against those values in the heap. If the
      specified time has past, then the message in the data structure will
      be sent.
     
***************************************************************************/
void SendMsgAfter(deltaT, entry, msgToSend, size, pChareID)
unsigned int  deltaT;
int             entry;
void            *msgToSend;
int             size;
ChareIDType     *pChareID;
{
unsigned int tPrime, currT;
ChareDataEntry *newEntry;

if((newEntry = (ChareDataEntry *) CmiAlloc(sizeof(ChareDataEntry))) == NULL) {
    CkMemError(newEntry);
    }
else {
    currT  = CkTimer();                    /* get current time */
    tPrime = currT + deltaT;               /* add delta to detrmn what tme */
                                           /* to actually send the message */
    newEntry->entry     = entry;
    newEntry->msg       = msgToSend;
    newEntry->size      = size;
    newEntry->chareID   = *pChareID;        /* fill up our data structure   */
    TRACE(CmiPrintf("SendMsgAfter: currT[%d] Will sendmsg [%d]\n",currT,tPrime));
    InsertInHeap(tPrime, ITSACHARE,  newEntry); /* insert into tmr hp */
    }
} 



/***************************************************************************
Author: Wayne Fenton
Date:   1/20/90

Info:   This routine is again exactly the same as SendmsgAfter, except
        that when the time elapses and TimerChecks notices, then it will
        call the boc function, and not send a msg.

***************************************************************************/
void CallBocAfter(fn_ptr,bocNum,deltaT)
FUNCTION_PTR   fn_ptr;
int            bocNum; 
unsigned int deltaT;
{
unsigned int tPrime, currT;
BocDataEntry *newEntry;

if((newEntry = (BocDataEntry *) CmiAlloc(sizeof(BocDataEntry))) == NULL) {
    CkMemError(newEntry);
    }
else {
    currT            = CkTimer();
    tPrime           = currT + deltaT;
    newEntry->bocNum = bocNum;  
    newEntry->fn_ptr = fn_ptr;  
    TRACE(CmiPrintf("CallBocAfter nd[%d] tPrime[%d] boc[%d]\n",
             CmiMyPe(),tPrime,bocNum));
    InsertInHeap(tPrime, ITSABOC, newEntry);
    } 
} 


/***************************************************************************
Author: Wayne Fenton
Date:   1/20/90

Info:   This routine does some time-related checks, ie those tasks that are
        to be executed after a certain delay, are taken care of here. We
        go through the heap, looking at entries with the smallest time values
        first. If the mentioned time has already past, then we either send
        the corresponding msg or call the boc function, whichever is applicable.
        We then remove the element from the heap. We continue this procedure
        until either the smallest mentioned time hasn't arrived yet, or we
        exhaust the elements in the heap.

***************************************************************************/
void TimerChecks()
{
unsigned int currTime;
int            index;
ChareDataEntry *chareData;
BocDataEntry   *bocData;

currTime = CkTimer();

	/*TRACE(CmiPrintf("(TimerChecks nd[%d]), currT [%d] #in hp[%d]\n",
             CmiMyPe(),currTime,CpvAccess(numHeapEntries)));*/

while ((CpvAccess(numHeapEntries) > 0) && (LowestTime(&index) < currTime)) {
    if(CpvAccess(timerHeap)[index].bocOrChare == ITSACHARE) {
        TRACE(CmiPrintf("(TimerCheck nd[%d])Snding Msg ndx[%d] currT[%d] schedT[%d]\n",
                 CmiMyPe(),index, currTime, LowestTime(&index)));
        chareData = (ChareDataEntry *) CpvAccess(timerHeap)[index].theData;
        RemoveFromHeap(index); 
        SendMsg(chareData->entry, chareData->msg,      /* send the message */
                 &chareData->chareID);
	CmiFree(chareData);
        } 
    else if(CpvAccess(timerHeap)[index].bocOrChare == ITSABOC) {  /* call the boc fn */
        TRACE(CmiPrintf("(TimerCheck nd[%d])Cllng Boc ndx[%d] currT[%d] schedT[%d]\n",
                 CmiMyPe(),index, currTime, CpvAccess(timerHeap)[1].timeVal));
        bocData = (BocDataEntry *) CpvAccess(timerHeap)[index].theData;
        RemoveFromHeap(index); 
        (*(bocData->fn_ptr)) (bocData->bocNum);
	CmiFree(bocData);
        } 
    else {
        CmiPrintf("Internal consistency error, (TimerChecks), exiting...\n");
        CkExit();
        }
    } 
} 

/*****************************************************************************
Author: Wayne Fenton
Date:   1/20/90

Info:   Here we just initialize some of the data structures 
*****************************************************************************/
void CondSendInit()
{
    int i;

    CpvAccess(numRaiseCondArryElts) = NUMSYSCONDARISEELTS;

    CpvAccess(timerHeap) = (HeapIndexType *)
          CmiAlloc((MAXTIMERHEAPENTRIES + 1)*sizeof(HeapIndexType));

    CpvAccess(condChkArray) = (CondArrayEltType *)
          CmiAlloc(MAXCONDCHKARRAYELTS*sizeof(CondArrayEltType));

    CpvAccess(ifCondArisesArray) = (IfCondArisesArrayEltType *)
          CmiAlloc(MAXIFCONDARISESARRAYELTS*sizeof(IfCondArisesArrayEltType));



    for(i = 0; i < MAXIFCONDARISESARRAYELTS; i++) {
       CpvAccess(ifCondArisesArray)[i].isCondRaised = 0;
       CpvAccess(ifCondArisesArray)[i].dataListPtr  = NULL;
    }
}

/*****************************************************************************
Author: Wayne Fenton
Date:   1/20/90

Info:  This routine is called periodically to check various conditions and 
       do housekeeping. Several things are checked.
       1) we go through the condChkArray and either call the condition 
       function or make the boc access function call. If the cond function 
       call returns non-zero, then we send the corresponding msg. If either
       return non-zero, then that entry is then removed from the array.
       2) We go through the ifCondArisesArray checking to see if any of the
       conditions has been raised (by a call to RaiseCondition). If one has,
       we traverse its corresponding linked list and either send the msg or
       call the boc function. We then remove the elements from the list.
       If things are added again later, they will then be taken care of on
       the next call to this routine, since the condition has already been
       raised.

*****************************************************************************/
void PeriodicChecks()
{
int i,j;
ChareDataEntry *chareData;
BocDataEntry   *bocData;
LinkRec        *listPtr,*oldPtr;

for(i = 0; i < CpvAccess(numCondChkArryElts); i++) {
    if(CpvAccess(condChkArray)[i].bocOrChare == ITSACHARE) {
        chareData = (ChareDataEntry *) CpvAccess(condChkArray)[i].theData;
        if((*chareData->cond_fn)()){
           SendMsg(chareData->entry, chareData->msg, &chareData->chareID);
           CmiFree(CpvAccess(condChkArray)[i].theData);
           for(j = i; j < CpvAccess(numCondChkArryElts) - 1; j++) /* fill in hole in arry*/
               CpvAccess(condChkArray)[i] = CpvAccess(condChkArray)[i + 1];
           i--;                            /* keep checking where we left off */
           CpvAccess(numCondChkArryElts)--;           /* but we have one fewer elt       */
           } 
        } 
    else if(CpvAccess(condChkArray)[i].bocOrChare == ITSABOC) {
        bocData = (BocDataEntry *) CpvAccess(condChkArray)[i].theData;
        if((*(bocData->fn_ptr))(bocData->bocNum)) {
            CmiFree(CpvAccess(condChkArray)[i].theData);
            for(j = i; j < CpvAccess(numCondChkArryElts) - 1; j++)   /* fill in hole */
                CpvAccess(condChkArray)[i] = CpvAccess(condChkArray)[i + 1];
            i--;                       /* keep checking where we left off */
            CpvAccess(numCondChkArryElts)--;      /* but we have one fewer elt       */
            }
        } 
    else {
        CmiPrintf("Internal inconsistency (PeriodicChecks), exiting...\n");
        CkExit();
        } 
    } 

       /* now we check the ifCondArisesArray to see if any have arisen */
for(i = 0; i < CpvAccess(numRaiseCondArryElts); i++) {
    if(CpvAccess(ifCondArisesArray)[i].isCondRaised) {         /* if the condition is T */
        listPtr = CpvAccess(ifCondArisesArray)[i].dataListPtr;
        while(listPtr != NULL) {                    /* traverse the list */
            if(listPtr->bocOrChare == ITSACHARE) {  /* and take care of tasks */
                chareData = (ChareDataEntry *) listPtr->theData;
                SendMsg(chareData->entry, chareData->msg, /* send the msg */
                        &chareData->chareID);
                }
            else if(listPtr->bocOrChare == ITSABOC) {
                bocData = (BocDataEntry *) listPtr->theData;
                (*(bocData->fn_ptr)) (bocData->bocNum);
                }
            oldPtr = listPtr;
            listPtr = listPtr->next;
            CmiFree(oldPtr->theData);           /* free the task */
            CmiFree(oldPtr);                    /* free the link struct */
            }
        CpvAccess(ifCondArisesArray)[i].isCondRaised = 0; /* now reset condition flg */
        CpvAccess(ifCondArisesArray)[i].dataListPtr  = NULL;  /* as well as list head    */
        }
    }
} 

/***************************************************************************
Author: Wayne Fenton
Date:   1/20/90

info:   here we add an element to the heap.
***************************************************************************/
void InsertInHeap(theTime, bocOrChare, heapData)
unsigned int theTime;
int            bocOrChare;
void           *heapData;
{
int child, parent;

if(CpvAccess(numHeapEntries) > MAXTIMERHEAPENTRIES) {
    CmiPrintf("Heap overflow (InsertInHeap), exiting...\n");
    CkExit();
    }
else {
    CpvAccess(numHeapEntries)++;
    CpvAccess(timerHeap)[CpvAccess(numHeapEntries)].timeVal    = theTime;
    CpvAccess(timerHeap)[CpvAccess(numHeapEntries)].bocOrChare = bocOrChare;
    CpvAccess(timerHeap)[CpvAccess(numHeapEntries)].theData    = (void *) heapData;
    child  = CpvAccess(numHeapEntries);
    parent = child / 2;
    while(parent > 0) {
        if(CpvAccess(timerHeap)[child].timeVal < CpvAccess(timerHeap)[parent].timeVal)
            SwapHeapEntries(child,parent);
        child  = parent;
        parent = parent / 2;
        }
    }
} 


/***************************************************************************
Author: Wayne Fenton
Date:   1/20/90

Info:   Just removes an element from the heap corresponding to the index.
        it also frees up the memory associated with the heap element (ie,
        the space that was pointed to by the heap element). 
***************************************************************************/
void RemoveFromHeap(index)
int index;
{
int parent,child;

parent = index;
if(!CpvAccess(numHeapEntries) || (index != 1)) {
    CmiPrintf("Internal inconsistency (RemoveFromHeap), exiting ...\n");
    CkExit();
    } 
else {
    CpvAccess(timerHeap)[index].theData = NULL;
    SwapHeapEntries(index,CpvAccess(numHeapEntries)); /* put value at end of heap */
    CpvAccess(numHeapEntries)--;
    if(CpvAccess(numHeapEntries)) {             /* if any left, then bubble up values */
        child = 2 * parent;
        while(child <= CpvAccess(numHeapEntries)) {
            if(((child + 1) <= CpvAccess(numHeapEntries))  &&
               (CpvAccess(timerHeap)[child].timeVal > CpvAccess(timerHeap)[child + 1].timeVal))
                child++;              /* use the smaller of the two */
            if(CpvAccess(timerHeap)[parent].timeVal > CpvAccess(timerHeap)[child].timeVal) {
                SwapHeapEntries(parent,child);
                parent  = child;      /* go down the tree one more step */
                child  = 2 * child;
                }
            else
                break;                /* no more bubbling so break */
            }
        }
    } 
/*
CmiPrintf("Removed from Heap ndx[%d] #HpEntries[%d]\n",index,CpvAccess(numHeapEntries));
*/
} 


/***************************************************************************
Author: Wayne Fenton
Date:   1/20/90

Info:   Just swaps two entries in the heap.
***************************************************************************/
void SwapHeapEntries(index1, index2)
int index1,index2;
{
HeapIndexType temp;

temp              = CpvAccess(timerHeap)[index1];
CpvAccess(timerHeap)[index1] = CpvAccess(timerHeap)[index2];
CpvAccess(timerHeap)[index2] = temp;
} 


/***************************************************************************
Author: Wayne Fenton
Date:   1/20/90

Info:   Just return the lowest value in the heap, which should always be
        at location 0. It also sets the index to this value.
***************************************************************************/
unsigned int LowestTime(indexPtr)
int *indexPtr;
{
if(!CpvAccess(numHeapEntries)) {
    CmiPrintf("Internal inconsistency (LowestTime), exiting...\n");
    CkExit();
    }
else {
    *indexPtr = 1;
    return(CpvAccess(timerHeap)[1].timeVal);
    }
} 


/******************************************************************************
Author: Wayne Fenton
Date:   2/12/90

Info:  Returns a boolean indicating if there are any delayed messages in the
 	   heap (ie are there any outstanding SendMsgAfter or CallBocAfter calls) 
***************************************************************************/
int  NoDelayedMsgs()
{
int i,cat;

/*
CmiPrintf("entering NoDelayedMsgs(), [%d] in heap\n",CpvAccess(numHeapEntries));
*/
for(i = 1; i <= CpvAccess(numHeapEntries); i++) {
	if(CpvAccess(timerHeap)[i].bocOrChare == ITSABOC) {
        if(((BocDataEntry *)CpvAccess(timerHeap)[i].theData)->bocNum >= NumSysBoc)
            return 0;      /* if there is a user level boc func call */
        }
    else if(CpvAccess(timerHeap)[i].bocOrChare == ITSACHARE) {

	ENVELOPE *env;

	env = (ENVELOPE *) ENVELOPE_UPTR(((ChareDataEntry*)CpvAccess(timerHeap)[i].theData)->msg);
        cat = GetEnv_category(env);
        CmiPrintf("NoDelayedMsgs, category is[%d]\n",cat);
        if(cat == USERcat)
            return 0;        /* if there is a user level message */
        }
    else {
        CmiPrintf("Error, node[%d], (NoDelayedMsgs ndx[%d] boc[%d])\n",
                 CmiMyPe(),i,CpvAccess(timerHeap)[i].bocOrChare);
        CkExit();
        break;
        }
    }
return 1;
}
