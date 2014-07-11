/** 
    @file 
    Routines for modifying the Charm++ prioritized message queue
    @ingroup CharmScheduler
    
    @addtogroup CharmScheduler
    @{
 */


#include "charm++.h"
#include "queueing.h" // For access to scheduler data structures
#include "conv-trace.h"


// Predeclarations:
int CqsFindRemoveSpecificPrioq(_prioq q, void *&msgPtr, const int *entryMethod, const int numEntryMethods );
int CqsFindRemoveSpecificDeq(_deq q, void *&msgPtr, const int *entryMethod, const int numEntryMethods );


/** Search Queue for messages associated with a specified entry method */ 
void CqsIncreasePriorityForEntryMethod(Queue q, const int entrymethod){
    void *removedMsgPtr;
    int numRemoved;
    
    int entryMethods[1];
    entryMethods[0] = entrymethod;

    numRemoved = CqsFindRemoveSpecificPrioq(&(q->negprioq), removedMsgPtr, entryMethods, 1 );
    if(numRemoved == 0)
	numRemoved = CqsFindRemoveSpecificDeq(&(q->zeroprio), removedMsgPtr, entryMethods, 1 );
    if(numRemoved == 0)
	numRemoved = CqsFindRemoveSpecificPrioq(&(q->posprioq), removedMsgPtr, entryMethods, 1 );
    
    if(numRemoved > 0){
	CkAssert(numRemoved==1); // We need to reenqueue all removed messages, but we currently only handle one
	int prio = -1000000; 
	CqsEnqueueGeneral(q, removedMsgPtr, CQS_QUEUEING_IFIFO, 0, (unsigned int*)&prio);

#if CMK_TRACE_ENABLED
	char traceStr[64];
	sprintf(traceStr, "Replacing %p in message queue with NULL", removedMsgPtr);
	traceUserSuppliedNote(traceStr);
#endif
    }
}
 
#ifdef ADAPT_SCHED_MEM
/** Search Queue for messages associated with memory-critical entry methods */ 
void CqsIncreasePriorityForMemCriticalEntries(Queue q){
    void *removedMsgPtr;
    int numRemoved;

    numRemoved = CqsFindRemoveSpecificPrioq(&(q->negprioq), removedMsgPtr, memCriticalEntries, numMemCriticalEntries);
    if(numRemoved == 0)
	numRemoved = CqsFindRemoveSpecificDeq(&(q->zeroprio), removedMsgPtr, memCriticalEntries, numMemCriticalEntries);
    if(numRemoved == 0)
	numRemoved = CqsFindRemoveSpecificPrioq(&(q->posprioq), removedMsgPtr, memCriticalEntries, numMemCriticalEntries);
    
    if(numRemoved > 0){
	CkAssert(numRemoved==1); // We need to reenqueue all removed messages, but we currently only handle one
	int prio = -1000000; 
	CqsEnqueueGeneral(q, removedMsgPtr, CQS_QUEUEING_IFIFO, 0, (unsigned int*)&prio);

#if CMK_TRACE_ENABLED
	char traceStr[64];
	sprintf(traceStr, "Replacing %p in message queue with NULL", removedMsgPtr);
	traceUserSuppliedNote(traceStr);
#endif
    }
}
#endif

static bool checkAndRemoveMatching(void *&msgPtr, const int *entryMethod, const int numEntryMethods, envelope *env, void** &p) {
	if(env != NULL && (env->getMsgtype() == ForArrayEltMsg ||
                       env->getMsgtype() == ForChareMsg)
      ){
	    const int ep = env->getsetArrayEp();
	    bool foundMatch = false;
	    // Search for ep by linearly searching through entryMethod
	    for(int i=0;i<numEntryMethods;++i){
		if(ep==entryMethod[i]){
		    foundMatch=true;
		    break;
		}
	    }
	    if(foundMatch){
		// Remove the entry from the queue
		*p = NULL;
		msgPtr = env;
		return true;
	    }
	}
}

/** Find and remove the first 1 occurences of messages that matches a specified entry method index.
    The size of the deq will not change, it will just contain an entry for a NULL pointer.

    @return number of entries that were replaced with NULL

    @param [in] q A circular double ended queue
    @param [out] msgPtr returns the message that was removed from the prioq
    @param [in] entryMethod An array of entry method ids that should be considered for removal
    @param [in] numEntryMethods The number of the values in the entryMethod array.
*/
int CqsFindRemoveSpecificDeq(_deq q, void *&msgPtr, const int *entryMethod, const int numEntryMethods ){
    void **iter = q->head; ///< An iterator used to iterate through the circular queue

    while(iter != q->tail){
	// *iter is contains a pointer to a message
	envelope *env = (envelope*)*iter;
	if (checkAndRemoveMatching(msgPtr, entryMethod, numEntryMethods, env, iter))
	  return 1;

	// Advance head to the next item in the circular queue
	iter++;
	if(iter == q->end)
	    iter = q->bgn;
    }
    return 0;
}

 

/** Find and remove the first 1 occurences of messages that matches a specified entry method index.
    The size of the prioq will not change, it will just contain an entry for a NULL pointer.

    @return number of entries that were replaced with NULL

    @param [in] q A priority queue
    @param [out] msgPtr returns the message that was removed from the prioq
    @param [in] entryMethod An array of entry method ids that should be considered for removal
    @param [in] numEntryMethods The number of the values in the entryMethod array.
*/
int CqsFindRemoveSpecificPrioq(_prioq q, void *&msgPtr, const int *entryMethod, const int numEntryMethods ){

    // A priority queue is a heap of circular queues
    for(int i = 1; i < q->heapnext; i++){
	// For each of the circular queues:
        _prioqelt pe = (q->heap)[i];
	void **head; ///< An iterator used to iterate through a circular queue
	void **tail; ///< The end of the circular queue
	head = pe->data.head;
        tail = pe->data.tail;
        while(head != tail){
	    // *head contains a pointer to a message
	    envelope *env = (envelope*)*head;
	    if (checkAndRemoveMatching(msgPtr, entryMethod, numEntryMethods, env, head))
	      return 1;

	    // Advance head to the next item in the circular queue
	    head++;
            if(head == (pe->data).end)
                head = (pe->data).bgn;
        }
    }
    return 0;
}





/** @} */
