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

/** Search Queue for messages associated with a specified entry method */ 
void CqsIncreasePriorityForEntryMethod(Queue q, const int entrymethod){
  int i;
  void **entries;
  int numMessages = q->length;
  
  CqsEnumerateQueue(q, &entries);
  
  for(i=0;i<numMessages;i++){
    envelope *env = (envelope*)entries[i];
    if(env != NULL){
      if(env->getMsgtype() == ForArrayEltMsg || env->getMsgtype() == ForChareMsg){
	const int ep = env->getsetArrayEp();
	if(ep==entrymethod){
	  // Remove the entry from the queue	  
	  CqsRemoveSpecific(q,env);
	  
	  int prio = -50000; 
	  CqsEnqueueGeneral(q, (void*)env, CQS_QUEUEING_IFIFO, 0, (unsigned int*)&prio);

	  char traceStr[64];
	  sprintf(traceStr, "Replacing %p in prioq with NULL", env);
	  traceUserSuppliedNote(traceStr);

	  break;
	}
      }  
    }
  }
  
  CmiFree(entries);
}


/** @} */
