/**
   @addtogroup CharmComlib
   @{
   @file
   Implementation of the functions in ComlibManager.h and handler for message
   transportation.
*/

#include "ComlibManager.h"
#include "comlib.h"
#include "ck.h"
#include "envelope.h"


// We only want to print debug information for a single strategy. Otherwise we'll get incredibly confused
#undef ComlibManagerPrintf
//#define ComlibManagerPrintf  if(instid==1)ComlibPrintf
#define ComlibManagerPrintf  ComlibPrintf

#define getmax(a,b) ((a)>(b)?(a):(b))

CkpvExtern(int, RecvdummyHandle);
 
CkpvDeclare(CkGroupID, cmgrID);

/***************************************************************************
 * Handlers section:
 *
 * all the handlers used by the ComlibManager to coordinate the work, and
 * propagate the messages from one processor to another
 ***************************************************************************/

//Handler to receive array messages
CkpvDeclare(int, RecvmsgHandle);

void recv_array_msg(void *msg){

  //	ComlibPrintf("%d:In recv_msg\n", CkMyPe());

	if(msg == NULL)
		return;

	register envelope* env = (envelope *)msg;
	env->setUsed(0);
	env->getsetArrayHops()=1; 
	CkUnpackMessage(&env);

	int srcPe = env->getSrcPe();
	int sid = ((CmiMsgHeaderExt *) env)->stratid;

	//	ComlibPrintf("%d: Recording receive %d, %d, %d\n", CkMyPe(), sid, env->getTotalsize(), srcPe);

	RECORD_RECV_STATS(sid, env->getTotalsize(), srcPe);

	CkArray *a=(CkArray *)_localBranch(env->getsetArrayMgr());
	
	a->deliver((CkArrayMessage *)EnvToUsr(env), CkDeliver_queue);

	//	ComlibPrintf("%d:Out of recv_msg\n", CkMyPe());
	return;
}



/**
   A debugging routine that will periodically print out the status of the message queues. 
   The Definition is at bottom of this file.
 */
static void periodicDebugPrintStatus(void* ptr, double currWallTime);





/***************************************************************************
 * Initialization section:
 *
 * Routines used by Comlib to initialize itself and all the strategies on all
 * the processors. Used only at the beginning of the program.
 ***************************************************************************/

// // initialized at startup before the main chare main methods are called:
// void initCharmComlibManager(){
//   CkPrintf("[%d] initCharmComlibManager()\n", CkMyPe());
//   fflush(stdout);
// }



ComlibManager::ComlibManager(){
	init();
}

void ComlibManager::init(){

  //   CcdCallFnAfterOnPE((CcdVoidFn)periodicDebugPrintStatus, (void*)this, 4000, CkMyPe());


  if(CkNumPes() == 1 ){
    ComlibPrintf("Doing nothing in ComlibManager::init() because we are running on 1 pe.\n");
  } else {

	if (CkMyRank() == 0) {
		PUPable_reg(CharmMessageHolder);
	}

	numStatsReceived = 0;
	curComlibController = 0;
	clibIteration = 0;

	CkpvInitialize(comRectHashType *, com_rect_ptr); 
	CkpvAccess(com_rect_ptr)= new comRectHashType;

	CkpvInitialize(int, RecvmsgHandle);
	CkpvAccess(RecvmsgHandle) =CkRegisterHandler((CmiHandler)recv_array_msg);

	bcast_pelist = new int [CkNumPes()];
	_MEMCHECK(bcast_pelist);
	for(int brcount = 0; brcount < CkNumPes(); brcount++)
		bcast_pelist[brcount] = brcount;

	section_send_event = traceRegisterUserEvent("ArraySectionMulticast");

	CkpvInitialize(CkGroupID, cmgrID);
	CkpvAccess(cmgrID) = thisgroup;

	dummyArrayIndex.nInts = 0;

	CkAssert(CkpvInitialized(conv_com_object));
	converseManager = &CkpvAccess(conv_com_object);

	setupComplete = 0;

	CkpvInitialize(int, migrationDoneHandlerID);
	CkpvAccess(migrationDoneHandlerID) = 
		CkRegisterHandler((CmiHandler) ComlibNotifyMigrationDoneHandler);

	CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
	cgproxy[curComlibController].barrier();
  }
}

//First barrier makes sure that the communication library group 
//has been created on all processors
void ComlibManager::barrier(){
	static int bcount = 0;
	ComlibPrintf("barrier %d\n", bcount);
	if(CkMyPe() == 0) {
		bcount ++;
		if(bcount == CkNumPes()){
			bcount = 0;
			CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
			cgproxy.resumeFromSetupBarrier();
		}
	}
}



/**
   Due to the possibility of race conditions in the initialization of charm, this
   barrier prevents comlib from being activated before all group branches are created.
   This function completes the initialization of the charm layer of comlib.

   In this function we also call ComlibDoneCreating for the user (basically
   triggering the broadcast of the strategies created in Main::Main. Here the
   Main::Main has for sure executed, otherwise we will not have received
   confirmation by all other processors.
   
   
 */
void ComlibManager::resumeFromSetupBarrier(){
	ComlibPrintf("[%d] resumeFromSetupBarrier Charm group ComlibManager setup finished\n", CkMyPe());

	setupComplete = 1;
	ComlibDoneCreating();
	ComlibPrintf("[%d] resumeFromSetupBarrier calling ComlibDoneCreating to tell converse layer strategies to set themselves up\n", CkMyPe());
	sendBufferedMessagesAllStrategies();

}

/***************************************************************************
 Determine whether the delegated messages should be buffered until the 
 strategy has recovered from any error conditions and startup. Once the 
 buffers can be flushed, ComlibManager::sendBufferedMessages() will be called
***************************************************************************/
bool ComlibManager::shouldBufferMessagesNow(int instid){
  StrategyTableEntry *myEntry = converseManager->getStrategyTable(instid);
  return (!setupComplete) || myEntry->getErrorMode() == ERROR_MODE || myEntry->getErrorMode() == CONFIRM_MODE || myEntry->bufferOutgoing;
}


/***************************************************************************
 Calls ComlibManager::sendBufferedMessages for each strategy.
***************************************************************************/
void ComlibManager::sendBufferedMessagesAllStrategies(){
  int nstrats = converseManager->getNumStrats();
  for(int i=0;i<nstrats;i++){
    sendBufferedMessages(i);
  }
}


/***************************************************************************
   Send all the buffered messages once startup has completed, and we have 
   recovered from all errors.
***************************************************************************/
void ComlibManager::sendBufferedMessages(int instid, int step){
  StrategyTableEntry *myEntry = converseManager->getStrategyTable(instid);
  
  if(shouldBufferMessagesNow(instid)){
    ComlibPrintf("[%d] sendBufferedMessages is not flushing buffered messages for strategy %d because shouldBufferMessagesNow()==true step %d\n", CkMyPe(), instid, step);  
  } else if(delayMessageSendBuffer[instid].size() == 0){
    ComlibPrintf("[%d] sendBufferedMessages: no bufferedmessages to send for strategy %d step %d\n", CkMyPe(), instid, step);  
  } else{
    ComlibPrintf("[%d] sendBufferedMessages Sending %d buffered messages for instid=%d step %d\n", CkMyPe(), delayMessageSendBuffer[instid].size(), instid, step);
 
    for (std::set<CharmMessageHolder*>::iterator iter = delayMessageSendBuffer[instid].begin(); iter != delayMessageSendBuffer[instid].end(); ++iter) {
      CharmMessageHolder* cmsg = *iter;
	  
      switch(cmsg->type){
	    
      case CMH_ARRAYSEND:
	CkpvAccess(conv_com_object).insertMessage(cmsg, instid);
	CkpvAccess(conv_com_object).doneInserting(instid);
	break;
	    
      case CMH_GROUPSEND:
	CkAbort("CMH_GROUPSEND unimplemented");
	break;
	    
      case CMH_ARRAYBROADCAST:
      case CMH_ARRAYSECTIONSEND: 
      case CMH_GROUPBROADCAST:
	// Multicast/broadcast to an array or a section:
	cmsg->sec_id = cmsg->copy_of_sec_id;
	CkpvAccess(conv_com_object).insertMessage(cmsg, instid);
	CkpvAccess(conv_com_object).doneInserting(instid);
	break;
	    
      default:
	CkAbort("Unknown cmsg->type was found in buffer of delayed messages\n");
      }
	  
    }

    delayMessageSendBuffer[instid].clear();
  }

}



/***************************************************************************
 * Routine for bracketed strategies, for detecting count errors after 
 * objects migrate.
 * 
 * This function must only be called after all messages from the previous 
 * iteration have been delivered. This is the application's responsibility to 
 * ensure. The iteration values provided must increase monotonically, 
 * and must be greater than STARTUP_ITERATION.
 *
 ***************************************************************************/
/// Called when the array/group element starts sending messages
void ComlibManager::beginIteration(int instid, int iteration){
	StrategyTableEntry *myEntry = converseManager->getStrategyTable(instid);
			    
	ComlibManagerPrintf("[%d] beginIteration iter=%d lastKnownIteration=%d  %s %s %s\n", CkMyPe(), iteration, myEntry->lastKnownIteration, myEntry->errorModeString(),  myEntry->errorModeServerString(),  myEntry->discoveryModeString() );

	CkAssert(myEntry->getErrorMode() == NORMAL_MODE || myEntry->getErrorMode() == ERROR_MODE);
	if(CkMyPe()==0)
	  CkAssert(myEntry->getErrorModeServer() == NORMAL_MODE_SERVER ||  myEntry->getErrorModeServer() == ERROR_MODE_SERVER);

	if(iteration > myEntry->lastKnownIteration){
	      ComlibManagerPrintf("[%d] beginIteration Starting Next Iteration ( # %d )\n", CkMyPe(), iteration);

	      myEntry->lastKnownIteration = iteration;
	      myEntry->nBeginItr = 1; // we are the first time to be called this iteration
	      myEntry->nEndItr = 0;
	      myEntry->nProcSync = 0;
	      myEntry->totalEndCounted = 0;
	      myEntry->nEndSaved = 0;
	      	      
	} else if(iteration == myEntry->lastKnownIteration){
		ComlibManagerPrintf("[%d] beginIteration continuing iteration # %d\n", CkMyPe(), iteration);
		myEntry->nBeginItr++;
	} else {
	  CkPrintf("[%d] ERROR: ComlibManager::beginIteration iteration=%d < myEntry->lastKnownIteration=%d", iteration, myEntry->lastKnownIteration);
	  CkAbort("[%d] ERROR: ComlibManager::beginIteration iteration < myEntry->lastKnownIteration");
	}
	
	
	// We need to check for error conditions here as well as EndIteration.
	// This will ensure that if we are the processor that detects this error, 
	// we won't deliver at least this message until the strategy is fixed
	if (myEntry->nBeginItr > myEntry->numElements) {
	  ComlibManagerPrintf("[%d] beginIteration BUFFERING OUTGOING because nBeginItr=%d > numElements=%d\n",CkMyPe(), myEntry->nBeginItr, myEntry->numElements);
	  myEntry->bufferOutgoing = 1;
	}
	
}


/** Called by user program when each element has finished sending its messages.

    If no errors are detected, ConvComlibManager::doneInserting() is called on the 
    underlying converse strategy. If an error is detected, then ConvComlibManager::doneInserting
    is NOT called. This likely causes the the underlying converse strategy to buffer the 
    messages until we recover from the error mode, although we have buffered at least some of
    the messages, so the user program cannot run ahead and start the next iteration.

*/

void ComlibManager::endIteration(int instid, int step){ 
	StrategyTableEntry *myEntry = converseManager->getStrategyTable(instid);
	
	// migration is not allowed between begin and end of iteration, 
	// so each element must call end iteration on the same processor 
	// as it called begin iteration
	CkAssert(myEntry->nEndItr <= myEntry->nBeginItr);
	CkAssert(step == myEntry->lastKnownIteration);

	CkAssert(myEntry->getErrorMode() == NORMAL_MODE || myEntry->getErrorMode() == ERROR_MODE);
	if(CkMyPe()==0)
	  CkAssert(myEntry->getErrorModeServer() == NORMAL_MODE_SERVER ||  myEntry->getErrorModeServer() == ERROR_MODE_SERVER);

	myEntry->nEndItr++;
	
	ComlibManagerPrintf("[%d] endIteration called\n",CkMyPe());
	
	
	if (myEntry->bufferOutgoing) {
	  // If migration was detected in ComlibManager::beginIteration and hence messages have been buffered:
	  CkAssert(delayMessageSendBuffer[instid].size() > 0);
	  CProxy_ComlibManager myProxy(thisgroup);
	  myProxy[CkMyPe()].bracketedStartErrorRecoveryProcess(instid, step);
	} 
	else if(myEntry->nEndItr == myEntry->numElements) {
	  // If all the objects have called beginIteration and endIteration and no errors were detected
	  CkAssert(converseManager->isReady(instid));
	  converseManager->doneInserting(instid);
	}
	
}


/** Start recovery of errors. 
    This entry method calls itself repeatedly if the underlying 
    converse strategy is not yet ready.

    If the PE is already in error mode, then only the difference 
    in the counts will be contributed.
*/
void ComlibManager::bracketedStartErrorRecoveryProcess(int instid, int step){  
  CProxy_ComlibManager myProxy(thisgroup);
  StrategyTableEntry *myEntry = converseManager->getStrategyTable(instid);     
  CkAssert(step >= myEntry->lastKnownIteration);


  if(converseManager->isReady(instid)){
    ComlibManagerPrintf("[%d] bracketedStartErrorRecoveryProcess(instid=%d step=%d) %s %s %s\n", CkMyPe(), instid, step, myEntry->errorModeString(),  myEntry->errorModeServerString(),  myEntry->discoveryModeString() );

    CkAssert(myEntry->strategy != NULL);
    CkAssert(myEntry->getErrorMode() == NORMAL_MODE || myEntry->getErrorMode() == ERROR_MODE);
	  
    if (!myEntry->strategy->isBracketed()) {
      CkPrintf("[%d] endIteration called unecessarily for a non-bracketed strategy\n", CkMyPe());
      return;
    }
       
    if (myEntry->getErrorMode() == NORMAL_MODE) {
      ComlibManagerPrintf("[%d] bracketedStartErrorRecoveryProcess()\n", CkMyPe());
      myEntry->nEndSaved = myEntry->nEndItr;
      myProxy[0].bracketedReceiveCount(instid, CkMyPe(), myEntry->nEndSaved, 1, step);
      myEntry->setErrorMode(ERROR_MODE);
      bracketedStartDiscovery(instid);
    } else {
      // Send the updated count
      int update = myEntry->nEndItr - myEntry->nEndSaved;
      if (update > 0) {
	//	ComlibManagerPrintf("bracketedStartErrorRecoveryProcess sending update to bracketedReceiveCount\n");
	CProxy_ComlibManager myProxy(thisgroup);
	myProxy[0].bracketedReceiveCount(instid, CkMyPe(), update, 0, step);
	myEntry->nEndSaved = myEntry->nEndItr;
      }
      
    }
  } else {
    ComlibManagerPrintf("[%d] bracketedStartErrorRecoveryProcess() REENQUEUE\n", CkMyPe() );
    // Re-enqueue myself because we can't start error recovery process until converse strategy is ready
    myProxy[CkMyPe()].bracketedStartErrorRecoveryProcess(instid, step);
  }
}


/// Invoked on all processors to inform that an error has been detected.
void ComlibManager::bracketedErrorDetected(int instid, int step) {
	StrategyTableEntry *myEntry = converseManager->getStrategyTable(instid);
	
	bracketedCatchUpPE(instid,step);
	CkAssert(step == myEntry->lastKnownIteration);
	CkAssert(myEntry->getErrorMode() == NORMAL_MODE || myEntry->getErrorMode() == ERROR_MODE);

	ComlibManagerPrintf("[%d] bracketedErrorDetected()\n", CkMyPe());

	if (myEntry->getErrorMode() == NORMAL_MODE) {
	  // save the value we are sending to bracketedReceiveCount
	  myEntry->nEndSaved = myEntry->nEndItr; // save the value we are sending to bracketedReceiveCount
	  CProxy_ComlibManager myProxy(thisgroup);
	  myProxy[0].bracketedReceiveCount(instid, CkMyPe(), myEntry->nEndSaved, 1, step);
	  bracketedStartDiscovery(instid);
	  myEntry->setErrorMode(ERROR_MODE);

	} else { // ERROR_MODE
	  // If we have an update count, send it
	  int update = myEntry->nEndItr - myEntry->nEndSaved;
	  ComlibManagerPrintf("bracketedErrorDetected update=%d\n", update);
	  if (update > 0) {
	    ComlibManagerPrintf("bracketedErrorDetected sending update to bracketedReceiveCount\n");
	    CProxy_ComlibManager myProxy(thisgroup);
	    myProxy[0].bracketedReceiveCount(instid, CkMyPe(), update, 0, step);
	    myEntry->nEndSaved = myEntry->nEndItr;
	  }
	}
	
}

/// Invoked on all processors. After processor 0 has a count match, it sends out
/// a broadcast on this entry method to get a confirmation from all others.
void ComlibManager::bracketedConfirmCount(int instid, int step) {
        StrategyTableEntry *myEntry = converseManager->getStrategyTable(instid);
	CkAssert(myEntry->getErrorMode() == ERROR_MODE);
	myEntry->setErrorMode(CONFIRM_MODE);
	CProxy_ComlibManager myProxy(thisgroup);
	ComlibManagerPrintf("[%d] bracketedConfirmCount\n", CkMyPe());
	myProxy[0].bracketedCountConfirmed(instid, myEntry->nEndSaved, step);
}

/// Invoked on processor 0 upon a request to confirm the count. If the count is
/// correct a "NewPeList" is sent out, otherwise the error mode is returned with
/// "ErrorDetected"
void ComlibManager::bracketedCountConfirmed(int instid, int count, int step) {
	StrategyTableEntry *myEntry = converseManager->getStrategyTable(instid);
	CkAssert(CkMyPe() == 0);
	// Advance PE0 to current step if we had no local objects
	bracketedCatchUpPE(instid, step);
	CkAssert(myEntry->getErrorModeServer() == CONFIRM_MODE_SERVER);
	CkAssert(step == myEntry->lastKnownIteration);

	myEntry->total += count;
	
	ComlibManagerPrintf("[%d] bracketedCountConfirmed\n", CkMyPe());
	
	if (++myEntry->peConfirmCounter == CkNumPes()) {
		myEntry->peConfirmCounter = 0;

		CkAssert(myEntry->total == myEntry->totalEndCounted);
		  
		CProxy_ComlibManager(thisgroup).bracketedReceiveNewCount(instid, step);
		myEntry->setErrorModeServer(ERROR_FIXED_MODE_SERVER);
		  
		myEntry->total = 0;
	}

}



/** Update the state for a PE that was likely lacking any local objects.
    If no local objects exist, noone will call begin iteration, and hence,
    the lastKnownIteration value will be old. We need to advance to the 
    current iteration before we can proceed.
*/
void ComlibManager::bracketedCatchUpPE(int instid, int step){
  StrategyTableEntry *myEntry = converseManager->getStrategyTable(instid);
  CkAssert(step >= myEntry->lastKnownIteration);
  if(step > myEntry->lastKnownIteration){
    
    myEntry->total = 0;
    myEntry->lastKnownIteration = step;
    myEntry->nBeginItr = 0;
    myEntry->nEndItr = 0;
    myEntry->nProcSync = 0;
    myEntry->totalEndCounted = 0;
    myEntry->nEndSaved = 0;
    myEntry->setErrorMode(NORMAL_MODE);
    myEntry->setDiscoveryMode(NORMAL_DISCOVERY_MODE);

    if(CkMyPe()==0){
      CkAssert(myEntry->getErrorModeServer() == NORMAL_MODE_SERVER || myEntry->getErrorModeServer() == ERROR_FIXED_MODE_SERVER);
      myEntry->setErrorModeServer(NORMAL_MODE_SERVER);
    }    
    
  }
}

/// Invoked on processor 0 when a processor sends a count of how many elements
/// have already deposited. This number is incremental, and it refers to the
/// previous one sent by that processor. Processor 0 stores temporarily all the
/// numbers it receives. When a match happens, processor 0 switches to "confirm"
/// mode and send out a request for confirmation to all other processors.
/// The final parameter, "step", is used so that if  no objects exist on a processor,
/// then the counts sent by the processor will be tagged with an old timestep, and can 
/// safely be ignored. If no objects are on a PE, then beginIteration will never
/// be called there.
void ComlibManager::bracketedReceiveCount(int instid, int pe, int count, int isFirst, int step) {
	StrategyTableEntry *myEntry = converseManager->getStrategyTable(instid);
	ComlibPrintf("[%d] bracketedReceiveCount begins step=%d lastKnownIteration=%d totalEndCounted=%d count=%d\n", CkMyPe(), step, myEntry->lastKnownIteration, myEntry->totalEndCounted, count);
	CkAssert(step >= myEntry->lastKnownIteration);

	CkAssert(CkMyPe() == 0);
	CkAssert(myEntry->getErrorModeServer() == NORMAL_MODE_SERVER || myEntry->getErrorModeServer() == ERROR_MODE_SERVER );
	
	
	// Advance PE0 to current step if we had no local objects
	bracketedCatchUpPE(instid, step);

	// Encountering a message from an old step
	CkAssert(step == myEntry->lastKnownIteration);

	myEntry->totalEndCounted += count;


	
	myEntry->nProcSync += isFirst; // isFirst is 1 the first time a processor send a message,
	CkAssert(myEntry->nProcSync <= CkNumPes());
	
	if (myEntry->getErrorModeServer() == NORMAL_MODE_SERVER) { 
	        // first time this is called
	        CkAssert(myEntry->nProcSync == 1);
	        CProxy_ComlibManager(thisgroup).bracketedErrorDetected(instid, step);
		myEntry->setErrorModeServer(ERROR_MODE_SERVER);
		ComlibManagerPrintf("[%d] bracketedReceiveCount first time\n", CkMyPe());
	}



	CharmStrategy* s = dynamic_cast<CharmStrategy*>(myEntry->strategy);
	ComlibArrayInfo ainfo = s->ainfo;
	int totalsrc =  ainfo.getTotalSrc() ;
	  
	if(myEntry->nProcSync == CkNumPes() && myEntry->totalEndCounted == totalsrc) {
	  // ok, we received notifications from all PEs and all objects have called endIteration
	  myEntry->setErrorModeServer(CONFIRM_MODE_SERVER); 
	  ComlibManagerPrintf("[%d] bracketedReceiveCount errorModeServer is now CONFIRM_MODE calling bracketedConfirmCount totalsrc=%d\n", CkMyPe(), (int)totalsrc);
	  CProxy_ComlibManager(thisgroup).bracketedConfirmCount(instid, step);
			
	}
		
}


/// Invoked on all processors. After the count has been checked and it matches
/// the number of emements involved in the bracketed operation, processor 0
/// sends a broadcast to acknowledge the success. 
/// The strategy is disabled until a new Pe list is received.
void ComlibManager::bracketedReceiveNewCount(int instid, int step) {
	StrategyTableEntry *myEntry = converseManager->getStrategyTable(instid);
	CkAssert(myEntry->getErrorMode() == CONFIRM_MODE);

	myEntry->setErrorMode(ERROR_FIXED_MODE);
	
	myEntry->nEndItr -= myEntry->nEndSaved;
	myEntry->nBeginItr -= myEntry->nEndSaved;

	myEntry->nEndSaved = 0;

	bracketedFinalBarrier(instid, step);
}




/// Invoked on all processors. When all processors have discovered all elements
/// involved in the operation, processor 0 uses this method to broadcast the
/// entire processor list to all. Currently the discovery process HAS to go
/// together with the counting process, otherwise the strategy is updated at an
/// undetermined state. In future it may be useful to implement the discovery
/// process alone.
void ComlibManager::bracketedReceiveNewPeList(int instid, int step, int *count) {
	StrategyTableEntry *myEntry = converseManager->getStrategyTable(instid);
	CkAssert(myEntry->getDiscoveryMode() == STARTED_DISCOVERY_MODE);
	myEntry->setDiscoveryMode(FINISHED_DISCOVERY_MODE);

	myEntry->strategy->bracketedUpdatePeKnowledge(count);
	
	ComlibManagerPrintf("[%d] bracketedReceiveNewPeList Updating numElements\n", CkMyPe());
	ComlibArrayInfo *myInfo = &dynamic_cast<CharmStrategy*>(myEntry->strategy)->ainfo;
	CkAssert((unsigned long)myInfo > 0x1000);
	myEntry->numElements = myInfo->getLocalSrc();
	
	ComlibManagerPrintf("[%d] delayMessageSendBuffer[%d].size()=%d\n",CkMyPe(), instid, delayMessageSendBuffer[instid].size() );
	ComlibManagerPrintf("[%d] delayMessageSendBuffer[%d].size()=%d\n", CkMyPe(), instid, delayMessageSendBuffer[instid].size());
		
	bracketedFinalBarrier(instid, step);
}


/** Start a barrier phase where all processes will enter it once both 
    the counting and discovery processes complete.

    ComlibManager::bracketedReceiveNewPeList calls this method. 
    ComlibManager::bracketedReceiveNewPeList is called as an array broadcast to thisProxy, 
    so every PE will call this method.
 */
void ComlibManager::bracketedFinalBarrier(int instid, int step) {
  StrategyTableEntry *myEntry = converseManager->getStrategyTable(instid);
  ComlibManagerPrintf("[%d] ComlibManager::bracketedFinalBarrier %s %s %s\n", CkMyPe(), myEntry->errorModeString(),  myEntry->errorModeServerString(),  myEntry->discoveryModeString() );

  
  if (myEntry->getDiscoveryMode() == FINISHED_DISCOVERY_MODE &&  myEntry->getErrorMode() == ERROR_FIXED_MODE) {       
    myEntry->setDiscoveryMode(NORMAL_DISCOVERY_MODE);
    myEntry->setErrorMode(NORMAL_MODE);

    // Update destination and source element lists for use in the next step
    ComlibArrayInfo *myInfo = &dynamic_cast<CharmStrategy*>(myEntry->strategy)->ainfo;
    myInfo->useNewSourceAndDestinations();	
    
    CProxy_ComlibManager myProxy(thisgroup);
    myProxy[0].bracketedReleaseCount(instid, step);
  }
}


/** 
    Once all PEs report here, we will allow them to release the buffered messages from this iteration
 */
void ComlibManager::bracketedReleaseCount(int instid, int step) {

    StrategyTableEntry *myEntry = converseManager->getStrategyTable(instid);
    ComlibPrintf("[%d] ComlibManager::bracketedReleaseCount myEntry->numBufferReleaseReady was %d\n", CkMyPe(), myEntry->numBufferReleaseReady);

    CkAssert(CkMyPe() == 0);
    CkAssert(myEntry->getErrorModeServer() == ERROR_FIXED_MODE_SERVER);
    
    myEntry->numBufferReleaseReady++;
    if(myEntry->numBufferReleaseReady == CkNumPes()) {
      myEntry->setErrorModeServer(NORMAL_MODE_SERVER);
      CProxy_ComlibManager(thisgroup).bracketedReleaseBufferedMessages(instid, step);
      myEntry->numBufferReleaseReady = 0;
    }
}

/** 
    Release any buffered messages.
 */
void ComlibManager::bracketedReleaseBufferedMessages(int instid, int step) {
  ComlibManagerPrintf("[%d] ComlibManager::bracketedReleaseBufferedMessages step=%d\n", CkMyPe(), step);
  StrategyTableEntry *myEntry = converseManager->getStrategyTable(instid);

  CkAssert(myEntry->getErrorModeServer() == NORMAL_MODE_SERVER);
  CkAssert(myEntry->getErrorMode() == NORMAL_MODE);
  CkAssert(myEntry->getDiscoveryMode() == NORMAL_DISCOVERY_MODE);
  
  myEntry->bufferOutgoing = 0;
  sendBufferedMessages(instid, step);
  
  converseManager->doneInserting(instid);
}






/** Called when an error is discovered. Such errors occur when it is observed
    that an array element has migrated to a new PE. Specifically if a strategy
    on some PE determines that it has received messages for more array elements
    than the strategy knew were local, then some array element must have 
    migrated to the PE. The strategy instance detecting the error will call this
    method to initiate a global update operation that determines the locations
    of all array elements.

    This is called on each PE by the bracketedErrorDetected() method. 

    Each array element previously located on this PE is examined to determine if
    it is still here. If the array element has migrated away, then the
    bracketedDiscover() method is called on the new PE.

*/
void ComlibManager::bracketedStartDiscovery(int instid) {
	StrategyTableEntry *myEntry = converseManager->getStrategyTable(instid);
	CkAssert(myEntry->getDiscoveryMode() == NORMAL_DISCOVERY_MODE);
	myEntry->setDiscoveryMode(STARTED_DISCOVERY_MODE);
	ComlibArrayInfo *myInfo = &dynamic_cast<CharmStrategy*>(myEntry->strategy)->ainfo;
	const CProxy_ComlibManager myProxy(thisgroup);

	ComlibManagerPrintf("[%d] bracketedStartDiscovery\n", CkMyPe());

	int countSrc = 0;
	int countDest = 0;

	if (myInfo->isSourceArray()) {

	  const CkVec<CkArrayIndexMax> & srcElements = myInfo->getSourceElements();
	  const int nelem = srcElements.size();
	  const CkArrayID aid = myInfo->getSourceArrayID(); 
	  const CkArray *a = (CkArray*)_localBranch(aid);

	  for (int i=0; i<nelem; ++i) {
	    int pe = a->lastKnown(srcElements[i]);
	    if (pe == CkMyPe()) {
	      countSrc++;
	      myInfo->addNewLocalSource(srcElements[i]);
	    }
	    else {
	      myProxy[pe].bracketedDiscover(instid, aid, srcElements[i], true);
	    }
	  }

	}

	if (myInfo->isDestinationArray()) {
// 	  CkAssert(myInfo->newDestinationListSize() == 0);

	  const CkVec<CkArrayIndexMax> & destElements = myInfo->getDestinationElements();
	  const int nelem = destElements.size();
	  const CkArrayID aid = myInfo->getDestinationArrayID();
	  const CkArray *a = (CkArray*)_localBranch(aid);

	  for (int i=0; i<nelem; ++i) {
	    int pe = a->lastKnown(destElements[i]);
	    if (pe == CkMyPe()) {
	      countDest++;
	      myInfo->addNewLocalDestination(destElements[i]);
	    }
	    else {
	      ComlibPrintf("[%d] destination element %d is no longer local\n", CkMyPe(), (int)destElements[i].data()[0]);
	      myProxy[pe].bracketedDiscover(instid, aid, destElements[i], false);
	    }
	  }
	}

	// Report the number of elements that are now local to this PE (if >0).
	// The new owner PEs will report the counts for those objects that are no longer local to this PE
	if (countSrc > 0 || countDest > 0) {
	  myProxy[0].bracketedContributeDiscovery(instid, CkMyPe(), countSrc, countDest, myEntry->lastKnownIteration);
	}
	
}



/** Determine the PE of a given array index. This will be called for any 
    array element by the previous owner PE.

    If the index is local to the processor, then record this in the local 
    strategy. Also send a message to PE 0 informing PE 0 that the array
    element was located.

    If the array element is not found locally, call this method on the last 
    known location for the element.

*/
void ComlibManager::bracketedDiscover(int instid, CkArrayID aid, CkArrayIndexMax &idx, int isSrc) {
	ComlibManagerPrintf("[%d] bracketedDiscover\n", CkMyPe());
	CkArray *a = (CkArray *)_localBranch(aid);
	int pe = a->lastKnown(idx);
	CProxy_ComlibManager myProxy(thisgroup);
	StrategyTableEntry *myEntry = converseManager->getStrategyTable(instid);

	if (pe == CkMyPe()) {
	        // Object was found locally
	 
	        // notify PE 0
	        myProxy[0].bracketedContributeDiscovery(instid, pe, isSrc?1:0, isSrc?0:1, myEntry->lastKnownIteration);
	  

		// Find local strategy instance

		// ComlibArrayInfo *myInfo = &(dynamic_cast<CharmStrategy*>(getStrategy(instid))->ainfo);
		CkAssert((unsigned long)myEntry->strategy > 0x1000);
		ComlibArrayInfo *myInfo = &dynamic_cast<CharmStrategy*>(myEntry->strategy)->ainfo;
		CkAssert((unsigned long)myInfo > 0x1000);

		if (isSrc) {
		        // Add the element as a source element for the strategy
			ComlibManagerPrintf("[%d] bracketedDiscover addSource\n", CkMyPe());
			CkAssert((unsigned long)myInfo > 0x1000);
			myInfo->addNewLocalSource(idx);

			ComlibManagerPrintf("[%d] bracketedDiscover updating numElements\n", CkMyPe());
			myEntry->numElements = myInfo->getLocalSrc();		
		}
		else {
		        // Add the element as a Destination element for the strategy
			ComlibManagerPrintf("[%d] bracketedDiscover addNewDestination\n", CkMyPe());
			myInfo->addNewLocalDestination(idx);
		}
	} else {
		ComlibManagerPrintf("Keep On Forwarding*********************\n");
		// forward to next potential processor
		myProxy[pe].bracketedDiscover(instid, aid, idx, isSrc);
	}
}



/** On PE 0, record the notifications from all PEs about the actual locations of each
    array element. Count the number of elements discovered. Once the new location of 
    every elements has been discovered, broadcast the new locations to all PEs by 
    invoking bracketedReceiveNewPeList.
*/
void ComlibManager::bracketedContributeDiscovery(int instid, int pe, int nsrc, int ndest, int step) {
	CkAssert(CkMyPe() == 0);
	ComlibManagerPrintf("[%d] bracketedContributeDiscovery pe=%d nsrc=%d ndest=%d step=%d\n", CkMyPe(), pe, nsrc, ndest, step);
	StrategyTableEntry *myEntry = converseManager->getStrategyTable(instid);
	if (myEntry->peList == 0) {
		myEntry->peList = new int[CkNumPes()+2];
		// peList[CkNumPes()] keeps the sum of all source objects discovered
		// peList[CkNumPes()+1] keeps the sum of all destination objects discovered
		for (int i=0; i<CkNumPes()+2; ++i) myEntry->peList[i]=0;
		ComlibManagerPrintf("[%d] bracketedContributeDiscovery zeroing new peList\n", CkMyPe());
	}
	myEntry->peList[CkNumPes()] += nsrc;
	myEntry->peList[CkNumPes()+1] += ndest;
	// update the count for the sender processor. peList[i] is:
	// 0 if proc "i" has no objects,
	// 1 if proc "i" has only source objects,
	// 2 if proc "i" has only destination objects,
	// 3 if proc "i" has both source and destination objects
	// the following code maintains the property of peList!
	if (nsrc > 0) myEntry->peList[pe] |= 1;
	if (ndest > 0) myEntry->peList[pe] |= 2;

	ComlibArrayInfo *myInfo = &dynamic_cast<CharmStrategy*>(myEntry->strategy)->ainfo;
	CkAssert((unsigned long)myInfo > 0x1000);

	
//	ComlibManagerPrintf("[%d] bracketedContributeDiscovery myEntry->peList[CkNumPes()]=%d  myInfo->getTotalSrc()=%d\n", CkMyPe(), myEntry->peList[CkNumPes()], myInfo->getTotalSrc());
//	ComlibManagerPrintf("[%d] bracketedContributeDiscovery myEntry->peList[CkNumPes()+1]=%d  myInfo->getTotalDest()=%d\n", CkMyPe(), myEntry->peList[CkNumPes()+1], myInfo->getTotalDest());
//		
	if (myEntry->peList[CkNumPes()] == myInfo->getTotalSrc() &&
			myEntry->peList[CkNumPes()+1] == myInfo->getTotalDest()) {
		// discovery process finished, broadcast the new pe list

		CProxy_ComlibManager myProxy(thisgroup);
		
		ComlibManagerPrintf("[%d] bracketedContributeDiscovery calling bracketedReceiveNewPeList %d/%d, %d/%d\n", 	
				CkMyPe(), 
				myEntry->peList[CkNumPes()], myInfo->getTotalSrc(), 
				myEntry->peList[CkNumPes()+1], myInfo->getTotalDest() );
		
		printPeList("bracketedContributeDiscovery peList=", myEntry->peList);
		myProxy.bracketedReceiveNewPeList(instid, step, myEntry->peList);
		delete myEntry->peList;
		myEntry->peList = NULL;
	} else {
		ComlibManagerPrintf("[%d] bracketedContributeDiscovery NOT calling bracketedReceiveNewPeList %d/%d, %d/%d\n", 
				CkMyPe(), 
				myEntry->peList[CkNumPes()], myInfo->getTotalSrc(), 
				myEntry->peList[CkNumPes()+1], myInfo->getTotalDest() );
	}

}


void ComlibManager::printPeList(const char* note, int *count)
{
	char *buf = (char*)malloc(1024*64);
	sprintf(buf, "[%d] %s ", CkMyPe(), note);
	for(int i=0;i<CkNumPes();i++){
		switch (count[i]){
		case 0:
			sprintf(buf+strlen(buf), " %d:no ", i);
			break;
		case 1:
			sprintf(buf+strlen(buf), " %d:source ", i);
			break;
		case 2:
			sprintf(buf+strlen(buf), " %d:dest ", i);
			break;
		case 3:
			sprintf(buf+strlen(buf), " %d:both ", i);
			break;
		}
	}
	
	sprintf(buf+strlen(buf), ", all source objects discovered =  %d, all destination objects discovered = %d",  count[CkNumPes()] ,  count[CkNumPes()+1] );
		
	ComlibPrintf("%s\n", buf);
	free(buf);
}

/***************************************************************************
 * Delegation framework section:
 *
 * Reimplementation of the main routines needed for the delegation framework:
 * this routines will be called when a message is sent through a proxy which has
 * been associated with comlib.
 ***************************************************************************/

extern int _charmHandlerIdx;
void msg_prepareSend_noinline(CkArrayMessage *msg, int ep,CkArrayID aid);


/** 
    Handle messages sent via ArraySend. These are single point to point messages 
    to array elements.

    This method should not optimize direct sends in the case where buffering occurs

 */
void ComlibManager::ArraySend(CkDelegateData *pd,int ep, void *msg, 
		const CkArrayIndexMax &idx, CkArrayID a){

	CkAssert(pd != NULL);
	ComlibDelegateData *ci = static_cast<ComlibDelegateData *>(pd);
	int instid = ci->getID();
	CkAssert(instid != 0);

	// Reading from two hash tables is a big overhead
	CkArray *amgr = CkArrayID::CkLocalBranch(a);
	int dest_proc = amgr->lastKnown(idx);

	register envelope * env = UsrToEnv(msg);
	msg_prepareSend_noinline((CkArrayMessage*)msg, ep, a);

	env->getsetArrayIndex()=idx;
	env->setUsed(0);
	((CmiMsgHeaderExt *)env)->stratid = instid;

	CkPackMessage(&env);
	CmiSetHandler(env, CkpvAccess(RecvmsgHandle));        
	
	CharmMessageHolder *cmsg = new CharmMessageHolder((char *)msg, dest_proc, CMH_ARRAYSEND);
	
	if(shouldBufferMessagesNow(instid)){
	  delayMessageSendBuffer[instid].insert(cmsg);
	  StrategyTableEntry *myEntry = converseManager->getStrategyTable(instid);
	  int step = myEntry->lastKnownIteration;
	  ComlibManagerPrintf("[%d] ComlibManager::ArraySend BUFFERED OUTGOING: now buffer contains %d messages step=%d\n",CkMyPe(), delayMessageSendBuffer[instid].size(), step);
	} else {
	  ComlibPrintf("ComlibManager::ArraySend NOT BUFFERING inserting message into strategy %d\n",instid);
	  
	  if(dest_proc == CkMyPe()){  
	    // Directly send if object is local
	    amgr->deliver((CkArrayMessage *)msg, CkDeliver_queue);
	    return;
	  } else {
	    // Send through converse level strategy if non-local
	    converseManager->insertMessage(cmsg, instid);
	  }
	  
	}

}


#include "qd.h"
//CkpvExtern(QdState*, _qd);

void ComlibManager::GroupSend(CkDelegateData *pd,int ep, void *msg, int onPE, CkGroupID gid){

	CkAssert(pd != NULL);
	ComlibDelegateData *ci = static_cast<ComlibDelegateData *>(pd);
	int instid = ci->getID();

	int dest_proc = onPE;

	ComlibPrintf("Send Data %d %d %d\n", CkMyPe(), dest_proc, 
			UsrToEnv(msg)->getTotalsize());

	register envelope * env = UsrToEnv(msg);
	if(dest_proc == CkMyPe()){
		_SET_USED(env, 0);
		CkSendMsgBranch(ep, msg, dest_proc, gid);
		return;
	}

	((CmiMsgHeaderExt *)env)->stratid = instid;
	CpvAccess(_qd)->create(1);

	env->setMsgtype(ForBocMsg);
	env->setEpIdx(ep);
	env->setGroupNum(gid);
	env->setSrcPe(CkMyPe());
	env->setUsed(0);

	CkPackMessage(&env);
	CmiSetHandler(env, _charmHandlerIdx);

	CharmMessageHolder *cmsg = new CharmMessageHolder((char *)msg, dest_proc, CMH_GROUPSEND); 

	if(shouldBufferMessagesNow(instid)){
	  StrategyTableEntry *myEntry = converseManager->getStrategyTable(instid);
	  int step = myEntry->lastKnownIteration;
	  ComlibPrintf("ComlibManager::GroupSend Buffering message for %d step=%d\n",instid, step);
	  delayMessageSendBuffer[instid].insert(cmsg);
	} else {
	  ComlibPrintf("ComlibManager::GroupSend inserting message into strategy %d\n",instid);
	  converseManager->insertMessage(cmsg, instid);
	}


}

void ComlibManager::ArrayBroadcast(CkDelegateData *pd,int ep,void *m,CkArrayID a){
	ComlibPrintf("[%d] Array Broadcast \n", CkMyPe());

	CkAssert(pd != NULL);
	ComlibDelegateData *ci = static_cast<ComlibDelegateData *>(pd);
	int instid = ci->getID();

	register envelope * env = UsrToEnv(m);
	msg_prepareSend_noinline((CkArrayMessage*)m, ep, a);

	env->getsetArrayIndex()= dummyArrayIndex;
	((CmiMsgHeaderExt *)env)->stratid = instid;

	CmiSetHandler(env, CkpvAccess(RecvmsgHandle));

	CharmMessageHolder *cmsg = new CharmMessageHolder((char *)m, IS_BROADCAST, CMH_ARRAYBROADCAST);
	cmsg->npes = 0;
	cmsg->sec_id = NULL;
	cmsg->array_id = a;

	multicast(cmsg, instid);
}

void ComlibManager::ArraySectionSend(CkDelegateData *pd,int ep, void *m, 
		int nsid, CkSectionID *s, int opts) {

    CkAssert(nsid == 1);
	CkAssert(pd != NULL);
	ComlibDelegateData *ci = static_cast<ComlibDelegateData *>(pd);
	int instid = ci->getID();

	ComlibPrintf("[%d] Array Section Send \n", CkMyPe());


	//Provide a dummy dest proc as it does not matter for mulitcast 
	CharmMessageHolder *cmsg = new CharmMessageHolder((char *)m, IS_SECTION_MULTICAST, CMH_ARRAYSECTIONSEND);
	cmsg->npes = 0;
	cmsg->sec_id = s;
	cmsg->array_id = s->_cookie.aid;
	

	msg_prepareSend_noinline((CkArrayMessage*)m, ep, s->_cookie.aid);
	
	register envelope * env = UsrToEnv(m);
	env->getsetArrayIndex()= dummyArrayIndex;
	((CmiMsgHeaderExt *)env)->stratid = instid;
	
	CmiSetHandler(env, CkpvAccess(RecvmsgHandle));
	
	env->setUsed(0);
	CkPackMessage(&env);
	

	CkSectionInfo minfo;
	minfo.type = COMLIB_MULTICAST_MESSAGE;
	minfo.sInfo.cInfo.instId = ci->getID();
	//minfo.sInfo.cInfo.status = COMLIB_MULTICAST_ALL;  
	minfo.sInfo.cInfo.id = 0; 
	minfo.pe = CkMyPe();
	((CkMcastBaseMsg *)env)->_cookie = minfo;    
	
	multicast(cmsg, instid);
}

void ComlibManager::GroupBroadcast(CkDelegateData *pd,int ep,void *m,CkGroupID g) {

	CkAssert(pd != NULL);
	ComlibDelegateData *ci = static_cast<ComlibDelegateData *>(pd);
	int instid = ci->getID();
	CkAssert(instid!=0);

	register envelope * env = UsrToEnv(m);

	CpvAccess(_qd)->create(1);

	env->setMsgtype(ForBocMsg);
	env->setEpIdx(ep);
	env->setGroupNum(g);
	env->setSrcPe(CkMyPe());
	env->setUsed(0);
	((CmiMsgHeaderExt *)env)->stratid = instid;

	CkPackMessage(&env);
	CmiSetHandler(env, _charmHandlerIdx);

	//Provide a dummy dest proc as it does not matter for mulitcast 
	CharmMessageHolder *cmsg = new CharmMessageHolder((char *)m,IS_BROADCAST, CMH_GROUPBROADCAST);

	cmsg->npes = 0;
	//cmsg->pelist = NULL;

	multicast(cmsg, instid);
}


/** 
    Multicast the message with the specified strategy(instid).
    This method is used in: ArrayBroadcast, ArraySectionSend, and GroupBroadcast
 */
void ComlibManager::multicast(CharmMessageHolder *cmsg, int instid) {
        CkAssert(instid != 0);
	register envelope * env = UsrToEnv(cmsg->getCharmMessage());

#if DEBUG
	StrategyTableEntry *myEntry = converseManager->getStrategyTable(instid);
	ComlibPrintf("[%d] multicast setupComplete=%d %s %s\n", CkMyPe(), setupComplete, myEntry->getErrorModeString(), myEntry->getErrorModeServerString());
#endif
 
	env->setUsed(0);
	CkPackMessage(&env);
	
	if(shouldBufferMessagesNow(instid)){
	  cmsg->saveCopyOf_sec_id();
	  StrategyTableEntry *myEntry = converseManager->getStrategyTable(instid);
	  int step = myEntry->lastKnownIteration;
	  ComlibPrintf("[%d] ComlibManager::multicast Buffering message for %d lastKnownIteration=%d\n", CkMyPe(),instid, step);
	  delayMessageSendBuffer[instid].insert(cmsg);
	} else {
	  converseManager->insertMessage(cmsg, instid);
	}
	
}


void ComlibManager::printDiagnostics(int instid){
  CkPrintf("[%d]     delayMessageSendBuffer.size()=%d\n", CkMyPe(), delayMessageSendBuffer[instid].size());
}


void ComlibManager::printDiagnostics(){


  std::map<ComlibInstanceHandle, std::set<CharmMessageHolder*> >::iterator iter;
  for(iter = delayMessageSendBuffer.begin(); iter != delayMessageSendBuffer.end(); ++iter){
    int instid = iter->first;
    int size = iter->second.size();
    
    if(size>0 || true){
      CkPrintf("[%d] delayMessageSendBuffer[instid=%d] contains %d messages\n", CkMyPe(), instid, size);
      
      if(! shouldBufferMessagesNow(instid)){
	CkPrintf("[%d] printDiagnostics: No messages should be still in delayMessageSendBuffer[instid=%d]\n", CkMyPe(), instid);
      } else {
	CkPrintf("[%d] printDiagnostics: Messages still ought to be delayed in delayMessageSendBuffer[instid=%d]\n", CkMyPe(), instid);
      }
      
      StrategyTableEntry *myEntry = converseManager->getStrategyTable(instid);
      CkPrintf("[%d] printDiagnostics[instid=%d] setupComplete=%d %s %s %s bufferOutgoing=%d\n", (int)CkMyPe(), (int)instid, (int)setupComplete, myEntry->errorModeString(),  myEntry->errorModeServerString(),  myEntry->discoveryModeString(), (int)myEntry->bufferOutgoing);



    }
  }
  
  CkpvAccess(conv_com_object).printDiagnostics();
  
}



CkDelegateData* ComlibManager::ckCopyDelegateData(CkDelegateData *data) {
	//ComlibDelegateData *inst = static_cast<ComlibDelegateData *>(data);
	data->ref();
	return data;
	//return (new ComlibDelegateData(inst->getID()));
}

CkDelegateData* ComlibManager::DelegatePointerPup(PUP::er &p,
		CkDelegateData *pd) {
	if (!p.isUnpacking() && pd == NULL) CkAbort("Tryed to pup a null ComlibDelegateData!\n");
	//CmiBool isNotNull = pd!=NULL ? CmiTrue : CmiFalse;
	ComlibDelegateData *inst = static_cast<ComlibDelegateData *>(pd);

	// call a migration constructor
	if (p.isUnpacking()) inst = new ComlibDelegateData((CkMigrateMessage*)0);
	inst->pup(p);
	/*
  p | isNotNull;
  if (isNotNull) {
    if (p.isUnpacking()) inst = new ComlibInstanceHandle();
    inst->pup(p);
  }
	 */
	return inst;
}


//Collect statistics from all the processors, also gets the list of
//array elements on each processor.
void ComlibManager::collectStats(ComlibLocalStats &stat, int pe) {
}


/// @TODO: eliminate AtSync and move toward anytime migration!
void ComlibManager::AtSync() {

}


/***************************************************************************
 * User section:
 *
 * Interface available to the user to interact with the ComlibManager
 ***************************************************************************/

/** Permanently associate the given proxy with comlib, to use the instance
    represented by cinst. All messages sent through the proxy will be forwarded
    by comlib. */
void ComlibAssociateProxy(ComlibInstanceHandle cinst, CProxy &proxy) {
  if(CkNumPes() > 1){
	CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
	proxy.ckDelegate(cgproxy.ckLocalBranch(), new ComlibDelegateData(cinst));
  } else {
    ComlibPrintf("Doing nothing in ComlibAssociateProxy because we have only 1 pe\n"); 
  }
}

/** Permanently assiciate the given proxy with comlib, to use the strategy
    represented by strat. All messages sent through the proxy will be forwarded
    by comlib. */
void ComlibAssociateProxy(Strategy *strat, CProxy &proxy) {
	ComlibAssociateProxy(strat->getHandle(), proxy);
}  

/* OLD DESCRIPTION! Register a strategy to the comlib framework, and return a
    handle to be used in the future with ComlibAssociateProxy to associate the
    strategy with a proxy. If one strategy is registered more than once, the
    handles will be different, but they has to be considered as aliases. */

/** Provided only for backward compatibility. A strategy is registered at the
    converse level as soon as it is created (see Strategy constructor). This
    function just retrieve a handle from the strategy itself. */
ComlibInstanceHandle ComlibRegister(Strategy *strat) {
	return strat->getHandle();
}

/** Call beginIteration on the ComlibManager with the instance handle for the strategy associated with the proxy. 
 *   If no strategy has been associated with the proxy, nothing is done in this function.
 */
void ComlibBegin(CProxy &proxy, int iteration) {
  if(CkNumPes() > 1){
	ComlibDelegateData *cinst = static_cast<ComlibDelegateData *>(proxy.ckDelegatedPtr());
	if(cinst==NULL)
		return;
	CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
	(cgproxy.ckLocalBranch())->beginIteration(cinst->getID(), iteration);
  } else {
    ComlibPrintf("Doing nothing in ComlibBegin because we have only 1 pe");  
  }
}

/** Call endIteration on the ComlibManager with the instance handle for the strategy associated with the proxy. 
 *   If no strategy has been associated with the proxy, nothing is done in this function.
 */
void ComlibEnd(CProxy &proxy, int iteration) {
  if(CkNumPes() > 1){
	ComlibDelegateData *cinst = static_cast<ComlibDelegateData *>(proxy.ckDelegatedPtr());
	if(cinst==NULL)
		return;
	CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
	(cgproxy.ckLocalBranch())->endIteration(cinst->getID(), iteration);
  } else {
    ComlibPrintf("Doing nothing in ComlibEnd because we have only 1 pe");  
  }
}

char *routerName;
int sfactor=0;


/** A mainchare, used to initialize the comlib framework at the program startup.
    Its main purpose is to create the ComlibManager group. */
class ComlibManagerMain {
public:
	ComlibManagerMain(CkArgMsg *msg) {

		if(CkMyPe() == 0 && msg !=  NULL)
			CmiGetArgString(msg->argv, "+strategy", &routerName);         

		if(CkMyPe() == 0 && msg !=  NULL)
			CmiGetArgInt(msg->argv, "+spanning_factor", &sfactor);

		CProxy_ComlibManager::ckNew();
	}
};

/***************************************************************************
 * ComlibInstanceHandle section:
 *
 * Implementation of the functions defined in the ComlibInstanceHandle class.
 ***************************************************************************/

ComlibDelegateData::ComlibDelegateData(int instid) : CkDelegateData(), _instid(instid) {
	ComlibManagerPrintf("[%d] Constructing ComlibDelegateData\n", CkMyPe());
	ref();
}


void ComlibInitSectionID(CkSectionID &sid){

	sid._cookie.type = COMLIB_MULTICAST_MESSAGE;
	sid._cookie.pe = CkMyPe();

	sid._cookie.sInfo.cInfo.id = 0;    
	sid.npes = 0;
	sid.pelist = NULL;
}


/** For backward compatibility - for old name commlib. The function
    _registercomlib() is generated by the translator. */
void _registercommlib(void)
{
	static int _done = 0; 
	if(_done) 
		return; 
	_done = 1;
	_registercomlib();
}


void ComlibNotifyMigrationDoneHandler(void *msg) {
	CmiFree(msg);
	CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
	ComlibManager *cmgr_ptr = cgproxy.ckLocalBranch();
	if(cmgr_ptr)
		cmgr_ptr->AtSync();    
}






static void periodicDebugPrintStatus(void* ptr, double currWallTime){
  CkPrintf("[%d] periodicDebugPrintStatus()\n", CkMyPe());

  ComlibManager *mgr = (ComlibManager*)ptr;
  mgr->printDiagnostics();

  CcdCallFnAfterOnPE((CcdVoidFn)periodicDebugPrintStatus, ptr, 4000, CkMyPe());

}





#include "comlib.def.h"

/*@}*/
