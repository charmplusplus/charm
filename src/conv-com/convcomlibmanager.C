/**
   @addtogroup ConvComlib
   @{
   @file

   @brief Implementations of convcomlibmanager.h classes. It also defines all
   the converse handlers to perform the broadcast of the strategies to all the
   processors (synchronization process). These routines provide this support
   also to the charm layer.

   @author Sameer Kumar 28/03/04
   @author Heavily revised, Filippo Gioachin 01/06
*/

#include "convcomlibmanager.h"
#include "routerstrategy.h"
#include "StreamingStrategy.h"
#include "MeshStreamingStrategy.h"
#include "pipebroadcastconverse.h"
#include "converse.h"

int com_debug=0;

/// The location in the global table where the converse Comlib manager is located.
CkpvDeclare(ConvComlibManager, conv_com_object);

/***************************************************************************
 * Handlers section:
 *
 * Declarations of all the converse handler IDs used, together with all the
 * functions they are associated with.
 ***************************************************************************/

/** Handler to which send messages inside the comlib framework. To this handler
    arrive all the messages sent by common strategies without specific needs.
    Strategies that need particular rerouting of their messages use their own
    specific handlers. */
CkpvDeclare(int, comlib_handler);
/// Method invoked upon receipt a message routed through comlib.
void *strategyHandler(void *msg) {
    CmiMsgHeaderExt *conv_header = (CmiMsgHeaderExt *) msg;
    int instid = conv_header->stratid;

#ifndef CMK_OPTIMIZE
    // check that the instid is not zero, meaning a possibly uninitialized value
    if (instid == 0) {
      CmiAbort("Comlib strategy ID is zero, did you forget to initialize a variable?\n");
    }
#endif
    
    // when handling a message always call the lowest level
    Strategy *strat = ConvComlibGetStrategy(instid);
    
    strat->handleMessage(msg);
    return NULL;
}

/** @file
 * 
 *  Strategies synchronization process:
 *
 * 1) the strategies are inserted on processor 0 (and possibly in other
 *    processors with the same order. The strategies are marked as "new"
 *
 * 2) when ComlibDoneCreating is called, processor 0 broadcast all the new
 *    strategies to all the processors, and marks them as "inSync"
 *
 * 3) when a processor receives a table it updates its personal table with the
 *    incoming, it marks all the strategies just arrived as "inSync", and it
 *    sends an acknowledgement back to processor 0.
 *
 * 4) when an acknowledgement is received by processor 0, a counter is
 *    decremented. When it reaches 0, all the "inSync" strategies are switched
 *    to status "ready" and they can start working. All the messages in the
 *    tmplist are delivered. The sync is broadcasted.
 *
 * 5) when an acknowledgement is received by a processor other than 0, all the
 *    "inSync" strategies are switched to "ready" and the messages in tmplist
 *    are delivered.
 *
 * 6) in order to prevent two consecutive table broadcasts to interfere with
 *    each other, an additional acknowledgement is sent back by each processor
 *    to processor 0 to allow a new table update to happen.
 */

/** Handler to accept the second acknowledgements, after which a new table can
    be broadcasted. Executed only on processor 0. */
CkpvDeclare(int, comlib_ready);
/// Method to acknowledge the ready status
void *comlibReadyHandler(void *msg) {
  ComlibPrintf("[%d] Received ready acknowledgement\n",CmiMyPe());
  CmiAssert(CkpvAccess(conv_com_object).acksReceived > 0);
  if (--CkpvAccess(conv_com_object).acksReceived == 0) {
    // ok, we are done. Do we have to broadcast a new table?
    ComlibPrintf("Strategy table propagation finished\n");
    CkpvAccess(conv_com_object).busy = false;
    if (CkpvAccess(conv_com_object).doneCreatingScheduled) {
      CkpvAccess(conv_com_object).doneCreating();
    }
  }
  CmiFree(msg);
  return NULL;
}

/** Handler to count and accept the acknowledgements of table received. On
    processor zero this handler counts the number of acks. On all other
    processors it simply accept it */
CkpvDeclare(int, comlib_table_received);

/// Method invoked upon receipt of an acknowledgement of table received
void *comlibTableReceivedHandler(void *msg) {
  if (CmiMyPe() == 0) {
	//    CmiPrintf("Num acks to go: %d\n",CkpvAccess(conv_com_object).acksReceived);
    if (--CkpvAccess(conv_com_object).acksReceived == 0) {
      CkpvAccess(conv_com_object).tableReady();
      // reset acksReceived for the second step
      //CmiPrintf("All acks received, broadcasting message to table_received\n");
      CkpvAccess(conv_com_object).acksReceived = CmiNumPes() - 1;
      CmiSyncBroadcastAndFree(CmiReservedHeaderSize, (char*)msg);
    } else {
      CmiFree(msg);
    }
  } else {
    CkpvAccess(conv_com_object).tableReady();
    CmiSetHandler(msg, CkpvAccess(comlib_ready));
    CmiSyncSendAndFree(0, CmiReservedHeaderSize, (char*)msg);
  }  
  return NULL;
}

/** Handler to broadcast all the strategies to all the processors. This is
    invoked on all processors except processor zero, upon a broadcast message
    from processor zero (in ComlibDoneInserting). */
CkpvDeclare(int, comlib_receive_table);

/// Method invoked upon receipt of the strategy table
void *comlibReceiveTableHandler(void *msg) {
  // unpack the message into a StrategyWrapper
  ComlibPrintf("Received new strategy table\n");
  StrategyWrapper sw;
  PUP::fromMem pm(((char*)msg)+CmiReservedHeaderSize);
  pm|sw;

  // insert the strategies into the local table
  for (int i=0; i<sw.nstrats; ++i) {
    Strategy *current = CkpvAccess(conv_com_object).getStrategy(sw.position[i]);
    if (sw.replace[i] && current != NULL) {
      // delete the old strategy. Since it is requested, it is safe
      delete current;
      current = NULL;
      CkpvAccess(conv_com_object).decrementNumStrats(); 
    }
    if (current == NULL) {
      // if current is NULL either the strategy has never been set yet, or we
      // are replacing it
      CkpvAccess(conv_com_object).setStrategy(sw.position[i], sw.strategy[i]);
      CkpvAccess(conv_com_object).incrementNumStrats(); 
    } else {
      // let's delete the incoming strategy since it is not used
      delete sw.strategy[i];
    }
    CkpvAccess(conv_com_object).inSync(sw.position[i]);
  }

  // cheat about the size of the message
  CmiSetHandler(msg, CkpvAccess(comlib_table_received));
  CmiSyncSendAndFree(0, CmiReservedHeaderSize, (char*)msg);
  return NULL;
}

/***************************************************************************
 * ConvComlibManager section:
 *
 * Implementation of the functions defined in the ConvComlibManager class.
 ***************************************************************************/

ConvComlibManager::ConvComlibManager(): strategyTable(MAX_NUM_STRATS+1){
  nstrats = 0;
  init_flag = false;
  acksReceived = 0;
  doneCreatingScheduled = false;
  busy = false;
}

/** Insert a strategy into the system table, and return a handle to be used to
    access the strategy later. This handle is the location in the strategy
    table. From the strategy itself the user can get a handle for its usage. */
int ConvComlibManager::insertStrategy(Strategy *s) {

  // This is not right, remove this stupid restriction!
  if(nstrats >= MAX_NUM_STRATS)
    CmiAbort("Too Many strategies\n");

  int index = ++nstrats;
  StrategyTableEntry &st = strategyTable[index];
  
  if(st.strategy != NULL) CmiAbort("Trying to insert a strategy over another one!");
  
  st.strategy = s;
  st.isNew = 1;
  st.bracketedSetupFinished = 2*CkNumPes();
  
  //s->setInstance(index);
  
  // if the strategy is pure converse or pure charm the following
  // line is a duplication, but if a charm strategy embed a converse
  // strategy it is necessary to set the instanceID in both
  //s->getConverseStrategy()->setInstance(index); DEPRECATED

  return index;
}

void ConvComlibManager::doneCreating() {
  ComlibPrintf("Called doneCreating\n");
  if (busy) {
    // we have to delay the table broadcast because we are in the middle of another one
    doneCreatingScheduled = true;
    return;
  }
  // if we reach here it means we are not busy and we can proceed
  busy = true;
  acksReceived = CmiNumPes() - 1;
  int count = 0;
  for (int i=1; i<=nstrats; ++i) {
    if (strategyTable[i].isNew) {
      count++;
    }
  }

  if (count > 0) {
    // create the wrapper and link the strategies there
    StrategyWrapper sw(count);
    count = 0;
    for (int i=1; i<=nstrats; ++i) {
      if (strategyTable[i].isNew) {
    	  sw.position[count] = i;
    	  sw.replace[count] = false;
    	  sw.strategy[count] = strategyTable[i].strategy;
    	  count++;
    	  CkpvAccess(conv_com_object).inSync(i);
      }
    }

    // pup the wrapper into a message
    PUP::sizer ps;
    ps|sw;
    char *msg = (char*)CmiAlloc(ps.size() + CmiReservedHeaderSize);
    PUP::toMem pm(msg+CmiReservedHeaderSize);
    //int size = ps.size();
    //pm|size;
    pm|sw;
    //for (int i=CmiReservedHeaderSize; i<CmiReservedHeaderSize+size; ++i) {
    //  CmiPrintf("%x",((char*)msg)[i]);
    //}
    //CmiPrintf("\n");
    CmiSetHandler(msg, CkpvAccess(comlib_receive_table));
    CmiSyncBroadcastAndFree(ps.size()+CmiReservedHeaderSize, msg);

    /* NOT USED NOW!
    // call the finalizeCreation after the strategies has been packed
    for (int i=0; i<strategyTable.size(); ++i) {
      if (strategyTable[i].isNew) strategyTable[i].strategy->finalizeCreation();
    }
    */
  } else {
    busy = false;
  }
}

void ConvComlibManager::tableReady() {
  for (int i=1; i<strategyTable.size(); ++i) {
    if (strategyTable[i].isInSync) {
      ComlibPrintf("[%d] ConvComlibManager::tableReady Enabling strategy %d\n",CmiMyPe(),i);
      strategyTable[i].isInSync = 0;
      enableStrategy(i);
    }
  }
}

#include "ComlibStrategy.h"

void ConvComlibManager::enableStrategy(int i) {
  strategyTable[i].isReady = 1;
  // deliver all the messages in the tmplist to the strategy
  MessageHolder *mh;
  while ((mh=strategyTable[i].tmplist.deq()) != NULL) {
    CharmMessageHolder*cmh = (CharmMessageHolder*)mh;
    //    cmh->checkme();
    cmh->sec_id = cmh->copy_of_sec_id;

#if DEBUG_MULTICAST
    int nelem =cmh->sec_id->_nElems;
    CkPrintf("[%d] enableStrategy() pushing message into strategy %d using copy of sec_id stored when enqueuing message (message=%p nelem=%d)\n",CmiMyPe(),i, mh, nelem);
#endif

    strategyTable[i].strategy->insertMessage(mh);
    //    ((CharmMessageHolder*)mh)->freeCopyOf_sec_id();
  }
  for (int j=0; j<strategyTable[i].call_doneInserting; ++j) {
    strategyTable[i].strategy->doneInserting();
  }
  strategyTable[i].call_doneInserting = 0;
}



/// Handler for dummy messages...
/// @TODO: Find out why we need this stupid empty messages and get rid of them (they are used only in router strategies)
CkpvDeclare(int, RecvdummyHandle);
//handler for dummy messages
void recv_dummy(void *msg){
    ComlibPrintf("Received Dummy %d\n", CmiMyPe());
    CmiFree(msg);
}

/// @TODO: hack for PipeBroadcastStrategy to register its handlers, fix it.
//extern void propagate_handler(void *);
extern void propagate_handler_frag(void *);


/** At startup on each processor, this method is called. 
    This sets up the converse level comlib strategies.

    This is called before any mainchare main functions.
 */
void initConvComlibManager(){ 

    if(!CkpvInitialized(conv_com_object))
      CkpvInitialize(ConvComlibManager, conv_com_object);
    
    
    if(CkpvAccess(conv_com_object).getInitialized()) {
      CmiPrintf("Comlib initialized more than once!\n");
      return;
    }
    
    CkpvInitialize(int, RecvdummyHandle);
    CkpvAccess(RecvdummyHandle) = CkRegisterHandler((CmiHandler)recv_dummy);

    CkpvInitialize(int, comlib_receive_table);
    CkpvAccess(comlib_receive_table) = CkRegisterHandler((CmiHandler)comlibReceiveTableHandler);
    CkpvInitialize(int, comlib_table_received);
    CkpvAccess(comlib_table_received) = CkRegisterHandler((CmiHandler)comlibTableReceivedHandler);
    CkpvInitialize(int, comlib_ready);
    CkpvAccess(comlib_ready) = CkRegisterHandler((CmiHandler)comlibReadyHandler);

    // init strategy specific variables

    // router strategy
    CkpvInitialize(int, RouterRecvHandle);
    CkpvAccess(RouterRecvHandle) = CkRegisterHandler((CmiHandler)routerRecvManyCombinedMsg);
    CkpvInitialize(int, RouterProcHandle);
    CkpvAccess(RouterProcHandle) = CkRegisterHandler((CmiHandler)routerProcManyCombinedMsg);
    CkpvInitialize(int, RouterDummyHandle);
    CkpvAccess(RouterDummyHandle) = CkRegisterHandler((CmiHandler)routerDummyMsg);    

    // streaming strategy
    CpvInitialize(int, streaming_handler_id);
    CpvAccess(streaming_handler_id) = CmiRegisterHandler(StreamingHandlerFn);

    // mesh streaming strategy
    CkpvInitialize(int, streaming_column_handler_id);
    CkpvAccess(streaming_column_handler_id) = CkRegisterHandler(streaming_column_handler);

    // pipelined broadcast
    CkpvInitialize(int, pipeline_handler);
    CkpvInitialize(int, pipeline_frag_handler);
    CkpvAccess(pipeline_handler) = CkRegisterHandler((CmiHandler)PipelineHandler);
    CkpvAccess(pipeline_frag_handler) = CkRegisterHandler((CmiHandler)PipelineFragmentHandler);
    
    // general handler
    CkpvInitialize(int, comlib_handler);
    CkpvAccess(comlib_handler) = CkRegisterHandler((CmiHandler) strategyHandler);

    //PUPable_reg(Strategy); ABSTRACT
    //PUPable_reg(ConvComlibInstanceHandle);
    if (CmiMyRank() == 0) {
   	  PUPable_reg(RouterStrategy);
      PUPable_reg(StreamingStrategy);
      PUPable_reg(MeshStreamingStrategy);
      PUPable_reg(PipeBroadcastConverse);
      PUPable_reg(MessageHolder);
    }
    CkpvAccess(conv_com_object).setInitialized();
}

// #ifdef __cplusplus
// extern "C" {
// #endif
//   void ComlibInit() {initComlibManager();}
// #ifdef __cplusplus
// }
// #endif


/***************************************************************************
 * User section:
 *
 * Implementation of the functions used by the user
 ***************************************************************************/

Strategy *ConvComlibGetStrategy(int loc) {
    //Calling converse strategy lets Charm++ strategies one strategy
    //table entry but multiple layers of strategies (Charm on top of Converse).
    return CkpvAccess(conv_com_object).getStrategy(loc);
}

// Why is this here for? Guess it is for the routers...
void ConvComlibScheduleDoneInserting(int loc) {
  CkpvAccess(conv_com_object).getStrategyTable(loc)->call_doneInserting++;
}


  
void ConvComlibManager::insertMessage(MessageHolder* msg, int instid) {
  ComlibPrintf("[%d] enqueuing message for strategy %d in tmplist\n",CmiMyPe(),instid);
#ifndef CMK_OPTIMIZE
  if (instid == 0) CmiAbort("Trying to send a message through comlib strategy zero, did you forget to initialize zome variable?\n");
#endif
  if (isReady(instid)) {
    ComlibPrintf("[%d] insertMessage inserting into strategy\n", CmiMyPe());
    strategyTable[instid].strategy->insertMessage(msg);
  }
  else{
    // In this case, we will enqueue the messages, so they can be delivered 
    // once the strategy is initialized and ready to be used. This is tricky
    // because the message contains a pointer to its "CkSectionID *sec_id".
    // Somewhere this structure is freed before these messages are dequeued.
    // Thus we try to copy the sec_id by calling saveCopyOf_sec_id() which
    // performs a shallow copy of sec_id, which appears to be sufficient.
    // This is only a shallow copy, so some pointers inside may well be 
    // dead.
    //
    // FIXME: This should really be fixed in some other manner, such 
    // as not letting the sec_id get deleted in the first place. Or the relevant 
    // sec_id fields could be copied into the CharmMessageHolder object.

#if DEBUG_MULTICAST
    int nelem = ((CharmMessageHolder*)msg)->sec_id->_nElems;
    void * si = ((CharmMessageHolder*)msg)->sec_id;
    ComlibPrintf("[%d] insertMessage inserting into tmplist with msg=%p si=%p nelem=%d\n", CmiMyPe(), msg, si, nelem);
#endif

    ComlibPrintf("[%d] msg=%p\n", CkMyPe(), dynamic_cast<CharmMessageHolder*>(msg));
    ((CharmMessageHolder*)msg)->saveCopyOf_sec_id();
    ComlibPrintf("[%d] insertMessage inserting into tmplist  intid=%d\n", CmiMyPe(), instid);
    strategyTable[instid].tmplist.enq(msg);
  }

}
  



void ConvComlibManager::printDiagnostics(){


  //  CkVec<StrategyTableEntry> strategyTable
  int ready = 0;
  int tmplistTotal = 0;

  int size = strategyTable.size();
  //  CkPrintf("[%d]   converse level strategyTable.size()=%d\n", CkMyPe(), size);
  for(int i=0;i<size;i++){
    if(strategyTable[i].isReady){
      ready++;
      //  CkPrintf("[%d]   strategyTable[%d] is ready\n", CkMyPe(), i);
    } else {
      // CkPrintf("[%d]   strategyTable[%d] is not ready\n", CkMyPe(), i);
    }
    
    int nmsg = strategyTable[i].tmplist.length();
    
    tmplistTotal += nmsg;
    //      CkPrintf("[%d]   strategyTable[%d] has %d messages in tmplist\n", CkMyPe(), i, nmsg);
  }

  if(tmplistTotal>0){
    CkPrintf("[%d]  %d of %d converse strategies are ready (%d msgs buffered)\n", CkMyPe(), ready, size, tmplistTotal);
  }
}





/*@}*/
