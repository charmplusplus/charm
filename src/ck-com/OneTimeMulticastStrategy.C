/**
   @addtogroup ComlibCharmStrategy
   @{
   @file

*/


#include "OneTimeMulticastStrategy.h"
#include <string>

CkpvExtern(CkGroupID, cmgrID);


OneTimeMulticastStrategy::OneTimeMulticastStrategy()
  : Strategy(), CharmStrategy() {
  //  ComlibPrintf("OneTimeMulticastStrategy constructor\n");
  setType(ARRAY_STRATEGY);
}

OneTimeMulticastStrategy::~OneTimeMulticastStrategy() {
}

void OneTimeMulticastStrategy::pup(PUP::er &p){
  Strategy::pup(p);
  CharmStrategy::pup(p);
}


/** Called when the user invokes the entry method on the delegated proxy. */
void OneTimeMulticastStrategy::insertMessage(CharmMessageHolder *cmsg){
  if(cmsg->dest_proc != IS_SECTION_MULTICAST && cmsg->sec_id == NULL) { 
    CkAbort("OneTimeMulticastStrategy can only be used with an array section proxy");
  }
    
  // Create a multicast message containing all information about remote destination objects 
  int needSort = 0;
  ComlibMulticastMsg * multMsg = sinfo.getNewMulticastMessage(cmsg, needSort, getInstance());
  
  // local multicast will re-extract a list of local destination objects (FIXME to make this more efficient)
  localMulticast(cmsg);
  
  // The remote multicast method will send the message to the remote PEs, as specified in multMsg
  remoteMulticast(multMsg);
   
  delete cmsg;    
}



/** Deliver the message to the local elements. */
void OneTimeMulticastStrategy::localMulticast(CharmMessageHolder *cmsg) {
  CkSectionID *sec_id = cmsg->sec_id;
  CkVec< CkArrayIndexMax > localIndices;
  sinfo.getLocalIndices(sec_id->_nElems, sec_id->_elems, sec_id->_cookie.aid, localIndices);
  deliverToIndices(cmsg->getCharmMessage(), localIndices );
}



/** 
    Forward multicast message to all other processors containing destination objects. 
    Uses CmiSyncListSendAndFree for delivery to this strategy's OneTimeMulticastStrategy::handleMessage method.
*/
void OneTimeMulticastStrategy::remoteMulticast(ComlibMulticastMsg * multMsg) {
  envelope *env = UsrToEnv(multMsg);

  // double StartTime = CmiWallTimer();

  int npes = multMsg->nPes;
  
  if(npes == 0) {
    CmiFree(env);
    return;    
  }

  // ComlibPrintf("[%d] remoteMulticast Sending to %d PEs: \n", CkMyPe(), npes);
 
  int *pelist = new int[npes];
  for (int i=0; i<npes; ++i) {
    pelist[i] = multMsg->indicesCount[i].pe;
    //  ComlibPrintf("[%d]   %d messages to pe %d\n", CkMyPe(), multMsg->indicesCount[i].count, multMsg->indicesCount[i].pe);
  }
  
  CmiSetHandler(env, CkpvAccess(comlib_handler));

  ((CmiMsgHeaderBasic *) env)->stratid = getInstance();

  //Collect Multicast Statistics
  RECORD_SENDM_STATS(getInstance(), env->getTotalsize(), pelist, npes);
    
  CkPackMessage(&env);
     
  CmiSyncListSendAndFree(npes, pelist, env->getTotalsize(), (char*)env);
  //CmiSyncBroadcastAndFree(env->getTotalsize(), (char*)env);

  //	traceUserBracketEvent( 2201, StartTime, CmiWallTimer()); 

}




/** 
    Receive an incoming multicast message(sent from OneTimeMulticastStrategy::remoteMulticast).
    Deliver the message to all local elements 
*/
void OneTimeMulticastStrategy::handleMessage(void *msg){
  // ComlibPrintf("[%d] In handleMulticastMessage\n", CkMyPe());

  //	double StartTime = CmiWallTimer();
  
  envelope *env = (envelope *)msg;
  RECORD_RECV_STATS(getInstance(), env->getTotalsize(), env->getSrcPe());
  
  // Extract the list of elements to be delivered locally

  int localElems;
  envelope *newenv;
  CkArrayIndexMax *local_idx_list;    
    
  CkUnpackMessage(&env);
  sinfo.unpack(env, localElems, local_idx_list, newenv);

  ComlibMulticastMsg *newmsg = (ComlibMulticastMsg *)EnvToUsr(newenv);  

  deliverToIndices(newmsg, localElems, local_idx_list );
    
}



/*@}*/
