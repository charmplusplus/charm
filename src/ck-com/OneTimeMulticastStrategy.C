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
  remoteMulticast(multMsg, true);
   
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
    Forward multicast message to our successor processors in the spanning tree. 
    Uses CmiSyncListSendAndFree for delivery to this strategy's OneTimeMulticastStrategy::handleMessage method.
*/
void OneTimeMulticastStrategy::remoteMulticast(ComlibMulticastMsg * multMsg, bool rootPE) {

  envelope *env = UsrToEnv(multMsg);
    
  int npes;
  int *pelist;   
  
  /// The index into the PE list in the message
  int myIndex = -10000; 
  const int totalDestPEs = multMsg->nPes;
  const int myPe = CkMyPe();
  
  // Find my index in the list of all destination PEs
  if(rootPE){
    myIndex = -1;
  } else {
    for (int i=0; i<totalDestPEs; ++i) {
      if(multMsg->indicesCount[i].pe == myPe){
	myIndex = i;
	break;
      }
    }
  }
  
  CkAssert(myIndex != -10000); // Sanity check
    
  determineNextHopPEs(multMsg, myIndex, pelist, npes );
  
  if(npes == 0) {
    CmiFree(env);
    return;
  }
  
  CmiSetHandler(env, CkpvAccess(comlib_handler));
  ((CmiMsgHeaderBasic *) env)->stratid = getInstance();  
  CkPackMessage(&env);

  //Collect Multicast Statistics
  RECORD_SENDM_STATS(getInstance(), env->getTotalsize(), pelist, npes);
  

#if DEBUG
  for(int i=0;i<npes;i++){
    CkPrintf("[%d] Multicast to %d  rootPE=%d\n", CkMyPe(), pelist[i], (int)rootPE);
  }
#endif

  //  if(npes > 0)
  CmiSyncListSendAndFree(npes, pelist, env->getTotalsize(), (char*)env);
  
  delete[] pelist;
}


/** 
    Fill in pelist and npes to which the multicast message will be forwarded from this PE.
    Caller should delete pelist if npes>0.
*/
void OneTimeMulticastStrategy::determineNextHopPEs(ComlibMulticastMsg * multMsg, int myIndex, int * &pelist, int &npes ) {
  if(myIndex==-1){
    // We are at a root node of the spanning tree. 
    // We will forward the message to all other PEs in the destination list.
    npes = multMsg->nPes;
    
    pelist = new int[npes];
    for (int i=0; i<npes; ++i) {
      pelist[i] = multMsg->indicesCount[i].pe;
    }
  } else {
    // We are at a leaf node of the spanning tree. 
    npes = 0;
  }
  
}




/** 
    Receive an incoming multicast message(sent from OneTimeMulticastStrategy::remoteMulticast).
    Deliver the message to all local elements 
*/
void OneTimeMulticastStrategy::handleMessage(void *msg){
  envelope *env = (envelope *)msg;
  CkUnpackMessage(&env);
  
  ComlibMulticastMsg* multMsg = (ComlibMulticastMsg*)EnvToUsr(env);
  
  // Don't use msg after this point. Instead use the unpacked env
  
  RECORD_RECV_STATS(getInstance(), env->getTotalsize(), env->getSrcPe());
  
  // Deliver to objects marked as local in the message
  int localElems;
  envelope *newenv;
  CkArrayIndexMax *local_idx_list;  
  sinfo.unpack(env, localElems, local_idx_list, newenv);
  ComlibMulticastMsg *newmsg = (ComlibMulticastMsg *)EnvToUsr(newenv);  
  deliverToIndices(newmsg, localElems, local_idx_list );
  
  // Forward on to other processors if necessary
  remoteMulticast( multMsg, false);

}





/** 
    Fill in pelist and npes to which the multicast message will be forwarded from this PE.
    Caller should delete pelist if npes>0.
*/
void OneTimeRingMulticastStrategy::determineNextHopPEs(ComlibMulticastMsg * multMsg, int myIndex, int * &pelist, int &npes ) {
  const int totalDestPEs = multMsg->nPes;
  const int myPe = CkMyPe();

  if(myIndex == totalDestPEs-1){
    // Final PE won't send to anyone
    npes = 0;
    return;
  } else {
    // All non-final PEs will send to next PE in list
    npes = 1;
    pelist = new int[1];
    pelist[0] = multMsg->indicesCount[myIndex+1].pe;
  }

}




/** 
    Fill in pelist and npes to which the multicast message will be forwarded from this PE.
    Caller should delete pelist if npes>0.
*/
void OneTimeTreeMulticastStrategy::determineNextHopPEs(ComlibMulticastMsg * multMsg, int myIndex, int * &pelist, int &npes ) {
  const int totalDestPEs = multMsg->nPes;
  const int myPe = CkMyPe();
  
  // The logical indices start at 0 = root node. Logical index i corresponds to the entry i+1 in the array of PEs in the message
  
  // All non-final PEs will send to next PE in list
  int sendLogicalIndexStart = degree*(myIndex+1) + 1;       // inclusive
  int sendLogicalIndexEnd = sendLogicalIndexStart + degree - 1;   // inclusive
  
  if(sendLogicalIndexEnd-1 >= totalDestPEs){
    sendLogicalIndexEnd = totalDestPEs;
  }

  int numSend = sendLogicalIndexEnd - sendLogicalIndexStart + 1;
  if(numSend <= 0){
    npes = 0;
    return;
  }
  

#if DEBUG
  if(numSend > 0)
    CkPrintf("Tree logical index %d sending to logical %d to %d (totalDestPEs excluding root=%d)  numSend=%d\n",
	     myIndex+1, sendLogicalIndexStart, sendLogicalIndexEnd, totalDestPEs, numSend);
#endif

  npes = numSend;
  pelist = new int[npes];
  
  for(int i=0;i<numSend;i++){
    CkAssert(sendLogicalIndexStart-1+i < totalDestPEs);
    pelist[i] = multMsg->indicesCount[sendLogicalIndexStart-1+i].pe;
#if DEBUG
    CkPrintf("Tree logical index %d sending to PE %d\n", myIndex+1, pelist[i]);
#endif
    CkAssert(pelist[i] < CkNumPes());
  }
  
}



/*@}*/
