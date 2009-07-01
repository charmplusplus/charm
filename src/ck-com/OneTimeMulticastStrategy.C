/**
   @addtogroup ComlibCharmStrategy
   @{
   @file

*/


#include "OneTimeMulticastStrategy.h"
#include <string>
#include <set>

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
  double start = CmiWallTimer();
  CkSectionID *sec_id = cmsg->sec_id;
  CkVec< CkArrayIndexMax > localIndices;
  sinfo.getLocalIndices(sec_id->_nElems, sec_id->_elems, sec_id->_cookie.aid, localIndices);
  deliverToIndices(cmsg->getCharmMessage(), localIndices );
  traceUserBracketEvent(10000, start, CmiWallTimer());
}





/** 
    Forward multicast message to our successor processors in the spanning tree. 
    Uses CmiSyncListSendAndFree for delivery to this strategy's OneTimeMulticastStrategy::handleMessage method.
*/
void OneTimeMulticastStrategy::remoteMulticast(ComlibMulticastMsg * multMsg, bool rootPE) {
  double start = CmiWallTimer();

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
  
  determineNextHopPEs(totalDestPEs, multMsg->indicesCount, myIndex, pelist, npes );
  
  if(npes == 0) {
    traceUserBracketEvent(10001, start, CmiWallTimer());
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
  traceUserBracketEvent(10001, start, CmiWallTimer());

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




void OneTimeMulticastStrategy::determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, int * &pelist, int &npes) {
  if(myIndex==-1){
    // We are at a root node of the spanning tree. 
    // We will forward the message to all other PEs in the destination list.
    npes = totalDestPEs;
    
    pelist = new int[npes];
    for (int i=0; i<npes; ++i) {
      pelist[i] = destPEs[i].pe;
    }
  } else {
    // We are at a leaf node of the spanning tree. 
    npes = 0;
  }
  
}


void OneTimeRingMulticastStrategy::determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, int * &pelist, int &npes) {
  const int myPe = CkMyPe();

  if(myIndex == totalDestPEs-1){
    // Final PE won't send to anyone
    npes = 0;
    return;
  } else {
    // All non-final PEs will send to next PE in list
    npes = 1;
    pelist = new int[1];
    pelist[0] = destPEs[myIndex+1].pe;
  }

}


void OneTimeTreeMulticastStrategy::determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, int * &pelist, int &npes){
  const int myPe = CkMyPe();
  
  // The logical indices start at 0 = root node. Logical index i corresponds to the entry i+1 in the array of PEs in the message
  
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
    pelist[i] = destPEs[sendLogicalIndexStart-1+i].pe;
#if DEBUG
    CkPrintf("Tree logical index %d sending to PE %d\n", myIndex+1, pelist[i]);
#endif
    CkAssert(pelist[i] < CkNumPes());
  }
  
}




void OneTimeNodeTreeMulticastStrategy::determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, int * &pelist, int &npes){
  const int myPe = CkMyPe();

  std::set<int> nodePERepresentatives;
  
  // create a list of PEs, with one for each node to which the message must be sent
  for(int i=0; i<totalDestPEs; i++){
    int pe = destPEs[i].pe;
    int representative = CmiGetFirstPeOnPhysicalNode(pe);
    nodePERepresentatives.insert(representative);    
  }
  
  int numRepresentativePEs = nodePERepresentatives.size();

  CkPrintf("Multicasting to %d PEs on %d physical nodes\n", totalDestPEs, numRepresentativePEs );
  fflush(stdout);

  int repForMyPe = CmiGetFirstPeOnPhysicalNode(CkMyPe());
  
  if(CkMyPe() == repForMyPe){
    // This representative PE for the node should forward on this message along the tree, and deliver to local PEs
    
    
    // flatten the data structure for nodePERepresentatives
    int *repPeList = new int[numRepresentativePEs];
    int myRepIndex = -1;
    std::set<int>::iterator iter;
    int p=0;
    for(iter=nodePERepresentatives.begin(); iter != nodePERepresentatives.end(); iter++){
      repPeList[p] = *iter;
      if(*iter == repForMyPe)
	myRepIndex = p;
      p++;
    }
    CkAssert(myRepIndex >=0);
    
    
       
    // The logical indices start at 0 = root node. Logical index i corresponds to the entry i+1 in the array of PEs in the message
    int sendLogicalIndexStart = degree*(myRepIndex+1) + 1;       // inclusive
    int sendLogicalIndexEnd = sendLogicalIndexStart + degree - 1;   // inclusive
    
    if(sendLogicalIndexEnd-1 >= totalDestPEs){
      sendLogicalIndexEnd = totalDestPEs;
    }
    
    int numSendTree = sendLogicalIndexEnd - sendLogicalIndexStart + 1;
    int numSendLocal = CmiNumPesOnPhysicalNode(CkMyPe())-1;
    
    CkPrintf("[%d] numSendTree=%d numSendLocal=%d\n", CkMyPe(), numSendTree, numSendLocal);
    fflush(stdout);

    int numSend = numSendTree + numSendLocal;
    if(numSend <= 0){
      npes = 0;
      return;
    }
    
    npes = numSend;
    pelist = new int[npes];
  
    for(int i=0;i<numSendTree;i++){
      CkAssert(sendLogicalIndexStart-1+i < totalDestPEs);
      pelist[i] = repPeList[sendLogicalIndexStart-1+i];
      CkAssert(pelist[i] < CkNumPes());
    }
    
    int num;
    int *pelist;
    CmiGetPesOnPhysicalNode(CkMyPe(), &pelist, &num);
    for(int i=0;i<numSendLocal;i++){
      pelist[i+numSendTree] = pelist[1+i];
    }

    
    char buf[1024];
    sprintf(buf, "PE %d is sending to PEs: ", CkMyPe() );
    
    for(int i=0;i<numSend;i++){
      sprintf(buf+strlen(buf), "%d ", pelist[i]);
    }
    
    CkPrintf("%s\n", buf);
        
  } else {
    // We are a leaf PE
    npes = 0;
    return;
  }

  
  
}



/*@}*/
