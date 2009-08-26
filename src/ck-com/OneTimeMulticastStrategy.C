/**
   @addtogroup ComlibCharmStrategy
   @{
   @file

*/


#include "OneTimeMulticastStrategy.h"
#include <string>
#include <set>
#include <vector>
#include <list>

//#define DEBUG 1

using std::list;
using std::set;
using std::vector;


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
#if DEBUG
  CkPrintf("[%d] OneTimeMulticastStrategy::insertMessage\n", CkMyPe());
  fflush(stdout);
#endif 

  if(cmsg->dest_proc != IS_SECTION_MULTICAST && cmsg->sec_id == NULL) { 
    CkAbort("OneTimeMulticastStrategy can only be used with an array section proxy");
  }
    

  envelope *env = UsrToEnv(cmsg->getCharmMessage());
  int npes = 1;
  int pes[1] = {0};
  _TRACE_CREATION_MULTICAST(env, npes, pes);

#if DEBUG
  CkPrintf("[%d] after TRACE_CREATION_MULTICAST menv->event=%d\n", CkMyPe(), (int)env->getEvent());  
#endif
  
  // Create a multicast message containing all information about remote destination objects 
  ComlibMulticastMsg * multMsg = sinfo.getNewMulticastMessage(cmsg, 0, getInstance());


#if DEBUG
    CkPrintf("[%d] after TRACE_CREATION_MULTICAST multMsg->event=%d\n", CkMyPe(), (int)( UsrToEnv(multMsg)->getEvent() ) );  
#endif

  // The remote multicast method will send the message to the remote PEs, as specified in multMsg
  remoteMulticast(multMsg, true);

  // local multicast will re-extract a list of local destination objects (FIXME to make this more efficient)
  localMulticast(cmsg);

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
    Uses CmiSyncListSend for delivery to this strategy's OneTimeMulticastStrategy::handleMessage method.
*/
void OneTimeMulticastStrategy::remoteMulticast(ComlibMulticastMsg * multMsg, bool rootPE) {
  double start = CmiWallTimer();

  envelope *env = UsrToEnv(multMsg);
    
  
  /// The index into the PE list in the message
  int myIndex = -10000; 
  const int totalDestPEs = multMsg->nPes;
  const int myPe = CkMyPe();
  
  // Find my index in the list of all destination PEs
  if(rootPE){
    CkAssert(CkMyPe() == multMsg->_cookie.pe);
    myIndex = -1;
  } else {
    for (int i=0; i<totalDestPEs; ++i) {
      if(multMsg->indicesCount[i].pe == myPe){
	myIndex = i;
	break;
      }
    }
  }
  
  if(myIndex == -10000)
    CkAbort("My PE was not found in the list of destination PEs in the ComlibMulticastMsg");
  
  int npes;
  int *pelist = NULL;

  if(totalDestPEs > 0)
    determineNextHopPEs(totalDestPEs, multMsg->indicesCount, myIndex, multMsg->_cookie.pe, pelist, npes );
  else {
    npes = 0;
  }

  if(npes == 0) {
#if DEBUG
    CkPrintf("[%d] OneTimeMulticastStrategy::remoteMulticast is not forwarding to any other PEs\n", CkMyPe());
#endif
    traceUserBracketEvent(10001, start, CmiWallTimer());
    CmiFree(env);
    return;
  }
  
  //Collect Multicast Statistics
  RECORD_SENDM_STATS(getInstance(), env->getTotalsize(), pelist, npes);
  

  CmiSetHandler(env, CkpvAccess(comlib_handler));
  ((CmiMsgHeaderBasic *) env)->stratid = getInstance();  
  CkPackMessage(&env);

  double middle = CmiWallTimer();

  
  // CkPrintf("[%d] before CmiSyncListSendAndFree env->event=%d\n", CkMyPe(), (int)env->getEvent());


  CkAssert(npes > 0);
  CmiSyncListSend(npes, pelist, env->getTotalsize(), (char*)env);
  
  delete[] pelist;

  double end = CmiWallTimer();
  traceUserBracketEvent(10001, start, middle);
  traceUserBracketEvent(10002, middle, end);
  
}



/** 
    Receive an incoming multicast message(sent from OneTimeMulticastStrategy::remoteMulticast).
    Deliver the message to all local elements 
*/
void OneTimeMulticastStrategy::handleMessage(void *msg){
#if DEBUG
  //  CkPrintf("[%d] OneTimeMulticastStrategy::handleMessage\n", CkMyPe());
#endif
  envelope *env = (envelope *)msg;
  CkUnpackMessage(&env);
  ComlibMulticastMsg* multMsg = (ComlibMulticastMsg*)EnvToUsr(env);
  
  // Don't use msg after this point. Instead use the unpacked env
  
  RECORD_RECV_STATS(getInstance(), env->getTotalsize(), env->getSrcPe()); // DOESN'T DO ANYTHING IN NEW COMLIB
  
  // Deliver to objects marked as local in the message
  int localElems;
  envelope *newenv;
  CkArrayIndexMax *local_idx_list;  
  sinfo.unpack(env, localElems, local_idx_list, newenv);
  ComlibMulticastMsg *newmsg = (ComlibMulticastMsg *)EnvToUsr(newenv);  

  //  CkPrintf("[%d] in OneTimeMulticastStrategy::handleMessage before  deliverToIndices newenv->event=%d\n", CkMyPe(), (int)newenv->getEvent());

  // Forward on to other processors if necessary
  remoteMulticast(multMsg, false);

  // Deliver locally
  deliverToIndices(newmsg, localElems, local_idx_list );
  
  // Finally delete the reference counted message because remoteMulticast does not do this.
  CmiFree(multMsg);
  
}




void OneTimeMulticastStrategy::determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, const int rootPE, int * &pelist, int &npes) {
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


void OneTimeRingMulticastStrategy::determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, const int rootPE, int * &pelist, int &npes) {
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


void OneTimeTreeMulticastStrategy::determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, const int rootPE, int * &pelist, int &npes){
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


/** Find a unique representative PE for a node containing pe, with the restriction that the returned PE is in the list destPEs. */
int getFirstPeOnPhysicalNodeFromList(int pe, const int totalDestPEs, const ComlibMulticastIndexCount* destPEs){
  int num;
  int *nodePeList;
  CmiGetPesOnPhysicalNode(pe, &nodePeList, &num);
  
  for(int i=0;i<num;i++){
    // Scan destPEs for the pe
    int p = nodePeList[i];
    
    for(int j=0;j<totalDestPEs;j++){
      if(p == destPEs[j].pe){
	// found the representative PE for the node that is in the destPEs list
	return p;
      }
    }
  }
  
  CkAbort("ERROR: Could not find an entry for pe in destPEs list.\n");
  return -1;
}


/** Find a unique representative PE for a node containing pe, with the restriction that the returned PE is in the list destPEs. */
int getNthPeOnPhysicalNodeFromList(int n, int pe, const int totalDestPEs, const ComlibMulticastIndexCount* destPEs){
  int num;
  int *nodePeList;
  CmiGetPesOnPhysicalNode(pe, &nodePeList, &num);
  
  int count = 0;
  int lastFound = -1;
  
  // Foreach PE on this physical node
  for(int i=0;i<num;i++){
    int p = nodePeList[i];
    
    // Scan destPEs for the pe
    for(int j=0;j<totalDestPEs;j++){
      if(p == destPEs[j].pe){
	lastFound = p;
	if(count==n)
	  return p;
	count++;
      }
    }
  }
  
  if(lastFound != -1)
    return lastFound;

  CkAbort("ERROR: Could not find an entry for pe in destPEs list.\n");
  return -1;
}


/** List all the PEs from the list that share the physical node */ 
vector<int> getPesOnPhysicalNodeFromList(int pe, const int totalDestPEs, const ComlibMulticastIndexCount* destPEs){ 
   
  vector<int> result; 
 
  int num; 
  int *nodePeList; 
  CmiGetPesOnPhysicalNode(pe, &nodePeList, &num); 
  
  for(int i=0;i<num;i++){ 
    // Scan destPEs for the pe 
    int p = nodePeList[i]; 
    for(int j=0;j<totalDestPEs;j++){ 
      if(p == destPEs[j].pe){ 
	// found the representative PE for the node that is in the
	// destPEs list 
	result.push_back(p); 
	break; 
      } 
    } 
  } 
  
  return result; 
}



/** List all the other PEs from the list that share the physical node */
vector<int> getOtherPesOnPhysicalNodeFromList(int pe, const int totalDestPEs, const ComlibMulticastIndexCount* destPEs){
  
  vector<int> result;

  int num;
  int *nodePeList;
  CmiGetPesOnPhysicalNode(pe, &nodePeList, &num);
  
  for(int i=0;i<num;i++){
    // Scan destPEs for the pe
    int p = nodePeList[i];
    if(p != pe){
      for(int j=0;j<totalDestPEs;j++){
	if(p == destPEs[j].pe){
	  // found the representative PE for the node that is in the destPEs list
	  result.push_back(p);
	  break;
	}
      }
    }
  }
  
  return result;
}


void OneTimeNodeTreeMulticastStrategy::determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, const int rootPE, int * &pelist, int &npes){
  const int myPe = CkMyPe();

  set<int> nodePERepresentatives;
  
  // create a list of PEs, with one for each node to which the message must be sent
  for(int i=0; i<totalDestPEs; i++){
    int pe = destPEs[i].pe;
    int representative = getFirstPeOnPhysicalNodeFromList(pe, totalDestPEs, destPEs);
    nodePERepresentatives.insert(representative);    
  }
  
  // Create an ordered list of PEs to send to, based upon the rootPE
  // This should distribute load more evenly than the sorted list previously used
  set<int>::iterator splitter = nodePERepresentatives.upper_bound(rootPE);
  vector<int> nodePERepresentativesOrdered;
  for(set<int>::iterator iter = splitter; iter!=nodePERepresentatives.end(); ++iter){
    nodePERepresentativesOrdered.push_back(*iter);
  }
  for(set<int>::iterator iter = nodePERepresentatives.begin(); iter!=splitter; ++iter){
    nodePERepresentativesOrdered.push_back(*iter);
  }

  CkAssert(nodePERepresentativesOrdered.size() == nodePERepresentatives.size());
    
  int numRepresentativePEs = nodePERepresentatives.size();
  
  int repForMyPe=-1;
  if(myIndex != -1)
    repForMyPe = getFirstPeOnPhysicalNodeFromList(CkMyPe(), totalDestPEs, destPEs);
  
#if DEBUG
  CkPrintf("[%d] Multicasting to %d PEs on %d physical nodes  repForMyPe=%d\n", CkMyPe(), totalDestPEs, numRepresentativePEs, repForMyPe);
  fflush(stdout);
#endif
  
  // If this PE is part of the multicast tree, then it should forward the message along
  if(CkMyPe() == repForMyPe || myIndex == -1){
    // I am an internal node in the multicast tree
    
    // flatten the nodePERepresentativesOrdered data structure
    int *repPeList = new int[numRepresentativePEs];
    int myRepIndex = -1;
    int p=0;
    for(vector<int>::iterator iter=nodePERepresentativesOrdered.begin(); iter != nodePERepresentativesOrdered.end(); iter++){
      repPeList[p] = *iter;
      if(*iter == repForMyPe)
	myRepIndex = p;
      p++;
    }
    CkAssert(myRepIndex >=0 || myIndex==-1);
      
    // The logical indices start at 0 = root node. Logical index i corresponds to the entry i+1 in the array of PEs in the message
    int sendLogicalIndexStart = degree*(myRepIndex+1) + 1;       // inclusive
    int sendLogicalIndexEnd = sendLogicalIndexStart + degree - 1;   // inclusive
    
    if(sendLogicalIndexEnd-1 >= numRepresentativePEs){
      sendLogicalIndexEnd = numRepresentativePEs;
    }
    
    int numSendTree = sendLogicalIndexEnd - sendLogicalIndexStart + 1;
    if(numSendTree < 0)
      numSendTree = 0;
    
    vector<int> otherLocalPes = getOtherPesOnPhysicalNodeFromList(CkMyPe(), totalDestPEs, destPEs);
    int numSendLocal;
    if(myIndex == -1)
      numSendLocal = 0;
    else 
      numSendLocal = otherLocalPes.size();
    
    

#if DEBUG
    CkPrintf("[%d] numSendTree=%d numSendLocal=%d sendLogicalIndexStart=%d sendLogicalIndexEnd=%d\n", CkMyPe(), numSendTree, numSendLocal,  sendLogicalIndexStart, sendLogicalIndexEnd);
    fflush(stdout);
#endif

    int numSend = numSendTree + numSendLocal;
    if(numSend <= 0){
      npes = 0;
      return;
    }
    
    npes = numSend;
    pelist = new int[npes];
  
    for(int i=0;i<numSendTree;i++){
      CkAssert(sendLogicalIndexStart-1+i < numRepresentativePEs);
      pelist[i] = repPeList[sendLogicalIndexStart-1+i];
      CkAssert(pelist[i] < CkNumPes() && pelist[i] >= 0);
    }
    
    delete[] repPeList;
    repPeList = NULL;

    for(int i=0;i<numSendLocal;i++){
      pelist[i+numSendTree] = otherLocalPes[i];
      CkAssert(pelist[i] < CkNumPes() && pelist[i] >= 0);
    }
    
    
#if DEBUG
    char buf[1024];
    sprintf(buf, "PE %d is sending to Remote Node PEs: ", CkMyPe() );
    for(int i=0;i<numSend;i++){
      if(i==numSendTree)
	sprintf(buf+strlen(buf), " and Local To Node PEs: ", pelist[i]);

      sprintf(buf+strlen(buf), "%d ", pelist[i]);
    }    
    CkPrintf("%s\n", buf);
    fflush(stdout);
#endif
        
  } else {
    // We are a leaf PE
    npes = 0;
    return;
  }

  
  
}


void OneTimeNodeTreeRingMulticastStrategy::determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, const int rootPE, int * &pelist, int &npes){
  const int myPe = CkMyPe();

  set<int> nodePERepresentatives;
  
  // create a list of PEs, with one for each node to which the message must be sent
  for(int i=0; i<totalDestPEs; i++){
    int pe = destPEs[i].pe;
    int representative = getFirstPeOnPhysicalNodeFromList(pe, totalDestPEs, destPEs);
    nodePERepresentatives.insert(representative);    
  }

   // Create an ordered list of PEs to send to, based upon the rootPE
  // This should distribute load more evenly than the sorted list previously used
  set<int>::iterator splitter = nodePERepresentatives.upper_bound(rootPE);
  vector<int> nodePERepresentativesOrdered;
  for(set<int>::iterator iter = splitter; iter!=nodePERepresentatives.end(); ++iter){
    nodePERepresentativesOrdered.push_back(*iter);
  }
  for(set<int>::iterator iter = nodePERepresentatives.begin(); iter!=splitter; ++iter){
    nodePERepresentativesOrdered.push_back(*iter);
  }

  CkAssert(nodePERepresentativesOrdered.size() == nodePERepresentatives.size());
  int numRepresentativePEs = nodePERepresentatives.size();
  
  int repForMyPe=-1;
  if(myIndex != -1)
    repForMyPe = getFirstPeOnPhysicalNodeFromList(CkMyPe(), totalDestPEs, destPEs);
  
#if DEBUG
  CkPrintf("[%d] Multicasting to %d PEs on %d physical nodes  repForMyPe=%d\n", CkMyPe(), totalDestPEs, numRepresentativePEs, repForMyPe);
  fflush(stdout);
#endif
  
  // If this PE is part of the multicast tree, then it should forward the message along
  if(CkMyPe() == repForMyPe || myIndex == -1){
    // I am an internal node in the multicast tree
    
    // flatten the data structure for nodePERepresentatives
    int *repPeList = new int[numRepresentativePEs];
    int myRepIndex = -1;
    int p=0;
    for(vector<int>::iterator iter=nodePERepresentativesOrdered.begin(); iter != nodePERepresentativesOrdered.end(); iter++){
      repPeList[p] = *iter;
      if(*iter == repForMyPe)
	myRepIndex = p;
      p++;
    }
    CkAssert(myRepIndex >=0 || myIndex==-1);
      
    // The logical indices start at 0 = root node. Logical index i corresponds to the entry i+1 in the array of PEs in the message
    int sendLogicalIndexStart = degree*(myRepIndex+1) + 1;       // inclusive
    int sendLogicalIndexEnd = sendLogicalIndexStart + degree - 1;   // inclusive
    
    if(sendLogicalIndexEnd-1 >= numRepresentativePEs){
      sendLogicalIndexEnd = numRepresentativePEs;
    }
    
    int numSendTree = sendLogicalIndexEnd - sendLogicalIndexStart + 1;
    if(numSendTree < 0)
      numSendTree = 0;


    // Send in a ring to the PEs on this node
    vector<int> otherLocalPes = getOtherPesOnPhysicalNodeFromList(CkMyPe(), totalDestPEs, destPEs);
    int numSendLocal = 0;
    if(myIndex == -1)
      numSendLocal = 0;
    else {
      if(otherLocalPes.size() > 0)
	numSendLocal = 1;
      else
	numSendLocal = 0;
    }
    

#if DEBUG
    CkPrintf("[%d] numSendTree=%d numSendLocal=%d sendLogicalIndexStart=%d sendLogicalIndexEnd=%d\n", CkMyPe(), numSendTree, numSendLocal,  sendLogicalIndexStart, sendLogicalIndexEnd);
    fflush(stdout);
#endif

    int numSend = numSendTree + numSendLocal;
    if(numSend <= 0){
      npes = 0;
      return;
    }
    
    npes = numSend;
    pelist = new int[npes];
  
    for(int i=0;i<numSendTree;i++){
      CkAssert(sendLogicalIndexStart-1+i < numRepresentativePEs);
      pelist[i] = repPeList[sendLogicalIndexStart-1+i];
      CkAssert(pelist[i] < CkNumPes() && pelist[i] >= 0);
    }
    
    delete[] repPeList;
    repPeList = NULL;

    for(int i=0;i<numSendLocal;i++){
      pelist[i+numSendTree] = otherLocalPes[i];
      CkAssert(pelist[i] < CkNumPes() && pelist[i] >= 0);
    }
    
    
#if DEBUG
    char buf[1024];
    sprintf(buf, "PE %d is sending to Remote Node PEs: ", CkMyPe() );
    for(int i=0;i<numSend;i++){
      if(i==numSendTree)
	sprintf(buf+strlen(buf), " and Local To Node PEs: ", pelist[i]);

      sprintf(buf+strlen(buf), "%d ", pelist[i]);
    }    
    CkPrintf("%s\n", buf);
    fflush(stdout);
#endif
        
  } else {
    // We are a leaf PE, so forward in a ring to the PEs on this node
    const vector<int> otherLocalPes = getPesOnPhysicalNodeFromList(CkMyPe(), totalDestPEs, destPEs);
    
    npes = 0;
    pelist = new int[1];
    
    //    CkPrintf("[%d] otherLocalPes.size=%d\n", CkMyPe(), otherLocalPes.size() ); 
    const int numOthers = otherLocalPes.size() ;
    
    for(int i=0;i<numOthers;i++){
      if(otherLocalPes[i] == CkMyPe()){
	// found me in the PE list for this node
	if(i+1<otherLocalPes.size()){
	  // If we have a successor in the ring
	  pelist[0] = otherLocalPes[i+1];
	  npes = 1;
	}
      }
    }
    
    
#if DEBUG
    if(npes==0)
      CkPrintf("[%d] At end of ring\n", CkMyPe() );
    else
      CkPrintf("[%d] sending along ring to %d\n", CkMyPe(), pelist[0] );
    
    fflush(stdout);
#endif
    
    
  }
  
  
  
}








void OneTimeDimensionOrderedMulticastStrategy::determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, const int rootPE, int * &pelist, int &npes){
 //  const int myPe = CkMyPe();

//   set<int> nodePERepresentatives;
  
//   // create a list of PEs, with one for each node to which the message must be sent
//   for(int i=0; i<totalDestPEs; i++){
//     int pe = destPEs[i].pe;
//     int representative = getFirstPeOnPhysicalNodeFromList(pe, totalDestPEs, destPEs);
//     nodePERepresentatives.insert(representative);    
//   }
  
//   int numRepresentativePEs = nodePERepresentatives.size();
  
//   int repForMyPe=-1;
//   if(myIndex != -1)
//     repForMyPe = getFirstPeOnPhysicalNodeFromList(CkMyPe(), totalDestPEs, destPEs);
  
// #if DEBUG
//   CkPrintf("[%d] Multicasting to %d PEs on %d physical nodes  repForMyPe=%d\n", CkMyPe(), totalDestPEs, numRepresentativePEs, repForMyPe);
//   fflush(stdout);
// #endif
  
//   // If this PE is part of the multicast tree, then it should forward the message along
//   if(CkMyPe() == repForMyPe || myIndex == -1){
//     // I am an internal node in the multicast tree

//     TopoManager tmgr;
    
//     // flatten the data structure for nodePERepresentatives
//     int *repPeList = new int[numRepresentativePEs];
//     bool *inTreeYet = new bool[numRepresentativePEs];
//     int *repPeTopoX = new int[numRepresentativePEs];
//     int *repPeTopoY = new int[numRepresentativePEs];
//     int *repPeTopoZ = new int[numRepresentativePEs];

//     set<int> xcoords;

//     int myX, myY, myZ, myT;
//     int rootX, rootY, rootZ, rootT;
//     tmgr->rankToCoordinates(CkMyPe(), myX, myY, myZ, myT);        
//     tmgr->rankToCoordinates(CkMyPe(), rootX, rootY, rootZ, rootT);        

//     int myRepIndex = -1;
//     set<int>::iterator iter;
//     int p=0;
//     for(iter=nodePERepresentatives.begin(); iter != nodePERepresentatives.end(); iter++){
//       int pe =  *iter;
//       repPeList[p] = pe;
//       inTreeYet[p] = false;
//       int t; // ignore the 't' dimension (which PE on the node)
//       tmgr->rankToCoordinates(pe, repPeTopoX[p], repPeTopoY[p], repPeTopoZ[p], t);     
//       xcoords.insert(repPeTopoX[p]);
//       if(*iter == repForMyPe)
// 	myRepIndex = p;
//       p++;
//     }
//     CkAssert(myRepIndex >=0 || myIndex==-1);
    

//     // Holds the spanning tree as we build it. It maps from index in the repPeList to sets of indices for repPeList
//     map<int, set<int> > spanningTreeChildren;

//     //---------------------------------------------------------------------------------------------
//     // Determine the children of the root of the multicast tree
 
//     int nearestXp = 100000;
//     int nearestXpidx = -1;
//     int nearestXm = -100000;
//     int nearestXmidx = -1;

//     for(int i=0; i<numRepresentativePEs; i++){
//       // Find the nearest +x neighbor
//       if(repPeTopoX[i] > rootX && repPeTopoX[i] < nearestXp){
// 	nearestXp = repPeTopoX[i] ;
// 	nearestXpidx = i;
//       }
      
//       // Find the nearest -x neighbor
//       if(repPeTopoX[i] < rootX && repPeTopoX[i] > nearestXm){
// 	nearestXm = repPeTopoX[i] ;
// 	nearestXmidx = i;
//       }    
      
//     //   // send to some destinations with same X coordinate (just one per Y coordinate)
// //       if(repPeTopoX == rootX){
// // 	set<int>::iterator iter;
// // 	bool found = false;
// // 	for(iter = spanningTreeChildren[-1].begin(); iter != spanningTreeChildren[-1].end(); ++iter){
// // 	  iterYcoord = repPeTopoY[*iter];
// // 	  if(iterYcoord == repPeTopoY[i]){
// // 	    found = true;
// // 	    break;
// // 	  } 
// // 	}
// // 	if(! found) {
// // 	  // haven't seen this Y coordinate yet
// // 	  spanningTreeChildren[-1].insert(i);
// // 	  inTreeYet[i] = true;
// // 	}
// //       }
//     }
    
    
//     // The root sends to nearest +x neighbor
//     if(nearestXpidx != -1){
//       spanningTreeChildren[-1].insert(nearestXpidx);
//       inTreeYet[nearestXpidx] = true;
//     }
    
    
//     // The root sends to nearest -x neighbor
//     if(nearestXpidx != -1){
//       spanningTreeChildren[-1].insert(nearestXpidx);
//       inTreeYet[nearestXpidx] = true;
//     }
    
    
//     //---------------------------------------------------------------------------------------------
//     // Make sure we span all X coordinates 
    
    
    
    





    
    
   
//     int numSendTree = 1;
//     if(numSendTree < 0)
//       numSendTree = 0;
    
//     vector<int> otherLocalPes = getOtherPesOnPhysicalNodeFromList(CkMyPe(), totalDestPEs, destPEs);
//     int numSendLocal;
//     if(myIndex == -1)
//       numSendLocal = 0;
//     else 
//       numSendLocal = otherLocalPes.size();
    
//     int numSend = numSendTree + numSendLocal;
//     if(numSend <= 0){
//       npes = 0;
//       return;
//     }
    
//     npes = numSend;
//     pelist = new int[npes];
  
//     for(int i=0;i<numSendTree;i++){
//       CkAssert(sendLogicalIndexStart-1+i < numRepresentativePEs);
//       pelist[i] = repPeList[sendLogicalIndexStart-1+i];
//       CkAssert(pelist[i] < CkNumPes() && pelist[i] >= 0);
//     }
    
//     delete[] repPeList;
//     delete[] repPeTopoX;
//     delete[] repPeTopoY;
//     delete[] repPeTopoZ;
//     repPeList = NULL;

//     for(int i=0;i<numSendLocal;i++){
//       pelist[i+numSendTree] = otherLocalPes[i];
//       CkAssert(pelist[i] < CkNumPes() && pelist[i] >= 0);
//     }
    
// #if 1
//     char buf[1024];
//     sprintf(buf, "PE %d is sending to Remote Node PEs: ", CkMyPe() );
//     for(int i=0;i<numSend;i++){
//       if(i==numSendTree)
// 	sprintf(buf+strlen(buf), " and Local To Node PEs: ", pelist[i]);

//       sprintf(buf+strlen(buf), "%d ", pelist[i]);
//     }    
//     CkPrintf("%s\n", buf);
//     fflush(stdout);
// #endif
        
//   } else {
//     // We are a leaf PE
//     npes = 0;
//     return;
//   }

  
  
}














/*@}*/
