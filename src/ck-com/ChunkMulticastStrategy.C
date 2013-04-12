/**
   @addtogroup ComlibCharmStrategy
   @{
   @file

*/


#include "ChunkMulticastStrategy.h"
#include <string>
#include <set>
#include <vector>
#include "queueing.h"
#include "ck.h"
#include "spanningTreeStrategy.h"

#define DEBUG 0
#define CHUNK_LL 2048	//minimum chunk size

CkpvExtern(CkGroupID, cmgrID);

ChunkMulticastStrategy::ChunkMulticastStrategy()
  : Strategy(), CharmStrategy() {
  //  ComlibPrintf("ChunkMulticastStrategy constructor\n");
  setType(ARRAY_STRATEGY);
  //numChunks = 0;
  //nrecv = 0;
  sentCount = 0;
}

ChunkMulticastStrategy::~ChunkMulticastStrategy() {
}

void ChunkMulticastStrategy::pup(PUP::er &p){
  Strategy::pup(p);
  CharmStrategy::pup(p);
}


/** Called when the user invokes the entry method on the delegated proxy. */
void ChunkMulticastStrategy::insertMessage(CharmMessageHolder *cmsg){
#if DEBUG
  CkPrintf("[%d] ChunkMulticastStrategy::insertMessage\n", CkMyPe());
  fflush(stdout);
#endif 

  if(cmsg->dest_proc != IS_SECTION_MULTICAST && cmsg->sec_id == NULL) { 
    CkAbort("ChunkMulticastStrategy can only be used with an array section proxy");
  }
    

  envelope *env = UsrToEnv(cmsg->getCharmMessage());
  int npes = 1;
  int pes[1] = {0};



  //THIS IS A TEMPORARY HACK, WILL WORK ONLY FOR RING
  const CkArrayID destArrayID(env->getArrayMgr());
  int nRemotePes=-1, nRemoteIndices=-1;
  ComlibMulticastIndexCount *indicesCount;
  int *belongingList;
  sinfo.getPeCount(cmsg->sec_id->_nElems, cmsg->sec_id->_elems, destArrayID, nRemotePes, nRemoteIndices, indicesCount, belongingList);
//  int numChunks = nRemotePes/2;
  
  delete [] belongingList;
  delete [] indicesCount;

#if DEBUG
  CkPrintf("[%d] after TRACE_CREATION_MULTICAST menv->event=%d\n", CkMyPe(), (int)env->getEvent());  
#endif

  //message needs to be unpacked to correctly access envelope information
  CkUnpackMessage(&env);
  //char* msg = EnvToUsr(env);
  int totalSize = cmsg->getSize() -sizeof(envelope) - env->getPrioBytes(); //totalsize = envelope size + usermsg size + priobits
  int numChunks;
  if (totalSize/CHUNK_LL < nRemotePes) numChunks = totalSize/CHUNK_LL;
  else numChunks = nRemotePes;
  if (numChunks == 0) numChunks = 1;
  int chunkSize, sendingSize;
  char **sendingMsgArr = new char*[numChunks];
  char *sendingMsg;
  envelope *envchunk;
  CharmMessageHolder *holder;
  ComlibMulticastMsg *commsg;
  ChunkInfo *info;

  //increment send counter;
  sentCount++;

  //set up message chunks and define chunk information
  for(int i = 0; i < numChunks; i++){
    chunkSize = totalSize / numChunks;
    if (i < totalSize % numChunks)
      chunkSize++;
    chunkSize = CkMsgAlignLength(chunkSize);
    sendingSize = chunkSize+CkMsgAlignLength(sizeof(ChunkInfo));
    sendingMsg = (char*)CkAllocBuffer(EnvToUsr(env), sendingSize);
    info = (ChunkInfo*)(sendingMsg);
    info->srcPe = CkMyPe();
    info->chunkNumber = i;
    info->numChunks = numChunks;
    info->chunkSize = chunkSize;
    //info->messageSize = sendingSize;
    info->idx = sentCount;
    sendingMsgArr[i] = sendingMsg;
  }
  
  //pack message before copying data for correct varsize packing
  CkPackMessage(&env);
  char *nextChunk = (char*)EnvToUsr(env);
  for(int i = 0; i < numChunks; i++){
    sendingMsg = sendingMsgArr[i];
    info = (ChunkInfo*)(sendingMsg);
    CmiMemcpy(sendingMsg+CkMsgAlignLength(sizeof(ChunkInfo)), nextChunk, info->chunkSize);
    envchunk = UsrToEnv(sendingMsg);
    
    nextChunk += info->chunkSize;

    CkPackMessage(&envchunk);
    envchunk->setPacked(1);
   
    holder = cmsg;
    holder->data = (char*)envchunk;
    holder->size = envchunk->getTotalsize();
    // Create a multicast message containing all information about remote destination objects 
    _TRACE_CREATION_MULTICAST(envchunk, npes, pes);
    
    commsg = sinfo.getNewMulticastMessage(holder, 0, getInstance()); 

    envchunk = UsrToEnv(commsg);

    // The remote multicast method will send the message to the remote PEs, as specified in multMsg
    remoteMulticast(commsg, true, i, numChunks);
  //  delete holder->data;
  }


#if DEBUG
//    CkPrintf("[%d] after TRACE_CREATION_MULTICAST multMsg->event=%d\n", CkMyPe(), (int)( UsrToEnv(commsg)->getEvent() ) );  
#endif

  // local multicast will re-extract a list of local destination objects (FIXME to make this more efficient)
  cmsg->data =  (char*)env;
  cmsg->size = env->getTotalsize();
  localMulticast(cmsg);

  for (int i = 0; i < numChunks; i++){
    CmiFree(UsrToEnv(sendingMsgArr[i]));
  }
  delete [] sendingMsgArr;
  delete cmsg;    
}



/** Deliver the message to the local elements. */
void ChunkMulticastStrategy::localMulticast(CharmMessageHolder *cmsg) {
  double start = CmiWallTimer();
  CkSectionID *sec_id = cmsg->sec_id;
  CkVec< CkArrayIndex > localIndices;
  CkArrayID aid(sec_id->_cookie.get_aid());
  sinfo.getLocalIndices(sec_id->_nElems, sec_id->_elems, aid, localIndices);
  deliverToIndices(cmsg->getCharmMessage(), localIndices.size(), localIndices.getVec() );
  //if (deliverToIndices(cmsg->getCharmMessage(), localIndices.size(), localIndices.getVec() ) == 0) 
    //CkFreeMsg(cmsg->getCharmMessage());
  traceUserBracketEvent(10000, start, CmiWallTimer());
}





/** 
    Forward multicast message to our successor processors in the spanning tree. 
    Uses CmiSyncListSendAndFree for delivery to this strategy's ChunkMulticastStrategy::handleMessage method.
*/
void ChunkMulticastStrategy::remoteMulticast(ComlibMulticastMsg * multMsg, bool rootPE, int chunkNumber, int numChunks) {
  double start = CmiWallTimer();

  envelope *env = UsrToEnv(multMsg);
    
  
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
  
  if(myIndex == -10000)
    CkAbort("My PE was not found in the list of destination PEs in the ComlibMulticastMsg");
  
  int npes;
  int *pelist = NULL;
  //CkPrintf("totalDestPEs = %d\n",totalDestPEs);
  if(totalDestPEs > 0)
    determineNextHopPEs(totalDestPEs, multMsg->indicesCount, myIndex, pelist, npes, chunkNumber, numChunks );
  else {
    npes = 0;
  }

  if(npes == 0) {
#if DEBUG
    CkPrintf("[%d] ChunkMulticastStrategy::remoteMulticast is not forwarding to any other PEs\n", CkMyPe());
#endif
    traceUserBracketEvent(10001, start, CmiWallTimer());
    CmiFree(env);
    return;
  }
  
  //Collect Multicast Statistics
  RECORD_SENDM_STATS(getInstance(), env->getTotalsize(), pelist, npes);
  

  CmiSetHandler(env, CkpvAccess(comlib_handler));
  ((CmiMsgHeaderExt *) env)->stratid = getInstance();  
  CkPackMessage(&env);
  double middle = CmiWallTimer();

  
  // CkPrintf("[%d] before CmiSyncListSendAndFree env->event=%d\n", CkMyPe(), (int)env->getEvent());

#if DEBUG
  CkPrintf("[%d] remoteMulticast Sending to %d PEs: numChunks = %d\n", CkMyPe(), npes, numChunks);
  for(int i=0;i<npes;i++){
    CkPrintf("[%d]    %d\n", CkMyPe(), pelist[i]);
  } 
#endif

  CkAssert(npes > 0);
  CmiSyncListSendAndFree(npes, pelist, env->getTotalsize(), (char*)env);
  
  delete[] pelist;

  double end = CmiWallTimer();
  traceUserBracketEvent(10001, start, middle);
  traceUserBracketEvent(10002, middle, end);
  
}

/** 
    Receive an incoming multicast message(sent from ChunkMulticastStrategy::remoteMulticast).
    Deliver the message to all local elements 
*/
void ChunkMulticastStrategy::handleMessage(void *msg){
#if DEBUG
  //  CkPrintf("[%d] ChunkMulticastStrategy::handleMessage\n", CkMyPe());
#endif
  envelope *env = (envelope *)msg;
  // CkPrintf("[%d] in ChunkMulticastStrategy::handleMessage env->event=%d\n", CkMyPe(), (int)env->getEvent());
  
  CkUnpackMessage(&env);
  
  // CkPrintf("[%d] in ChunkMulticastStrategy::handleMessage after CkUnpackMessage env->event=%d\n", CkMyPe(), (int)env->getEvent());
  


  ComlibMulticastMsg* multMsg = (ComlibMulticastMsg*)EnvToUsr(env);
  
  // Don't use msg after this point. Instead use the unpacked env
  
  RECORD_RECV_STATS(getInstance(), env->getTotalsize(), env->getSrcPe()); // DOESN'T DO ANYTHING IN NEW COMLIB
  
  // Deliver to objects marked as local in the message
  int localElems;
  envelope *newenv;
  CkArrayIndex *local_idx_list;  
  sinfo.unpack(env, localElems, local_idx_list, newenv);
  ComlibMulticastMsg *newmsg = (ComlibMulticastMsg *)EnvToUsr(newenv);  

  ChunkInfo *inf = (ChunkInfo*)newmsg;
  std::list< recvBuffer* >::iterator iter;
  recvBuffer* buf;
  int nrecv = -1;
  int cnumber = inf->chunkNumber;
  int numChunks;
  envelope** recvChunks;
  for (iter=recvList.begin(); iter != recvList.end(); iter++){
    buf = *iter;
    if (inf->srcPe == buf->srcPe && inf->idx == buf->idx){
      buf->nrecv++;
      nrecv = buf->nrecv;
      numChunks = buf->numChunks;
      recvChunks = buf->recvChunks;
      if (nrecv == numChunks){
	delete buf;
	recvList.erase(iter);
      }
      break;
    }
  }
  if ( nrecv == -1){
    numChunks = inf->numChunks;
    nrecv = 1;
    recvChunks = new envelope*[inf->numChunks];
    if (numChunks > 1){
      buf = new recvBuffer();
      buf->numChunks = inf->numChunks;
      buf->srcPe = inf->srcPe;
      buf->idx = inf->idx;
      buf->nrecv = 1;
      buf->recvChunks = recvChunks;
      recvList.push_back(buf);
    }
  }
#if DEBUG
  CkPrintf("proc %d received %d chunks out of %d for message idx %d src %d chunk # = %d\n", CkMyPe(), nrecv, numChunks, inf->idx, inf->srcPe, inf->chunkNumber);
#endif
  recvChunks[inf->chunkNumber] = newenv;
  if (nrecv == numChunks){
    void *wholemsg;
    int totalSize = 0;
    ChunkInfo* cinfo;
    for (int i = 0; i < numChunks; i++){
      cinfo = (ChunkInfo*)(EnvToUsr(recvChunks[i]));
      totalSize += cinfo->chunkSize;
    }
    wholemsg = CkAllocBuffer(newmsg, totalSize);
    cinfo = (ChunkInfo*)(EnvToUsr(recvChunks[0]));
    int offset = 0;
    for (int i = 0; i < numChunks; i++){
      cinfo = (ChunkInfo*)(EnvToUsr(recvChunks[i]));
      CmiMemcpy(((char*)wholemsg)+offset, ((char*)cinfo)+CkMsgAlignLength(sizeof(ChunkInfo)), cinfo->chunkSize);
      offset += cinfo->chunkSize;
    }
    envelope *envc = UsrToEnv(wholemsg);
    envc->setPacked(1);
    CkUnpackMessage(&envc);
    ComlibMulticastMsg *cmmsg = (ComlibMulticastMsg *)EnvToUsr(envc);
    deliverToIndices(cmmsg, localElems, local_idx_list );
    for (int i = 0; i < numChunks; i++){
      CmiFree(recvChunks[i]);
    }
    delete [] recvChunks;
 //   CmiFree(newenv);
  }
  // Forward on to other processors if necessary
  remoteMulticast(multMsg, false, cnumber, numChunks);
  if (nrecv == numChunks) {
    nrecv = 0;
    numChunks = 0;
  }
}


    


void ChunkMulticastStrategy::determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, int * &pelist, int &npes, int chunkNumber, int numChunks) {
  //numChunks = totalDestPEs;
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

void ChunkRingMulticastStrategy::determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, int * &pelist, int &npes, int chunkNumber, int numChunks) {
  if (myIndex == -1){
    npes = 1;
    pelist = new int[1];
    pelist[0] = destPEs[chunkNumber*(totalDestPEs/numChunks)].pe;
  }
  else if (chunkNumber*(totalDestPEs/numChunks) != (myIndex+1) % totalDestPEs){
    // All non-final PEs will send to next PE in list
    npes = 1;
    pelist = new int[1];
    pelist[0] = destPEs[(myIndex+1) % totalDestPEs].pe;
  }
  else {
    npes = 0;
  }
}

/* Chunk Tree multicast strategy
 * 1. Send chunks from source to unique places in array, with chunk i of c going to destination i*(d/c) of d.
 * 2. Destination processor i*(d/c) forwards chunk to (i+1)*(d/c), unless last chunk
 * 3. Destination processor i*(d/c) also forwards chunk to children defined recursively within (i*(d/c), (i+1)*(d/c)), with hopsize = numChunks
 *
 * Note that the communication is persistent. (Any intermediate processor sends to the same processors regardless of chunk number.)
 */

//FIX: remainer pes are handled inefficiently. i.e. if destPes = 15 and numchunks = 8, pe 7 will send to 7 pes, while pe 0-6 send to no one.
//FIX: is there a way to calculate children statically instead of recalculating them each iteration?
void ChunkTreeMulticastStrategy::determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, int * &pelist, int &npes, int chunkNumber, int numChunks) {
  int hop;
  //if i am source
  if (myIndex == -1){
    npes = 1;
    pelist = new int[1];
    hop = totalDestPEs/numChunks;
    pelist[0] = destPEs[chunkNumber*(totalDestPEs/numChunks)].pe;
  }
  else {
    int depth = 1;
    int idx = myIndex;
    //ipes keeps track of how many pes we are sending to
    int ipes = totalDestPEs;
    while (1){
      hop = ipes/numChunks;
      if (hop == 0) hop = 1;
      //if i am in remainder (at end)
      if (idx >= hop*(numChunks-1)){
	idx = idx - hop*(numChunks-1);
	ipes = ipes - hop*(numChunks-1) - 1;
      }
      else {
	idx = idx % hop;
	ipes = hop - 1;
      }
      depth++;
      if (idx == 0) break;
      else idx--;
    }
    //if i receive the chunk first and if i need to pass it on (the chunk is not the last one)
    if ( depth == 2 && ((chunkNumber-1+numChunks)%numChunks)*(totalDestPEs/numChunks) != myIndex){
      if (numChunks < ipes) npes = numChunks + 1;
      else npes = ipes + 1;
      pelist = new int[npes];
      //send chunk to next 2nd depth node along with all my children
      if (myIndex == (totalDestPEs/numChunks)*(numChunks-1))
	pelist[0] = destPEs[0].pe;
      else
	pelist[0] = destPEs[(myIndex+(totalDestPEs/numChunks))].pe;
      hop = ipes/npes;
      if (hop == 0) hop = 1;
      for ( int i = 1; i < npes; i++ ){
	pelist[i] = destPEs[(i-1)*hop + myIndex + 1].pe;
      }
    }
    //pass chunk onto children
    else {
      //if no children
      if (ipes <= 0){
	npes = 0;
	return;
      }
      if (numChunks < ipes) npes = numChunks;
      else npes = ipes ;

      pelist = new int[npes];
      hop = ipes/npes;
      if (hop == 0) hop = 1;
      for ( int i = 0; i < npes; i++ ){
	pelist[i] = destPEs[i*hop + myIndex + 1].pe;
      }
    }
  }
}

//like tree except with the mother node sending chunks instead of a ring between depth 1 processors
void ChunkPipeTreeMulticastStrategy::determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, int * &pelist, int &npes, int chunkNumber, int numChunks) {
  /*  int hop;*/
  int *allpelist;
  CkPrintf("myindex = %d\n", myIndex);
  if (myIndex == -1) {
    allpelist = new int[totalDestPEs+1];
    allpelist[0] = CkMyPe();
    for (int i = 1; i < totalDestPEs; i++){
      allpelist[i] = destPEs[i-1].pe;
    }
  } else {
    allpelist = new int[totalDestPEs];
    for (int i = myIndex; i < totalDestPEs + myIndex; i++){
      allpelist[i-myIndex] = destPEs[i%totalDestPEs].pe;
    }
  }
  topo::SpanningTreeVertex *nextGenInfo;
  nextGenInfo = topo::buildSpanningTreeGeneration(allpelist, allpelist + totalDestPEs, degree);
  npes = nextGenInfo->childIndex.size();
  pelist = new int[npes];
  for (int i = 0; i < npes; i++){
    pelist[i] = nextGenInfo->childIndex[i];
  }


  //if i am source
  /*if (myIndex == -1){
    npes = degree;
    if (degree > totalDestPEs) npes = totalDestPEs;
    pelist = new int[npes];
    hop = totalDestPEs/npes;
    if (hop == 0) hop = 1;
    for (int i = 0; i < npes; i++){
      pelist[i] = destPEs[i*hop].pe;
    }
  }
  else {
    int depth = 1;
    int idx = myIndex;
    //ipes keeps track of how many pes we are sending to
    int ipes = totalDestPEs;
    while (1){
      hop = ipes/degree;
      if (hop == 0) hop = 1;
      //if i am in remainder (at end)
      if (idx >= hop*(degree-1)){
	idx = idx - hop*(degree-1);
	ipes = ipes - hop*(degree-1) - 1;
      }
      else {
	idx = idx % hop;
	ipes = hop - 1;
      }
      depth++;
      if (idx == 0) break;
      else idx--;
    }
    //pass chunk onto children
      //if no children
    if (ipes <= 0){
      npes = 0;
      return;
    }
    if (degree < ipes) npes = degree;
    else npes = ipes;

    pelist = new int[npes];
    hop = ipes/npes;
    if (hop == 0) hop = 1;
    for ( int i = 0; i < npes; i++ ){
      pelist[i] = destPEs[i*hop + myIndex + 1].pe;
    }
  }*/
}

/*@}*/
