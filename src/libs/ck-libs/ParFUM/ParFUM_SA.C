/** File: ParFUM_SA.C
 *  Author: Nilesh Choudhury
 *
 *  This file defines the functions for the ParFUMShadowArray class
 */

#include "ParFUM.h"
#include "ParFUM_internals.h"
#include "ParFUM_SA.h"
#include "bulk_adapt_ops.h"

CProxy_ParFUMShadowArray meshSA;
#ifdef DEBUG
#undef DEBUG
#endif
#define DEBUG(x) x
//#define DEBUG(x) 


void ParFUM_SA_Init(int meshId) {
  CkArrayID ParfumSAId;
  int idx;
  int size;
  FEM_Mesh_allocate_valid_attr(meshId, FEM_ELEM+0);
  FEM_Mesh_allocate_valid_attr(meshId, FEM_NODE);
  _registerParFUM_SA();
  TCharm *tc=TCharm::get();
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm,&idx);
  MPI_Comm_size(comm,&size);
  if(idx==0) {
    CkArrayOptions opts;
    opts.bindTo(tc->getProxy()); //bind to the current proxy
    ParfumSAId = CProxy_ParFUMShadowArray::ckNew(size, idx, opts);
  }
  MPI_Bcast(&ParfumSAId, sizeof(CkArrayID), MPI_BYTE, 0, comm);
  meshSA = ParfumSAId;
  meshSA[idx].insert(size, idx);
  FEM_Mesh *m = FEM_Mesh_lookup(meshId,"ParFUM_SA_Init");

  //initializing the lock code
  FEM_DataAttribute *lockAttr = (FEM_DataAttribute *) m->node.lookup(FEM_ADAPT_LOCK,"ParFUM_SA_Init");
  AllocTable2d<int> &lockTable = lockAttr->getInt();
  for(int i=0;i<lockAttr->getMax();i++){
    lockTable[i][0] = -1; // locking chunkID
    lockTable[i][1] = -1; // locking region
  }

  FEMMeshMsg *msg = new FEMMeshMsg(m,tc,meshId); 
  meshSA[idx].setFemMesh(msg);
  return;
}


ParFUMShadowArray::ParFUMShadowArray(int s, int i) {
  numChunks = s;
  idx = i;
  fmMesh = NULL;
  regionCount = 0;
  pendingLock.localID = -1;  
  holdingLock.localID = -1;
}

ParFUMShadowArray::ParFUMShadowArray(CkMigrateMessage *m) {
  tc = NULL;
  fmMesh = NULL;
  regionCount = 0;
  pendingLock.localID = -1;
}

ParFUMShadowArray::~ParFUMShadowArray() {
}


void ParFUMShadowArray::pup(PUP::er &p) {
  p|numChunks;
  p|idx;
  p|tproxy;
}

enum {FEM_globalID=33};
void ParFUMShadowArray::ckJustMigrated(void) {
  ArrayElement1D::ckJustMigrated();
  tc = tproxy[idx].ckLocal();
  CkVec<TCharm::UserData> &v=tc->sud;
  FEM_chunk *c = (FEM_chunk*)(v[FEM_globalID].getData());
  fmMesh = c->getMesh("ckJustMigrated");
  fmMesh->setParfumSA(this);
}

void ParFUMShadowArray::setFemMesh(FEMMeshMsg *msg) {
  fmMesh = msg->m;
  tc = msg->t;
  tproxy = tc->getProxy();
  fmMesh->setParfumSA(this);
  bulkAdapt = new BulkAdapt(msg->meshid,fmMesh,idx,thisProxy);
  return;
}

/** Helper function that uses bubble sort to sort the indices in 
    the given array in increasing order
    These lists are usually small, so bubble sort is good enough
 */
void ParFUMShadowArray::sort(int *chkList, int chkListSize) {
  for(int i=0; i<chkListSize; i++) {
    for(int j=i+1; j<chkListSize; j++) {
      if(chkList[j] < chkList[i]) {
  int tmp = chkList[i];
  chkList[i] = chkList[j];
  chkList[j] = tmp;
      }
    }
  }
  return;
}


void uniquify(CkVec<int> &vec){
  if(vec.length() != 0){
    vec.quickSort(8);
    int count=0;
    for(int i=1;i<vec.length();i++){
      if(vec[count] == vec[i]){
      }else{
        count++;
        if(i != count){
          vec[count] = vec[i];
        }
      }
    }
    vec.resize(count+1);
  }
}


bool hasGreaterPrio(RegionID x, RegionID y) {
  return ((x.prio > y.prio) || ((x.prio == y.prio) && (x.localID < y.localID)) 
  	  || ((x.prio == y.prio) && (x.localID == y.localID) && (x.chunkID < y.chunkID)));
}

/** This method locks all the nodes belonging to the 
 * specified elements. Elements might be on remote chunk.
 * Corresponding idxls between other chunks must also be locked */
int ParFUMShadowArray::lockRegion(int numElements,adaptAdj *elements,RegionID *regionID,double prio){
  //create a regionID for this region
  if (regionID->localID == -1) {
    regionID->chunkID = idx;
    regionID->localID = regionCount;
    regionID->prio = prio;
    regionCount++;
  }

  CkPrintf("[%d] INCOMING: (%d,%d,%6.4f)\n", idx, regionID->chunkID,regionID->localID,regionID->prio);
  CkPrintf("[%d] HOLDING:  (%d,%d,%6.4f)\n", idx, holdingLock.chunkID,holdingLock.localID,holdingLock.prio);
  CkPrintf("[%d] PENDING:  (%d,%d,%6.4f)\n", idx, pendingLock.chunkID,pendingLock.localID,pendingLock.prio);

  if (holdingLock.localID != -1) { // someone holds a lock on this chunk
    CkAssert((holdingLock == (*regionID)) || (holdingLock.chunkID != regionID->chunkID));
    if (holdingLock == (*regionID)) { // this regionID already has the lock
      DEBUG(printf("[%d] lockRegion SUCCEEDED: This region: (%d,%d,%6.4f) is the holding lock.\n", 
		   idx, regionID->chunkID,regionID->localID,regionID->prio));
    }
    else if (hasGreaterPrio(holdingLock, *regionID)) { // it is locked; fail
      DEBUG(printf("[%d] lockRegion FAILED: Partition is locked by higher prio region: (%d,%d,%6.4f) we have: (%d,%d,%6.4f)\n", 
		   idx,holdingLock.chunkID,holdingLock.localID,holdingLock.prio,
		   regionID->chunkID,regionID->localID,regionID->prio));
      return 0;
    }
    else { // holder is lower prio
      if (pendingLock.localID != -1) { // there is a pending region
	if (pendingLock == (*regionID)) {
	  DEBUG(printf("[%d] lockRegion PENDING: Partition is already pending with: (%d,%d,%6.4f), lock held by (%d,%d,%6.4f)\n", 
		       idx,regionID->chunkID,regionID->localID,regionID->prio,holdingLock.chunkID,holdingLock.localID,holdingLock.prio));

	  return 1;
	}
	else if (hasGreaterPrio(pendingLock, *regionID)) { //pending has prio, fail
	  DEBUG(printf("[%d] lockRegion FAILED: Partition has pending lock of higher prio region: (%d,%d,%6.4f) we have: (%d,%d,%6.4f)\n", 
		       idx,pendingLock.chunkID,pendingLock.localID,pendingLock.prio,
		       regionID->chunkID,regionID->localID,regionID->prio));
	  return 0;
	}
	else { // pending is lower priority, replace with this region
	  pendingLock = (*regionID);
	  DEBUG(printf("[%d] lockRegion PENDING: Partition has pending lower prio region: (%d,%d,%6.4f) replaced with: (%d,%d,%6.4f)\n", 
		       idx,pendingLock.chunkID,pendingLock.localID,pendingLock.prio,
		       regionID->chunkID,regionID->localID,regionID->prio));
	  return 1;
	}
      }
      else { // there is no pending region; make this pending
	pendingLock = (*regionID);
	DEBUG(printf("[%d] lockRegion PENDING: Partition has no pending region; we have: (%d,%d,%6.4f)\n", 
		     idx,regionID->chunkID,regionID->localID,regionID->prio));
	return 1;
      }
    }
  }
  else { // no one holds the lock
    if (pendingLock.localID != -1) { // there is a pending lock
      CkAssert((pendingLock == (*regionID)) || (pendingLock.chunkID != regionID->chunkID));
      if (pendingLock == (*regionID)) { // this region was pending; take the lock
	holdingLock = (*regionID);
	pendingLock.localID = pendingLock.chunkID = -1;
	DEBUG(printf("[%d] lockRegion SUCCEEDED: This region: (%d,%d,%6.4f) was pending, now has lock.\n", 
		   idx, regionID->chunkID,regionID->localID,regionID->prio));
      }
      else if (hasGreaterPrio(pendingLock, *regionID)) { // this region has lower prio to pending
	DEBUG(printf("[%d] lockRegion FAILED: Pending region should lock: (%d,%d,%6.4f) we have: (%d,%d,%6.4f)\n", 
		   idx,pendingLock.chunkID,pendingLock.localID,pendingLock.prio,
		   regionID->chunkID,regionID->localID,regionID->prio));
	return 0;
      }
      else { // this region has higher prio; take the lock
	holdingLock = (*regionID);
	DEBUG(printf("[%d] lockRegion SUCCEEDED: This region gets lock: pending (%d,%d,%6.4f) we have: (%d,%d,%6.4f)\n", 
		   idx,pendingLock.chunkID,pendingLock.localID,pendingLock.prio,
		   regionID->chunkID,regionID->localID,regionID->prio));
      }
    }
    else { //no one is pending; take the lock
      holdingLock = (*regionID);
      DEBUG(printf("[%d] lockRegion SUCCEEDED: This region gets lock with: (%d,%d,%6.4f)\n", 
		   idx, regionID->chunkID,regionID->localID,regionID->prio));
    }
  }

  LockRegion *region = new LockRegion;
  region->myID = *regionID;
  DEBUG(CkPrintf("[%d] LockRegion (%d,%d) created \n",idx,regionID->chunkID,regionID->localID));
  regionTable.put(*regionID) = region;
  
  //collect all the local nodes in this list of elements into one list
  collectLocalNodes(numElements,elements,region->localNodes);
  
  //Try to lock all the local nodes. If any fails unlock all and return
  //add this region to the regionTable if the locals have all been locked
  FEM_Node &fmNode = fmMesh->node;
  IDXL_Side &shared = fmNode.shared;
  /*
  bool success = lockLocalNodes(region);
  DEBUG(CkPrintf("[%d] LockRegion %d lockLocalNodes success=%d  \n",idx,regionID->localID,success));
  if(!success){ //we could not lock at least one of the local nodes
    unlockLocalNodes(region);
    delete region;
    pendingLock.localID = -1;
    return 0;
  }else{ //all local nodes were sucessfully locked
    regionTable.put(*regionID) = region;
  }
  */
  
  //find list of chunks that share nodes (have a shared idxl) with the local nodes of this region
  region->sharedIdxls.push_back(idx);
  for(int i=0;i<region->localNodes.length();i++){
    int node = region->localNodes[i];
    const IDXL_Rec *rec=shared.getRec(node);
    if(rec!=NULL){ //Put in the list of shared Nodes
      for(int j=0;j<rec->getShared();j++){
	region->sharedIdxls.push_back(rec->getChk(j));
      }
    }
  }
  uniquify(region->sharedIdxls);	
  DEBUG(CkPrintf("[%d] LockRegion %d number of SharedIdxls %d \n",idx,regionID->localID,region->sharedIdxls.length()));
  if(region->sharedIdxls.length() == 1){ //no remote chunks are invovled in this region's locking
    DEBUG(CkPrintf("[%d] LockRegion %d successfully locked with no remote messages \n",idx,regionID->localID));
    pendingLock.localID = -1;
    return 2;
  }
  
  /*
  //lock the local shared idxls
  success = lockSharedIdxls(region);
  if(!success){ //we could not lock one of the idxls owned by us
    //The idxls that were locked are already been unlocked; we need to unlock only the local nodes
    unlockLocalNodes(region);
    regionTable.remove(*regionID);
    delete region;
    pendingLock.localID = -1;
    return 0;
  }
  */

  //separate the list of remote elements by chunk
  for(int i=0;i<numElements;i++){
    if(elements[i].partID != idx){ //remote element
      CkVec<adaptAdj> *list = region->remoteElements.get(elements[i].partID);
      if(list == NULL){	//first element of this type
	list = new CkVec<adaptAdj>;
	DEBUG(printf("[%d] Adding remoteElements %p for partition %d\n",idx,list,elements[i].partID););
	region->remoteElements.put(elements[i].partID) = list;
	CkAssert(region->remoteElements.get(elements[i].partID) ==list);
      }
      list->push_back(elements[i]);
    }
  }
  
  //send message to lock all the idxls for all the chunks and corresponding remote elements
  DEBUG(printf("[%d] Begin locking remote nodes and IDXLs for (%d,%d)...\n", idx, regionID->chunkID, regionID->localID));
  for(int i=0;i<region->sharedIdxls.length();i++){
    int chunk = region->sharedIdxls[i];
    if(chunk == idx) { // local chunk is already locked
      continue;
    }
    CkVec<adaptAdj> *list = region->remoteElements.get(chunk);
    adaptAdj *elementsForChunk=NULL;
    int numElementsForChunk=0;
    if(list != NULL){
      elementsForChunk = list->getVec();
      numElementsForChunk = list->size();
    }
    thisProxy[chunk].lockRegionForRemote(region->myID,region->sharedIdxls.getVec(),region->sharedIdxls.size(),elementsForChunk,numElementsForChunk);		
  }
  region->numReplies = 1; //this chunk itself has locked all its local nodes and idxl
  region->success = 2;
  region->tid = CthSelf();
  CthSuspend();
  
  if(region->success==2){
    DEBUG(printf("[%d] Remote locking successful for (%d,%d)...\n", idx, regionID->chunkID, regionID->localID););
    return 2;
  }
  else if (region->success == 1) {
    DEBUG(printf("[%d] Remote locking pending for (%d,%d). Trying again...\n", idx, regionID->chunkID, regionID->localID););
    return 1;
  }
  else {
    DEBUG(printf("[%d] Remote locking failed for (%d,%d)...\n", idx, regionID->chunkID, regionID->localID););
    unpendRegion(*regionID);
    unlockRegion(*regionID);
    regionTable.remove(*regionID);
    //delete region;
    holdingLock.localID = -1;
    return 0;
  }
}


/**  Find all the local nodes in the connectivity of the list of elements 
  * and store it in the vector localNodes */
void ParFUMShadowArray::collectLocalNodes(int numElements,adaptAdj *elements,CkVec<int> &localNodes){
  for(int i=0;i<numElements;i++){
    //check if the element is local
    if(elements[i].partID < 0 || elements[i].localID < 0) {
      CkAssert(elements[i].partID < 0 && elements[i].localID < 0);
      continue;
    }
    if(elements[i].partID == idx){
      int elemType = elements[i].elemType;
      const FEM_Elem &elem = fmMesh->getElem(elemType);
      const int *conn = elem.connFor(elements[i].localID);
      const int numNodes = elem.getNodesPer();
      DEBUG(printf("Nodes for local element %d: ",elements[i].localID));
      for(int j=0;j<numNodes;j++){
        int node = conn[j];
        CkAssert(node >= 0);
        localNodes.push_back(node);
	DEBUG(printf("%d ",node));
      }
      DEBUG(printf("\n"));
    }
  }
  //now we need to uniquify the list of localNodes
  uniquify(localNodes);
}


bool ParFUMShadowArray::lockLocalNodes(LockRegion *region){
  FEM_Node &fmNode = fmMesh->node;
  AllocTable2d<int> &lockTable = ((FEM_DataAttribute *)fmNode.lookup(FEM_ADAPT_LOCK,"lockRegion"))->getInt();
  
  bool success=true;
  for(int i=0;i<region->localNodes.length();i++){
    int node = region->localNodes[i];
    
    if(lockTable[node][0]!= -1){
      //the node is locked
      DEBUG(printf("[%d] Node %d was locked \n",thisIndex,node));
      success = false;
      break;
    }
    else {
      //the node is unlocked
      lockTable[node][0] = region->myID.chunkID;
      lockTable[node][1] = region->myID.localID;
      
      DEBUG(printf("[%d] Locking Node %d \n",thisIndex,node));
    }
  }
  return success;
}


/** Lock the sharedidxls specified in the region.
 *    If idx < chunkID of shared idxl, lock the idxl with chunkID on this chunk
 *    If we cant lock all the idxls we need to, clean up the locked ones and return false
 */
bool ParFUMShadowArray::lockSharedIdxls(LockRegion *region){
  bool success=true;
  int lockedIdx=-1;
  for(int i=0;i<region->sharedIdxls.length();i++){
    int chunkID = region->sharedIdxls[i];
    if(idx < chunkID){
      IDXL_List *list =  fmMesh->node.shared.getIdxlListN(chunkID);
      if(list != NULL){
	if(list->isLocked()){ //list is locked 
	  success = false;
	  lockedIdx= i;
	  break;
	}else{
	  list->lockIdxl();
	}
      }
    }
  }

  if(!success){
    for(int i=0;i<lockedIdx;i++){
      int chunkID = region->sharedIdxls[i];
      if(idx < chunkID){
	IDXL_List *list =  fmMesh->node.shared.getIdxlListN(chunkID);
	if(list != NULL){
	  if(list->isLocked()){
	    list->unlockIdxl();
	  }
	}
      }
    }
  }
  return success;
}

/** Process a request from another chunk to lock these idxls and the nodes belonging to these elements
 */
void ParFUMShadowArray::lockRegionForRemote(RegionID regionID,int *sharedIdxls,int numSharedIdxls,adaptAdj *elements,int numElements){
  LockRegion *region = new LockRegion;
  region->myID = regionID;
  static int tag=0;
  int success;

  DEBUG(printf("lockRegionForRemote: regionID=(%d,%d,%6.4f) holdingLock=(%d,%d,%6.4f) pendingLock=(%d,%d,%6.4f)\n", regionID.chunkID, regionID.localID, regionID.prio, holdingLock.chunkID, holdingLock.localID, holdingLock.prio, pendingLock.chunkID, pendingLock.localID, pendingLock.prio));
  if (holdingLock.localID != -1) { // someone holds a lock on this chunk
    CkAssert((holdingLock == regionID) || (holdingLock.chunkID != regionID.chunkID));
    if (holdingLock == regionID) { // this regionID already has the lock
      success=2;
      DEBUG(printf("[%d] lockRegionForRemote SUCCEEDED: This region: (%d,%d,%6.4f) is the holding lock.\n", 
		   idx, regionID.chunkID,regionID.localID,regionID.prio));
    }
    else if (hasGreaterPrio(holdingLock, regionID)) { // it is locked; fail
      success=0;
      DEBUG(printf("[%d] lockRegionForRemote FAILED: Partition is locked by higher prio region: (%d,%d,%6.4f) we have: (%d,%d,%6.4f)\n", 
		   idx,holdingLock.chunkID,holdingLock.localID,holdingLock.prio,
		   regionID.chunkID,regionID.localID,regionID.prio));
    }
    else {
      if (pendingLock.localID != -1) {
	if (pendingLock == regionID) {
	  success = 1;
	  DEBUG(printf("[%d] lockRegion PENDING: Partition is already pending with: (%d,%d,%6.4f), lock held by (%d,%d,%6.4f)\n", 
		       idx,regionID.chunkID,regionID.localID,regionID.prio,holdingLock.chunkID,holdingLock.localID,holdingLock.prio));
	}
	else if (hasGreaterPrio(pendingLock, regionID)) {
	  success=0;
	  DEBUG(printf("[%d] lockRegion FAILED: Partition has pending lock of higher prio region: (%d,%d,%6.4f) we have: (%d,%d,%6.4f)\n", 
		       idx,pendingLock.chunkID,pendingLock.localID,pendingLock.prio,
		       regionID.chunkID,regionID.localID,regionID.prio));
	}
	else {
	  pendingLock = regionID;
	  success=1;
	  DEBUG(printf("[%d] lockRegion PENDING: Partition has pending lower prio region: (%d,%d,%6.4f) replaced with: (%d,%d,%6.4f)\n", 
		       idx,pendingLock.chunkID,pendingLock.localID,pendingLock.prio,
		       regionID.chunkID,regionID.localID,regionID.prio));
	}
      }
      else { // there is no pending region; make this pending
	pendingLock = regionID;
	success=1;
	DEBUG(printf("[%d] lockRegion PENDING: Partition has no pending region; we have: (%d,%d,%6.4f)\n", 
		     idx,regionID.chunkID,regionID.localID,regionID.prio));
      }
    }
  }
  else { // no one holds the lock
    if (pendingLock.localID != -1) { // there is a pending lock
      CkAssert((pendingLock == regionID) || (pendingLock.chunkID != regionID.chunkID));
      if (pendingLock == regionID) {
	holdingLock = regionID;
	pendingLock.localID = pendingLock.chunkID = -1;
	success=2;
	DEBUG(printf("[%d] lockRegion SUCCEEDED: This region: (%d,%d,%6.4f) was pending, now has lock.\n", 
		   idx, regionID.chunkID,regionID.localID,regionID.prio));
      }
      else if (hasGreaterPrio(pendingLock, regionID)) {
	success=0;
	DEBUG(printf("[%d] lockRegion FAILED: Pending region should lock: (%d,%d,%6.4f) we have: (%d,%d,%6.4f)\n", 
		     idx,pendingLock.chunkID,pendingLock.localID,pendingLock.prio,
		     regionID.chunkID,regionID.localID,regionID.prio));
      }
      else {
	holdingLock = regionID;
	success=2;
	DEBUG(printf("[%d] lockRegion SUCCEEDED: This region gets lock: pending (%d,%d,%6.4f) we have: (%d,%d,%6.4f)\n", 
		     idx,pendingLock.chunkID,pendingLock.localID,pendingLock.prio,
		     regionID.chunkID,regionID.localID,regionID.prio));

      }
    }
    else {
      holdingLock = regionID;
      success=2;
      DEBUG(printf("[%d] lockRegion SUCCEEDED: This region gets lock with: (%d,%d,%6.4f)\n", 
		   idx, regionID.chunkID,regionID.localID,regionID.prio));
      
    }
  }

  /*
  //make a list of all the nodes that are specified in the elements
  collectLocalNodes(numElements,elements,region->localNodes);
  //try locking all the local nodes,
  success = 0;
  if (lockLocalNodes(region)) {
    success = 2;
    for(int i=0;i<numSharedIdxls;i++){
      region->sharedIdxls.push_back(sharedIdxls[i]);
    }
    if (!(lockSharedIdxls(region)))
      success = 0;
  }

  // could not lock either the shared idxls or the local nodes; need to unlock local nodes before replying
  if (success == 0) {
    unlockLocalNodes(region);
    pendingLock.localID = -1;
    delete region;
  }
  else if (success == 1) {
    unlockLocalNodes(region);
    delete region;
  }
  else{ //if the locking was successful, store the region
    regionTable.put(regionID) = region;
  }
  */
    
  if (success == 2) {
    regionTable.put(regionID) = region;
  }

  DEBUG(CkPrintf("[%d] Lockregion (%d,%d,%6.4f) success=%d sending lockReply (tag=%d) to %d\n",thisIndex,regionID.chunkID,regionID.localID,regionID.prio,success,tag,regionID.chunkID));
  //send a reply back to the requesting chunk with the result of the lock attempt
  thisProxy[regionID.chunkID].lockReply(idx,regionID,success,tag,success);
  DEBUG(CkPrintf("[%d] Lockregion (%d,%d,%6.4f) success=%d sent lockReply (tag=%d) to %d\n",thisIndex,regionID.chunkID,regionID.localID,regionID.prio,success,tag,regionID.chunkID));
  tag++;
}

/** Collect the replies from all the remote chunks that were 
 * requested to lock regions of the mesh. 
 * If all chunks reply that they have locked successfully, 
 * then we just wake up the thread waiting for this region to lock.
 * If some chunks fail, we need to unlock the region before replying
 * */
void ParFUMShadowArray::lockReply(int remoteChunk,RegionID regionID, int foosuccess, int tag, int otherSuccess){
  LockRegion *region = regionTable.get(regionID);
  DEBUG(CkPrintf("[%d] Lockregion reply (tag=%d) received (%d,%d,%6.4f) foosuccess=%d otherSuccess=%d from chunk %d region=%x\n",thisIndex,tag,regionID.chunkID,regionID.localID,regionID.prio,foosuccess, otherSuccess, remoteChunk, region));
  CkAssert(region != NULL);
  region->numReplies++;
  if (region->success > otherSuccess)
    region->success = otherSuccess;
  if(region->numReplies == region->sharedIdxls.size()){
    if(region->success == 2){
      CthAwaken(region->tid);
    }
    else if (region->success == 1) {
      CthAwaken(region->tid);
    }	
    else {
      unlockRegion(region);
    }	
  }
}

void ParFUMShadowArray::unpendRegion(RegionID regionID){
  CkAssert(regionID.chunkID == idx);
  LockRegion *region = regionTable.get(regionID);
  if (region!=NULL) {
    for(int i=0;i<region->sharedIdxls.length();i++){
      int chunk = region->sharedIdxls[i];
      if(chunk != idx){
	thisProxy[chunk].unpendForRemote(region->myID);
      }
    }
    region->numReplies = 1; //this chunk has unlocked all the stuff on it
    if(region->sharedIdxls.length() == 1){ //there are no remote chunks involved in this region
    }else{ //wait for unlock to finish on other chunks
      CthSuspend();
    }
    regionTable.remove(regionID);
    delete region;
  }
  if (holdingLock == regionID) 
    holdingLock.localID = -1;
  if (pendingLock == regionID)
    pendingLock.localID = -1;
}

void ParFUMShadowArray::unpendForRemote(RegionID regionID){
  LockRegion *region = regionTable.get(regionID);
  if (region != NULL) {
    delete region;
    regionTable.remove(regionID);
  }
  if (holdingLock == regionID) 
    holdingLock.localID = -1;
  if (pendingLock == regionID)
    pendingLock.localID = -1;
  thisProxy[regionID.chunkID].unlockReply(idx,regionID);
}

void ParFUMShadowArray::unlockRegion(RegionID regionID){
  CkAssert(regionID.chunkID == idx);
  CkAssert((regionID == holdingLock) || (regionID.localID == -1));
  if (regionID.localID != -1) {
    LockRegion *region = regionTable.get(regionID);
    CkAssert(region!=NULL);
    unlockRegion(region);
    if(region->sharedIdxls.length() == 1){ //there are no remote chunks involved in this region
    }else{ //wait for unlock to finish on other chunks
      CthSuspend();
    }
    holdingLock.localID = -1;
    regionTable.remove(regionID);
    delete region;
  }
  DEBUG(CkPrintf("[%d] LockRegion (%d,%d,%6.4f) unlocked\n",idx,regionID.chunkID,regionID.localID,regionID.prio));
  CkPrintf("[%d] INCOMING: (%d,%d,%6.4f)\n", idx, regionID.chunkID,regionID.localID,regionID.prio);
  CkPrintf("[%d] HOLDING:  (%d,%d,%6.4f)\n", idx, holdingLock.chunkID,holdingLock.localID,holdingLock.prio);
  CkPrintf("[%d] PENDING:  (%d,%d,%6.4f)\n", idx, pendingLock.chunkID,pendingLock.localID,pendingLock.prio);
}

/** Free up all the local nodes, idxl and elements on remote chunk locked by this node */
void ParFUMShadowArray::unlockRegion(LockRegion *region){
  //unlock the local nodes
  //unlockLocalNodes(region);
  //unlock the sharedIdxls on this chunk
  //unlockSharedIdxls(region);
  //free the idxl and elements on remote chunks
  for(int i=0;i<region->sharedIdxls.length();i++){
    int chunk = region->sharedIdxls[i];
    if(chunk != idx){
      thisProxy[chunk].unlockForRemote(region->myID);
    }
  }
  region->numReplies = 1; //this chunk has unlocked all the stuff on it
}

void ParFUMShadowArray::unlockLocalNodes(LockRegion *region){
  //unlock the localNodes
  FEM_Node &fmNode = fmMesh->node;
  IDXL_Side &shared = fmNode.shared;
  AllocTable2d<int> &lockTable = ((FEM_DataAttribute *)fmNode.lookup(FEM_ADAPT_LOCK,"lockRegion"))->getInt();
  
  for(int i=0;i<region->localNodes.length();i++){
    //is the node locked and has it been locked by this region
    if(lockTable[region->localNodes[i]][0] == region->myID.chunkID && lockTable[region->localNodes[i]][1] == region->myID.localID){ //the node had been locked by this region earlier; unlocking it now
      lockTable[region->localNodes[i]][0] = -1;
      lockTable[region->localNodes[i]][1] = -1;
      
      DEBUG(printf("[%d] Un Locking Node  %d \n",thisIndex,region->localNodes[i]));
    }
  }
}

/** Unlock all the shared idxl on this chunk */
void ParFUMShadowArray::unlockSharedIdxls(LockRegion *region){
  for(int i=0;i<region->sharedIdxls.length();i++){
    int chunkID = region->sharedIdxls[i];
    if(idx < chunkID){
      IDXL_List *list =  fmMesh->node.shared.getIdxlListN(chunkID);
      if(list != NULL){
	if(list->isLocked()){
	  list->unlockIdxl();
	}
	else {
	  CmiAbort("IDXL was not locked. Attempting to unlock it?");
	}
      }
    }
  }
}

/** Unlock the local nodes and sharedIdxls locked for the remote region specified by regionID.
 * We need to send an ack back
 */
void ParFUMShadowArray::unlockForRemote(RegionID regionID){
  if ((regionID == holdingLock) || (holdingLock.localID == -1)) {
    LockRegion *region = regionTable.get(regionID);
    if(region != NULL){
      //unlockLocalNodes(region);
      //unlockSharedIdxls(region);
      regionTable.remove(regionID);
      delete region;
    }
    CkPrintf("[%d] INCOMING: (%d,%d,%6.4f)\n", idx, regionID.chunkID,regionID.localID,regionID.prio);
    CkPrintf("[%d] HOLDING:  (%d,%d,%6.4f)\n", idx, holdingLock.chunkID,holdingLock.localID,holdingLock.prio);
    CkPrintf("[%d] PENDING:  (%d,%d,%6.4f)\n", idx, pendingLock.chunkID,pendingLock.localID,pendingLock.prio);
    if (holdingLock == regionID) 
      holdingLock.localID = -1;
    DEBUG(CkPrintf("[%d] Lockregion (%d,%d,%6.4f) unlocked \n",thisIndex,regionID.chunkID,regionID.localID,regionID.prio));
  }
  thisProxy[regionID.chunkID].unlockReply(idx,regionID);
}

/** Collect the replies to unlock requests from all the chunks that are involved
 * in this region. Once all the chunks have replied wake up the calling thread
 */
void ParFUMShadowArray::unlockReply(int remoteChunk,RegionID regionID){
  LockRegion *region = regionTable.get(regionID);
  CkAssert(region != NULL);
  region->numReplies++;
  if(region->numReplies == region->sharedIdxls.length()){
    CthAwaken(region->tid);
  }
}


/** Helper function that translates the sharedChk and idxlType 
    to the corresponding idxl_Side (FEM_Comm)
 */
FEM_Comm *ParFUMShadowArray::FindIdxlSide(int idxlType) {
  if(idxlType == 0) { //shared node
    return &(fmMesh->node.shared);
  }
  else if(idxlType == 1) { //ghost node send 
    return &(fmMesh->node.ghostSend);
  }
  else if(idxlType == 2) { //ghost node recv 
    return &(fmMesh->node.ghost->ghostRecv);
  }
  else if((idxlType-3)%2 == 0) { //ghost elem send 
    int elemType = (idxlType-3)/2;
    return &(fmMesh->elem[elemType].ghostSend);
  }
  else if((idxlType-4)%2 == 0) { //ghost elem recv 
    int elemType = (idxlType-4)/2;
    return &(fmMesh->elem[elemType].ghost->ghostRecv);
  }
  return NULL;
}


/** This operation adds an entry in the idxl list on the primary with
    'sharedChk'. The idxl list is again determined by idxlType. It retuns the
    index on the idxl list where it was added, which is used by the following
    operation. Note that here Primary does not have any relation to the global
    number, it is just the first chunk in the idxl pair where this entry is
    being added. The localId has a globalNo associated with it, so we just
    store the localId in the IDXL list, no change in IDXL necessary :)
*/
int ParFUMShadowArray::IdxlAddPrimary(int localId, int sharedChk, int idxlType){
  if(idxlType>0 && idxlType%2==0) {
    localId = FEM_From_ghost_index(sharedChk);
  }
  return (FindIdxlSide(idxlType))->addNode(localId,sharedChk);
}

/**  This is pretty self-explanatory. Add the entry 'loclaId' on the
     idxl list defined by 'idxltype' with 'sharedChk' at the location
     'sharedIdx'. Returns success/failure
 */
bool ParFUMShadowArray::IdxlAddSecondary(int localId, int sharedChk, int sharedIdx, int idxlType){
  if(idxlType>0 && idxlType%2==0) {
    localId = FEM_From_ghost_index(sharedChk);
  }
  return (FindIdxlSide(idxlType))->setNode(localId,sharedChk,sharedIdx);
}

/** Search for this 'localId' on this idxl list and remove this entry.
    and return the index, which will be used by the following operation.
*/
int ParFUMShadowArray::IdxlRemovePrimary(int localId, int sharedChk, int idxlType){
  if(idxlType>0 && idxlType%2==0) {
    localId = FEM_From_ghost_index(sharedChk);
  }
  return (FindIdxlSide(idxlType))->removeNode(localId,sharedChk);
}

/** Remove the entry that is at sharedIdx on the idxllist specified by
    sharedChk and IdxlType
*/
bool ParFUMShadowArray::IdxlRemoveSecondary(int sharedChk, int sharedIdx, int idxlType){
  return (FindIdxlSide(idxlType))->unsetNode(sharedChk,sharedIdx);
}

/** Search for this localId in this Idxl list and return the entry
    index; Return -1 if it does not exist in this IDXL list
*/
int ParFUMShadowArray::IdxlLookUpPrimary(int localId, int sharedChk, int idxlType){
  if(idxlType>0 && idxlType%2==0) {
    localId = FEM_From_ghost_index(sharedChk);
  }
  return (FindIdxlSide(idxlType))->existsNode(localId,sharedChk);
}

/** Return the localId at this index 'sharedIdx' on the idxl list
    specified by sharedChk and IdxlType.
*/
int ParFUMShadowArray::IdxlLookUpSecondary(int sharedChk, int sharedIdx, int idxlType){
  return (FindIdxlSide(idxlType))->getNode(sharedChk,sharedIdx);
}


// BULK ADAPT ENTRY METHODS START HERE:
adaptAdjMsg *ParFUMShadowArray::remote_bulk_edge_bisect_2D(adaptAdj &nbrElem, adaptAdj &splitElem, int new_idxl, int n1_idxl, int n2_idxl, int partitionID)
{
  adaptAdjMsg *am = new adaptAdjMsg;
  am->elem = bulkAdapt->remote_edge_bisect_2D(nbrElem, splitElem, new_idxl, n1_idxl, n2_idxl, partitionID);
  return am;
}

longestMsg *ParFUMShadowArray::isLongest(int elem, int elemType, double len)
{
  longestMsg *m = new longestMsg;
  m->longest = bulkAdapt->isLongest(elem, elemType, len);
  return m;
}

void ParFUMShadowArray::remote_adaptAdj_replace(adaptAdj &elem, adaptAdj &oldElem, adaptAdj &newElem)
{
  bulkAdapt->remote_adaptAdj_replace(elem, oldElem, newElem);
}

void ParFUMShadowArray::remote_edgeAdj_replace(int remotePartID, adaptAdj &adj, 
					       adaptAdj &elem, 
					       adaptAdj &splitElem, int n1_idxl,
					       int n2_idxl)
{
  bulkAdapt->remote_edgeAdj_replace(remotePartID, adj, elem, splitElem, 
				    n1_idxl, n2_idxl);
}

void ParFUMShadowArray::remote_edgeAdj_add(int remotePartID, adaptAdj &adj, 
					   adaptAdj &splitElem, int n1_idxl, 
					   int n2_idxl)
{
  bulkAdapt->remote_edgeAdj_add(remotePartID, adj, splitElem, n1_idxl, n2_idxl);
}

void ParFUMShadowArray::recv_split_3D(int pos, int tableID, adaptAdj &elem, 
				      adaptAdj &splitElem)
{
  bulkAdapt->recv_split_3D(pos, tableID, elem, splitElem);
}

void ParFUMShadowArray::handle_split_3D(int remotePartID, int pos, int tableID,
					adaptAdj &elem, RegionID lockRegionID, int n1_idxl, 
					int n2_idxl, int n5_idxl)
{
  CkAssert(holdingLock == lockRegionID);
  bulkAdapt->handle_split_3D(remotePartID, pos, tableID, elem, lockRegionID, n1_idxl, 
			     n2_idxl, n5_idxl);
}

void ParFUMShadowArray::recv_splits(int tableID, int expectedSplits)
{
  while (!(bulkAdapt->all_splits_received(tableID, expectedSplits)))
    CthYield();
}

void ParFUMShadowArray::update_asterisk_3D(int remotePartID, int i, 
					   adaptAdj &elem, int numElemPairs, 
					   adaptAdj *elemPairs, RegionID lockRegionID,
					   int n1_idxl, int n2_idxl, int n5_idxl)
{
  CkAssert(holdingLock == lockRegionID);
  bulkAdapt->update_asterisk_3D(remotePartID, i, elem, numElemPairs, elemPairs,
				lockRegionID, n1_idxl, n2_idxl, n5_idxl);
}


#include "ParFUM_SA.def.h"
