/** File: ParFUM_SA.C
 *  Author: Nilesh Choudhury
 *
 *  This file defines the functions for the ParFUMShadowArray class
 */

#include "ParFUM.h"
#include "ParFUM_internals.h"
#include "ParFUM_SA.h"

CProxy_ParFUMShadowArray meshSA;



void ParFUM_SA_Init(int meshId) {
  CkArrayID ParfumSAId;
  int idx;
  int size;
  FEM_Mesh_allocate_valid_attr(meshId, FEM_ELEM+0);
  FEM_Mesh_allocate_valid_attr(meshId, FEM_NODE);
  //_registerParFUM_SA();
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
  
	FEMMeshMsg *msg = new FEMMeshMsg(m,tc); 
  meshSA[idx].setFemMesh(msg);
  return;
}



ParFUMShadowArray::ParFUMShadowArray(int s, int i) {
  numChunks = s;
  idx = i;
  fmMesh = NULL;
	regionCount = 0;
}

ParFUMShadowArray::ParFUMShadowArray(CkMigrateMessage *m) {
  tc = NULL;
  fmMesh = NULL;
	regionCount = 0;
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


/** This method locks all the nodes belonging to the 
 * specified elements. Elements might be on remote chunk.
 * Corresponding idxls between other chunks must also be locked */
bool ParFUMShadowArray::lockRegion(int numElements,adaptAdj *elements,RegionID *regionID){
  
	//create a regionID for this region
	regionID->chunkID = idx;
	regionID->localID = regionCount;
	regionCount++;

	LockRegion * region = new LockRegion;
	region->myID = *regionID;

  //collect all the local nodes in this list of elements into one list
  collectLocalNodes(numElements,elements,region->localNodes);
  
	//Try to lock all the local nodes. If any fails unlock all and
  //return
	
	FEM_Node &fmNode = fmMesh->node;
	IDXL_Side &shared = fmNode.shared;
	AllocTable2d<int> &lockTable = ((FEM_DataAttribute *)fmNode.lookup(FEM_ADAPT_LOCK,"lockRegion"))->getInt();
	bool success = true;

	for(int i=0;i<region->localNodes.length();i++){
		int node = region->localNodes[i];
		if(shared.getRec(node)==NULL){
			//Purely Local node
		}else{
			//Shared node
		}

		if(lockTable[node][0]!= -1){
			//the node is locked
			success = false;
			break;
		}else{
			//the node is unlocked
			lockTable[node][0] = idx;
			lockTable[node][1] = region->myID.localID;
		}
	}

	//add this region to the regionTable if the locals have all been locked
	if(!success){
		//we could not lock one of the local nodes
		freeRegion(region);
		return false;
	}else{
		//all local nodes were sucessfully locked
		regionTable.put(*regionID) = region;
	}
  
  //Of the local nodes some are shared. Find all the chunks that 
  //share these nodes and lock all the idxls for that
  
  

  //elements that are remote need to be sent to a remote 
  //processor and their nodes need to be locked on the remote
  
  
}

/** Free up all the local nodes, idxl and elements 
 * on remote chunk locked by this node */
void ParFUMShadowArray::freeRegion(LockRegion *region){
	//free the localNodes
	FEM_Node &fmNode = fmMesh->node;
	IDXL_Side &shared = fmNode.shared;
	AllocTable2d<int> &lockTable = ((FEM_DataAttribute *)fmNode.lookup(FEM_ADAPT_LOCK,"lockRegion"))->getInt();
	
	for(int i=0;i<region->localNodes.length();i++){
		//is the node locked and has it been locked by this region
		if(lockTable[region->localNodes[i]][0] == region->myID.chunkID && lockTable[region->localNodes[i]][1] == region->myID.localID ){
			//the node had been locked by this region earlier
			//unlocking it now
			lockTable[region->localNodes[i]][0] = -1;
			lockTable[region->localNodes[i]][1] = -1;
		}
	}

	//TODO: free the idxl and elements on remote chunks
	
}

void ParFUMShadowArray::unlockRegion(RegionID regionID){
	CkAssert(regionID.chunkID == idx);
	LockRegion *region = regionTable.get(regionID);
	freeRegion(region);
}



/**  Find all the local nodes in the connectivity of the list of elements 
  * and store it in the vector localNodes */
void ParFUMShadowArray::collectLocalNodes(int numElements,adaptAdj *elements,CkVec<int> &localNodes){
  for(int i=0;i<numElements;i++){
    //check if the element is local
    CkAssert(elements[i].partID >= 0 && elements[i].localID >= 0);
    if(elements[i].partID == idx){
      int elemType = elements[i].elemType;
      const FEM_Elem &elem = fmMesh->getElem(elemType);
      const int *conn = elem.connFor(elements[i].localID);
      const int numNodes = elem.getNodesPer();
      for(int j=0;j<numNodes;j++){
        int node = conn[j];
        CkAssert(node >= 0);
        localNodes.push_back(node);
      }
    }
  }
  //now we need to uniquify the list of localNodes
  if(localNodes.length() != 0){
    localNodes.quickSort(8);
    int count=0;
    for(int i=1;i<localNodes.length();i++){
      if(localNodes[count] == localNodes[i]){
      }else{
        count++;
        if(i != count){
          localNodes[count] = localNodes[i];
        }
      }
    }
    localNodes.resize(count+1);
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

/** This locks all the IDXL lists for all chunks mentioned in the
    list, as well as the IDXL lists among the chunks not involving the
    primary. In graph theory jargon, it will lock the clique formed by all
    these chunks. This will be a blocking lock. So, the user should use it
    cautiously. The IDXL lock will always be locked on the smaller of the two
    chunks that form the idxl list. 'idxlType' determines which idxl list to
    lock. There are 5 types of idxl lists on a chunk -- shared node, ghost
    send node, ghost recv node, ghost send elem, ghost recv elem (assuming
    only one set of elements). If it does not get a lock it will block, so it
    returns only if the job suceeds. Hence no need for a return value
*/
void ParFUMShadowArray::IdxlLockChunks(int* chkList, int chkListSize, int idxlType){
  //sort the list of chunks
  sort(chkList, chkListSize);
  //now that the list is sorted, call the following function to grab the locks
  for(int i=0; i<chkListSize-1; i++) {
    int size = chkListSize-1-i;
    int *chks = (chkList+i+1);
    if(chkList[i]!=idx) {
      lockChunksMsg *lm = new(size)lockChunksMsg(chks,size,idxlType);
      memcpy(lm->chkList,chks,size*sizeof(int));
      meshSA[chkList[i]].IdxlLockChunksSecondaryRemote(lm);
    }
    else {
      IdxlLockChunksSecondary(chks, size, idxlType);
    }
  }
  return;
}

/** Calls the above function, it is just a remote interface
 */
void ParFUMShadowArray::IdxlLockChunksRemote(lockChunksMsg *lm) {
  IdxlLockChunks(lm->getChks(), lm->getSize(), lm->getType());
  delete lm;
  return;
}

/** The difference between this and the previous function is that this function
    locks only the idxl lists with chunks mentioned in the list of arguments
    and this chunk, it does not bother about locking the idxl lists among
    the set of chunks themselves.
    It should also be noted that this chunk is the minimum index compared
    to the list of chunks passed as arguments
 */
void ParFUMShadowArray::IdxlLockChunksSecondary(int *chkList, int chkListSize, int idxlType) {
  IDXL_List *ll;
  //since this list is sorted, we will lock in this order, helps avoiding deadlocks
  for(int i=0; i<chkListSize; i++) {
    ll = (FindIdxlSide(idxlType))->getIdxlListN(chkList[i]);
    CkAssert(ll!=NULL);
    bool locked = false;
    while(!locked) {
      locked = ll->lockIdxl();
      if(!locked) CthYield();
    }
  }
  return;
}

/** Calls the above function, it is just a remote interface
 */
void ParFUMShadowArray::IdxlLockChunksSecondaryRemote(lockChunksMsg *lm) {
  IdxlLockChunksSecondary(lm->getChks(), lm->getSize(), lm->getType());
  delete lm;
  return;
}



/** This operation just asserts that locks exist on every edge in the
    clique of all chunks passed as arguments and unlocks the chunks.
*/
void ParFUMShadowArray::IdxlUnlockChunks(int* chkList, int chkListSize, int idxlType){
  //sort the list of chunks
  sort(chkList, chkListSize);
  //now that the list is sorted, call the following function to unlock the locks
  for(int i=0; i<chkListSize-1; i++) {
    int size = chkListSize-1-i;
    int *chks = (chkList+i+1);
    if(chkList[i]!=idx) {
      lockChunksMsg *lm = new(size)lockChunksMsg(chks,size,idxlType);
      memcpy(lm->chkList,chks,size*sizeof(int));
      meshSA[chkList[i]].IdxlUnlockChunksSecondaryRemote(lm);
    }
    else {
      IdxlUnlockChunksSecondary(chks, size, idxlType);
    }
  }
  return;
}

/** Calls the above function, it is just a remote interface
 */
void ParFUMShadowArray::IdxlUnlockChunksRemote(lockChunksMsg *lm) {
  IdxlUnlockChunks(lm->getChks(), lm->getSize(), lm->getType());
  delete lm;
  return;
}

/** The difference between this and the previous function is that this function
    unlocks only the idxl lists with chunks mentioned in the list of arguments
    and this chunk, it does not bother about unlocking the idxl lists among
    the set of chunks themselves.
    It should also be noted that this chunk is the minimum index compared
    to the list of chunks passed as arguments
 */
void ParFUMShadowArray::IdxlUnlockChunksSecondary(int *chkList, int chkListSize, int idxlType) {
  IDXL_List *ll;
  //since this list is sorted, we will unlock in this order
  for(int i=0; i<chkListSize; i++) {
    ll = (FindIdxlSide(idxlType))->getIdxlListN(chkList[i]);
    CkAssert(ll!=NULL);
    CkAssert(ll->isLocked());
    ll->unlockIdxl();
  }
  return;
}

/** Calls the above function, it is just a remote interface
 */
void ParFUMShadowArray::IdxlUnlockChunksSecondaryRemote(lockChunksMsg *lm) {
  IdxlUnlockChunksSecondary(lm->getChks(), lm->getSize(), lm->getType());
  delete lm;
  return;
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

#include "ParFUM_SA.def.h"
