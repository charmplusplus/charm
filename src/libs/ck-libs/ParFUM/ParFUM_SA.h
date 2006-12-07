/** The shadow array class attached to a fem chunk to perform all communication
 *  Author: Nilesh Choudhury
 */

#ifndef __PARFUM_SA_H
#define __PARFUM_SA_H

#include "tcharm.h"
#include "charm++.h"
#include "ParFUM.h"
#include "idxl.h"
#include "ParFUM_internals.h"


class RegionID{
public:
	int chunkID;
	int localID;
 	operator CkHashCode() const {
		return (CkHashCode )(localID + 1<< chunkID); 
	};
	inline bool operator ==(const RegionID &rhs) const {
		return (rhs.chunkID == chunkID) && (rhs.localID == localID);
	}
};
PUPbytes(RegionID);

/** Class that represent a region that is being locked 
 */
class LockRegion {
public:
	RegionID myID;
	CkVec<int> localNodes;
	CkVec<int> sharedIdxls;
	CkHashtableT<CkHashtableAdaptorT<int>, CkVec<adaptAdj> *> remoteElements;
	CthThread tid;
	int numReplies;
	bool success;
	~LockRegion(){
		CkHashtableIterator *iter = remoteElements.iterator();
		while(iter->hasNext()){
			CkVec<adaptAdj> *list = (CkVec<adaptAdj> *)iter->next();
			delete list;
		}
	}
};

#include "ParFUM_SA.decl.h"

///The shadow array attached to a fem chunk to perform all communication
/** This is a shadow array that should be used by all operations
    which want to perform any operations on meshes and for that
    purpose want to communicate among chunks on different processors
 */
class ParFUMShadowArray : public CBase_ParFUMShadowArray {
 private:
  ///Total number of chunks
  int numChunks;
  ///Index of this chunk (the chunk this is attached to)
  int idx;
  ///The Tcharm pointer to set it even outside the thread..
  TCharm *tc;
  ///The proxy for the current Tcharm object
  CProxy_TCharm tproxy;
  ///cross-pointer to the fem mesh on this chunk
  FEM_Mesh *fmMesh;
  ///Deprecated: used to lock this chunk

	CkHashtableT<CkHashtableAdaptorT<RegionID>,LockRegion *> regionTable;
	int regionCount;

 public:
  ///constructor
  ParFUMShadowArray(int s, int i);
  ///constructor for migration
  ParFUMShadowArray(CkMigrateMessage *m);
  ///destructor
  ~ParFUMShadowArray();

  ///Pup to transfer this object's data
  void pup(PUP::er &p);
  ///This function is overloaded, it is called on this object just after migration
  void ckJustMigrated(void);

  ///Get number of chunks
  int getNumChunks(){return numChunks;}
  ///Get the index of this chunk
  int getIdx(){return idx;}
  ///Get the pointer to the mesh object
  FEM_Mesh *getfmMesh(){return fmMesh;}

  ///Initialize the mesh pointer for this chunk
  void setFemMesh(FEMMeshMsg *m);

  ///Sort this list of numbers in increasing order
  void sort(int *chkList, int chkListSize);
  
	//Lock all the nodes belonging to these elements
	//Some of the elements might be remote
	//You also have to lock idxls with all those chunks with which
	//any of the nodes are shared
	bool lockRegion(int numElements,adaptAdj *elements,RegionID *regionID);

	void unlockRegion(RegionID regionID);
	
	void collectLocalNodes(int numElements,adaptAdj *elements,CkVec<int> &localNodes);
	bool lockLocalNodes(LockRegion *region);
	bool lockSharedIdxls(LockRegion *region);
	void lockRegionForRemote(RegionID regionID,int *sharedIdxls,int numSharedIdxls,adaptAdj *elements,int numElements);
	void lockReply(int remoteChunk,RegionID regionID,bool success);
	void unlockRegion(LockRegion *region);
	void unlockLocalNodes(LockRegion *region);
	void unlockSharedIdxls(LockRegion *region);
	void unlockForRemote(RegionID regionID);
	void unlockReply(int remoteChunk,RegionID regionID);


	void freeRegion(LockRegion *region);


	
	
	///Translates the sharedChk and the idxlType to the idxl side
  FEM_Comm *FindIdxlSide(int idxlType);


  ///Add this 'localId' to this idxl list and return the shared entry index
  int IdxlAddPrimary(int localId, int sharedChk, int idxlType);
  ///Add this 'sharedIdx' to this idxl list
  bool IdxlAddSecondary(int localId, int sharedChk, int sharedIdx, int idxlType);
  ///Remove this 'localId' from this idxl list and return the shared entry index
  int IdxlRemovePrimary(int localId, int sharedChk, int idxlType);
  ///Remove the entry in this idxl list at index 'sharedIdx'
  bool IdxlRemoveSecondary(int sharedChk, int sharedIdx, int idxlType);

  ///Lookup this 'localId' in this idxl list and return the shared entry index
  int IdxlLookUpPrimary(int localId, int sharedChk, int idxlType);
  ///Return the localIdx at this 'sharedIdx' on this idxl list
  int IdxlLookUpSecondary(int sharedChk, int sharedIdx, int idxlType);
};

///This is a message which packs all the chunk indices together
class lockChunksMsg : public CMessage_lockChunksMsg {
 public:
  ///list of chunks
  int *chkList;
  ///number of chunks in the list
  int chkListSize;
  ///type of idxl list
  int idxlType;

 public:
  lockChunksMsg(int *c, int s, int type) {
    chkListSize = s;
    idxlType = type;
  }

  ~lockChunksMsg() {
    ///if(chkList!=NULL) delete chkList;
  }

  int *getChks() {
    return chkList;
  }

  int getSize() {
    return chkListSize;
  }

  int getType() {
    return idxlType;
  }
};

#endif

