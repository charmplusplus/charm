/**
IDXL--C++ communication structures used by idxl library.

Orion Sky Lawlor, olawlor@acm.org, 1/7/2003
*/

#ifndef __CHARM_IDXL_COMM_H
#define __CHARM_IDXL_COMM_H

#include "pup.h"
#include "cklists.h"
#include "ckhashtable.h"
#include "charm.h"


/**
 * IDXL_Share describes how one entity is shared with one other chunk.
 * It lists the chunk and the entity's location in the communication list;
 * this location in the communication list is useful because it may be
 * the *only* way to specify a particular entity to another chunk
 * if we don't have global numbers to identify things with.
 */
class IDXL_Share {
 public:
  int chk; //Chunk we're shared with
  int idx; //Our index in the local comm. list for that chunk
  IDXL_Share(int x=0) {chk=idx=-1;}
  IDXL_Share(int c,int i) :chk(c), idx(i) {}
  void pup(PUP::er &p) {p(chk); p(idx);}
};
PUPmarshall(IDXL_Share)

/**
 * IDXL_Rec lists all the chunks that share an entity.
 */
class IDXL_Rec {
	int entity; //Index of entity (node or element) we describe
	CkVec<IDXL_Share> shares;
	int oldlength;
public:
	IDXL_Rec(int entity_=-1);
	~IDXL_Rec();

	void pup(PUP::er &p);
	
	inline int getEntity(void) const {return entity;}
	inline int getShared(void) const {return shares.size();}
	inline int getChk(int shareNo) const {return shares[shareNo].chk;}
	inline int getIdx(int shareNo) const {return shares[shareNo].idx;}
	bool hasChk(int chk) const {
		for (int i=0;i<getShared();i++)
			if (getChk(i)==chk) return true;
		return false;
	}
	void add(int chk,int idx);
};

/**
 * Map an entity to its IDXL_Rec.  We only bother listing entities
 * that are actually shared with other chunks; completely local entities
 * are not present.
 */
class IDXL_Map {
	CkHashtableT<CkHashtableAdaptorT<int>,IDXL_Rec *> map;
public:
	IDXL_Map();
	~IDXL_Map();

	//Add a IDXL_Rec entry for this entity
	void add(int entity,int chk,int idx);
	
	//Look up this entity's IDXL_Rec.  Returns NULL if entity is not shared.
	const IDXL_Rec *get(int entity) const;
};

/**
 * IDXL_List lists the entities we share with one other chunk. 
 * This list is used to build outgoing and interpret incoming messages--
 * for an outgoing message, the listed entities are copied into the message;
 * for an incoming message, the listed entities are copied out of the message.
 */
class IDXL_List {
	int chunk; //Global number of other chunk	
	CkVec<int> shared; //Local indices of shared entities
	bool lock;
public:
	IDXL_List();
	IDXL_List(int otherchunk);
	~IDXL_List();
	int getDest(void) const {return chunk;}
	int size(void) const {return shared.size();}
	int &operator[](int idx) {return shared[idx]; }
	int operator[](int idx) const {return shared[idx]; }
	const int *getVec(void) const {return &shared[0];}
	bool lockIdxl();
	void unlockIdxl();
	bool isLocked();
	/// We share this local index with this other chunk.
	int push_back(int localIdx);
	bool set(int localIdx, int sharedIdx);
	bool unset(int sharedIdx);
	int exists(int localIdx);
	int get(int sharedIdx);
	void pup(PUP::er &p);
	void sort2d(double *coord);
	void sort3d(double *coord);
};
PUPmarshall(IDXL_List)

/// This class formats a local index for output.
/// This is used, e.g., to print 1-based global numbers 
/// instead of 0-based local numbers during output.
class IDXL_Print_Map {
public:
	virtual void map(int srcIdx) const =0;
};

/**
 * IDXL_Side describes all the shared entities for a given communication
 * direction.
 * It provides both chunk->entities shared with it (comm)
 * and the more rarely used entity->chunks that share it (map)
 */
class IDXL_Side : public CkNoncopyable {
	/// Communication lists indexed by (local) chunk number
	CkPupPtrVec<IDXL_List, CkPupAlwaysAllocatePtr<IDXL_List> > comm; 

	/// Communication lists indexed by local entity number.
	/// Because this is just another form for the data in comm,
	/// this pointer is normally NULL until somebody needs it.
	IDXL_Map *cached_map;
	IDXL_Map &getMap(void);
	
	//Return the Comm_List associated with this chunk, or NULL
	IDXL_List *getListN(int chunk) { 
		for (int i=0;i<comm.size();i++)
			if (comm[i]->getDest()==chunk)
				return comm[i];
		return NULL; 
	}
public:
	IDXL_Side(void);
	void pup(PUP::er &p);
	~IDXL_Side();
	/// Return total number of entries in all our lists
	int total() const;
	
	/// Return the number of different chunks we communicate with
	int size(void) const {return comm.size();}
	/// Return the i'th (local) chunk we communicate with
	const IDXL_List &getLocalList(int idx) const { return *comm[idx]; }
	IDXL_List &setLocalList(int idx) { return *comm[idx]; }
	
	/// Return the local chunk number for this global chunk number
	int findLocalList(int chunk) const {
		for (int i=0;i<comm.size();i++) 
			if (comm[i]->getDest()==chunk)
				return i;
		return -1;
	}
	
	/// Return the IDXL_List associated with this global chunk number
	const IDXL_List &getList(int chunk) const { 
		const IDXL_List *ret=((IDXL_Side *)this)->getListN(chunk);
		if (ret==NULL) CkAbort("FEM> Communication lists corrupted (unexpected message)");
		return *ret; 
	}
	/// Return the IDXL_List for this global chunk, adding if needed
	IDXL_List &addList(int chunk) {
		IDXL_List *ret=getListN(chunk);
		if (ret==NULL) { //Have to add a new list:
			ret=new IDXL_List(chunk);
			comm.push_back(ret);
		}
		return *ret;
	}

	IDXL_List *getIdxlListN(int chunk) {
	  return getListN(chunk);
	}
	
	/// Look up an entity's IDXL_Rec by the entity's local number
	const IDXL_Rec *getRec(int entity) const;
	

	/// The communication lists just changed-- flush any cached information.
	void flushMap(void);
	
	/**
	 * Used in creating comm. lists:
	 *  myLocalNo on (global chunk number) myChunk should be sent to
	 *  hisLocalNo on (global chunk number) hisChunk.
	 */
	void add(int myChunk,int myLocalNo,
		 int hisChunk,int hisLocalNo,IDXL_Side &hisList);
	
	
	/**
	 * Method to clear remove all the IDXL Lists stored in this IDXL_Side
	 * Memory leak ?
	 */
	void clear();
	
	/***
	 * added by Nilesh, needs comments.
	 * */
	bool lockIdxl(int sharedWithChk);
	void unlockIdxl(int sharedWithChk);
	bool isLocked(int sharedWithChk);
	int addNode(int localNo, int sharedWithChk);
	int removeNode(int localNo, int sharedWithChk);
	bool setNode(int localNo, int sharedWithChk, int sharedIdx);
	bool unsetNode(int sharedWithChk, int sharedIdx);
	int existsNode(int localNo, int sharedWithChk);
	int getNode(int sharedWithChk, int sharedIdx);

	void print(const IDXL_Print_Map *idxmap=NULL) const;

	void sort2d(double *coord);

	void sort3d(double *coord);
};

/**
 * IDXL: A two-way communication list.
 */
class IDXL : public CkNoncopyable {
	IDXL_Side *send,*recv;
	IDXL_Side *alloc_send, *alloc_recv;
public:
	/// Use this single list for both send and recv.  Never deletes the list.
	IDXL(IDXL_Side *sendrecv) {
		send=sendrecv; recv=sendrecv;
		alloc_send=alloc_recv=NULL;
	}
	
	/// Use these lists.  Never deletes the lists.
	IDXL(IDXL_Side *send_, IDXL_Side *recv_) {
		send=send_; recv=recv_;
		alloc_send=alloc_recv=NULL;
	}
	
	/// Empty list (used during migration).
	IDXL(void) {
		send=alloc_send=NULL;
		recv=alloc_recv=NULL;
	}
	
	/// Allocate new send list; with recv identical
	void allocateSingle(void) {
		send=alloc_send=new IDXL_Side;
		recv=send; alloc_recv=NULL;
	}
	/// Allocate new, empty send and recv lists:
	void allocateDual(void) {
		send=alloc_send=new IDXL_Side;
		recv=alloc_recv=new IDXL_Side;
	}
	
	void pup(PUP::er &p) {
		int isSendRecv= (send==recv);
		p|isSendRecv;
		if (!send) send=alloc_send=new IDXL_Side;
		send->pup(p);
		if (isSendRecv) {
			recv=send;
		}
		else {
			if (!recv) recv=alloc_recv=new IDXL_Side;
			recv->pup(p);
		}
	}
	~IDXL(void) {
		delete alloc_send; delete alloc_recv;
	}
	
	/// Return true if our send and recv lists are identical
	bool isSingle(void) const {return send==recv;}
	
	IDXL_Side &getSend(void) {return *send;}
	const IDXL_Side &getSend(void) const {return *send;}
	IDXL_Side &getRecv(void) {return *recv;}
	const IDXL_Side &getRecv(void) const {return *recv;}

	void sort2d(double *coord);

	void sort3d(double *coord);
};

#endif
