/*
 * parCollide: parallel interface part of collision detection system
 * Orion Sky Lawlor, olawlor@acm.org, 4/8/2001
 */
#ifndef __OSL_PARCOLLIDE_H
#define __OSL_PARCOLLIDE_H

#include "parCollide.decl.h"

/******************** objListMsg *********************
A pile of objects sent to a collision voxel.
Declared as a "packed" message
type, which converts the message to a flat byte array
only when needed (for sending across processors).
*/
class objListMsg : public CMessage_objListMsg
{
public:
	class returnReceipt {
		CkGroupID gid;
		int onPE;
	public:
		returnReceipt() {}
		returnReceipt(CkGroupID gid_,int onPE_) :gid(gid_),onPE(onPE_) {}
		void send(void);
	};
private:
	bool isHeapAllocated;
	returnReceipt receipt;
	
	int n;
	crossObjRec *obj; //Bounding boxes & IDs
	
	void freeHeapAllocated();
public:
	objListMsg() :isHeapAllocated(false) {}
	
	//Hand control of these arrays over to this message,
	// which will delete them when appropriate.
	objListMsg(int n_,crossObjRec *obj_,
		const returnReceipt &receipt_);
	~objListMsg() {freeHeapAllocated();}
	
	void sendReceipt(void) {receipt.send();}
	
	int getObjects(void) const {return n;}
	const crossObjRec &getObj(int i) const {return obj[i];}
	const bbox3d &bbox(int i) const {return obj[i].box;}
	
	static void *pack(objListMsg *m);
	static objListMsg *unpack(void *m);
};

/********************* syncReductionMgr *****************
A group to synchronize on some event across the machine.
Maintains a reduction tree and waits for the "advance"
method to be called from each processor.  To handle 
non-autonomous cases, calls "pleaseAdvance" when an advance
is first expected from that PE.
*/
class syncReductionMgr : public Group 
{
	CProxy_syncReductionMgr thisproxy;
	void status(const char *msg) {
		CkPrintf("SRMgr pe %d> %s\n",CkMyPe(),msg);
	}
	//Describes the reduction tree
	int onPE;
	enum {TREE_WID=4};
	int treeParent;//Parent in reduction tree
	int treeChildStart,treeChildEnd;//First and last+1 child
	int nChildren;//Number of children in the reduction tree
	void startStep(int stepNo,bool withProd);
	
	//State data
	int stepCount;//Increments by one every reduction, from zero
	bool stepFinished;//prior step complete
	bool localFinished;//Local advance called
	int childrenCount;//Number of tree children in delivered state
	void tryFinish(void);//Try to finish reduction

protected:
	//This is called by subclasses
	void advance(void);
	//This is offered for subclasses's optional use
	virtual void pleaseAdvance(void);
	//This is called on PE 0 once the reduction is finished
	virtual void reductionFinished(void);
public:
	int getStepCount(void) const {return stepCount;}
	syncReductionMgr();
	
	//Called by parent-- will you contribute?
	void childProd(int stepCount);
	//Called by tree children-- me and my children are finished
	void childDone(int stepCount);
};

/*********************** collideMgr **********************
A group that synchronizes the collision detection process.
A single collision operation consists of:
	-collect contributions from clients
	-aggregate the contributions
	-send off triangle lists to voxels
	-wait for replies from the voxels indicating sucessful delivery
	-collideMgr reduction to make sure everybody's messages are delivered
	-root broadcasts startCollision to collideVoxel array
	-collideVoxel array reduces over collisionLists
	-resulting summed collisionList is delivered on processor 0
*/

class collideMgr : public syncReductionMgr
{
	CProxy_collideMgr thisproxy;
public:
	//Set the client function to call with collision data
	typedef void (*collisionClientFn)(void *param,int nColl,collision *colls);
private:
	void status(const char *msg) {
		CkPrintf("CMgr pe %d> %s\n",CkMyPe(),msg);
	}
	collisionClientFn clientFn;void *clientParam;
	
	int nContrib;//Number of registered contributors
	int contribCount;//Number of contribute calls given this step

	collisionAggregator aggregator;
	CProxy_collideVoxel voxelProxy;
	int msgsSent;//Messages sent out to voxels
	int msgsRecvd;//Return-receipt messages received from voxels
	void tryAdvance(void);
	static void reductionClient(void *param,int dataSize,void *data);
protected:
	//Check if we're barren-- if so, advance now
	virtual void pleaseAdvance(void);
	//This is called on PE 0 once the voxel send reduction is finished
	virtual void reductionFinished(void);
public:
	collideMgr(CkArrayID voxels);

	//Use this routine on node 0 to report collisions
	void setClient(collisionClientFn clientFn_,void *clientParam_)
	  {clientFn=clientFn_; clientParam=clientParam_; }
	
	//Maintain contributor registration count
	void registerContributor(int chunkNo);
	void unregisterContributor(int chunkNo);
	
	//Clients call this to contribute their objects
	void contribute(int chunkNo,
		int n,const bbox3d *boxes);
	
	//voxelAggregators deliver messages to voxels via this bottleneck
	void sendVoxelMessage(const gridLoc3d &dest,
		int n,crossObjRec *obj);
	
	//collideVoxels send a return receipt here
	void voxelMessageRecvd(void);
};

/********************** collideVoxel ********************
A sparse 3D array that represents a region of space where
collisions may occur.  Each step it accumulates triangle
lists and then computes their intersection
*/

class collideVoxel : public ArrayElement3D
{
	growableBufferT<objListMsg *> msgs;
	bbox3d territory;
	void status(const char *msg);
	void emptyMessages();
	void collide(collisionList &dest);
public:
	collideVoxel();
	collideVoxel(CkMigrateMessage *m);
	~collideVoxel();
	void pup(PUP::er &p);
	
	void add(objListMsg *msg);
	void startCollision(void);
};


/*  External Interface:   */
//Create a collider group to contribute triangles to.  
// Should be called at init. time on node 0
CkGroupID createCollider(const vector3d &gridStart,const vector3d &gridSize,
	collideMgr::collisionClientFn client,void *clientParam);

//Create collider voxel array
CkArrayID createColliderVoxels(const vector3d &gridStart,const vector3d &gridSize);

#endif //def(thisHeader)
