/*
 * threadCollide: threaded interface to collision detection library
 * Orion Sky Lawlor, olawlor@acm.org, 7/19/2001
 */
#ifndef __OSL_THREADCOLLIDE_H
#define __OSL_THREADCOLLIDE_H

#include "parCollide.h"
#include "collide.decl.h"

//This class describes one contributor of objects to a collision
class contribRec {
	CthThread tid; // waiting thread, 0 if no one is waiting
	int chunk; //Contributor's chunk number
	growableBufferT<collision> colls; //Accumulated collisions
public:
	int hasContributed;

	contribRec(int chunk_)
		:tid(0), chunk(chunk_), hasContributed(0)
		{ }
	void suspend(void) {
		tid=CthSelf();
		CthSuspend();
	}
	void resume(void) {
		CthAwaken(tid);
		tid=0;
	}
	void addCollision(const globalObjID &a,const globalObjID &b) {
		colls.push_back(collision(a,b));
	}
	int getNcoll(void) const {return colls.length();}
	collision *detachCollisions(void) {
		return colls.detachBuffer();
	}
};

class threadedCollideMgr : public collideMgr
{
	CProxy_threadedCollideMgr thisproxy;
	void status(const char *msg) {
		CkPrintf("TCMgr pe %d> %s\n",CkMyPe(),msg);
	}

	//Map chunk number to contributor record
	CkHashtableT<CkHashtableAdaptorT<int>,contribRec*> chunk2contrib;

	static void collisionClient(void *param,int nColl,collision *colls);
public:
	threadedCollideMgr(CkArrayID voxels);

	//Maintain contributor registration count
	void registerContributor(int chunkNo);
	contribRec &lookupContributor(int chunkNo);
	void unregisterContributor(int chunkNo);

	//Contribute to collision and suspend
	void contribute(int chunkNo,
			int n,const bbox3d *boxes);
	
	//Collision list is delivered here on every pe--
	// separate collision list per chunk and resume
	void collisionList(int nColl,collision *colls);
};

/************** Client API: *****************/

#include "charm-api.h"

//Call this once at system-init time:
CDECL void CollideInit(const double *gridStart,const double *gridSize);

//Each chunk should call this when created/arriving
//  Note that *everybody* has to call this before *anybody* collides!
CDECL void CollideRegister(int chunkNo);
//Chunks call this when leaving a processor
CDECL void CollideUnregister(int chunkNo);

//Collide these boxes (boxes[0..6*nBox])
CDECL void Collide(int chunkNo,int nBox,const double *boxes);
//Immediately after a collision, get the number of collisions
CDECL int CollideCount(int chunkNo);
//Immediately after a collision, get the colliding objects:
// [0] is local object ID; [1] is remote chunk; [2] is remote chunk ID
//  Total of (collisions[0..3*nColl])
CDECL void CollideList(int chunkNo,int *collisions);

#endif //def(thisHeader)






