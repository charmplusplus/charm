/*
 * parCollide: parallel interface part of collision detection system
 * Orion Sky Lawlor, olawlor@acm.org, 4/8/2001
 */
#include "collision.h"
#include "threadCollide.h"
#include "collide.h"

#if 1
//copious annoying debugging output
#  define CM_STATUS(x) x
#else
//No debugging output
#  define CM_STATUS(x) /*emtpy*/
#endif


/********************** threadMgr **********************/
threadedCollideMgr::threadedCollideMgr(CkArrayID voxels)
	:collideMgr(voxels), thisproxy(thisgroup)
{
  setClient(collisionClient,(void *)this);
}

//Maintain contributor registration count
void threadedCollideMgr::registerContributor(int chunkNo) 
{
	collideMgr::registerContributor(chunkNo);
	chunk2contrib.put(chunkNo)=new contribRec(chunkNo);
}
contribRec &threadedCollideMgr::lookupContributor(int chunkNo) 
{
	return *chunk2contrib.getRef(chunkNo);
}
void threadedCollideMgr::unregisterContributor(int chunkNo) 
{
	collideMgr::unregisterContributor(chunkNo);
	delete &lookupContributor(chunkNo);
	chunk2contrib.remove(chunkNo);
}

//Contribute to collision and suspend
void threadedCollideMgr::contribute(int chunkNo,
			int n,const bbox3d *boxes)
{
  contribRec &r=lookupContributor(chunkNo);
  if (r.hasContributed) CkAbort("Multiple collision contributions from one chunk!");
  r.hasContributed=1;
  collideMgr::contribute(chunkNo,n,boxes);
  r.suspend();
}

void threadedCollideMgr::collisionClient(void *param,int nColl,collision *colls)
{
	threadedCollideMgr *m=(threadedCollideMgr *)param;
	m->thisproxy.collisionList(nColl,colls);
}

//Collision list is delivered here on every processor
void threadedCollideMgr::collisionList(int nColl,collision *colls)
{
	CM_STATUS(status("threadedCollideMgr::collisionList"));
	if (nColl!=0) {
		//Figure out which collisions are meant for us and
		// send them to our clients
		int myPe=CkMyPe();
		for (int i=0;i<nColl;i++) {
			const collision &c=colls[i];
			int Amine=(c.A.pe==myPe);
			int Bmine=(c.B.pe==myPe);
			if ((!Amine) && (!Bmine))
				continue;//Collision for another processor
			if (Amine)
				lookupContributor(c.A.chunk).addCollision(c.A,c.B);
			if (Bmine) {
				//Make sure we don't add collision twice
				if (Amine && (c.A.chunk == c.B.chunk))
					continue;
				lookupContributor(c.B.chunk).addCollision(c.B,c.A);
			}
		}
	}
	//Resume all clients
	CM_STATUS(status("threadedCollideMgr waking up clients"));
	CkHashtableIterator *it=chunk2contrib.iterator();
	void *contribPtr;
	while (NULL!=(contribPtr=it->next())) {
		contribRec &r=**(contribRec **)contribPtr;
		r.hasContributed=0; //Ready for next
		r.resume();
	}
}

/********************** C Client API *********************/
static readonly<CkGroupID> tcm_gid;

static threadedCollideMgr *collideMgr(void)
{
  return CProxy_threadedCollideMgr(tcm_gid).ckLocalBranch();
}

//Call this once at system-init time:
CDECL void CollideInit(const double *gridStart,const double *gridSize)
{
  CkArrayID voxels=createColliderVoxels(*(vector3d *)gridStart,
					*(vector3d *)gridSize);
  tcm_gid=CProxy_threadedCollideMgr::ckNew(voxels);
}

//Each chunk should call this once
CDECL void CollideRegister(int chunkNo) {
  collideMgr()->registerContributor(chunkNo);
}
//Chunks call this when leaving a processor
CDECL void CollideUnregister(int chunkNo) {
  collideMgr()->unregisterContributor(chunkNo);
}

//Collide these boxes (boxes[0..6*nBox])
CDECL void Collide(int chunkNo,int nBox,const double *boxes) {
  collideMgr()->contribute(chunkNo,nBox,(const bbox3d *)boxes);
}

//Immediately after a collision, get the number of collisions
CDECL int CollideCount(int chunkNo) {
  return collideMgr()->lookupContributor(chunkNo).getNcoll();
}


static void getCollisionList(int chunkNo,int *out,int indexBase)
{
  contribRec &r=collideMgr()->lookupContributor(chunkNo);
  int n=r.getNcoll();
  collision *in=r.detachCollisions();
  for (int i=0;i<n;i++) {
    out[3*i+0]=in[i].A.number+indexBase;
    out[3*i+1]=in[i].B.chunk+indexBase;
    out[3*i+2]=in[i].B.number+indexBase;
  }
  free(in);	
}

//Immediately after a collision, get the colliding boxes (collisions[0..3*nColl])
CDECL void CollideList(int chunkNo,int *out) {
	getCollisionList(chunkNo,out,0);
}

/******************* f90 Client API *******************/
FDECL void FTN_NAME(COLLIDEINIT,collideinit)
	(const double *gridStart,const double *gridSize) 
{ 
	CollideInit(gridStart,gridSize); 
}

FDECL void FTN_NAME(COLLIDEREGISTER,collideregister) 
	(int *chunkNo)
{
	CollideRegister(*chunkNo-1);
}
FDECL void FTN_NAME(COLLIDEUNREGISTER,collideunregister) 
	(int *chunkNo)
{
	CollideUnregister(*chunkNo-1);
}
FDECL void FTN_NAME(COLLIDE,collide) 
	(int *chunkNo,int *nBoxes,const double *boxes)
{
	Collide(*chunkNo-1,*nBoxes,boxes);
}
FDECL int FTN_NAME(COLLIDECOUNT,collidecount) 
	(int *chunkNo)
{
	return CollideCount(*chunkNo-1);
}
FDECL void FTN_NAME(COLLIDELIST,collidelist) 
	(int *chunkNo, int *arrayLen, int *out)
{
	getCollisionList(*chunkNo-1,out,1);
}

#include "collide.def.h"



