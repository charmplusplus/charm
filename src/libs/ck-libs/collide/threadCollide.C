/*
 * threadCollide: threaded interface to Collision detection library
 * Orion Sky Lawlor, olawlor@acm.org, 7/19/2001
 */
#include "collidecharm_impl.h"
#include "collidecharm.h"
#include "collidec.h"
#include "tcharm.h"
#include "collide.decl.h"

#define COLLIDE_TRACE 0
#if COLLIDE_TRACE
  define TRACE(x) ckout<<"["<<CkMyPe()<<"] "<<x<<endl;
#else
#  define TRACE(x) /* empty */
#endif

/* TCharm Semaphore ID for collision startup */
#define COLLIDE_TCHARM_SEMAID 0x0C0771DE /* _COLLIDE */

class threadCollide;

/**
Collision message: just a list of collisions.
*/
class threadCollisions : public CMessage_threadCollisions {
public:
	int src; /* processor number of source group */
	int nColls; /* number of collision records below */
	Collision *colls; /* points into message body */
};

/**
Threaded collision group--collects collisions as they come
from voxels, and sends the collisions to the source chunks.
*/
class threadCollideMgr : public CBase_threadCollideMgr
{
	//Map chunk number to contributor (big, but fast indexing scheme)
	CkVec<threadCollide *> contrib;
	inline threadCollide *lookup(int chunkNo) {
		threadCollide *ret=contrib[chunkNo];
#if CMK_ERROR_CHECKING
		if (ret==NULL) CkAbort("threadCollideMgr can't find contributor");
#endif
		return ret;
	}
	
	//Temporarily store collisions to be sent to each processor:
	CkVec<CollisionList> toPE;
	
	//Temporarily store collisions sent in from each processor:
	CkVec<threadCollisions *> fromPE;
	
	//Counter for collisions from remote processors:
	int nRemote;
	
	//Cached version of CkMyPe
	int myPe;
	
public:
	threadCollideMgr(void) 
		:toPE(CkNumPes()), fromPE(CkNumPes())
	{
		for (int p=0;p<CkNumPes();p++) fromPE[p]=0;
		nRemote=0;
		myPe=CkMyPe();
	}
	
	/// Maintain contributor lists:
	void registerContributor(threadCollide *chunk,int chunkNo) 
	{
		while (contrib.size()<=chunkNo) contrib.push_back(0);
		contrib[chunkNo]=chunk;
	}
	void unregisterContributor(int chunkNo) {
		contrib[chunkNo]=NULL;
	}
	
	/// collideClient interface (called by voxels)
	/// Splits up collisions by destination PE
	void collisions(ArrayElement *src,int step,CollisionList &colls);
	
	/// All voxels have now reported their collisions:
	///  Send off the accumulated collisions to each destination PE
	void sendRemote(CkReductionMsg *m);
	
	/// Accept and buffer these remote collisions
	void remoteCollisions(threadCollisions *m);
	
	/// Add these remote collisions to each local chunk
	void sift(int nColl,const Collision *colls);
};

/**
Threaded collision client array--provides interface between
threadCollideMgr and API routines.
*/
class threadCollide : public TCharmClient1D {
	typedef TCharmClient1D super;
	// Outgoing collision requests:
	CollideHandle collide;
	// Incoming collision lists:
	CProxy_threadCollideMgr mgr;
protected:
	virtual void setupThreadPrivate(CthThread th) {}
public:
	growableBufferT<Collision> colls; //Accumulated Collisions
	
	threadCollide(const CProxy_TCharm &threads,
		const CProxy_threadCollideMgr &mgr_,
    		const CollideHandle &collide_) 
		:super(threads), mgr(mgr_), collide(collide_)
	{
		arriving();
		/// Wake up the blocked thread in COLLIDE_Init
		thread->semaPut(COLLIDE_TCHARM_SEMAID,this);
	}
	threadCollide(CkMigrateMessage *m) :super(m) {}
	
	void arriving(void) {
		CollideRegister(collide,thisIndex);
		mgr.ckLocalBranch()->registerContributor(this,thisIndex);
	}
	void pup(PUP::er &p) {
		super::pup(p);
		p|mgr;
		p|collide;
	}
	void ckJustMigrated(void) {
		super::ckJustMigrated();
		arriving();
	}
	void leaving(void) {
		CollideUnregister(collide,thisIndex);
		mgr.ckLocalBranch()->unregisterContributor(thisIndex);
	}
	~threadCollide() {
		leaving();
	}
	inline const CkArrayID &getArrayID(void) const {return thisArrayID;}
	
	
	/// Contribute to Collision and suspend the caller
	void contribute(int n,const bbox3d *boxes,const int *prio) 
	{
		CollideBoxesPrio(collide,thisIndex,n,boxes,prio);
		thread->suspend(); //Will be resumed by call to resultsDone()
	}
	
	/// No more collisions will arrive this step.
	void resultsDone(void) {
		thread->resume();
	}
};


/// collideClient interface (called by voxels)
/// Splits up collisions by destination PE
void threadCollideMgr::collisions(ArrayElement *src,int step,
				  CollisionList &colls) {
  // Do a fake reduction, so we'll know when all voxels have reported:
  src->contribute(0,0,CkReduction::sum_int,
		  CkCallback(CkIndex_threadCollideMgr::sendRemote(0),thisProxy));
  
  // Split out this voxel's contribution
  int i=0, n=colls.size();
  static int count=0;
  
  TRACE("Voxel contributes "<<n<<" collisions")
    
    //printf("COLLIDE: Total collisions contributed so far: %d\n", count+=n);
    for (i=0;i<n;i++) {
      const Collision &c=colls[i];
      toPE[c.A.pe].push_back(c);
      if (c.B.pe!=c.A.pe) { //Report collision to both processors
	Collision cB(c.B,c.A); //Swap so B is listed first
	toPE[c.B.pe].push_back(cB);
      }
    }
}

/// Destroy this collision list, and create a message
/// from it.
threadCollisions *listToMessage(CollisionList &l) 
{
	int n=l.size();
	Collision *c=l.detachBuffer();
	threadCollisions *m=new (n,0) threadCollisions;
	m->nColls=n;
	for (int i=0;i<n;i++) m->colls[i]=c[i];
	free(c);
	return m;
}

/// All voxels have now reported their collisions:
///  Send off the accumulated collisions to each destination PE
void threadCollideMgr::sendRemote(CkReductionMsg *m) {
	// FIXME: optimize this all-to-all
	int p,n=CkNumPes();
	for (p=0;p<n;p++) { // Loop over destination processors:
		TRACE("Sending "<<toPE[p].size()<<" collisions to "<<p)
		threadCollisions *m=listToMessage(toPE[p]);
		m->src=myPe;
		if (p==myPe) /* local */
			remoteCollisions(m);
		else /* remote */
			thisProxy[p].remoteCollisions(m);
	}
}

/// Accept these remote collisions
void threadCollideMgr::remoteCollisions(threadCollisions *m) {
	/*
	Subtle: to guarantee that matching collisions are presented 
	in the same order everywhere, we order each chunk's collisions
	by reporting (source) processor.  We do this without a sort by
	buffering, then "sifting" each collision in the proper order.
	*/
	
	// Just buffer this message 
	// (FIXME: if it's the one we're waiting for, sift it right away)
	if (fromPE[m->src]!=NULL) 
		CkAbort("threadCollideMgr::remoteCollisions unexpected message");
	fromPE[m->src]=m;
	
	// See if we're done yet
	if (++nRemote==CkNumPes()) 
	{	
		// Sift out our collisions to each array element
		TRACE("Sifting collisions out to each array element")
		int p,n=CkNumPes();
		for (p=0;p<n;p++) {
			sift(fromPE[p]->nColls,fromPE[p]->colls);
			delete fromPE[p]; fromPE[p]=NULL;
		}
		
		// Get ready for the next step
		nRemote=0;
		
		// Tell all our array elements that the results are now in
		for (int i=0;i<contrib.size();i++)
			if (contrib[i])
				contrib[i]->resultsDone();
	}
}

/// Add these remote collisions to each local chunk
void threadCollideMgr::sift(int nColl,const Collision *colls) 
{
	for (int i=0;i<nColl;i++) {
		const Collision &c=colls[i];
#if CMK_ERROR_CHECKING
		if (c.A.pe!=myPe) CkAbort("Should only have local collisions now");
#endif
		lookup(c.A.chunk)->colls.push_back(c);
		if (c.A.pe==c.B.pe && c.A.chunk!=c.B.chunk) 
		{ //Report this collision to both local chunks:
			Collision cB(c.B,c.A); //Swap so B is listed first
			lookup(c.B.chunk)->colls.push_back(cB);
		}
			
	}
}

/*************** API Routines *****************/
//Declare this at the start of every API routine:
#define COLLIDEAPI(routineName) TCHARM_API_TRACE(routineName,"collide")


int TCHARMLIB_Get_rank(TCharm *tc,int mpi_comm) {
	// FIXME: call AMPI_Get_rank if given a real AMPI communicator
	return tc->getElement();
}
CkArrayOptions TCHARMLIB_Bound_array(TCharm *tc,int mpi_comm) {
	// FIXME: bind to AMPI if given a real AMPI communicator
	CkArrayOptions opts(tc->getNumElements());
	opts.bindTo(tc->getProxy());
	return opts;
}

CDECL collide_t COLLIDE_Init(int mpi_comm,
	const double *gridStart,const double *gridSize)
{
	COLLIDEAPI("COLLIDE_Init");
	TCharm *tc=TCharm::get();
	if (tc==NULL) CkAbort("Must call COLLIDE_Init from driver");
	int rank=TCHARMLIB_Get_rank(tc,mpi_comm);
	if (rank==0) { // I am the master: I must create the array
	  CkArrayOptions opts(TCHARMLIB_Bound_array(tc,mpi_comm));
	  CProxy_threadCollideMgr client=
	    CProxy_threadCollideMgr::ckNew();
	  CollideGrid3d gridMap(*(vector3d *)gridStart, *(vector3d *)gridSize);
	  CollideHandle collide=
	    CollideCreate(gridMap,client);
	  CProxy_threadCollide::ckNew(tc->getProxy(),client,collide,opts);
	  // As array elements are created, they will
	  //  do tc->semaPut(COLLIDE_TCHARM_SEMAID,this);
	}
	// Block until the collision objects are all created:
	threadCollide *coll=(threadCollide *)tc->semaGet(COLLIDE_TCHARM_SEMAID);
	// hideous: extract the groupID's "idx" to use as a "collide_t"
	CkGroupID g=coll->getArrayID();
	collide_t c=g.idx;
	return c;
}
FORTRAN_AS_C_RETURN(int,COLLIDE_INIT,COLLIDE_Init,collide_init,
	(int *comm,double *s,double *e), (*comm,s,e))

threadCollide *COLLIDE_Lookup(collide_t c) {
	CkGroupID g; g.idx=c;
	CProxy_threadCollide coll(g);
	threadCollide *ret=coll[TCharm::get()->getElement()].ckLocal();
#if CMK_ERROR_CHECKING
	if (ret==NULL) CkAbort("COLLIDE can't find its collision array element.");
#endif	
	return ret;
}

CDECL void COLLIDE_Boxes(collide_t c,int nBox,const double *boxes)
{
	COLLIDEAPI("COLLIDE_Boxes");
	COLLIDE_Lookup(c)->contribute(nBox,(const bbox3d *)boxes,NULL);
}
FORTRAN_AS_C(COLLIDE_BOXES,COLLIDE_Boxes,collide_boxes,
	(int *c,int *n,double *box),(*c,*n,box))

CDECL void COLLIDE_Boxes_prio(collide_t c,int nBox,const double *boxes,const int *prio)
{
	COLLIDEAPI("COLLIDE_Boxes_prio");
	COLLIDE_Lookup(c)->contribute(nBox,(const bbox3d *)boxes,prio);
}
FORTRAN_AS_C(COLLIDE_BOXES_PRIO,COLLIDE_Boxes_prio,collide_boxes_prio,
	(int *c,int *n,double *box,int *prio),(*c,*n,box,prio))

CDECL int COLLIDE_Count(collide_t c) {
	COLLIDEAPI("COLLIDE_Count");
	return COLLIDE_Lookup(c)->colls.size();
}
FORTRAN_AS_C_RETURN(int,COLLIDE_COUNT,COLLIDE_Count,collide_count,
	(int *c),(*c))

static void getCollisionList(collide_t c,int *out,int indexBase) {
	growableBufferT<Collision> &colls=COLLIDE_Lookup(c)->colls;
	int i,n=colls.size();
	Collision *in=colls.detachBuffer();
	for (i=0;i<n;i++) {
		out[3*i+0]=in[i].A.number+indexBase;
		out[3*i+1]=in[i].B.chunk+indexBase;
		out[3*i+2]=in[i].B.number+indexBase;
	}
	free(in);
}

CDECL void COLLIDE_List(collide_t c,int *out) {
	COLLIDEAPI("COLLIDE_List");
	getCollisionList(c,out,0);
}
FDECL void FTN_NAME(COLLIDE_LIST,collide_list)(collide_t *c,int *out) {
	COLLIDEAPI("COLLIDE_List");
	getCollisionList(*c,out,1);
}

CDECL void COLLIDE_Destroy(collide_t c) {
	COLLIDEAPI("COLLIDE_Destroy");
	/* FIXME: delete entire array */
}
FORTRAN_AS_C(COLLIDE_DESTROY,COLLIDE_Destroy,collide_destroy,
	(int *c),(*c))

#include "collide.def.h"
