/*
 * parCollide: parallel interface part of collision detection system
 * Orion Sky Lawlor, olawlor@acm.org, 4/8/2001
 */
#include "collision.h"
#include "parCollide.h"

#if 0
//copious annoying debugging output
#  define SRM_STATUS(x) x
#  define CM_STATUS(x) x
#  define CC_STATUS(x) x
#  define COL_STATUS(x) x
#else
//No debugging output
#  define SRM_STATUS(x) /*emtpy*/
#  define CM_STATUS(x) /*emtpy*/
#  define CC_STATUS(x) /*emtpy*/
#  define COL_STATUS(x) /*emtpy*/
#endif

/*************** External Interface *****************/

//Create the collider voxel array
CkArrayID createColliderVoxels(const vector3d &gridStart,const vector3d
&gridSize)
{
	gridMap.init(gridStart,gridSize);
	
	//Build the parallel objects (should be via a subroutine call)
	CkArrayID voxels=CProxy_collideVoxel::ckNew();
	CProxy_collideVoxel(voxels).doneInserting();	
	return voxels;
}


//Create a collider group to contribute to.  
// Should be called at init. time on node 0
CkGroupID createCollider(const vector3d &gridStart,const vector3d &gridSize,
	collideMgr::collisionClientFn client,void *clientParam)
{
	CkArrayID voxels=createColliderVoxels(gridStart,gridSize);
	CkGroupID ret=CProxy_collideMgr::ckNew(voxels);
	CProxy_collideMgr(ret).ckLocalBranch()->
		setClient(client,clientParam);
	return ret;
}

/************** gridMapping ***********/
gridMapping gridMap;

static void testMapping(gridMapping &map,int axis,
	double origin,double size)
{
	int m1=map.world2grid(axis,rSeg1d(-1.0e20,origin-0.6*size)).getMax();
	int m2=map.world2grid(axis,rSeg1d(-1.0e20,origin-0.4*size)).getMax();
	int m3=map.world2grid(axis,rSeg1d(-1.0e20,origin-0.1*size)).getMax();
	int e1=map.world2grid(axis,rSeg1d(-1.0e20,origin+0.1*size)).getMax();
	int e2=map.world2grid(axis,rSeg1d(-1.0e20,origin+0.4*size)).getMax();
	int e3=map.world2grid(axis,rSeg1d(-1.0e20,origin+0.6*size)).getMax();
	int e4=map.world2grid(axis,rSeg1d(-1.0e20,origin+0.9*size)).getMax();
	int p1=map.world2grid(axis,rSeg1d(-1.0e20,origin+1.1*size)).getMax();
	int p2=map.world2grid(axis,rSeg1d(-1.0e20,origin+1.4*size)).getMax();
	int p3=map.world2grid(axis,rSeg1d(-1.0e20,origin+1.6*size)).getMax();
	int p4=map.world2grid(axis,rSeg1d(-1.0e20,origin+1.9*size)).getMax();
	if (m1!=m2 || m1!=m3) 
		CkAbort("gridMapping::Grid initialization error (m)!\n");
	if (e1!=e2 || e1!=e3 || e1!=e4) 
		CkAbort("gridMapping::Grid initialization error (e)!\n");
	if (p1!=p2 || p1!=p3 || p1!=p4) 
		CkAbort("gridMapping::Grid initialization error (p)!\n");
}

void gridMapping::init(const vector3d &Norigin,//Grid voxel corner 0,0,0
                const vector3d &desiredSize)//Size of each voxel
{
	origin=Norigin;
        for (int i=0;i<3;i++) {
#if COLLISION_USE_FLOAT_HACK
                //Compute gridhack shift-- round grid size down 
                //  to nearest smaller power of two
                double s=(1<<20);
                while (s>(1.25*desiredSize[i])) s*=0.5;
                sizes[i]=s;
                hakShift[i]=(1.5*(1<<23)-0.5)*s-origin[i];
                float o=(float)(hakShift[i]);
                hakStart[i]=*(int *)&o;
#else
                sizes[i]=desiredSize[i];
#endif
                scales[i]=1.0/sizes[i];
                testMapping(*this,i,origin[i],sizes[i]);
        }
        
}

/******************** objListMsg **********************/
objListMsg::objListMsg(
		int n_,crossObjRec *obj_, 
		const returnReceipt &receipt_) 
	:isHeapAllocated(true),receipt(receipt_),
	 n(n_),obj(obj_)
{}

void objListMsg::freeHeapAllocated()
{
	if (!isHeapAllocated) return;
	free(obj);obj=NULL;
}

//This gives the byte offsets & sizes needed by each field
#define objListMsg_OFFSETS \
	int offM=0; \
	int cntM=sizeof(objListMsg); \
	int offB=offM+ALIGN8(cntM); \
	int cntB=sizeof(crossObjRec)*m->n;\

void *objListMsg::pack(objListMsg *m)
{
	objListMsg_OFFSETS
	int offEnd=offB+cntB;
	//Allocate a new message buffer and copy our fields into it
	void *buf = CkAllocBuffer(m, offEnd);
	char *cbuf=(char *)buf;
	memcpy(cbuf+offM,m,cntM);
	memcpy(cbuf+offB,m->obj,cntB);
	delete m;
	return buf;
}

objListMsg *objListMsg::unpack(void *buf)
{
	//Unpack ourselves in-place from the buffer; with
	// our arrays pointing into the allocated message data.
	objListMsg *m = new (buf) objListMsg;
	char *cbuf=(char *)buf;
	objListMsg_OFFSETS
	m->obj=(crossObjRec *)(cbuf+offB);
	return m;
}

/********************* syncReductionMgr ******************/
syncReductionMgr::syncReductionMgr()
  :thisproxy(thisgroup)
{
	stepCount=-1;
	stepFinished=true;
	localFinished=false;
	
	//Set up the reduction tree
	onPE=CkMyPe();
	if (onPE==0) treeParent=-1;
	else treeParent=(onPE-1)/TREE_WID;
	treeChildStart=(onPE*TREE_WID)+1;
	treeChildEnd=treeChildStart+TREE_WID;
	if (treeChildStart>CkNumPes()) treeChildStart=CkNumPes();
	if (treeChildEnd>CkNumPes()) treeChildEnd=CkNumPes();
	nChildren=treeChildEnd-treeChildStart;
}
void syncReductionMgr::startStep(int stepNo,bool withProd)
{
	SRM_STATUS(status("syncReductionMgr::startStep"));
	if (stepNo<1+stepCount) return;//Already started
	if (stepNo>1+stepCount) CkAbort("Tried to start SRMgr step from future\n");
	stepCount++;
	stepFinished=false;
	localFinished=false;
	childrenCount=0;
	if (nChildren>0)
	  for (int i=0;i<TREE_WID;i++) 
	    if (treeChildStart+i<CkNumPes())
	      thisproxy[treeChildStart+i].childProd(stepCount);
	if (withProd)
		pleaseAdvance();//Advise subclass to advance
}

void syncReductionMgr::advance(void)
{
	SRM_STATUS(status("syncReductionMgr::advance"));
	if (stepFinished) startStep(stepCount+1,false);
	localFinished=true;
	tryFinish();
}

void syncReductionMgr::pleaseAdvance(void)
	{ /*Child advisory only*/ }

//This is called on PE 0 once the reduction is finished
void syncReductionMgr::reductionFinished(void)
	{ /*Child use only */ }

void syncReductionMgr::tryFinish(void) //Try to finish reduction
{
	SRM_STATUS(status("syncReductionMgr::tryFinish"));
	if (localFinished && (!stepFinished) && childrenCount==nChildren) 
	{
		stepFinished=true;
		if (treeParent!=-1)
			thisproxy[treeParent].childDone(stepCount);
		else
			reductionFinished();
	}
}
//Called by parent-- will you contribute?
void syncReductionMgr::childProd(int stepCount)
{
	SRM_STATUS(status("syncReductionMgr::childProd"));
	if (stepFinished) startStep(stepCount,true);
	tryFinish();
}
//Called by tree children-- me and my children are finished
void syncReductionMgr::childDone(int stepCount)
{
	SRM_STATUS(status("syncReductionMgr::childDone"));
	if (stepFinished) startStep(stepCount,true);
	childrenCount++;
	tryFinish();
}


/*********************** collideMgr ***********************/
collideMgr::collideMgr(CkArrayID voxels)
	:thisproxy(thisgroup), aggregator(this), voxelProxy(voxels)
{
	nContrib=0;
	contribCount=0;
	msgsSent=msgsRecvd=0;
	voxelProxy.setReductionClient(reductionClient,(void *)this);
	clientFn=NULL;clientParam=NULL;
}

//Maintain contributor registration count
void collideMgr::registerContributor(int chunkNo) 
{
	CM_STATUS(status("Contributor register"));
	nContrib++;
}
void collideMgr::unregisterContributor(int chunkNo) 
{
	CM_STATUS(status("Contributor unregister"));
	nContrib--;
}

//Clients call this to contribute their triangle lists
void collideMgr::contribute(int chunkNo,
	int n,const bbox3d *boxes)
{
	CM_STATUS(status("collideMgr::contribute"));
	aggregator.aggregate(CkMyPe(),chunkNo,n,boxes);
	if (++contribCount==nContrib) //That's everybody
	{
		aggregator.send(); //Deliver all outgoing messages
		if (getStepCount()%8==0)
			aggregator.compact();//Blow away all the old voxels (saves memory)
		tryAdvance();
	}
}

/*Declared here instead of aggregate.cpp so it can see the collideMgr definition*/
void voxelAggregator::sendMessage(void)//Send off accumulated points
{
	//Detach accumulated data and send it off
	crossObjRec *o=obj.getData(); int n=obj.length();
	obj.detachBuffer();
	mgr->sendVoxelMessage(destination,n,o);
}

inline CkArrayIndex3D buildIndex(const gridLoc3d &l) 
	{return CkArrayIndex3D(l.x,l.y,l.z);}

//voxelAggregators deliver messages to voxels via this bottleneck
void collideMgr::sendVoxelMessage(const gridLoc3d &dest,
	int n,crossObjRec *obj)
{
	CM_STATUS(status("collideMgr::sendVoxelMessage"));
	msgsSent++;
	objListMsg *msg=new objListMsg(n,obj,
			objListMsg::returnReceipt(thisgroup,CkMyPe()));
	voxelProxy[buildIndex(dest)].add(msg);
}

void objListMsg::returnReceipt::send(void)
{
	CProxy_collideMgr p(gid);
	if (onPE==CkMyPe()) //Just send directly to the local branch
		p.ckLocalBranch()->voxelMessageRecvd();
	else //Deliver via the network
		p[onPE].voxelMessageRecvd();
}

//collideVoxels send a return receipt here
void collideMgr::voxelMessageRecvd(void)
{
	CM_STATUS(status("collideMgr::voxelMessageRecvd"));
	msgsRecvd++;
	//All the voxels we send messages to have received them--
	tryAdvance();
}

//Check if we're barren-- if so, advance now
void collideMgr::pleaseAdvance(void)
{
	CM_STATUS(status("collideMgr::pleaseAdvance"));
	tryAdvance();
}

//Attempt to finish the voxel send/recv step
void collideMgr::tryAdvance(void)
{
	CM_STATUS(CkPrintf("Cmgr Pe %d> %d contrib, %d = %d msgs\n",CkMyPe(),
			nContrib-contribCount, msgsSent, msgsRecvd);)
	if ((contribCount==nContrib) && (msgsSent==msgsRecvd)) {
		CM_STATUS(status("collideMgr::advancing"));
		advance();
		contribCount=0;
		msgsSent=msgsRecvd=0;
	}
}

//This is called on PE 0 once the voxel send/recv reduction is finished
void collideMgr::reductionFinished(void)
{
	CM_STATUS(status("collideMgr::reductionFinished"));
	//Broadcast collision start
	voxelProxy.startCollision();
}

//This is called after the collideVoxel collision reduction completes
void collideMgr::reductionClient(void *param,int len,void *data)
{
	CM_STATUS(status("collideMgr::reductionClient (serial)"));
	collideMgr *m=(collideMgr *)param;
	int nColl=len/sizeof(collision);
	if (m->clientFn!=NULL) //User wants collisions on node 0
	  (m->clientFn)(m->clientParam,nColl,(collision *)data);
	else //User wants collisions distributed
	  CkAbort("Forgot to call collideMgr::setClient!\n");
}

/********************** collideVoxel *********************/
//Extract the (signed) low 23 bits of src--
// this is the IEEE floating-point mantissa used as a grid index
static int low23(unsigned int src)
{
	unsigned int loMask=0x007fFFffu;//Low 23 bits set
	unsigned int offset=0x00400000u;
	return (src&loMask)-offset;
}

iSeg1d collideVoxel_extents[3]={
	iSeg1d(100,-100),iSeg1d(100,-100),iSeg1d(100,-100)};

//Print debugging state information
void collideVoxel::status(const char *msg)
{
	int x=low23(thisIndex.x);
	int y=low23(thisIndex.y);
	int z=low23(thisIndex.z);
	CkPrintf("Pe %d, voxel (%d,%d,%d)> %s\n",CkMyPe(),x,y,z,msg);
}
void collideVoxel::emptyMessages()
{
	for (int i=0;i<msgs.length();i++) delete msgs[i];
	msgs.length()=0;
}
collideVoxel::collideVoxel()
	:territory(
		gridMap.grid2world(0,rSeg1d(thisIndex.x,thisIndex.x+1)),
		gridMap.grid2world(1,rSeg1d(thisIndex.y,thisIndex.y+1)),
		gridMap.grid2world(2,rSeg1d(thisIndex.z,thisIndex.z+1))
	)
{
	CC_STATUS(status("created"));
	collideVoxel_extents[0].expand(low23(thisIndex.x));
	collideVoxel_extents[1].expand(low23(thisIndex.y));
	collideVoxel_extents[2].expand(low23(thisIndex.z));
}
collideVoxel::collideVoxel(CkMigrateMessage *m) : ArrayElement3D(m) 
{
	CC_STATUS(status("arrived from migration"));
}
collideVoxel::~collideVoxel()
{
	emptyMessages();
	CC_STATUS(status("deleted. (migration depart)"));
}
void collideVoxel::pup(PUP::er &p) {
	ArrayElement3D::pup(p);
	if (msgs.length()!=0) {
		status("Error!  Cannot migrate with messages in tow!\n");
		CkAbort("collideVoxel::pup cannot handle message case");
	}
	p|territory;
}
void collideVoxel::add(objListMsg *msg)
{
	CC_STATUS(status("add"));
	msg->sendReceipt();
	msgs.push_back(msg);
}

void collideVoxel::collide(collisionList &dest)
{
	int m;
	
#if 0  //Check if all the priorities are identical-- early exit if so
	CC_STATUS(status("      early-exit (all prio. identical) test"));
	int firstPrio=msgs[0]->tri(0).id.prio;
	bool allIdentical=true;
	for (m=0;allIdentical && m<msgs.length();m++)
		for (int i=0;i<msgs[m]->getNtris();i++)
			if (msgs[m]->tri(i).id.prio!=firstPrio)
			{ allIdentical=false; break;}
	if (allIdentical) {CC_STATUS(status("      early-exit used!"));return;}
#endif
	
	//Figure out how many objects we have total
	int n=0;
	for (m=0;m<msgs.length();m++) n+=msgs[m]->getObjects();
	octant o(n,territory);
	o.length()=0;
	bbox3d big;big.infinity();
	o.setBbox(big);
	
	CC_STATUS(status("      creating bbox, etc. for polys"));
	//Create records for each referenced poly
	for (m=0;m<msgs.length();m++) {
		const objListMsg &msg=*(msgs[m]);
		for (int i=0;i<msg.getObjects();i++)
			o.push_fast(&msg.getObj(i));
	}
	o.markHome(o.length());
	COL_STATUS(status("    colliding polys"));
	COL_STATUS(CkPrintf("\t\t\tXXX %d polys, %d msgs\n",o.length(),msgs.length());)
	o.findCollisions(dest);
}


void collideVoxel::startCollision(void)
{
	char buf[200];sprintf(buf,"startCollision on %d msgs {\n",msgs.length());
	CC_STATUS(status(buf));
	
	collisionList colls;
	collide(colls);
	contribute(colls.length()*sizeof(collision),colls.getData(),
		   CkReduction::concat);
	
	emptyMessages();
	CC_STATUS(status("startCollision }"));
}

#include "parCollide.def.h"
