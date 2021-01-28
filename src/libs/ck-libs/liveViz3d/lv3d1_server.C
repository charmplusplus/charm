/**
  Interface to server portion of a more complicated
  interface to the liveViz3d library-- it defines a 
  single array of objects, each of which represent a 
  single CkViewable.

  Orion Sky Lawlor, olawlor@acm.org, 2003
*/
#include "lv3d1_server.h"

/* Readonlies, set only via a special CCS request */
int LV3D_Disable_Render_Prio;
int LV3D_Verbosity;

/**
  Return the priority for this object.
  Charm priorities have lower numbers with higher priority.
*/
unsigned int LV3D_build_priority(int frameNo,double prioAdj) {
	if (LV3D_Disable_Render_Prio) return 0;
	const double prioMax=1.0;
	if (!(prioAdj<prioMax)) prioAdj=prioMax; /* reset big & NaN values */
	return 0xC0000000u+frameNo-(int)(1000*prioAdj);	
}

/************** LV3D_Array: client interface & render requests ************
Represents the list of objects to be viewed in the scene.
*/
class impl_LV3D_Array {
	LV3D_Array *array;
	
	CkViewable *viewable; // Should be array of viewables
	
// All this state should be per-client:

	/// Last-rendered view for this viewable.
	CkView *view; // Should be per-viewable
	CkVector3d lastCamera; /// render position for last view
	// Throw out old views, and allow new render request
	void flush(void) {
		if (view) view->unref();
		view=NULL;
		renderRequested=false;
	}
	bool renderRequested;
	CkViewpoint renderViewpoint;
	int renderFrameID;
	void renderUpdate(const LV3D_ViewpointMsg *m) {
		renderViewpoint=m->viewpoint;
		renderFrameID=m->frameID;
	}
	
	CkViewableID makeViewableID(int vi) {
		CkViewableID ret;
		int i;
		for (i=0;i<array->thisIndexMax.nInts;i++)
			ret.id[i]=array->thisIndexMax.data()[i];
		for (;i<3;i++)
			ret.id[i]=0;
		ret.id[3]=vi;
		return ret;
	}
	void status(const char *where) {
		// CkPrintf("[%d] %s",array->thisIndexMax.data()[0],where);
	}
public:
	impl_LV3D_Array(LV3D_Array *array_) :array(array_) {
		viewable=NULL;
		view=NULL;
		renderRequested=false;
	}
	~impl_LV3D_Array() {
		/* viewables belong to caller; do not delete them */
		flush();
	}
	
	inline void add(CkViewable *v_) {
		if (viewable) CkAbort("LV3D_Array::addViewable> Added too many viewables!\n");
		viewable=v_;
	}
	inline void remove(CkViewable *v_) {
		if (!viewable) CkAbort("LV3D_Array::removeViewable> No viewable to remove!\n");
		if (viewable!=v_)  CkAbort("LV3D_Array::removeViewable> Can't remove wrong viewable!\n");
		viewable=NULL;
		flush();
	}
	
	inline void newClient(int clientID) {
		// FIXME: reinitialize that clientID here.
		flush();
	}
	
	/**
	  This request is broadcast every time a client viewpoint changes.
	  Internally, it asks the stored CkViewables if they should redraw,
	  and if so, queues up a LV3D_RenderMsg.
	 */
	void viewpoint(const LV3D_ViewpointMsg *m) {
		if (renderRequested) 
		{ // Already wanting to render-- just update the viewpoint:
	status("Updating pending render...\n");
			renderUpdate(m);
		}
		else if (viewable) {
		  if ((!view) || viewable->shouldRender(m->viewpoint,*view)) 
		  { // Need to ask for a new rendering:
	status("Reqesting rendering...\n");
			renderRequested=true;
			renderUpdate(m);
			double sizeAdjust=viewable->getSize(renderViewpoint);
			LV3D_RenderMsg *rm= LV3D_RenderMsg::new_(
				m->clientID,m->frameID,0,sizeAdjust);
			array->thisProxy[array->thisIndexMax].LV3D_Render(rm);
		  }
		}
	}
	
	/**
	  This method is used to prioritize rendering.
	*/
	void render(LV3D_RenderMsg *m) {
	status("Rendering...\n");
		CkView *v=viewable->renderView(renderViewpoint);
		flush(); /* subtle: otherwise a NULL-returning render never updates again! */
		if (v) {
			view=v;
			view->id=makeViewableID(0);
			
			double sizeAdjust=viewable->getSize(renderViewpoint);
			view->prio=LV3D_build_priority(renderFrameID,sizeAdjust);
			LV3D0_Deposit(view,m->clientID);
		}
		LV3D_RenderMsg::delete_(m);
	}
	
#if LV3D_USE_FLAT
	void LV3D_FlatRender(liveVizRequestMsg *m,LV3D_Array *arr);
#endif
	
};

/* The array itself just forwards everything to the impl_LV3D_Array */
void LV3D_Array::init(void) {
	impl=new impl_LV3D_Array(this);
	usesAtSync=true;
}

void LV3D_Array::addViewable(CkViewable *v) {
	impl->add(v);
}
void LV3D_Array::removeViewable(CkViewable *v) {
	impl->remove(v);
}
void LV3D_Array::pup(PUP::er &p) {
	/* everything in impl can be thrown out during a migration */
	if (LV3D_Verbosity>1) {
		const int *data=thisIndexMax.data();
		CkPrintf("LV3DArray(%d,%d,%d) pup on PE %d\n",
			data[0],data[1],data[2],CkMyPe());
	}
}
	
LV3D_Array::~LV3D_Array() {
	delete impl;
	impl=NULL;
}
	
// Network-called methods:
/**
  This request is broadcast every time a client connects.
*/
void LV3D_Array::LV3D_NewClient(int clientID) 
{
	impl->newClient(clientID);
}

/// Perform load balancing now.	
void LV3D_Array::LV3D_DoBalance(void) {
	AtSync();
}

/**
  This request is broadcast every time a client viewpoint changes.
  Internally, it asks the stored CkViewables if they should redraw,
  and if so, queues up a LV3D_RenderMsg.
 */
void LV3D_Array::LV3D_Viewpoint(LV3D_ViewpointMsg *m) 
{
	LV3D_Prepare();
	impl->viewpoint(m);
}

/**
  This method is used to prioritize rendering.
*/
void LV3D_Array::LV3D_Render(LV3D_RenderMsg *m)
{
	impl->render(m);
}

/**
  This entry method is only used when rendering to
    plain old server-assembled liveViz 2d.
*/
void LV3D_Array::LV3D_FlatRender(liveVizRequestMsg *m)
{
#if LV3D_USE_FLAT
	LV3D_Prepare();
	impl->LV3D_FlatRender(m,this);
#endif
}

void LV3D_Array::LV3D_Prepare(void) {}

#if LV3D_USE_FLAT
void impl_LV3D_Array::LV3D_FlatRender(liveVizRequestMsg *m,LV3D_Array *arr)
{
   {
	if (!viewable) goto skipit;
	CkViewpoint vp;
	liveVizRequestUnpack(m,vp);
	CkQuadView *v=(CkQuadView *)viewable->renderView(vp);
	if (v==NULL) goto skipit; /* out of bounds */
	CkVector3d topLeft=vp.project(v->corners[0]);
	CkAllocImage &src=v->getImage(); // FIXME: assumed to be RGBA
	int w=src.getWidth(), h=src.getHeight();
	CkAllocImage dest(w, h, 3);
	int x,y;
	for (y=0;y<h;y++) {
		unsigned char *i=src.getPixel(0,y);
		const int ip=4; /*byte size of pixel */
		int r, g, b; /* byte offsets from pixel start */
		if (src.getLayout()==CkImage::layout_reversed)
			{r=2; g=1; b=0;}
		else
			{r=1; g=2; b=3;}
		unsigned char *o=dest.getPixel(0,y);
		const int op=3; /*byte size of pixel*/
		for (x=0;x<w;x++) {
			o[op*x+0] = i[x*ip+r];
			o[op*x+1] = i[x*ip+g];
			o[op*x+2] = i[x*ip+b];
		}
	}
	x=(int)(topLeft.x+0.5), y=(int)(topLeft.y+0.5);
	liveVizDeposit(m, x,y, w,h, dest.getData(), arr);
	delete v;
	return;
   }
skipit:/* nothing to show: deposit empty image (so reduction completes) */
	// printf("Skipping deposit for %d\n",arr->thisIndex.data[0]);
	liveVizDeposit(m, 0,0, 0,0, 0, arr);
}
#endif

/**************** Map ********************/
/*
 Simple map that stays away from PE 0.
*/
class LV3D1_Map : public CkArrayMap {
	unsigned int shiftPE, numPE;
	void init(void) {
		shiftPE=0;
		numPE=CkNumPes();
		if (numPE>1) {
			shiftPE=1; /* don't put anything on PE 1 */
			numPE-=shiftPE;
		}
	}
public:
	LV3D1_Map() {init();}
	LV3D1_Map(CkMigrateMessage *m):CkArrayMap(m){init();}
  
  int procNum(int arrayHdl, const CkArrayIndex &i)
  {
#if 1
    if (i.nInts==1) {
      //Map 1D integer indices in simple round-robin fashion
      return shiftPE+((i.data()[0])%numPE);
    }
    else 
#endif
      {
        //Map other indices based on their hash code, mod a big prime.
        unsigned int hash=(i.hash()+739)%1280107;
        return shiftPE+(hash % numPE);
      }
  }
};

void LV3D1_Attach(CkArrayOptions &opts)
{
	opts.setMap(CProxy_LV3D1_Map::ckNew());
}

/**************** Network Messages **************/

/// Make a prioritized LV3D_RenderMsg:
LV3D_RenderMsg *LV3D_RenderMsg::new_(
	int client,int frame,int viewable,double prioAdj) 
{
	int prioBits=8*sizeof(prioAdj);
	LV3D_RenderMsg *m=new (prioBits) LV3D_RenderMsg;
	m->clientID=client;
	m->frameID=frame;
	m->viewableID=viewable;
	unsigned int *p=(unsigned int *)CkPriorityPtr(m);
	p[0]=LV3D_build_priority(frame,prioAdj);
	if (LV3D_Disable_Render_Prio) p[0]=0;
	CkSetQueueing(m,CK_QUEUEING_BFIFO);
	return m;
}
void LV3D_RenderMsg::delete_(LV3D_RenderMsg *m) {
	delete m;
}

void LV3D1_ServerMgr::doBalance(void)
{
	a.LV3D_DoBalance();
}	

void LV3D1_Init(CkArrayID aid,LV3D_Universe *theUniverse)
{
	LV3D0_Init(theUniverse,new LV3D1_ServerMgr(aid));
#if LV3D_USE_FLAT
	// Broadcast to LV3D_FlatRender for 2D views.
	CkCallback flatUpdate(CkIndex_LV3D_Array::LV3D_FlatRender(0),aid);
	liveVizInit(liveVizConfig(true,false),aid,flatUpdate);
#endif
}




#include "lv3d1.def.h"
