/*
  Interface to server portion of the sixty library.
*/
#include "pup.h"
#include "viewpoint.h"
#include "ckvector3d.h"
#include "conv-ccs.h"
#include "liveViz3d_impl.h"
#include "liveViz3d.h"
#include "pup_toNetwork4.h"
#include <vector>

static CProxy_liveViz3dManager mgrProxy;

class liveViz3dManager : public CBase_liveViz3dManager {
//FIXME: support multiple clients here
	//These are all the views that have not yet been sent off to the client
	std::vector<CkView *> views;
	
	bool hasDelayed;
	CcsDelayedReply delayedReply;
	
	void sendReply(CcsDelayedReply repl) {
		int i,n=views.size();
		CmiPrintf("Sending off %d new views\n",n);
		PUP_toNetwork4_sizer ps;
		ps|n;
		for (i=0;i<n;i++) ps|*views[i];
		int len=ps.size();
		char *retMsg=new char[len];
		PUP_toNetwork4_pack pp(retMsg);
		pp|n;
		for (i=0;i<n;i++) {
			pp|*views[i];
			delete views[i];
		}
		views.clear();//erase(views.begin(),views.end());
		CcsSendDelayedReply(repl,len,retMsg);
		delete[] retMsg;
		CmiPrintf("Done sending off %d views\n",n);
	}
	
public:
	liveViz3dManager(void) {
		mgrProxy=thisgroup;
		hasDelayed=false;
	}
	
	//This object has created a new view for client ID:
	void addView(pupPtrHolder<CkView> &view,int clientID) {
		views.push_back(view.release());
		if (hasDelayed) { //There's already somebody waiting
			hasDelayed=false;
			sendReply(delayedReply);
		}
	}
	
	//This client is requesting the latest views:
	void getViews(int clientID) {
	#if 1
		if (views.size()==0) { //Nothing to send yet-- wait for it
			hasDelayed=true;
			delayedReply=CcsDelayReply();
		}
		else 
	#endif
		{
			sendReply(CcsDelayReply());
		}	
	}
	
};


/* CCS Interface */
//These are only set on processor 0:
static CkBbox3d pe0_box;
static CkCallback pe0_request; 

extern "C" void lv3d_newViewpoint(char *msg) {
	liveViz3dRequestMsg *rm=new liveViz3dRequestMsg;
	PUP_toNetwork4_unpack p(&msg[CmiMsgHeaderSizeBytes]);
	rm->nv.pup(p);
	pe0_request.send(rm);
	CmiFree(msg);
		CmiPrintf("New user viewpoint\n");
}

extern "C" void lv3d_getViews(char *msg) {
	int clientID;
	PUP_toNetwork4_unpack p(&msg[CmiMsgHeaderSizeBytes]);
	p|clientID;
	mgrProxy.ckLocalBranch()->getViews(clientID);
	CmiFree(msg);
}

/*
Register for libsixty redraw requests.  This routine
must be called exactly once on processor 0.
The callback will be executed whenever the user updates
their viewpoint--it will be called with a liveViz3dRequestMsg.
*/
void liveViz3dInit(const CkBbox3d &box,CkCallback incomingRequest)
{
	pe0_box=box;
	pe0_request=incomingRequest;
	CcsRegisterHandler("lv3d_getViews",(CmiHandler)lv3d_getViews);
	CcsRegisterHandler("lv3d_newViewpoint",(CmiHandler)lv3d_newViewpoint);
	CProxy_liveViz3dManager::ckNew();
}

/******************* liveViz3dViewable ***********************
This object should live on every viewable: its 
"handleRequest" method should be called every time
the user changes their viewpoint.  You must implement
a subclass of this class that implements the "view" 
method to draw yourself.
*/
class liveViz3dViewableImpl {
	CkView *lastView; //Last rendered viewpoint
public:	
	liveViz3dViewableImpl() {
		lastView=NULL;
	}
	~liveViz3dViewableImpl() {
		delete lastView;
	}
	void pup(PUP::er &p) {
		/*empty-- view cache is flushed on migration*/
	}
	
	
	//Return true if we're out of date under this view:
	bool outOfDate(const CkViewpoint &vp) {
		if (lastView==NULL) return true; 
		double viewTol=2.0; //Accept up to this many pixels of error:
		if (lastView->rmsError(vp)>viewTol) return true;
		return false;
	}
	
	//Get the best view for this (out of date) viewpoint:
	CkView *getView(const CkViewpoint &vp,CkViewable *obj) {
		if (lastView) delete lastView;
		lastView=new CkView(vp,obj);
		return lastView;
	}
};


liveViz3dViewable::liveViz3dViewable(void) {
	impl=new liveViz3dViewableImpl;
}
liveViz3dViewable::~liveViz3dViewable() {
	delete impl;
}

void liveViz3dViewable::handleRequest(liveViz3dRequestMsg *m)
{
	const CkViewpoint &vp=m->nv.vp;
	if (!impl->outOfDate(vp)) {delete m; return;}
	//Make a new view and send it off to processor 0:
	CkView *v=impl->getView(vp,this);
	mgrProxy[0].addView(v,m->nv.clientID);
}

void liveViz3dViewable::pup(PUP::er &p)
{
	p|id;
	p|univPoints;
	impl->pup(p);
}

#include "liveViz3d.def.h" //For liveViz3dRequestMsg
