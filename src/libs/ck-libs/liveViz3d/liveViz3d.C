/*
  Interface to server portion of the liveViz3d library.
*/
#include "pup.h"
#include "ckviewpoint.h"
#include "ckvector3d.h"
#include "conv-ccs.h"
#include "liveViz3d.h"
#include "pup_toNetwork4.h"
#include <vector>


static /* readonly */ CProxy_liveViz3dManager mgrProxy;
#define masterProcessor 0

/** 
 * The liveViz3dManager group stores up outgoing views before
 * they're sent off to clients.
 */ 
class liveViz3dManager : public CBase_liveViz3dManager {
//FIXME: support multiple clients here
	//These are all the views that have not yet been sent off to the client
	std::vector<CkView *> views;
	
	bool hasDelayed;
	CcsDelayedReply delayedReply;
	
	/// Pack up and send off up to 100KB of stored views:
	void sendReply(CcsDelayedReply repl) {
		int i,n;
		PUP_toNetwork4_sizer ps;
		ps|n;
		for (n=0;n<views.size();n++) {
			pup_pack(ps,*views[n]);
			if (ps.size()>100*1024) {n++; break;}
		}
		CmiPrintf("Sending off %d new views (%d bytes)\n",n,ps.size());
		int len=ps.size();
		char *retMsg=new char[len];
		PUP_toNetwork4_pack pp(retMsg);
		pp|n;
		for (i=0;i<n;i++) {
			pup_pack(pp,*views[i]);
			views[i]->unref();
		}
		if (len!=pp.size()) {
			CkError("Sizing pup was %d bytes; packing pup was %d!\n",
				len,pp.size());
			CkAbort("Pup size mismatch (logic error) in liveViz3d!\n");
		}
		
		// views.clear();
		views.erase(views.begin(),n+views.begin());
		CcsSendDelayedReply(repl,len,retMsg);
		delete[] retMsg;
		CmiPrintf("Done sending off %d views\n",n);
	}
	
	// FIXME: should have a vector of consumers, for each client
	CkViewConsumer *cons;
public:
	liveViz3dManager(void);
	
	/// This remote object has created a new view:
	void addView(CkViewHolder &view) {
		addView(view.release());
	}
	/// This local object has a new view:
	void addView(CkView *view) {
		view->ref();
		views.push_back(view);
		if (hasDelayed) { //There's already somebody waiting
			hasDelayed=false;
			sendReply(delayedReply);
		}
	}
	
	/// This client is requesting the latest views:
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
	
	/// Viewable needs a place to put its views:
	inline CkViewConsumer &getConsumer(const liveViz3dRequestMsg *m) {
		return *cons;
	}
};


/// Used to send views off to the master processor
class RemoteManagerConsumer : public CkViewConsumer {
	// Prevent uploading duplicate views:
	CkHashtableT<CkViewableID,CkView *> views;
public:
	virtual void add(CkView *view);
};
void RemoteManagerConsumer::add(CkView *view) {
	CkView *oldView=views.get(view->id);
	if (oldView!=view) 
	{ // The view has changed: add it to our hashtable and pass it on
		views.put(view->id)=view;
		mgrProxy[masterProcessor].addView(view);
	}
}

/// Used to hand views off to the local group
class LocalManagerConsumer : public CkViewConsumer {
	// Prevent uploading duplicate views:
	CkHashtableT<CkViewableID,CkView *> views;
	liveViz3dManager *mgr;
public:
	LocalManagerConsumer(liveViz3dManager *mgr_) 
		:mgr(mgr_) {}
	
	virtual void add(CkView *view);
};
void LocalManagerConsumer::add(CkView *view) {
	CkView *oldView=views.get(view->id);
	if (oldView!=view) 
	{ // The view has changed: add it to our hashtable and pass it on
		views.put(view->id)=view;
		mgr->addView(view);
	}
}

liveViz3dManager::liveViz3dManager(void)
{
	mgrProxy=thisgroup;
	hasDelayed=false;
	
	if (CkMyPe()==masterProcessor)
		cons=new LocalManagerConsumer(this);
	else
		cons=new RemoteManagerConsumer();
}

/* CCS Interface */
//These are only set on processor 0:
static CkBbox3d pe0_box;
static CkCallback pe0_request; 

/*
"lv3d_newViewpoint" handler:
	Sent by the client when his viewpoint changes,
to request updated views of the objects.

Outgoing request is a CkViewpoint.
There is no response.
*/
extern "C" void lv3d_newViewpoint(char *msg) {
	liveViz3dRequestMsg *rm=new liveViz3dRequestMsg;
	PUP_toNetwork4_unpack p(&msg[CmiMsgHeaderSizeBytes]);
	rm->clientID=0; //FIXME
	rm->vp.pup(p);
	pe0_request.send(rm);
	CmiFree(msg);
	CmiPrintf("New user viewpoint\n");
}

/*
"lv3d_getViews" handler:
	Sent by the client to request the actual images
of any updated object views.

Outgoing request is a single integer, the client ID
Incoming response is a set of updated CkViews:
    int n; p|n;
    for (int i=0;i<n;i++) p|view[i];
*/
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

/**
 * Call this routine with each viewable each time the
 * incomingRequest callback is executed.  
 * Can be called on any processor.
 * Be sure to delete the message after all the calls.
 */
void liveViz3dHandleRequest(const liveViz3dRequestMsg *m,CkViewable &v) 
{
	v.view(m->vp,mgrProxy.ckLocalBranch()->getConsumer(m));
}

#include "liveViz3d.def.h" //For liveViz3dRequestMsg
