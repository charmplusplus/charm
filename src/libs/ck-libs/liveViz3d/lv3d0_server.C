/*
  Implementation of the simplest server portion 
  of the liveViz3d library.
  
  Orion Sky Lawlor, olawlor@acm.org, 2003/9/13
*/
#include "lv3d0_server.h"
#include "pup_toNetwork4.h"
#include <vector>
#include <map>
#include <algorithm>

#include "lv3d0.decl.h" //For LV3D0_ViewMsg
#include "lv3d1.decl.h" //For LV3D1 registration (for .def file)

static /* readonly */ CProxy_LV3D0_Manager mgrProxy;
#define masterProcessor 0

/**
  This message is used to pack up and ship views from where they're
  computed to the master processor, where they're accumulated
  for CCS transport to the client.  
*/
class LV3D0_ViewMsg : public CMessage_LV3D0_ViewMsg {
public:
	/// This is the viewable we describe
	CkViewableID id;
	
	/// This is our approximate network priority. Prio==0 is highest.
	int prio;
	
	/// This is the client that requested the view.
	int clientID;
	
	/// Number of bytes in view array:
	int view_size;
	/// A PUP::toNetwork4'd CkView:
	unsigned char *view;
	
	/// Pack this CkView into a new message.  Does not keep the view.
	static LV3D0_ViewMsg *new_(CkView *view);
	static void delete_(LV3D0_ViewMsg *m);
};

/// Pack this CkView into a new message:
LV3D0_ViewMsg *LV3D0_ViewMsg::new_(CkView *vp) {
	PUP_toNetwork4_sizer ps; ps|vp;
	int view_size=ps.size();
	
	LV3D0_ViewMsg *m=new (view_size,0) LV3D0_ViewMsg;
	m->id=vp->id;
	m->prio=vp->prio;
	m->view_size=view_size;
	PUP_toNetwork4_pack pp(m->view); pp|vp;
	return m;
}
void LV3D0_ViewMsg::delete_(LV3D0_ViewMsg *m) {
	delete m;
}


/********** LV3D0_ClientManager: outgoing images ********/

/// Holds a view in the priority heap.
class CkViewPrioHolder {
public:
	LV3D0_ViewMsg *v;
	
	CkViewPrioHolder(LV3D0_ViewMsg *v_) :v(v_) {}
	
	// Comparison operator, used to keep a priority queue.
	bool operator<(const CkViewPrioHolder &h) const {
		if (v->prio<h.v->prio) return true;
		if (v->prio>h.v->prio) return false;
		/* else priorities are equal: compare ID's */
		int i;
		for (i=0;i<4;i++) {
			if (v->id.id[i]<h.v->id.id[i]) return true;
			if (v->id.id[i]>h.v->id.id[i]) return false;
		}
		// Woa-- they're *identical*.  Just pick smaller one.
		return v < h.v;
	}
};

/**
 * A LV3D0_ClientManager buffers up outgoing views before
 * they're requested by, and sent off to, a client.
 */
class LV3D0_ClientManager {
	/*
	  We generally want to process views in priority order;
	  but we never want to send an earlier view to a client
	  when a later view is available.
	  
	  This means we need to keep views in a priority queue
	  (for extraction), but also provide a way to look up 
	  the view (to prevent duplicates).
	*/
	
	// This indexes unsent views by ID:
	CkHashtableT<CkViewableID,CkViewPrioHolder> id2view;

	// This indexes unsent views by priority:
	typedef std::map<CkViewPrioHolder,char> prio2view_t;
	prio2view_t prio2view;
	
	bool hasDelayed;
	CcsDelayedReply delayedReply;
	
	/// Pack up and send off up to 100KB of stored views:
	void sendReply(CcsDelayedReply repl) {
	
	// Figure out how many views we can afford to send at once:
		int len=4; /* leave room for the "length" field */
		int n=0;
		prio2view_t::iterator it=prio2view.begin();
		while (len<100*1024 && it!=prio2view.end()) {
			LV3D0_ViewMsg *v=(*it++).first.v;
			len+=v->view_size;
			n++;
		}
		prio2view_t::iterator sendEnd=it;
	
	// Pack the views into a network buffer:
		// CmiPrintf("[OUT] Sending off %d new views (%d bytes)\n",n,len);
		
		// FIXME: special case this when n==1.
		char *retMsg=new char[len];
		PUP_toNetwork4_pack pp(retMsg);
		pp|n;
		for (it=prio2view.begin(); it!=sendEnd; ++it) {
			prio2view_t::iterator doomed=it;
			LV3D0_ViewMsg *v=(*it).first.v;
			pp(v->view,v->view_size);
			prio2view.erase(doomed);
			id2view.remove(v->id);
			delete v;
		}
		
		if (len!=pp.size()) {
			CkError("Sizing pup was %d bytes; packing pup was %d!\n",
				len,pp.size());
			CkAbort("Pup size mismatch (logic error) in LV3D0_!\n");
		}

		CcsSendDelayedReply(repl,len,retMsg);
		delete[] retMsg;
		
		// CmiPrintf("[OUT] Done sending off %d views\n",n);
	}

public:
	LV3D0_ClientManager() 
		:id2view(129,0.2)
	{
		hasDelayed=false;
	}
	~LV3D0_ClientManager() {
		for (prio2view_t::iterator it=prio2view.begin();
		  it!=prio2view.end();
		  ++it)
			delete (*it).first.v;
	}
	
	/// A local object has a new view.
	///  We are passed in our reference to this object.
	void add(LV3D0_ViewMsg *v) {
		CkViewPrioHolder old=id2view.get(v->id);
		/// FIXME: messages may come out of order-- check framenumber
		if (old.v!=0) 
		{ /* An old entry from this viewable exists: remove it.
		     This is so we always send off the most up-to-date view. */
			// printf("[OUT] Replacing old view for viewable %d\n",v->id.id[0]);
			prio2view.erase(prio2view.find(old));
			delete old.v;
		}
		else {
			// printf("[OUT] Queueing new view for viewable %d\n",v->id.id[0]);
		}
		id2view.put(v->id)=CkViewPrioHolder(v);
		prio2view.insert(std::make_pair(CkViewPrioHolder(v),(char)1));
		
		if (hasDelayed) { //There's already somebody waiting
			hasDelayed=false;
			sendReply(delayedReply);
		}
	}
	
	/// A client is requesting the latest views
	///   This routine must be called from a CCS handler.
	void getViews(void) {
		if (prio2view.size()==0) { //Nothing to send yet-- wait for it
			// printf("[OUT] Views requested, but none available\n");
			hasDelayed=true;
			delayedReply=CcsDelayReply();
		}
		else 
		{
			sendReply(CcsDelayReply());
		}	
	}
};

/** 
 * The LV3D0_Manager group stores up outgoing views before
 * they're sent off to clients.
 */ 
class LV3D0_Manager : public CBase_LV3D0_Manager {
	// FIXME: should have a vector of clients
	LV3D0_ClientManager *client;
public:
	LV3D0_Manager(void);
	
	/// This client is requesting the latest views.
	///  This routine must be called from a CCS handler.
	void getViews(int clientID) {
		client->getViews();
	}
	
	/// A viewable is adding this view for this client.
	inline void addView(LV3D0_ViewMsg *m) {
		if (CkMyPe()==masterProcessor) {
			// FIXME: look up based on clientID
			client->add(m);
		} else /* forward to master processor */
			thisProxy[masterProcessor].addView(m);
	}
};


LV3D0_Manager::LV3D0_Manager(void)
{
	mgrProxy=thisgroup;
	
	if (CkMyPe()==masterProcessor)
		client=new LV3D0_ClientManager();
	else
		client=NULL;
}


/**
  Send this view back to this client.
  You must set the view's id and prio fields.
 */
void LV3D0_Deposit(CkView *v,int clientID) {
	LV3D0_ViewMsg *vm=LV3D0_ViewMsg::new_(v);
	vm->clientID=clientID;
    mgrProxy.ckLocalBranch()->addView(vm);
}


/*************** CCS Interface ***************/
//This only ever gets set on processor 0 (FIXME: checkpointing)
static LV3D_Universe *theUniverse=0;
static CkCallback frameUpdate;

/**
"lv3d_setup" CCS handler:
	Request a new clientID and universe.

Outgoing request is empty.
Response is a 1-int clientID followed by a PUP::able universe.
*/
extern "C" void LV3D0_setup(char *msg) {
	CmiFree(msg);
	int clientID=0; // FIXME: actually care about client
	PUP_toNetwork4_sizer sp;
	sp|clientID;
	sp|theUniverse;
	unsigned char *buf=new unsigned char[sp.size()];
	PUP_toNetwork4_pack pp(buf);
	pp|clientID;
	pp|theUniverse;
	CcsSendReply(sp.size(),buf);
	// CmiPrintf("Registered (client %d)\n",clientID);
	delete[] buf;
}


/**
"lv3d_newViewpoint" CCS handler:
	Sent by the client when his viewpoint changes,
to request updated views of the objects.

Outgoing request is an integer clientID, frameID, 
and a CkViewpoint. There is no response.
*/
extern "C" void LV3D0_newViewpoint(char *msg) {
	PUP_toNetwork4_unpack p(&msg[CmiMsgHeaderSizeBytes]);
	LV3D_ViewpointMsg *m=new LV3D_ViewpointMsg;
	p|m->clientID;
	p|m->frameID;
	m->viewpoint.pup(p);
	// CmiPrintf("New user viewpoint (client %d, frame %d)\n",clientID,frameID);
	frameUpdate.send(m);
}

/**
"lv3d_getViews" CCS handler:
	Sent by the client to request the actual images
of any updated object views.  These come via LV3D0_Deposit.

Outgoing request is a single integer, the client ID
Incoming response is a set of updated CkViews:
    int n; p|n;
    for (int i=0;i<n;i++) p|view[i];
*/
extern "C" void LV3D0_getViews(char *msg) {
	PUP_toNetwork4_unpack p(&msg[CmiMsgHeaderSizeBytes]);
	int clientID; p|clientID; CmiFree(msg);
	mgrProxy.ckLocalBranch()->getViews(clientID);
}

/*
"lv3d_qd" CCS handler:
	Sent by the client to wait until all views have been
updated and sent back.  No input or output data: control flow only.
*/
static void qdDoneFn(void *param,void *msg) 
{
	CcsDelayedReply *repl=(CcsDelayedReply *)param;
	CcsSendDelayedReply(*repl,0,0);
}
extern "C" void LV3D0_qd(char *msg) 
{
	CmiFree(msg);
	CcsDelayedReply *repl=new CcsDelayedReply(CcsDelayReply());
	CkCallback cb(qdDoneFn,repl);
	CkStartQD(cb); /* finish CCS reply after quiescence */
}


/**
Register for libsixty redraw requests.  This routine
must be called exactly once on processor 0.
*/
void LV3D0_Init(LV3D_Universe *clientUniverse,const CkCallback &frameUpdate_)
{
	if (clientUniverse==0)
		clientUniverse=new LV3D_Universe();
	theUniverse=clientUniverse;
	frameUpdate=frameUpdate_;
	CcsRegisterHandler("lv3d_setup",(CmiHandler)LV3D0_setup);
	CcsRegisterHandler("lv3d_newViewpoint",(CmiHandler)LV3D0_newViewpoint);
	CcsRegisterHandler("lv3d_getViews",(CmiHandler)LV3D0_getViews);
	CcsRegisterHandler("lv3d_qd",(CmiHandler)LV3D0_qd);
	CProxy_LV3D0_Manager::ckNew();
}

/**
 Per-processor initialization routine:
*/
void LV3D0_ProcInit(void) {
	
}
void LV3D0_NodeInit(void) {
	CkViewNodeInit();
}


#include "lv3d0.def.h"
#include "liveViz3d.def.h"

