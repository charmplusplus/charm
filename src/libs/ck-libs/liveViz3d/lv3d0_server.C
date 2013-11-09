/*
  Implementation of the simplest server portion 
  of the liveViz3d library.
  
  Orion Sky Lawlor, olawlor@acm.org, 2003/9/13
*/
#include "lv3d0_server.h"
#include "pup_toNetwork.h"
#include <vector>
#include <map>
#include <algorithm>
#include "stats.h"
#include "LBDatabase.h" /* for set_avail_vector */

#include "lv3d0.decl.h" //For LV3D0_ViewMsg
void _registerlv3d1(void);
void _registerliveViz(void);
// #include "lv3d1.decl.h" //For LV3D1 registration (for .def file)

static /* readonly */ CProxy_LV3D0_Manager mgrProxy;
static /* readonly */ int LV3D_dosave_views=0; ///< Write views to file (not network).
static /* readonly */ int LV3D_disable_ship_prio=0; ///< Set priorities to zero
static /* readonly */ int LV3D_disable_ship_replace=0; ///< Don't replace old images from same impostor
static /* readonly */ int LV3D_disable_ship_throttle=0; ///< Don't send a limited number at once
static /* readonly */ int LV3D_disable_ship=0;  ///< Throw away images, don't ship any
#define masterProcessor 0

/**
  This message is used to pack up and ship views from where they're
  computed to the master processor, where they're accumulated
  for CCS transport to the client.  
*/
class LV3D0_ViewMsg : public CMessage_LV3D0_ViewMsg {
public:
// (these fields are all copied from our viewable)
	/// This is the viewable we describe
	CkViewableID id;
	
	/// This is our approximate network priority
	///   Prio==0 is highest.
	int prio;
	
	/// Pixels we represent
	int pixels;
	
	/// This is the client that requested the view.
	int clientID;
	
	/// Number of bytes in view array:
	int view_size;
	/// A PUP::toNetwork'd CkView:
	unsigned char *view;
	
	/// Pack this CkView into a new message.  Does not keep the view.
	static LV3D0_ViewMsg *new_(CkView *view);
	static void delete_(LV3D0_ViewMsg *m);
};

/// Pack this CkView into a new message:
LV3D0_ViewMsg *LV3D0_ViewMsg::new_(CkView *vp) {
	PUP_toNetwork_sizer ps; ps|vp;
	int view_size=ps.size();
	
	LV3D0_ViewMsg *m=new (view_size,0) LV3D0_ViewMsg;
	m->id=vp->id;
	m->prio=vp->prio;
	m->pixels=vp->pixels;
	m->view_size=view_size;
	PUP_toNetwork_pack pp(m->view); pp|vp;
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
	//  Returns true if we should be processed before this guy.
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
 Stores a set of view messages:
   - Sorts outgoing views by priority.
   - Removes duplicate views from same viewable.
*/
class CkViewPrioSorter {

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
	//   The "key" is all that's needed here-- the "obj" field is useless.
	typedef std::map<CkViewPrioHolder,char> prio2view_t;
	prio2view_t prio2view;
	
public:
	CkViewPrioSorter() 
		:id2view(129,0.2)
	{
	}
	~CkViewPrioSorter() {
		for (iterator it=begin();it!=end();++it)
			delete (*it).first.v;
	}
	
	/// Return true if we have no stored, unsent views.
	bool isEmpty(void) const {return prio2view.size()==0;}
	
	/// Add this view to our set.
	void add(LV3D0_ViewMsg *v)
	{
		if (LV3D_disable_ship_prio) v->prio=0;
		CkViewPrioHolder old=id2view.get(v->id);
		/// FIXME: messages may come out of order-- check framenumber
		if (old.v!=0 && !LV3D_disable_ship_replace) 
		{ /* An old entry from this viewable exists: remove it.
		     This is so we always send off the most up-to-date view. */
			// printf("[OUT] Replacing old view for viewable %d\n",v->id.id[0]);
			if (v->prio>old.v->prio)  // keep old priority if it's higher
				v->prio=old.v->prio;
			prio2view.erase(prio2view.find(old));
			delete old.v;
		}
		else {
			// printf("[OUT] Queueing new view for viewable %d\n",v->id.id[0]);
		}
		id2view.put(v->id)=CkViewPrioHolder(v);
		prio2view.insert(std::make_pair(CkViewPrioHolder(v),(char)1));
	}
	
	/// Peek through the set of high-priority views.
	///  Get the iterator v, extract with (*v)
	typedef prio2view_t::iterator iterator;
	iterator begin(void) {return prio2view.begin();}
	iterator end(void) {return prio2view.end();}
	
	/// Extract the corresponding view for this iterator.
	///   Often it==begin().
	LV3D0_ViewMsg *extract(const iterator &doomed) {
		LV3D0_ViewMsg *v=(*doomed).first.v;
		prio2view.erase(doomed);
		id2view.remove(v->id);
		return v;
	}
};

/// Generic superclass for client managers.
class LV3D0_ClientManager : public CkViewPrioSorter {
public:
	virtual ~LV3D0_ClientManager() {}
	virtual void add(LV3D0_ViewMsg *v) =0;
	virtual void getViews(void) {}
	virtual void whenEmptyCallback(const CkCallback &cb) {}
};

/******************** Shipping views to Master ***************/
/// Communication parameter: number of bytes to send per 10ms window.
///  NOTE: this is reset in procInit, below.
int LV3D0_toMaster_bytesPer=100*1024;
/// Communication parameter: maximum number of bytes to store in bucket.
int LV3D0_toMaster_bytesMax=2*LV3D0_toMaster_bytesPer;

static void toMaster_fillBucket(void *ptr,double timer);

static stats::op_t op_master_count=stats::count_op("master.count","Number of final master impostors","count");
static stats::op_t op_master_bytes=stats::count_op("master.bytes","Number of final master bytes","bytes");
static stats::op_t op_master_pixels=stats::count_op("master.pixels","Number of final master pixels","pixels");

/**
 * A LV3D0_ClientManager that buffers up outgoing views 
 * for delivery to the master processor.  This throttles the
 * cluster network utilization, preventing ancient views from
 * piling up at the network interconnect.
 */
class LV3D0_ClientManager_toMaster : public LV3D0_ClientManager {
	/// Bucket algorithm: remaining bytes we're allowed to send.
	int bucket_bytes;
	int cidx;
public:
	LV3D0_ClientManager_toMaster() {
		bucket_bytes=LV3D0_toMaster_bytesMax;
		cidx=CcdCallOnConditionKeep(CcdPERIODIC_10ms,toMaster_fillBucket,this);
	}
	~LV3D0_ClientManager_toMaster() {
		CcdCancelCallOnConditionKeep(CcdPERIODIC_10ms,cidx);
	}
	
	/// Add a view from a new viewable.
	virtual void add(LV3D0_ViewMsg *v) {
		CkViewPrioSorter::add(v);
		progress();
	}
	/// Send everything we're ready to send.
	void progress(void) {
		// printf("progress: bucket=%d\n",bucket_bytes);
		if (CmiLongSendQueue(CmiNodeOf(masterProcessor),LV3D0_toMaster_bytesMax))
			return; /* too many outstanding messages already! */
		while ((!isEmpty()) && (bucket_bytes>0 || LV3D_disable_ship_throttle)) 
		{
			LV3D0_ViewMsg *v=extract(begin());
			bucket_bytes-=v->view_size;
			stats::get()->add(1.0,op_master_count);
			stats::get()->add(v->view_size,op_master_bytes);
			stats::get()->add(v->pixels,op_master_pixels);
			mgrProxy[masterProcessor].addView(v);
		}
	}
	/// Add bytes to our bucket, which allows us to send.
	///  Called every 10ms via the CcdPERIODIC_10ms stuff.
	void fillBucket(void) {
		bucket_bytes+=LV3D0_toMaster_bytesPer;
		if (bucket_bytes>LV3D0_toMaster_bytesMax) 
			bucket_bytes=LV3D0_toMaster_bytesMax;
		progress();
	}
};
/// Called every 10ms via CcdPERIODIC_10ms.
static void toMaster_fillBucket(void *ptr,double timer) {
	LV3D0_ClientManager_toMaster *p=(LV3D0_ClientManager_toMaster *)ptr;
	p->fillBucket();
}

/************** Shipping views to Client ***************/
/// Communication parameter: maximum number of bytes to send per client request.
int LV3D0_toClient_bytesPer=100*1024;

static stats::op_t op_client_pack=stats::time_op("client.pack","Time spent packing final client impostors");
static stats::op_t op_client_count=stats::count_op("client.count","Number of final client impostors","count");
static stats::op_t op_client_bytes=stats::count_op("client.bytes","Number of final client bytes","bytes");
static stats::op_t op_client_pixels=stats::count_op("client.pixels","Final client impostor pixels","pixels");

/**
 * A LV3D0_ClientManager buffers up outgoing views before
 * they're requested by, and sent off to, a client.
 */
class LV3D0_ClientManager_toClient : public LV3D0_ClientManager
{
	typedef CkViewPrioSorter super;
	bool hasDelayed;
	CcsDelayedReply delayedReply;
	
	/// Emptiness detection (used for QD, below)
	CkVec<CkCallback> emptyCallbacks;
	void checkEmpty(void) {
		if (isEmpty()) { /* no views left-- send all empty callbacks */
			for (int i=0;i<emptyCallbacks.size();i++)
				emptyCallbacks[i].send();
			emptyCallbacks.resize(0);
		}
	}
	
	/// Pack up and send off up to 100KB of stored views:
	void sendReply(CcsDelayedReply repl) {
		stats::op_sentry stats_sentry(op_client_pack);
		
	// Figure out how many views we can afford to send at once:
		int len=4; /* leave room for the "length" field */
		int n=0;
		iterator it=begin();
		while (it!=end() && (len<LV3D0_toClient_bytesPer || LV3D_disable_ship_throttle)) {
			LV3D0_ViewMsg *v=(*it++).first.v;
			len+=v->view_size;
			n++;
		}
	
	// Pack the views into a network buffer:
		// CmiPrintf("[OUT] Sending off %d new views (%d bytes)\n",n,len);
		
		// FIXME: special case this when n==1.
		char *retMsg=new char[len];
		PUP_toNetwork_pack pp(retMsg);
		pp|n;
		for (int i=0;i<n;i++) {
			LV3D0_ViewMsg *v=extract(begin());
			pp(v->view,v->view_size);
			stats::get()->add(v->pixels,op_client_pixels);
			delete v;
		}
		
		if (len!=pp.size()) {
			CkError("Sizing pup was %d bytes; packing pup was %d!\n",
				len,pp.size());
			CkAbort("Pup size mismatch (logic error) in LV3D0_!\n");
		}

		stats::get()->add(1.0,op_client_count);
		stats::get()->add(len,op_client_bytes);
		CcsSendDelayedReply(repl,len,retMsg);
		delete[] retMsg;
		
		// CmiPrintf("[OUT] Done sending off %d views\n",n);
		checkEmpty();
	}
	
public:
	LV3D0_ClientManager_toClient() 
	{
		hasDelayed=false;
	}
	
	/// A local object has a new view.
	///  We are passed in our reference to this object.
	virtual void add(LV3D0_ViewMsg *v) {
		super::add(v);
		
		if (hasDelayed) { //There's already somebody waiting
			hasDelayed=false;
			sendReply(delayedReply);
		}
	}
	
	/// A client is requesting the latest views
	///   This routine must be called from a CCS handler.
	virtual void getViews(void) {
		if (isEmpty()) { //Nothing to send yet-- wait for it
			// printf("[OUT] Views requested, but none available\n");
			hasDelayed=true;
			delayedReply=CcsDelayReply();
		}
		else 
		{
			sendReply(CcsDelayReply());
		}	
	}
	
	/// Call this callback as soon as we have nothing further to send.
	virtual void whenEmptyCallback(const CkCallback &cb) {
		emptyCallbacks.push_back(cb);
		checkEmpty();
	}
};

/** 
 * The LV3D0_Manager group stores up outgoing views before
 * they're sent off to clients.
 */ 
class LV3D0_Manager : public CBase_LV3D0_Manager {
	/// Given a clientID, look up the corresponding LV3D0_ClientManager
	CkHashtableT<CkHashtableAdaptorT<int>,LV3D0_ClientManager *> clientTable;
	
	/// Next unassigned clientID.
	int nextClientID;
public:
	LV3D0_Manager(void);
	
	/// Create a new client.  Called only on masterProcessor.
	///   Returns new client's clientID.
	int newClient(void);
	
	/// Get this client manager
	LV3D0_ClientManager *getClient(int clientID);
	
	/// This client is requesting the latest views.
	///  This routine must be called from a CCS handler.
	///  This routine is only called on processor 0.
	void getViews(int clientID) {
		getClient(clientID)->getViews();
	}
	
	/// A local or remote viewable is adding this view for this client.
	inline void addView(LV3D0_ViewMsg *m) {
		getClient(m->clientID)->add(m);
	}
};


LV3D0_Manager::LV3D0_Manager(void)
{
	mgrProxy=thisgroup;
	nextClientID=1;
}

int LV3D0_Manager::newClient(void)
{
	return nextClientID++;
}

/// Get this client manager
LV3D0_ClientManager *LV3D0_Manager::getClient(int clientID)
{
	LV3D0_ClientManager *m=clientTable.get(clientID);
	if (m==NULL) {
		if (CkMyPe()==masterProcessor) 
			m=new LV3D0_ClientManager_toClient;
		else
			m=new LV3D0_ClientManager_toMaster;
		clientTable.put(clientID)=m;
	}
	return m;
}

static stats::op_t op_deposit_views=stats::count_op("deposit.views","CkView count","CkViews");
static stats::op_t op_deposit_bytes=stats::count_op("deposit.bytes","CkView sizes","bytes");
static stats::op_t op_deposit_pixels=stats::count_op("deposit.pixels","CkView pixels","pixels");

/***** Saving Views to File ********/
/// Destination to write incoming views to, or 0 if none (the normal case).
CkpvStaticDeclare(FILE *,LV3D_save_views);
static double LV3D_save_viewStart=0; ///< Time at view start
static char *LV3D_copy_view_src=0, *LV3D_copy_view_dest=0;
static stats::op_t op_save=stats::time_op("save.time","Time spent saving views to disk");

static void LV3D_save_init(void) {
	if (LV3D_copy_view_src==0) return;
	if (CkpvAccess(LV3D_save_views)) { /* file already open: close and re-open */
		fclose(CkpvAccess(LV3D_save_views));
	}
	
	char fName[1024];
	sprintf(fName,LV3D_copy_view_src,CkMyPe());
	FILE *f=fopen(fName,"wb");
	if (f==NULL) CmiAbort("Couldn't create save view file!\n");
	CkpvAccess(LV3D_save_views)=f;
	CkPrintf("Created views file %s\n",fName);
}

static void LV3D_save_start(void)
{
	if (!LV3D_dosave_views) return;
	LV3D_save_init();
	LV3D_save_viewStart=CkWallTimer(); //< HACK!
}

struct savedViewRecord {
public:
	double t; /* Time view was rendered */
	int view_size; /* size, in bytes, of view buffer */
	void pup(PUP::er &p) {
		p|t;
		p|view_size;
	}
};

static int LV3D_save_view(LV3D0_ViewMsg *v) {
	if (!LV3D_dosave_views) return 0;
	if (!CkpvAccess(LV3D_save_views)) return 0;
	stats::op_sentry stats_sentry(op_save);
	savedViewRecord rec;
	rec.t=CkWallTimer()-LV3D_save_viewStart;
	rec.view_size=v->view_size;
	enum {bufLen=sizeof(rec)};
	char buf[bufLen];
	PUP_toNetwork_pack p(buf); p|rec;
	FILE *f=CkpvAccess(LV3D_save_views);
	if (1!=fwrite(buf,p.size(),1,f)) CmiAbort("Can't write header to saved view file!\n");
	if (1!=fwrite(v->view,v->view_size,1,f)) CmiAbort("Can't write view to saved view file!\n");
	delete v;
	return 1;
}

static void LV3D_save_finish(void) {
	if (!LV3D_dosave_views) return;
	if (!CkpvAccess(LV3D_save_views)) return;
	fclose(CkpvAccess(LV3D_save_views));
	CkpvAccess(LV3D_save_views)=0;
	if (LV3D_copy_view_dest) { /* Copy view file to dest directory */
		char fSrc[1024], fDest[1024], cmd[2048];
		sprintf(fSrc,LV3D_copy_view_src,CkMyPe());
		sprintf(fDest,LV3D_copy_view_dest,CkMyPe());
		sprintf(cmd,"cp '%s' '%s' && rm '%s'", fSrc,fDest, fSrc);
		CkPrintf("Copying views file from %s to %s\n",fSrc,fDest);
		system(cmd);
		CkPrintf("Views file copied.\n");
	}
}


/**
  Send this view back to this client.
  You must set the view's id and prio fields.
 */
void LV3D0_Deposit(CkView *v,int clientID) {
	stats::stats *s=stats::get();
	s->add(1.0,op_deposit_views);
	s->add(v->pixels,op_deposit_pixels);
	LV3D0_ViewMsg *vm=LV3D0_ViewMsg::new_(v);
	if (LV3D_save_view(vm)) return;
	s->add(vm->view_size,op_deposit_bytes);
	if (LV3D_disable_ship) {delete vm; return;}
	vm->clientID=clientID;
	mgrProxy.ckLocalBranch()->addView(vm);
}


/**************** Performance collection ***************/

static stats::op_t op_pes=stats::count_op("cmi.pes","Processors","pes");
static stats::op_t op_time=stats::time_op("cmi.time","Elapsed wall-clock time");
static stats::op_t op_unknown=stats::time_op("cmi.unknown","Unaccounted-for time");
static stats::op_t op_idle=stats::time_op("cmi.idle","Time spent waiting for data");


static CcsDelayedReply statsReply;

/// Reduction handler, used to print statistics.
static void printStats(void *rednMsg) {
	CkReductionMsg *m=(CkReductionMsg *)rednMsg;
	
	// FIXME: make stats::print return a std::string, instead
	//  of this horrible "print to a file and read it back" business.
	
	char tmpFileName[100];
	sprintf(tmpFileName,"/tmp/stats.%d.%d",CkMyPe(),(int)getpid());
	FILE *f=fopen(tmpFileName,"w");
	int len=0; void *buf=0;
	if (f!=NULL) {
		/* write stats to temp file */
		const stats::stats *s=(const stats::stats *)m->getData();
		s->print(f,"total",1.0);
		s->print(f,"per_second",1.0/s->get(op_time));
		s->print(f,"per_pe-second",1.0/(CkNumPes()*s->get(op_time)));
		fclose(f);
		
		/* Read stats back */
		f=fopen(tmpFileName,"r");
		fseek(f,0,SEEK_END);
		len=ftell(f);
		buf=malloc(len);
		fseek(f,0,SEEK_SET);
		fread(buf,1,len,f);
		fclose(f);
		
		/* print to screen */
		write(1,buf,len);
		printf("\n");
	
	}
	unlink(tmpFileName);
	CcsSendDelayedReply(statsReply,len,buf);
	free(buf);
	delete m;
}

/// Ccd idle handler: notifies stats collection of our idleness.
static void perfmanager_stats_idle(void *ptr,double timer)
{
	stats::swap(op_idle);
}

class LV3D_PerfManager : public CBase_LV3D_PerfManager {
	double startTime;
public:
	LV3D_PerfManager(void) {
		zero();
		CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,perfmanager_stats_idle,0);
		CcdCallOnConditionKeep(CcdPROCESSOR_END_IDLE,perfmanager_stats_idle,0);
		
		/* Don't let the load balancer move stuff onto node 0: */
		char *bitmap=new char[CkNumPes()];
		for (int i=0;i<CkNumPes();i++)
			bitmap[i]=(i!=masterProcessor);
		set_avail_vector(bitmap);
	}
	/** Zero out collected statistics.  This is broadcast before each run. 
	*/
	void zero(void) { 
		stats::stats *s=stats::get();
		stats::swap(op_unknown); // so unknown gets zerod out
		s->zero();
		s->add(1.0,op_pes);
		startTime=stats::time();
		stats::swap(op_unknown);
		LV3D_save_start();
	}
	/** Contribute current stats to reduction 
	*/
	void collect(void) { 
		stats::stats *s=stats::get();
		if (CkMyPe()==0) s->set(stats::time()-startTime,op_time);
		else s->set(0,op_time);
		stats::swap(op_unknown);
		contribute(sizeof(double)*stats::op_len,&s->t[0],CkReduction::sum_double,
			CkCallback(printStats));
		LV3D_save_finish();
		zero();
	}
	void traceOn(void) {
		traceBegin();
	}
	void startBalance(void) {
		LBClearLoads();
		LBTurnInstrumentOn();
	}
	void doneBalance(void) {
		LBTurnInstrumentOff();
	}
	void throttle(int throttleOn) { LV3D_disable_ship_throttle=!throttleOn; }
};


/*************** CCS Interface ***************/
//This only ever gets set on processor 0 (FIXME: checkpointing)
static LV3D_Universe *theUniverse=0;
static LV3D_ServerMgr *theMgr=0;
static CProxy_LV3D_PerfManager perfMgr;

LV3D_ServerMgr::~LV3D_ServerMgr() {}

/**
"lv3d_setup" CCS handler:
	Request a new clientID and universe.

Outgoing request is empty.
Response is a 1-int clientID followed by a PUP::able universe.
*/
extern "C" void LV3D0_setup(char *msg) {
	CmiFree(msg);
	int clientID=mgrProxy.ckLocalBranch()->newClient();
	PUP_toNetwork_sizer sp;
	sp|clientID;
	sp|theUniverse;
	unsigned char *buf=new unsigned char[sp.size()];
	PUP_toNetwork_pack pp(buf);
	pp|clientID;
	pp|theUniverse;
	CcsSendReply(sp.size(),buf);
	theMgr->newClient(clientID);
	CmiPrintf("Registered (client %d)\n",clientID);
	delete[] buf;
	perfMgr.zero(); // New client implicitly zeros out stats.
}


static stats::op_t op_view_count=stats::count_op("client.views","New viewpoints","Views");

/**
"lv3d_newViewpoint" CCS handler:
	Sent by the client when his viewpoint changes,
to request updated views of the objects.

Outgoing request is an integer clientID, frameID, 
and a CkViewpoint. There is no response.
*/
extern "C" void LV3D0_newViewpoint(char *msg) {
	PUP_toNetwork_unpack p(&msg[CmiMsgHeaderSizeBytes]);
	LV3D_ViewpointMsg *m=new LV3D_ViewpointMsg;
	p|m->clientID;
	p|m->frameID;
	m->viewpoint.pup(p);
	// CmiPrintf("New user viewpoint (client %d, frame %d)\n",clientID,frameID);
	theMgr->newViewpoint(m);
	stats::get()->add(1.0,op_view_count);
}

/**
"lv3d_getViews" CCS handler:
	Sent by the client to request the actual images
of any updated object views.  These come in via LV3D0_Deposit.

Outgoing request is a single integer, the client ID
Incoming response is a set of updated CkViews:
    int n; p|n;
    for (int i=0;i<n;i++) p|view[i];
*/
extern "C" void LV3D0_getViews(char *msg) {
	PUP_toNetwork_unpack p(&msg[CmiMsgHeaderSizeBytes]);
	int clientID=0; p|clientID; CmiFree(msg);
	mgrProxy.ckLocalBranch()->getViews(clientID);
}

/*
"lv3d_qd" CCS handler:
	Sent by the client to wait until all views have been
updated and sent back.  No input or output data: control flow only.
*/
struct lv3d_qdState {
	/// ID of requesting client
	int clientID;
	
	/// CCS reply to send back to.
	CcsDelayedReply reply;
};

static void qdDoneFn(void *param,void *msg);
static void emptyDoneFn(void *param,void *msg);

extern "C" void LV3D0_qd(char *msg) /* stage 1 */
{
	lv3d_qdState *s=new lv3d_qdState;
	PUP_toNetwork_unpack p(&msg[CmiMsgHeaderSizeBytes]);
	p|s->clientID;
	s->reply=CcsDelayReply();
	CmiFree(msg);
	CkCallback cb(qdDoneFn,s);
	CkStartQD(cb); /* finish CCS reply after quiescence */
}
static void qdDoneFn(void *param,void *msg)  /* stage 2 */
{
	lv3d_qdState *s=(lv3d_qdState *)param;
	mgrProxy.ckLocalBranch()->getClient(s->clientID)->whenEmptyCallback(
		CkCallback(emptyDoneFn,s));
}
static void emptyDoneFn(void *param,void *msg) /* stage 3 */
{
	lv3d_qdState *s=(lv3d_qdState *)param;
	CcsSendDelayedReply(s->reply,0,0);
	delete s;
}

/**
"lv3d_flush" CCS handler:
	Throw away views for this client.

Outgoing request a 1-int clientID.
*/
extern "C" void LV3D0_flush(char *msg) {
	int clientID=0;
	PUP_toNetwork_unpack p(&msg[CmiMsgHeaderSizeBytes]);
	p|clientID;
	CmiFree(msg);
	theMgr->newClient(clientID);
}

/*
"lv3d_startbal" CCS handler:
	Begin a load balancing phase.  Normally followed by 
	some rendering, then an endbalance.
*/
extern "C" void LV3D0_startbalance(char *msg) 
{
	CkPrintf("CCS call to LV3D0_startbalance\n");
	perfMgr.startBalance();
	CmiFree(msg);
}
/*
"lv3d_endbal" CCS handler:
	Go to sync.  Exactly one end must follow 
	each startbalance call.
*/
extern "C" void LV3D0_endbalance(char *msg) 
{
	CkPrintf("CCS call to LV3D0_endbalance\n");
	theMgr->doBalance();
	perfMgr.doneBalance();
	LV3D0_qd(msg); /* wait for quiescence */
}

/*
"lv3d_quit" CCS handler:
	Shut down the program.
*/
extern "C" void LV3D0_quit(char *msg) 
{
	CkPrintf("Exiting: CCS call to LV3D0_quit\n");
	CcsSendReply(0,0);
	CmiFree(msg);
	CkExit();
}
/// lv3d_zero CCS handler: clear all collected statistics
extern "C" void LV3D0_zero(char *msg) 
{
	CkPrintf("Zeroing statistics\n");
	perfMgr.zero();
	CmiFree(msg);
}
/// lv3d_stats CCS handler: sum and print all collected statistics, then zero
extern "C" void LV3D0_stats(char *msg) 
{
	CkPrintf("Printing statistics\n");
	statsReply=CcsDelayReply();
	perfMgr.collect();
	CmiFree(msg);
}
/// lv3d_trace CCS handler: turn on tracing
extern "C" void LV3D0_trace(char *msg) 
{
	CkPrintf("Tracing turned on\n");
	perfMgr.traceOn();
	CmiFree(msg);
}
extern "C" void LV3D0_throttle0(char *msg) { perfMgr.throttle(0); CmiFree(msg); }
extern "C" void LV3D0_throttle1(char *msg) { perfMgr.throttle(1); CmiFree(msg); }

/**
Register for libsixty redraw requests.  This routine
must be called exactly once on processor 0.
*/
void LV3D0_Init(LV3D_Universe *clientUniverse,LV3D_ServerMgr *mgr)
{
	if (clientUniverse==0)
		clientUniverse=new LV3D_Universe();
	theUniverse=clientUniverse;
	theMgr=mgr;
	CcsRegisterHandler("lv3d_setup",(CmiHandler)LV3D0_setup);
	CcsRegisterHandler("lv3d_flush",(CmiHandler)LV3D0_flush);
	CcsRegisterHandler("lv3d_newViewpoint",(CmiHandler)LV3D0_newViewpoint);
	CcsRegisterHandler("lv3d_getViews",(CmiHandler)LV3D0_getViews);
	CcsRegisterHandler("lv3d_qd",(CmiHandler)LV3D0_qd);
	CcsRegisterHandler("lv3d_startbal",(CmiHandler)LV3D0_startbalance);
	CcsRegisterHandler("lv3d_endbal",(CmiHandler)LV3D0_endbalance);
	CcsRegisterHandler("lv3d_quit",(CmiHandler)LV3D0_quit);
	CcsRegisterHandler("lv3d_zero",(CmiHandler)LV3D0_zero);
	CcsRegisterHandler("lv3d_stats",(CmiHandler)LV3D0_stats);
	CcsRegisterHandler("lv3d_trace",(CmiHandler)LV3D0_trace);
	CcsRegisterHandler("lv3d_throttle0",(CmiHandler)LV3D0_throttle0);
	CcsRegisterHandler("lv3d_throttle1",(CmiHandler)LV3D0_throttle1);
	CProxy_LV3D0_Manager::ckNew();
	perfMgr=CProxy_LV3D_PerfManager::ckNew();
}

/**
 Per-processor initialization routine:
*/
void LV3D0_ProcInit(void) {
	CkpvInitialize(FILE *,LV3D_save_views);
	CkpvAccess(LV3D_save_views)=0;
	CmiGetArgStringDesc(CkGetArgv(),"+LV3D_save_views",&LV3D_copy_view_src,"Save rendered views to a file with this pattern.  Use like '/tmp/views.%d.pe'");
	CmiGetArgStringDesc(CkGetArgv(),"+LV3D_copy_views",&LV3D_copy_view_dest,"Copy view files to this pattern.  Use like 'views.%d.pe'");
	LV3D0_toMaster_bytesPer=LV3D0_toMaster_bytesPer/CkNumPes();
	LV3D0_toMaster_bytesMax=LV3D0_toMaster_bytesMax/CkNumPes();
}
void LV3D0_NodeInit(void) {
	CkViewNodeInit();
}


#include "lv3d0.def.h"
#include "liveViz3d.def.h"

