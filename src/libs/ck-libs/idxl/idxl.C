/**
IDXL--Index List communication library.

This is a low-level bare-bones communication library.
The basic primitive is an "Index List", a list of user
array entries to send and receive.  This basic communication
primitive is enough to represent (for example) FEM shared
nodes or an FEM ghost layer; the ghost cells in a stencil-based
CFD computation, etc.

Orion Sky Lawlor, olawlor@acm.org, 1/7/2003
*/
#include "idxl.h"
#include "idxl.decl.h"

class IDXL_DataMsg : public CMessage_IDXL_DataMsg
{
 public:
  int seqnum, tag; //Sequence number and user tag
  int from; //Source's chunk number
  int length; //Length in bytes of below array
  int nSto; //Number of separate send/recv entries in message
  void *data; // User data
  double alignPad; //Makes sure this structure is double-aligned
  
  IDXL_DataMsg(int s,int t, int f,int l, int sto) 
    :seqnum(s), tag(t), from(f), length(l), nSto(sto) 
  	{ data = (void*) (this+1); }
  IDXL_DataMsg(void) { data = (void*) (this+1); }
  static void *pack(IDXL_DataMsg *);
  static IDXL_DataMsg *unpack(void *);
  static void *alloc(int, size_t, int*, int);
};


//_idxlptr gives the current chunk
CtvStaticDeclare(IDXL_Chunk*, _idxlptr);
void IDXLnodeInit(void) {
	CtvInitialize(IDXL_Chunk*, _idxlptr);
}

void IDXL_Chunk::setupThreadPrivate(CthThread forThread) {
	CtvAccessOther(forThread,_idxlptr)=this;
}
IDXL_Chunk *IDXL_Chunk::lookup(const char *callingRoutine) {
	if(TCharm::getState()!=inDriver) 
		IDXL_Abort(callingRoutine,"Can only be called from driver (from parallel context)");
	return CtvAccess(_idxlptr);
}
void IDXL_Abort(const char *callingRoutine,const char *msg,int m0,int m1,int m2)
{
	char msg1[1024], msg2[1024];
	sprintf(msg1,msg,m0,m1,m2);
	sprintf(msg2,"Fatal error in IDXL routine %s:\n%s",callingRoutine,msg1);
	CkAbort(msg2);
}

IDXL_Chunk::IDXL_Chunk(const CkArrayID &threadArrayID) 
	:super(threadArrayID)
{
	tcharmClientInit();
	updateSeqnum=1000;
	init();
}
IDXL_Chunk::IDXL_Chunk(CkMigrateMessage *m) :super(m)
{
	init();
}
void IDXL_Chunk::init(void) {
	blockedOnComm=NULL;
	for (int lat=0;lat<LAST_IDXL;lat++) idxls[lat]=NULL;
}

void IDXL_Chunk::pup(PUP::er &p) {
	super::pup(p);
	
	p|layouts;
	p|messages;
	p|updateSeqnum;
	if (!currentComm.isDone()) CkAbort("Cannot migrate with ongoing IDXL communication");
	if (blockedOnComm!=NULL) CkAbort("Cannot migrate while blocked in IDXL communication");
	
	//Pack the dynamic IDXLs (static IDXLs must re-add themselves)
	int lat, nDynamic=0;
	if (!p.isUnpacking()) //Count the number of non-NULL idxls:
		for (lat=STATIC_IDXL;lat<LAST_IDXL;lat++) if (idxls[lat]) nDynamic++;
	p|nDynamic;
	if (p.isUnpacking()) {
		for (int d=0;d<nDynamic;d++) //Loop over non-NULL IDXLs
		{
			p|lat;
			idxls[lat]=new IDXL;
			p|*idxls[lat];
		}
	} else /* packing */ {
		for (lat=STATIC_IDXL;lat<LAST_IDXL;lat++) //Loop over non-NULL IDXLs
		if (idxls[lat]) {
			p|lat;
			p|*idxls[lat];
		}
	}
}

IDXL_Chunk::~IDXL_Chunk() {
}

/****** IDXL List ******/
/// Dynamically create a new empty IDXL:
IDXL_t IDXL_Chunk::addDynamic(void) 
{ //Pick the next free dynamic index:
	for (int ret=STATIC_IDXL;ret<LAST_IDXL;ret++)
		if (idxls[ret]==NULL) {
			idxls[ret]=new IDXL;
			idxls[ret]->allocateDual(); //FIXME: add way to allocate single, too
			return FIRST_IDXL+ret;
		}
	CkAbort("Ran out of room in (silly fixed-size) IDXL table");
	return -1; //<- for whining compilers
}
/// Register this statically-allocated IDXL at this existing index
IDXL_t IDXL_Chunk::addStatic(IDXL *idx,IDXL_t at) {
	if (at==-1) { //Pick the next free static index
		for (int ret=0;ret<STATIC_IDXL;ret++)
			if (idxls[ret]==NULL) {
				idxls[ret]=idx;
				return FIRST_IDXL+ret;
			}
		CkAbort("Ran out of room in (silly fixed-size static) IDXL table");
	}
	else /* at!=-1 */ { //User provided a (previously returned) index 
		if (at<FIRST_IDXL || at>=FIRST_IDXL+STATIC_IDXL)
			CkAbort("Provided bad fixed address to IDXL_Chunk::add!");
		int lat=at-FIRST_IDXL;
		if (idxls[lat]!=NULL)
			CkAbort("Cannot re-use a fixed IDXL address!");
		idxls[lat]=idx;
		return at;
	}
	return -1; //<- for whining compilers
}
IDXL &IDXL_Chunk::lookup(IDXL_t at,const char *callingRoutine) {
	if (at<FIRST_IDXL || at>=FIRST_IDXL+LAST_IDXL)
			IDXL_Abort(callingRoutine,"Invalid IDXL_t %d",at);
	int lat=at-FIRST_IDXL;
	return *idxls[lat];
}
const IDXL &IDXL_Chunk::lookup(IDXL_t at,const char *callingRoutine) const {
	if (at<FIRST_IDXL || at>=FIRST_IDXL+LAST_IDXL)
			IDXL_Abort(callingRoutine,"Invalid IDXL_t %d",at);
	int lat=at-FIRST_IDXL;
	return *idxls[lat];
}
void IDXL_Chunk::destroy(IDXL_t at,const char *callingRoutine) {
	lookup(at, callingRoutine); //For side-effect of checking t's validity
	int lat=at-FIRST_IDXL;
	if (lat>=STATIC_IDXL) /*dynamically allocated: destroy */
		delete idxls[lat]; 
	idxls[lat]=NULL;
}

/****** User Datatype list ******/

/// Get the currently active layout list.
///  In driver, this is IDXL_Chunk's "layouts" member.
///  Elsewhere (e.g., in init), this is a local member.
IDXL_Layout_List &getLayouts(void) {
	static IDXL_Layout_List static_list;
	if (TCharm::getState()==inDriver) return CtvAccess(_idxlptr)->layouts;
	else return static_list;
}

/**** Data Message ****/
void *
IDXL_DataMsg::pack(IDXL_DataMsg *in)
{
  return (void*) in;
}

IDXL_DataMsg *
IDXL_DataMsg::unpack(void *in)
{
  return new (in) IDXL_DataMsg;
}

void *
IDXL_DataMsg::alloc(int mnum, size_t size, int *sizes, int pbits)
{
  return CkAllocMsg(mnum, size+sizes[0], pbits);
}



/**** Messaging logic ***/

IDXL_Comm_t IDXL_Chunk::addComm(int tag,int context)
{
	if (!currentComm.isDone()) CkAbort("Cannot start two IDXL_Comms at once");
	currentComm=IDXL_Comm(updateSeqnum++,tag,context);
	return 27; //Silly: there's only one outstanding comm, so this is easy!
}
IDXL_Comm *IDXL_Chunk::lookupComm(IDXL_Comm_t uc,const char *callingRoutine)
{
	if (uc!=27) CkAbort("Invalid idxl_comm id");
	return &currentComm;
} 

IDXL_Comm::IDXL_Comm(int seqnum_, int tag_,int context_) {
	nSto=nStoAdd=nStoRecv=0;
	seqnum=seqnum_; tag=tag_; context=context_;
	nRecv=0;
	isSent=false;
	beginRecv=false;
}

// prepare to write this field to the message:
void IDXL_Comm::send(const IDXL_Side *idx,const IDXL_Layout *dtype,const void *src)
{
	if (isSent) CkAbort("Cannot call IDXL_Comm_send after IDXL_Comm_send!");
	sto[nSto++]=sto_t(idx,dtype,(void *)src,add_t); 
	nStoAdd++;
}
void IDXL_Comm::recv(const IDXL_Side *idx,const IDXL_Layout *dtype,void *dest)
{ 
	if (beginRecv) CkAbort("Cannot call IDXL_Comm_recv after a blocking call!");
	sto[nSto++]=sto_t(idx,dtype,dest,recv_t); 
	nStoRecv++;
}
void IDXL_Comm::sum(const IDXL_Side *idx,const IDXL_Layout *dtype,void *srcdest)
{ 
	if (beginRecv) CkAbort("Cannot call IDXL_Comm_sum after a blocking call!");
	sto[nSto++]=sto_t(idx,dtype,srcdest,sum_t); 
	nStoRecv++;
}
void IDXL_Comm::flush(int src,const CkArrayID &chunkArray) {
	if (isSent) CkAbort("Cannot flush the same IDXL_Comm_t more than once");
	isSent=true;
	//Send off our values to those processors that need them:
	if (nStoAdd!=1) CkAbort("FIXME: IDXL cannot do multiple-send communication");
	CProxy_IDXL_Chunk dest(chunkArray);
	sto_t *send=NULL;
	int s;
	for (s=0;s<nSto;s++)
		if (sto[s].op==add_t)
			send=&sto[s];
	const IDXL_Side *comm=send->idx;
	const IDXL_Layout &dt=*send->dtype;
	void *buf=send->data;
	for(int ll=0;ll<comm->size();ll++) {
		const IDXL_List &l=comm->getLocalList(ll);
		int msgLen=l.size()*dt.compressedBytes();
		IDXL_DataMsg *msg = new (&msgLen, 0) IDXL_DataMsg(
			seqnum, tag, src, msgLen, nStoAdd);
		dt.gather(l.size(),l.getVec(),buf,msg->data);
		dest[l.getDest()].idxl_recv(msg);
	}
	
	//Figure out how many messages we'll need to receive
	nRecv=0;
	for (s=0;s<nSto;s++)
		if (sto[s].op!=add_t)
			nRecv=sto[s].idx->size();
}

void IDXL_Chunk::idxl_recv(IDXL_DataMsg *m) {
	//FIXME: add loop over active comm's
	IDXL_Comm *c=&currentComm;
	if (c->recv(m)) 
	{ //This message actually was for this comm:
		if (c->isDone() && blockedOnComm==c) 
		{ //That's the last message we need: wake up the sleeping waitComm
			blockedOnComm=NULL;
			thread->resume();
		}
	}
	else /* not for the current update--stick it in the queue */
		messages.enq(m);
}

bool IDXL_Comm::recv(IDXL_DataMsg *msg) {
	if (seqnum!=msg->seqnum) return false; //Not my iteration
	if (tag!=msg->tag) return false; //Not my tag
	beginRecv=true;
	if (nRecv<=0) 
		CkAbort("IDXL: Received unexpected message--corrupted communication lists?");
	if (nStoRecv!=msg->nSto)
		CkAbort("IDXL: Message has unexpected number of entries.  Do send/recv's match?");
	const char *msgBuf=(const char *)msg->data;
	for (int s=0;s<nSto;s++)
	if (sto[s].op==recv_t || sto[s].op==sum_t) 
	{ // There may be something in the message for us:
		sto_t *recv=&sto[s];
		const IDXL_Side *comm=recv->idx;
		const IDXL_Layout &dt=*recv->dtype;
		int ll=comm->findLocalList(msg->from);
		if (ll==-1) continue; //This sto doesn't need anything from this processor
		const IDXL_List &l=comm->getLocalList(ll);
		int length=l.size()*dt.compressedBytes();
		if (recv->op==recv_t)
			dt.scatter(l.size(),l.getVec(),msgBuf,recv->data);
		else
			dt.scatteradd(l.size(),l.getVec(),msgBuf,recv->data);
		msgBuf+=length;
	}
	if (msgBuf!=msg->length+(const char *)msg->data)
		CkAbort("IDXL: Message length mismatch--corrupted communication lists?");
	delete msg;
	nRecv--;
	return true;
}

void IDXL_Chunk::waitComm(IDXL_Comm *comm)
{
	if (!comm->isFlushed()) 
		comm->flush(thisIndex,thisArrayID);
	if (!comm->isDone()) 
	{ 
		//Check if anything from the future queue might help:
		int i,len=messages.length();
		for (i=0;i<len;i++) {
			IDXL_DataMsg *m=messages.deq();
			if (!comm->recv(m)) // Not ours--put it back
				messages.enq(m);
		}
		
		if (!comm->isDone()) { //Future queue didn't help: block the thread
			blockedOnComm=comm;
			thread->suspend(); //Will be awakened by idxl_recv
			if (blockedOnComm!=NULL || !comm->isDone()) 
				CkAbort("IDXL thread suspended; resumed too early");
		}
	}
	//comm is now done
}

#include "idxl.def.h"

