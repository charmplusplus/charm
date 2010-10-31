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
#include "charm-api.h"

void IDXL_Abort(const char *callingRoutine,const char *msg,int m0,int m1,int m2)
{
	char msg1[1024], msg2[1024];
	sprintf(msg1,msg,m0,m1,m2);
	sprintf(msg2,"Fatal error in IDXL routine %s:\n%s",callingRoutine,msg1);
	CkAbort(msg2);
}

CDECL void pupIDXL_Chunk(pup_er cp) {
	PUP::er &p=*(PUP::er *)cp;
	IDXL_Chunk *c=(IDXL_Chunk *)TCHARM_Get_global(IDXL_globalID);
	if (c==NULL) {
		c=new IDXL_Chunk((CkMigrateMessage *)0);
		TCHARM_Set_global(IDXL_globalID,c,pupIDXL_Chunk);
	}
	c->pup(p);
	if (p.isDeleting()) {
		delete c;
		/// Zero out our global entry now that we're gone--
		///   this prevents use-after-delete bugs from creeping in.
		TCHARM_Set_global(IDXL_globalID,0,pupIDXL_Chunk);
	}
}
IDXL_Chunk *IDXL_Chunk::get(const char *callingRoutine) {
	IDXL_Chunk *c=getNULL();
	if(!c) IDXL_Abort(callingRoutine,"IDXL is not initialized");
	return c;
}

CDECL void
IDXL_Init(int mpi_comm) {
	if (!TCHARM_Get_global(IDXL_globalID)) {
		IDXL_Chunk *c=new IDXL_Chunk(mpi_comm);
		TCHARM_Set_global(IDXL_globalID,c,pupIDXL_Chunk);
	}
}
FORTRAN_AS_C(IDXL_INIT,IDXL_Init,idxl_init,  (int *comm), (*comm))

IDXL_Chunk::IDXL_Chunk(int mpi_comm_) 
	:mpi_comm(mpi_comm_), currentComm(0)
{
	init();
}
IDXL_Chunk::IDXL_Chunk(CkMigrateMessage *m) :currentComm(0)
{
	init();
}
void IDXL_Chunk::init(void) {
	
}

void IDXL_Chunk::pup(PUP::er &p) {
	p|mpi_comm;

	p|layouts;
	if (currentComm && !currentComm->isComplete()){
	  CkPrintf("ERROR: Cannot migrate with ongoing IDXL communication: currentComm=%p ispacking=%d isunpacking=%d\n", currentComm, (int)p.isPacking(), (int)p.isUnpacking());
	  CkAbort("Cannot migrate with ongoing IDXL communication");
	}	

	//Pack the dynamic IDXLs (static IDXLs must re-add themselves)
	int i, nDynamic=0;
	if (!p.isUnpacking()) //Count the number of non-NULL idxls:
		for (i=0;i<dynamic_idxls.size();i++) 
			if (dynamic_idxls[i]) nDynamic++;
	p|nDynamic;
	if (p.isUnpacking()) {
		for (int d=0;d<nDynamic;d++) //Loop over non-NULL IDXLs
		{
			p|i;
			dynamic_idxls[i]=new IDXL;
			p|*dynamic_idxls[i];
		}
	} else /* packing */ {
		for (i=0;i<dynamic_idxls.size();i++) //Loop over non-NULL IDXLs
		if (dynamic_idxls[i]) {
			p|i;
			p|*dynamic_idxls[i];
		}
	}
}

IDXL_Chunk::~IDXL_Chunk() {
	// we do not delete static_idxls
	for (int i=0;i<dynamic_idxls.size();i++) 
		if (dynamic_idxls[i]) delete dynamic_idxls[i];
	delete currentComm;
	currentComm = 0;
}

/****** IDXL List ******/
/// Dynamically create a new empty IDXL:
IDXL_t IDXL_Chunk::addDynamic(void) 
{ //Pick the next free dynamic index:
	IDXL *ret=new IDXL;
	ret->allocateDual(); //FIXME: add way to allocate single, too
	return IDXL_DYNAMIC_IDXL_T+storeToFreeIndex(dynamic_idxls,ret);
}
/// Register this statically-allocated IDXL at this existing index
IDXL_t IDXL_Chunk::addStatic(IDXL *idx,IDXL_t at) {
	if (at!=-1) 
	{ //User provided a (previously returned) index-- try that first
		if (at<IDXL_STATIC_IDXL_T || at>=IDXL_LAST_IDXL_T)
			CkAbort("Provided bad fixed address to IDXL_Chunk::add!");
		int lat=at-IDXL_STATIC_IDXL_T;
		while (static_idxls.size()<=lat) static_idxls.push_back(NULL);
		if (static_idxls[lat]==NULL)
		{ /* that slot is free-- re-use the fixed address */
			static_idxls[lat]=idx;
			return at;
		}
		else /* idxls[lat]!=NULL, somebody's already there-- fall through */
			at=-1;
	}
	if (at==-1) { //Pick the next free static index
		return IDXL_STATIC_IDXL_T+storeToFreeIndex(static_idxls,idx);
	}
	return -1; //<- for whining compilers
}

/// Check this IDXL for validity
void IDXL_Chunk::check(IDXL_t at,const char *callingRoutine) const {
	if (at<IDXL_DYNAMIC_IDXL_T || at>=IDXL_LAST_IDXL_T)
			IDXL_Abort(callingRoutine,"Invalid IDXL_t %d",at);
}
IDXL &IDXL_Chunk::lookup(IDXL_t at,const char *callingRoutine) {
	IDXL *ret=0;
	if (at>=IDXL_DYNAMIC_IDXL_T && at<IDXL_DYNAMIC_IDXL_T+dynamic_idxls.size())
		ret=dynamic_idxls[at-IDXL_DYNAMIC_IDXL_T];
	else if (at>=IDXL_STATIC_IDXL_T && at<IDXL_STATIC_IDXL_T+static_idxls.size())
		ret=static_idxls[at-IDXL_STATIC_IDXL_T];
	if (ret==NULL) 
		IDXL_Abort(callingRoutine,"Trying to look up invalid IDXL_t %d",at);
	return *ret;
}
const IDXL &IDXL_Chunk::lookup(IDXL_t at,const char *callingRoutine) const {
	IDXL_Chunk *dthis=(IDXL_Chunk *)this; // Call non-const version
	return dthis->lookup(at,callingRoutine);
}
void IDXL_Chunk::destroy(IDXL_t at,const char *callingRoutine) {
	IDXL **ret=NULL;
	if (at>=IDXL_DYNAMIC_IDXL_T && at<IDXL_DYNAMIC_IDXL_T+dynamic_idxls.size())
		ret=&dynamic_idxls[at-IDXL_DYNAMIC_IDXL_T];
	else if (at>=IDXL_STATIC_IDXL_T && at<IDXL_STATIC_IDXL_T+static_idxls.size())
		ret=&static_idxls[at-IDXL_STATIC_IDXL_T];
	if (ret==NULL)
		IDXL_Abort(callingRoutine,"Trying to destroy invalid IDXL_t %d",at);
	if (*ret==NULL)
		IDXL_Abort(callingRoutine,"Trying to destroy already deleted IDXL_t %d",at);
	if (at<IDXL_STATIC_IDXL_T)
		delete *ret; /* only destroy dynamically allocated IDXLs */
	*ret=NULL;
}

/****** User Datatype list ******/

/// Get the currently active layout list.
IDXL_Layout_List &IDXL_Layout_List::get(void) 
{
	return IDXL_Chunk::get("IDXL_Layouts::get")->layouts; 
}

/**** Messaging logic ***/

IDXL_Comm_t IDXL_Chunk::addComm(int tag,int context)
{
	if (currentComm && !currentComm->isComplete()) CkAbort("Cannot start two IDXL_Comms at once");
	if (context==0) context=mpi_comm;
	if (currentComm==0) 
		currentComm=new IDXL_Comm(tag,context);
	else 
		currentComm->reset(tag,context);
	return 27; //FIXME: there's only one outstanding comm!
}
IDXL_Comm *IDXL_Chunk::lookupComm(IDXL_Comm_t uc,const char *callingRoutine)
{
	if (uc!=27) CkAbort("Invalid idxl_comm id");
	return currentComm;
}

IDXL_Comm::IDXL_Comm(int tag_,int context) {
	reset(tag_,context);
}
void IDXL_Comm::reset(int tag_,int context) {
	tag=tag_;
	if (context==0) comm=MPI_COMM_WORLD;
	else comm=(MPI_Comm)context; /* silly: not all MPI's use "int" for MPI_Comm */
	sto.resize(0);
	nMsgs=0;
	/* don't delete the msg array, because we want to avoid
	   reallocating its message buffers whenever possible. */
	msgReq.resize(0);
	msgSts.resize(0);
	
	isPost=false;
	isDone=false;
}
IDXL_Comm::~IDXL_Comm() {
	for (int i=0;i<msg.size();i++)
		delete msg[i];
}


// prepare to write this field to the message:
void IDXL_Comm::send(const IDXL_Side *idx,const IDXL_Layout *dtype,const void *src)
{
	if (isPost) CkAbort("Cannot call IDXL_Comm_send after IDXL_Comm_flush!");
	sto.push_back(sto_t(idx,dtype,(void *)src,send_t)); 
}
void IDXL_Comm::recv(const IDXL_Side *idx,const IDXL_Layout *dtype,void *dest)
{ 
	if (isPost) CkAbort("Cannot call IDXL_Comm_recv after IDXL_Comm_flush!");
	sto.push_back(sto_t(idx,dtype,dest,recv_t)); 
}
void IDXL_Comm::sum(const IDXL_Side *idx,const IDXL_Layout *dtype,void *srcdest)
{ 
	if (isPost) CkAbort("Cannot call IDXL_Comm_sum after IDXL_Comm_flush!");
	sto.push_back(sto_t(idx,dtype,srcdest,sum_t)); 
}

void IDXL_Comm::post(void) {
	if (isPost) CkAbort("Cannot post the same IDXL_Comm_t more than once");
	isPost=true;
	
	//Post all our sends and receives:
	nMsgs=0;
	for (int s=0;s<sto.size();s++) {
		const IDXL_Side *idx=sto[s].idx;
		const IDXL_Layout *dtype=sto[s].dtype;
		for (int ll=0;ll<idx->size();ll++) {
			const IDXL_List &l=idx->getLocalList(ll);
			
			// Create message struct
			++nMsgs;
			if (nMsgs>msg.size()) {
				msg.resize(nMsgs);
				msg[nMsgs-1]=new msg_t;
			}
			msg_t *m=msg[nMsgs-1];
			m->sto=&sto[s];
			m->ll=ll;
			
			// Allocate storage for message data
			int len=l.size()*dtype->compressedBytes();
			m->allocate(len);
			
			// Copy data and post MPI request
			MPI_Request req;
			switch (sto[s].op) {
			case send_t:
				sto[s].dtype->gather(l.size(),l.getVec(),sto[s].data,m->getBuf());
				MPI_Isend(m->getBuf(),len,MPI_BYTE,l.getDest(),tag,comm,&req);
				break;
			case recv_t:case sum_t:
				MPI_Irecv(m->getBuf(),len,MPI_BYTE,l.getDest(),tag,comm,&req);
				break;
			};
			msgReq.push_back(req);
		}
	}
}

void IDXL_Comm::wait(void) {
	if (!isPosted()) post();
	CkAssert(msg.size()>=nMsgs);
	CkAssert(msgReq.size()==nMsgs);
        if (nMsgs == 0) { isDone=true; return; }
	msgSts.resize(nMsgs);
	MPI_Waitall(nMsgs,&msgReq[0],&msgSts[0]);
	//Process all received messages:
	for (int im=0;im<nMsgs;im++) {
		msg_t *m=msg[im];
		sto_t *s=m->sto;
		const IDXL_List &l=s->idx->getLocalList(m->ll);
		switch (s->op) {
		case send_t: /* nothing else to do */
			break;
		case recv_t:
			s->dtype->scatter(l.size(),l.getVec(),m->getBuf(),s->data);
			break;
		case sum_t:
			s->dtype->scatteradd(l.size(),l.getVec(),m->getBuf(),s->data);
			break;
		};
	}
	isDone=true;
}

void IDXL_Chunk::waitComm(IDXL_Comm *comm)
{
	comm->wait();
}

