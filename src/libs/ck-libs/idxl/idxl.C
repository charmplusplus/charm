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
	for (int lat=0;lat<LAST_IDXL;lat++) idxls[lat]=NULL;
}

void IDXL_Chunk::pup(PUP::er &p) {
	p|mpi_comm;

	p|layouts;
	if (currentComm && !currentComm->isComplete()) CkAbort("Cannot migrate with ongoing IDXL communication");
	
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
	for (int i=STATIC_IDXL;i<LAST_IDXL;i++) //Loop over non-NULL IDXLs
		if (idxls[i]) delete idxls[i];
	delete currentComm;
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
	if (at!=-1) 
	{ //User provided a (previously returned) index-- try that first
		if (at<FIRST_IDXL || at>=FIRST_IDXL+STATIC_IDXL)
			CkAbort("Provided bad fixed address to IDXL_Chunk::add!");
		int lat=at-FIRST_IDXL;
		if (idxls[lat]==NULL)
		{ /* that slot is free-- re-use the fixed address */
			idxls[lat]=idx;
			return at;
		}
		else /* idxls[lat]!=NULL, somebody's already there-- fall through */
			at=-1;
	}
	if (at==-1) { //Pick the next free static index
		for (int ret=0;ret<STATIC_IDXL;ret++)
			if (idxls[ret]==NULL) {
				idxls[ret]=idx;
				return FIRST_IDXL+ret;
			}
		CkAbort("Ran out of room in (silly fixed-size static) IDXL table");
	}
	return -1; //<- for whining compilers
}

/// Check this IDXL for validity
void IDXL_Chunk::check(IDXL_t at,const char *callingRoutine) const {
	if (at<FIRST_IDXL || at>=FIRST_IDXL+LAST_IDXL)
			IDXL_Abort(callingRoutine,"Invalid IDXL_t %d",at);
}
IDXL &IDXL_Chunk::lookup(IDXL_t at,const char *callingRoutine) {
	check(at,callingRoutine);
	int lat=at-FIRST_IDXL;
	return *idxls[lat];
}
const IDXL &IDXL_Chunk::lookup(IDXL_t at,const char *callingRoutine) const {
	check(at,callingRoutine);
	int lat=at-FIRST_IDXL;
	return *idxls[lat];
}
void IDXL_Chunk::destroy(IDXL_t at,const char *callingRoutine) {
	check(at,callingRoutine);
	int lat=at-FIRST_IDXL;
	if (lat>=STATIC_IDXL) /*dynamically allocated: destroy */
		delete idxls[lat]; 
	idxls[lat]=NULL;
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
	nSto=nMsg=0;
	isPost=false;
	isDone=false;
}


// prepare to write this field to the message:
void IDXL_Comm::send(const IDXL_Side *idx,const IDXL_Layout *dtype,const void *src)
{
	if (isPost) CkAbort("Cannot call IDXL_Comm_send after IDXL_Comm_flush!");
	sto[nSto++]=sto_t(idx,dtype,(void *)src,send_t); 
}
void IDXL_Comm::recv(const IDXL_Side *idx,const IDXL_Layout *dtype,void *dest)
{ 
	if (isPost) CkAbort("Cannot call IDXL_Comm_recv after IDXL_Comm_flush!");
	sto[nSto++]=sto_t(idx,dtype,dest,recv_t); 
}
void IDXL_Comm::sum(const IDXL_Side *idx,const IDXL_Layout *dtype,void *srcdest)
{ 
	if (isPost) CkAbort("Cannot call IDXL_Comm_sum after IDXL_Comm_flush!");
	sto[nSto++]=sto_t(idx,dtype,srcdest,sum_t); 
}

void IDXL_Comm::post(void) {
	if (isPost) CkAbort("Cannot post the same IDXL_Comm_t more than once");
	isPost=true;
	
	//Post all our sends and receives:
	nMsg=0;
	for (int s=0;s<nSto;s++) {
		const IDXL_Side *idx=sto[s].idx;
		const IDXL_Layout *dtype=sto[s].dtype;
		for (int ll=0;ll<idx->size();ll++) {
			const IDXL_List &l=idx->getLocalList(ll);
			msg_t *m=&msg[nMsg];
			m->sto=&sto[s];
			m->ll=ll;
			int len=l.size()*dtype->compressedBytes();
			m->allocate(len);
			switch (sto[s].op) {
			case send_t:
				sto[s].dtype->gather(l.size(),l.getVec(),sto[s].data,m->buf);
				MPI_Isend(m->buf,len,MPI_BYTE,l.getDest(),tag,comm,&msgReq[nMsg]);
				break;
			case recv_t:case sum_t:
				MPI_Irecv(m->buf,len,MPI_BYTE,l.getDest(),tag,comm,&msgReq[nMsg]);
				break;
			};
			nMsg++;
		}
	}
}

void IDXL_Comm::wait(void) {
	if (!isPosted()) post();
	MPI_Status sts[maxMsg];
	MPI_Waitall(nMsg,msgReq,sts);
	//Process all received messages:
	for (int im=0;im<nMsg;im++) {
		msg_t *m=&msg[im];
		sto_t *s=m->sto;
		const IDXL_List &l=s->idx->getLocalList(m->ll);
		switch (s->op) {
		case send_t: /* nothing else to do */
			break;
		case recv_t:
			s->dtype->scatter(l.size(),l.getVec(),m->buf,s->data);
			break;
		case sum_t:
			s->dtype->scatteradd(l.size(),l.getVec(),m->buf,s->data);
			break;
		};
	}
	isDone=true;
}

void IDXL_Chunk::waitComm(IDXL_Comm *comm)
{
	comm->wait();
}

