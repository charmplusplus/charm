/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "ampiimpl.h"
// for strlen
#include <string.h>

argvPupable::~argvPupable()
{
	if (!isSeparate) return;
/* //This causes a crash if the thread migrates, so just leak it (!)
	int argc=getArgc();
	for (int i=0;i<argc;i++)
		delete[] argv[i];
	delete[] argv;
*/
}

argvPupable::argvPupable(const argvPupable &p)
{
	isSeparate=true;
	int argc=p.getArgc();
	char **nu_argv=new char*[argc+1];
	for (int i=0;i<argc;i++)
	{
		int len;
		len=strlen(p.argv[i])+1;
		nu_argv[i]=new char[len];
		strcpy(nu_argv[i],p.argv[i]);
	}
	nu_argv[argc]=NULL;
	argv=nu_argv;
}

void argvPupable::pup(PUP::er &p)
{
	int argc=0;
	if (!p.isUnpacking()) argc=getArgc();
	p(argc);
	if (p.isUnpacking()) {
		argv=new char*[argc+1];
		isSeparate=true;
	}
	for (int i=0;i<argc;i++) {
		int len;
		if (!p.isUnpacking()) len=strlen(argv[i])+1;
		p(len);
		if (p.isUnpacking()) argv[i]=new char[len];
		p(argv[i],len);
	}
	if (p.isUnpacking()) argv[argc]=NULL;
}

//------------- startup -------------
static mpi_comm_structs mpi_comms;
int mpi_ncomms;
int MPI_COMM_UNIVERSE[MPI_MAX_COMM];

CDECL void MPI_Setup_Switch(void);

CtvDeclare(ampi*, ampiPtr);
static void ampiNodeInit(void)
{
  CtvInitialize(ampi *, ampiPtr);
  mpi_ncomms=0;
  for(int i=0;i<mpi_ncomms; i++)
  {
    MPI_COMM_UNIVERSE[i] = i+1;
  }
  TCharmSetFallbackSetup(MPI_Setup_Switch);
}

static void 
ampiAllReduceHandler(void *arg, void *r_msg)
{
  mpi_comm_struct *commspec = (mpi_comm_struct *) arg;
  CkReductionMsg *msg=(CkReductionMsg *)r_msg;
  
  ampi::bcastraw(msg->getData(), msg->getSize(), commspec->aid);
  delete msg;
}

void ampi::reduceResult(CkReductionMsg *msg)
{
  ampi::sendraw(MPI_REDUCE_TAG, 0, msg->getData(), msg->getSize(), 
             thisArrayID,thisIndex);
  delete msg;
}

class MPI_threadstart_t {
public:
	argvPupable args;
	MPI_MainFn fn;
	MPI_threadstart_t() {}
	MPI_threadstart_t(const argvPupable &args_,MPI_MainFn fn_)
		:args(args_), fn(fn_) {}
	void start(void) {
		char **argv=args.getArgv();
		int argc=args.getArgc();
		(fn)(argc,argv);
	}
	void pup(PUP::er &p) {
		p|args;
		p|fn;
	}
};
PUPmarshall(MPI_threadstart_t);

extern "C" void MPI_threadstart(void *data)
{
	MPI_threadstart_t t;
	pupFromBuf(data,t);
	t.start();
}

static void ampiAttach(void);

void ampiCreateMain(MPI_MainFn mainFn)
{
	int _nchunks=TCharmGetNumChunks();
	
	//Make a new threads array
	argvPupable args(TCharmArgv());
	MPI_threadstart_t s(args,mainFn);
	memBuf b; pupIntoBuf(b,s);
	TCharmCreateData( _nchunks,MPI_threadstart,
			  b.getData(), b.getSize());
}

static void ampiAttach(const char *name,int namelen)
{
        TCharmSetupCookie *tc=TCharmSetupCookie::get();
	if (!tc->hasThreads())
		CkAbort("You must create a thread array with TCharmCreate before calling MPI_Attach!\n");
	int _nchunks=tc->getNumElements();
	CkArrayID threads=tc->getThreads();

	//Allocate the next communicator  
	if(mpi_ncomms == MPI_MAX_COMM)
	{
		CkAbort("AMPI> Number of registered comm_worlds exceeded limit.\n");
	}
	int commidx=mpi_ncomms++;	
	mpi_comms[commidx].mainfunc = NULL; //mainFn;
	mpi_comms[commidx].name = new char[namelen+1];
	memcpy(mpi_comms[commidx].name, name, namelen);
	mpi_comms[commidx].name[namelen] = '\0';
	mpi_comms[commidx].nobj=_nchunks;
        
        //Create and attach the new ampi array
        CkArrayOptions opts(_nchunks);
        opts.bindTo(threads);
        CProxy_ampi arr= CProxy_ampi::ckNew(commidx,threads,opts);
	mpi_comms[commidx].aid=arr;

        tc->addClient(arr);        
}


ampi::ampi(int commidx_,CProxy_TCharm threads_)
{
  commidx = commidx_;
  msgs = CmmNew();
  nbcasts = 0;
  nrequests = 0;
  myDDT = new DDT() ;
  nirequests = 0;
  firstfree = 0;
  ampiBlockedThread=0;
  threads=threads_;
  prepareCtv();

  int i;
  for(i=0;i<100;i++) {
    irequests[i].nextfree = (i+1)%100;
    irequests[i].prevfree = ((i-1)+100)%100;
  }
  oorder = new AmpiSeqQ[numElements];
  nextseq = new int[numElements];
  for(i=0;i<numElements;i++) {
    nextseq[i] = 0;
    oorder[i].init();
  }
  thread->ready();
}
ampi::ampi(CkMigrateMessage *msg)
{
	msgs=NULL;
	ampiBlockedThread=0;
}
void ampi::ckJustMigrated(void)
{
	ArrayElement1D::ckJustMigrated();
	prepareCtv();
}
void ampi::prepareCtv(void)
{
	thread=threads[thisIndex].ckLocal();
	if (thread==NULL) CkAbort("Ampi cannot find its TCharm!\n");
	CtvAccessOther(thread->getThread(),ampiPtr) = this;   
}

ampi::~ampi()
{
  delete[] oorder;
  delete[] nextseq;
  CmmFree(msgs);
}

//------------------------ communication -----------------------
void
ampi::generic(AmpiMsg* msg)
{
  if(msg->comm == MPI_COMM_WORLD && msg->tag <= MPI_TAG_UB) {
    int src = msg->src;
    oorder[src].put(msg->seq, msg);
    while((msg=oorder[src].get())!=0) {
      inorder(msg);
    }
  } else {
    inorder(msg);
  }
  if(ampiBlockedThread)
	  thread->resume();
}

void
ampi::inorder(AmpiMsg* msg)
{
  int tags[3];
  tags[0] = msg->tag; tags[1] = msg->src; tags[2] = msg->comm;
  CmmPut(msgs, 3, tags, msg);
}

void 
ampi::send(int t, int s, void* buf, int count, int type,  int idx, int comm)
{
  CkArrayID aid = thisArrayID;
  int mycomm = MPI_COMM_WORLD;
  int seq = -1;
  if(comm != MPI_COMM_WORLD) {
    mycomm = MPI_COMM_UNIVERSE[commidx];
    aid = mpi_comms[comm-1].aid;
  } else {
    if(t <= MPI_TAG_UB) {
      seq = nextseq[idx]++;
    }
  }
  DDT_DataType *ddt = myDDT->getType(type);
  int len = ddt->getSize(count);
  AmpiMsg *msg = new (&len, 0) AmpiMsg(seq, t, s, len, mycomm);
  ddt->serialize((char*)buf, (char*)msg->data, count, 1);
  CProxy_ampi pa(aid);
  pa[idx].generic(msg);
}

void 
ampi::sendraw(int t, int s, void* buf, int len, CkArrayID aid, int idx)
{
  AmpiMsg *msg = new (&len, 0) AmpiMsg(-1, t, s, len, 0);
  memcpy(msg->data, buf, len);
  CProxy_ampi pa(aid);
  pa[idx].generic(msg);
}

void 
ampi::recv(int t, int s, void* buf, int count, int type, int comm, int *sts)
{
  int tags[3];
  AmpiMsg *msg = 0;
  DDT_DataType *ddt = myDDT->getType(type);
  int len = ddt->getSize(count);
  ampiBlockedThread=1;
  while(1) {
    tags[0] = t; tags[1] = s; tags[2] = comm;
    msg = (AmpiMsg *) CmmGet(msgs, 3, tags, sts);
    if (msg) break;
    thread->suspend();
  }
  ampiBlockedThread=0;
  if(sts)
    ((MPI_Status*)sts)->MPI_LENGTH = msg->length;
  if (msg->length > len) {
    CkError("AMPI: (type=%d, count=%d) Expecting msg of len %d, received %d\n",
            type, count, len, msg->length);
    CkAbort("Exiting.\n");
  }
  ddt->serialize((char*)buf, (char*)msg->data, msg->length/(ddt->getSize(1)), (-1));
  delete msg;
}

void 
ampi::probe(int t, int s, int comm, int *sts)
{
  int tags[3];
  AmpiMsg *msg = 0;
  while(1) {
    tags[0] = t; tags[1] = s; tags[2] = comm;
    msg = (AmpiMsg *) CmmProbe(msgs, 3, tags, sts);
    if (msg) break;
    thread->schedule();
  }
  if(sts)
    ((MPI_Status*)sts)->MPI_LENGTH = msg->length;
}

int 
ampi::iprobe(int t, int s, int comm, int *sts)
{
  int tags[3];
  AmpiMsg *msg = 0;
  tags[0] = t; tags[1] = s; tags[2] = comm;
  msg = (AmpiMsg *) CmmProbe(msgs, 3, tags, sts);
  if (msg) {
    if(sts)
      ((MPI_Status*)sts)->MPI_LENGTH = msg->length;
    return 1;
  }
  thread->schedule();
  return 0;
}

void 
ampi::barrier(void)
{
  if(thisIndex) {
    send(MPI_BARR_TAG, 0, 0, 0, 0, 0, MPI_COMM_WORLD);
    recv(MPI_BARR_TAG, 0, 0, 0, 0, MPI_COMM_WORLD);
  } else {
    int i;
    for(i=1;i<numElements;i++) 
      recv(MPI_BARR_TAG, 0, 0, 0, 0, MPI_COMM_WORLD);
    for(i=1;i<numElements;i++) 
      send(MPI_BARR_TAG, 0, 0, 0, 0, i, MPI_COMM_WORLD);
  }
}

void 
ampi::bcast(int root, void* buf, int count, int type)
{
  if(root==thisIndex) {
    int i;
    for(i=0;i<numElements;i++)
      send(MPI_BCAST_TAG, nbcasts, buf, count, type, i, MPI_COMM_WORLD);
  }
  recv(MPI_BCAST_TAG, nbcasts, buf, count, type, MPI_COMM_WORLD);
  nbcasts++;
}

void
ampi::bcastraw(void* buf, int len, CkArrayID aid)
{
  AmpiMsg *msg = new (&len, 0) AmpiMsg(-1, MPI_BCAST_TAG, 0, len, 0);
  memcpy(msg->data, buf, len);
  CProxy_ampi pa(aid);
  pa.generic(msg);
}

//------------------- maintainance -----------------
#if 0
//Need to figure out how to support checkpoint/restart properly
void ampi::checkpoint(DirMsg *msg)
{
  sprintf(str, "%s/%d", msg->dname, commidx);
  mkdir(str, 0777);
  sprintf(str, "%s/%d/%d.cpt", msg->dname, commidx, thisIndex);
  delete msg;
  CProxy_ampimain pm(ampimain::handle); 
  pm.checkpoint(); 
}

CDECL void MPI_Checkpoint(char *dirname)
{
  mkdir(dirname, 0777);
  ampi *ptr = CtvAccess(ampiPtr);
  ptr->cthread_id = CthSelf();
  int idx = ptr->thisIndex;
  CProxy_ampi aproxy(ampimain::mpi_comms[ptr->commidx].aid);
  aproxy[idx].checkpoint(new DirMsg(dirname));
  ptr->stop_running();
  CthSuspend();
  ptr = CtvAccess(ampiPtr);
  if(ptr->cthread_id != 0)
    CkAbort("cthread_id not 0 upon return !!\n");
  ptr->start_running();
}
#endif

void ampi::pup(PUP::er &p)
{
  if(!p.isUserlevel())
    ArrayElement1D::pup(p);//Pack superclass
  p(commidx);
  msgs=CmmPup((pup_er)&p,msgs);

  p|threads;

  p(nbcasts);
  // persistent comm requests will have to be re-registered after
  // migration anyway, so no need to pup them
  // migrate is called only when all irequests are complete, so no need
  // to pup them as well.
  if(p.isUnpacking())
  {
    myDDT = new DDT((void*)0);
    nrequests = 0;
    nirequests = 0;
    firstfree = 0;
    int i;
    for(i=0;i<100;i++) {
      irequests[i].nextfree = (i+1)%100;
      irequests[i].prevfree = ((i-1)+100)%100;
    }
    oorder = new AmpiSeqQ[numElements];
    nextseq = new int[numElements];
  }
  for(int i=0; i<numElements; i++)
    p | oorder[i];
  p(nextseq, numElements);
  myDDT->pup(p);
}

//------------------ External Interface -----------------

static ampi *getAmpiInstance(void) {
  ampi *ret = CtvAccess(ampiPtr);
  return ret;
}

CDECL void MPI_Migrate(void)
{
  AMPIAPI("MPI_Migrate");
  TCharmMigrate();
}

CDECL int MPI_Init(int *argc, char*** argv)
{
  AMPIAPI("MPI_Init");
  return 0;
}

CDECL int MPI_Comm_rank(MPI_Comm comm, int *rank)
{
  AMPIAPI("MPI_Comm_rank");
  *rank = TCharmElement();
  return 0;
}

CDECL
int MPI_Comm_size(MPI_Comm comm, int *size)
{
  AMPIAPI("MPI_Comm_size");
  *size = TCharmNumElements();
  return 0;
}

CDECL
int MPI_Finalize(void)
{
  AMPIAPI("MPI_Finalize");
  return 0;
}

CDECL
int MPI_Send(void *msg, int count, MPI_Datatype type, int dest, 
                        int tag, MPI_Comm comm)
{
  AMPIAPI("MPI_Send");
  ampi *ptr = CtvAccess(ampiPtr);
  ptr->send(tag, ptr->getIndex(), msg, count, type, dest, comm);
  return 0;
}

CDECL
int MPI_Ssend(void *msg, int count, MPI_Datatype type, int dest, 
                        int tag, MPI_Comm comm)
{
  AMPIAPI("MPI_Ssend");
  ampi *ptr = CtvAccess(ampiPtr);
  ptr->send(tag, ptr->getIndex(), msg, count, type, dest, comm);
  return 0;
}

CDECL
int MPI_Recv(void *msg, int count, MPI_Datatype type, int src, int tag, 
              MPI_Comm comm, MPI_Status *status)
{
  AMPIAPI("MPI_Recv");
  ampi *ptr = CtvAccess(ampiPtr);
  ptr->recv(tag,src,msg,count,type, comm, (int*) status);
  return 0;
}

CDECL
int MPI_Probe(int src, int tag, MPI_Comm comm, MPI_Status *status)
{
  AMPIAPI("MPI_Probe");
  ampi *ptr = CtvAccess(ampiPtr);
  ptr->probe(tag,src, comm, (int*) status);
  return 0;
}

CDECL
int MPI_Iprobe(int src,int tag,MPI_Comm comm,int *flag,MPI_Status *status)
{
  AMPIAPI("MPI_Iprobe");
  ampi *ptr = CtvAccess(ampiPtr);
  *flag = ptr->iprobe(tag,src,comm,(int*) status);
  return 0;
}

CDECL
int MPI_Sendrecv(void *sbuf, int scount, int stype, int dest, 
                  int stag, void *rbuf, int rcount, int rtype,
                  int src, int rtag, MPI_Comm comm, MPI_Status *sts)
{
  AMPIAPI("MPI_Sendrecv");
  return (MPI_Send(sbuf,scount,stype,dest,stag,comm) ||
          MPI_Recv(rbuf,rcount,rtype,src,rtag,comm,sts));
}

CDECL
int MPI_Barrier(MPI_Comm comm)
{
  AMPIAPI("MPI_Barrier");
  if(comm != MPI_COMM_WORLD)
  {
    CkAbort("AMPI> Cannot have global operations across communicators.\n");
  }
  ampi *ptr = CtvAccess(ampiPtr);
  ptr->barrier();
  return 0;
}

CDECL
int MPI_Bcast(void *buf, int count, MPI_Datatype type, int root, 
                         MPI_Comm comm)
{
  AMPIAPI("MPI_Bcast");
  if(comm != MPI_COMM_WORLD)
  {
    CkAbort("AMPI> Cannot have global operations across communicators.\n");
  }
  ampi *ptr = CtvAccess(ampiPtr);
  ptr->bcast(root, buf, count, type);
  return 0;
}

static CkReduction::reducerType 
getReductionType(int type, int op)
{
  CkReduction::reducerType mytype;
  switch(op) {
    case MPI_MAX :
      switch(type) {
        case MPI_FLOAT : mytype = CkReduction::max_float; break;
        case MPI_INT : mytype = CkReduction::max_int; break;
        case MPI_DOUBLE : mytype = CkReduction::max_double; break;
        default:
          ckerr << "Type " << type << " not supported." << endl;
          CmiAbort("exiting");
      }
      break;
    case MPI_MIN :
      switch(type) {
        case MPI_FLOAT : mytype = CkReduction::min_float; break;
        case MPI_INT : mytype = CkReduction::min_int; break;
        case MPI_DOUBLE : mytype = CkReduction::min_double; break;
        default:
          ckerr << "Type " << type << " not supported." << endl;
          CmiAbort("exiting");
      }
      break;
    case MPI_SUM :
      switch(type) {
        case MPI_FLOAT : mytype = CkReduction::sum_float; break;
        case MPI_INT : mytype = CkReduction::sum_int; break;
        case MPI_DOUBLE : mytype = CkReduction::sum_double; break;
        default:
          ckerr << "Type " << type << " not supported." << endl;
          CmiAbort("exiting");
      }
      break;
    case MPI_PROD :
      switch(type) {
        case MPI_FLOAT : mytype = CkReduction::product_float; break;
        case MPI_INT : mytype = CkReduction::product_int; break;
        case MPI_DOUBLE : mytype = CkReduction::product_double; break;
        default:
          ckerr << "Type " << type << " not supported." << endl;
          CmiAbort("exiting");
      }
      break;
    default:
      ckerr << "Op " << op << " not supported." << endl;
      CmiAbort("exiting");
  }
  return mytype;
}

CDECL
int MPI_Reduce(void *inbuf, void *outbuf, int count, int type, MPI_Op op, 
                int root, MPI_Comm comm)
{
  AMPIAPI("MPI_Reduce");
  if(comm != MPI_COMM_WORLD)
  {
    CkAbort("AMPI> Cannot have global operations across communicators.\n");
  }
  ampi *ptr = CtvAccess(ampiPtr);
  CkReduction::reducerType mytype = getReductionType(type,op);
  int size = ptr->myDDT->getType(type)->getSize(count) ;
  CkCallback reduceCB(CkIndex_ampi::reduceResult(0),CkArrayIndex1D(root),
  	mpi_comms[ptr->commidx].aid,true);
  ptr->contribute(size, inbuf, mytype, reduceCB); 
  
  if (ptr->thisIndex == root) /*HACK: Use recv() to block until reduction data comes back*/
    ptr->recv(MPI_REDUCE_TAG, 0, outbuf, count, type, comm);
  return 0;
}

CDECL
int MPI_Allreduce(void *inbuf, void *outbuf, int count, int type,
                   MPI_Op op, MPI_Comm comm)
{
  AMPIAPI("MPI_Allreduce");
  if(comm != MPI_COMM_WORLD)
  {
    CkAbort("AMPI> Cannot have global operations across communicators.\n");
  }
  ampi *ptr = CtvAccess(ampiPtr);
  CkReduction::reducerType mytype = getReductionType(type,op);
  int size = ptr->myDDT->getType(type)->getSize(count) ;
  CkCallback allreduceCB(ampiAllReduceHandler,(void *)&mpi_comms[ptr->commidx]);
  ptr->contribute(size, inbuf, mytype, allreduceCB);
  /*HACK: Use recv() to block until the reduction data comes back*/ 
  ptr->recv(MPI_BCAST_TAG, 0, outbuf, count, type, comm);
  return 0;
}

CDECL
double MPI_Wtime(void)
{
  return TCharmWallTimer();
}

CDECL
int MPI_Start(MPI_Request *reqnum)
{
  AMPIAPI("MPI_Start");
  ampi *ptr = CtvAccess(ampiPtr);
  if(*reqnum >= ptr->nrequests) {
    CkAbort("Invalid persistent Request..\n");
  }
  PersReq *req = &(ptr->requests[*reqnum]);
  if(req->sndrcv == 1) { // send request
    ptr->send(req->tag, ptr->getIndex(), req->buf, req->count, req->type, 
              req->proc, req->comm);
  }
  return 0;
}

CDECL
int MPI_Waitall(int count, MPI_Request *request, MPI_Status *sts)
{
  AMPIAPI("MPI_Waitall");
  ampi *ptr = CtvAccess(ampiPtr);
  int i;
  for(i=0;i<count;i++) {
    if(request[i] == (-1))
      continue;
    if(request[i] < 100) { // persistent request
      PersReq *req = &(ptr->requests[request[i]]);
      if(req->sndrcv == 2) { // recv request
        ptr->recv(req->tag, req->proc, req->buf, req->count, 
                  req->type, req->comm, (int*)(sts+i));
      }
    } else { // irecv request
      int index = request[i] - 100;
      PersReq *req = &(ptr->irequests[index]);
      ptr->recv(req->tag, req->proc, req->buf, req->count, 
                req->type, req->comm, (int*) (sts+i));
      // now free the request
      ptr->nirequests--;
      PersReq *ireq = &(ptr->irequests[0]);
      req->nextfree = ptr->firstfree;
      req->prevfree = ireq[ptr->firstfree].prevfree;
      ireq[req->prevfree].nextfree = index;
      ireq[req->nextfree].prevfree = index;
      ptr->firstfree = index;
    }
  }
  return 0;
}

CDECL
int MPI_Waitany(int count, MPI_Request *request, int *idx, MPI_Status *sts)
{
  AMPIAPI("MPI_Waitany");
  ampi *ptr = CtvAccess(ampiPtr);
  while(1) {
    for(*idx=0;(*idx)<count;(*idx)++) {
      if(request[*idx] == (-1))
        return 0;
      if(request[*idx] < 100) { // persistent request
        PersReq *req = &(ptr->requests[request[*idx]]);
        if(req->sndrcv == 2) { // recv request
          if(ptr->iprobe(req->tag, req->proc, req->comm, (int*) sts)) {
            ptr->recv(req->tag, req->proc, req->buf, req->count, 
                      req->type, req->comm, (int*)sts);
            return 0;
          }
        }
      } else { // irecv request
        int index = request[*idx] - 100;
        PersReq *req = &(ptr->irequests[index]);
        if(ptr->iprobe(req->tag, req->proc, req->comm, (int*) sts)) {
          ptr->recv(req->tag, req->proc, req->buf, req->count, 
                    req->type, req->comm, (int*)sts);
          // now free the request
          ptr->nirequests--;
          PersReq *ireq = &(ptr->irequests[0]);
          req->nextfree = ptr->firstfree;
          req->prevfree = ireq[ptr->firstfree].prevfree;
          ireq[req->prevfree].nextfree = index;
          ireq[req->nextfree].prevfree = index;
          ptr->firstfree = index;
          return 0;
        }
      }
    }
  }
  // should never come here
  return 0;
}

CDECL
int MPI_Wait(MPI_Request *request, MPI_Status *sts)
{
  AMPIAPI("MPI_Wait");
  ampi *ptr = CtvAccess(ampiPtr);
  if(*request == (-1))
      return 0;
  if(*request < 100) { // persistent request
    PersReq *req = &(ptr->requests[*request]);
    if(req->sndrcv == 2) { // recv request
      ptr->recv(req->tag, req->proc, req->buf, req->count, 
                req->type, req->comm, (int*)sts);
    }
  } else { // irecv request
    int index = *request - 100;
    PersReq *req = &(ptr->irequests[index]);
    ptr->recv(req->tag, req->proc, req->buf, req->count, 
              req->type, req->comm, (int*) sts);
    // now free the request
    ptr->nirequests--;
    PersReq *ireq = &(ptr->irequests[0]);
    req->nextfree = ptr->firstfree;
    req->prevfree = ireq[ptr->firstfree].prevfree;
    ireq[req->prevfree].nextfree = index;
    ireq[req->nextfree].prevfree = index;
    ptr->firstfree = index;
  }
  return 0;
}

CDECL
int MPI_Test(MPI_Request *request, int *flag, MPI_Status *sts)
{
  AMPIAPI("MPI_Test");
  ampi *ptr = CtvAccess(ampiPtr);
  if(*request==(-1)) {
    *flag = 1;
    return 0;
  }
  if(*request < 100) { // persistent request
    PersReq *req = &(ptr->requests[*request]);
    if(req->sndrcv == 2) // recv request
      *flag = ptr->iprobe(req->tag, req->proc, req->comm, (int*)sts);
    else
      *flag = 1; // send request
  } else { // irecv request
    int index = *request - 100;
    PersReq *req = &(ptr->irequests[index]);
    *flag = ptr->iprobe(req->tag, req->proc, req->comm, (int*) sts);
  }
  return 0;
}

CDECL
int MPI_Testall(int count, MPI_Request *request, int *flag, MPI_Status *sts)
{
  AMPIAPI("MPI_Testall");
  int i;
  int tmpflag;
  *flag = 1;
  for(i=0;i<count;i++)
  {
    MPI_Test(&request[i], &tmpflag, sts+i);
    *flag = *flag && tmpflag;
  }
  return 0;
}

CDECL
int MPI_Recv_init(void *buf, int count, int type, int src, int tag,
                   MPI_Comm comm, MPI_Request *req)
{
  AMPIAPI("MPI_Recv_init");

  ampi *ptr = CtvAccess(ampiPtr);

  if(ptr->nrequests == 100) {
    CmiAbort("Too many persistent commrequests.\n");
  }
  ptr->requests[ptr->nrequests].sndrcv = 2;
  ptr->requests[ptr->nrequests].buf = buf;
  ptr->requests[ptr->nrequests].count = count;
  ptr->requests[ptr->nrequests].type = type;
  ptr->requests[ptr->nrequests].proc = src;
  ptr->requests[ptr->nrequests].tag = tag;
  ptr->requests[ptr->nrequests].comm = comm;
  *req = ptr->nrequests;
  ptr->nrequests ++;
  return 0;
}

CDECL
int MPI_Send_init(void *buf, int count, int type, int dest, int tag,
                   MPI_Comm comm, MPI_Request *req)
{
  AMPIAPI("MPI_Send_init");
  ampi *ptr = CtvAccess(ampiPtr);
  if(ptr->nrequests == 100) {
    CmiAbort("Too many persistent commrequests.\n");
  }
  ptr->requests[ptr->nrequests].sndrcv = 1;
  ptr->requests[ptr->nrequests].buf = buf;
  ptr->requests[ptr->nrequests].count = count;
  ptr->requests[ptr->nrequests].type = type;
  ptr->requests[ptr->nrequests].proc = dest;
  ptr->requests[ptr->nrequests].tag = tag;
  ptr->requests[ptr->nrequests].comm = comm;
  *req = ptr->nrequests;
  ptr->nrequests ++;
  return 0;
}

CDECL
int MPI_Type_contiguous(int count, MPI_Datatype oldtype, 
                         MPI_Datatype *newtype)
{
  AMPIAPI("MPI_Type_contiguous");
  ampi *ptr = CtvAccess(ampiPtr);
  ptr->myDDT->newContiguous(count, oldtype, newtype); 
  return 0;
}

extern  "C"  
int MPI_Type_vector(int count, int blocklength, int stride, 
                     MPI_Datatype oldtype, MPI_Datatype*  newtype)
{
  AMPIAPI("MPI_Type_vector");
  ampi  *ptr = CtvAccess(ampiPtr);
  ptr->myDDT->newVector(count, blocklength, stride, oldtype, newtype);
  return 0 ;
}

extern  "C"  
int MPI_Type_hvector(int count, int blocklength, int stride, 
                      MPI_Datatype oldtype, MPI_Datatype*  newtype)
{
  AMPIAPI("MPI_Type_hvector");
  ampi  *ptr = CtvAccess(ampiPtr);
  ptr->myDDT->newHVector(count, blocklength, stride, oldtype, newtype);
  return 0 ;
}

extern  "C"  
int MPI_Type_indexed(int count, int* arrBlength, int* arrDisp, 
                      MPI_Datatype oldtype, MPI_Datatype*  newtype)
{
  AMPIAPI("MPI_Type_indexed");
  ampi  *ptr = CtvAccess(ampiPtr);
  ptr->myDDT->newIndexed(count, arrBlength, arrDisp, oldtype, newtype);
  return 0 ;
}

extern  "C"  
int MPI_Type_hindexed(int count, int* arrBlength, int* arrDisp, 
                       MPI_Datatype oldtype, MPI_Datatype*  newtype)
{
  AMPIAPI("MPI_Type_hindexed");
  ampi  *ptr = CtvAccess(ampiPtr);
  ptr->myDDT->newHIndexed(count, arrBlength, arrDisp, oldtype, newtype);
  return 0 ;
}

extern  "C"  
int MPI_Type_struct(int count, int* arrBlength, int* arrDisp, 
                     MPI_Datatype* oldtype, MPI_Datatype*  newtype)
{
  AMPIAPI("MPI_Type_struct");
  ampi  *ptr = CtvAccess(ampiPtr);
  ptr->myDDT->newStruct(count, arrBlength, arrDisp, oldtype, newtype);
  return 0 ;
}

CDECL
int MPI_Type_commit(MPI_Datatype *datatype)
{
  AMPIAPI("MPI_Type_commit");
  return 0;
}

CDECL
int MPI_Type_free(MPI_Datatype *datatype)
{
  AMPIAPI("MPI_Type_free");
  ampi  *ptr = CtvAccess(ampiPtr);
  ptr->myDDT->freeType(datatype);
  return 0;
}


CDECL
int MPI_Type_extent(MPI_Datatype datatype, MPI_Aint extent)
{
  AMPIAPI("MPI_Type_extent");
  ampi *ptr = CtvAccess(ampiPtr) ;
  *extent = ptr->myDDT->getExtent(datatype);
  return 0;
}

CDECL
int MPI_Type_size(MPI_Datatype datatype, MPI_Aint size)
{
  AMPIAPI("MPI_Type_size");
  ampi *ptr = CtvAccess(ampiPtr) ;
  *size = ptr->myDDT->getSize(datatype);
  return 0;
}

CDECL
int MPI_Isend(void *buf, int count, MPI_Datatype type, int dest, 
              int tag, MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("MPI_Isend");
  ampi *ptr = CtvAccess(ampiPtr);
  
  ptr->send(tag, ptr->getIndex(), buf, count, type, dest, comm);
  *request = (-1);
  return 0;
}

CDECL
int MPI_Issend(void *buf, int count, MPI_Datatype type, int dest, 
              int tag, MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("MPI_Issend");
  ampi *ptr = CtvAccess(ampiPtr);
  
  ptr->send(tag, ptr->getIndex(), buf, count, type, dest, comm);
  *request = (-1);
  return 0;
}

CDECL
int MPI_Irecv(void *buf, int count, MPI_Datatype type, int src, 
              int tag, MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("MPI_Irecv");
  ampi *ptr = CtvAccess(ampiPtr);
  if(ptr->nirequests == 100) {
    CmiAbort("Too many Irecv requests.\n");
  }

  PersReq *req = &(ptr->irequests[ptr->firstfree]);
  req->sndrcv = 2;
  req->buf = buf;
  req->count = count;
  req->type = type;
  req->proc = src;
  req->tag = tag;
  req->comm = comm;
  *request = ptr->firstfree + 100;
  ptr->nirequests ++;
  // remove this request from the free list
  PersReq *ireq = &(ptr->irequests[0]);
  ptr->firstfree = ireq[ptr->firstfree].nextfree;
  ireq[req->nextfree].prevfree = req->prevfree;
  ireq[req->prevfree].nextfree = req->nextfree;
  return 0;
}

CDECL
int MPI_Allgatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                   void *recvbuf, int *recvcounts, int *displs, 
                   MPI_Datatype recvtype, MPI_Comm comm) 
{
  AMPIAPI("MPI_Allgatherv");
  if(comm != MPI_COMM_WORLD)
  {
    CkAbort("AMPI> Cannot have global operations across communicators.\n");
  }
  ampi *ptr = CtvAccess(ampiPtr);
  int size = ptr->getArraySize();
  int i;
  for(i=0;i<size;i++) {
    MPI_Send(sendbuf, sendcount, sendtype, i, MPI_GATHER_TAG, comm);
  }

  MPI_Status status;
  DDT_DataType* dttype = ptr->myDDT->getType(recvtype) ;
  int itemsize = dttype->getSize() ;
  
  for(i=0;i<size;i++) {
    MPI_Recv(((char*)recvbuf)+(itemsize*displs[i]), recvcounts[i], recvtype,
             i, MPI_GATHER_TAG, comm, &status);
  }
  return 0;
}

CDECL
int MPI_Allgather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm)
{
  AMPIAPI("MPI_Allgather");
  if(comm != MPI_COMM_WORLD)
  {
    CkAbort("AMPI> Cannot have global operations across communicators.\n");
  }
  ampi *ptr = CtvAccess(ampiPtr);
  int size = ptr->getArraySize();
  int i;
  for(i=0;i<size;i++) {
    MPI_Send(sendbuf, sendcount, sendtype, i, MPI_GATHER_TAG, comm);
  }

  MPI_Status status;
  DDT_DataType* dttype = ptr->myDDT->getType(recvtype) ;
  int itemsize = dttype->getSize(recvcount) ;
  
  for(i=0;i<size;i++) {
    MPI_Recv(((char*)recvbuf)+(itemsize*i), recvcount, recvtype,
             i, MPI_GATHER_TAG, comm, &status);
  }
  return 0;
}

CDECL
int MPI_Gatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int *recvcounts, int *displs,
                MPI_Datatype recvtype, int root, MPI_Comm comm)
{
  AMPIAPI("MPI_Gatherv");
  if(comm != MPI_COMM_WORLD)
  {
    CkAbort("AMPI> Cannot have global operations across communicators.\n");
  }
  ampi *ptr = CtvAccess(ampiPtr);
  int size = ptr->getArraySize();
  int i;

  MPI_Send(sendbuf, sendcount, sendtype, root, MPI_GATHER_TAG, comm);

  if(ptr->getIndex() == root) {
    MPI_Status status;
    DDT_DataType* dttype = ptr->myDDT->getType(recvtype) ;
    int itemsize = dttype->getSize() ;
  
    for(i=0;i<size;i++) {
      MPI_Recv(((char*)recvbuf)+(itemsize*displs[i]), recvcounts[i], recvtype,
               i, MPI_GATHER_TAG, comm, &status);
    }
  }
  return 0;
}

CDECL
int MPI_Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype, 
               int root, MPI_Comm comm)
{
  AMPIAPI("MPI_Gather");
  if(comm != MPI_COMM_WORLD)
  {
    CkAbort("AMPI> Cannot have global operations across communicators.\n");
  }
  ampi *ptr = CtvAccess(ampiPtr);
  int size = ptr->getArraySize();
  int i;
  MPI_Send(sendbuf, sendcount, sendtype, root, MPI_GATHER_TAG, comm);

  if(ptr->getIndex()==root) {
    MPI_Status status;
    DDT_DataType* dttype = ptr->myDDT->getType(recvtype) ;
    int itemsize = dttype->getSize(recvcount) ;
  
    for(i=0;i<size;i++) {
      MPI_Recv(((char*)recvbuf)+(itemsize*i), recvcount, recvtype,
               i, MPI_GATHER_TAG, comm, &status);
    }
  }
  return 0;
}

CDECL
int MPI_Alltoallv(void *sendbuf, int *sendcounts, int *sdispls,
                  MPI_Datatype sendtype, void *recvbuf, int *recvcounts,
                  int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
  AMPIAPI("MPI_Alltoallv");
  if(comm != MPI_COMM_WORLD)
  {
    CkAbort("AMPI> Cannot have global operations across communicators.\n");
  }
  ampi *ptr = CtvAccess(ampiPtr);
  int size = ptr->getArraySize();
  DDT_DataType* dttype = ptr->myDDT->getType(sendtype) ;
  int itemsize = dttype->getSize() ;
  int i;
  for(i=0;i<size;i++) {
    MPI_Send(((char*)sendbuf)+(itemsize*sdispls[i]), sendcounts[i], sendtype,
             i, MPI_GATHER_TAG, comm);
  }

  MPI_Status status;
  dttype = ptr->myDDT->getType(recvtype) ;
  itemsize = dttype->getSize() ;
  
  for(i=0;i<size;i++) {
    MPI_Recv(((char*)recvbuf)+(itemsize*rdispls[i]), recvcounts[i], recvtype,
             i, MPI_GATHER_TAG, comm, &status);
  }
  return 0;
}

CDECL
int MPI_Alltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                 void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                 MPI_Comm comm)
{
  AMPIAPI("MPI_Alltoall");
  if(comm != MPI_COMM_WORLD)
  {
    CkAbort("AMPI> Cannot have global operations across communicators.\n");
  }
  ampi *ptr = CtvAccess(ampiPtr);
  int size = ptr->getArraySize();
  DDT_DataType* dttype = ptr->myDDT->getType(sendtype) ;
  int itemsize = dttype->getSize(sendcount) ;
  int i;
  for(i=0;i<size;i++) {
    MPI_Send(((char*)sendbuf)+(itemsize*i), sendcount, sendtype,
             i, MPI_GATHER_TAG, comm);
  }

  MPI_Status status;
  dttype = ptr->myDDT->getType(recvtype) ;
  itemsize = dttype->getSize(recvcount) ;
  
  for(i=0;i<size;i++) {
    MPI_Recv(((char*)recvbuf)+(itemsize*i), recvcount, recvtype,
             i, MPI_GATHER_TAG, comm, &status);
  }
  return 0;
}

CDECL
int MPI_Comm_dup(int comm, int *newcomm)
{
  AMPIAPI("MPI_Comm_dup");
  *newcomm = comm;
  return 0;
}

CDECL
int MPI_Comm_free(int *comm)
{
  AMPIAPI("MPI_Comm_free");
  return 0;
}

CDECL
int MPI_Abort(int comm, int errorcode)
{
  AMPIAPI("MPI_Abort");
  CkAbort("AMPI: User called MPI_Abort!\n");
  return errorcode;
}

CDECL
int MPI_Get_count(MPI_Status *sts, MPI_Datatype dtype, int *count)
{
  AMPIAPI("MPI_Get_count");
  ampi *ptr = CtvAccess(ampiPtr);
  DDT_DataType* dttype = ptr->myDDT->getType(dtype) ;
  int itemsize = dttype->getSize() ;
  *count = sts->MPI_LENGTH/itemsize;
  return 0;
}

CDECL
int MPI_Pack(void *inbuf, int incount, MPI_Datatype dtype, void *outbuf,
              int outsize, int *position, MPI_Comm comm)
{
  AMPIAPI("MPI_Pack");
  ampi *ptr = CtvAccess(ampiPtr);
  DDT_DataType* dttype = ptr->myDDT->getType(dtype) ;
  int itemsize = dttype->getSize();
  dttype->serialize((char*)inbuf, ((char*)outbuf)+(*position), incount, 1);
  *position += (itemsize*incount);
  return 0;
}

CDECL
int MPI_Unpack(void *inbuf, int insize, int *position, void *outbuf,
              int outcount, MPI_Datatype dtype, MPI_Comm comm)
{
  AMPIAPI("MPI_Unpack");
  ampi *ptr = CtvAccess(ampiPtr);
  DDT_DataType* dttype = ptr->myDDT->getType(dtype) ;
  int itemsize = dttype->getSize();
  dttype->serialize(((char*)inbuf+(*position)), (char*)outbuf, outcount, 1);
  *position += (itemsize*outcount);
  return 0;
}

CDECL
int MPI_Pack_size(int incount,MPI_Datatype datatype,MPI_Comm comm,int *sz)
{
  AMPIAPI("MPI_Pack_size");
  ampi *ptr = CtvAccess(ampiPtr);
  DDT_DataType* dttype = ptr->myDDT->getType(datatype) ;
  return incount*dttype->getSize() ;
}

/* Error handling */
CDECL
int MPI_Error_string(int errorcode, char *string, int *resultlen)
{
  AMPIAPI("MPI_Error_string");
  const char *ret="";
  switch(errorcode) {
  case MPI_SUCCESS:
	   ret="Success";
	   break;
  default:
	   return 1;/*LIE: should be MPI_ERR_something */
  };
  *resultlen=strlen(ret);
  strcpy(string,ret);
  return MPI_SUCCESS;
}


/* Charm++ Extentions to MPI standard: */
CDECL
void MPI_Print(char *str)
{
  AMPIAPI("MPI_Print");
  ampi *ptr = CtvAccess(ampiPtr);
  CkPrintf("[%d] %s\n", ptr->thisIndex, str);
}

CDECL
int MPI_Register(void *d, MPI_PupFn f)
{
	AMPIAPI("MPI_Register");
	return TCharmRegister(d,f);
}

CDECL
void *MPI_Get_userdata(int idx)
{
	AMPIAPI("MPI_Get_userdata");
	return TCharmGetUserdata(idx);
}

CDECL void MPI_Register_main(MPI_MainFn mainFn,const char *name)
{
	AMPIAPI("MPI_Register_main");
	ampiCreateMain(mainFn);
	ampiAttach(name,strlen(name));
}
FDECL void FTN_NAME(MPI_REGISTER_MAIN,mpi_register_main)
	(MPI_MainFn mainFn,const char *name,int nameLen)
{
	AMPIAPI("MPI_register_main");
	ampiCreateMain(mainFn);	
	ampiAttach(name,nameLen);
}

CDECL void MPI_Attach(const char *name)
{
	AMPIAPI("MPI_Attach");
	ampiAttach(name,strlen(name));	
}
FDECL void FTN_NAME(MPI_ATTACH,mpi_attach)(const char *name,int nameLen)
{
	AMPIAPI("MPI_attach");
	ampiAttach(name,nameLen);
}


void _registerampif(void)
{
  _registerampi();
}
#include "ampi.def.h"
