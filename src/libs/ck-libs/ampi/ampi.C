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
	int argc=getArgc();
	for (int i=0;i<argc;i++)
		delete[] argv[i];
	delete[] argv;
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
static ampi_comm_structs ampi_comms;
int ampi_ncomms;
int AMPI_COMM_UNIVERSE[AMPI_MAX_COMM];

extern "C" void AMPI_Setup_Switch(void);

CtvDeclare(ampi*, ampiPtr);
static void ampiNodeInit(void)
{
  CtvInitialize(ampi *, ampiPtr);
  ampi_ncomms=0;
  for(int i=0;i<ampi_ncomms; i++)
  {
    AMPI_COMM_UNIVERSE[i] = i+1;
  }
  TCharmSetFallbackSetup(AMPI_Setup_Switch);
}

static void 
ampiAllReduceHandler(void *arg, int dataSize, void *data)
{
  ampi_comm_struct *commspec = (ampi_comm_struct *) arg;
  int type = commspec->rspec.type;
  if (type==-1)
	  CkAbort("ERROR! Never set the AMPI reduction type-- is there an element on processor 0?\n");

  if(type==0) 
  { // allreduce
    ampi::bcastraw(data, dataSize, commspec->aid);
  } else 
  { // reduce
    ampi::sendraw(0, AMPI_REDUCE_TAG, data, dataSize, commspec->aid, 
                  commspec->rspec.root);
  }
  commspec->rspec.type=-1;
}

class AMPI_threadstart_t {
public:
	argvPupable args;
	AMPI_MainFn fn;
	AMPI_threadstart_t() {}
	AMPI_threadstart_t(const argvPupable &args_,AMPI_MainFn fn_)
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
PUPmarshall(AMPI_threadstart_t);

extern "C" void AMPI_threadstart(void *data)
{
	AMPI_threadstart_t t;
	pupFromBuf(data,t);
	t.start();
}

static void ampiAttach(void);

void ampiCreateMain(AMPI_MainFn mainFn)
{
	int _nchunks=TCharmGetNumChunks();
	
	//Make a new threads array
	argvPupable args(TCharmArgv());
	AMPI_threadstart_t s(args,mainFn);
	memBuf b; pupIntoBuf(b,s);
	TCharmCreateData( _nchunks,AMPI_threadstart,
			  b.getData(), b.getSize());
}

static void ampiAttach(const char *name,int namelen)
{
        TCharmSetupCookie *tc=TCharmSetupCookie::get();
	if (!tc->hasThreads())
		CkAbort("You must create a thread array with TCharmCreate before calling AMPI_Attach!\n");
	int _nchunks=tc->getNumElements();
	CkArrayID threads=tc->getThreads();

	//Allocate the next communicator  
	if(ampi_ncomms == AMPI_MAX_COMM)
	{
		CkAbort("AMPI> Number of registered comm_worlds exceeded limit.\n");
	}
	int commidx=ampi_ncomms++;	
	ampi_comms[commidx].mainfunc = NULL; //mainFn;
	ampi_comms[commidx].name = new char[namelen+1];
	memcpy(ampi_comms[commidx].name, name, namelen);
	ampi_comms[commidx].name[namelen] = '\0';
	ampi_comms[commidx].nobj=_nchunks;
	ampi_comms[commidx].rspec.type=-1;
	ampi_comms[commidx].rspec.root=-1;
        
        //Create and attach the new ampi array
        CkArrayOptions opts(_nchunks);
        opts.bindTo(threads);
        CProxy_ampi arr= CProxy_ampi::ckNew(commidx,threads,opts);
	ampi_comms[commidx].aid=arr;
        arr.setReductionClient(ampiAllReduceHandler, &ampi_comms[commidx]);

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
	CmmFree(msgs);
}

//------------------------ communication -----------------------
void
ampi::generic(AmpiMsg* msg)
{
  int tags[3];
  tags[0] = msg->tag1; tags[1] = msg->tag2; tags[2] = msg->comm;
  CmmPut(msgs, 3, tags, msg);
  if(ampiBlockedThread)
	  thread->resume();
}

void 
ampi::send(int t1, int t2, void* buf, int count, int type,  int idx, int comm)
{
  CkArrayID aid = thisArrayID;
  int mycomm = AMPI_COMM_WORLD;
  if(comm != AMPI_COMM_WORLD)
  {
    mycomm = AMPI_COMM_UNIVERSE[commidx];
    aid = ampi_comms[comm-1].aid;
  }
  DDT_DataType *ddt = myDDT->getType(type);
  int len = ddt->getSize(count);
  AmpiMsg *msg = new (&len, 0) AmpiMsg(t1, t2, len, mycomm);
  ddt->serialize((char*)buf, (char*)msg->data, count, 1);
  CProxy_ampi pa(aid);
  pa[idx].generic(msg);
}

void 
ampi::sendraw(int t1, int t2, void* buf, int len, CkArrayID aid, int idx)
{
  AmpiMsg *msg = new (&len, 0) AmpiMsg(t1, t2, len, 0);
  memcpy(msg->data, buf, len);
  CProxy_ampi pa(aid);
  pa[idx].generic(msg);
}

void 
ampi::recv(int t1, int t2, void* buf, int count, int type, int comm, int *sts)
{
  int tags[3];
  AmpiMsg *msg = 0;
  DDT_DataType *ddt = myDDT->getType(type);
  int len = ddt->getSize(count);
  ampiBlockedThread=1;
  while(1) {
    tags[0] = t1; tags[1] = t2; tags[2] = comm;
    msg = (AmpiMsg *) CmmGet(msgs, 3, tags, sts);
    if (msg) break;
    thread->suspend();
  }
  ampiBlockedThread=0;
  if(sts)
    ((AMPI_Status*)sts)->AMPI_LENGTH = msg->length;
  if (msg->length < len) {
    CkError("AMPI: (type=%d, count=%d) Expecting msg of len %d, received %d\n",
            type, count, len, msg->length);
    CkAbort("Exiting.\n");
  }
  ddt->serialize((char*)buf, (char*)msg->data, count, (-1));
  delete msg;
}

void 
ampi::probe(int t1, int t2, int comm, int *sts)
{
  int tags[3];
  AmpiMsg *msg = 0;
  while(1) {
    tags[0] = t1; tags[1] = t2; tags[2] = comm;
    msg = (AmpiMsg *) CmmProbe(msgs, 3, tags, sts);
    if (msg) break;
    thread->schedule();
  }
  if(sts)
    ((AMPI_Status*)sts)->AMPI_LENGTH = msg->length;
}

int 
ampi::iprobe(int t1, int t2, int comm, int *sts)
{
  int tags[3];
  AmpiMsg *msg = 0;
  tags[0] = t1; tags[1] = t2; tags[2] = comm;
  msg = (AmpiMsg *) CmmProbe(msgs, 3, tags, sts);
  if (msg) {
    if(sts)
      ((AMPI_Status*)sts)->AMPI_LENGTH = msg->length;
    return 1;
  }
  thread->schedule();
  return 0;
}

void 
ampi::barrier(void)
{
  if(thisIndex) {
    send(AMPI_BARR_TAG, 0, 0, 0, 0, 0, AMPI_COMM_WORLD);
    recv(AMPI_BARR_TAG, 0, 0, 0, 0, AMPI_COMM_WORLD);
  } else {
    int i;
    for(i=1;i<numElements;i++) 
      recv(AMPI_BARR_TAG, 0, 0, 0, 0, AMPI_COMM_WORLD);
    for(i=1;i<numElements;i++) 
      send(AMPI_BARR_TAG, 0, 0, 0, 0, i, AMPI_COMM_WORLD);
  }
}

void 
ampi::bcast(int root, void* buf, int count, int type)
{
  if(root==thisIndex) {
    int i;
    for(i=0;i<numElements;i++)
      send(AMPI_BCAST_TAG, nbcasts, buf, count, type, i, AMPI_COMM_WORLD);
  }
  recv(AMPI_BCAST_TAG, nbcasts, buf, count, type, AMPI_COMM_WORLD);
  nbcasts++;
}

void
ampi::bcastraw(void* buf, int len, CkArrayID aid)
{
  AmpiMsg *msg = new (&len, 0) AmpiMsg(0, AMPI_BCAST_TAG, len, 0);
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
extern "C" void
AMPI_Checkpoint(char *dirname)
{
  mkdir(dirname, 0777);
  ampi *ptr = CtvAccess(ampiPtr);
  ptr->cthread_id = CthSelf();
  int idx = ptr->thisIndex;
  CProxy_ampi aproxy(ampimain::ampi_comms[ptr->commidx].aid);
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
  }
  myDDT->pup(p);
}

//------------------ External Interface -----------------

static ampi *getAmpiInstance(void) {
  ampi *ret = CtvAccess(ampiPtr);
  return ret;
}

extern "C" void
AMPI_Migrate(void)
{
  TCharmMigrate();
}

extern "C" 
int AMPI_Init(int *argc, char*** argv)
{
  return 0;
}

extern "C" 
int AMPI_Comm_rank(AMPI_Comm comm, int *rank)
{
  *rank = TCharmElement();
  return 0;
}

extern "C" 
int AMPI_Comm_size(AMPI_Comm comm, int *size)
{
  *size = TCharmNumElements();
  return 0;
}

extern "C" 
int AMPI_Finalize(void)
{
  return 0;
}

extern "C" 
int AMPI_Send(void *msg, int count, AMPI_Datatype type, int dest, 
                        int tag, AMPI_Comm comm)
{
  ampi *ptr = CtvAccess(ampiPtr);
  ptr->send(tag, ptr->getIndex(), msg, count, type, dest, comm);
  return 0;
}

extern "C" 
int AMPI_Ssend(void *msg, int count, AMPI_Datatype type, int dest, 
                        int tag, AMPI_Comm comm)
{
  ampi *ptr = CtvAccess(ampiPtr);
  ptr->send(tag, ptr->getIndex(), msg, count, type, dest, comm);
  return 0;
}

extern "C" 
int AMPI_Recv(void *msg, int count, AMPI_Datatype type, int src, int tag, 
              AMPI_Comm comm, AMPI_Status *status)
{
  ampi *ptr = CtvAccess(ampiPtr);
  ptr->recv(tag,src,msg,count,type, comm, (int*) status);
  return 0;
}

extern "C" 
int AMPI_Probe(int src, int tag, AMPI_Comm comm, AMPI_Status *status)
{
  ampi *ptr = CtvAccess(ampiPtr);
  ptr->probe(tag,src, comm, (int*) status);
  return 0;
}

extern "C" 
int AMPI_Iprobe(int src,int tag,AMPI_Comm comm,int *flag,AMPI_Status *status)
{
  ampi *ptr = CtvAccess(ampiPtr);
  *flag = ptr->iprobe(tag,src,comm,(int*) status);
  return 0;
}

extern "C" 
int AMPI_Sendrecv(void *sbuf, int scount, int stype, int dest, 
                  int stag, void *rbuf, int rcount, int rtype,
                  int src, int rtag, AMPI_Comm comm, AMPI_Status *sts)
{
  return (AMPI_Send(sbuf,scount,stype,dest,stag,comm) ||
          AMPI_Recv(rbuf,rcount,rtype,src,rtag,comm,sts));
}

extern "C" 
int AMPI_Barrier(AMPI_Comm comm)
{
  if(comm != AMPI_COMM_WORLD)
  {
    CkAbort("AMPI> Cannot have global operations across communicators.\n");
  }
  ampi *ptr = CtvAccess(ampiPtr);
  ptr->barrier();
  return 0;
}

extern "C" 
int AMPI_Bcast(void *buf, int count, AMPI_Datatype type, int root, 
                         AMPI_Comm comm)
{
  if(comm != AMPI_COMM_WORLD)
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
    case AMPI_MAX :
      switch(type) {
        case AMPI_FLOAT : mytype = CkReduction::max_float; break;
        case AMPI_INT : mytype = CkReduction::max_int; break;
        case AMPI_DOUBLE : mytype = CkReduction::max_double; break;
        default:
          ckerr << "Type " << type << " not supported." << endl;
          CmiAbort("exiting");
      }
      break;
    case AMPI_MIN :
      switch(type) {
        case AMPI_FLOAT : mytype = CkReduction::min_float; break;
        case AMPI_INT : mytype = CkReduction::min_int; break;
        case AMPI_DOUBLE : mytype = CkReduction::min_double; break;
        default:
          ckerr << "Type " << type << " not supported." << endl;
          CmiAbort("exiting");
      }
      break;
    case AMPI_SUM :
      switch(type) {
        case AMPI_FLOAT : mytype = CkReduction::sum_float; break;
        case AMPI_INT : mytype = CkReduction::sum_int; break;
        case AMPI_DOUBLE : mytype = CkReduction::sum_double; break;
        default:
          ckerr << "Type " << type << " not supported." << endl;
          CmiAbort("exiting");
      }
      break;
    case AMPI_PROD :
      switch(type) {
        case AMPI_FLOAT : mytype = CkReduction::product_float; break;
        case AMPI_INT : mytype = CkReduction::product_int; break;
        case AMPI_DOUBLE : mytype = CkReduction::product_double; break;
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

extern "C" 
int AMPI_Reduce(void *inbuf, void *outbuf, int count, int type, AMPI_Op op, 
                int root, AMPI_Comm comm)
{
  if(comm != AMPI_COMM_WORLD)
  {
    CkAbort("AMPI> Cannot have global operations across communicators.\n");
  }
  ampi *ptr = CtvAccess(ampiPtr);
  if(CkMyPe()==0)
  {
    ampi_comms[ptr->commidx].rspec.type = 1;
    ampi_comms[ptr->commidx].rspec.root = root;
  }
  CkReduction::reducerType mytype = getReductionType(type,op);
  int size = ptr->myDDT->getType(type)->getSize(count) ;
  ptr->contribute(size, inbuf, mytype);
  if (ptr->thisIndex == root)
    ptr->recv(0, AMPI_REDUCE_TAG, outbuf, count, type, comm);
  return 0;
}

extern "C" 
int AMPI_Allreduce(void *inbuf, void *outbuf, int count, int type,
                   AMPI_Op op, AMPI_Comm comm)
{
  if(comm != AMPI_COMM_WORLD)
  {
    CkAbort("AMPI> Cannot have global operations across communicators.\n");
  }
  ampi *ptr = CtvAccess(ampiPtr);
  if(CkMyPe()==0)
  {
    ampi_comms[ptr->commidx].rspec.type = 0;
  }
  CkReduction::reducerType mytype = getReductionType(type,op);
  int size = ptr->myDDT->getType(type)->getSize(count) ;
  ptr->contribute(size, inbuf, mytype);
  ptr->recv(0, AMPI_BCAST_TAG, outbuf, count, type, comm);
  return 0;
}

extern "C" 
double AMPI_Wtime(void)
{
  return CmiWallTimer();
}

extern "C" 
int AMPI_Start(AMPI_Request *reqnum)
{
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

extern "C" 
int AMPI_Waitall(int count, AMPI_Request *request, AMPI_Status *sts)
{
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

extern "C" 
int AMPI_Waitany(int count, AMPI_Request *request, int *idx, AMPI_Status *sts)
{
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

extern "C" 
int AMPI_Wait(AMPI_Request *request, AMPI_Status *sts)
{
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

extern "C" 
int AMPI_Test(AMPI_Request *request, int *flag, AMPI_Status *sts)
{
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

extern "C" 
int AMPI_Testall(int count, AMPI_Request *request, int *flag, AMPI_Status *sts)
{
  int i;
  int tmpflag;
  *flag = 1;
  for(i=0;i<count;i++)
  {
    AMPI_Test(&request[i], &tmpflag, sts+i);
    *flag = *flag && tmpflag;
  }
  return 0;
}

extern "C" 
int AMPI_Recv_init(void *buf, int count, int type, int src, int tag,
                   AMPI_Comm comm, AMPI_Request *req)
{

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

extern "C" 
int AMPI_Send_init(void *buf, int count, int type, int dest, int tag,
                   AMPI_Comm comm, AMPI_Request *req)
{
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

extern "C" 
int AMPI_Type_contiguous(int count, AMPI_Datatype oldtype, 
                         AMPI_Datatype *newtype)
{
  ampi *ptr = CtvAccess(ampiPtr);
  ptr->myDDT->newContiguous(count, oldtype, newtype); 
  return 0;
}

extern  "C"  
int AMPI_Type_vector(int count, int blocklength, int stride, 
                     AMPI_Datatype oldtype, AMPI_Datatype*  newtype)
{
  ampi  *ptr = CtvAccess(ampiPtr);
  ptr->myDDT->newVector(count, blocklength, stride, oldtype, newtype);
  return 0 ;
}

extern  "C"  
int AMPI_Type_hvector(int count, int blocklength, int stride, 
                      AMPI_Datatype oldtype, AMPI_Datatype*  newtype)
{
  ampi  *ptr = CtvAccess(ampiPtr);
  ptr->myDDT->newHVector(count, blocklength, stride, oldtype, newtype);
  return 0 ;
}

extern  "C"  
int AMPI_Type_indexed(int count, int* arrBlength, int* arrDisp, 
                      AMPI_Datatype oldtype, AMPI_Datatype*  newtype)
{
  ampi  *ptr = CtvAccess(ampiPtr);
  ptr->myDDT->newIndexed(count, arrBlength, arrDisp, oldtype, newtype);
  return 0 ;
}

extern  "C"  
int AMPI_Type_hindexed(int count, int* arrBlength, int* arrDisp, 
                       AMPI_Datatype oldtype, AMPI_Datatype*  newtype)
{
  ampi  *ptr = CtvAccess(ampiPtr);
  ptr->myDDT->newHIndexed(count, arrBlength, arrDisp, oldtype, newtype);
  return 0 ;
}

extern  "C"  
int AMPI_Type_struct(int count, int* arrBlength, int* arrDisp, 
                     AMPI_Datatype* oldtype, AMPI_Datatype*  newtype)
{
  ampi  *ptr = CtvAccess(ampiPtr);
  ptr->myDDT->newStruct(count, arrBlength, arrDisp, oldtype, newtype);
  return 0 ;
}

extern "C" 
int AMPI_Type_commit(AMPI_Datatype *datatype)
{
  return 0;
}

extern "C" 
int AMPI_Type_free(AMPI_Datatype *datatype)
{
  ampi  *ptr = CtvAccess(ampiPtr);
  ptr->myDDT->freeType(datatype);
  return 0;
}


extern "C" 
int AMPI_Type_extent(AMPI_Datatype datatype, AMPI_Aint extent)
{
  ampi *ptr = CtvAccess(ampiPtr) ;
  *extent = ptr->myDDT->getExtent(datatype);
  return 0;
}

extern "C" 
int AMPI_Type_size(AMPI_Datatype datatype, AMPI_Aint size)
{
  ampi *ptr = CtvAccess(ampiPtr) ;
  *size = ptr->myDDT->getSize(datatype);
  return 0;
}

extern "C" 
int AMPI_Isend(void *buf, int count, AMPI_Datatype type, int dest, 
              int tag, AMPI_Comm comm, AMPI_Request *request)
{
  ampi *ptr = CtvAccess(ampiPtr);
  
  ptr->send(tag, ptr->getIndex(), buf, count, type, dest, comm);
  *request = (-1);
  return 0;
}

extern "C" 
int AMPI_Issend(void *buf, int count, AMPI_Datatype type, int dest, 
              int tag, AMPI_Comm comm, AMPI_Request *request)
{
  ampi *ptr = CtvAccess(ampiPtr);
  
  ptr->send(tag, ptr->getIndex(), buf, count, type, dest, comm);
  *request = (-1);
  return 0;
}

extern "C" 
int AMPI_Irecv(void *buf, int count, AMPI_Datatype type, int src, 
              int tag, AMPI_Comm comm, AMPI_Request *request)
{
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

extern "C" 
int AMPI_Allgatherv(void *sendbuf, int sendcount, AMPI_Datatype sendtype, 
                   void *recvbuf, int *recvcounts, int *displs, 
                   AMPI_Datatype recvtype, AMPI_Comm comm) 
{
  if(comm != AMPI_COMM_WORLD)
  {
    CkAbort("AMPI> Cannot have global operations across communicators.\n");
  }
  ampi *ptr = CtvAccess(ampiPtr);
  int size = ptr->getArraySize();
  int i;
  for(i=0;i<size;i++) {
    AMPI_Send(sendbuf, sendcount, sendtype, i, AMPI_GATHER_TAG, comm);
  }

  AMPI_Status status;
  DDT_DataType* dttype = ptr->myDDT->getType(recvtype) ;
  int itemsize = dttype->getSize() ;
  
  for(i=0;i<size;i++) {
    AMPI_Recv(((char*)recvbuf)+(itemsize*displs[i]), recvcounts[i], recvtype,
             i, AMPI_GATHER_TAG, comm, &status);
  }
  return 0;
}

extern "C"
int AMPI_Allgather(void *sendbuf, int sendcount, AMPI_Datatype sendtype,
                  void *recvbuf, int recvcount, AMPI_Datatype recvtype,
                  AMPI_Comm comm)
{
  if(comm != AMPI_COMM_WORLD)
  {
    CkAbort("AMPI> Cannot have global operations across communicators.\n");
  }
  ampi *ptr = CtvAccess(ampiPtr);
  int size = ptr->getArraySize();
  int i;
  for(i=0;i<size;i++) {
    AMPI_Send(sendbuf, sendcount, sendtype, i, AMPI_GATHER_TAG, comm);
  }

  AMPI_Status status;
  DDT_DataType* dttype = ptr->myDDT->getType(recvtype) ;
  int itemsize = dttype->getSize(recvcount) ;
  
  for(i=0;i<size;i++) {
    AMPI_Recv(((char*)recvbuf)+(itemsize*i), recvcount, recvtype,
             i, AMPI_GATHER_TAG, comm, &status);
  }
  return 0;
}

extern "C"
int AMPI_Gatherv(void *sendbuf, int sendcount, AMPI_Datatype sendtype,
                void *recvbuf, int *recvcounts, int *displs,
                AMPI_Datatype recvtype, int root, AMPI_Comm comm)
{
  if(comm != AMPI_COMM_WORLD)
  {
    CkAbort("AMPI> Cannot have global operations across communicators.\n");
  }
  ampi *ptr = CtvAccess(ampiPtr);
  int size = ptr->getArraySize();
  int i;

  AMPI_Send(sendbuf, sendcount, sendtype, root, AMPI_GATHER_TAG, comm);

  if(ptr->getIndex() == root) {
    AMPI_Status status;
    DDT_DataType* dttype = ptr->myDDT->getType(recvtype) ;
    int itemsize = dttype->getSize() ;
  
    for(i=0;i<size;i++) {
      AMPI_Recv(((char*)recvbuf)+(itemsize*displs[i]), recvcounts[i], recvtype,
               i, AMPI_GATHER_TAG, comm, &status);
    }
  }
  return 0;
}

extern "C"
int AMPI_Gather(void *sendbuf, int sendcount, AMPI_Datatype sendtype,
               void *recvbuf, int recvcount, AMPI_Datatype recvtype, 
               int root, AMPI_Comm comm)
{
  if(comm != AMPI_COMM_WORLD)
  {
    CkAbort("AMPI> Cannot have global operations across communicators.\n");
  }
  ampi *ptr = CtvAccess(ampiPtr);
  int size = ptr->getArraySize();
  int i;
  AMPI_Send(sendbuf, sendcount, sendtype, root, AMPI_GATHER_TAG, comm);

  if(ptr->getIndex()==root) {
    AMPI_Status status;
    DDT_DataType* dttype = ptr->myDDT->getType(recvtype) ;
    int itemsize = dttype->getSize(recvcount) ;
  
    for(i=0;i<size;i++) {
      AMPI_Recv(((char*)recvbuf)+(itemsize*i), recvcount, recvtype,
               i, AMPI_GATHER_TAG, comm, &status);
    }
  }
  return 0;
}

extern "C" 
int AMPI_Alltoallv(void *sendbuf, int *sendcounts, int *sdispls,
                  AMPI_Datatype sendtype, void *recvbuf, int *recvcounts,
                  int *rdispls, AMPI_Datatype recvtype, AMPI_Comm comm)
{
  if(comm != AMPI_COMM_WORLD)
  {
    CkAbort("AMPI> Cannot have global operations across communicators.\n");
  }
  ampi *ptr = CtvAccess(ampiPtr);
  int size = ptr->getArraySize();
  DDT_DataType* dttype = ptr->myDDT->getType(sendtype) ;
  int itemsize = dttype->getSize() ;
  int i;
  for(i=0;i<size;i++) {
    AMPI_Send(((char*)sendbuf)+(itemsize*sdispls[i]), sendcounts[i], sendtype,
             i, AMPI_GATHER_TAG, comm);
  }

  AMPI_Status status;
  dttype = ptr->myDDT->getType(recvtype) ;
  itemsize = dttype->getSize() ;
  
  for(i=0;i<size;i++) {
    AMPI_Recv(((char*)recvbuf)+(itemsize*rdispls[i]), recvcounts[i], recvtype,
             i, AMPI_GATHER_TAG, comm, &status);
  }
  return 0;
}

extern "C" 
int AMPI_Alltoall(void *sendbuf, int sendcount, AMPI_Datatype sendtype, 
                 void *recvbuf, int recvcount, AMPI_Datatype recvtype, 
                 AMPI_Comm comm)
{
  if(comm != AMPI_COMM_WORLD)
  {
    CkAbort("AMPI> Cannot have global operations across communicators.\n");
  }
  ampi *ptr = CtvAccess(ampiPtr);
  int size = ptr->getArraySize();
  DDT_DataType* dttype = ptr->myDDT->getType(sendtype) ;
  int itemsize = dttype->getSize(sendcount) ;
  int i;
  for(i=0;i<size;i++) {
    AMPI_Send(((char*)sendbuf)+(itemsize*i), sendcount, sendtype,
             i, AMPI_GATHER_TAG, comm);
  }

  AMPI_Status status;
  dttype = ptr->myDDT->getType(recvtype) ;
  itemsize = dttype->getSize(recvcount) ;
  
  for(i=0;i<size;i++) {
    AMPI_Recv(((char*)recvbuf)+(itemsize*i), recvcount, recvtype,
             i, AMPI_GATHER_TAG, comm, &status);
  }
  return 0;
}

extern "C"
int AMPI_Comm_dup(int comm, int *newcomm)
{
  *newcomm = comm;
  return 0;
}

extern "C"
int AMPI_Comm_free(int *comm)
{
  return 0;
}

extern "C"
int AMPI_Abort(int comm, int errorcode)
{
  CkAbort("AMPI: User called MPI_Abort!\n");
  return errorcode;
}

extern "C"
int AMPI_Get_count(AMPI_Status *sts, AMPI_Datatype dtype, int *count)
{
  ampi *ptr = CtvAccess(ampiPtr);
  DDT_DataType* dttype = ptr->myDDT->getType(dtype) ;
  int itemsize = dttype->getSize() ;
  *count = sts->AMPI_LENGTH/itemsize;
  return 0;
}

extern "C"
int AMPI_Pack(void *inbuf, int incount, AMPI_Datatype dtype, void *outbuf,
              int outsize, int *position, AMPI_Comm comm)
{
  ampi *ptr = CtvAccess(ampiPtr);
  DDT_DataType* dttype = ptr->myDDT->getType(dtype) ;
  int itemsize = dttype->getSize();
  dttype->serialize((char*)inbuf, ((char*)outbuf)+(*position), incount, 1);
  *position += (itemsize*incount);
  return 0;
}

extern "C"
int AMPI_Unpack(void *inbuf, int insize, int *position, void *outbuf,
              int outcount, AMPI_Datatype dtype, AMPI_Comm comm)
{
  ampi *ptr = CtvAccess(ampiPtr);
  DDT_DataType* dttype = ptr->myDDT->getType(dtype) ;
  int itemsize = dttype->getSize();
  dttype->serialize(((char*)inbuf+(*position)), (char*)outbuf, outcount, 1);
  *position += (itemsize*outcount);
  return 0;
}

extern "C"
int AMPI_Pack_size(int incount,AMPI_Datatype datatype,AMPI_Comm comm,int *sz)
{
  ampi *ptr = CtvAccess(ampiPtr);
  DDT_DataType* dttype = ptr->myDDT->getType(datatype) ;
  return incount*dttype->getSize() ;
}

extern "C"
void AMPI_Print(char *str)
{
  ampi *ptr = CtvAccess(ampiPtr);
  CkPrintf("[%d] %s\n", ptr->thisIndex, str);
}

extern "C"
int AMPI_Register(void *d, AMPI_PupFn f)
{
	return TCharmRegister(d,f);
}

extern "C"
void *AMPI_Get_userdata(int idx)
{
	return TCharmGetUserdata(idx);
}

CDECL void AMPI_Register_main(AMPI_MainFn mainFn,const char *name)
{
	ampiCreateMain(mainFn);
	ampiAttach(name,strlen(name));
}
FDECL void FTN_NAME(AMPI_REGISTER_MAIN,ampi_register_main)
	(AMPI_MainFn mainFn,const char *name,int nameLen)
{
	ampiCreateMain(mainFn);	
	ampiAttach(name,nameLen);
}

CDECL void AMPI_Attach(const char *name)
{
	ampiAttach(name,strlen(name));	
}
FDECL void FTN_NAME(AMPI_ATTACH,ampi_attach)(const char *name,int nameLen)
{
	ampiAttach(name,nameLen);
}


#include "ampi.def.h"
