/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "ampiimpl.h"
// for strlen
#include <string.h>

int AMPI_COMM_UNIVERSE[AMPI_MAX_COMM];

// Default ampi_setup
#if AMPI_FORTRAN
#include "ampimain.decl.h"
#if CMK_FORTRAN_USES_ALLCAPS
extern "C" void AMPI_MAIN(int, char **);
extern "C" void AMPI_SETUP(void){AMPI_REGISTER_MAIN(AMPI_MAIN);}
#else
extern "C" void ampi_main_(int, char **);
extern "C" void ampi_setup_(void){ampi_register_main_(ampi_main_);}
#endif
#else
extern "C" void AMPI_Main(int, char **);
extern "C" void AMPI_Setup(void){AMPI_Register_main(AMPI_Main);}
#endif

void*
ArgsInfo::pack(ArgsInfo* msg)
{
  int argsize=0, i;
  for(i=0;msg->argv[i]!=0;i++) {
    argsize += (strlen(msg->argv[i])+1); // +1 for '\0'
  }
  msg->argc = i;
  void *p = CkAllocBuffer(msg, sizeof(ArgsInfo) +
                               (msg->argc*sizeof(char*)) + 
                                argsize);
  memcpy(p,msg,sizeof(ArgsInfo));
  char *args = (char *)((char*)p+sizeof(ArgsInfo)+(msg->argc*sizeof(char*)));
  for(i=0;i<msg->argc;i++) {
    char *tmp = msg->argv[i];
    while(*tmp) { *args++ = *tmp++; }
    *args++ = '\0';
  }
  delete msg;
  return p;
}

ArgsInfo*
ArgsInfo::unpack(void *in)
{
  ArgsInfo* msg = new (in) ArgsInfo();
  char **argv = (char**)((char*)in+sizeof(ArgsInfo));
  msg->setargs(msg->argc, argv);
  char *tmp = ((char*)in+sizeof(ArgsInfo)+(msg->argc*sizeof(char*)));
  for(int i=0;i<msg->argc;i++) {
    argv[i] = tmp;
    while(*tmp) { tmp++; }
    tmp++;
  }
  return msg;
}

CtvDeclare(ampi*, ampiPtr);


ampi::ampi(AmpiStartMsg *msg)
{
  usesAtSync = CmiTrue;
  commidx = msg->commidx;
  delete msg;
  msgs = CmmNew();
  thread_id = 0;
  nbcasts = 0;
  nrequests = 0;
  myDDT = new DDT() ;
  nirequests = 0;
  firstfree = 0;
  int i;
  for(i=0;i<100;i++) {
    irequests[i].nextfree = (i+1)%100;
    irequests[i].prevfree = ((i-1)+100)%100;
  }
  for(i=0;i<ampimain::ncomms; i++)
  {
    AMPI_COMM_UNIVERSE[i] = i+1;
  }
}

ampi::~ampi()
{
  if (thread_id!=0)
    CthFree(thread_id);
  CmmFree(msgs);
}

void
ampi::generic(AmpiMsg* msg)
{
  int tags[3];
  tags[0] = msg->tag1; tags[1] = msg->tag2; tags[2] = msg->comm;
  CmmPut(msgs, 3, tags, msg);
  if(thread_id) {
    CthAwaken(thread_id);
    thread_id = 0;
  }
}

void 
ampi::send(int t1, int t2, void* buf, int count, int type,  int idx, int comm)
{
  CkArrayID aid = thisArrayID;
  int mycomm = AMPI_COMM_WORLD;
  if(comm != AMPI_COMM_WORLD)
  {
    mycomm = AMPI_COMM_UNIVERSE[commidx];
    aid = ampimain::ampi_comms[comm-1].aid;
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
ampi::recv(int t1, int t2, void* buf, int count, int type, int comm)
{
  int tags[3];
  AmpiMsg *msg = 0;
  DDT_DataType *ddt = myDDT->getType(type);
  int len = ddt->getSize(count);
  while(1) {
    tags[0] = t1; tags[1] = t2; tags[2] = comm;
    msg = (AmpiMsg *) CmmGet(msgs, 3, tags, 0);
    if (msg) break;
    thread_id = CthSelf();
    stop_running();
    CthSuspend();
    start_running();
  }
  if (msg->length < len) {
    CkError("AMPI: (type=%d, count=%d) Expecting msg of len %d, received %d\n",
            type, count, len, msg->length);
    CkAbort("Exiting.\n");
  }
  ddt->serialize((char*)buf, (char*)msg->data, count, (-1));
  delete msg;
}

void 
ampi::barrier(void)
{
  if(thisIndex) {
    send(AMPI_BARR_TAG, 0, 0, 0, 0, 0, AMPI_COMM_WORLD);
    recv(AMPI_BARR_TAG, 0, 0, 0, 0, AMPI_COMM_WORLD);
  } else {
    int i;
    for(i=1;i<numElements;i++) recv(AMPI_BARR_TAG, 0, 0, 0, 0, AMPI_COMM_WORLD);
    for(i=1;i<numElements;i++) send(AMPI_BARR_TAG, 0, 0, 0, 0, i, AMPI_COMM_WORLD);
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

void ampi::pup(PUP::er &p)
{
  ArrayElement1D::pup(p);//Pack superclass
  p(commidx);
  if(p.isDeleting())
  {//Resend saved messages to myself
    AmpiMsg *msg;
    int snum[3];
    snum[0] = CmmWildCard;
    snum[1] = CmmWildCard;
    snum[2] = CmmWildCard;
    CProxy_ampi ap(thisArrayID);
    while(msg = (AmpiMsg*)CmmGet(msgs,3,snum,0))
      ap[thisIndex].generic(msg);
  }
  //This seekBlock allows us to reorder the packing/unpacking--
  // This is needed because the userData depends on the thread's stack
  // both at pack and unpack time.
  PUP::seekBlock s(p,2);
  if (p.isUnpacking()) 
  {//In this case, unpack the thread before the user data
  	s.seek(1);
  	thread_id = CthPup((pup_er) &p, thread_id);
  }
  //Pack all user data
  s.seek(0);
  p(nudata);
  int i;
  for(i=0;i<nudata;i++) {
    p((void*)&(userdata[i]), sizeof(void*));
    p((void*)&(pup_ud[i]), sizeof(AMPI_PupFn));
#if AMPI_FORTRAN
    pup_ud[i]((pup_er) &p, userdata[i]);
#else
    userdata[i] = pup_ud[i]((pup_er) &p, userdata[i]);
#endif
  }
  if (p.isPacking() || p.isSizing()) 
  {//In this case, pack the thread after the user data
  	s.seek(1);
  	thread_id = CthPup((pup_er) &p, thread_id);
  }
  s.endBlock(); //End of seeking block

  p(nbcasts);
  // persistent comm requests will have to be re-registered after
  // migration anyway, so no need to pup them
  // migrate is called only when all irequests are complete, so no need
  // to pup them as well.
  if(p.isUnpacking())
  {
    msgs = CmmNew();
    CtvAccessOther(thread_id, ampiPtr) = this;
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

int
ampi::register_userdata(void *d, AMPI_PupFn f)
{
  if(nudata==AMPI_MAXUDATA)
    CkAbort("AMPI> UserData registration limit exceeded.!\n");
  userdata[nudata] = d;
  pup_ud[nudata] = f;
  nudata++;
  return (nudata-1);
}

void *
ampi::get_userdata(int idx)
{
  return userdata[idx];
}

// This is invoked in the Fortran (and C ?) version of AMPI
void
ampi::run(ArgsInfo *msg)
{
  int argc = msg->argc;
  char **argv = msg->argv;
  delete msg;
  CtvInitialize(ampi *, ampiPtr);
  CtvAccess(ampiPtr) = this;
  ampimain::ampi_comms[commidx].mainfunc(argc, argv);
  CProxy_ampimain mp(ampimain::handle);
  mp.done();
}

// This is invoked in the C++ version of AMPI
void
ampi::run(void)
{
  CtvInitialize(ampi *, ampiPtr);
  CtvAccess(ampiPtr) = this;
  start();
}

void
ampi::start(void)
{
  CkPrintf("You should write your own start(). \n");
}

extern "C" void
AMPI_Migrate(void)
{
  ampi *ptr = CtvAccess(ampiPtr);
  ptr->thread_id = CthSelf();
  int idx = ptr->thisIndex;
  CProxy_ampi aproxy(ampimain::ampi_comms[ptr->commidx].aid);
  aproxy[idx].migrate();
  ptr->stop_running();
  CthSuspend();
  ptr = CtvAccess(ampiPtr);
  ptr->start_running();
}

extern "C" 
int AMPI_Init(int *argc, char*** argv)
{
  return 0;
}

extern "C" 
int AMPI_Comm_rank(AMPI_Comm comm, int *rank)
{
  *rank = CtvAccess(ampiPtr)->getIndex();
  return 0;
}

extern "C" 
int AMPI_Comm_size(AMPI_Comm comm, int *size)
{
  *size = CtvAccess(ampiPtr)->getArraySize();
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
int AMPI_Recv(void *msg, int count, AMPI_Datatype type, int src, int tag, 
              AMPI_Comm comm, AMPI_Status *status)
{
  ampi *ptr = CtvAccess(ampiPtr);
  ptr->recv(tag,src,msg,count,type, comm);
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
    ampimain::ampi_comms[ptr->commidx].rspec.type = 1;
    ampimain::ampi_comms[ptr->commidx].rspec.root = root;
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
    ampimain::ampi_comms[ptr->commidx].rspec.type = 0;
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
                  req->type, req->comm);
      }
    } else { // irecv request
      int index = request[i] - 100;
      PersReq *req = &(ptr->irequests[index]);
      ptr->recv(req->tag, req->proc, req->buf, req->count, 
                req->type, req->comm);
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
void AMPI_Print(char *str)
{
  ampi *ptr = CtvAccess(ampiPtr);
  CkPrintf("[%d] %s\n", ptr->thisIndex, str);
}

extern "C"
int AMPI_Register(void *d, AMPI_PupFn f)
{
  ampi *ptr = CtvAccess(ampiPtr);
  return ptr->register_userdata(d, f);
}

extern "C"
void *AMPI_Get_userdata(int idx)
{
  ampi *ptr = CtvAccess(ampiPtr);
  return ptr->get_userdata(idx);
}

#include "ampi.def.h"
