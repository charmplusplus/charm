/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "ampiimpl.h"
// for strlen
#include <string.h>

#if AMPI_FORTRAN
#include "ampimain.decl.h"
extern "C" void main_(int, char **);
#endif
// FIXME: find good names for these user-provided functions
extern "C" void get_size_(int *, int *, int *, int *);
extern "C" void pack_(char*,int*,int*,int*,float*,int*,int*,int*);
extern "C" void unpack_(char*,int*,int*,int*,float*,int*,int*,int*);

void*
ArgsInfo::pack(ArgsInfo* msg)
{
  int argsize, i;
  for(i=0;i<msg->argc;i++) {
    argsize += (strlen(msg->argv[i])+1); // +1 for '\0'
  }
  void *p = CkAllocBuffer(msg, sizeof(ArgsInfo) +
                               (msg->argc*sizeof(char*)) + 
			        argsize);
  memcpy(p,msg,sizeof(ArgsInfo));
  char *args = (char *)((char*)p+sizeof(ArgsInfo)+(msg->argc+sizeof(char*)));
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
  int argc = *((int*)in);
  char **argv = (char**)((char*)in+sizeof(ArgsInfo));
  ArgsInfo* msg = new (in) ArgsInfo(argc, argv);
  char *tmp = ((char*)in+sizeof(ArgsInfo)+(argc*sizeof(char*)));
  for(int i=0;i<argc;i++) {
    argv[i] = tmp;
    while(*tmp) { tmp++; }
    tmp++;
  }
  return msg;
}

int migHandle;
CtvDeclare(ampi *, ampiPtr);
CtvDeclare(int, numMigrateCalls);
static CkArray *ampiArray;


ampi::ampi(void)
{
  msgs = CmmNew();
  thread_id = 0;
  nbcasts = 0;
  ampiArray = thisArray;
  nrequests = 0;
  myDDT = new DDT ;
  nirequests = 0;
  firstfree = 0;
  nReductions = 0;
  nAllReductions = 0;
  niRecvs = niSends = biRecv = biSend = 0;
  int i;
  for(i=0;i<100;i++) {
    irequests[i].nextfree = (i+1)%100;
    irequests[i].prevfree = ((i-1)+100)%100;
  }
}

ampi::ampi(CkMigrateMessage *msg)
{
  ampiArray = thisArray;
  nrequests = 0;
}

void
ampi::generic(AmpiMsg* msg)
{
  int tags[2];
  tags[0] = msg->tag1; tags[1] = msg->tag2;
  CmmPut(msgs, 2, tags, msg);
  if(thread_id) {
    CthAwaken(thread_id);
    thread_id = 0;
  }
}

void 
ampi::send(int t1, int t2, void* buf, int count, int type,  int idx)
{
  DDT_DataType *ddt = myDDT->getType(type);
  int len = ddt->getSize(count);
  AmpiMsg *msg = new (&len, 0) AmpiMsg(t1, t2, len);
  ddt->serialize((char*)buf, (char*)msg->data, count, 1);
  CProxy_ampi pa(thisArrayID);
  pa[idx].generic(msg);
}

void 
ampi::sendraw(int t1, int t2, void* buf, int len, CkArrayID aid, int idx)
{
  AmpiMsg *msg = new (&len, 0) AmpiMsg(t1, t2, len);
  memcpy(msg->data, buf, len);
  CProxy_ampi pa(aid);
  pa[idx].generic(msg);
}

void 
ampi::recv(int t1, int t2, void* buf, int count, int type)
{
  int tags[2];
  AmpiMsg *msg = 0;
  DDT_DataType *ddt = myDDT->getType(type);
  int len = ddt->getSize(count);
  while(1) {
    tags[0] = t1; tags[1] = t2;
    msg = (AmpiMsg *) CmmGet(msgs, 2, tags, 0);
    if (msg) break;
    thread_id = CthSelf();
    CthSuspend();
  }
  if (msg->length < len) {
    CkError("AMPI: Expecting message of length %d, received %d\n",
            len, msg->length);
    CkAbort("Exiting.\n");
  }
  ddt->serialize((char*)buf, (char*)msg->data, count, (-1));
  delete msg;
}

#define AMPI_BCAST_TAG  1025
#define AMPI_BARR_TAG   1026
#define AMPI_REDUCE_TAG 1027
#define AMPI_GATHER_TAG 1028

void 
ampi::barrier(void)
{
  if(thisIndex) {
    send(AMPI_BARR_TAG, 0, 0, 0, 0, 0);
    recv(AMPI_BARR_TAG, 0, 0, 0, 0);
  } else {
    int i;
    for(i=1;i<numElements;i++) recv(AMPI_BARR_TAG, 0, 0, 0, 0);
    for(i=1;i<numElements;i++) send(AMPI_BARR_TAG, 0, 0, 0, 0, i);
  }
}

void 
ampi::bcast(int root, void* buf, int count, int type)
{
  if(root==thisIndex) {
    int i;
    for(i=0;i<numElements;i++)
      send(AMPI_BCAST_TAG, nbcasts, buf, count, type, i);
  }
  recv(AMPI_BCAST_TAG, nbcasts, buf, count, type);
  nbcasts++;
}

void 
ampi::reduce(int root, int op, void* inb, void *outb, int count, int type)
{
}

void ampi::pup(PUP::er &p)
{
  ArrayElement1D::pup(p);//Pack superclass
  CkAbort("Pupper for AMPI is not implemented yet.\n");
}

void
ampi::run(ArgsInfo *msg)
{
#if AMPI_FORTRAN
  CtvInitialize(ampi *, ampiPtr);
  CtvAccess(ampiPtr) = this;
  CtvInitialize(int, numMigrateCalls);
  CtvAccess(numMigrateCalls) = 0;

  main_(msg->argc, msg->argv);

  CProxy_ampimain mp(mainhandle);
  mp.done();
  CthSuspend();
#else
  CkPrintf("You should link ampi using -language ampif\n");
#endif
}

void
ampi::run(void)
{
  CtvInitialize(ampi *, ampiPtr);
  CtvAccess(ampiPtr) = this;
  CtvInitialize(int, numMigrateCalls);
  CtvAccess(numMigrateCalls) = 0;

  start();

  CthSuspend();
}

void
ampi::start(void)
{
  CkPrintf("You should write your own start(). \n");
}

// migrate to next processor every 2 iterations
// needs to be changed in a major way
// to provide some sort of user control over migration
extern "C" 
void migrate_(void *gptr)
{
  ampi *ptr = CtvAccess(ampiPtr);;
  CtvAccess(numMigrateCalls)++;
  if(CtvAccess(numMigrateCalls)%2 == 0) {
    int index = ptr->getIndex();
    int where = (CkMyPe()+1) % CkNumPes();
    if(where == CkMyPe())
      return;
    CProxy_migrator pmg(migHandle);
    pmg.migrateElement(new MigrateInfo(ptr, where), CkMyPe());
    int csize, isize, rsize, fsize;
    get_size_(&csize, &isize, &rsize, &fsize);
    int totsize = MyAlign8(csize)+isize+rsize+fsize;
    void *pb = malloc(totsize);
    char *cb = (char *)pb;
    int *ib = (int *) (cb+MyAlign8(csize));
    float *rb = (float *)(ib+isize/sizeof(int));
    int *fb = (int *)(rb+rsize/sizeof(float));
    pack_(cb, &csize, ib, &isize, rb, &rsize, fb, &fsize);
    ptr->csize = csize; ptr->isize = isize; ptr->rsize = rsize; 
    ptr->fsize = fsize; ptr->totsize = totsize; ptr->packedBlock = pb;
    CthSuspend();
    CtvAccess(ampiPtr) = ptr = 
      (ampi*) ampiArray->getElement(CkArrayIndex1D(index));
    pb = ptr->packedBlock; csize = ptr->csize; isize = ptr->isize;
    rsize = ptr->rsize; fsize = ptr->fsize;
    cb = (char *)pb;
    ib = (int *) (cb+MyAlign8(csize));
    rb = (float *)(ib+isize/sizeof(int));
    fb = (int *)(rb+rsize/sizeof(float));
    unpack_(cb, &csize, ib, &isize, rb, &rsize, fb, &fsize);
    free(pb);
  }
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
  *size = CtvAccess(ampiPtr)->getSize();
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
  ptr->send(tag, ptr->getIndex(), msg, count, type, dest);
  return 0;
}

extern "C" 
int AMPI_Recv(void *msg, int count, AMPI_Datatype type, int src, int tag, 
              AMPI_Comm comm, AMPI_Status *status)
{
  ampi *ptr = CtvAccess(ampiPtr);
  ptr->recv(tag,src,msg,count,type);
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
  ampi *ptr = CtvAccess(ampiPtr);
  ptr->barrier();
  return 0;
}

extern "C" 
int AMPI_Bcast(void *buf, int count, AMPI_Datatype type, int root, 
                         AMPI_Comm comm)
{
  ampi *ptr = CtvAccess(ampiPtr);
  ptr->bcast(root, buf, count, type);
  return 0;
}

extern "C" 
int AMPI_Reduce(void *inbuf, void *outbuf, int count, int type, AMPI_Op op, 
                int root, AMPI_Comm comm)
{
  ampi *ptr = CtvAccess(ampiPtr);
  ptr->reduce(root,op,inbuf,outbuf,count,type);
  return 0;
}

extern "C" 
int AMPI_Allreduce(void *inbuf, void *outbuf, int count, int type,
                   AMPI_Op op, AMPI_Comm comm)
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
  ampi *ptr = CtvAccess(ampiPtr);
  int size = ptr->myDDT->getType(type)->getSize(count) ;
  ptr->contribute(size, inbuf, mytype);
  ptr->nAllReductions++;
  ptr->recv(0, AMPI_BCAST_TAG, outbuf, count, type);
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
              req->proc);
  } // recv request is handled in waitall
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
        ptr->recv(req->tag, req->proc, req->buf, req->count, req->type);
      }
    } else { // irecv request
      int index = request[i] - 100;
      PersReq *req = &(ptr->irequests[index]);
      ptr->recv(req->tag, req->proc, req->buf, req->count, req->type);
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
  
  ptr->niSends++;
  ptr->send(tag, ptr->getIndex(), buf, count, type, dest);
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

  ptr->niRecvs++;
  PersReq *req = &(ptr->irequests[ptr->firstfree]);
  req->sndrcv = 2;
  req->buf = buf;
  req->count = count;
  req->type = type;
  req->proc = src;
  req->tag = tag;
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
  ampi *ptr = CtvAccess(ampiPtr);
  int size = ptr->getSize();
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
  ampi *ptr = CtvAccess(ampiPtr);
  int size = ptr->getSize();
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
  ampi *ptr = CtvAccess(ampiPtr);
  int size = ptr->getSize();
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
  ampi *ptr = CtvAccess(ampiPtr);
  int size = ptr->getSize();
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
  ampi *ptr = CtvAccess(ampiPtr);
  int size = ptr->getSize();
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
  ampi *ptr = CtvAccess(ampiPtr);
  int size = ptr->getSize();
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

#include "ampi.def.h"
