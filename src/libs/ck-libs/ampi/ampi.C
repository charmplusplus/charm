#include "ampiimpl.h"

extern void _initCharm(int argc, char **argv);

extern "C" void conversemain_(int *argc,char _argv[][80],int length[])
{
  int i;
  char **argv = new char*[*argc+2];

  for(i=0;i <= *argc;i++) {
    if (length[i] < 100) {
      _argv[i][length[i]]='\0';
      argv[i] = &(_argv[i][0]);
    } else {
      argv[i][0] = '\0';
    }
  }
  argv[*argc+1]=0;
  
  ConverseInit(*argc, argv, _initCharm, 0, 0);
}

CkChareID mainhandle;

main::main(CkArgMsg *m)
{
  int i;
  for(i=1;i<m->argc;i++) {
    if(strncmp(m->argv[i], "+vp", 3) == 0) {
      if (strlen(m->argv[i]) > 2) {
        sscanf(m->argv[i], "+vp%d", &nblocks);
      } else {
        if (m->argv[i+1]) {
          sscanf(m->argv[i+1], "%d", &nblocks);
        }
      }
      break;
    }
  }
  CProxy_migrator::ckNew();
  numDone = 0;
  delete m;
  // CkGroupID mapID = CProxy_BlockMap::ckNew();
  // CProxy_ampi jarray(nblocks, mapID);
  CProxy_ampi jarray(nblocks);
  for(i=0; i<nblocks; i++)
    jarray[i].run();
  mainhandle = thishandle;
}

void
main::qd(void)
{
  // CkWaitQD();
  // CkPrintf("Created Elements\n");
  CProxy_ampi jarray(arr);
  for(int i=0; i<nblocks; i++)
    jarray[i].run();
  return;
}

void
main::done(void)
{
  numDone++;
  if(numDone==nblocks) {
    ckout << "Exiting" << endl;
    CkExit();
  }
}

int migHandle;

CtvDeclare(ampi *, ampiPtr);
CtvDeclare(int, numMigrateCalls);
extern "C" void main_(void);
static Array1D *ampiArray;

extern "C" void get_size_(int *, int *, int *, int *);
extern "C" void pack_(char*,int*,int*,int*,float*,int*,int*,int*);
extern "C" void unpack_(char*,int*,int*,int*,float*,int*,int*,int*);

ampi::ampi(ArrayElementCreateMessage *msg) : TempoArray(msg)
{
  ampiArray = thisArray;
  nrequests = 0;
  ntypes = 0;
  nirequests = 0;
  firstfree = 0;
  int i;
  for(i=0;i<100;i++) {
    irequests[i].nextfree = (i+1)%100;
    irequests[i].prevfree = ((i-1)+100)%100;
  }
  nbcasts = 0;
  // delete msg;
}

ampi::ampi(ArrayElementMigrateMessage *msg) : TempoArray(msg)
{
  //CkPrintf("[%d] called migration constructor\n", thisIndex);
  nrequests = 0;
  ampiArray = thisArray;
  void *buf = msg->packData;
  totsize = *(int *)buf;
  buf = (void *) ((int *)buf + 1);
  csize = *(int *)buf;
  buf = (void *) ((int *)buf + 1);
  isize = *(int *)buf;
  buf = (void *) ((int *)buf + 1);
  rsize = *(int *)buf;
  buf = (void *) ((int *)buf + 1);
  fsize = *(int *)buf;
  buf = (void *) ((int *)buf + 1);
  packedBlock = malloc(totsize);
  memcpy(packedBlock, buf, totsize);
  buf = (void *) ((char *)buf + totsize);
  thread_id = CthUnpackThread(buf);
  CthAwaken(thread_id);
  finishMigration(); 
  delete msg;
  //CkPrintf("[%d] finished migration constructor\n", thisIndex);
}

int
ampi::packsize(void)
{
  return 5*sizeof(int) + totsize + CthPackBufSize(thread_id);
}

void
ampi::pack(void *buf)
{
  TempoMessage *msg;
  int itags[2];
  itags[0] = TEMPO_ANY; itags[1] = TEMPO_ANY;
  // resend pending messages in table
  while((msg=(TempoMessage *)CmmGet(tempoMessages, 2, itags, 0))) {
    ckTempoSendElem(msg->tag1, msg->tag2, msg->data, msg->length, thisIndex);
  }
  *(int *)buf = totsize;
  buf = (void *) ((int *)buf + 1);
  *(int *)buf = csize;
  buf = (void *) ((int *)buf + 1);
  *(int *)buf = isize;
  buf = (void *) ((int *)buf + 1);
  *(int *)buf = rsize;
  buf = (void *) ((int *)buf + 1);
  *(int *)buf = fsize;
  buf = (void *) ((int *)buf + 1);
  memcpy(buf, packedBlock, totsize);
  free(packedBlock);
  buf = (void *) ((char *)buf + totsize);
  CthPackThread(thread_id, buf);
}

void
ampi::run(void)
{
  static int initCtv = 0;
  int myIdx = getIndex();
  ampi *myThis;

  if(!initCtv) {
    CtvInitialize(ampi *, ampiPtr);
    CtvInitialize(int, numMigrateCalls);
    initCtv = 1;
  }
  CtvAccess(ampiPtr) = this;
  CtvAccess(numMigrateCalls) = 0;

  // CkPrintf("[%d] main_ called\n", getIndex());
  main_();
  myThis = (ampi*) ampiArray->getElement(myIdx);
  //CkPrintf("[%d] main_ finished\n", myThis->getIndex());

  // itersDone();

  myThis->ckTempoBarrier();
  if(myThis->getIndex()==0)
    CmiAbort("");
  else
    CthSuspend();
}

extern "C" void migrate_(void *gptr)
{
  ampi *ptr = CtvAccess(ampiPtr);;
  CtvAccess(numMigrateCalls)++;
  // migrate to next processor every 2 iterations
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
    //CkPrintf("[%d] Migrating from %d to %d\n", index, CkMyPe(), where);
    CthSuspend();
    //CkPrintf("[%d] awakened on %d \n", index, CkMyPe());
    CtvAccess(ampiPtr) = ptr = (ampi*) ampiArray->getElement(index);
    pb = ptr->packedBlock; csize = ptr->csize; isize = ptr->isize;
    rsize = ptr->rsize; fsize = ptr->fsize;
    cb = (char *)pb;
    ib = (int *) (cb+MyAlign8(csize));
    rb = (float *)(ib+isize/sizeof(int));
    fb = (int *)(rb+rsize/sizeof(float));
    unpack_(cb, &csize, ib, &isize, rb, &rsize, fb, &fsize);
    free(pb);
    //CkPrintf("[%d] Migrated to %d\n", index, CkMyPe());
  }
}
extern "C" int MPI_Init(int *argc, char*** argv)
{
  return 0;
}

extern "C" void mpi_init_(int *ierr)
{
  *ierr = MPI_Init(0,0);
}

extern "C" int MPI_Comm_rank(MPI_Comm comm, int *rank)
{
  *rank = CtvAccess(ampiPtr)->getIndex();
  return 0;
}

extern "C" void mpi_comm_rank_(int *comm, int *rank, int *ierr)
{
  *ierr = MPI_Comm_rank(*comm, rank);
}

extern "C" int MPI_Comm_size(MPI_Comm comm, int *size)
{
  *size = CtvAccess(ampiPtr)->getSize();
  return 0;
}

extern "C" void mpi_comm_size_(int *comm, int *size, int *ierr)
{
  *ierr = MPI_Comm_size(*comm, size);
}

extern "C" int MPI_Finalize(void)
{
  return 0;
}

extern "C" void mpi_finalize_(int *ierr)
{
  *ierr = MPI_Finalize();
}

static int typesize(int type, int count, ampi* ptr)
{
  switch(type) {
    case MPI_DOUBLE_PRECISION : return count*sizeof(double);
    case MPI_INTEGER : return count*sizeof(int);
    case MPI_REAL : return count*sizeof(float);
    case MPI_COMPLEX: return 2*count*sizeof(double);
    case MPI_LOGICAL: return count*sizeof(int);
    case MPI_CHARACTER: return count*sizeof(char);
    case MPI_BYTE: return count;
    case MPI_PACKED: return count;
    default:
      if((type >= 100) && 
         (type-100 < ptr->ntypes))
      { 
        // user-defined type
        return count*(ptr->types[type-100]);
      }
      CmiError("Type %d not supported yet!\n", type);
      CmiAbort("");
      return 0; // keep compiler happy
  }
}

extern "C" int MPI_Send(void *msg, int count, MPI_Datatype type, int dest, 
                        int tag, MPI_Comm comm)
{
  ampi *ptr = CtvAccess(ampiPtr);
  int size = typesize(type, count, ptr);
  //CkPrintf("[%d] sending %d bytes to %d tagged %d\n", ptr->getIndex(), 
           //size, dest, tag);
  ptr->ckTempoSendElem(tag, ptr->getIndex(), msg, size, dest);
  return 0;
}

extern "C" void mpi_send_(void *msg, int *count, int *type, int *dest, 
                          int *tag, int *comm, int *ierr)
{
  *ierr = MPI_Send(msg, *count, *type, *dest, *tag, *comm);
}

extern "C" int MPI_Recv(void *msg, int count, int type, int src, int tag,
                        MPI_Comm comm, MPI_Status *status)
{
  ampi *ptr = CtvAccess(ampiPtr);
  int size = typesize(type, count, ptr);
  //CkPrintf("[%d] waits for %d bytes tagged %d\n",ptr->getIndex(),size,tag);
  ptr->ckTempoRecv(tag, src, msg, size);
  //CkPrintf("[%d] received %d bytes tagged %d\n", ptr->getIndex(), size, tag);
  return 0;
}

extern "C" void mpi_recv_(void *msg, int *count, int *type, int *src,
                          int *tag, int *comm, int *status, int *ierr)
{
  *ierr = MPI_Recv(msg, *count, *type, *src, *tag, *comm, (MPI_Status*)status);
}

extern "C" int MPI_Sendrecv(void *sbuf, int scount, int stype, int dest, 
                            int stag, void *rbuf, int rcount, int rtype,
                            int src, int rtag, MPI_Comm comm, MPI_Status *sts)
{
  return (MPI_Send(sbuf,scount,stype,dest,stag,comm) ||
          MPI_Recv(rbuf,rcount,rtype,src,rtag,comm,sts));
}

extern "C" void mpi_sendrecv_(void *sndbuf, int *sndcount, int *sndtype, 
                              int *dest, int *sndtag, void *rcvbuf, 
                              int *rcvcount, int *rcvtype, int *src, 
                              int *rcvtag, int *comm, int *status, int *ierr)
{
  mpi_send_(sndbuf, sndcount, sndtype, dest, sndtag, comm, ierr);
  mpi_recv_(rcvbuf, rcvcount, rcvtype, src, rcvtag, comm, status, ierr);
}

extern "C" int MPI_Barrier(MPI_Comm comm)
{
  ampi *ptr = CtvAccess(ampiPtr);
  //CkPrintf("[%d] Barrier called\n", ptr->getIndex());
  ptr->ckTempoBarrier();
  //CkPrintf("[%d] Barrier finished\n", ptr->getIndex());
  return 0;
}

extern "C" void mpi_barrier_(int *comm, int *ierr)
{
  *ierr = MPI_Comm(*comm);
}

#define MPI_BCAST_TAG 999

extern "C" int MPI_Bcast(void *buf, int count, int type, int root, 
                         MPI_Comm comm)
{
  ampi *ptr = CtvAccess(ampiPtr);
  int size = typesize(type, count, ptr);
  ptr->nbcasts++;
  // CkPrintf("[%d] %dth Broadcast called size=%d\n", ptr->getIndex(), 
           // ptr->nbcasts, size);
  ptr->ckTempoBcast(((root)==ptr->getIndex())?1:0, 
                    MPI_BCAST_TAG+ptr->nbcasts, buf, size);
  // CkPrintf("[%d] %dth Broadcast finished\n", ptr->getIndex(), ptr->nbcasts);
  return 0;
}

extern "C" void mpi_bcast_(void *buf, int *count, int *type, int *root, 
                           int *comm, int *ierr)
{
  *ierr = MPI_Bcast(buf, *count, *type, *root, *comm);
}

static void optype(int inop, int intype, int *outop, int *outtype)
{
  switch(inop) {
    case MPI_MAX : *outop = TEMPO_MAX; break;
    case MPI_MIN : *outop = TEMPO_MIN; break;
    case MPI_SUM : *outop = TEMPO_SUM; break;
    case MPI_PROD : *outop = TEMPO_PROD; break;
    default:
      ckerr << "Op " << inop << " not supported." << endl;
      CmiAbort("exiting");
  }
  switch(intype) {
    case MPI_REAL : *outtype = TEMPO_FLOAT; break;
    case MPI_INTEGER : *outtype = TEMPO_INT; break;
    case MPI_DOUBLE_PRECISION : *outtype = TEMPO_DOUBLE; break;
    default:
      ckerr << "Type " << intype << " not supported." << endl;
      CmiAbort("exiting");
  }
}

extern "C" int MPI_Reduce(void *inbuf, void *outbuf, int count, int type,
                          MPI_Op op, int root, MPI_Comm comm)
{
  int myop, mytype;
  optype(op, type, &myop, &mytype);
  ampi *ptr = CtvAccess(ampiPtr);
  //CkPrintf("[%d] reduction called\n", ptr->getIndex());
  ptr->ckTempoReduce(root,myop,inbuf,outbuf,count,mytype);
  //CkPrintf("[%d] reduction finished\n", ptr->getIndex());
  return 0;
}

extern "C" void mpi_reduce_(void *inbuf, void *outbuf, int *count, int *type,
                            int *op, int *root, int *comm, int *ierr)
{
  *ierr = MPI_Reduce(inbuf, outbuf, *count, *type, *op, *root, *comm);
}

extern "C" int MPI_Allreduce(void *inbuf, void *outbuf, int count, int type,
                          MPI_Op op, MPI_Comm comm)
{
  int myop, mytype;
  optype(op, type, &myop, &mytype);
  ampi *ptr = CtvAccess(ampiPtr);
  //CkPrintf("[%d] Allreduce called\n", ptr->getIndex());
  ptr->ckTempoAllReduce(myop,inbuf,outbuf,count,mytype);
  //CkPrintf("[%d] Allreduce finished\n", ptr->getIndex());
  return 0;
}

extern "C" void mpi_allreduce_(void *inbuf,void *outbuf,int *count,int *type,
                               int *op, int *comm, int *ierr)
{
  *ierr = MPI_Allreduce(inbuf, outbuf, *count, *type, *op, *comm);
}

extern "C" double MPI_Wtime(void)
{
  return CmiWallTimer();
}

extern "C" double mpi_wtime_(void)
{
  return MPI_Wtime();
}

extern "C" int MPI_Start(MPI_Request *reqnum)
{
  ampi *ptr = CtvAccess(ampiPtr);
  if(*reqnum >= ptr->nrequests) {
    CmiAbort("Invalid persistent Request..\n");
  }
  PersReq *req = &(ptr->requests[*reqnum]);
  if(req->sndrcv == 1) { // send request
    // CkPrintf("[%d] sending buf=%p, size=%d, tag=%d to %d\n", ptr->getIndex(),
             // req->buf, req->size, req->tag, req->proc);
    ptr->ckTempoSendElem(req->tag, ptr->getIndex(), req->buf, 
                         req->size, req->proc);
  } // recv request is handled in waitall
  return 0;
}

extern "C" void mpi_start_(int *reqnum, int *ierr)
{
  *ierr = MPI_Start((MPI_Request*) reqnum);
}

extern "C" int MPI_Waitall(int count, MPI_Request *request, MPI_Status *sts)
{
  ampi *ptr = CtvAccess(ampiPtr);
  int i;
  for(i=0;i<count;i++) {
    if(request[i] == (-1))
      continue;
    if(request[i] < 100) { // persistent request
      PersReq *req = &(ptr->requests[request[i]]);
      if(req->sndrcv == 2) { // recv request
        ptr->ckTempoRecv(req->tag, req->proc, req->buf, req->size);
        // CkPrintf("[%d] received buf=%p, size=%d, tag=%d from %d\n", 
                // ptr->getIndex(), req->buf, req->size, req->tag, req->proc);
      }
    } else { // irecv request
      int index = request[i] - 100;
      PersReq *req = &(ptr->irequests[index]);
      // CkPrintf("[%d] waiting for size=%d, tag=%d from %d\n", 
              // ptr->getIndex(), req->size, req->tag, req->proc);
      ptr->ckTempoRecv(req->tag, req->proc, req->buf, req->size);
      // CkPrintf("[%d] received buf=%p, size=%d, tag=%d from %d\n", 
              // ptr->getIndex(), req->buf, req->size, req->tag, req->proc);
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

extern "C" void mpi_waitall_(int *count, int *request, int *status, int *ierr)
{
  *ierr = MPI_Waitall(*count, (MPI_Request*) request, (MPI_Status*) status);
}

extern "C" int MPI_Recv_init(void *buf, int count, int type, int src, int tag,
                             MPI_Comm comm, MPI_Request *req)
{
  ampi *ptr = CtvAccess(ampiPtr);
  if(ptr->nrequests == 100) {
    CmiAbort("Too many persistent commrequests.\n");
  }
  int size = typesize(type, count, ptr);
  ptr->requests[ptr->nrequests].sndrcv = 2;
  ptr->requests[ptr->nrequests].buf = buf;
  ptr->requests[ptr->nrequests].size = size;
  ptr->requests[ptr->nrequests].proc = src;
  ptr->requests[ptr->nrequests].tag = tag;
  *req = ptr->nrequests;
  ptr->nrequests ++;
  // CkPrintf("[%d] recv request %d buf=%p, count=%d size=%d,tag=%d from %d\n",
            // ptr->getIndex(), ptr->nrequests-1, buf, count, size, tag, *src);
  return 0;
}

extern "C" void mpi_recv_init_(void *buf, int *count, int *type, int *srcpe,
                               int *tag, int *comm, int *req, int *ierr)
{
  *ierr = MPI_Recv_init(buf,*count,*type,*srcpe,*tag,*comm,(MPI_Request*)req);
}

extern "C" int MPI_Send_init(void *buf, int count, int type, int dest, int tag,
                             MPI_Comm comm, MPI_Request *req)
{
  ampi *ptr = CtvAccess(ampiPtr);
  if(ptr->nrequests == 100) {
    CmiAbort("Too many persistent commrequests.\n");
  }
  int size = typesize(type, count, ptr);
  ptr->requests[ptr->nrequests].sndrcv = 1;
  ptr->requests[ptr->nrequests].buf = buf;
  ptr->requests[ptr->nrequests].size = size;
  ptr->requests[ptr->nrequests].proc = dest;
  ptr->requests[ptr->nrequests].tag = tag;
  *req = ptr->nrequests;
  ptr->nrequests ++;
  // CkPrintf("[%d] send request %d buf=%p, count=%d size=%d, tag=%d, to %d\n",
           // ptr->getIndex(), ptr->nrequests-1, buf, count, size, tag, destpe);
  return 0;
}

extern "C" void mpi_send_init_(void *buf, int *count, int *type, int *destpe,
                               int *tag, int *comm, int *req, int *ierr)
{
  *ierr = MPI_Send_init(buf,*count,*type,*destpe,*tag,*comm,(MPI_Request*)req);
}

extern "C" int MPI_Type_contiguous(int count, MPI_Datatype oldtype, 
                                   MPI_Datatype *newtype)
{
  ampi *ptr = CtvAccess(ampiPtr);
  *newtype = ptr->ntypes;
  ptr->types[*newtype] =  typesize(oldtype, count, ptr);
  ptr->ntypes ++;
  *newtype += 100;
  return 0;
}

extern "C" int MPI_Type_commit(MPI_Datatype *datatype)
{
  return 0;
}

extern "C" int MPI_Type_free(MPI_Datatype *datatype)
{
  return 0;
}

extern "C" void mpi_type_contiguous_(int *count, int *oldtype, int *newtype, 
                                int *ierr)
{
  *ierr = MPI_Type_contiguous(*count, *oldtype, newtype);
}

extern "C" void mpi_type_commit_(int *type, int *ierr)
{
  *ierr = MPI_Type_commit(type);
}

extern "C" void mpi_type_free_(int *type, int *ierr)
{
  *ierr = MPI_Type_free(type);
}

extern "C" int MPI_Isend(void *buf, int count, MPI_Datatype datatype, int dest, 
              int tag, MPI_Comm comm, MPI_Request *request)
{
    ampi *ptr = CtvAccess(ampiPtr);
    ptr->ckTempoSendElem(tag, ptr->getIndex(), buf, 
                         typesize(datatype, count, ptr), dest);
    *request = (-1);
    return 0;
}

extern "C" int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int src, 
              int tag, MPI_Comm comm, MPI_Request *request)
{
  ampi *ptr = CtvAccess(ampiPtr);
  if(ptr->nirequests == 100) {
    CmiAbort("Too many Irecv requests.\n");
  }
  int size = typesize(datatype, count, ptr);
  PersReq *req = &(ptr->irequests[ptr->firstfree]);
  req->sndrcv = 2;
  req->buf = buf;
  req->size = size;
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

extern "C" void mpi_isend_(void *buf, int *count, int *datatype, int *dest,
                           int *tag, int *comm, int *request, int *ierr)
{
  *ierr = MPI_Isend(buf, *count, *datatype, *dest, *tag, *comm, request);
}

extern "C" void mpi_irecv_(void *buf, int *count, int *datatype, int *src,
                           int *tag, int *comm, int *request, int *ierr)
{
  *ierr = MPI_Irecv(buf, *count, *datatype, *src, *tag, *comm, request);
}

#include "ampi.def.h"
