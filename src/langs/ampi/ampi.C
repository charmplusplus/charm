/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "ampiimpl.h"

#ifdef AMPI_FORTRAN
#include "ampimain.decl.h"
#endif

/*
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
CkArrayID _ampiAid;

static void allReduceHandler(void *,int dataSize,void *data)
{
  TempoArray::ckTempoBcast(0, data, dataSize, _ampiAid);
}

ampimain::ampimain(CkArgMsg *m)
{
  int i;
  nblocks = CkNumPes();
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
  _ampiAid = CProxy_ampi::ckNew(nblocks);
  // CkRegisterArrayReductionHandler(_ampiAid,allReduceHandler,0);
  CProxy_ampi jarray(_ampiAid);
  jarray.setReductionClient(allReduceHandler,0);
  for(i=0; i<nblocks; i++) {
    jarray[i].run();
  }
  mainhandle = thishandle;
}

void
ampimain::qd(void)
{
  // CkWaitQD();
  // CkPrintf("Created Elements\n");
  CProxy_ampi jarray(arr);
  for(int i=0; i<nblocks; i++) {
    ArgsInfo *argsinfo = new ArgsInfo(0, NULL);
    jarray[i].run(argsinfo);
  }
  return;
}

//CpvExtern(int, _numSwitches);

void
ampimain::done(void)
{
  numDone++;
  if(numDone==nblocks) {
    // ckout << "Exiting" << endl;
    CkExit();
  }
}

*/

int migHandle;

CtvDeclare(ampi *, ampiPtr);
CtvDeclare(int, numMigrateCalls);
extern "C" void main_(int, char **);
static CkArray *ampiArray;

extern "C" void get_size_(int *, int *, int *, int *);
extern "C" void pack_(char*,int*,int*,int*,float*,int*,int*,int*);
extern "C" void unpack_(char*,int*,int*,int*,float*,int*,int*,int*);

ampi::ampi(void)
{
  ampiArray = thisArray;
  nrequests = 0;
  ntypes = 0;
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
  nbcasts = 0;
}

ampi::ampi(ArrayElementMigrateMessage *msg)
{
  ampiArray = thisArray;
  nrequests = 0;
}

void ampi::pup(PUP::er &p)
{
	ArrayElement1D::pup(p);//Pack superclass
	
	if (p.isPacking())
	{ // resend pending messages in table
	  TempoMessage *msg;
	  int itags[2];
	  itags[0] = TEMPO_ANY; itags[1] = TEMPO_ANY;
	  while((msg=(TempoMessage *)CmmGet(tempoMessages, 2, itags, 0))) {
	    ckTempoSendElem(msg->tag1, msg->tag2, msg->data, msg->length, thisIndex);
	  }
	}
	
	
	//Pack/unpack all our data
	p(csize);p(isize);p(rsize);p(fsize);
	p(totsize);
	if (p.isUnpacking()) packedBlock=malloc(totsize);
	p(packedBlock,totsize);
	
	//Pack our thread structure [HACK: we need a CthPup]
	int thBytes;
	if (!p.isUnpacking())  thBytes=CthPackBufSize(thread_id);
	p(thBytes);
	void *buf=malloc(thBytes);//<- temporary pack/unpack buffer
	if (p.isPacking())  CthPackThread(thread_id, buf);
	p(buf,thBytes);
	if (p.isUnpacking())  thread_id=CthUnpackThread(buf);
	free(buf);
	
	//Start our thread on arrival
	if (p.isUnpacking()) 
		CthAwaken(thread_id);
}

void
ampi::run(ArgsInfo *msg)
{
#ifdef AMPI_FORTRAN
  static int initCtv = 0;

  if(!initCtv) {
    CtvInitialize(ampi *, ampiPtr);
    CtvInitialize(int, numMigrateCalls);
    initCtv = 1;
  }

  CtvAccess(ampiPtr) = this;
  CtvAccess(numMigrateCalls) = 0;

  main_(msg->argc, msg->argv);

  // myThis = (ampi*) ampiArray->getElement(myIdx);

  CProxy_ampimain mp(mainhandle);
  mp.done();
  CthSuspend();
#else
  CkPrintf("You should link ampi use -lampif\n");
#endif
}

void
ampi::run(void)
{
  static int initCtv = 0;

  if(!initCtv) {
    CtvInitialize(ampi *, ampiPtr);
    CtvInitialize(int, numMigrateCalls);
    initCtv = 1;
  }

  CtvAccess(ampiPtr) = this;
  CtvAccess(numMigrateCalls) = 0;

  start();

  // myThis = (ampi*) ampiArray->getElement(myIdx);

  CthSuspend();
}

void
ampi::start(void)
{
  CkPrintf("You should write your own start(). \n");
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
    CtvAccess(ampiPtr) = ptr = (ampi*) ampiArray->getElement(CkArrayIndex1D(index));
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
extern "C" int AMPI_Init(int *argc, char*** argv)
{
  return 0;
}

extern "C" void ampi_init_(int *ierr)
{
  *ierr = AMPI_Init(0,0);
}

extern "C" int AMPI_Comm_rank(AMPI_Comm comm, int *rank)
{
  *rank = CtvAccess(ampiPtr)->getIndex();
  return 0;
}

extern "C" void ampi_comm_rank_(int *comm, int *rank, int *ierr)
{
  *ierr = AMPI_Comm_rank(*comm, rank);
}

extern "C" int AMPI_Comm_size(AMPI_Comm comm, int *size)
{
  *size = CtvAccess(ampiPtr)->getSize();
  return 0;
}

extern "C" void ampi_comm_size_(int *comm, int *size, int *ierr)
{
  *ierr = AMPI_Comm_size(*comm, size);
}

extern "C" int AMPI_Finalize(void)
{
  return 0;
}

extern "C" void ampi_finalize_(int *ierr)
{
  *ierr = AMPI_Finalize();
}

static int typesize(int type, int count, ampi* ptr)
{
  switch(type) {
    case AMPI_DOUBLE : return count*sizeof(double);
    case AMPI_INT : return count*sizeof(int);
    case AMPI_FLOAT : return count*sizeof(float);
    case AMPI_COMPLEX: return 2*count*sizeof(double);
    case AMPI_LOGICAL: return count*sizeof(int);
    case AMPI_CHAR: return count*sizeof(char);
    case AMPI_BYTE: return count;
    case AMPI_PACKED: return count;
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

extern "C" int AMPI_Send(void *msg, int count, AMPI_Datatype type, int dest, 
                        int tag, AMPI_Comm comm)
{
  ampi *ptr = CtvAccess(ampiPtr);
  int size = typesize(type, count, ptr);
  //CkPrintf("[%d] sending %d bytes to %d tagged %d\n", ptr->getIndex(), 
           //size, dest, tag);
  ptr->ckTempoSendElem(tag, ptr->getIndex(), msg, size, dest);
  return 0;
}

extern "C" void ampi_send_(void *msg, int *count, int *type, int *dest, 
                          int *tag, int *comm, int *ierr)
{
  *ierr = AMPI_Send(msg, *count, *type, *dest, *tag, *comm);
}

extern "C" int AMPI_Recv(void *msg, int count, int type, int src, int tag,
                        AMPI_Comm comm, AMPI_Status *status)
{
  ampi *ptr = CtvAccess(ampiPtr);
  int size = typesize(type, count, ptr);
  //CkPrintf("[%d] waits for %d bytes tagged %d\n",ptr->getIndex(),size,tag);
  ptr->ckTempoRecv(tag, src, msg, size);
  //CkPrintf("[%d] received %d bytes tagged %d\n", ptr->getIndex(), size, tag);
  return 0;
}

extern "C" void ampi_recv_(void *msg, int *count, int *type, int *src,
                          int *tag, int *comm, int *status, int *ierr)
{
  *ierr = AMPI_Recv(msg, *count, *type, *src, *tag, *comm, (AMPI_Status*)status);
}

extern "C" int AMPI_Sendrecv(void *sbuf, int scount, int stype, int dest, 
                            int stag, void *rbuf, int rcount, int rtype,
                            int src, int rtag, AMPI_Comm comm, AMPI_Status *sts)
{
  return (AMPI_Send(sbuf,scount,stype,dest,stag,comm) ||
          AMPI_Recv(rbuf,rcount,rtype,src,rtag,comm,sts));
}

extern "C" void ampi_sendrecv_(void *sndbuf, int *sndcount, int *sndtype, 
                              int *dest, int *sndtag, void *rcvbuf, 
                              int *rcvcount, int *rcvtype, int *src, 
                              int *rcvtag, int *comm, int *status, int *ierr)
{
  ampi_send_(sndbuf, sndcount, sndtype, dest, sndtag, comm, ierr);
  ampi_recv_(rcvbuf, rcvcount, rcvtype, src, rcvtag, comm, status, ierr);
}

extern "C" int AMPI_Barrier(AMPI_Comm comm)
{
  ampi *ptr = CtvAccess(ampiPtr);
  //CkPrintf("[%d] Barrier called\n", ptr->getIndex());
  ptr->ckTempoBarrier();
  //CkPrintf("[%d] Barrier finished\n", ptr->getIndex());
  return 0;
}

extern "C" void ampi_barrier_(int *comm, int *ierr)
{
  *ierr = AMPI_Comm(*comm);
}

#define AMPI_BCAST_TAG 999

extern "C" int AMPI_Bcast(void *buf, int count, int type, int root, 
                         AMPI_Comm comm)
{
  ampi *ptr = CtvAccess(ampiPtr);
  int size = typesize(type, count, ptr);
  ptr->nbcasts++;
  // CkPrintf("[%d] %dth Broadcast called size=%d\n", ptr->getIndex(), 
           // ptr->nbcasts, size);
  ptr->ckTempoBcast(((root)==ptr->getIndex())?1:0, 
                    AMPI_BCAST_TAG+ptr->nbcasts, buf, size);
  // CkPrintf("[%d] %dth Broadcast finished\n", ptr->getIndex(), ptr->nbcasts);
  return 0;
}

extern "C" void ampi_bcast_(void *buf, int *count, int *type, int *root, 
                           int *comm, int *ierr)
{
  *ierr = AMPI_Bcast(buf, *count, *type, *root, *comm);
}

static void optype(int inop, int intype, int *outop, int *outtype)
{
  switch(inop) {
    case AMPI_MAX : *outop = TEMPO_MAX; break;
    case AMPI_MIN : *outop = TEMPO_MIN; break;
    case AMPI_SUM : *outop = TEMPO_SUM; break;
    case AMPI_PROD : *outop = TEMPO_PROD; break;
    default:
      ckerr << "Op " << inop << " not supported." << endl;
      CmiAbort("exiting");
  }
  switch(intype) {
    case AMPI_FLOAT : *outtype = TEMPO_FLOAT; break;
    case AMPI_INT : *outtype = TEMPO_INT; break;
    case AMPI_DOUBLE : *outtype = TEMPO_DOUBLE; break;
    default:
      ckerr << "Type " << intype << " not supported." << endl;
      CmiAbort("exiting");
  }
}

extern "C" int AMPI_Reduce(void *inbuf, void *outbuf, int count, int type,
                          AMPI_Op op, int root, AMPI_Comm comm)
{
  int myop, mytype;
  optype(op, type, &myop, &mytype);
  ampi *ptr = CtvAccess(ampiPtr);
  //CkPrintf("[%d] reduction called\n", ptr->getIndex());
  ptr->nReductions++;
  ptr->ckTempoReduce(root,myop,inbuf,outbuf,count,mytype);
  //CkPrintf("[%d] reduction finished\n", ptr->getIndex());
  return 0;
}

extern "C" void ampi_reduce_(void *inbuf, void *outbuf, int *count, int *type,
                            int *op, int *root, int *comm, int *ierr)
{
  *ierr = AMPI_Reduce(inbuf, outbuf, *count, *type, *op, *root, *comm);
}

extern "C" int AMPI_Allreduce(void *inbuf, void *outbuf, int count, int type,
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
  int size = typesize(type, count, ptr);
  ptr->contribute(size, inbuf, mytype);
  //CkPrintf("[%d] Allreduce called\n", ptr->getIndex());
  ptr->nAllReductions++;
  ptr->ckTempoRecv(0, BCAST_TAG, outbuf, size);
  //CkPrintf("[%d] Allreduce finished\n", ptr->getIndex());
  return 0;
}

extern "C" void ampi_allreduce_(void *inbuf,void *outbuf,int *count,int *type,
                               int *op, int *comm, int *ierr)
{
  *ierr = AMPI_Allreduce(inbuf, outbuf, *count, *type, *op, *comm);
}

extern "C" double AMPI_Wtime(void)
{
  return CmiWallTimer();
}

extern "C" double ampi_wtime_(void)
{
  return AMPI_Wtime();
}

extern "C" int AMPI_Start(AMPI_Request *reqnum)
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

extern "C" void ampi_start_(int *reqnum, int *ierr)
{
  *ierr = AMPI_Start((AMPI_Request*) reqnum);
}

extern "C" int AMPI_Waitall(int count, AMPI_Request *request, AMPI_Status *sts)
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

extern "C" void ampi_waitall_(int *count, int *request, int *status, int *ierr)
{
  *ierr = AMPI_Waitall(*count, (AMPI_Request*) request, (AMPI_Status*) status);
}

extern "C" int AMPI_Recv_init(void *buf, int count, int type, int src, int tag,
                             AMPI_Comm comm, AMPI_Request *req)
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

extern "C" void ampi_recv_init_(void *buf, int *count, int *type, int *srcpe,
                               int *tag, int *comm, int *req, int *ierr)
{
  *ierr = AMPI_Recv_init(buf,*count,*type,*srcpe,*tag,*comm,(AMPI_Request*)req);
}

extern "C" int AMPI_Send_init(void *buf, int count, int type, int dest, int tag,
                             AMPI_Comm comm, AMPI_Request *req)
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

extern "C" void ampi_send_init_(void *buf, int *count, int *type, int *destpe,
                               int *tag, int *comm, int *req, int *ierr)
{
  *ierr = AMPI_Send_init(buf,*count,*type,*destpe,*tag,*comm,(AMPI_Request*)req);
}

extern "C" int AMPI_Type_contiguous(int count, AMPI_Datatype oldtype, 
                                   AMPI_Datatype *newtype)
{
  ampi *ptr = CtvAccess(ampiPtr);
  *newtype = ptr->ntypes;
  ptr->types[*newtype] =  typesize(oldtype, count, ptr);
  ptr->ntypes ++;
  *newtype += 100;
  return 0;
}

extern "C" int AMPI_Type_commit(AMPI_Datatype *datatype)
{
  return 0;
}

extern "C" int AMPI_Type_free(AMPI_Datatype *datatype)
{
  return 0;
}

extern "C" void ampi_type_contiguous_(int *count, int *oldtype, int *newtype, 
                                int *ierr)
{
  *ierr = AMPI_Type_contiguous(*count, *oldtype, newtype);
}

extern "C" void ampi_type_commit_(int *type, int *ierr)
{
  *ierr = AMPI_Type_commit(type);
}

extern "C" void ampi_type_free_(int *type, int *ierr)
{
  *ierr = AMPI_Type_free(type);
}

extern "C" int AMPI_Isend(void *buf, int count, AMPI_Datatype datatype, int dest, 
              int tag, AMPI_Comm comm, AMPI_Request *request)
{
    ampi *ptr = CtvAccess(ampiPtr);
    int size = typesize(datatype, count, ptr);
    ptr->niSends++;
    ptr->biSend += size;
    ptr->ckTempoSendElem(tag, ptr->getIndex(), buf, size, dest);
    *request = (-1);
    return 0;
}

extern "C" int AMPI_Irecv(void *buf, int count, AMPI_Datatype datatype, int src, 
              int tag, AMPI_Comm comm, AMPI_Request *request)
{
  ampi *ptr = CtvAccess(ampiPtr);
  if(ptr->nirequests == 100) {
    CmiAbort("Too many Irecv requests.\n");
  }
  int size = typesize(datatype, count, ptr);
  ptr->niRecvs++;
  ptr->biRecv += size;
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

extern "C" void ampi_isend_(void *buf, int *count, int *datatype, int *dest,
                           int *tag, int *comm, int *request, int *ierr)
{
  *ierr = AMPI_Isend(buf, *count, *datatype, *dest, *tag, *comm, request);
}

extern "C" void ampi_irecv_(void *buf, int *count, int *datatype, int *src,
                           int *tag, int *comm, int *request, int *ierr)
{
  *ierr = AMPI_Irecv(buf, *count, *datatype, *src, *tag, *comm, request);
}

#define AMPI_GATHER_TAG 5000

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
  int itemsize = typesize(recvtype, 1, ptr);
  
  for(i=0;i<size;i++) {
    AMPI_Recv(((char*)recvbuf)+(itemsize*displs[i]), recvcounts[i], recvtype,
             i, AMPI_GATHER_TAG, comm, &status);
  }
  return 0;
}

extern "C"
void ampi_allgatherv_(void *sendbuf, int *sendcount, int *sendtype,
                     void *recvbuf, int *recvcounts, int *displs,
                     int *recvtype, int *comm, int *ierr)
{
  *ierr = AMPI_Allgatherv(sendbuf, *sendcount, *sendtype, recvbuf, recvcounts,
                         displs, *recvtype, *comm);
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
  int itemsize = typesize(recvtype, 1, ptr);
  
  for(i=0;i<size;i++) {
    AMPI_Recv(((char*)recvbuf)+(recvcount*itemsize*i), recvcount, recvtype,
             i, AMPI_GATHER_TAG, comm, &status);
  }
  return 0;
}

extern "C"
void ampi_allgather_(void *sendbuf, int *sendcount, int *sendtype,
                  void *recvbuf, int *recvcount, int *recvtype,
                  int *comm, int *ierr)
{
  *ierr = AMPI_Allgather(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount,
                        *recvtype, *comm);
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
    int itemsize = typesize(recvtype, 1, ptr);
  
    for(i=0;i<size;i++) {
      AMPI_Recv(((char*)recvbuf)+(itemsize*displs[i]), recvcounts[i], recvtype,
               i, AMPI_GATHER_TAG, comm, &status);
    }
  }
  return 0;
}

extern "C"
void ampi_gatherv_(void *sendbuf, int *sendcount, int *sendtype,
                  void *recvbuf, int *recvcounts, int *displs,
                  int *recvtype, int *root, int *comm, int *ierr)
{
  *ierr = AMPI_Gatherv(sendbuf, *sendcount, *sendtype, recvbuf, recvcounts,
                      displs, *recvtype, *root, *comm);
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
    int itemsize = typesize(recvtype, 1, ptr);
  
    for(i=0;i<size;i++) {
      AMPI_Recv(((char*)recvbuf)+(recvcount*itemsize*i), recvcount, recvtype,
               i, AMPI_GATHER_TAG, comm, &status);
    }
  }
  return 0;
}

extern "C"
void ampi_gather_(void *sendbuf, int *sendcount, int *sendtype,
                 void *recvbuf, int *recvcount, int *recvtype,
                 int *root, int *comm, int *ierr)
{
  *ierr = AMPI_Gather(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount, 
                     *recvtype, *root, *comm);
}

extern "C" 
int AMPI_Alltoallv(void *sendbuf, int *sendcounts, int *sdispls,
                  AMPI_Datatype sendtype, void *recvbuf, int *recvcounts,
                  int *rdispls, AMPI_Datatype recvtype, AMPI_Comm comm)
{
  ampi *ptr = CtvAccess(ampiPtr);
  int size = ptr->getSize();
  int itemsize = typesize(sendtype, 1, ptr);
  int i;
  for(i=0;i<size;i++) {
    AMPI_Send(((char*)sendbuf)+(itemsize*sdispls[i]), sendcounts[i], sendtype,
             i, AMPI_GATHER_TAG, comm);
  }

  AMPI_Status status;
  itemsize = typesize(recvtype, 1, ptr);
  
  for(i=0;i<size;i++) {
    AMPI_Recv(((char*)recvbuf)+(itemsize*rdispls[i]), recvcounts[i], recvtype,
             i, AMPI_GATHER_TAG, comm, &status);
  }
  return 0;
}

extern "C"
void ampi_alltoallv_(void *sendbuf, int *sendcounts, int *sdispls,
                    int *sendtype, void *recvbuf, int *recvcounts,
                    int *rdispls, int *recvtype, int *comm, int *ierr)
{
  *ierr = AMPI_Alltoallv(sendbuf, sendcounts, sdispls, *sendtype, recvbuf,
                        recvcounts, rdispls, *recvtype, *comm);
}

extern "C" 
int AMPI_Alltoall(void *sendbuf, int sendcount, AMPI_Datatype sendtype, 
                 void *recvbuf, int recvcount, AMPI_Datatype recvtype, 
                 AMPI_Comm comm)
{
  ampi *ptr = CtvAccess(ampiPtr);
  int size = ptr->getSize();
  int itemsize = typesize(sendtype, 1, ptr);
  int i;
  for(i=0;i<size;i++) {
    AMPI_Send(((char*)sendbuf)+(itemsize*sendcount*i), sendcount, sendtype,
             i, AMPI_GATHER_TAG, comm);
  }

  AMPI_Status status;
  itemsize = typesize(recvtype, 1, ptr);
  
  for(i=0;i<size;i++) {
    AMPI_Recv(((char*)recvbuf)+(itemsize*recvcount*i), recvcount, recvtype,
             i, AMPI_GATHER_TAG, comm, &status);
  }
  return 0;
}

extern "C"
void ampi_alltoall_(void *sendbuf, int *sendcount, int *sendtype,
                   void *recvbuf, int *recvcount, int *recvtype,
                   int *comm, int *ierr)
{
  *ierr = AMPI_Alltoall(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount,
                       *recvtype, *comm);
}

extern "C"
int AMPI_Comm_dup(int comm, int *newcomm)
{
  *newcomm = comm;
  return 0;
}

extern "C"
void ampi_comm_dup_(int *comm, int *newcomm, int *ierr)
{
  *newcomm = *comm;
  *ierr = 0;
}

extern "C"
int AMPI_Comm_free(int *comm)
{
  return 0;
}

extern "C"
void ampi_comm_free_(int *comm, int *ierr)
{
  *ierr = 0;
}

extern "C"
int AMPI_Abort(int comm, int errorcode)
{
  CmiAbort("MPI_Abort!\n");
  return errorcode;
}

extern "C"
void ampi_abort_(int *comm, int *errorcode, int *ierr)
{
  CmiAbort("MPI_Abort!\n");
  *ierr = 0;
}

#include "ampi.def.h"
