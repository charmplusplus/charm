#include "fem.h"

CkChareID _mainhandle;
CkArrayID _femaid;
unsigned int _nchunks;

CtvStaticDeclare(chunk*, _femptr);

void *
DataMsg::pack(DataMsg *in)
{
  return (void*) in;
}

DataMsg *
DataMsg::unpack(void *in)
{
  return new (in) DataMsg;
}

void *
DataMsg::alloc(int mnum, size_t size, int *sizes, int pbits)
{
  return CkAllocMsg(mnum, size+sizes[0], pbits);
}

static void 
_allReduceHandler(void *, int datasize, void *data)
{
  // the reduction handler is called on processor 0
  // with available reduction results
  DataMsg *dmsg = new (&datasize, 0) DataMsg(0,datasize,0);
  memcpy(dmsg->data, data, datasize);
  CProxy_chunk carr(_femaid);
  // broadcast the reduction results to all array elements
  carr.result(dmsg);
}

main::main(CkArgMsg *am)
{
  int i;
  _nchunks = CkNumPes();
  for(i=1;i<am->argc;i++) {
    if(strncmp(am->argv[i], "+vp", 3) == 0) {
      if (strlen(am->argv[i]) > 2) {
        sscanf(am->argv[i], "+vp%d", &_nchunks);
      } else {
        if (am->argv[i+1]) {
          sscanf(am->argv[i+1], "%d", &_nchunks);
        }
      }
      break;
    }
  }
  _femaid = CProxy_chunk::ckNew(_nchunks);
  CProxy_chunk farray(_femaid);
  farray.setReductionClient(_allReduceHandler, 0);
  for(i=0;i<_nchunks;i++) {
    farray[i].run();
  }
  delete am;
  _mainhandle = thishandle;
  numdone = 0;
  // call application-specific initialization
  init_();
}

void
main::done(void)
{
  numdone++;
  if (numdone == _nchunks) {
    // call application-specific finalization
    finalize_();
    CkExit();
  }
}

template<class d>
void sum(const int len, d* lhs, d* rhs)
{
  int i;
  for(i=0;i<len;i++) {
    *lhs++ += *rhs++;
  }
}

static inline void
combine(const DType& dt, void* lhs, void* rhs)
{
  switch(dt.base_type) {
    case FEM_BYTE : 
      sum(dt.vec_len,(unsigned char*)lhs, (unsigned char*)rhs); 
      break;
    case FEM_INT : sum(dt.vec_len,(int*)lhs, (int*) rhs); break;
    case FEM_REAL : sum(dt.vec_len,(float*)lhs, (float*) rhs); break;
    case FEM_DOUBLE : sum(dt.vec_len,(double*)lhs, (double*) rhs); break;
  }
}

chunk::chunk(void)
{
  ntypes = 0;
  new_DT(FEM_BYTE);
  new_DT(FEM_INT);
  new_DT(FEM_REAL);
  new_DT(FEM_DOUBLE);

  messages = CmmNew();
  seqnum = 1;
  wait_for = 0;
  tid = 0;
}

void
chunk::run(void)
{
  CtvInitialize(chunk*, _femptr);
  CtvAccess(_femptr) = this;
  readChunk();
  // call the application-specific driver
  driver_(&numNodes, gNodeNums, &numElems, gElemNums, &numNodesPerElem, conn);
  FEM_Done();
}

void
chunk::recv(DataMsg *dm)
{
  if (dm->tag == wait_for) {
    update_field(dm); // update the appropriate field value
    delete dm;
    nRecd++;
    if(nRecd==numPes) {
      wait_for = 0; // done waiting for seqnum
      CthAwaken(tid); // awaken the waiting thread
      tid = 0;
    }
  } else {
    CmmPut(messages, 1, &(dm->tag), dm);
  }
}

void
chunk::send(int fid, void *nodes)
{
  int i, j;
  for(i=0;i<numPes;i++) {
    int dest = peNums[i];
    int num = numNodesPerPe[i];
    int len = dtypes[fid].length(num);
    DataMsg *msg = new (&len, 0) DataMsg(seqnum, thisIndex, fid);
    len = dtypes[fid].length();
    void *data = msg->data;
    void *src = (void *) ((char *)nodes + dtypes[fid].init_offset);
    for(j=0;j<num;j++) {
      src = (void *)((char*)nodes+(nodesPerPe[i][j]*dtypes[fid].distance));
      memcpy(data, src, len);
      data = (void*) ((char*)data + len);
    }
    CProxy_chunk cp(thisArrayID);
    cp[dest].recv(msg);
  }
}

void
chunk::update(int fid, void *nodes)
{
  // first send my field values to all the processors that need it
  seqnum++;
  send(fid, nodes);
  curbuf = nodes;
  nRecd = 0;
  // now, if any of the field values have been received already,
  // process them
  DataMsg *dm;
  while (dm = (DataMsg*)CmmGet(messages, 1, &seqnum, 0)) {
    update_field(dm);
    delete dm;
    nRecd++;
  }
  // if any field values are still required, put myself to sleep
  if (nRecd != numPes) {
    wait_for = seqnum;
    tid = CthSelf();
    CthSuspend();
    wait_for = 0;
    tid = 0;
  }
}

void
chunk::update_field(DataMsg *msg)
{
  void *data = msg->data;
  int from = gPeToIdx[msg->from];
  int nnodes = numNodesPerPe[from];
  int i;
  for(i=0;i<nnodes;i++) {
    int cnum = nodesPerPe[from][i];
    void *cnode = (void*) ((char*)curbuf+cnum*dtypes[msg->dtype].distance);
    combine(dtypes[msg->dtype], cnode, data);
    data = (void *)((char*)data+(dtypes[msg->dtype].length()));
  }
}

void
chunk::reduce_field(int fid, void *nodes, void *outbuf)
{
  // first reduce over local nodes
  DType *dt = &dtypes[fid];
  void *src = (void *) ((char *) nodes + dt->init_offset);
  for(int i=0; i<numNodes; i++) {
    if(isPrimary[i]) {
      combine(*dt, outbuf, src);
    }
    src = (void *)((char *)src + dt->distance);
  }
  // and now reduce over partitions
  reduce(fid, outbuf, outbuf);
}

void
chunk::reduce(int fid, void *inbuf, void *outbuf)
{
  int len = dtypes[fid].length();
  CkReduction::reducerType rtype;
  switch(dtypes[fid].base_type) {
    case FEM_INT: rtype = CkReduction::sum_int; break;
    case FEM_REAL: rtype = CkReduction::sum_float; break;
    case FEM_DOUBLE: rtype = CkReduction::sum_double; break;
  }
  contribute(len, inbuf, rtype);
  curbuf = outbuf;
  tid = CthSelf();
  CthSuspend();
}

void
chunk::result(DataMsg *msg)
{
  //msg->from used as length
  memcpy(curbuf, msg->data, msg->from);
  CthAwaken(tid);
  tid = 0;
  delete msg;
}

void
chunk::readNodes(FILE* fp)
{
  fscanf(fp, "%d", &numNodes);
  gNodeNums = new int[numNodes];
  isPrimary = new int[numNodes];
  for(int i=0;i<numNodes;i++) {
    fscanf(fp, "%d%d", &gNodeNums[i], &isPrimary[i]);
    isPrimary[i] = ((isPrimary[i]==thisIndex) ? 1 : 0);
  }
}

void
chunk::readElems(FILE* fp)
{
  fscanf(fp, "%d%d", &numElems, &numNodesPerElem);
  gElemNums = new int[numElems];
  conn = new int[numElems*numNodesPerElem];
  for(int i=0; i<numElems; i++) {
    fscanf(fp, "%d", &gElemNums[i]);
    for(int j=0;j<numNodesPerElem;j++) {
      fscanf(fp, "%d", &conn[i*numNodesPerElem+j]);
    }
  }
}

void
chunk::readComm(FILE* fp)
{
  gPeToIdx = new int[numElements];
  for(int p=0;p<numElements;p++) {
    gPeToIdx[p] = (-1);
  }
  fscanf(fp, "%d", &numPes);
  peNums = new int[numPes];
  numNodesPerPe = new int[numPes];
  nodesPerPe = new int*[numPes];
  for(int i=0;i<numPes;i++) {
    fscanf(fp, "%d%d", &peNums[i], &numNodesPerPe[i]);
    gPeToIdx[peNums[i]] = i;
    nodesPerPe[i] = new int[numNodesPerPe[i]];
    for(int j=0;j<numNodesPerPe[i];j++) {
      fscanf(fp, "%d", &nodesPerPe[i][j]);
    }
  }
}

void
chunk::readChunk(void)
{
  char fname[32];
  sprintf(fname, "meshdata.Pe%d", thisIndex);
  FILE *fp = fopen(fname, "r");
  if(fp==0) {
    CkAbort("FEM: unable to open input file.\n");
  }
  readNodes(fp);
  readElems(fp);
  readComm(fp);
  fclose(fp);
}

void 
FEM_Done(void)
{
  CProxy_main mainproxy(_mainhandle);
  mainproxy.done();
}

int 
FEM_Create_Field(int base_type, int vec_len, int init_offset, int distance)
{
  chunk *cptr = CtvAccess(_femptr);
  return cptr->new_DT(base_type, vec_len, init_offset, distance);
}

void
FEM_Update_Field(int fid, void *nodes)
{
  chunk *cptr = CtvAccess(_femptr);
  cptr->update(fid, nodes);
}

void
FEM_Reduce_Field(int fid, void *nodes, void *outbuf)
{
  chunk *cptr = CtvAccess(_femptr);
  cptr->reduce_field(fid, nodes, outbuf);
}

void
FEM_Reduce(int fid, void *inbuf, void *outbuf)
{
  chunk *cptr = CtvAccess(_femptr);
  cptr->reduce(fid, inbuf, outbuf);
}

int
FEM_Id(void)
{
  chunk *cptr = CtvAccess(_femptr);
  return cptr->id();
}

// Fortran Bindings

extern "C" int
fem_create_field_(int *bt, int *vl, int *io, int *d)
{
  return FEM_Create_Field(*bt, *vl, *io, *d);
}

extern "C" void
fem_update_field_(int *fid, void *nodes)
{
  FEM_Update_Field(*fid, nodes);
}

extern "C" void
fem_reduce_field_(int *fid, void *nodes, void *outbuf)
{
  FEM_Reduce_Field(*fid, nodes, outbuf);
}

extern "C" void
fem_reduce_(int *fid, void *inbuf, void *outbuf)
{
  FEM_Reduce(*fid, inbuf, outbuf);
}

extern "C" int
fem_id_(void)
{
  return FEM_Id();
}

// Utility functions for Fortran

extern "C" int
offsetof_(void *first, void *second)
{
  return (int)((char *)second - (char*)first);
}

#include "fem.def.h"
