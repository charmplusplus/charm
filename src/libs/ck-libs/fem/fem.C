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
void sum(int len, d* lhs, d* rhs)
{
  int i;
  for(i=0;i<len;i++) {
    *lhs++ += *rhs++;
  }
}

static inline void
combine(int type, int len, void *lhs, void *rhs)
{
  switch(type) {
    case FEM_BYTE : sum(len,(char*)lhs, (char*)rhs); break;
    case FEM_INT : sum(len,(int*)lhs, (int*) rhs); break;
    case FEM_REAL : sum(len,(float*)lhs, (float*) rhs); break;
    case FEM_DOUBLE : sum(len,(double*)lhs, (double*) rhs); break;
  }
}

chunk::chunk(void)
{
  ntypes = 0;
  new_DT(FEM_BYTE, 1);
  new_DT(FEM_INT, 1);
  new_DT(FEM_REAL, 1);
  new_DT(FEM_DOUBLE, 1);

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
  // call the application-specific driver
  driver_(&numNodes, gNodeNums, &numElems, gElemNums, conn);
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
    int start = peStart[i];
    int num = numNodesPerPe[i];
    int len = dtypes[fid].length(num);
    DataMsg *msg = new (&len, 0) DataMsg(seqnum, thisIndex, len);
    len = dtypes[fid].length();
    void *data = msg->getData();
    void *src = nodes;
    for(j=start;j<(start+num);j++) {
      src = (void *)((char*)nodes+(j*nsize));
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
  curnodes = nodes;
  curfid = fid;
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
}

void 
FEM_Done(void)
{
  CProxy_main mainproxy(_mainhandle);
  mainproxy.done();
}

void
FEM_Set_Node_Size(int nsize)
{
  chunk *cptr = CtvAccess(_femptr);
  cptr->set_node_size(nsize);
}

int 
FEM_Create_Field(int base_type, int vec_len)
{
  chunk *cptr = CtvAccess(_femptr);
  return cptr->new_DT(base_type, vec_len);
}

void
FEM_Update_Field(int fid, void *nodes)
{
  chunk *cptr = CtvAccess(_femptr);
  cptr->update(fid, nodes);
}

// Fortran Bindings

extern "C" void
fem_set_node_size_(int *nsize)
{
  FEM_Set_Node_Size(*nsize);
}

extern "C" int
fem_create_field_(int *bt, int *vl)
{
  return FEM_Create_Field(*bt, *vl);
}

extern "C" void
fem_update_field_(int *fid, void *nodes)
{
  FEM_Update_Field(*fid, nodes);
}

// Utility functions for Fortran

extern "C" int
sizeof_(void *first, void *second)
{
  return (int)((char *)second - (char*)first);
}

#include "fem.def.h"
