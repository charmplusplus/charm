/*
Multiblock CFD Framework for Charm++
Parallel Programming Lab, Univ. of Illinois 2000

This file implements a C, C++, or Fortran-callable
library for parallel multiblock CFD computations.

We partition the user's grid using the makemblock programs.
We run the user's (timeloop) driver routine in a 
thread (in the style of AMPI), so communication looks
blocking to the user.  Internally, we suspend the
user's driver thread when communication is needed,
then resume the thread when the results arrive.
 */
#include "mblock_impl.h"
#include <limits.h>
#include <float.h> /*for FLT_MIN on non-Suns*/

CkChareID _mainhandle;
CkArrayID _mblockaid;
int _nchunks;

//_mblkptr gives the current chunk, and is only
// valid in routines called from driver().
CtvStaticDeclare(chunk*, _mblkptr);

//This enum defines possible places the MBLK_* calls
// can come from.  This is needed because, e.g., a
// MBLK_Print call can prepend the chunk# if called from
// driver; but not otherwise.
typedef enum {
  inInit,
  inPup,
  inBc,
  inDriver, //<- almost always here
} mblk_state_t;

CpvStaticDeclare(mblk_state_t,_mblk_state);

static void 
_allReduceHandler(void *, int datasize, void *data)
{
  // the reduction handler is called on processor 0
  // with available reduction results
  DataMsg *dmsg = new (&datasize, 0) DataMsg(0,datasize,0,0); CHK(dmsg);
  memcpy(dmsg->data, data, datasize);
  CProxy_chunk carr(_mblockaid);
  // broadcast the reduction results to all array elements
  carr.reductionResult(dmsg);
}

extern void CreateMetisLB(void);

static main *_mainptr = 0;

main::main(CkArgMsg *am)
{
  int i;
  delete am;
  CreateMetisLB();
  CpvInitialize(mblk_state_t,_mblk_state);
  CpvAccess(_mblk_state)=inInit;
  _mainptr = this;

#if MBLK_FORTRAN
  FTN_NAME(INIT,init_) ();
#else // C/C++
  init();
#endif // Fortran

  _mblockaid = CProxy_chunk::ckNew(nblocks);
  CProxy_chunk farray(_mblockaid);
  farray.setReductionClient(_allReduceHandler, 0);
  _mainhandle = thishandle;
  numdone = 0;
  for(i=0;i<nblocks;i++) {
    farray[i].run(new ChunkMsg((const char *)prefix, ndims));
  }
}

void
main::done(void)
{
  numdone++;
  if (numdone == _nchunks) {
    CkExit();
  }
}


/******************** Reduction Support **********************/

template<class d>
void sum(const int len, d* lhs, const d* rhs)
{
  int i;
  for(i=0;i<len;i++) {
    *lhs++ += *rhs++;
  }
}

#ifdef __CYGWIN__
#undef max 
#undef min
#endif

template<class d>
void max(const int len, d* lhs, const d* rhs)
{
  int i;
  for(i=0;i<len;i++) {
    *lhs = (*lhs > *rhs) ? *lhs : *rhs;
    lhs++; rhs++;
  }
}

template<class d>
void min(const int len, d* lhs, const d* rhs)
{
  int i;
  for(i=0;i<len;i++) {
    *lhs = (*lhs < *rhs) ? *lhs : *rhs;
    lhs++; rhs++;
  }
}

template<class d>
void assign(const int len, d* lhs, d val)
{
  int i;
  for(i=0;i<len;i++) {
    *lhs = val;
  }
}

static inline void
initialize(const DType& dt, void *lhs, int op)
{
  switch(op) {
    case MBLK_SUM:
      switch(dt.base_type) {
        case MBLK_BYTE : 
          assign(dt.vec_len,(unsigned char*)lhs, (unsigned char)0); 
          break;
        case MBLK_INT : assign(dt.vec_len,(int*)lhs, 0); break;
        case MBLK_REAL : assign(dt.vec_len,(float*)lhs, (float)0.0); break;
        case MBLK_DOUBLE : assign(dt.vec_len,(double*)lhs, 0.0); break;
      }
      break;
    case MBLK_MAX:
      switch(dt.base_type) {
        case MBLK_BYTE : 
          assign(dt.vec_len,(unsigned char*)lhs, (unsigned char)CHAR_MIN); 
          break;
        case MBLK_INT : assign(dt.vec_len,(int*)lhs, INT_MIN); break;
        case MBLK_REAL : assign(dt.vec_len,(float*)lhs, FLT_MIN); break;
        case MBLK_DOUBLE : assign(dt.vec_len,(double*)lhs, DBL_MIN); break;
      }
      break;
    case MBLK_MIN:
      switch(dt.base_type) {
        case MBLK_BYTE : 
          assign(dt.vec_len,(unsigned char*)lhs, (unsigned char)CHAR_MAX); 
          break;
        case MBLK_INT : assign(dt.vec_len,(int*)lhs, INT_MAX); break;
        case MBLK_REAL : assign(dt.vec_len,(float*)lhs, FLT_MAX); break;
        case MBLK_DOUBLE : assign(dt.vec_len,(double*)lhs, DBL_MAX); break;
      }
      break;
  }
}

typedef void (*combineFn)(const int len,void *lhs,const void *rhs);

typedef void (*combineFn_BYTE)(const int len,unsigned char *lhs,const unsigned char *rhs);
typedef void (*combineFn_INT)(const int len,int *lhs,const int *rhs);
typedef void (*combineFn_REAL)(const int len,float *lhs,const float *rhs);
typedef void (*combineFn_DOUBLE)(const int len,double *lhs,const double *rhs);


static combineFn
combine(const DType& dt, int op)
{
  switch(op) {
//This odd-looking define selects the appropriate templated type
    // of "fn", casts it to a void* type, and returns it.
#define combine_switch(fn) \
      switch(dt.base_type) {\
        case MBLK_BYTE : return (combineFn)(combineFn_BYTE)fn;\
        case MBLK_INT : return (combineFn)(combineFn_INT)fn;\
        case MBLK_REAL : return (combineFn)(combineFn_REAL)fn;\
        case MBLK_DOUBLE : return (combineFn)(combineFn_DOUBLE)fn;\
      }\
      break;
    case MBLK_SUM: combine_switch(sum);
    case MBLK_MIN: combine_switch(min);
    case MBLK_MAX: combine_switch(max);
  }
  return NULL;
}

/******************************* CHUNK *********************************/

chunk::chunk(void)
{
  usesAtSync = CmiTrue;
  ntypes = 0;
  new_DT(MBLK_BYTE);
  new_DT(MBLK_INT);
  new_DT(MBLK_REAL);
  new_DT(MBLK_DOUBLE);

  CpvInitialize(mblk_state_t,_mblk_state);
  CpvAccess(_mblk_state)=inDriver;
  
  messages = CmmNew();
  tid = 0;
  nudata = 0;
  seqnum = 0;
  wait_seqnum = -1;
  nRecvPatches = 0;
  nRecd = 0;
}

chunk::~chunk() //Destructor-- deallocate memory
{
	CmmFree(messages);
	if (tid!=0)
	  CthFree(tid);
}

int
chunk::check_userdata(int n)
{
  if (n<0 || n>=nudata)
    CkAbort("Invalid userdata ID passed to MBLK_[SG]et_Userdata\n");
  return n;
}

void
chunk::callDriver(void)
{
  // call the application-specific driver
#if MBLK_FORTRAN
  FTN_NAME(DRIVER,driver_) ();
#else // C/C++
  driver();
#endif // Fortran
  CProxy_main mainproxy(_mainhandle);
  mainproxy.done();
}

void
chunk::run(ChunkMsg *msg)
{
  start_running();
  CtvInitialize(chunk*, _mblkptr);
  CtvAccess(_mblkptr) = this;
  b = new block(msg->prefix, thisIndex);
  int n = b->getPatches();
  nRecvPatches = 0;
  for(int i=0;i<n;i++) {
    patch *p = b->getPatch(i);
    if(p->type == patch::recv)
      nRecvPatches++;
  }
  callDriver();
  // Note: "this may have changed after we come back here 
  CtvAccess(_mblkptr)->stop_running();
}

void
chunk::recv(DataMsg *dm)
{
  if(dm->seqnum = wait_seqnum) {
    update_field(dm); // update the appropriate field value
    delete dm;
    nRecd++;
    if(nRecd==nRecvPatches) {
      wait_seqnum = -1; // done waiting for seqnum
      if(tid)
        thread_resume();
    }
  } else {
    CmmPut(messages, 1, &(dm->seqnum), dm);
  }
}

//Send the values for my source patches out
void
chunk::send(int fid, const void *grid)
{
  const DType &dt=dtypes[fid];
  int len = dt.length();
  const char *fStart=(const char *)grid;
  fStart+=dt.init_offset;
  blockDim d = b->getDim();
  int n = b->getPatches();
  for(int m=0;m<n;m++) {
    patch *p = b->getPatch(m);
    if (p->type != patch::send)
      continue;
    internalBCpatch *ip = (internalBCpatch*) p;
    int dest = ip->dest;
    int num = (ip->end[0]-ip->start[0]) *
      (ip->end[1]-ip->start[1]) *
      (ip->end[2]-ip->start[2]);
    int msgLen=len*num;
    DataMsg *msg = new (&msgLen, 0) DataMsg(seqnum, thisIndex, fid, 
        ip->destPatch); 
    CHK(msg);
    void *data = msg->data;
    const void *src;
    for(int i=ip->start[0];i<ip->end[0];i++)
      for(int j=ip->start[1];j<ip->end[1];i++)
        for(int k=ip->start[2];k<ip->end[2];i++) {
          src = (const void *)(fStart+d.c_index(i,j,k)*dt.distance);
          memcpy(data, src, len);
          data = (void*) ((char*)data + len);
        }
    CProxy_chunk cp(thisArrayID);
    cp[dest].recv(msg);
  }
  seqnum++;
}


//Update my ghost cells based on these values
void
chunk::update_field(DataMsg *msg)
{
  const DType &dt=dtypes[msg->dtype];
  int length=dt.length();
  char *fStart=(char *)ogrid;
  fStart+=dt.init_offset;
  char *data = (char *) msg->data;
  int incr[3];
  internalBCpatch *p = (internalBCpatch*) (b->getPatch(msg->patchno));
  blockLoc start = p->start; blockLoc end = p->end;
  int d;
  int dmap[3];
  for(d=0; d<3; d++)
    dmap[d] = p->orient.getMap(d);
  for(d=0; d<3; d++) {
    if(p->orient.isFlipped(d)) {
      end[d] = p->start[dmap[d]]-1;
      start[d] = p->end[dmap[d]]-1;
      incr[d] = -1;
    } else {
      start[d] = p->start[dmap[d]];
      end[d] = p->end[dmap[d]];
      incr[d] = 1;
    }
  }
  blockDim dim = b->getDim();
  for(int i=start[0]; i!=end[0]; i+= incr[0])
    for(int j=start[1]; j!=end[1]; j+= incr[1])
      for(int k=start[2]; k!=end[2]; k+= incr[2])
      {
        void *dest = (void *)(fStart+dim.c_index(i,j,k)*dt.distance);
        memcpy(dest, data, length);
        data = data + length;
      }
}


void
chunk::update(int fid, void *ingrid, void *outgrid)
{
  // first send my field values to all the processors that need it
  send(fid, ingrid);
  ogrid = outgrid;
  nRecd = 0;
  // now, if any of the field values have been received already,
  // process them
  DataMsg *dm;
  while ((dm = (DataMsg*)CmmGet(messages, 1, &seqnum, 0))!=NULL) {
    update_field(dm);
    delete dm;
    nRecd++;
  }
  // if any field values are still required, note that
  if (nRecd != nRecvPatches)
    wait_seqnum = seqnum;
}

int
chunk::test_update(void)
{
  return (wait_seqnum==(-1))?MBLK_DONE : MBLK_NOTDONE;
}

int
chunk::wait_update(void)
{
  if(wait_seqnum != (-1))
    thread_suspend();
  return MBLK_SUCCESS;
}

void
chunk::reduce_field(int fid, const void *grid, void *outbuf, int op)
{
  // first reduce over local gridpoints
  const DType &dt = dtypes[fid];
  const void *src = (const void *) ((const char *) grid + dt.init_offset);
  initialize(dt,outbuf,op);
  combineFn fn=combine(dt,op);
  int start[3], end[3];
  MBLK_Get_extent(start, end);
  blockDim d = b->getDim();
  for(int i=start[0]; i<end[0]; i++)
    for (int j=start[1]; j<end[1]; j++)
      for(int k=start[2]; k<end[2]; k++)
        fn(dt.vec_len, outbuf, 
            ((const char *)src + d.c_index(i,j,k)*dt.distance));
  // and now reduce over partitions
  reduce(fid, outbuf, outbuf, op);
}

void
chunk::reduce(int fid, const void *inbuf, void *outbuf, int op)
{
  int len = dtypes[fid].length();
  if(numElements==1) {
    memcpy(outbuf,inbuf,len);
    return;
  }
  CkReduction::reducerType rtype;
  switch(op) {
    case MBLK_SUM:
      switch(dtypes[fid].base_type) {
        case MBLK_INT: rtype = CkReduction::sum_int; break;
        case MBLK_REAL: rtype = CkReduction::sum_float; break;
        case MBLK_DOUBLE: rtype = CkReduction::sum_double; break;
      }
      break;
    case MBLK_MAX:
      switch(dtypes[fid].base_type) {
        case MBLK_INT: rtype = CkReduction::max_int; break;
        case MBLK_REAL: rtype = CkReduction::max_float; break;
        case MBLK_DOUBLE: rtype = CkReduction::max_double; break;
      }
      break;
    case MBLK_MIN:
      switch(dtypes[fid].base_type) {
        case MBLK_INT: rtype = CkReduction::min_int; break;
        case MBLK_REAL: rtype = CkReduction::min_float; break;
        case MBLK_DOUBLE: rtype = CkReduction::min_double; break;
      }
      break;
  }
  contribute(len, (void *)inbuf, rtype);
  ogrid = outbuf;
  thread_suspend();
}

void
chunk::reductionResult(DataMsg *msg)
{
  //msg->from used as length
  memcpy(ogrid, msg->data, msg->from);
  thread_resume();
  delete msg;
}

/********** Thread/migration support ************/
void chunk::thread_suspend(void)
{
  if (tid!=0) 
    CkAbort("chunk::thread_suspend: Tried to suspend; but already waiting!\n");
  tid = CthSelf();
  stop_running();
  CthSuspend();
  /*We have to do the CtvAccess because "this" may have changed
    during a migration-suspend.*/
  CtvAccess(_mblkptr)->start_running();
}
void chunk::thread_resume(void)
{
  if (tid==0) 
    CkAbort("chunk::thread_resume: Tried to resume; but no waiting thread!\n");
  CthAwaken(tid);
  tid = 0;
}

static chunk *getCurChunk(void) 
{
  chunk *cptr=CtvAccess(_mblkptr);
  if (cptr==NULL) 
    CkAbort("Routine can only be called from driver()!\n");
  return cptr;
}


/******************************* C Bindings **********************************/

CDECL int 
MBLK_Set_prefix(const char *prefix)
{
  if(CpvAccess(_mblk_state) != inInit) {
    CkError("MBLK_Set_prefix called from outside init\n");
    return MBLK_FAILURE;
  }
  strcpy(_mainptr->prefix,prefix);
  return MBLK_SUCCESS;
}

CDECL int 
MBLK_Set_nblocks(const int n)
{
  if(CpvAccess(_mblk_state) != inInit) {
    CkError("MBLK_Set_nblocks called from outside init\n");
    return MBLK_FAILURE;
  }
  _mainptr->nblocks = n;
  return MBLK_SUCCESS;
}

CDECL int 
MBLK_Set_dim(const int n)
{
  if(CpvAccess(_mblk_state) != inInit) {
    CkError("MBLK_Set_dim called from outside init\n");
    return MBLK_FAILURE;
  }
  _mainptr->nblocks = n;
  return MBLK_SUCCESS;
}

CDECL int 
MBLK_Get_nblocks(int *n)
{
  if(CpvAccess(_mblk_state) != inDriver) {
    CkError("MBLK_Get_nblocks  called from outside driver\n");
    return MBLK_FAILURE;
  }
  chunk *cptr = getCurChunk();
  *n = cptr->total();
  return MBLK_SUCCESS;
}

CDECL int 
MBLK_Get_myblock(int *m)
{
  if(CpvAccess(_mblk_state) != inDriver) {
    CkError("MBLK_Get_myblock  called from outside driver\n");
    return MBLK_FAILURE;
  }
  chunk *cptr = getCurChunk();
  *m = cptr->id();
  return MBLK_SUCCESS;
}

CDECL int 
MBLK_Get_blocksize(int *start, int *end)
{
  if(CpvAccess(_mblk_state) != inDriver) {
    CkError("MBLK_Get_blocksize  called from outside driver\n");
    return MBLK_FAILURE;
  }
  chunk *cptr = getCurChunk();
  blockDim d = cptr->b->getDim();
  start[0] = start[1] = start[2] = 0;
  end[0] = d[0]; end[1] = d[1]; end[2] = d[2];
  return MBLK_SUCCESS;
}

// very nasty kludge. makemblock should tell me the extents specifically
CDECL int 
MBLK_Get_extent(int *start, int *end)
{
  if(CpvAccess(_mblk_state) != inDriver) {
    CkError("MBLK_Get_extent  called from outside driver\n");
    return MBLK_FAILURE;
  }
  chunk *cptr = getCurChunk();
  blockDim d = cptr->b->getDim();
  start[0] = start[1] = start[2] = 0;
  end[0] = d[0]; end[1] = d[1]; end[2] = d[2];
  int n = cptr->b->getPatches();
  for(int i=0; i<n; i++)
  {
    patch *p = cptr->b->getPatch(i);
    if(p->type==patch::recv || p->type==patch::ext)
    {
      blockLoc w;
      int dir = p->getWidth(w);
      if(p->getPos(dir) == 1)
        start[dir] += w[dir];
      else
        end[dir] -= w[dir];
    }
  }
  return MBLK_SUCCESS;
}

CDECL double 
MBLK_Timer(void)
{
  return CkTimer();
}

CDECL void 
MBLK_Print(const char *str)
{
  if(CpvAccess(_mblk_state)==inDriver) {
    chunk *cptr = getCurChunk();
    CkPrintf("[%d] %s\n", cptr->thisIndex, str);
  } else {
    CkPrintf("%s\n", str);
  }
}

CDECL int 
MBLK_Register(void *_ud,MBLK_PupFn _pup_ud, int *rid)
{
  if(CpvAccess(_mblk_state) != inDriver) {
    CkError("MBLK_Register  called from outside driver\n");
    return MBLK_FAILURE;
  }
  *rid = getCurChunk()->register_userdata(_ud,_pup_ud);
  return MBLK_SUCCESS;
}

CDECL int 
MBLK_Get_registered(int n, void **b)
{
  if(CpvAccess(_mblk_state) != inDriver) {
    CkError("MBLK_Get_registered  called from outside driver\n");
    return MBLK_FAILURE;
  }
  *b = getCurChunk()->get_userdata(n);
  return MBLK_SUCCESS;
}

CDECL int 
MBLK_Migrate(void)
{
  if(CpvAccess(_mblk_state) != inDriver) {
    CkError("MBLK_Migrate  called from outside driver\n");
    return MBLK_FAILURE;
  }
  getCurChunk()->readyToMigrate();
  return MBLK_SUCCESS;
}

CDECL int 
MBLK_Create_Field(int base_type, int vec_len, int init_offset, int distance,
    int *fid)
{
  if(CpvAccess(_mblk_state) != inDriver) {
    CkError("MBLK_Create_field  called from outside driver\n");
    return MBLK_FAILURE;
  }
  *fid = getCurChunk()->new_DT(base_type, vec_len, init_offset, distance);
  return ((*fid==(-1)) ? MBLK_FAILURE : MBLK_SUCCESS);
}

CDECL int
MBLK_Update_Field(int fid, void *ingrid, void *outgrid)
{
  if(CpvAccess(_mblk_state) != inDriver) {
    CkError("MBLK_Update_field  called from outside driver\n");
    return MBLK_FAILURE;
  }
  MBLK_Iupdate_field(fid, ingrid, outgrid);
  return MBLK_Wait_update();
}

CDECL int
MBLK_Iupdate_Field(int fid, void *ingrid, void *outgrid)
{
  if(CpvAccess(_mblk_state) != inDriver) {
    CkError("MBLK_Iupdate_field  called from outside driver\n");
    return MBLK_FAILURE;
  }
}

CDECL int
MBLK_Test_update(int *status)
{
  if(CpvAccess(_mblk_state) != inDriver) {
    CkError("MBLK_Test_update  called from outside driver\n");
    return MBLK_FAILURE;
  }
}

CDECL int
MBLK_Wait_update(void)
{
  if(CpvAccess(_mblk_state) != inDriver) {
    CkError("MBLK_Wait_update  called from outside driver\n");
    return MBLK_FAILURE;
  }
}

CDECL int
MBLK_Reduce_Field(int fid, void *ingrid, void *outbuf, int op)
{
  if(CpvAccess(_mblk_state) != inDriver) {
    CkError("MBLK_Reduce_field  called from outside driver\n");
    return MBLK_FAILURE;
  }
  getCurChunk()->reduce_field(fid, ingrid, outbuf, op);
  return MBLK_SUCCESS;
}

CDECL int
MBLK_Reduce(int fid, void *inbuf, void *outbuf, int op)
{
  if(CpvAccess(_mblk_state) != inDriver) {
    CkError("MBLK_Reduce  called from outside driver\n");
    return MBLK_FAILURE;
  }
  getCurChunk()->reduce(fid, inbuf, outbuf, op);
  return MBLK_SUCCESS;
}

CDECL int
MBLK_Register_bc(const int bcnum, const MBLK_BcFn bcfn)
{
  if(CpvAccess(_mblk_state) != inDriver) {
    CkError("MBLK_Register_bc  called from outside driver\n");
    return MBLK_FAILURE;
  }
  return getCurChunk()->register_bc(bcnum, bcfn);
}

CDECL int
MBLK_Apply_bc(const int bcnum, void *grid)
{
  if(CpvAccess(_mblk_state) != inDriver) {
    CkError("MBLK_Apply_bc  called from outside driver\n");
    return MBLK_FAILURE;
  }
  CpvAccess(_mblk_state) = inBc;
  int retval =  getCurChunk()->apply_bc(bcnum, grid);
  CpvAccess(_mblk_state) = inDriver;
  return retval;
}

CDECL int
MBLK_Apply_bc_all(void *grid)
{
  if(CpvAccess(_mblk_state) != inDriver) {
    CkError("MBLK_Apply_bc_all  called from outside driver\n");
    return MBLK_FAILURE;
  }
  CpvAccess(_mblk_state) = inBc;
  int retval =  getCurChunk()->apply_bc_all(grid);
  CpvAccess(_mblk_state) = inDriver;
  return retval;
}

CDECL int
MBLK_Get_boundary_extent(int *start, int *end)
{
  if(CpvAccess(_mblk_state) != inBc) {
    CkError("MBLK_Get_boundary_extent  called from outside BC handler\n");
    return MBLK_FAILURE;
  }
  return getCurChunk()->get_boundary_extent(start, end);
}

CDECL void 
MBLK_Print_block(void)
{
  if(CpvAccess(_mblk_state)==inDriver) {
    chunk *cptr = getCurChunk();
    cptr->print();
  }
}

#if MBLK_FORTRAN
/************************ Fortran Bindings *********************************/

// Utility functions for Fortran

FDECL int FTN_NAME(OFFSETOF,offsetof_)
  (void *first, void *second)
{
  return (int)((char *)second - (char*)first);
}

FDECL void FTN_NAME(MBLK_SET_PREFIX, mblk_set_prefix_)
  (const char *str, int *ret, int len)
{
  char *tmpstr = new char[len+1]; CHK(tmpstr);
  memcpy(tmpstr,str,len);
  tmpstr[len] = '\0';
  *ret = MBLK_Set_prefix(tmpstr);
  delete[] tmpstr;
}

FDECL void FTN_NAME(MBLK_SET_NBLOCKS, mblk_set_nblocks_)
  (int *nblocks, int *ret)
{
  *ret = MBLK_Set_nblocks(*nblocks);
}

FDECL void FTN_NAME(MBLK_SET_DIM, mblk_set_dim_)
  (int *dim, int *ret)
{
  *ret = MBLK_Set_dim(*dim);
}

FDECL void FTN_NAME(MBLK_GET_NBLOCKS, mblk_get_nblocks_)
  (int *n, int *ret)
{
  *ret = MBLK_Get_nblocks(n);
}

FDECL void FTN_NAME(MBLK_GET_MYBLOCK, mblk_get_myblock_)
  (int *m, int *ret)
{
  *ret = MBLK_Get_myblock(m);
}

FDECL void FTN_NAME(MBLK_GET_BLOCKSIZE, mblk_get_blocksize_)
  (int *start, int *end, int *ret)
{
  *ret = MBLK_Get_blocksize(start, end);
}

FDECL void FTN_NAME(MBLK_GET_EXTENT, mblk_get_extent_)
  (int *start, int *end, int *ret)
{
  *ret = MBLK_Get_extent(start, end);
}

FDECL double FTN_NAME(MBLK_TIMER, mblk_timer_)
  (void)
{
  return MBLK_Timer();
}

FDECL void FTN_NAME(MBLK_PRINT,mblk_print_)
  (char *str, int len)
{
  char *tmpstr = new char[len+1]; CHK(tmpstr);
  memcpy(tmpstr,str,len);
  tmpstr[len] = '\0';
  MBLK_Print(tmpstr);
  delete[] tmpstr;
}

FDECL void FTN_NAME(MBLK_PRINT_BLOCK,mblk_print_block_)
  (void)
{
  MBLK_Print_block();
}

FDECL void FTN_NAME(MBLK_CREATE_FIELD, mblk_create_field_)
  (int *b, int *l, int *o, int *d, int *f, int *ret)
{
  *ret = MBLK_Create_field(*b, *l, *o, *d, f);
}

FDECL void FTN_NAME(MBLK_UPDATE_FIELD, mblk_update_field_)
  (int *fid, void *igrid, void *ogrid, int *ret)
{
  *ret = MBLK_Update_field(*fid, igrid, ogrid);
}

FDECL void FTN_NAME(MBLK_IUPDATE_FIELD, mblk_iupdate_field_)
  (int *fid, void *igrid, void *ogrid, int *ret)
{
  *ret = MBLK_Iupdate_field(*fid, igrid, ogrid);
}

FDECL void FTN_NAME(MBLK_TEST_UPDATE, mblk_test_update_)
  (int *status, int *ret)
{
  *ret = MBLK_Test_update(status);
}

FDECL void FTN_NAME(MBLK_WAIT_UPDATE, mblk_wait_update_)
  (int *ret)
{
  *ret = MBLK_Wait_update();
}

FDECL void FTN_NAME(MBLK_REDUCE_FIELD, mblk_reduce_field_)
  (int *fid, void *grid, void *out, int *op, int *ret)
{
  *ret = MBLK_Reduce_field(*fid, grid, out, *op);
}

FDECL void FTN_NAME(MBLK_REDUCE, mblk_reduce_)
  (int *fid, void *in, void *out, int *op, int *ret)
{
  *ret = MBLK_Reduce_field(*fid, in, out, *op);
}

FDECL void FTN_NAME(MBLK_REGISTER_BC, mblk_register_bc_)
  (int *bcnum, MBLK_BcFn bcfn, int *ret)
{
  *ret = MBLK_Register_bc(*bcnum, bcfn);
}

FDECL void FTN_NAME(MBLK_APPLY_BC, mblk_apply_bc_)
  (int *bcnum, void *grid, int *ret)
{
  *ret = MBLK_Apply_bc(*bcnum, grid);
}

FDECL void FTN_NAME(MBLK_APPLY_BC_ALL, mblk_apply_bc_all_)
  (void *grid, int *ret)
{
  *ret = MBLK_Apply_bc_all(grid);
}

FDECL void FTN_NAME(MBLK_GET_BOUNDARY_EXTENT, mblk_get_boundary_extent_)
  (int *start, int *end, int *ret)
{
  *ret = MBLK_Get_boundary_extent(start, end);
}

FDECL void FTN_NAME(MBLK_REGISTER, mblk_register_)
  (void *block, MBLK_PupFn pupfn, int *rid, int *ret)
{
  *ret = MBLK_Register(block, pupfn, rid);
}

FDECL void FTN_NAME(MBLK_MIGRATE, mblk_migrate_)
  (int *ret)
{
  *ret = MBLK_Migrate();
}

#endif /*MBLK_FORTRAN*/


void
chunk::print(void)
{
  int i;
  CkPrintf("-------------------- Block %d --------------------\n",thisIndex);
  b->print();
  CkPrintf("--------------------------------------------------\n",thisIndex);
}  
  

void
chunk::pup(PUP::er &p)
{
//Pup superclass
  ArrayElement1D::pup(p);

  if(p.isDeleting())
  { //Resend saved messages to myself
    DataMsg *dm;
    int snum = CmmWildCard;
    CProxy_chunk cp(thisArrayID);
    while ((dm = (DataMsg*) CmmGet(messages, 1, &snum, 0))!=NULL)
      cp[thisIndex].recv(dm);
  }

  //This seekBlock allows us to reorder the packing/unpacking--
  // This is needed because the userData depends on the thread's stack
  // both at pack and unpack time.
  PUP::seekBlock s(p,2);
  if (p.isUnpacking()) 
  {//In this case, unpack the thread before the user data
    s.seek(1);
    tid = CthPup((pup_er) &p, tid);
  }
  
  //Pack all user data
  CpvAccess(_mblk_state)=inPup;
  s.seek(0);
  p(nudata);
  for(int i=0;i<nudata;i++) {
    //Save userdata for Fortran-- stack allocated
    p((void*)&(userdata[i]), sizeof(void *));
    //FIXME: function pointers may not be valid across processors
    p((void*)&(pup_ud[i]), sizeof(MBLK_PupFn));
#if MBLK_FORTRAN
    pup_ud[i]((pup_er) &p, userdata[i]);
#else
    userdata[i] = pup_ud[i]((pup_er) &p, userdata[i]);
#endif
  }
  CpvAccess(_mblk_state)=inDriver;

  if (!p.isUnpacking()) 
  {//In this case, pack the thread after the user data
    s.seek(1);
    tid = CthPup((pup_er) &p, tid);
  }
  s.endBlock(); //End of seeking block
  

  if(p.isUnpacking())
  {
    messages = CmmNew();
    b = new block();
    CtvAccessOther(tid,_mblkptr) = this;
  }
  b->pup(p);
//Pup all other fields
  p(ntypes);
  p((void*)dtypes, MBLK_MAXDT*sizeof(DType));
  p((void*)bcfns, MBLK_MAXBC*sizeof(MBLK_BcFn));
  p(seqnum);
  p(wait_seqnum);
  p(nRecd);
  p(nRecvPatches);
}


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


#include "mblock.def.h"
