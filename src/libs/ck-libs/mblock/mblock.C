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


#if 0
/*Many debugging statements:*/
#define DBG(x) ckout<<"["<<thisIndex<<"] MBLK> "<<x<<endl;

#else
/*No debugging statements*/
#define DBG(x) /*empty*/
#endif

//_mblkptr gives the current chunk, and is only
// valid in routines called from driver().
CtvStaticDeclare(MBlockChunk *, _mblkptr);

static void 
_allReduceHandler(void *proxy_v, int datasize, void *data)
{
  CProxy_MBlockChunk *proxy=(CProxy_MBlockChunk *)proxy_v;
  // the reduction handler is called on processor 0
  // with available reduction results
  MBlockDataMsg *dmsg = new (&datasize, 0) MBlockDataMsg(0,datasize,0,0); CHK(dmsg);
  memcpy(dmsg->data, data, datasize);

  // broadcast the reduction results to all array elements
  proxy->reductionResult(dmsg);
}


CDECL void init(void);
FDECL void FTN_NAME(INIT,init)(void);
CDECL void driver(void);
FDECL void FTN_NAME(DRIVER,driver)(void);

static void callDrivers(void) {
	driver();
	FTN_NAME(DRIVER,driver)();
}

//Startup routine to use if the user doesn't register one
static void MBlockFallbackSetup(void)
{
	TCharmCreate(TCharmGetNumChunks(),callDrivers);
        init();
        FTN_NAME(INIT,init)();
        MBLK_Attach();
}

static MBlockSetupCookie cookie;
void MBlockInit(void) 
{
  CtvInitialize(MBlockChunk *, _mblkptr);
  TCharmSetFallbackSetup(MBlockFallbackSetup);
}

MBlockSetupCookie::MBlockSetupCookie(void)
{
  nblocks=-1;
  ndims=3;	
}

void MBlockSetupCookie::createArray(void)
{
  if (nblocks==-1)
    CkAbort("You must call MBLCK_Set_nblocks during init!\n");

  CkArrayOptions opt(nblocks);
  TCharmSetupCookie *tc=TCharmSetupCookie::get();
  if (!tc->hasThreads())
	  CkAbort("You must create a threads array with TCharmCreate before calling MBLK_Attach!\n");
  opt.bindTo(tc->getThreads());
  CkArrayID aid=CProxy_MBlockChunk::ckNew(
	  new MBlockInitMsg((const char *)prefix, ndims,tc->getThreads()),opt);
  CProxy_MBlockChunk *proxy=new CProxy_MBlockChunk(aid);
  proxy->setReductionClient(_allReduceHandler,proxy);
  tc->addClient(aid);
}

CDECL void MBLK_Attach(void)
{
	cookie.createArray();
}
FDECL void FTN_NAME(MBLK_ATTACH,mblk_attach)(void)
{ MBLK_Attach(); }


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

/******************************* MBLOCKCHUNK *********************************/

void MBlockChunk::commonInit(void)
{
  nfields = 0;
  thisproxy=thisArrayID;

}

MBlockChunk::MBlockChunk(MBlockInitMsg *msg)
{
  commonInit();

  threads=msg->threads;
  migInit();

  for (int bc=0;bc<MBLK_MAXBC;bc++)
    bcs[bc].fn=NULL;
  
  seqnum = 0;
  update.nRecd = 0;
  update.wait_seqnum = -1;
  messages = CmmNew();	
  
  b = new block(msg->prefix, thisIndex);
  int n = b->getPatches();
  nRecvPatches = 0;
  for(int i=0;i<n;i++) {
    patch *p = b->getPatch(i);
    if(p->type == patch::internal)
      nRecvPatches++;
  }
  delete msg;
  thread->ready();
}

MBlockChunk::MBlockChunk(CkMigrateMessage *msg)
	:ArrayElement1D(msg)
{
  b=NULL;
  messages=NULL;
  commonInit();
}


//Update fields after migration
void MBlockChunk::migInit(void)
{
  thread=threads[thisIndex].ckLocal();
  if (thread==NULL) CkAbort("MBlock can't locate its thread!\n");
  CtvAccessOther(thread->getThread(),_mblkptr)=this;
}

void MBlockChunk::ckJustMigrated(void)
{
  ArrayElement1D::ckJustMigrated();
  migInit();
}

MBlockChunk::~MBlockChunk() //Destructor-- deallocate memory
{
        CmmFree(messages);
	delete b;
}

void
MBlockChunk::recv(MBlockDataMsg *dm)
{
  if(dm->seqnum == update.wait_seqnum) {
    DBG( "recv: handling ("<<dm->seqnum<<")" )
    update_field(dm); // update the appropriate field value
    if(update.nRecd==nRecvPatches) {
      update.wait_seqnum = -1; //This update is complete
      thread->resume();
    }
  } else if (dm->seqnum>seqnum) {
    DBG( "recv: stashing early ("<<dm->seqnum<<")" )
    CmmPut(messages, 1, &(dm->seqnum), dm);
  } else
    CkAbort("MBLOCK MBlockChunk received message from the past!\n");
}

//Send the values for my source patches out
void
MBlockChunk::send(int fid,const extrudeMethod &meth,const void *grid)
{
  const DType &dt=fields[fid]->dt;
  int len = dt.length();
  const char *fStart=(const char *)grid;
  fStart+=dt.init_offset;
  blockDim d = fields[fid]->dim;
  int n = b->getPatches();
  for(int m=0;m<n;m++) {
    patch *p = b->getPatch(m);
    if (p->type != patch::internal)
      continue;
    internalBCpatch *ip = (internalBCpatch*) p;
    blockSpan src=ip->getExtents(meth,fields[fid]->forVoxel,-1);
    int dest = ip->dest;
    int num = src.getDim().getSize();
    int msgLen=len*num;
    MBlockDataMsg *msg = new (&msgLen, 0) MBlockDataMsg(seqnum, thisIndex, fid, 
        ip->destPatch); 
    CHK(msg);
    blockLoc dsize; //Size in destination space
    blockLoc so; //Source origin-- corresponds to dest. start
    for (int axis=0;axis<3;axis++) {
      dsize[ip->orient[axis]]=src.getDim()[axis];
      if (ip->orient.dIsFlipped(axis)) 
	so[axis]=src.end[axis]-1;
      else
	so[axis]=src.start[axis]; 
    }
    blockLoc iDir=ip->orient.destAxis(0);
    blockLoc jDir=ip->orient.destAxis(1);
    blockLoc kDir=ip->orient.destAxis(2);
    
    //Loop over the destination space
    void *data = msg->data;
    for(int k=0;k<dsize[2];k++) 
      for(int j=0;j<dsize[1];j++)
        for(int i=0;i<dsize[0];i++)	
	{
	  blockLoc from=so+i*iDir+j*jDir+k*kDir;
#ifndef CMK_OPTIMIZE  /*Perform bounds check*/
	  if (!src.contains(from))
	    CkAbort("Read location out of bounds in mblock::MBlockChunk::send!\n");
#endif
          const void *src = (const void *)(fStart+d[from]*dt.distance);
          memcpy(data, src, len);
          data = (void*) ((char*)data + len);
        }
    thisproxy[dest].recv(msg);
    DBG("  sending "<<num<<" data items ("<<seqnum<<") to processor "<<dest)
  }
}


//Update my ghost cells based on these values
void
MBlockChunk::update_field(MBlockDataMsg *msg)
{
  DBG("    update_field ("<<msg->seqnum<<")")
  const DType &dt=fields[msg->fid]->dt;
  int length=dt.length();
  char *aStart=(char *)update.ogrid;
  char *aEnd=aStart+fields[msg->fid]->arrayBytes;
  char *fStart=aStart+dt.init_offset;
  char *data = (char *) msg->data;
  internalBCpatch *p = (internalBCpatch*) (b->getPatch(msg->patchno));
  blockSpan dest=p->getExtents(update.m,fields[msg->fid]->forVoxel);

  blockDim dim = fields[msg->fid]->dim;
  //Loop over the destination-space patch
  for(int k=dest.start[2]; k<dest.end[2]; k++)
    for(int j=dest.start[1]; j<dest.end[1]; j++)
      for(int i=dest.start[0]; i<dest.end[0]; i++) 
      {
        char *dest = fStart+dim.c_index(i,j,k)*dt.distance;
#ifndef CMK_OPTIMIZE /*Perform bounds check*/
	if (dest<aStart || dest>=aEnd)
	  CkAbort("MBlockChunk::update_field would write out of bounds!\n");
#endif
        memcpy(dest, data, length);
        data = data + length;
      }
  delete msg;
  update.nRecd++;
}

void
MBlockChunk::start_update(int fid,const extrudeMethod &m,void *ingrid, void *outgrid)
{
  //Update sequence number
  seqnum++;
  update.nRecd=0;
  update.wait_seqnum=seqnum;
  update.m=m;
  update.ogrid=outgrid;

  DBG("update("<<seqnum<<") {")
  // first send my field values to all the processors that need it
  send(fid, m,ingrid);

  // now, if any of the field values have been received already,
  // process them
  MBlockDataMsg *dm;
  while ((dm = (MBlockDataMsg*)CmmGet(messages, 1, &seqnum, 0))!=NULL) {
    DBG("   handling stashed message")
    update_field(dm);
  }
  if (update.nRecd == nRecvPatches)
    update.wait_seqnum = -1; // we now have everything

  DBG("} update("<<seqnum<<")")
}

int
MBlockChunk::test_update(void)
{
  return (update.wait_seqnum==(-1))?MBLK_DONE : MBLK_NOTDONE;
}

int
MBlockChunk::wait_update(void)
{
  if(update.wait_seqnum != (-1)) {
    DBG("sleeping for "<<nRecvPatches-nRecd<<" update messages")
    thread->suspend();
    DBG("got all update messages for "<<seqnum)  
  }
  return MBLK_SUCCESS;
}

void
MBlockChunk::reduce_field(int fid, const void *grid, void *outbuf, int op)
{
  // first reduce over local gridpoints
  const DType &dt = fields[fid]->dt;
  const void *src = (const void *) ((const char *) grid + dt.init_offset);
  ::initialize(dt,outbuf,op);
  combineFn fn=combine(dt,op);
  int dim[3];
  MBLK_Get_blocksize(dim);
  blockDim d = fields[fid]->dim;
  for(int k=0; k<dim[2]; k++)
    for(int j=0; j<dim[1]; j++)
      for(int i=0; i<dim[0]; i++)
        fn(dt.vec_len, outbuf, 
            ((const char *)src + d.c_index(i,j,k)*dt.distance));
  // and now reduce over partitions
  reduce(fid, outbuf, outbuf, op);
}

void
MBlockChunk::reduce(int fid, const void *inbuf, void *outbuf, int op)
{
  const DType &dt = fields[fid]->dt;
  int len = dt.length();
  if(numElements==1) {
    memcpy(outbuf,inbuf,len);
    return;
  }
  CkReduction::reducerType rtype;
  switch(op) {
    case MBLK_SUM:
      switch(dt.base_type) {
        case MBLK_INT: rtype = CkReduction::sum_int; break;
        case MBLK_REAL: rtype = CkReduction::sum_float; break;
        case MBLK_DOUBLE: rtype = CkReduction::sum_double; break;
      }
      break;
    case MBLK_MAX:
      switch(dt.base_type) {
        case MBLK_INT: rtype = CkReduction::max_int; break;
        case MBLK_REAL: rtype = CkReduction::max_float; break;
        case MBLK_DOUBLE: rtype = CkReduction::max_double; break;
      }
      break;
    case MBLK_MIN:
      switch(dt.base_type) {
        case MBLK_INT: rtype = CkReduction::min_int; break;
        case MBLK_REAL: rtype = CkReduction::min_float; break;
        case MBLK_DOUBLE: rtype = CkReduction::min_double; break;
      }
      break;
  }
  contribute(len, (void *)inbuf, rtype);
  reduce_output = outbuf;
  thread->suspend();
}

void
MBlockChunk::reductionResult(MBlockDataMsg *msg)
{
  //msg->from used as length
  memcpy(reduce_output, msg->data, msg->from);
  reduce_output=NULL;
  thread->resume();
  delete msg;
}

/********** Thread/migration support ************/
static MBlockChunk *getCurMBlockChunk(void) 
{
  MBlockChunk *cptr=CtvAccess(_mblkptr);
  if (cptr==NULL) 
    CkAbort("Routine can only be called from driver()!\n");
  return cptr;
}


/******************************* C Bindings **********************************/

CDECL int 
MBLK_Set_prefix(const char *prefix)
{
  if(TCharm::getState() != inInit) {
    CkError("MBLK_Set_prefix called from outside init\n");
    return MBLK_FAILURE;
  }
  strcpy(cookie.prefix,prefix);
  return MBLK_SUCCESS;
}

CDECL int 
MBLK_Set_nblocks(const int n)
{
  if(TCharm::getState() != inInit) {
    CkError("MBLK_Set_nblocks called from outside init\n");
    return MBLK_FAILURE;
  }
  cookie.nblocks = n;
  return MBLK_SUCCESS;
}

CDECL int 
MBLK_Set_dim(const int n)
{
  if(TCharm::getState() != inInit) {
    CkError("MBLK_Set_dim called from outside init\n");
    return MBLK_FAILURE;
  }
  cookie.ndims = n;
  return MBLK_SUCCESS;
}

CDECL int 
MBLK_Get_nblocks(int *n)
{
  *n = TCharmNumElements();
  return MBLK_SUCCESS;
}

CDECL int 
MBLK_Get_myblock(int *m)
{
  *m = TCharmElement();
  return MBLK_SUCCESS;
}

CDECL int 
MBLK_Get_blocksize(int *dim)
{
  if(TCharm::getState() != inDriver) {
    CkError("MBLK_Get_blocksize  called from outside driver\n");
    return MBLK_FAILURE;
  }
  MBlockChunk *cptr = getCurMBlockChunk();
  blockDim d = cptr->b->getDim();
  /*Subtract one to get to voxel coordinates*/
  dim[0] = d[0]-1; dim[1] = d[1]-1; dim[2] = d[2]-1;
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
  if(TCharm::getState()==inDriver) {
    MBlockChunk *cptr = getCurMBlockChunk();
    CkPrintf("[%d] %s\n", cptr->thisIndex, str);
  } else {
    CkPrintf("%s\n", str);
  }
}

CDECL int 
MBLK_Register(void *_ud,MBLK_PupFn _pup_ud, int *rid)
{
  *rid=TCharmRegister(_ud,_pup_ud);
  return MBLK_SUCCESS;
}

CDECL int 
MBLK_Get_registered(int n, void **b)
{
  *b=TCharmGetUserdata(n);
  return MBLK_SUCCESS;
}

CDECL int 
MBLK_Migrate(void)
{
  TCharm::get()->migrate();
  return MBLK_SUCCESS;
}

CDECL int 
MBLK_Create_field(int *dims,int isVoxel,
		  int base_type, int vec_len, 
		  int init_offset, int distance,
		  int *fid)
{
  if(TCharm::getState() != inDriver) {
    CkError("MBLK_Create_field  called from outside driver\n");
    return MBLK_FAILURE;
  }
  field_t *f=new field_t(DType(base_type,vec_len,init_offset,distance),
			 blockDim(dims[0],dims[1],dims[2]),isVoxel==1);
  *fid = getCurMBlockChunk()->add_field(f);
  return ((*fid==(-1)) ? MBLK_FAILURE : MBLK_SUCCESS);
}

CDECL int
MBLK_Update_field(int fid, int ghostWidth, void *grid)
{
  if(TCharm::getState() != inDriver) {
    CkError("MBLK_Update_field  called from outside driver\n");
    return MBLK_FAILURE;
  }
  MBLK_Iupdate_field(fid, ghostWidth, grid, grid);
  return MBLK_Wait_update();
}

CDECL int
MBLK_Iupdate_field(int fid, int ghostWidth, void *ingrid, void *outgrid)
{
  if(TCharm::getState() != inDriver) {
    CkError("MBLK_Iupdate_field  called from outside driver\n");
    return MBLK_FAILURE;
  }
  getCurMBlockChunk()->start_update(fid,extrudeMethod(ghostWidth),ingrid,outgrid);
  return MBLK_SUCCESS;
}

CDECL int
MBLK_Test_update(int *status)
{
  if(TCharm::getState() != inDriver) {
    CkError("MBLK_Test_update  called from outside driver\n");
    return MBLK_FAILURE;
  }
  *status=getCurMBlockChunk()->test_update();
  return MBLK_SUCCESS; 
}

CDECL int
MBLK_Wait_update(void)
{
  if(TCharm::getState() != inDriver) {
    CkError("MBLK_Wait_update  called from outside driver\n");
    return MBLK_FAILURE;
  }
  return getCurMBlockChunk()->wait_update();
}

CDECL int
MBLK_Reduce_field(int fid, void *ingrid, void *outbuf, int op)
{
  if(TCharm::getState() != inDriver) {
    CkError("MBLK_Reduce_field  called from outside driver\n");
    return MBLK_FAILURE;
  }
  getCurMBlockChunk()->reduce_field(fid, ingrid, outbuf, op);
  return MBLK_SUCCESS;
}

CDECL int
MBLK_Reduce(int fid, void *inbuf, void *outbuf, int op)
{
  if(TCharm::getState() != inDriver) {
    CkError("MBLK_Reduce  called from outside driver\n");
    return MBLK_FAILURE;
  }
  getCurMBlockChunk()->reduce(fid, inbuf, outbuf, op);
  return MBLK_SUCCESS;
}

CDECL int
MBLK_Register_bc(const int bcnum, int ghostWidth,const MBLK_BcFn bcfn)
{
  if(TCharm::getState() != inDriver) {
    CkError("MBLK_Register_bc  called from outside driver\n");
    return MBLK_FAILURE;
  }
  return getCurMBlockChunk()->register_bc(bcnum, bcfn, extrudeMethod(ghostWidth), false);
}

CDECL int
MBLK_Apply_bc(const int bcnum, void *p1,void *p2)
{
  if(TCharm::getState() != inDriver) {
    CkError("MBLK_Apply_bc  called from outside driver\n");
    return MBLK_FAILURE;
  }
  int retval =  getCurMBlockChunk()->apply_bc(bcnum, p1,p2);
  return retval;
}

CDECL int
MBLK_Apply_bc_all(void *p1,void *p2)
{
  if(TCharm::getState() != inDriver) {
    CkError("MBLK_Apply_bc_all  called from outside driver\n");
    return MBLK_FAILURE;
  }
  int retval =  getCurMBlockChunk()->apply_bc_all(p1,p2);
  return retval;
}

CDECL void 
MBLK_Print_block(void)
{
  if(TCharm::getState()==inDriver) {
    MBlockChunk *cptr = getCurMBlockChunk();
    cptr->print();
  }
}

/************************ Fortran Bindings *********************************/

// Utility functions for Fortran

FDECL int FTN_NAME(OFFSETOF,offsetof)
  (void *first, void *second)
{
  return (int)((char *)second - (char*)first);
}

FDECL void FTN_NAME(MBLK_SET_PREFIX, mblk_set_prefix)
  (const char *str, int *ret)
{
  int len=0; /*Find the end of the string by looking for the space*/
  while (str[len]!=' ') len++;
  char *tmpstr = new char[len+1]; CHK(tmpstr);
  memcpy(tmpstr,str,len);
  tmpstr[len] = '\0';
  *ret = MBLK_Set_prefix(tmpstr);
  delete[] tmpstr;
}

FDECL void FTN_NAME(MBLK_SET_NBLOCKS, mblk_set_nblocks)
  (int *nblocks, int *ret)
{
  *ret = MBLK_Set_nblocks(*nblocks);
}

FDECL void FTN_NAME(MBLK_SET_DIM, mblk_set_dim)
  (int *dim, int *ret)
{
  *ret = MBLK_Set_dim(*dim);
}

FDECL void FTN_NAME(MBLK_GET_NBLOCKS, mblk_get_nblocks)
  (int *n, int *ret)
{
  *ret = MBLK_Get_nblocks(n);
}

FDECL void FTN_NAME(MBLK_GET_MYBLOCK, mblk_get_myblock)
  (int *m, int *ret)
{
  *ret = MBLK_Get_myblock(m);
}
static void c2f_index3d(int *idx) {
  idx[0]++; idx[1]++; idx[2]++;
}
FDECL void FTN_NAME(MBLK_GET_BLOCKSIZE, mblk_get_blocksize)
  (int *dims, int *ret)
{
  *ret = MBLK_Get_blocksize(dims);
}

FDECL double FTN_NAME(MBLK_TIMER, mblk_timer)
  (void)
{
  return MBLK_Timer();
}

FDECL void FTN_NAME(MBLK_PRINT,mblk_print)
  (char *str, int len)
{
  char *tmpstr = new char[len+1]; CHK(tmpstr);
  memcpy(tmpstr,str,len);
  tmpstr[len] = '\0';
  MBLK_Print(tmpstr);
  delete[] tmpstr;
}

FDECL void FTN_NAME(MBLK_PRINT_BLOCK,mblk_print_block)
  (void)
{
  MBLK_Print_block();
}

FDECL void FTN_NAME(MBLK_CREATE_FIELD, mblk_create_field)
  (int *dims,int *forVox,int *b, int *l, int *o, int *d, int *f, int *ret)
{
  *ret = MBLK_Create_field(dims,*forVox,*b, *l, *o, *d, f);
}

FDECL void FTN_NAME(MBLK_UPDATE_FIELD, mblk_update_field)
  (int *fid, int *ghostWidth,void *grid, int *ret)
{
  *ret = MBLK_Update_field(*fid,*ghostWidth, grid);
}

FDECL void FTN_NAME(MBLK_IUPDATE_FIELD, mblk_iupdate_field)
  (int *fid, int *ghostWidth, void *igrid, void *ogrid, int *ret)
{
  *ret = MBLK_Iupdate_field(*fid,*ghostWidth, igrid, ogrid);
}

FDECL void FTN_NAME(MBLK_TEST_UPDATE, mblk_test_update)
  (int *status, int *ret)
{
  *ret = MBLK_Test_update(status);
}

FDECL void FTN_NAME(MBLK_WAIT_UPDATE, mblk_wait_update)
  (int *ret)
{
  *ret = MBLK_Wait_update();
}

FDECL void FTN_NAME(MBLK_REDUCE_FIELD, mblk_reduce_field)
  (int *fid, void *grid, void *out, int *op, int *ret)
{
  *ret = MBLK_Reduce_field(*fid, grid, out, *op);
}

FDECL void FTN_NAME(MBLK_REDUCE, mblk_reduce)
  (int *fid, void *in, void *out, int *op, int *ret)
{
  *ret = MBLK_Reduce_field(*fid, in, out, *op);
}

FDECL void FTN_NAME(MBLK_REGISTER_BC, mblk_register_bc)
  (int *bcnum, int *ghostWidth,MBLK_BcFn bcfn, int *ret)
{
  if(TCharm::getState() != inDriver) {
    CkError("MBLK_Register_bc  called from outside driver\n");
    *ret=MBLK_FAILURE;
  } else {
    *ret=getCurMBlockChunk()->register_bc(*bcnum, bcfn, extrudeMethod(*ghostWidth), true);	
  }
}

FDECL void FTN_NAME(MBLK_APPLY_BC, mblk_apply_bc)
  (int *bcnum, void *p1,void *p2, int *ret)
{
  *ret = MBLK_Apply_bc(*bcnum, p1,p2);
}

FDECL void FTN_NAME(MBLK_APPLY_BC_ALL, mblk_apply_bc_all)
  (void *p1,void *p2, int *ret)
{
  *ret = MBLK_Apply_bc_all(p1,p2);
}

FDECL void FTN_NAME(MBLK_REGISTER, mblk_register)
  (void *block, MBLK_PupFn pupfn, int *rid, int *ret)
{
  *ret = MBLK_Register(block, pupfn, rid);
}

FDECL void FTN_NAME(MBLK_MIGRATE, mblk_migrate)
  (int *ret)
{
  *ret = MBLK_Migrate();
}


void
MBlockChunk::print(void)
{
  CkPrintf("-------------------- Block %d --------------------\n",thisIndex);
  b->print();
  CkPrintf("--------------------------------------------------\n",thisIndex);
}  
  

void
MBlockChunk::pup(PUP::er &p)
{
//Pup superclass
  ArrayElement1D::pup(p);

  threads.pup(p);
  messages=CmmPup(&p,messages);

  if(p.isUnpacking())
    b = new block();
  b->pup(p);
//Pup all other fields
  p(nfields);
  for (int i=0;i<nfields;i++) {
    if (p.isUnpacking()) fields[i]=new field_t;
    fields[i]->pup(p);
  }
  p((void*)bcs, MBLK_MAXBC*sizeof(bcs[0]));
  p(seqnum);
  p(update.wait_seqnum);
  p(update.nRecd);
  p(nRecvPatches);
}

#include "mblock.def.h"
