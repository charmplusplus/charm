/*
Finite Element Method Framework for Charm++
Parallel Programming Lab, Univ. of Illinois 2000

This file implements a C, C++, or Fortran-callable
library for parallel finite-element computations.

The basic idea is to partition the serial mesh into
"chunks", which do nearly independent computations;
but occasionally neighboring mesh chunks must
synchronize the values (called fields) at the shared 
nodes.  For load balance, we create more chunks than
processors and occasionally migrate chunks for better
load balance.

We run the user's (timeloop) driver routine in a 
thread (in the style of AMPI), so communication looks
blocking to the user.  Internally, we suspend the
user's driver thread when communication is needed,
then resume the thread when the results arrive.
 */
#include "fem_impl.h"
#include <limits.h>
#include <float.h> /*for FLT_MIN on non-Suns*/

FDECL void FTN_NAME(INIT,init)(void);
FDECL void FTN_NAME(DRIVER,driver)(void);
FDECL void FTN_NAME(MESH_UPDATED,mesh_updated)(int *userParameter);

/*Startup:*/
static void callDrivers(void) {
        driver();
        FTN_NAME(DRIVER,driver)();
}

static void FEMfallbackSetup(void)
{
	int nChunks=TCharmGetNumChunks();
	char **argv=TCharmArgv();
	int initFlags=0;
	if (CmiGetArgFlag(argv,"-read")) initFlags|=FEM_INIT_READ;
	if (CmiGetArgFlag(argv,"-write")) initFlags|=FEM_INIT_WRITE;
	TCharmCreate(nChunks,callDrivers);
	if (!(initFlags&FEM_INIT_READ)) {
		init();
		FTN_NAME(INIT,init)();
	}
        FEM_Attach(initFlags);
}

//_femptr gives the current chunk, and is only
// valid in routines called from driver().
CtvStaticDeclare(FEMchunk*, _femptr);

void FEMnodeInit(void) {
	CtvInitialize(FEMchunk*, _femptr);
	TCharmSetFallbackSetup(FEMfallbackSetup);	
}

static void 
_allReduceHandler(void *proxy_v, int datasize, void *data)
{
  // the reduction handler is called on processor 0
  // with available reduction results
  FEM_DataMsg *dmsg = new (&datasize, 0) FEM_DataMsg(0,datasize,0); CHK(dmsg);
  memcpy(dmsg->data, data, datasize);
  CProxy_FEMchunk &proxy=*(CProxy_FEMchunk *)proxy_v;
  // broadcast the reduction results to all array elements
  proxy.reductionResult(dmsg);
}


//_meshptr gives the current serial mesh
// (NULL if none).  It is only valid during
// init, mesh_updated, and finalize.
static FEM_Mesh* _meshptr = 0;

//Maps element number to (0-based) chunk number, allocated with new[]
int *_elem2chunk=NULL;

//Partitions and splits the given mesh
static void mesh2msgs(const FEM_Mesh *mesh,int _nchunks,ChunkMsg **msgs) {
    int *elem2chunk=_elem2chunk;
    if (elem2chunk==NULL) 
    {//Partition the elements ourselves
    	elem2chunk=new int[mesh->nElems()];
    	fem_partition(mesh,_nchunks,elem2chunk);
    }
    //Build communication lists and split mesh data
    fem_map(mesh,_nchunks,elem2chunk,msgs);
    //Blow away old partitioning
    delete[] elem2chunk;
    _elem2chunk=NULL;
}

//Free the global serial mesh (if possible)
static void freeMesh(void) {
	if (_meshptr==NULL) return; //Nothing to do
	_meshptr->deallocate();
	delete _meshptr;
	_meshptr=NULL;
}

CDECL void FEM_Attach(int flags)
{
	//Make sure the threads array exists
	TCharmSetupCookie *tc=TCharmSetupCookie::get();
	if (!tc->hasThreads())
		CkAbort("You must create a thread array with TCharmCreate before calling FEM_Attach!\n");
	int _nchunks=tc->getNumElements();
	
	//Create a new chunk array
	CProxy_FEMcoordinator coord=CProxy_FEMcoordinator::ckNew(_nchunks);
	FEMinit init(_nchunks,tc->getThreads(),flags,coord);
	CkArrayOptions opts(_nchunks);
	opts.bindTo(tc->getThreads());
	CProxy_FEMchunk chunks= CProxy_FEMchunk::ckNew(init,opts);
	chunks.setReductionClient(_allReduceHandler, new CProxy_FEMchunk(chunks));
	coord.setArray(chunks);
	tc->addClient(chunks);
	
	//Send the mesh out to the chunks
	if (_meshptr!=NULL) {
		//Partition the serial mesh online
		ChunkMsg **cmsgs=new ChunkMsg*[_nchunks]; CHK(cmsgs);		
		mesh2msgs(_meshptr,_nchunks,cmsgs);
		freeMesh();
		for (int i=0;i<_nchunks;i++) {
			if (flags&FEM_INIT_WRITE) chunks[i].write(cmsgs[i]);
			else chunks[i].run(cmsgs[i]);
		}
		delete[] cmsgs;
	} else /*NULL==mesh*/ {
		chunks.run();
	}
}

//This coordinator manages mesh reassembly for a FEM array:
class FEMcoordinator : public Chare {
	int nChunks; //Number of array elements total
	CProxy_FEMchunk femChunks;
	ChunkMsg **cmsgs; //Messages from/for array elements
	int updateCount; //Number of mesh updates so far
	CkQ<ChunkMsg *> futureUpdates;
	CkQ<ChunkMsg *> curUpdates; 
	int numdone; //Length of curUpdates 
public:
	FEMcoordinator(int nChunks_) 
		:nChunks(nChunks_)
	{
		cmsgs=new ChunkMsg*[nChunks]; CHK(cmsgs);		
		numdone=0;
		updateCount=0;
	}
	~FEMcoordinator() {
		delete[] cmsgs;
	}
	
	void setArray(const CkArrayID &fem_) {femChunks=fem_;}
	void updateMesh(ChunkMsg *m);
};


//Called by a chunk on FEM_Update_Mesh
void FEMcoordinator::updateMesh(ChunkMsg *msg)
{
  if (msg->updateCount>updateCount) {
    //This is a message for a future update-- save it for later
    futureUpdates.enq(msg);
  } else if (msg->updateCount<updateCount) 
    CkAbort("main::updateMesh> Received mesh chunk from the past!\n");
  else /*(msg->updateCount==updateCount)*/{
    int _nchunks=nChunks;
    //A chunk for the current mesh
    curUpdates.enq(msg);
    while (curUpdates.length()==_nchunks) {
      //We have all the chunks of the current mesh-- process them and start over
      int i;
      for (i=0;i<_nchunks;i++) {
      	ChunkMsg *m=curUpdates.deq();
	cmsgs[m->fromChunk]=m;
      }
      //Save what to do with the mesh
      int callMeshUpdated=cmsgs[0]->callMeshUpdated;
      int doRepartition=cmsgs[0]->doRepartition;
      //Assemble the current chunks into a serial mesh
      freeMesh();
      _meshptr=fem_assemble(_nchunks,cmsgs);
      //Blow away the old chunks
      for (i=0;i<_nchunks;i++) {
      	delete cmsgs[i];
      	cmsgs[i]=NULL;
      }

      //Now that the mesh is assembled, handle it
      if (callMeshUpdated) {
	TCharm::setState(inInit);
	FTN_NAME(MESH_UPDATED,mesh_updated) (&callMeshUpdated);
	mesh_updated(callMeshUpdated);
	TCharm::setState(inDriver);
      }
      if (doRepartition) {
	mesh2msgs(_meshptr,_nchunks,cmsgs);
	freeMesh();
	for (i=0;i<_nchunks;i++) {
	  femChunks[i].meshUpdated(cmsgs[i]);
	}
      }

      //Check for relevant messages in the future buffer
      updateCount++;
      for (i=0;i<futureUpdates.length();i++)
	if (futureUpdates[i]->updateCount==updateCount) {
	  curUpdates.enq(futureUpdates[i]);
	  futureUpdates[i--]=futureUpdates.deq();
	}
    }
  }
  
}


/********************** Mesh Creation ************************/
/*Utility*/

/*Transpose matrix in (which is nx by ny) to out (which is ny by nx).
Equivalently, convert row-major in to column-major out (both nx by ny);
or convert column-major in to row-major out (both ny by nx).
in cannot be the same matrix as out.
*/
template <class dtype>
void transpose(int nx, int ny,const dtype *in,dtype *out)
{
  for(int y=0;y<ny;y++)
    for(int x=0;x<nx;x++)
      out[x*ny+y] = in[y*nx+x];
}

/*As above, but add the given value to each matrix element.
in cannot be the same matrix as out.
*/
static void transposeAdd(int nx, int ny,const int *in,int add,int *out)
{
  for(int y=0;y<ny;y++)
    for(int x=0;x<nx;x++)
      out[x*ny+y] = in[y*nx+x]+add;
}
/*Copy the given matrix, adding the given value to each element.
in may equal out.
*/
static void copyAdd(int nx,int ny,const int *in,int add,int *out)
{
  int n=nx*ny;
  for(int i=0;i<n;i++)
    out[i] = in[i]+add;
}

void FEM_Mesh::count::setUdata_r(const double *Nudata)
{
	allocUdata();
	memcpy(udata,Nudata,udataCount()*sizeof(double));
}
void FEM_Mesh::count::setUdata_c(const double *Nudata)
{
	allocUdata();
	transpose(n,dataPer,Nudata,udata);
}

void FEM_Mesh::count::getUdata_r(double *Nudata) const
{
	memcpy(Nudata,udata,udataCount()*sizeof(double));
}
void FEM_Mesh::count::getUdata_c(double *Nudata) const
{
	transpose(dataPer,n,udata,Nudata);
}

/***** Mesh getting and setting state ****/

static FEM_Mesh *setMesh(void) {
  if(TCharm::getState()==inDriver) {
    FEMchunk *cptr = CtvAccess(_femptr);
    if (cptr->updated_mesh==NULL)
      cptr->updated_mesh=new ChunkMsg;
    return &cptr->updated_mesh->m;
  } else {
    //Called from init, finalize, or meshUpdate
    if (_meshptr==NULL)
      _meshptr=new FEM_Mesh;
    return _meshptr;
  }
}

static const FEM_Mesh *getMesh(void) {
  if(TCharm::getState()==inDriver) {
    FEMchunk *cptr = CtvAccess(_femptr);
    return &cptr->m;
  } else {
    //Called from init, finalize, or meshUpdate
    if (_meshptr==NULL) {
      CkAbort("FEM: Cannot get mesh-- it was never set!\n");
    }
    return _meshptr;
  }
}

//Check an element type field for validitity-- abort if bad
static int chkET(int et)
{
  if (et<0 || et>2*FEM_MAX_ELEMTYPES)
    CkAbort("FEM Error> Invalid element type!\n");
  if (et>=FEM_MAX_ELEMTYPES)
    CkAbort("FEM Error> Registered too many element types!\n");
  return et;
}

/****** Custom Partitioning API *******/
static void Set_Partition(int *elem2chunk,int indexBase) {
	if (_elem2chunk!=NULL) delete[] _elem2chunk;
	const FEM_Mesh *m=getMesh();
	int nElem=m->nElems();
	_elem2chunk=new int[nElem];
	for (int i=0;i<nElem;i++)
		_elem2chunk[i]=elem2chunk[i]-indexBase;
}

//C bindings:
CDECL void FEM_Set_Partition(int *elem2chunk) {
	Set_Partition(elem2chunk,0);
}

//Fortran bindings:
FDECL void FTN_NAME(FEM_SET_PARTITION,fem_set_partition)
	(int *elem2chunk) 
{
	Set_Partition(elem2chunk,1);
}

/***** Mesh-Setting API: C bindings*/
CDECL void FEM_Set_Node(int nNodes,int dataPer) 
{
	FEM_Mesh *m=setMesh();
	m->node.dataPer=dataPer;
	m->node.n=nNodes;
}
CDECL void FEM_Set_Node_Data(const double *data) 
  {setMesh()->node.setUdata_r(data);}

CDECL void FEM_Set_Elem(int elType,int nElem,int dataPer,int nodePer) {
	FEM_Mesh *m=setMesh();
	chkET(elType);
	if (m->nElemTypes<=elType)
		m->nElemTypes=elType+1;
	m->elem[elType].n=nElem;
	m->elem[elType].dataPer=dataPer;
	m->elem[elType].nodesPer=nodePer;
}
CDECL void FEM_Set_Elem_Data(int elType,const double *data) 
  {setMesh()->elem[chkET(elType)].setUdata_r(data);}
CDECL void FEM_Set_Elem_Conn(int elType,const int *conn) {
	FEM_Mesh::elemCount &c=setMesh()->elem[chkET(elType)];
	c.allocConn();
	memcpy(c.conn,conn,c.connCount()*sizeof(int));
}

/*Convenience routine: for use when you only have one kind of element
and no userdata.*/
CDECL void FEM_Set_Mesh(int nelem, int nnodes, int ctype, int *conn)
{
	FEM_Set_Node(nnodes,0);
	FEM_Set_Elem(0,nelem,0,ctype);
	FEM_Set_Elem_Conn(0,conn);
}

FDECL void FTN_NAME(FEM_SET_NODE,fem_set_node)
	(int *nNodes,int *dataPer) 
  {FEM_Set_Node(*nNodes,*dataPer);}
FDECL void FTN_NAME(FEM_SET_NODE_DATA_R,fem_set_node_data_r)
	(double *data) 
  {setMesh()->node.setUdata_r(data);}
FDECL void FTN_NAME(FEM_SET_NODE_DATA_C,fem_set_node_data_c)
	(double *data) 
  {setMesh()->node.setUdata_c(data);}

FDECL void FTN_NAME(FEM_SET_ELEM,fem_set_elem)
	(int *elType,int *nElem,int *dataPer,int *nodePer)  
  {FEM_Set_Elem(*elType-1,*nElem,*dataPer,*nodePer);}
FDECL void FTN_NAME(FEM_SET_ELEM_DATA_R,fem_set_elem_data_r)
	(int *elType,double *data)
  {setMesh()->elem[chkET(*elType-1)].setUdata_r(data);}
FDECL void FTN_NAME(FEM_SET_ELEM_DATA_C,fem_set_elem_data_c)
	(int *elType,double *data)
  {setMesh()->elem[chkET(*elType-1)].setUdata_c(data);}

FDECL void FTN_NAME(FEM_SET_ELEM_CONN_R,fem_set_elem_conn_r)
	(int *elType,int *conn_r)
{
	FEM_Mesh::elemCount &c=setMesh()->elem[chkET(*elType-1)];
	c.allocConn();
	copyAdd(c.n,c.nodesPer,conn_r,-1,c.conn);
}
FDECL void FTN_NAME(FEM_SET_ELEM_CONN_C,fem_set_elem_conn_c)
	(int *elType,int *conn_c)
{
	FEM_Mesh::elemCount &c=setMesh()->elem[chkET(*elType-1)];
	c.allocConn();
	transposeAdd(c.n,c.nodesPer,conn_c,-1,c.conn);
}

/*Convenience routine: for use when you only have one kind of element*/
FDECL void FTN_NAME(FEM_SET_MESH,fem_set_mesh)
	(int *nelem, int *nnodes, int *ctype, int *conn)
{
	int elType=1,zero=0;
	FTN_NAME(FEM_SET_NODE,fem_set_node) (nnodes,&zero);
	FTN_NAME(FEM_SET_ELEM,fem_set_elem) (&elType,nelem,&zero,ctype);
	FTN_NAME(FEM_SET_ELEM_CONN_C,fem_set_elem_conn_c) (&elType,conn);
}

/***** Mesh-Getting API: C bindings*/

CDECL void FEM_Get_Node(int *nNodes,int *dataPer) 
{
	const FEM_Mesh *m=getMesh();
	if (nNodes!=NULL) *nNodes=m->node.n;
	if (dataPer!=NULL) *dataPer=m->node.dataPer;
}
CDECL void FEM_Get_Node_Data(double *data) 
  {getMesh()->node.getUdata_r(data);}

CDECL void FEM_Get_Elem(int elType,int *nElem,int *dataPer,int *nodePer) 
{
	const FEM_Mesh *m=getMesh();
	chkET(elType);
	if (nElem!=NULL) *nElem=m->elem[elType].n;
	if (dataPer!=NULL) *dataPer=m->elem[elType].dataPer;
	if (nodePer!=NULL) *nodePer=m->elem[elType].nodesPer;
}
CDECL void FEM_Get_Elem_Data(int elType,double *data) 
  {getMesh()->elem[chkET(elType)].getUdata_r(data);}
CDECL void FEM_Get_Elem_Conn(int elType,int *conn) {
	const FEM_Mesh::elemCount &c=getMesh()->elem[chkET(elType)];
	memcpy(conn,c.conn,c.n*c.nodesPer*sizeof(int));
}

FDECL void FTN_NAME(FEM_GET_NODE,fem_get_node)
	(int *nNodes,int *dataPer) 
  {FEM_Get_Node(nNodes,dataPer);}
FDECL void FTN_NAME(FEM_GET_NODE_DATA_R,fem_get_node_data_r)
	(double *data) 
  {getMesh()->node.getUdata_r(data);}
FDECL void FTN_NAME(FEM_GET_NODE_DATA_C,fem_get_node_data_c)
	(double *data) 
  {getMesh()->node.getUdata_c(data);}

FDECL void FTN_NAME(FEM_GET_ELEM,fem_get_elem)
	(int *elType,int *nElem,int *dataPer,int *nodePer)  
  {FEM_Get_Elem(*elType-1,nElem,dataPer,nodePer);}
FDECL void FTN_NAME(FEM_GET_ELEM_DATA_R,fem_get_elem_data_r)
	(int *elType,double *data) 
  {getMesh()->elem[chkET(*elType-1)].getUdata_r(data);}
FDECL void FTN_NAME(FEM_GET_ELEM_DATA_C,fem_get_elem_data_c)
	(int *elType,double *data) 
  {getMesh()->elem[chkET(*elType-1)].getUdata_c(data);}

FDECL void FTN_NAME(FEM_GET_ELEM_CONN_R,fem_get_elem_conn_r)
	(int *elType,int *conn)
{
	const FEM_Mesh::elemCount &c=getMesh()->elem[chkET(*elType-1)];
	copyAdd(c.nodesPer,c.n,c.conn,+1,conn);
}
FDECL void FTN_NAME(FEM_GET_ELEM_CONN_C,fem_get_elem_conn_c)
	(int *elType,int *conn)
{
	const FEM_Mesh::elemCount &c=getMesh()->elem[chkET(*elType-1)];
	transposeAdd(c.nodesPer,c.n,c.conn,+1,conn);
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

/*Several compilers "helpfully" define max and min-- confusing us completely*/
#undef max 
#undef min

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
    case FEM_SUM:
      switch(dt.base_type) {
        case FEM_BYTE : 
          assign(dt.vec_len,(unsigned char*)lhs, (unsigned char)0); 
          break;
        case FEM_INT : assign(dt.vec_len,(int*)lhs, 0); break;
        case FEM_REAL : assign(dt.vec_len,(float*)lhs, (float)0.0); break;
        case FEM_DOUBLE : assign(dt.vec_len,(double*)lhs, 0.0); break;
      }
      break;
    case FEM_MAX:
      switch(dt.base_type) {
        case FEM_BYTE : 
          assign(dt.vec_len,(unsigned char*)lhs, (unsigned char)CHAR_MIN); 
          break;
        case FEM_INT : assign(dt.vec_len,(int*)lhs, INT_MIN); break;
        case FEM_REAL : assign(dt.vec_len,(float*)lhs, FLT_MIN); break;
        case FEM_DOUBLE : assign(dt.vec_len,(double*)lhs, DBL_MIN); break;
      }
      break;
    case FEM_MIN:
      switch(dt.base_type) {
        case FEM_BYTE : 
          assign(dt.vec_len,(unsigned char*)lhs, (unsigned char)CHAR_MAX); 
          break;
        case FEM_INT : assign(dt.vec_len,(int*)lhs, INT_MAX); break;
        case FEM_REAL : assign(dt.vec_len,(float*)lhs, FLT_MAX); break;
        case FEM_DOUBLE : assign(dt.vec_len,(double*)lhs, DBL_MAX); break;
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
        case FEM_BYTE : return (combineFn)(combineFn_BYTE)fn;\
        case FEM_INT : return (combineFn)(combineFn_INT)fn;\
        case FEM_REAL : return (combineFn)(combineFn_REAL)fn;\
        case FEM_DOUBLE : return (combineFn)(combineFn_DOUBLE)fn;\
      }\
      break;
    case FEM_SUM: combine_switch(sum);
    case FEM_MIN: combine_switch(min);
    case FEM_MAX: combine_switch(max);
  }
  return NULL;
}

/******************************* CHUNK *********************************/

FEMchunk::FEMchunk(const FEMinit &init_)
	:init(init_), thisproxy(thisArrayID)
{
  initFields();

  ntypes = 0;
  new_DT(FEM_BYTE);
  new_DT(FEM_INT);
  new_DT(FEM_REAL);
  new_DT(FEM_DOUBLE);

  elemNums=nodeNums=isPrimary=gPeToIdx=NULL;
  updated_mesh=NULL;
  stored_mesh=NULL;
  
  messages = CmmNew();
  seqnum = 1;
  updateCount=1;
  wait_for = 0;

}
FEMchunk::FEMchunk(CkMigrateMessage *msg)
	:ArrayElement1D(msg), thisproxy(thisArrayID)
{
  elemNums=nodeNums=isPrimary=gPeToIdx=NULL;
  updated_mesh=NULL;
  stored_mesh=NULL;
  messages=NULL;	
}

FEMchunk::~FEMchunk() //Destructor-- deallocate memory
{
	CmmFree(messages);
	deallocate();
}

void FEMchunk::deallocate(void) {
	if (stored_mesh!=NULL) {
		delete stored_mesh;
		stored_mesh=NULL;
	} else {
		m.deallocate();
		comm.deallocate();
		delete[] elemNums; elemNums=NULL;
		delete[] nodeNums; nodeNums=NULL;
		delete[] isPrimary; isPrimary=NULL;
	}
	delete[] gPeToIdx; gPeToIdx=NULL;
}

//Update fields after creation/migration
void FEMchunk::initFields(void)
{
  CProxy_TCharm tp(init.threads);
  thread=tp[thisIndex].ckLocal();
  if (thread==NULL) CkAbort("FEM can't locate its thread!\n");
  CtvAccessOther(thread->getThread(),_femptr)=this;
}
void FEMchunk::ckJustMigrated(void)
{
  ArrayElement1D::ckJustMigrated(); //Call superclass
  initFields();
}

void FEMchunk::write(ChunkMsg *msg) {
  readChunk(msg);
  writeChunk();
}

void
FEMchunk::run(ChunkMsg *msg)
{
  readChunk(msg);
  thread->ready();
}

void
FEMchunk::run(void)
{
  readChunk();
  thread->ready();
}

void
FEMchunk::recv(FEM_DataMsg *dm)
{
  if (dm->tag == wait_for) {
    update_field(dm); // update the appropriate field value
    delete dm;
    nRecd++;
    if(nRecd==comm.nPes) {
      wait_for = 0; // done waiting for seqnum
      thread->resume();
    }
  } else {
    CmmPut(messages, 1, &(dm->tag), dm);
  }
}

/************************************************
"Gather" routines extract data distributed (sharedNodeIdx)
through the user's array (in) and collect it into a message (out).
 */
#define gather_args (int nVal,int valLen, \
		    const int *nodeIdx,int nodeScale, \
		    const char *in,char *out)
typedef void (*gather_fn) gather_args;

static void gather_general gather_args
{
  for(int i=0;i<nVal;i++) {
      const void *src = (const void *)(in+nodeIdx[i]*nodeScale);
      memcpy(out, src, valLen);
      out +=valLen;
  }
}

#define gather_doubles(n,copy) \
static void gather_double##n gather_args \
{ \
  double *od=(double *)out; \
  for(int i=0;i<nVal;i++) { \
      const double *src = (const double *)(in+nodeIdx[i]*nodeScale); \
      copy \
      od+=n; \
  } \
}

gather_doubles(1,od[0]=src[0];)
gather_doubles(2,od[0]=src[0];od[1]=src[1];)
gather_doubles(3,od[0]=src[0];od[1]=src[1];od[2]=src[2];)

//Gather and send the values for my shared nodes out
void
FEMchunk::send(int fid, const void *nodes)
{
  int p;
  const DType &dt=dtypes[fid];
  int len = dt.length();
  const char *fStart=(const char *)nodes;
  fStart+=dt.init_offset;
  gather_fn gather=gather_general;
  if (dt.base_type == FEM_DOUBLE) {
    switch(dt.vec_len) {
    case 1: gather=gather_double1;break;
    case 2: gather=gather_double2;break;
    case 3: gather=gather_double3;break;
    }
  }
  for(p=0;p<comm.nPes;p++) {
    int dest = comm.peNums[p];
    int num=comm.numNodesPerPe[p];
    int msgLen=len*num;
    FEM_DataMsg *msg = new (&msgLen, 0) FEM_DataMsg(seqnum, thisIndex, fid); CHK(msg);
    gather(num,len,comm.nodesPerPe[p],dt.distance,
	   fStart,(char *)msg->data);
    thisproxy[dest].recv(msg);
  }
}

/************************************************
"Scatter" routines add the message data (in) to the
shared nodes distributed through the user's data (out).
 */
#define scatter_args (int nVal, \
		    const int *nodeIdx,int nodeScale, \
		    const char *in,char *out)

#define scatter_doubles(n,copy) \
static void scatter_double##n scatter_args \
{ \
  const double *id=(const double *)in; \
  for(int i=0;i<nVal;i++) { \
      double *targ = (double *)(out+nodeIdx[i]*nodeScale); \
      copy \
      id+=n; \
  } \
}

scatter_doubles(1,targ[0]+=id[0];)
scatter_doubles(2,targ[0]+=id[0];targ[1]+=id[1];)
scatter_doubles(3,targ[0]+=id[0];targ[1]+=id[1];targ[2]+=id[2];)

//Update my shared nodes based on these values
void
FEMchunk::update_field(FEM_DataMsg *msg)
{
  int i;
  const DType &dt=dtypes[msg->dtype];
  int length=dt.length();
  char *fStart=(char *)curbuf;
  fStart+=dt.init_offset;
  const char *data = (const char *)msg->data;
  int from = gPeToIdx[msg->from];
  int num = comm.numNodesPerPe[from];
  const int *nodeIdx=comm.nodesPerPe[from];
#if 1
  /*First try for an accellerated version*/
  if (dt.base_type==FEM_DOUBLE) {
    switch(dt.vec_len) {
    case 1: scatter_double1(num,nodeIdx,dt.distance,data,fStart); return;
    case 2: scatter_double2(num,nodeIdx,dt.distance,data,fStart); return;
    case 3: scatter_double3(num,nodeIdx,dt.distance,data,fStart); return;
    }    
  }
#endif

  /*Otherwise we need the slow, general version*/
  combineFn fn=combine(dtypes[msg->dtype],FEM_SUM);
  for(i=0;i<num;i++) {
    void *cnode = (void*) (fStart+nodeIdx[i]*dt.distance);
    fn(dt.vec_len,cnode, data);
    data +=length;
  }
}


void
FEMchunk::update(int fid, void *nodes)
{
  // first send my field values to all the processors that need it
  seqnum++;
  send(fid, nodes);
  curbuf = nodes;
  nRecd = 0;
  // now, if any of the field values have been received already,
  // process them
  FEM_DataMsg *dm;
  while ((dm = (FEM_DataMsg*)CmmGet(messages, 1, &seqnum, 0))!=NULL) {
    update_field(dm);
    delete dm;
    nRecd++;
  }
  // if any field values are still required, put myself to sleep
  if (nRecd != comm.nPes) {
    wait_for = seqnum;
    thread->suspend();
  }
}

void
FEMchunk::reduce_field(int fid, const void *nodes, void *outbuf, int op)
{
  // first reduce over local nodes
  const DType &dt = dtypes[fid];
  const void *src = (const void *) ((const char *) nodes + dt.init_offset);
  initialize(dt,outbuf,op);
  combineFn fn=combine(dt,op);
  for(int i=0; i<m.node.n; i++) {
    if(isPrimary[i]) {
      fn(dt.vec_len,outbuf, src);
    }
    src = (const void *)((const char *)src + dt.distance);
  }
  // and now reduce over partitions
  reduce(fid, outbuf, outbuf, op);
}

void
FEMchunk::reduce(int fid, const void *inbuf, void *outbuf, int op)
{
  int len = dtypes[fid].length();
  if(numElements==1) {
    memcpy(outbuf,inbuf,len);
    return;
  }
  CkReduction::reducerType rtype;
  switch(op) {
    case FEM_SUM:
      switch(dtypes[fid].base_type) {
        case FEM_INT: rtype = CkReduction::sum_int; break;
        case FEM_REAL: rtype = CkReduction::sum_float; break;
        case FEM_DOUBLE: rtype = CkReduction::sum_double; break;
      }
      break;
    case FEM_MAX:
      switch(dtypes[fid].base_type) {
        case FEM_INT: rtype = CkReduction::max_int; break;
        case FEM_REAL: rtype = CkReduction::max_float; break;
        case FEM_DOUBLE: rtype = CkReduction::max_double; break;
      }
      break;
    case FEM_MIN:
      switch(dtypes[fid].base_type) {
        case FEM_INT: rtype = CkReduction::min_int; break;
        case FEM_REAL: rtype = CkReduction::min_float; break;
        case FEM_DOUBLE: rtype = CkReduction::min_double; break;
      }
      break;
  }
  contribute(len, (void *)inbuf, rtype);
  curbuf = outbuf;
  thread->suspend();
}

void
FEMchunk::reductionResult(FEM_DataMsg *msg)
{
  //msg->from used as length
  memcpy(curbuf, msg->data, msg->from);
  thread->resume();
  delete msg;
}

//Called by user to ask us to contribute our updated mesh chunk
void 
FEMchunk::updateMesh(int callMeshUpdated,int doRepartition) {
  if (updated_mesh==NULL)
    CkAbort("FEM_Update_Mesh> You must first set the mesh before updating it!\n");
  updated_mesh->updateCount=updateCount++;
  updated_mesh->fromChunk=thisIndex;
  updated_mesh->callMeshUpdated=callMeshUpdated;
  updated_mesh->doRepartition=doRepartition;

  int t,i;
  int newElemTot=updated_mesh->m.nElems();
  int newNode=updated_mesh->m.node.n;
  int oldNode=m.node.n;
  updated_mesh->elemNums=new int[newElemTot];
  updated_mesh->nodeNums=new int[newNode];
  updated_mesh->isPrimary=new int[newNode];

  //Copy over the old global node numbers, and fabricate the rest  
  int comNode=oldNode; if (comNode>newNode) comNode=newNode;
  memcpy(updated_mesh->nodeNums,nodeNums,comNode*sizeof(int));
  memcpy(updated_mesh->isPrimary,isPrimary,comNode*sizeof(int));
  for (i=comNode;i<newNode;i++) {
    updated_mesh->nodeNums[i]=-1;//New nodes have no global number
    updated_mesh->isPrimary[i]=1;//New nodes are not shared
  }

  //Copy over the old global element numbers, and fabricate the rest
  i=0;
  for (t=0; t<m.nElemTypes && t<updated_mesh->m.nElemTypes ;t++) {
    int oldElemStart=m.nElems(t);
    int newElemStart=updated_mesh->m.nElems(t);
    int oldElems=m.elem[t].n;
    int newElems=updated_mesh->m.elem[t].n;
    int comElems=oldElems;
    if (comElems>newElems) comElems=newElems;
    memcpy(&updated_mesh->elemNums[newElemStart],
	   &elemNums[oldElemStart],comElems*sizeof(int));
    for (i=newElemStart+comElems;i<newElemStart+newElems;i++)
      updated_mesh->elemNums[i]=-1;//New elements have no global number
  }
  for (;i<newElemTot;i++)
    updated_mesh->elemNums[i]=-1;//New element types have no global number

  //Send the mesh off to the coordinator
  CProxy_FEMcoordinator coord(init.coordinator);
  coord.updateMesh(updated_mesh);
  updated_mesh=NULL;
  if (doRepartition)
    thread->suspend();//Sleep until repartitioned mesh arrives
}

//Called by coordinator with a new, repartitioned mesh chunk for us
void 
FEMchunk::meshUpdated(ChunkMsg *newMesh) {
  deallocate(); //Destroy the old mesh
  readChunk(newMesh); //Read in the new mesh
  thread->resume();  //Start computing again
}

void
FEMchunk::readField(int fid, void *nodes, const char *fname)
{
  int btype = dtypes[fid].base_type;
  int typelen = dtypes[fid].vec_len;
  int btypelen = dtypes[fid].length()/typelen;
  char *data = (char *)nodes + dtypes[fid].init_offset;
  int distance = dtypes[fid].distance;
  FILE *fp = fopen(fname, "r");
  if(fp==0) {
    CkError("Cannot open file %s for reading.\n", fname);
    CkAbort("Exiting");
  }
  char str[80];
  char* pos;
  const char* fmt;
  int i, j, curline;
#if FEM_FORTRAN
  curline = 1;
#else
  curline = 0;
#endif
  switch(btype) {
    case FEM_INT: fmt = "%d%n"; break;
    case FEM_REAL: fmt = "%f%n"; break;
    case FEM_DOUBLE: fmt = "%lf%n"; break;
  }
  for(i=0;i<m.node.n;i++) {
    // skip lines to the next local node
    for(j=curline;j<nodeNums[i];j++)
      fgets(str,80,fp);
    curline = nodeNums[i]+1;
    fgets(str,80,fp);
    int curnode, numchars;
    sscanf(str,"%d%n",&curnode,&numchars);
    pos = str + numchars;
    if(curnode != nodeNums[i]) {
      CkError("Expecting info for node %d, got %d\n", nodeNums[i], curnode);
      CkAbort("Exiting");
    }
    for(j=0;j<typelen;j++) {
      sscanf(pos, fmt, data+(j*btypelen), &numchars);
      pos += numchars;
    }
    data += distance;
  }
  fclose(fp);
}

//-------------------- chunk I/O ---------------------
void FEMchunk::writeNodes(FILE *fp) const
{
    fprintf(fp, "%d %d\n", m.node.n,m.node.dataPer);
    for(int i=0;i<m.node.n;i++) {
      fprintf(fp, "%d %d ", nodeNums[i], isPrimary[i]);
      for(int d=0;d<m.node.dataPer;d++)
	fprintf(fp, "%lf ", m.node.udata[i*m.node.dataPer+d]);
      fprintf(fp,"\n");
    }
}

void FEMchunk::readNodes(FILE *fp)
{
    fscanf(fp, "%d%d", &m.node.n,&m.node.dataPer);
    nodeNums = new int[m.node.n]; CHK(nodeNums);
    isPrimary = new int[m.node.n]; CHK(isPrimary);
    m.node.allocUdata();
    for(int i=0;i<m.node.n;i++) {
      fscanf(fp, "%d%d", &nodeNums[i], &isPrimary[i]);
      for(int d=0;d<m.node.dataPer;d++)
	fscanf(fp, "%lf", &m.node.udata[i*m.node.dataPer+d]);
    }
}

void FEMchunk::writeElems(FILE *fp) const
{
    fprintf(fp,"%d\n",m.nElemTypes);
    int t;
    for (t=0;t<m.nElemTypes;t++) {
      fprintf(fp, "%d %d %d\n", m.elem[t].n, m.elem[t].nodesPer,m.elem[t].dataPer);
    }
    for (t=0;t<m.nElemTypes;t++) {
      int start=m.nElems(t);
      for(int i=0; i<m.elem[t].n; i++) {
        fprintf(fp, "%d ", elemNums[start+i]);
        for(int j=0;j<m.elem[t].nodesPer;j++)
          fprintf(fp, "%d ", m.elem[t].conn[i*m.elem[t].nodesPer+j]);
        for(int d=0;d<m.elem[t].dataPer;d++)
          fprintf(fp, "%lf ", m.elem[t].udata[i*m.elem[t].dataPer+d]);
	fprintf(fp,"\n");
      }
    }
}

void FEMchunk::readElems(FILE *fp)
{
    fscanf(fp,"%d",&m.nElemTypes);
    int t;
    for (t=0;t<m.nElemTypes;t++) {
      fscanf(fp, "%d%d%d", &m.elem[t].n, &m.elem[t].nodesPer,&m.elem[t].dataPer);
    }
    elemNums = new int[m.nElems()]; CHK(elemNums);
    for (t=0;t<m.nElemTypes;t++) {
      m.elem[t].allocUdata();
      m.elem[t].allocConn();
      int start=m.nElems(t);
      for(int i=0; i<m.elem[t].n; i++) {
        fscanf(fp, "%d", &elemNums[start+i]);
        for(int j=0;j<m.elem[t].nodesPer;j++)
          fscanf(fp, "%d", &m.elem[t].conn[i*m.elem[t].nodesPer+j]);
        for(int d=0;d<m.elem[t].dataPer;d++)
          fscanf(fp, "%lf", &m.elem[t].udata[i*m.elem[t].dataPer+d]);
      }
    }
}

void FEMchunk::writeComm(FILE *fp) const
{
    fprintf(fp, "%d\n", comm.nPes);
    for(int p=0;p<comm.nPes;p++) {
      fprintf(fp, "%d %d\n", comm.peNums[p], comm.numNodesPerPe[p]);
      for(int j=0;j<comm.numNodesPerPe[p];j++) {
        fprintf(fp, "%d ", comm.nodesPerPe[p][j]);
      }
      fprintf(fp,"\n");
    }
}
void FEMchunk::readComm(FILE *fp)
{
    fscanf(fp, "%d", &comm.nPes);
    comm.allocate();
    for(int p=0;p<comm.nPes;p++) {
      fscanf(fp, "%d%d", &comm.peNums[p], &comm.numNodesPerPe[p]);
      comm.nodesPerPe[p] = new int[comm.numNodesPerPe[p]];
      for(int j=0;j<comm.numNodesPerPe[p];j++) {
        fscanf(fp, "%d", &comm.nodesPerPe[p][j]);
      }
    }
}

static const char *meshFileNames="meshdata.pe%d";

void FEMchunk::writeChunk(void)
{
    char fname[256];
    sprintf(fname, meshFileNames, thisIndex);
    FILE *fp = fopen(fname, "w");
    if(fp==0) {
      CkAbort("FEM: unable to open output file.\n");
    }
    CkPrintf("FEM> Writing %s...\n",fname);
    writeNodes(fp);
    writeElems(fp);
    writeComm(fp);
    fclose(fp);
}

void
FEMchunk::readChunk(ChunkMsg *msg)
{
  if(msg==0) { /*Read mesh from file*/
    char fname[256];
    sprintf(fname, meshFileNames, thisIndex);
    FILE *fp = fopen(fname, "r");
    if(fp==0) {
      CkAbort("FEM: unable to open input file.\n");
    }
    CkPrintf("FEM> Reading %s...\n",fname);    
    readNodes(fp);
    readElems(fp);
    readComm(fp);
    fclose(fp);
  } else {
    //Just copy pointers right out of the message--
    // this saves pointless allocation and copying, but you *can't*
    // delete the message after this!
	stored_mesh=msg;
    m=msg->m; //Copy over the FEM_Mesh (incl. user data; connectivity)
    comm=msg->comm; //<- copies pointers, too
    elemNums=msg->elemNums;
    nodeNums=msg->nodeNums;
    isPrimary=msg->isPrimary;
  }
  
//Initialize global Pe to local Pe mapping table
  gPeToIdx = new int[numElements]; CHK(gPeToIdx);
  for(int p=0;p<numElements;p++)
    gPeToIdx[p] = (-1);
  for(int i=0;i<comm.nPes;i++)
    gPeToIdx[comm.peNums[i]] = i;

}

/******************************* C Bindings **********************************/
static FEMchunk *getCurChunk(void) 
{
  FEMchunk *cptr=CtvAccess(_femptr);
  if (cptr==NULL) 
    CkAbort("Routine can only be called from driver()!\n");
  return cptr;
}

CDECL void FEM_Update_Mesh(int callMeshUpdated,int doRepartition) 
{ 
  getCurChunk()->updateMesh(callMeshUpdated,doRepartition); 
}

CDECL int FEM_Register(void *_ud,FEM_PupFn _pup_ud)
{
  return TCharmRegister(_ud,_pup_ud);
}

CDECL void *FEM_Get_Userdata(int n)
{
  return TCharmGetUserdata(n);
}

CDECL void
FEM_Migrate(void)
{
  TCharmMigrate();
}

CDECL int *
FEM_Get_Node_Nums(void)
{
  return getCurChunk()->get_nodenums();
}

CDECL int *
FEM_Get_Elem_Nums(void)
{
  return getCurChunk()->get_elemnums();
}

CDECL int *
FEM_Get_Conn(int elemType)
{
  return getCurChunk()->m.elem[elemType].conn;
}

CDECL void 
FEM_Done(void)
{
  TCharmDone();
}

CDECL int 
FEM_Create_Field(int base_type, int vec_len, int init_offset, int distance)
{
  return getCurChunk()->new_DT(base_type, vec_len, init_offset, distance);
}

CDECL void
FEM_Update_Field(int fid, void *nodes)
{
  getCurChunk()->update(fid, nodes);
}

CDECL void
FEM_Reduce_Field(int fid, const void *nodes, void *outbuf, int op)
{
  getCurChunk()->reduce_field(fid, nodes, outbuf, op);
}

CDECL void
FEM_Reduce(int fid, const void *inbuf, void *outbuf, int op)
{
    getCurChunk()->reduce(fid, inbuf, outbuf, op);
}

CDECL void
FEM_Read_Field(int fid, void *nodes, const char *fname)
{
  getCurChunk()->readField(fid, nodes, fname);
}

CDECL int
FEM_My_Partition(void)
{
  return TCharmElement();
}

CDECL int
FEM_Num_Partitions(void)
{
  return TCharmNumElements();
}

CDECL double
FEM_Timer(void)
{
  return CkTimer();
}

CDECL void 
FEM_Print(const char *str)
{
  if(TCharm::getState()==inDriver) {
    FEMchunk *cptr = getCurChunk();
    CkPrintf("[%d] %s\n", cptr->thisIndex, str);
  } else {
    CkPrintf("%s\n", str);
  }
}

CDECL void 
FEM_Print_Partition(void)
{
  if(TCharm::getState()==inDriver) {
    FEMchunk *cptr = getCurChunk();
    cptr->print();
  } else {
    if (_meshptr==NULL)
      CkPrintf("[%d] No serial mesh available.\n",FEM_My_Partition());
    else
      _meshptr->print(l2g_t());
  }
}

CDECL int FEM_Get_Comm_Partners(void)
{
	return getCurChunk()->getComm().nPes;
}
CDECL int FEM_Get_Comm_Partner(int partnerNo)
{
	return getCurChunk()->getComm().peNums[partnerNo];
}
CDECL int FEM_Get_Comm_Count(int partnerNo)
{
	return getCurChunk()->getComm().numNodesPerPe[partnerNo];
}
CDECL void FEM_Get_Comm_Nodes(int partnerNo,int *nodeNos)
{
	const int *nNo=getCurChunk()->getComm().nodesPerPe[partnerNo];
	int len=FEM_Get_Comm_Count(partnerNo);
	for (int i=0;i<len;i++)
		nodeNos[i]=nNo[i];
}

/************************ Fortran Bindings *********************************/
FDECL void FTN_NAME(FEM_UPDATE_MESH,fem_update_mesh)
  (int *callMesh, int *repart)
{
  FEM_Update_Mesh(*callMesh,*repart);
}

FDECL int FTN_NAME(FEM_REGISTER,fem_register)
  (void *userData,FEM_PupFn _pup_ud)
{ 
  return FEM_Register(userData,_pup_ud);
}

FDECL void FTN_NAME(FEM_MIGRATE,fem_migrate)
  (void)
{
  FEM_Migrate();
}

FDECL int FTN_NAME(FEM_CREATE_FIELD,fem_create_field)
  (int *bt, int *vl, int *io, int *d)
{
  return FEM_Create_Field(*bt, *vl, *io, *d);
}

FDECL void FTN_NAME(FEM_UPDATE_FIELD,fem_update_field)
  (int *fid, void *nodes)
{
  FEM_Update_Field(*fid, nodes);
}

FDECL void  FTN_NAME(FEM_REDUCE_FIELD,fem_reduce_field)
  (int *fid, void *nodes, void *outbuf, int *op)
{
  FEM_Reduce_Field(*fid, nodes, outbuf, *op);
}

FDECL void FTN_NAME(FEM_REDUCE,fem_reduce)
  (int *fid, void *inbuf, void *outbuf, int *op)
{
  FEM_Reduce(*fid, inbuf, outbuf, *op);
}

FDECL void FTN_NAME(FEM_READ_FIELD,fem_read_field)
  (int *fid, void *nodes, char *fname, int len)
{
  char *tmp = new char[len+1]; CHK(tmp);
  memcpy(tmp, fname, len);
  tmp[len] = '\0';
  FEM_Read_Field(*fid, nodes, tmp);
  delete[] tmp;
}

FDECL int FTN_NAME(FEM_MY_PARTITION,fem_my_partition)
  (void)
{
  return FEM_My_Partition()+1;
}

FDECL int FTN_NAME(FEM_NUM_PARTITIONS,fem_num_partitions)
  (void)
{
  return FEM_Num_Partitions();
}

FDECL double FTN_NAME(FEM_TIMER_,fem_timer)
  (void)
{
  return CkTimer();
}

// Utility functions for Fortran

FDECL int FTN_NAME(OFFSETOF,offsetof)
  (void *first, void *second)
{
  return (int)((char *)second - (char*)first);
}

FDECL void FTN_NAME(FEM_PRINT,fem_print)
  (char *str, int len)
{
  char *tmpstr = new char[len+1]; CHK(tmpstr);
  memcpy(tmpstr,str,len);
  tmpstr[len] = '\0';
  FEM_Print(tmpstr);
  delete[] tmpstr;
}

FDECL void FTN_NAME(FEM_PRINT_PARTITION,fem_print_partition)
  (void)
{
  FEM_Print_Partition();
}

FDECL void FTN_NAME(FEM_DONE,fem_done)
  (void)
{
  FEM_Done();
}

FDECL int FTN_NAME(FEM_GET_COMM_PARTNERS,fem_get_comm_partners)
	(void)
{
	return FEM_Get_Comm_Partners();
}
FDECL int FTN_NAME(FEM_GET_COMM_PARTNER,fem_get_comm_partner)
	(int *partnerNo)
{
	return FEM_Get_Comm_Partner(*partnerNo-1)+1;
}
FDECL int FTN_NAME(FEM_GET_COMM_COUNT,fem_get_comm_count)
	(int *partnerNo)
{
	return FEM_Get_Comm_Count(*partnerNo-1);
}
FDECL void FTN_NAME(FEM_GET_COMM_NODES,fem_get_comm_nodes)
	(int *pNo,int *nodeNos)
{
	int partnerNo=*pNo-1;
	const int *nNo=getCurChunk()->getComm().nodesPerPe[partnerNo];
	int len=FEM_Get_Comm_Count(partnerNo);
	for (int i=0;i<len;i++)
		nodeNos[i]=nNo[i]+1;
}

FDECL void FTN_NAME(FEM_GET_ELEM_NUMBERS,fem_get_elem_numbers)
	(int *gNo)
{
	const int *no=getCurChunk()->get_elemnums();
	int n=getMesh()->nElems();
	for (int i=0;i<n;i++) gNo[i]=no[i]+1;
}
FDECL void FTN_NAME(FEM_GET_NODE_NUMBERS,fem_get_node_numbers)
	(int *gNo)
{
	const int *no=getCurChunk()->get_nodenums();
	int n=getMesh()->node.n;
	for (int i=0;i<n;i++) gNo[i]=no[i]+1;
}


/********* Debugging mesh printouts *******/
# define ARRSTART 0

void FEM_Mesh::count::print(const char *type,const l2g_t &l2g)
{
  CkPrintf("Number of %ss = %d\n", type, n);
  CkPrintf("User data doubles per %s = %d\n", type, dataPer);
  if (dataPer!=0)
    for (int i=0;i<n;i++) {
      CkPrintf("\t%s[%d] userdata:",type,l2g.no(i)+ARRSTART);
      for (int j=0;j<dataPer;j++)
      	CkPrintf("\t%f",udata[i*dataPer+j]);
      CkPrintf("\n");
    }
}

void FEM_Mesh::elemCount::print(const char *type,const l2g_t &l2g)
{
  CkPrintf("Number of %ss = %d\n", type, n);
  CkPrintf("User data doubles per %s = %d\n", type, dataPer);
  if (dataPer!=0)
    for (int i=0;i<n;i++) {
      CkPrintf("\t%s[%d] userdata:",type,l2g.el(i)+ARRSTART);
      for (int j=0;j<dataPer;j++)
      	CkPrintf("\t%f",udata[i*dataPer+j]);
      CkPrintf("\n");
    }
  CkPrintf("Nodes per %s = %d\n", type, nodesPer);
    for (int i=0;i<n;i++) {
      CkPrintf("\t%s[%d] nodes:",type,l2g.el(i)+ARRSTART);
      for (int j=0;j<nodesPer;j++)
      	CkPrintf("\t%d",l2g.no(conn[i*nodesPer+j])+ARRSTART);
      CkPrintf("\n");
    }	
}

void FEM_Mesh::print(const l2g_t &l2g)
{
  node.print("node",l2g);
  CkPrintf("%d kinds of element:\n",nElemTypes);
  if (nElemTypes==1)
	elem[0].print("element",l2g);
  else 
  	for (int elType=0;elType<nElemTypes;elType++) {
  		char elName[50];
  		sprintf(elName,"El type %d",elType+ARRSTART);
  		elem[elType].print(elName,l2g);
  	}
}

void commCounts::print(const l2g_t &l2g)
{
  CkPrintf("We communicate with %d other chunks:\n",nPes);
  for (int p=0;p<nPes;p++) {
    CkPrintf("  With chunk %d, we share %d nodes:\n",peNums[p],numNodesPerPe[p]);
    for (int n=0;n<numNodesPerPe[p];n++)
      CkPrintf("\t%d",l2g.no(nodesPerPe[p][n]));
    CkPrintf("\n");
  }
}

class l2g_arr:public l2g_t {
public:
	const int *elMap,*noMap;
	l2g_arr(const int *NelMap,const int *NnoMap)
	  {elMap=NelMap;noMap=NnoMap;}
	//Return the global number associated with this local element
	virtual int el(int localNo) const {return elMap[localNo];}
	//Return the global number associated with this local node
	virtual int no(int localNo) const {return noMap[localNo];}
};

void
FEMchunk::print(void)
{
  int i;
  CkPrintf("-------------------- Chunk %d --------------------\n",thisIndex);
  m.print(l2g_arr(elemNums,nodeNums));
  comm.print(l2g_arr(elemNums,nodeNums));
  CkPrintf("[%d] Element global numbers:\n", thisIndex);
  for(i=0;i<m.nElems();i++) 
    CkPrintf("%d\t",elemNums[i]);
  CkPrintf("\n[%d] Node global numbers (* is primary):\n",thisIndex);
  for(i=0;i<m.node.n;i++) 
    CkPrintf("%d%s\t",nodeNums[i],isPrimary[i]?"*":"");
  CkPrintf("\n\n");
}  
  

/***************** Mesh-Sending and Packing Utilities **********/
void FEM_Mesh::count::pup(PUP::er &p) {
	p(n);p(dataPer);
	if (dataPer!=0) {
		if (udata==NULL) allocUdata();
		p(udata,udataCount());
	}
}
void FEM_Mesh::elemCount::pup(PUP::er &p) {
	count::pup(p);
	p(nodesPer);
	if (conn==NULL) allocConn();
	p(conn,connCount());
}
void FEM_Mesh::copyType(const FEM_Mesh &from)//Copies nElemTypes and *Per fields
{
	nElemTypes=from.nElemTypes;
	node.dataPer=from.node.dataPer;
	for (int t=0;t<nElemTypes;t++) {
		elem[t].dataPer=from.elem[t].dataPer;
		elem[t].nodesPer=from.elem[t].nodesPer;
	}
}

int FEM_Mesh::nElems(int t_max) const //Return total number of elements before type t_max
{
	int ret=0;
	for (int t=0;t<t_max;t++) ret+=elem[t].n;
	return ret;
}
void FEM_Mesh::deallocate(void) { //Free all stored memory
	node.deallocate();
	for (int t=0;t<nElemTypes;t++) elem[t].deallocate();
}
int FEM_Mesh::size() const //Return total array storage size, in bytes
{
	 int ret=node.size();
	 for (int t=0;t<nElemTypes;t++)
		  ret+=elem[t].size();
	 return ret;
}
void FEM_Mesh::pup(PUP::er &p)  //For migration
{
	node.pup(p);
	p(nElemTypes);
	for (int t=0;t<nElemTypes;t++)
		elem[t].pup(p);
}

int commCounts::sharedNodes() const { //Return total number of shared nodes
	int ret=0;
	for (int p=0;p<nPes;p++) ret+=numNodesPerPe[p];
	return ret;
}
void commCounts::allocate(void) {
	peNums = new int[nPes]; CHK(peNums);
	numNodesPerPe = new int[nPes]; CHK(numNodesPerPe);
	nodesPerPe = new int*[nPes]; CHK(nodesPerPe);
}
void commCounts::deallocate(void) { //Free all stored memory
	delete [] peNums;peNums=NULL;
	delete [] numNodesPerPe;numNodesPerPe=NULL;
	if (nodesPerPe!=NULL) {
        	for (int p=0;p<nPes;p++)
	        	delete [] nodesPerPe[p];
		delete [] nodesPerPe;
		nodesPerPe=NULL;
	}
}
void commCounts::pup(PUP::er &p)  //For migration
{
	p(nPes);
	if (p.isUnpacking()) 
		allocate();
	p(peNums, nPes);
	p(numNodesPerPe,nPes);
	for(int i=0;i<nPes;i++)
	{
		if(p.isUnpacking())
			nodesPerPe[i] = new int[numNodesPerPe[i]];
		p(nodesPerPe[i], numNodesPerPe[i]);
	}
}
int commCounts::size() const //Return total array storage size, in bytes
{
	 int ret=2*sizeof(int)*nPes;
	 for (int i=0;i<nPes;i++)
		  ret+=sizeof(int)*numNodesPerPe[i];
	 return ret;
}

void ChunkMsg::deallocate(void) { //Free all stored memory
	if (isPacked) return; //Delete will free memory
	m.deallocate();
	comm.deallocate();
	delete [] elemNums;
	delete [] nodeNums;
	delete [] isPrimary;
}

void ChunkMsg::pup(PUP::er &p) { //For send/recv
	m.pup(p);
	comm.pup(p);
	if(p.isUnpacking()) {
		elemNums=new int[m.nElems()];
		nodeNums=new int[m.node.n];
		isPrimary=new int[m.node.n];
	}
	p(elemNums,m.nElems());
	p(nodeNums,m.node.n);
	p(isPrimary,m.node.n);
}

void
FEMchunk::pup(PUP::er &p)
{
//Pup superclass
  ArrayElement1D::pup(p);

  messages=CmmPup(&p,messages);

// Pup the mesh fields
  init.pup(p);
  m.pup(p);
  comm.pup(p);
  if(p.isUnpacking())
  {
    elemNums = new int[m.nElems()];
    nodeNums = new int[m.node.n];
    isPrimary = new int[m.node.n];
    gPeToIdx = new int[numElements];
  }
  p(elemNums, m.nElems());
  p(nodeNums, m.node.n);
  p(isPrimary, m.node.n);
  p(gPeToIdx, numElements);

//Pup all other fields
  p(ntypes);
  p((void*)dtypes, MAXDT*sizeof(DType));
  p(wait_for);

  p(seqnum);
  p(nRecd);
  // update should not be in progress when migrating, so curbuf is not valid
  p(doneCalled);
  // fp is not valid, because it has been closed a long time ago
}


int ChunkMsg::size() const //Return total array storage size, in bytes
{
	 return sizeof(ChunkMsg)+sizeof(int)*(m.nElems()+2*m.node.n)+
	   m.size()+comm.size();
}

#define PACK(buf,sz) do { memcpy(pos,(buf),(sz)); pos += (sz); } while(0)
void *
ChunkMsg::pack(ChunkMsg *c)
{
  int totalsize = c->size()+sizeof(double);
  void *msg = CkAllocBuffer(c, totalsize); CHK(msg);
  char *pos = (char *) msg;
  
  //Pack all non-array data
  PACK(c, sizeof(ChunkMsg));

  //Handle all integer arrays
  PACK(c->elemNums,c->m.nElems()*sizeof(int));
  PACK(c->nodeNums,c->m.node.n*sizeof(int));
  PACK(c->isPrimary,c->m.node.n*sizeof(int));
  PACK(c->comm.peNums,c->comm.nPes*sizeof(int));
  PACK(c->comm.numNodesPerPe,c->comm.nPes*sizeof(int));
  for (int i=0;i<c->comm.nPes;i++)
    PACK(c->comm.nodesPerPe[i],c->comm.numNodesPerPe[i]*sizeof(int));
  int t;
  for (t=0;t<c->m.nElemTypes;t++)
        PACK(c->m.elem[t].conn,c->m.elem[t].connCount()*sizeof(int));

  //Handle double arrays (user data)
  int off=pos-(char *)msg;
  off=(off+sizeof(double)-1)&(~(sizeof(double)-1));//Round up to double boundary
  pos=(char *)msg+off;
  
  PACK(c->m.node.udata,c->m.node.udataCount()*sizeof(double));
  for (t=0;t<c->m.nElemTypes;t++)
    PACK(c->m.elem[t].udata,c->m.elem[t].udataCount()*sizeof(double));

  //Blow away old arrays  
  c->deallocate();
  return msg;
}
#undef PACK

#define UNPACK(buf,sz) do { buf = (int *) pos; pos += (sz); } while(0)
#define UNPACK_D(buf,sz) do { buf = (double *) pos; pos += (sz); } while(0)

ChunkMsg *
ChunkMsg::unpack(void *msg)
{
  ChunkMsg* c = (ChunkMsg *)msg;
  char *pos = (char *)msg + sizeof(ChunkMsg);
  c->isPacked=1;
  //Handle all integer arrays
  UNPACK(c->elemNums,c->m.nElems()*sizeof(int));
  UNPACK(c->nodeNums,c->m.node.n*sizeof(int));
  UNPACK(c->isPrimary,c->m.node.n*sizeof(int));
  UNPACK(c->comm.peNums,c->comm.nPes*sizeof(int));
  UNPACK(c->comm.numNodesPerPe,c->comm.nPes*sizeof(int));
  c->comm.nodesPerPe=new int*[c->comm.nPes];
  for (int i=0;i<c->comm.nPes;i++)
    UNPACK(c->comm.nodesPerPe[i],c->comm.numNodesPerPe[i]*sizeof(int));
  int t;
  for (t=0;t<c->m.nElemTypes;t++)
        UNPACK(c->m.elem[t].conn,c->m.elem[t].connCount()*sizeof(int));

  //Handle double arrays (user data)
  int off=pos-(char *)msg;
  off=(off+sizeof(double)-1)&(~(sizeof(double)-1));//Round up to double boundary
  pos=(char *)msg+off;
  
  UNPACK_D(c->m.node.udata,c->m.node.udataCount()*sizeof(double));
  for (t=0;t<c->m.nElemTypes;t++)
    UNPACK_D(c->m.elem[t].udata,c->m.elem[t].udataCount()*sizeof(double));
  
  return c;
}
#undef UNPACK
#undef UNPACK_D

void *
FEM_DataMsg::pack(FEM_DataMsg *in)
{
  return (void*) in;
}

FEM_DataMsg *
FEM_DataMsg::unpack(void *in)
{
  return new (in) FEM_DataMsg;
}

void *
FEM_DataMsg::alloc(int mnum, size_t size, int *sizes, int pbits)
{
  return CkAllocMsg(mnum, size+sizes[0], pbits);
}


#include "fem.def.h"
