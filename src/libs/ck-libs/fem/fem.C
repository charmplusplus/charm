/*
Finite Element Method (FEM) Framework for Charm++
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

/* TCharm semaphore ID, used for mesh startup */
#define FEM_TCHARM_SEMAID 0x00FE300 /* __FEM__ */

void FEM_Abort(const char *msg) {
	CkAbort(msg);
}
void FEM_Abort(const char *caller,const char *sprintf_msg,
   int int0,int int1, int int2) 
{
	char userSprintf[1024];
	char msg[1024];
	sprintf(userSprintf,sprintf_msg,int0,int1,int2);
	sprintf(msg,"FEM Routine %s fatal error:\n          %s",
		caller,userSprintf);
	FEM_Abort(msg);
}

/*******************************************************
  Communication tools
*/

#define checkMPI(err) checkMPIerr(err,__FILE__,__LINE__);
inline void checkMPIerr(int mpi_err,const char *file,int line) {
	if (mpi_err!=MPI_SUCCESS) {
		CkError("MPI Routine returned error %d at %s:%d\n",
			mpi_err,file,line);
		CkAbort("MPI Routine returned error code");
	}
}

/// Return the number of dt's in the next message from/tag/comm
int myMPI_Incoming(MPI_Datatype dt,int from,int tag,MPI_Comm comm) {
	MPI_Status sts;
	checkMPI(MPI_Probe(from,tag,comm,&sts));
	int len; checkMPI(MPI_Get_count(&sts,dt,&len));
	return len;
}

/// MPI_Recv, but using a T with a pup routine
template <class T>
inline void MPI_Recv_pup(T &t, int from,int tag,MPI_Comm comm) {
	int len=myMPI_Incoming(MPI_BYTE,from,tag,comm);
	MPI_Status sts;
	char *buf=new char[len];
	checkMPI(MPI_Recv(buf,len,MPI_BYTE, from,tag,comm,&sts));
	PUP::fromMemBuf(t,buf,len);
	delete[] buf;
}

/// MPI_Send, but using a T with a pup routine
template <class T>
inline void MPI_Send_pup(T &t, int to,int tag,MPI_Comm comm) {
	int len=PUP::size(t); char *buf=new char[len];
	PUP::toMemBuf(t,buf,len);
	checkMPI(MPI_Send(buf,len,MPI_BYTE, to,tag,comm));
	delete[] buf;
}

/******** Startup and initialization *******/

// This is our TCharm global ID:
enum {FEM_globalID=33};

CDECL void pupFEM_Chunk(pup_er cp) {
	PUP::er &p=*(PUP::er *)cp;
	FEMchunk *c=(FEMchunk *)TCHARM_Get_global(FEM_globalID);
	if (c==NULL) {
		c=new FEMchunk((CkMigrateMessage *)0);
		TCHARM_Set_global(FEM_globalID,c,pupFEM_Chunk);
	}
	c->pup(p);
	if (p.isDeleting())
		delete c;
}
FEMchunk *FEMchunk::get(const char *caller) {
	FEMchunk *c=(FEMchunk *)TCHARM_Get_global(FEM_globalID);
	if(!c) FEM_Abort(caller,"FEM is not initialized");
	return c;
}

CDECL void FEM_Init(FEM_Comm_t defaultComm)
{
	IDXL_Init(defaultComm);
	if (!TCHARM_Get_global(FEM_globalID)) {
		FEMchunk *c=new FEMchunk(defaultComm);
		TCHARM_Set_global(FEM_globalID,c,pupFEM_Chunk);
	}
}
FORTRAN_AS_C(FEM_INIT,FEM_Init,fem_init, (int *comm), (*comm))

// This lets FEM be a "-module", too.  (Normally comes from .ci file...)
void _registerfem(void) {}

/*******************************************************
  Mesh basics
*/
void FEM_Mesh_list::bad(int l,int bad_code,const char *caller) const {
	if (bad_code==0)
		FEM_Abort(caller,"Invalid FEM Mesh ID %d (should be like %d)",
			l,FEM_MESH_FIRST);
	else /* bad_code==1 */
		FEM_Abort(caller,"Re-used a deleted FEM Mesh ID %d",l);
}

CDECL int 
FEM_Mesh_allocate(void) /* build new mesh */  
{
	const char *caller="FEM_Mesh_allocate";FEMAPI(caller);
	FEMchunk *c=FEMchunk::get(caller);
	FEM_Mesh *m=new FEM_Mesh;
	m->becomeSetting();
	m->registerIDXL(IDXL_Chunk::get(caller));
	return c->meshes.put(m);
}
FORTRAN_AS_C_RETURN(int,
	FEM_MESH_ALLOCATE,FEM_Mesh_allocate,fem_mesh_allocate, (void),())

/// Return a new'd copy of this class, by calling pup.
template <class T>
inline T *clonePointerViaPup(T *old) {
	int len=PUP::size(*old);
	char *buf=new char[len];
	PUP::toMemBuf(*old,buf,len);
	T *nu=new T;
	PUP::fromMemBuf(*nu,buf,len);
	return nu;
}

CDECL int
FEM_Mesh_copy(int fem_mesh) {
	const char *caller="FEM_Mesh_copy";FEMAPI(caller);
	FEMchunk *c=FEMchunk::get(caller);
	return c->meshes.put(clonePointerViaPup(c->meshes.lookup(fem_mesh,caller)));
}
FORTRAN_AS_C_RETURN(int,
	FEM_MESH_COPY,FEM_Mesh_copy,fem_mesh_copy, (int *m),(*m))


CDECL void 
FEM_Mesh_deallocate(int fem_mesh) /* delete this local mesh */
{
	const char *caller="FEM_Mesh_deallocate";FEMAPI(caller);
	FEMchunk *c=FEMchunk::get(caller);
	c->meshes.destroy(fem_mesh,caller);
}
FORTRAN_AS_C(FEM_MESH_DEALLOCATE,FEM_Mesh_deallocate,fem_mesh_deallocate, (int *m),(*m))

/* Mesh I/O */
CDECL int 
FEM_Mesh_read(const char *prefix,int partNo,int nParts) /* read parallel mesh from file */
{
	const char *caller="FEM_Mesh_read";FEMAPI(caller);
	FEMchunk *c=FEMchunk::get(caller);
	return c->meshes.put(FEM_readMesh(prefix,partNo,nParts));
}
FDECL int
FTN_NAME(FEM_MESH_READ,fem_mesh_read)(const char *n,int *partNo,int *nParts,int len) {
	char *s=new char[len+1]; strncpy(s,n,len); s[len]=(char)0;
	int ret=FEM_Mesh_read(s,*partNo,*nParts);
	delete[] s;
	return ret;
}

CDECL void 
FEM_Mesh_write(int fem_mesh,const char *prefix,int partNo,int nParts) /* write parallel mesh to files */
{
	const char *caller="FEM_Mesh_write";FEMAPI(caller);
	FEMchunk *c=FEMchunk::get(caller);
	FEM_writeMesh(c->meshes.lookup(fem_mesh,caller),prefix,partNo,nParts);
}
FDECL void
FTN_NAME(FEM_MESH_WRITE,fem_mesh_write)(int *m,const char *n,int *partNo,int *nParts,int len) {
	char *s=new char[len+1]; strncpy(s,n,len); s[len]=(char)0;
	FEM_Mesh_write(*m,s,*partNo,*nParts);
	delete[] s;
}

/* Mesh assembly/disassembly */
CDECL int 
FEM_Mesh_assemble(int nParts,const int *srcMeshes) {
	const char *caller="FEM_Mesh_assemble";FEMAPI(caller);
	FEMchunk *c=FEMchunk::get(caller);
	FEM_Mesh **chunks=new FEM_Mesh*[nParts];
	for (int p=0;p<nParts;p++) chunks[p]=c->meshes.lookup(srcMeshes[p],caller);
	int ret=c->meshes.put(FEM_Mesh_assemble(nParts,chunks));
	delete[] chunks;
	return ret;
}
FORTRAN_AS_C_RETURN(int,FEM_MESH_ASSEMBLE,FEM_Mesh_assemble,fem_mesh_assemble,
	(int *nParts,const int *src),(*nParts,src))

static FEM_Partition *partition=NULL;
FEM_Partition &FEM_curPartition(void) {
	if (partition==NULL) partition=new FEM_Partition();
	return *partition;
}
static void clearPartition(void) {delete partition; partition=NULL;}

FEM_Partition::FEM_Partition() {
	elem2chunk=NULL;
	sym=NULL; 
}
FEM_Partition::~FEM_Partition() {
	if (elem2chunk) {delete[] elem2chunk;elem2chunk=NULL;}
	for (int i=0;i<getLayers();i++) delete layers[i];
}

void FEM_Partition::setPartition(const int *e, int nElem, int idxBase) {
	if (elem2chunk) {delete[] elem2chunk;elem2chunk=NULL;}
	elem2chunk=CkCopyArray(e,nElem,idxBase);
}
const int *FEM_Partition::getPartition(FEM_Mesh *src,int nChunks) const {
	if (!elem2chunk) { /* Create elem2chunk based on Metis partitioning: */
		int *e=new int[src->nElems()];
		FEM_Mesh_partition(src,nChunks,e);
		((FEM_Partition *)this)->elem2chunk=e;
	}
	return elem2chunk;
}

static void FEM_Mesh_partition(FEM_Mesh *src,int _nchunks,FEM_Mesh_Output *out) {
    src->setAscendingGlobalno();
    FEM_Mesh_split(src,_nchunks,FEM_curPartition(),out);
    clearPartition();
}


class FEM_Mesh_Partition_List : public FEM_Mesh_Output {
	FEMchunk *c;
	int *dest;
public:
	FEM_Mesh_Partition_List(FEMchunk *c_, int *dest_)
		:c(c_), dest(dest_) {}
	
	void accept(int chunkNo,FEM_Mesh *m) {
		dest[chunkNo]=c->meshes.put(m);
	}
};

CDECL void 
FEM_Mesh_partition(int fem_mesh,int nParts,int *destMeshes) {
	const char *caller="FEM_Mesh_partition"; FEMAPI(caller);
	FEMchunk *c=FEMchunk::get(caller);
	FEM_Mesh *m=c->lookup(fem_mesh,caller);
	FEM_Mesh_Partition_List l(c,destMeshes);
	if (m->node.size()>0) { /* partition normally */
		FEM_Mesh_partition(m,nParts,&l);
	} else { /* no geometric data in mesh-- just copy mesh */
		for (int i=0;i<nParts;i++)
			destMeshes[i]=FEM_Mesh_copy(fem_mesh);
	}
}
FORTRAN_AS_C(FEM_MESH_PARTITION,FEM_Mesh_partition,fem_mesh_partition,
	(int *mesh,int *nParts,int *dest),(*mesh,*nParts,dest))

/* Mesh communication */

CDECL int 
FEM_Mesh_recv(int fromRank,int tag,FEM_Comm_t comm_context)
{
	const char *caller="FEM_Mesh_recv";FEMAPI(caller);
	FEMchunk *c=FEMchunk::get(caller);
	marshallNewHeapCopy<FEM_Mesh> m;
	MPI_Recv_pup(m,fromRank,tag,(MPI_Comm)comm_context);
	return c->meshes.put(m);
}
FORTRAN_AS_C_RETURN(int,FEM_MESH_RECV,FEM_Mesh_recv,fem_mesh_recv, 
	(int *from,int *tag,int *comm),(*from,*tag,*comm))

CDECL void 
FEM_Mesh_send(int fem_mesh,int toRank,int tag,FEM_Comm_t comm_context)
{
	const char *caller="FEM_Mesh_send";FEMAPI(caller);
	FEMchunk *c=FEMchunk::get(caller);
	marshallNewHeapCopy<FEM_Mesh> m(c->meshes.lookup(fem_mesh,caller));
	MPI_Send_pup(m,toRank,tag,(MPI_Comm)comm_context);
}
FORTRAN_AS_C(FEM_MESH_SEND,FEM_Mesh_send,fem_mesh_send, 
	(int *mesh,int *to,int *tag,int *comm),(*mesh,*to,*tag,*comm))


CDECL int 
FEM_Mesh_reduce(int fem_mesh,int masterRank,FEM_Comm_t comm_context)
{
	int tag=89374;
	int myRank; MPI_Comm_rank((MPI_Comm)comm_context,&myRank);
	if (myRank!=masterRank) 
	{ /* I'm a slave-- send to master: */
		FEM_Mesh_send(fem_mesh,masterRank,tag,comm_context);
		return 0;
	}
	else /* myRank==masterRank */ 
	{ /* I'm the master-- recv the mesh pieces and assemble */
		int p, nParts; MPI_Comm_size((MPI_Comm)comm_context,&nParts);
		int *parts=new int[nParts];
		for (p=0;p<nParts;p++)
			if (p!=masterRank) /* recv from rank p */
				parts[p]=FEM_Mesh_recv(p,tag,comm_context);
			else
				parts[p]=fem_mesh; /* my part */
		int new_mesh=FEM_Mesh_assemble(nParts,parts);
		for (p=0;p<nParts;p++)
			if (p!=masterRank) /* Delete received meshes */
				FEM_Mesh_deallocate(parts[p]);
		delete[] parts;
		return new_mesh;
	}
}
FORTRAN_AS_C_RETURN(int,FEM_MESH_REDUCE,FEM_Mesh_reduce,fem_mesh_reduce, 
	(int *mesh,int *rank,int *comm),(*mesh,*rank,*comm))

CDECL int 
FEM_Mesh_broadcast(int fem_mesh,int masterRank,FEM_Comm_t comm_context)
{
	int tag=89375;
	int myRank; MPI_Comm_rank((MPI_Comm)comm_context,&myRank);
	if (myRank==masterRank) 
	{ /* I'm the master-- split up and send */
		int p, nParts; MPI_Comm_size((MPI_Comm)comm_context,&nParts);
		int *parts=new int[nParts];
		FEM_Mesh_partition(fem_mesh,nParts,parts);
		int new_mesh=0;
		for (p=0;p<nParts;p++)
			if (p!=masterRank) { /* Send off received meshes */
				FEM_Mesh_send(parts[p],p,tag,comm_context);
				FEM_Mesh_deallocate(parts[p]);
			}
			else /* Just keep my own partition */
				new_mesh=parts[p];
		return new_mesh;
	}
	else
	{ /* I'm a slave-- recv new mesh from master: */
		return FEM_Mesh_recv(masterRank,tag,comm_context);
	}
}
FORTRAN_AS_C_RETURN(int,FEM_MESH_BROADCAST,FEM_Mesh_broadcast,fem_mesh_broadcast, 
	(int *mesh,int *rank,int *comm),(*mesh,*rank,*comm))


CDECL void 
FEM_Mesh_copy_globalno(int src_mesh,int dest_mesh)
{
	const char *caller="FEM_Mesh_copy_globalno";FEMAPI(caller);
	FEMchunk *c=FEMchunk::get(caller);
	c->lookup(dest_mesh,caller)->
		copyOldGlobalno(*c->lookup(src_mesh,caller));
}

/* Tiny accessors */
CDECL int FEM_Mesh_default_read(void)  /* return default fem_mesh used for read (get) calls below */
{
	return FEMchunk::get("FEM_Mesh_default_read")->default_read;
}
FORTRAN_AS_C_RETURN(int,
	FEM_MESH_DEFAULT_READ,FEM_Mesh_default_read,fem_mesh_default_read,
	(void),())

CDECL int FEM_Mesh_default_write(void) /* return default fem_mesh used for write (set) calls below */
{
	return FEMchunk::get("FEM_Mesh_default_write")->default_write;
}
FORTRAN_AS_C_RETURN(int,
	FEM_MESH_DEFAULT_WRITE,FEM_Mesh_default_write,fem_mesh_default_write,
	(void),())

CDECL void FEM_Mesh_set_default_read(int fem_mesh)
{
	FEMchunk::get("FEM_Mesh_set_default_read")->default_read=fem_mesh;
}
FORTRAN_AS_C(FEM_MESH_SET_DEFAULT_READ,FEM_Mesh_set_default_read,fem_mesh_set_default_read,
	(int *m),(*m))
CDECL void FEM_Mesh_set_default_write(int fem_mesh)
{
	FEMchunk::get("FEM_Mesh_set_default_write")->default_write=fem_mesh;
}
FORTRAN_AS_C(FEM_MESH_SET_DEFAULT_WRITE,FEM_Mesh_set_default_write,fem_mesh_set_default_write,
	(int *m),(*m))

/********************** Mesh Creation ************************/
/*Utility*/

//Make a heap-allocated copy of this (len-entity) array, changing the index as spec'd
int *CkCopyArray(const int *src,int len,int indexBase)
{
	int *ret=new int[len];
	for (int i=0;i<len;i++) ret[i]=src[i]-indexBase;
	return ret;
}



/***** Mesh getting and setting state ****/
#ifndef CMK_OPTIMIZE
void FEMchunk::check(const char *where) {
	// CkPrintf("[%d] FEM> %s\n",thisIndex,where);
}
#endif

static FEM_Mesh *setMesh(void) {
  const char *caller="::setMesh";
  return FEMchunk::get(caller)->setMesh(caller);
}

static const FEM_Mesh *getMesh(void) {
  const char *caller="::getMesh";
  return FEMchunk::get(caller)->getMesh(caller);
}

FEM_Mesh *FEM_Mesh_lookup(int fem_mesh,const char *caller) {
	return FEMchunk::get(caller)->lookup(fem_mesh,caller);
}

/****** Custom Partitioning API *******/

//C bindings:
CDECL void FEM_Set_partition(int *elem2chunk) {
	FEMAPI("FEM_Set_partition");
	FEM_curPartition().setPartition(elem2chunk,setMesh()->nElems(),0);
}

//Fortran bindings:
FDECL void FTN_NAME(FEM_SET_PARTITION,fem_set_partition)
	(int *elem2chunk) 
{
	FEMAPI("FEM_Set_partition");
	FEM_curPartition().setPartition(elem2chunk,setMesh()->nElems(),1);
}


/******************************* CHUNK *********************************/

FEMchunk::FEMchunk(FEM_Comm_t defaultComm_)
	:defaultComm(defaultComm_)
{
  default_read=-1;
  default_write=-1;
  checkMPI(MPI_Comm_rank((MPI_Comm)defaultComm,&thisIndex));
  initFields();
}
FEMchunk::FEMchunk(CkMigrateMessage *msg)
{
  initFields();
}

void FEMchunk::pup(PUP::er &p)
{
//Pup superclass (none)

// Pup the meshes
  p|default_read;
  p|default_write;
  p|meshes;
  
  p|defaultComm;
  p|thisIndex;
}

PUPable_def(FEM_Sym_Linear);
//Update fields after creation/migration
void FEMchunk::initFields(void)
{
  PUPable_reg(FEM_Sym_Linear);
}

FEMchunk::~FEMchunk()
{
}


void
FEMchunk::reduce_field(int fid, const void *nodes, void *outbuf, int op)
{
  check("reduce_field precondition");
  // first reduce over local nodes
  const IDXL_Layout &dt = IDXL_Layout_List::get().get(fid,"FEM_Reduce_field");
  const byte *src = (const byte *) nodes;
  reduction_initialize(dt,outbuf,op);
  reduction_combine_fn fn=reduction_combine(dt,op);
  int nNodes=getMesh("reduce_field")->node.size();
  for(int i=0; i<nNodes; i++) {
    if(getPrimary(i)) {
      fn((byte *)outbuf, src, &dt);
    }
    src += dt.userBytes();
  }
  // and now reduce over partitions
  reduce(fid, outbuf, outbuf, op);
  check("reduce_field postcondition");
}

void
FEMchunk::reduce(int fid, const void *inbuf, void *outbuf, int op)
{
  const char *caller="FEM_Reduce";
  check("reduce precondition");
  const IDXL_Layout &dt = IDXL_Layout_List::get().get(fid,caller);
  int len = dt.compressedBytes();
  // MPI does not allow inbuf==outbuf, so make a copy:
  char *tmpbuf=new char[len];
  memcpy(tmpbuf,inbuf,len);
  // Map IDXL datatypes to MPI:
  MPI_Datatype mpidt;
  switch (dt.type) {
  case IDXL_BYTE: mpidt=MPI_BYTE; break;
  case IDXL_INT: mpidt=MPI_INT; break;
  case IDXL_REAL: mpidt=MPI_FLOAT; break;
  case IDXL_DOUBLE: mpidt=MPI_DOUBLE; break;
  default: FEM_Abort(caller,"cannot map FEM datatype %d to MPI",dt.type); break;
  };
  
  MPI_Op mpiop;
  switch (op) {
  case IDXL_SUM: mpiop=MPI_SUM; break;
  case IDXL_PROD: mpiop=MPI_PROD; break;
  case IDXL_MAX: mpiop=MPI_MAX; break;
  case IDXL_MIN: mpiop=MPI_MIN; break;
  default: FEM_Abort(caller,"cannot map FEM operation %d to MPI",op); break;
  };
  MPI_Comm comm=(MPI_Comm)defaultComm;
  MPI_Allreduce(tmpbuf,outbuf,dt.width,mpidt,mpiop,comm);
  delete[] tmpbuf;
  check("reduce postcondition");
}

void
FEMchunk::readField(int fid, void *nodes, const char *fname)
{
  const IDXL_Layout &dt = IDXL_Layout_List::get().get(fid,"FEM_Read_field");
  int type = dt.type;
  int width = dt.width;
  int offset = dt.offset;
  int distance = dt.distance;
  int skew = dt.skew;
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
  switch(type) {
    case FEM_INT: fmt = "%d%n"; break;
    case FEM_REAL: fmt = "%f%n"; break;
    case FEM_DOUBLE: fmt = "%lf%n"; break;
    default: CkAbort("FEM_Read_field doesn't support that data type");
  }
  FEM_Mesh *cur_mesh=getMesh("FEM_Read_field");
  int nNodes=cur_mesh->node.size();
  for(i=0;i<nNodes;i++) {
    // skip lines to the next local node 
    // (FIXME: assumes nodes are in ascending global order, which they ain't)
    int target=cur_mesh->node.getGlobalno(i);
    for(j=curline;j<target;j++)
      fgets(str,80,fp);
    curline = target+1;
    fgets(str,80,fp);
    int curnode, numchars;
    sscanf(str,"%d%n",&curnode,&numchars);
    pos = str + numchars;
    if(curnode != target) {
      CkError("Expecting info for node %d, got %d\n", target, curnode);
      CkAbort("Exiting");
    }
    for(j=0;j<width;j++) {
      sscanf(pos, fmt, &IDXL_LAYOUT_DEREF(byte,nodes,i,j), &numchars);
      pos += numchars;
    }
  }
  fclose(fp);
}

/******************************* C Bindings **********************************/

CDECL int FEM_Register(void *_ud,FEM_PupFn _pup_ud)
{
  FEMAPI("FEM_Register");
  return TCHARM_Register(_ud,_pup_ud);
}

CDECL void *FEM_Get_userdata(int n)
{
  FEMAPI("FEM_Get_userdata");
  return TCHARM_Get_userdata(n);
}

CDECL void FEM_Barrier(void) {TCHARM_Barrier();}
FDECL void FTN_NAME(FEM_BARRIER,fem_barrier)(void) {TCHARM_Barrier();}

CDECL void
FEM_Migrate(void)
{
  TCHARM_Migrate();
}

CDECL void 
FEM_Done(void)
{
  TCHARM_Done();
}

CDECL int 
FEM_Create_simple_field(int base_type, int vec_len)
{
  return IDXL_Layout_create(base_type,vec_len);
}

CDECL int 
FEM_Create_field(int base_type, int vec_len, int init_offset, int distance)
{
  return IDXL_Layout_offset(base_type,vec_len,init_offset,distance,0);
}

CDECL void
FEM_Update_field(int fid, void *nodes)
{
  int mesh=FEM_Mesh_default_read();
  IDXL_t list=FEM_Comm_shared(mesh,FEM_NODE);
  IDXL_Comm_sendsum(0,list,fid,nodes);
}

CDECL void
FEM_Reduce_field(int fid, const void *nodes, void *outbuf, int op)
{
  const char *caller="FEM_Reduce_field"; FEMAPI(caller);
  FEMchunk::get(caller)->reduce_field(fid, nodes, outbuf, op);
}

CDECL void
FEM_Reduce(int fid, const void *inbuf, void *outbuf, int op)
{
  const char *caller="FEM_Reduce";FEMAPI(caller);
  FEMchunk::get(caller)->reduce(fid, inbuf, outbuf, op);
}

CDECL void
FEM_Read_field(int fid, void *nodes, const char *fname)
{
  const char *caller="FEM_Read_field";FEMAPI(caller);
  FEMchunk::get(caller)->readField(fid, nodes, fname);
}

CDECL int
FEM_My_partition(void)
{
  return TCHARM_Element();
}

CDECL int
FEM_Num_partitions(void)
{
  return TCHARM_Num_elements();
}

CDECL double
FEM_Timer(void)
{
  return TCHARM_Wall_timer();
}

CDECL void 
FEM_Print(const char *str)
{
  const char *caller="FEM_Print"; FEMAPI(caller);
  FEMchunk *cptr = FEMchunk::get(caller);
  CkPrintf("[%d] %s\n", cptr->thisIndex, str);
}

static void do_print_partition(int fem_mesh,int idxBase) {
  
  const char *caller="FEM_Mesh_print"; FEMAPI(caller);
  FEMchunk *cptr = FEMchunk::get(caller);
  cptr->print(fem_mesh,idxBase);
}

CDECL void 
FEM_Mesh_print(int fem_mesh)
{
  do_print_partition(fem_mesh,0);
}

CDECL void
FEM_Print_partition(void) {
  do_print_partition(FEM_Mesh_default_read(),0);
}

CDECL int FEM_Get_comm_partners(void)
{
	const char *caller="FEM_Get_Comm_Partners"; FEMAPI(caller);
	return FEMchunk::get(caller)->getComm().size();
}
CDECL int FEM_Get_comm_partner(int partnerNo)
{
	const char *caller="FEM_Get_Comm_Partner"; FEMAPI(caller);
	return FEMchunk::get(caller)->getComm().getLocalList(partnerNo).getDest();
}
CDECL int FEM_Get_comm_count(int partnerNo)
{
	const char *caller="FEM_Get_Comm_Count"; FEMAPI(caller);
	return FEMchunk::get(caller)->getComm().getLocalList(partnerNo).size();
}
CDECL void FEM_Get_comm_nodes(int partnerNo,int *nodeNos)
{
	const char *caller="FEM_Get_comm_nodes"; FEMAPI(caller);
	const int *nNo=FEMchunk::get(caller)->getComm().getLocalList(partnerNo).getVec();
	int len=FEM_Get_comm_count(partnerNo);
	for (int i=0;i<len;i++)
		nodeNos[i]=nNo[i];
}

/************************ Fortran Bindings *********************************/

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

FDECL int FTN_NAME(FEM_CREATE_SIMPLE_FIELD,fem_create_simple_field)
  (int *bt, int *vl)
{
  return FEM_Create_simple_field(*bt, *vl);
}
FDECL int FTN_NAME(FEM_CREATE_FIELD,fem_create_field)
  (int *bt, int *vl, int *io, int *d)
{
  return FEM_Create_field(*bt, *vl, *io, *d);
}

FDECL void FTN_NAME(FEM_UPDATE_FIELD,fem_update_field)
  (int *fid, void *nodes)
{
  FEM_Update_field(*fid, nodes);
}

FDECL void  FTN_NAME(FEM_REDUCE_FIELD,fem_reduce_field)
  (int *fid, void *nodes, void *outbuf, int *op)
{
  FEM_Reduce_field(*fid, nodes, outbuf, *op);
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
  FEM_Read_field(*fid, nodes, tmp);
  delete[] tmp;
}

FDECL int FTN_NAME(FEM_MY_PARTITION,fem_my_partition)
  (void)
{
  return FEM_My_partition()+1;
}

FDECL int FTN_NAME(FEM_NUM_PARTITIONS,fem_num_partitions)
  (void)
{
  return FEM_Num_partitions();
}

FDECL double FTN_NAME(FEM_TIMER,fem_timer)
  (void)
{
  return FEM_Timer();
}

// Utility functions for Fortran
FDECL int FTN_NAME(FOFFSETOF,foffsetof)
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

FDECL void FTN_NAME(FEM_MESH_PRINT,fem_mesh_print)
  (int *m)
{
  do_print_partition(*m,1);
}
FDECL void FTN_NAME(FEM_PRINT_PARTITION,fem_print_partition)(void) {
  do_print_partition(FEM_Mesh_default_read(),0);
}

FDECL void FTN_NAME(FEM_DONE,fem_done)
  (void)
{
  FEM_Done();
}

FDECL int FTN_NAME(FEM_GET_COMM_PARTNERS,fem_get_comm_partners)
	(void)
{
	return FEM_Get_comm_partners();
}
FDECL int FTN_NAME(FEM_GET_COMM_PARTNER,fem_get_comm_partner)
	(int *partnerNo)
{
	return FEM_Get_comm_partner(*partnerNo-1)+1;
}
FDECL int FTN_NAME(FEM_GET_COMM_COUNT,fem_get_comm_count)
	(int *partnerNo)
{
	return FEM_Get_comm_count(*partnerNo-1);
}
FDECL void FTN_NAME(FEM_GET_COMM_NODES,fem_get_comm_nodes)
	(int *pNo,int *nodeNos)
{
	const char *caller="FEM_GET_COMM_NODES"; FEMAPI(caller);
	int partnerNo=*pNo-1;
	const int *nNo=FEMchunk::get(caller)->getComm().getLocalList(partnerNo).getVec();
	int len=FEM_Get_comm_count(partnerNo);
	for (int i=0;i<len;i++)
		nodeNos[i]=nNo[i]+1;
}

/******************** Ghost Layers *********************/
CDECL void FEM_Add_ghost_layer(int nodesPerTuple,int doAddNodes)
{
	FEMAPI("FEM_Add_ghost_layer");
	ghostLayer *cur=FEM_curPartition().addLayer();
	cur->nodesPerTuple=nodesPerTuple;
	cur->addNodes=(doAddNodes!=0);
	cur->elem.makeLonger(20);
}
FDECL void FTN_NAME(FEM_ADD_GHOST_LAYER,fem_add_ghost_layer)
	(int *nodesPerTuple,int *doAddNodes)
{ FEM_Add_ghost_layer(*nodesPerTuple,*doAddNodes); }

static void add_ghost_elem(int elType,int tuplesPerElem,const int *elem2tuple,int idxBase) {
	FEMAPI("FEM_Add_ghost_elem");
	ghostLayer *cur=FEM_curPartition().curLayer();
	cur->elem[elType].add=true;
	cur->elem[elType].tuplesPerElem=tuplesPerElem;
	cur->elem[elType].elem2tuple=CkCopyArray(elem2tuple,
		          tuplesPerElem*cur->nodesPerTuple,idxBase);
}

CDECL void FEM_Add_ghost_elem(int elType,int tuplesPerElem,const int *elem2tuple)
{ add_ghost_elem(elType,tuplesPerElem,elem2tuple,0); }
FDECL void FTN_NAME(FEM_ADD_GHOST_ELEM,fem_add_ghost_elem)
	(int *FelType,int *FtuplesPerElem,const int *elem2tuple)
{ add_ghost_elem(*FelType,*FtuplesPerElem,elem2tuple,1); }

CDECL void FEM_Update_ghost_field(int fid, int elType, void *v_data)
{
	int mesh=FEM_Mesh_default_read();
	int entity;
	if (elType==-1) entity=FEM_NODE;
	else entity=FEM_ELEM+elType;
	IDXL_t src=FEM_Comm_ghost(mesh,entity);
	int nReal=FEM_Mesh_get_length(mesh,entity);
	int bytesPerRec=IDXL_Get_layout_distance(fid);
	char *data=(char *)v_data;
	IDXL_Comm_t comm=IDXL_Comm_begin(1,0);
	IDXL_Comm_send(comm,src,fid,data);
	IDXL_Comm_recv(comm,src,fid,&data[nReal*bytesPerRec]); //Begin recv'ing ghosts after all reals
	IDXL_Comm_wait(comm);
}
FDECL void FTN_NAME(FEM_UPDATE_GHOST_FIELD,fem_update_ghost_field)
	(int *fid, int *elemType, void *data)
{
	int fElType=*elemType;
	if (fElType==0) //Ghost node update
		FEM_Update_ghost_field(*fid,-1,data);
	else //Ghost element update
		FEM_Update_ghost_field(*fid,fElType,data);
}

/*********** Mesh modification **********
It's the *user's* responsibility to ensure no update_field calls
are outstanding when the mesh is begin modified.  This typically
means keeping track of outstanding communication, and/or wrapping things 
in FEM_Barrier calls.
*/

//Entity list exchange: (should probably be moved to IDXL library)
// FIXME: I assume only one outstanding list exchange.
// This implies the presence of a global barrier before or after the exchange.
void FEMchunk::exchangeGhostLists(int elemType,
	     int inLen,const int *inList,int idxbase)
{
	check("exchangeGhostLists");
	FEM_Mesh *cur_mesh=getMesh("exchangeGhostLists");
	const FEM_Entity &e=cur_mesh->getCount(elemType);
	const FEM_Comm &cnt=e.getGhostSend();
	int tag=89376;
	MPI_Comm comm=(MPI_Comm)defaultComm;
	
//Send build an index list for each of our neighbors
	int i,chk,nChk=cnt.size();
	CkVec<int> *outIdx=new CkVec<int>[nChk];
	//Loop over (the shared entries in) the input list
	for (i=0;i<inLen;i++) {
		int localNo=inList[i]-idxbase;
		const FEM_Comm_Rec *rec=cnt.getRec(localNo);
		if (NULL==rec) continue; //This entity isn't shared
		//This entity is shared-- add its comm. idx to each chk
		for (int s=0;s<rec->getShared();s++) {
			int localChk=cnt.findLocalList(rec->getChk(s));
			outIdx[localChk].push_back(rec->getIdx(s));
		}
	}

//Send off the comm. idx list to each chk:
	MPI_Request *sends=new MPI_Request[nChk];
	for (chk=0;chk<nChk;chk++) {
		checkMPI(MPI_Isend(outIdx[chk].getVec(),outIdx[chk].size(),MPI_INT,
			cnt.getLocalList(chk).getDest(),tag,comm,&sends[chk]));
	}
	
//Wait until all the replies have arrived
	int listStart=0;
	for (chk=0;chk<e.getGhostRecv().size();chk++) {
		int src=e.getGhostRecv().getLocalList(chk).getDest();
		int nRecv=myMPI_Incoming(MPI_INT,src,tag,comm);
		listTmp.resize(listStart+nRecv);
		int *list=&listTmp[listStart];
		MPI_Status sts;
		checkMPI(MPI_Recv(list,nRecv,MPI_INT,src,tag,comm,&sts));
		// Convert from comm. list entries to real ghost indices,
		//  by looking up in the comm. list:
		int firstGhost=e.size();
		const FEM_Comm_List &l=e.getGhostRecv().getLocalList(chk);
		for (i=0;i<nRecv;i++)
			list[i]=firstGhost+l[list[i]];
		listStart+=nRecv;
	}
	
// Finish the communication and free buffers
	MPI_Status *sts=new MPI_Status[nChk];
	checkMPI(MPI_Waitall(nChk,sends,sts));
	delete[] sts;
	delete[] sends;
	delete[] outIdx;
}

//List exchange API
CDECL void FEM_Exchange_ghost_lists(int elemType,int nIdx,const int *localIdx)
{
	const char *caller="FEM_Exchange_Ghost_Lists"; FEMAPI(caller);
	FEMchunk::get(caller)->exchangeGhostLists(elemType,nIdx,localIdx,0);
}
FDECL void FTN_NAME(FEM_EXCHANGE_GHOST_LISTS,fem_exchange_ghost_lists)
	(int *elemType,int *nIdx,const int *localIdx)
{
	const char *caller="FEM_exchange_ghost_lists"; FEMAPI(caller);
	FEMchunk::get(caller)->exchangeGhostLists(*elemType,*nIdx,localIdx,1);
}
CDECL int FEM_Get_ghost_list_length(void) 
{
	const char *caller="FEM_Get_Ghost_List_Length"; FEMAPI(caller);
	return FEMchunk::get(caller)->getList().size();
}
FDECL int FTN_NAME(FEM_GET_GHOST_LIST_LENGTH,fem_get_ghost_list_length)(void)
{ return FEM_Get_ghost_list_length();}

CDECL void FEM_Get_ghost_list(int *dest)
{
	const char *caller="FEM_Get_Ghost_List"; FEMAPI(caller);
	int i,len=FEM_Get_ghost_list_length();
	const int *src=FEMchunk::get(caller)->getList().getVec();
	for (i=0;i<len;i++) dest[i]=src[i];
	FEMchunk::get(caller)->emptyList();
}
FDECL void FTN_NAME(FEM_GET_GHOST_LIST,fem_get_ghost_list)
	(int *dest)
{
	const char *caller="FEM_get_ghost_list"; FEMAPI(caller);
	int i,len=FEM_Get_ghost_list_length();
	const int *src=FEMchunk::get(caller)->getList().getVec();
	for (i=0;i<len;i++) dest[i]=src[i]+1;
	FEMchunk::get(caller)->emptyList();
}


/********* Debugging mesh printouts *******/

class localToGlobal : public IDXL_Print_Map {
	FEM_Entity &e;
public:
	int idxBase;
	localToGlobal(FEM_Entity &e_, int idxBase_)
		:e(e_), idxBase(idxBase_) {}
	
	void map(int l) const {
		if (l<0) CkPrintf("(%d)  ",l+idxBase);
		else {
			int g=e.getGlobalno(l);
			if (g==-1) CkPrintf("(%d)  ",l+idxBase);
			else CkPrintf("%d  ",g+idxBase);
		}
	}
};

void FEM_Entity::print(const char *type,const IDXL_Print_Map &l2g)
{
  CkPrintf("\nLength of %s = %d", type, size());
  if (ghost->size()>0) CkPrintf(" (and %d ghosts)\n",ghost->size());
  else CkPrintf("\n");
  
  //FIXME: also print out data (properly formatted) and symmetries
}

void FEM_Node::print(const char *type,const IDXL_Print_Map &l2g)
{
  super::print(type,l2g);
  CkPrintf("Node global numbers: * marks primary, () surrounds local-only:\n");
  for(int i=0;i<size();i++) {
    if (getPrimary(i)) CkPrintf("*");
    l2g.map(i);
  }
  CkPrintf("\n");
}



void FEM_Elem::print(const char *type,const IDXL_Print_Map &l2g)
{
  super::print(type,l2g);
  
  localToGlobal *src_l2g=(localToGlobal *)&l2g; //HACK!
  localToGlobal elt_l2g(*this,src_l2g->idxBase);
  
  CkPrintf("%s Connectivity: \n",type);
  for (int r=0;r<size();r++) {
    CkPrintf("  Entry "); elt_l2g.map(r); CkPrintf("|  ");
    for (int c=0;c<getNodesPer();c++)
       l2g.map(conn->get()(r,c));
    CkPrintf("\n");
  }
}

void FEM_Mesh::print(int idxBase)
{
  localToGlobal l2g(node,idxBase);
  node.print("FEM_NODE",l2g);
  int t;
  for (t=0;t<elem.size();t++) 
  if (elem.has(t)) {
  	char name[50]; sprintf(name,"FEM_ELEM+%d",t);
  	elem[t].print(name,l2g);
  }
  for (t=0;t<sparse.size();t++) 
  if (sparse.has(t)) {
  	char name[50]; sprintf(name,"FEM_SPARSE+%d",t);
  	sparse[t].print(name,l2g);
  }
}

void
FEMchunk::print(int fem_mesh,int idxBase)
{
  CkPrintf("-------------------- Chunk %d --------------------\n",thisIndex);
  lookup(fem_mesh,"FEM_Mesh_print")->print(idxBase);
  CkPrintf("\n\n");
}  
