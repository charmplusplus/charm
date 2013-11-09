/**
 * \addtogroup ParFUM
 */
/*@{*/


/*
ParFUM - Parallel Framework for Unstructured Meshing
Parallel Programming Lab, Univ. of Illinois 2006

 */

#include "ParFUM.h"
#include "ParFUM_internals.h"


/* Some Globals */
int femVersion = 1;
static FEM_Partition *mypartition=NULL;

enum PartitionMode {
    SerialPartitionMode = 1,
    ParallelPartitionMode = 2,
    ManualPartitionMode
};
PartitionMode FEM_Partition_Mode = SerialPartitionMode;

enum MetisGraphType {
    NodeNeighborMode,
    FaceNeighborMode
};
MetisGraphType FEM_Partition_Graph_Type = NodeNeighborMode;

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

/******** Startup and initialization *******/

// This is our TCharm global ID:
enum {FEM_globalID=33};

CDECL void pupFEM_Chunk(pup_er cp) {
	PUP::er &p=*(PUP::er *)cp;
	FEM_chunk *c=(FEM_chunk *)TCHARM_Get_global(FEM_globalID);
	if (c==NULL) {
		c=new FEM_chunk((CkMigrateMessage *)0);
		TCHARM_Set_global(FEM_globalID,c,pupFEM_Chunk);
	}
	c->pup(p);
	if (p.isDeleting())
		delete c;
}
FEM_chunk *FEM_chunk::get(const char *caller) {
	FEM_chunk *c=(FEM_chunk *)TCHARM_Get_global(FEM_globalID);
	if(!c) FEM_Abort(caller,"FEM is not initialized");
	return c;
}

CDECL void FEM_Init(FEM_Comm_t defaultComm)
{
	IDXL_Init(defaultComm);
	if (!TCHARM_Get_global(FEM_globalID)) {
		FEM_chunk *c=new FEM_chunk(defaultComm);
		TCHARM_Set_global(FEM_globalID,c,pupFEM_Chunk);
	}
	char **argv = CkGetArgv();
        if(CmiGetArgFlagDesc(argv,"+Parfum_parallel_partition","ParFUM should use the parallel partitioner")){
            FEM_Partition_Mode = ParallelPartitionMode;
	}
        if (CmiGetArgFlagDesc(argv, "+Parfum_manual_partition", "Specify manual mesh partitioning")) {
            FEM_Partition_Mode = ManualPartitionMode;
        }
        if (CmiGetArgFlagDesc(argv,
                    "+Parfum_face_neighbor_graph",
                    "Specify partitioning based on face neighbors instead of node neighbors")) {
            FEM_Partition_Graph_Type = FaceNeighborMode;
        }
}
FORTRAN_AS_C(FEM_INIT,FEM_Init,fem_init, (int *comm), (*comm))


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
	FEM_chunk *c=FEM_chunk::get(caller);
	FEM_Mesh *m=new FEM_Mesh;
	m->becomeSetting();
	return c->meshes.put(m);
}
FORTRAN_AS_C_RETURN(int,
	FEM_MESH_ALLOCATE,FEM_Mesh_allocate,fem_mesh_allocate, (void),())

/// Return a new'd copy of this class, by calling pup.
template <class T>
inline T *clonePointerViaPup(T *old) {
	size_t len=PUP::size(*old);
	char *buf=new char[len];
	PUP::toMemBuf(*old,buf,len);
	T *nu=new T;
	PUP::fromMemBuf(*nu,buf,len);
	return nu;
}

CDECL int
FEM_Mesh_copy(int fem_mesh) {
	const char *caller="FEM_Mesh_copy";FEMAPI(caller);
	FEM_chunk *c=FEM_chunk::get(caller);
	return c->meshes.put(clonePointerViaPup(c->meshes.lookup(fem_mesh,caller)));
}
FORTRAN_AS_C_RETURN(int,
	FEM_MESH_COPY,FEM_Mesh_copy,fem_mesh_copy, (int *m),(*m))


CDECL void 
FEM_Mesh_deallocate(int fem_mesh) /* delete this local mesh */
{
	if (fem_mesh!=-1) {
		const char *caller="FEM_Mesh_deallocate";FEMAPI(caller);
		FEM_chunk *c=FEM_chunk::get(caller);
		c->meshes.destroy(fem_mesh,caller);
	}
}
FORTRAN_AS_C(FEM_MESH_DEALLOCATE,FEM_Mesh_deallocate,fem_mesh_deallocate, (int *m),(*m))

/* Mesh I/O */
CDECL int 
FEM_Mesh_read(const char *prefix,int partNo,int nParts) /* read parallel mesh from file */
{
	const char *caller="FEM_Mesh_read";FEMAPI(caller);
	FEM_chunk *c=FEM_chunk::get(caller);
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
	FEM_chunk *c=FEM_chunk::get(caller);
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
	FEM_chunk *c=FEM_chunk::get(caller);
	FEM_Mesh **chunks=new FEM_Mesh*[nParts];
	for (int p=0;p<nParts;p++) chunks[p]=c->meshes.lookup(srcMeshes[p],caller);
	int ret=c->meshes.put(FEM_Mesh_assemble(nParts,chunks));
	delete[] chunks;
	return ret;
}
FORTRAN_AS_C_RETURN(int,FEM_MESH_ASSEMBLE,FEM_Mesh_assemble,fem_mesh_assemble,
	(int *nParts,const int *src),(*nParts,src))


/* mypartition is static to this file, and is declared above */
FEM_Partition &FEM_curPartition(void) {
  if (mypartition==NULL) mypartition=new FEM_Partition();
	return *mypartition;
}
void clearPartition(void) {delete mypartition; mypartition=NULL;}

FEM_Partition::FEM_Partition() 
{
	elem2chunk=NULL;
	sym=NULL; 
	lastLayer=0;
}
FEM_Partition::~FEM_Partition() {
	if (elem2chunk) {delete[] elem2chunk;elem2chunk=NULL;}
	for (int i=0;i<getRegions();i++) {
		delete regions[i].layer;
		delete regions[i].stencil;
	}
}

void FEM_Partition::setPartition(const int *e, int nElem, int idxBase) {
	if (elem2chunk) {delete[] elem2chunk;elem2chunk=NULL;}
	elem2chunk=CkCopyArray(e,nElem,idxBase);
}
const int *FEM_Partition::getPartition(FEM_Mesh *src,int nChunks) const {
	if (!elem2chunk) { /* Create elem2chunk based on Metis partitioning: */
		int *e=new int[src->nElems()];
		FEM_Mesh_partition(src,nChunks,e,FEM_Partition_Graph_Type == FaceNeighborMode);
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
	FEM_chunk *c;
	int *dest;
public:
	FEM_Mesh_Partition_List(FEM_chunk *c_, int *dest_)
		:c(c_), dest(dest_) {}
	
	void accept(int chunkNo,FEM_Mesh *m) {
		dest[chunkNo]=c->meshes.put(m);
	}
};

CDECL void 
FEM_Mesh_partition(int fem_mesh,int nParts,int *destMeshes) {
	const char *caller="FEM_Mesh_partition"; FEMAPI(caller);
	FEM_chunk *c=FEM_chunk::get(caller);
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
	FEM_chunk *c=FEM_chunk::get(caller);
	marshallNewHeapCopy<FEM_Mesh> m;
	MPI_Recv_pup(m,fromRank,tag,(MPI_Comm)comm_context);
	FEM_Mesh *msh=m; 
	msh->becomeGetting();
	return c->meshes.put(msh);
}
FORTRAN_AS_C_RETURN(int,FEM_MESH_RECV,FEM_Mesh_recv,fem_mesh_recv, 
	(int *from,int *tag,int *comm),(*from,*tag,*comm))

CDECL void 
FEM_Mesh_send(int fem_mesh,int toRank,int tag,FEM_Comm_t comm_context)
{
	const char *caller="FEM_Mesh_send";FEMAPI(caller);
	FEM_chunk *c=FEM_chunk::get(caller);
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


extern int FEM_Mesh_Parallel_broadcast(int fem_mesh,int masterRank,FEM_Comm_t comm_context);

CDECL int 
FEM_Mesh_broadcast(int fem_mesh,int masterRank,FEM_Comm_t comm_context)
{
    int myRank;
    MPI_Comm_rank((MPI_Comm)comm_context,&myRank);
    if (FEM_Partition_Mode == ManualPartitionMode && myRank == masterRank) {
        const char* caller = "FEM_Mesh_broadcast";
        FEMAPI(caller);
    	FEM_chunk* c = FEM_chunk::get(caller);
    	FEM_Mesh* mesh = c->lookup(fem_mesh,caller);    	

        // Look into each element type and read a map of elements to
        // partitions out of FEM_PARTITION
        int nelems = mesh->nElems();
        int* elem2chunk = new int [nelems];
        int index=0;
        for (int elemType=0; elemType<mesh->elem.size(); ++elemType) {
            FEM_Elem& elems = mesh->elem[elemType];
            FEM_DataAttribute* partitionMap = (FEM_DataAttribute*) elems.lookup(FEM_PARTITION, caller);
            for (int elem=0; elem<elems.size(); ++elem) {
                elem2chunk[index++] = partitionMap->getInt().getData()[elem];
            }
        }
        // Setting the partition preemptively prevents Metis from being called
        FEM_Set_partition(elem2chunk);
    }
	if (FEM_Partition_Mode == SerialPartitionMode || FEM_Partition_Mode == ManualPartitionMode) {
		int tag=89375;
		if (myRank==masterRank) { /* I'm the master-- split up and send */
			int p, nParts; MPI_Comm_size((MPI_Comm)comm_context,&nParts);
			int *parts=new int[nParts];
			FEM_Mesh_partition(fem_mesh,nParts,parts);
			int new_mesh=0;
			for (p=0;p<nParts;p++)
				if (p!=masterRank) { /* Send off received meshes */
					FEM_Mesh_send(parts[p],p,tag,comm_context);
					FEM_Mesh_deallocate(parts[p]);
				}
				else { /* Just keep my own partition */
					new_mesh=parts[p];
				}
			return new_mesh;
		} else { /* I'm a slave-- recv new mesh from master: */
			return FEM_Mesh_recv(masterRank,tag,comm_context);
		}
    } else if (FEM_Partition_Mode == ParallelPartitionMode) {
		//parallel partition
		MPI_Barrier((MPI_Comm)comm_context);
		//_registerfem();
		MPI_Barrier((MPI_Comm)comm_context);
		return FEM_Mesh_Parallel_broadcast(fem_mesh,masterRank,comm_context);
	} else {
        CkAbort("Unrecognized ParFUM partitioning mode");
    }
    return -1; // we should never get here
}
FORTRAN_AS_C_RETURN(int,FEM_MESH_BROADCAST,FEM_Mesh_broadcast,fem_mesh_broadcast, 
	(int *mesh,int *rank,int *comm),(*mesh,*rank,*comm))


CDECL void 
FEM_Mesh_copy_globalno(int src_mesh,int dest_mesh)
{
	const char *caller="FEM_Mesh_copy_globalno";FEMAPI(caller);
	FEM_chunk *c=FEM_chunk::get(caller);
	c->lookup(dest_mesh,caller)->
		copyOldGlobalno(*c->lookup(src_mesh,caller));
}

/* Tiny accessors */
CDECL int FEM_Mesh_default_read(void)  /* return default fem_mesh used for read (get) calls below */
{
	return FEM_chunk::get("FEM_Mesh_default_read")->default_read;
}
FORTRAN_AS_C_RETURN(int,
	FEM_MESH_DEFAULT_READ,FEM_Mesh_default_read,fem_mesh_default_read,
	(void),())

CDECL int FEM_Mesh_default_write(void) /* return default fem_mesh used for write (set) calls below */
{
	return FEM_chunk::get("FEM_Mesh_default_write")->default_write;
}
FORTRAN_AS_C_RETURN(int,
	FEM_MESH_DEFAULT_WRITE,FEM_Mesh_default_write,fem_mesh_default_write,
	(void),())

CDECL void FEM_Mesh_set_default_read(int fem_mesh)
{
	FEM_chunk::get("FEM_Mesh_set_default_read")->default_read=fem_mesh;
}
FORTRAN_AS_C(FEM_MESH_SET_DEFAULT_READ,FEM_Mesh_set_default_read,fem_mesh_set_default_read,
	(int *m),(*m))
CDECL void FEM_Mesh_set_default_write(int fem_mesh)
{
	FEM_chunk::get("FEM_Mesh_set_default_write")->default_write=fem_mesh;
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
#if CMK_ERROR_CHECKING
void FEM_chunk::check(const char *where) {
	// CkPrintf("[%d] FEM> %s\n",thisIndex,where);
}
#endif

static FEM_Mesh *setMesh(void) {
  const char *caller="::setMesh";
  return FEM_chunk::get(caller)->setMesh(caller);
}

static const FEM_Mesh *getMesh(void) {
  const char *caller="::getMesh";
  return FEM_chunk::get(caller)->getMesh(caller);
}

FEM_Mesh *FEM_Mesh_lookup(int fem_mesh, const char *caller) {
  return FEM_chunk::get(caller)->lookup(fem_mesh,caller);
}

CDECL void FEM_Mesh_Become_Setting(int fem_mesh) {
  FEM_Mesh *meshP = FEM_Mesh_lookup(fem_mesh,"driver");
  meshP->becomeSetting();
}
FDECL void FTN_NAME(FEM_MESH_BECOME_SETTING,fem_mesh_become_setting)
  (int *mesh)
{
  FEM_Mesh_Become_Setting(*mesh);
}

CDECL void FEM_Mesh_Become_Getting(int fem_mesh) {
  FEM_Mesh *meshP = FEM_Mesh_lookup(fem_mesh,"driver");
  meshP->becomeGetting();
}
FDECL void FTN_NAME(FEM_MESH_BECOME_GETTING,fem_mesh_become_getting)
  (int *mesh)
{
  FEM_Mesh_Become_Getting(*mesh);
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

FEM_chunk::FEM_chunk(FEM_Comm_t defaultComm_)
	:defaultComm(defaultComm_)
{
  default_read=-1;
  default_write=-1;
  checkMPI(MPI_Comm_rank((MPI_Comm)defaultComm,&thisIndex));
  initFields();
}
FEM_chunk::FEM_chunk(CkMigrateMessage *msg)
{
  initFields();
}

void FEM_chunk::pup(PUP::er &p)
{
//Pup superclass (none)

// Pup the meshes
  p|default_read;
  p|default_write;
  p|thisIndex;
  p|meshes;
  
  p|defaultComm;
}

PUPable_def(FEM_Sym_Linear)
//Update fields after creation/migration
void FEM_chunk::initFields(void)
{
  if (CkMyRank() == 0) 
    PUPable_reg(FEM_Sym_Linear);
}

FEM_chunk::~FEM_chunk()
{
}


void
FEM_chunk::reduce_field(int fid, const void *nodes, void *outbuf, int op)
{
  check("reduce_field precondition");
  // first reduce over local nodes
  const IDXL_Layout &dt = IDXL_Layout_List::get().get(fid,"FEM_Reduce_field");
  const unsigned char *src = (const unsigned char *) nodes;
  reduction_initialize(dt,outbuf,op);
  reduction_combine_fn fn=reduction_combine(dt,op);
  int nNodes=getMesh("reduce_field")->node.size();
  for(int i=0; i<nNodes; i++) {
    if(getPrimary(i)) {
      fn((unsigned char *)outbuf, src, &dt);
    }
    src += dt.userBytes();
  }
  // and now reduce over partitions
  reduce(fid, outbuf, outbuf, op);
  check("reduce_field postcondition");
}

void
FEM_chunk::reduce(int fid, const void *inbuf, void *outbuf, int op)
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
FEM_chunk::readField(int fid, void *nodes, const char *fname)
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
      sscanf(pos, fmt, &IDXL_LAYOUT_DEREF(unsigned char,nodes,i,j), &numchars);
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
FEM_Async_Migrate(void)
{
  TCHARM_Async_Migrate();
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
  FEM_chunk::get(caller)->reduce_field(fid, nodes, outbuf, op);
}

CDECL void
FEM_Reduce(int fid, const void *inbuf, void *outbuf, int op)
{
  const char *caller="FEM_Reduce";FEMAPI(caller);
  FEM_chunk::get(caller)->reduce(fid, inbuf, outbuf, op);
}

CDECL void
FEM_Read_field(int fid, void *nodes, const char *fname)
{
  const char *caller="FEM_Read_field";FEMAPI(caller);
  FEM_chunk::get(caller)->readField(fid, nodes, fname);
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
  CkPrintf("[%d] %s\n", TCHARM_Element(), str);
}

static void do_print_partition(int fem_mesh,int idxBase) {
  
  const char *caller="FEM_Mesh_print"; FEMAPI(caller);
  FEM_chunk *cptr = FEM_chunk::get(caller);
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
	return FEM_chunk::get(caller)->getComm().size();
}
CDECL int FEM_Get_comm_partner(int partnerNo)
{
	const char *caller="FEM_Get_Comm_Partner"; FEMAPI(caller);
	return FEM_chunk::get(caller)->getComm().getLocalList(partnerNo).getDest();
}
CDECL int FEM_Get_comm_count(int partnerNo)
{
	const char *caller="FEM_Get_Comm_Count"; FEMAPI(caller);
	return FEM_chunk::get(caller)->getComm().getLocalList(partnerNo).size();
}
CDECL void FEM_Get_comm_nodes(int partnerNo,int *nodeNos)
{
	const char *caller="FEM_Get_comm_nodes"; FEMAPI(caller);
	const int *nNo=FEM_chunk::get(caller)->getComm().getLocalList(partnerNo).getVec();
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

FDECL void FTN_NAME(FEM_ASYNC_MIGRATE,fem_async_migrate)
  (void)
{
  FEM_Async_Migrate();
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
	const int *nNo=FEM_chunk::get(caller)->getComm().getLocalList(partnerNo).getVec();
	int len=FEM_Get_comm_count(partnerNo);
	for (int i=0;i<len;i++)
		nodeNos[i]=nNo[i]+1;
}

/******************** Ghost Stencil *********************/

/**
 Create a stencil with this number of elements, 
  and these adjacent elements.
 
 In C, userEnds[i] is the 0-based one-past-last element;
 in F90, userEnds[i] is the 1-based last element, which
 amounts to the same thing!
  
 In C, userAdj is 0-based; in F90, the even elType fields
 are 0-based, and the odd elemNum fields are 1-based.
*/
FEM_Ghost_Stencil::FEM_Ghost_Stencil(int elType_, int n_,bool addNodes_,
	const int *userEnds,
	const int *userAdj,
	int idxBase)
	:elType(elType_), n(n_), addNodes(addNodes_),  
	 ends(new int[n]), adj(new int[2*userEnds[n-1]])
{
	int i;
	int lastEnd=0;
	for (i=0;i<n;i++) {
		ends[i]=userEnds[i];
		if (ends[i]<lastEnd) {
			CkError("FEM_Ghost_Stencil> ends array not monotonic!\n"
				"   ends[%d]=%d < previous value %d\n",
				i,ends[i],lastEnd);
			CkAbort("FEM_Ghost_Stencil> ends array not monotonic");
		}
		lastEnd=ends[i];
	}
	int m=ends[n-1];
	for (i=0;i<m;i++) {
		adj[2*i+0]=userAdj[2*i+0];
		adj[2*i+1]=userAdj[2*i+1]-idxBase;
	}
}

CDECL void FEM_Add_ghost_stencil_type(int elType,int nElts,int addNodes,
	const int *ends,const int *adj2)
{
	FEMAPI("FEM_Add_ghost_stencil_type");
	FEM_Ghost_Stencil *s=new FEM_Ghost_Stencil(
		elType, nElts, addNodes==1, ends, adj2, 0);
	FEM_curPartition().addGhostStencil(s);
}
FDECL void FTN_NAME(FEM_ADD_GHOST_STENCIL_TYPE,fem_add_ghost_stencil_type)(
	int *elType,int *nElts,int *addNodes,
	const int *ends,const int *adj2)
{
	FEMAPI("FEM_Add_ghost_stencil_type");
	FEM_Ghost_Stencil *s=new FEM_Ghost_Stencil(
		*elType, *nElts, *addNodes==1, ends, adj2, 1);
	FEM_curPartition().addGhostStencil(s);
}


inline int globalElem2elType(const FEM_Mesh *m,int globalElem) {
	int curStart=0;
	for (int t=0;t<m->elem.size();t++) if (m->elem.has(t)) {
		int n=m->elem[t].size();
		if (globalElem>=curStart && globalElem<curStart+n)
			return t;
		curStart+=n;
	}
	CkError("FEM> Invalid global element number %d!\n",globalElem);
	CkAbort("FEM> Invalid global element number!");
	return 0;
}

/*
 These non "_type" routines describe element adjacencies
 using one big array, instead of one array per element type.
 This one-piece format is more convenient for most users.
*/
CDECL void FEM_Add_ghost_stencil(int nElts,int addNodes,
	const int *ends,const int *adj)
{
	FEMAPI("FEM_Add_ghost_stencil");
	const FEM_Mesh *m=setMesh();
	int adjSrc=0, endsSrc=0;
	for (int t=0;t<m->elem.size();t++) if (m->elem.has(t)) {
		const FEM_Elem &el=m->elem[t];
		int n=el.size();
		int nAdjLocal=ends[endsSrc+n-1]-adjSrc;
		int *adjLocal=new int[2*nAdjLocal];
		int *endsLocal=new int[n];
		int adjDest=0, endsDest=0;
		while (endsDest<n) {
			while (adjSrc<ends[endsSrc]) {
				/* Make adj array into elType, elNum pairs */
				int et=globalElem2elType(m,adj[adjSrc]);
				adjLocal[2*adjDest+0]=et;
				adjLocal[2*adjDest+1]=adj[adjSrc]-m->nElems(et);
				adjDest++; adjSrc++;
			}
			/* Use local end numbers in adjLocal array */
			endsLocal[endsDest]=adjDest;
			endsDest++; endsSrc++;
		}
		if (adjDest!=nAdjLocal)
			CkAbort("FEM_Add_ghost_stencil adjacency count mismatch!");
		FEM_Add_ghost_stencil_type(t,n,addNodes,endsLocal,adjLocal);
		delete []endsLocal;
		delete []adjLocal;
	}
	if (endsSrc!=nElts) {
		CkError("FEM_Add_ghost_stencil: FEM thinks there are %d elements, but nElts is %d!\n",endsSrc,nElts);
		CkAbort("FEM_Add_ghost_stencil elements mismatch!");
	}
	FEM_curPartition().markGhostStencilLayer();
}
FDECL void FTN_NAME(FEM_ADD_GHOST_STENCIL,fem_add_ghost_stencil)(
	int *nElts,int *addNodes,
	const int *ends,int *adj)
{
	FEMAPI("FEM_Add_ghost_stencil");
	int i, n=ends[*nElts-1];
	for (i=0;i<n;i++) adj[i]--; /* 1-based to 0-based */
	FEM_Add_ghost_stencil(*nElts,*addNodes,ends,adj);
	for (i=0;i<n;i++) adj[i]++; /* 0-based to 1-based */
}

/******************** Ghost Layers *********************/
CDECL void FEM_Add_ghost_layer(int nodesPerTuple,int doAddNodes)
{
	FEMAPI("FEM_Add_ghost_layer");
	FEM_Ghost_Layer *cur=FEM_curPartition().addLayer();
	cur->nodesPerTuple=nodesPerTuple;
	cur->addNodes=(doAddNodes!=0);
}
FDECL void FTN_NAME(FEM_ADD_GHOST_LAYER,fem_add_ghost_layer)
	(int *nodesPerTuple,int *doAddNodes)
{ FEM_Add_ghost_layer(*nodesPerTuple,*doAddNodes); }

static void add_ghost_elem(int elType,int tuplesPerElem,const int *elem2tuple,int idxBase) {
	FEMAPI("FEM_Add_ghost_elem");
	FEM_Ghost_Layer *cur=FEM_curPartition().curLayer();
	if (elType>=FEM_ELEM) elType-=FEM_ELEM;
	if (elType>=FEM_GHOST) elType-=FEM_GHOST;
	if (elType>=FEM_MAX_ELTYPE) CkAbort("Element exceeds (stupid) hardcoded maximum element types\n");
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
void FEM_chunk::exchangeGhostLists(int elemType,
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
		int nRecv=MPI_Incoming_pup(MPI_INT,src,tag,comm);
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
	FEM_chunk::get(caller)->exchangeGhostLists(elemType,nIdx,localIdx,0);
}
FDECL void FTN_NAME(FEM_EXCHANGE_GHOST_LISTS,fem_exchange_ghost_lists)
	(int *elemType,int *nIdx,const int *localIdx)
{
	const char *caller="FEM_exchange_ghost_lists"; FEMAPI(caller);
	FEM_chunk::get(caller)->exchangeGhostLists(*elemType,*nIdx,localIdx,1);
}
CDECL int FEM_Get_ghost_list_length(void) 
{
	const char *caller="FEM_Get_Ghost_List_Length"; FEMAPI(caller);
	return FEM_chunk::get(caller)->getList().size();
}
FDECL int FTN_NAME(FEM_GET_GHOST_LIST_LENGTH,fem_get_ghost_list_length)(void)
{ return FEM_Get_ghost_list_length();}

CDECL void FEM_Get_ghost_list(int *dest)
{
	const char *caller="FEM_Get_Ghost_List"; FEMAPI(caller);
	int i,len=FEM_Get_ghost_list_length();
	const int *src=FEM_chunk::get(caller)->getList().getVec();
	for (i=0;i<len;i++) dest[i]=src[i];
	FEM_chunk::get(caller)->emptyList();
}
FDECL void FTN_NAME(FEM_GET_GHOST_LIST,fem_get_ghost_list)
	(int *dest)
{
	const char *caller="FEM_get_ghost_list"; FEMAPI(caller);
	int i,len=FEM_Get_ghost_list_length();
	const int *src=FEM_chunk::get(caller)->getList().getVec();
	for (i=0;i<len;i++) dest[i]=src[i]+1;
	FEM_chunk::get(caller)->emptyList();
}


/********* Roccom utility interface *******/

/** Extract an IDXL_Side_t into Roccom format. */
static void getRoccomPconn(IDXL_Side_t is,int bias,CkVec<int> &pconn,const int *paneFmChunk)
{
	int p,np=IDXL_Get_partners(is);
	pconn.push_back(np);
	for (p=0;p<np;p++) {
		int chunk=IDXL_Get_partner(is,p);
		int pane=1+chunk;
		if(paneFmChunk) pane=paneFmChunk[chunk];
		pconn.push_back(pane);
		int n,nn=IDXL_Get_count(is,p);
		pconn.push_back(nn); /* number of shared nodes */
		for (n=0;n<nn;n++)
			pconn.push_back(IDXL_Get_index(is,p,n)+1+bias); /* nodes are 1-based */
	}
}

/** Extract all FEM communication information into Roccom format. */
static CkVec<int> getRoccomPconn(int fem_mesh,int *ghost_len,const int *paneFmChunk)
{
	CkVec<int> pconn;
	// Shared nodes come first:
	getRoccomPconn(IDXL_Get_send(FEM_Comm_shared(fem_mesh,FEM_NODE)),0,pconn,paneFmChunk);
	int realLen=pconn.size();
	
	// Sent ghost nodes:
	getRoccomPconn(IDXL_Get_send(FEM_Comm_ghost(fem_mesh,FEM_NODE)),0,pconn,paneFmChunk);
	// Received ghost nodes (use bias to switch to Roccom ghost node numbering)
	getRoccomPconn(IDXL_Get_recv(FEM_Comm_ghost(fem_mesh,FEM_NODE)),
		FEM_Mesh_get_length(fem_mesh,FEM_NODE),pconn,paneFmChunk);
	
// Handle elements (much tougher!)
	// Find list of element types
	int elems[1024];
	int e, ne=FEM_Mesh_get_entities(fem_mesh,elems);
	for (e=0;e<ne;e++)
		if (elems[e]<FEM_ELEM || elems[e]>=FEM_SPARSE)
			elems[e--]=elems[--ne]; // swap out bad entity with the end
	
	// Make one output IDXL that combines all element types:
	IDXL_t elghost=IDXL_Create();
	int out_r=0, out_g=0; // output indices for real; ghost
	for (e=0;e<ne;e++) {
		IDXL_Combine(elghost,FEM_Comm_ghost(fem_mesh,elems[e]), out_r,out_g);
		out_r+=FEM_Mesh_get_length(fem_mesh,elems[e]); 
		out_g+=FEM_Mesh_get_length(fem_mesh,elems[e]+FEM_GHOST);
	}
	
	// Sent ghost elements:
	getRoccomPconn(IDXL_Get_send(elghost),0,pconn,paneFmChunk);
	// Received ghost elements (shift all ghosts to start after real elements)
	getRoccomPconn(IDXL_Get_recv(elghost),out_r,pconn,paneFmChunk);
	IDXL_Destroy(elghost);
	
	if (ghost_len) *ghost_len=pconn.size()-realLen;
	return pconn;
}

CDECL void FEM_Get_roccom_pconn_size(int fem_mesh,int *total_len,int *ghost_len)
{
	CkVec<int> pconn=getRoccomPconn(fem_mesh,ghost_len,NULL);
	*total_len=pconn.size();
}
FORTRAN_AS_C(FEM_GET_ROCCOM_PCONN_SIZE,FEM_Get_roccom_pconn_size,fem_get_roccom_pconn_size,
        (int *mesh,int *tl,int *gl), (*mesh,tl,gl)
)

CDECL void FEM_Get_roccom_pconn(int fem_mesh,const int *paneFmChunk,int *dest)
{
	CkVec<int> pconn=getRoccomPconn(fem_mesh,NULL,paneFmChunk);
	for (unsigned int i=0;i<pconn.size();i++)
		dest[i]=pconn[i];
}
FORTRAN_AS_C(FEM_GET_ROCCOM_PCONN,FEM_Get_roccom_pconn,fem_get_roccom_pconn,
        (int *mesh,int *paneFmChunk,int *dest), (*mesh,paneFmChunk,dest)
)

int findChunk(int paneID,const int *paneFmChunk)
{
	for (int c=0;paneFmChunk[c]>0;c++)
		if (paneFmChunk[c]==paneID)
			return c;
	CkError("Can't find chunk for pconn paneID %d!\n",paneID);
	CmiAbort("Bad pconn array");
	return -1;
}

const int* makeIDXLside(const int *src,const int *paneFmChunk,IDXL_Side &s) {
	int nPartner=*src++;
	for (int p=0;p<nPartner;p++) {
		int paneID=*src++;
		int chunk=findChunk(paneID,paneFmChunk);
		int nComm=*src++;
		for (int i=0;i<nComm;i++)
			s.addNode(*src++ - 1,chunk); /* nodeID's are 1-based */
	}
	return src;
}

CDECL void FEM_Set_roccom_pconn(int fem_mesh,const int *paneFmChunk,const int *src,int total_len,int ghost_len)
{
	const char *caller="FEM_Set_roccom_pconn"; FEMAPI(caller);
	FEM_Mesh *m=FEM_Mesh_lookup(fem_mesh,caller);
	/// FIXME: only sets shared nodes for now (should grab ghosts too)
	makeIDXLside(src,paneFmChunk,m->node.shared);
}
FORTRAN_AS_C(FEM_SET_ROCCOM_PCONN,FEM_Set_roccom_pconn,fem_set_roccom_pconn,
        (int *mesh,const int *paneFmChunk,const int *src,int *tl,int *gl), (*mesh,paneFmChunk,src,*tl,*gl)
)

enum {comm_unshared, comm_shared, comm_primary};
int commState(int entityNo,const IDXL_Side &s)
{
	const IDXL_Rec *r=s.getRec(entityNo);
	if (r==NULL || r->getShared()==0) return comm_unshared;
	int thisChunk=FEM_My_partition();
	for (int p=0;p<r->getShared();p++) {
		if (r->getChk(p)<thisChunk) // not primary -- somebody else smaller
			return comm_shared;
		// FIXME: handle symmetry nodes here too (r->getChk(p)==thisChunk)
	}
	// If we get here, we're the smallest chunk holding this entity,
	//   so we're primary.
	return comm_primary;
}

/**
    Based on shared node communication list, compute 
    FEM_NODE FEM_GLOBALNO and FEM_NODE_PRIMARY
*/
CDECL void FEM_Make_node_globalno(int fem_mesh,FEM_Comm_t comm_context)
{
	const char *caller="FEM_Make_node_globalno"; FEMAPI(caller);
	FEM_Mesh *m=FEM_Mesh_lookup(fem_mesh,caller);
	int n, nNo=m->node.size();
	const IDXL_Side &shared=m->node.shared;
	CkVec<int> globalNo(nNo);
	CkVec<char> nodePrimary(nNo);
	
	// Figure out how each of our nodes is shared
	int nLocal=0;
	for (n=0;n<nNo;n++) {
		switch (commState(n,shared)) {
		case comm_unshared: 
			nodePrimary[n]=0;
			globalNo[n]=nLocal++; 
			break;
		case comm_shared: 
			nodePrimary[n]=0;
			globalNo[n]=-1; // will be filled in during sendsum, below
			break;
		case comm_primary: 
			nodePrimary[n]=1;
			globalNo[n]=nLocal++; 
			break;
		};
	}
	
	// Compute global numbers across processors
	//  as the sum of local (unshared and primary) nodes:
	MPI_Comm comm=(MPI_Comm)comm_context;
	int firstGlobal=0; // global number of first local element
	MPI_Scan(&nLocal,&firstGlobal, 1,MPI_INT, MPI_SUM,comm);
	firstGlobal-=nLocal; /* sum of all locals before me, but *not* including */
	for (n=0;n<nNo;n++) {
		if (globalNo[n]==-1) globalNo[n]=0;
		else globalNo[n]+=firstGlobal;
	}
	
	// Get globalNo for shared nodes, by copying from primary.
	IDXL_Layout_t l=IDXL_Layout_create(IDXL_INT,1);
	IDXL_Comm_t c=IDXL_Comm_begin(72173841,comm_context);
	IDXL_Comm_sendsum(c,FEM_Comm_shared(fem_mesh,FEM_NODE),l,&globalNo[0]);
	IDXL_Comm_wait(c);
	IDXL_Layout_destroy(l);
	
	// Copy globalNo and primary into fem
	FEM_Mesh_set_data(fem_mesh,FEM_NODE, FEM_GLOBALNO,
		&globalNo[0], 0,nNo, FEM_INDEX_0,1);
	FEM_Mesh_set_data(fem_mesh,FEM_NODE, FEM_NODE_PRIMARY,
		&nodePrimary[0], 0,nNo, FEM_BYTE,1);
}

FORTRAN_AS_C(FEM_MAKE_NODE_GLOBALNO,FEM_Make_node_globalno,fem_make_node_globalno,
        (int *mesh,int *comm), (*mesh,*comm)
)

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
FEM_chunk::print(int fem_mesh,int idxBase)
{
  CkPrintf("-------------------- Chunk %d --------------------\n",thisIndex);
  lookup(fem_mesh,"FEM_Mesh_print")->print(idxBase);
  CkPrintf("\n\n");
}  


/** 
 *  Register a set of tuples for a single element type
 *  Must be called once for each element type.
 * 
 *  After a sequence of calls to this function, a call 
 *  should be made to
 *  FEM_Mesh_create_elem_elem_adjacency(int fem_mesh)
 *
 *   TODO: Make this work with multiple types of faces for a single element
 */

CDECL void FEM_Add_elem2face_tuples(int fem_mesh, int elem_type, int nodesPerTuple, int tuplesPerElem,const int *elem2tuple) 
{
	FEMAPI("FEM_Add_elem2face_tuples");
	FEM_Mesh *m = FEM_Mesh_lookup(fem_mesh,"FEM_Add_elem2face_tuples");
	FEM_ElemAdj_Layer *cur = m->getElemAdjLayer(); // creates the structure on first access
	cur->initialized = 1;

	if(cur->elem[elem_type].tuplesPerElem != 0)
	  CkPrintf("FEM> WARNING: Don't call FEM_Add_elem2face_tuples() repeatedly with the same element type\n");
	
	int idxBase=0;
	cur->nodesPerTuple=nodesPerTuple;
	cur->elem[elem_type].tuplesPerElem=tuplesPerElem;
	cur->elem[elem_type].elem2tuple=CkCopyArray(elem2tuple, tuplesPerElem*cur->nodesPerTuple,idxBase);

}
FORTRAN_AS_C(FEM_ADD_ELEM2FACE_TUPLES,FEM_Add_elem2face_tuples,fem_add_elem2face_tuples,
  (int *fem_mesh,int *elem_type,int *nodesPerTuple, int *tuplesPerElem, int *elem2tuple), 
  (*fem_mesh, *elem_type, *nodesPerTuple, *tuplesPerElem, elem2tuple)
)

/*@}*/
