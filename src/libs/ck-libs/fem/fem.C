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

void FEM_Abort(const char *msg) {
	CkAbort(msg);
}
void FEM_Abort(const char *callingRoutine,const char *sprintf_msg,
   int int0,int int1, int int2) 
{
	char userSprintf[1024];
	char msg[1024];
	sprintf(userSprintf,sprintf_msg,int0,int1,int2);
	sprintf(msg,"FEM Routine %s fatal error:\n          %s",
		callingRoutine,userSprintf);
	FEM_Abort(msg);
}


CDECL void fem_impl_call_init(void);

FDECL void FTN_NAME(INIT,init)(void);
FDECL void FTN_NAME(DRIVER,driver)(void);

/*Startup:*/
static void callDrivers(void) {
        driver();
#ifndef CMK_FORTRAN_USES_NOSCORE
        FTN_NAME(DRIVER,driver)();
#endif
}

static int initFlags=0;

static void FEMfallbackSetup(void)
{
	int nChunks=TCHARM_Get_num_chunks();
	TCHARM_Create(nChunks,callDrivers);
	if (!(initFlags&FEM_INIT_READ)) {
		fem_impl_call_init(); // init();
#ifndef CMK_FORTRAN_USES_NOSCORE
		FTN_NAME(INIT,init)();
#endif
	}
        FEM_Attach(initFlags);
}

//_femptr gives the current chunk, and is only
// valid in routines called from driver().
CtvStaticDeclare(FEMchunk*, _femptr);

PUPable_def(FEM_Sym_Linear);
void FEMnodeInit(void) {
	PUPable_reg(FEM_Sym_Linear);
	CtvInitialize(FEMchunk*, _femptr);
	TCHARM_Set_fallback_setup(FEMfallbackSetup);
	CmiArgGroup("Library","FEM Framework");
	char **argv=CkGetArgv();
	if (CmiGetArgFlagDesc(argv,"-read","Skip init()--read mesh from files")) 
		initFlags|=FEM_INIT_READ;
	if (CmiGetArgFlagDesc(argv,"-write","Skip driver()--write mesh to files")) 
		initFlags|=FEM_INIT_WRITE;
}

static void 
_allReduceHandler(void *proxy_v, int datasize, void *data)
{
  // the reduction handler is called on processor 0
  CProxy_FEMchunk &proxy=*(CProxy_FEMchunk *)proxy_v;
  // broadcast the reduction results to all array elements
  proxy.reductionResult(datasize,(char *)data);
}


//These fields give the current serial mesh
// (NULL if none).  They are only valid during
// init, mesh_updated, and finalize.
static FEM_Mesh* _meshptr = 0;

//Maps element number to (0-based) chunk number, allocated with new[]
static int *_elem2chunk=NULL;


static FEM_Ghost ghosts;
static ghostLayer *curGhostLayer=NULL;

//Partitions and splits the current serial mesh into the given number of pieces
static void mesh_split(int _nchunks,FEM_Mesh_Output *out) {
    int *elem2chunk=_elem2chunk;
    if (elem2chunk==NULL) 
    {//Partition the elements ourselves
    	elem2chunk=new int[_meshptr->nElems()];
    	FEM_Mesh_partition(_meshptr,_nchunks,elem2chunk);
    }
    _meshptr->setAscendingGlobalno();
    //Build communication lists and split mesh data
    FEM_Mesh_split(_meshptr,_nchunks,elem2chunk,ghosts,out);
    //Blow away old partitioning
    delete[] elem2chunk; _elem2chunk=NULL;
    delete _meshptr; _meshptr=NULL;
}

class FEM_Mesh_Writer : public FEM_Mesh_Output {
  int nchunks;
public:
  FEM_Mesh_Writer(int nc) :nchunks(nc) {}
  void accept(int chunkNo,FEM_Mesh *chk);
};

void FEM_Mesh_Writer::accept(int chunkNo,FEM_Mesh *chk)
{
	FEM_Mesh_write(chk,chunkNo,nchunks,NULL);
	delete chk;
}

class FEM_Mesh_Sender : public FEM_Mesh_Output {
	CProxy_FEMchunk dest;
public:
	FEM_Mesh_Sender(const CProxy_FEMchunk &dest_)
		:dest(dest_) {}
	void accept(int chunkNo,FEM_Mesh *chk)
	{
		dest[chunkNo].run(chk);
		delete chk;
	}
};

static void RestoreTCharmGlobals(int _nchunks) {
  PUP::fromTextFile pg(FEM_openMeshFile(_nchunks,_nchunks,true));
  TCharmReadonlys::pupAllReadonlys(pg);
}
static void SaveTCharmGlobals(int _nchunks) {
  PUP::toTextFile pg(FEM_openMeshFile(_nchunks,_nchunks,false));
  TCharmReadonlys::pupAllReadonlys(pg);
}

FDECL void FTN_NAME(FEM_ATTACH,fem_attach)(int *flags) 
{
	FEM_Attach(*flags);
}

CDECL void FEM_Attach(int flags)
{
	FEMAPI("FEM_Attach");
	CkArrayID threadsAID; int _nchunks;
	CkArrayOptions opts=TCHARM_Attach_start(&threadsAID,&_nchunks);
	
	if (flags&FEM_INIT_WRITE) 
	{ //First save the user's globals (if any):
		SaveTCharmGlobals(_nchunks);
		
		//Split the mesh and write it out
		FEM_Mesh_Writer w(_nchunks);
		mesh_split(_nchunks,&w);
		CkExit();
	}
	if (flags&FEM_INIT_READ)
	{ //Restore the user's globals on PE 0-- they'll get copied elsewhere
	  RestoreTCharmGlobals(_nchunks);
	}
	
	//Create a new chunk array
	CProxy_FEMcoordinator coord=CProxy_FEMcoordinator::ckNew(_nchunks);
	FEMinit init(_nchunks,threadsAID,flags,coord);
	
	CProxy_FEMchunk chunks= CProxy_FEMchunk::ckNew(init,opts);
	chunks.setReductionClient(_allReduceHandler, new CProxy_FEMchunk(chunks));
	coord.setArray(chunks);
	TCHARM_Attach_finish(chunks);
	
	//Send the mesh out to the chunks
	if (_meshptr!=NULL) 
	{ //Partition the serial mesh online
		FEM_Mesh_Sender s(chunks);
		mesh_split(_nchunks,&s);
	} else /*NULL==mesh*/ 
	{ //Each chunk will just read the mesh locally
		chunks.run();
	}
}

//This coordinator manages mesh reassembly for a FEM array:
class FEMcoordinator : public Chare {
	int nChunks; //Number of array elements total
	CProxy_FEMchunk femChunks;
	FEM_Mesh **cmsgs; //Messages from/for array elements
	int updateCount; //Number of mesh updates so far
	CkQ<UpdateMeshChunk *> futureUpdates;
	CkQ<UpdateMeshChunk *> curUpdates; 
	int numdone; //Length of curUpdates 
public:
	FEMcoordinator(int nChunks_) 
		:nChunks(nChunks_)
	{
		cmsgs=new FEM_Mesh*[nChunks]; CHK(cmsgs);		
		numdone=0;
		updateCount=1;
	}
	~FEMcoordinator() {
		delete[] cmsgs;
	}
	
	void setArray(const CkArrayID &fem_) {femChunks=fem_;}
	void updateMesh(marshallUpdateMeshChunk &chk);
};

class FEM_Mesh_Update : public FEM_Mesh_Output {
	CProxy_FEMchunk dest;
public:
	FEM_Mesh_Update(const CProxy_FEMchunk &dest_)
		:dest(dest_) {}
	void accept(int chunkNo,FEM_Mesh *chk)
	{
		dest[chunkNo].meshUpdated(chk);
		delete chk;
	}
};

//Called by a chunk on FEM_Update_Mesh
void FEMcoordinator::updateMesh(marshallUpdateMeshChunk &chk)
{
  UpdateMeshChunk *msg=chk;
  if (msg->updateCount>updateCount) {
    //This is a message for a future update-- save it for later
    futureUpdates.enq(msg);
  } else if (msg->updateCount<updateCount) 
    FEM_Abort("FEM_Update_mesh","Received mesh chunk from iteration %d while on iteration %d",
    	msg->updateCount,updateCount);
  else /*(msg->updateCount==updateCount)*/{
    int _nchunks=nChunks;
    //A chunk for the current mesh
    curUpdates.enq(msg);
    while (curUpdates.length()==_nchunks) {
      //We have all the chunks of the current mesh-- process them and start over
      int i;
      //Save what to do with the mesh
      CallMeshUpdated meshUpdated;
      int doWhat;
      
      for (i=0;i<_nchunks;i++) {
      	UpdateMeshChunk *m=curUpdates.deq();
	cmsgs[m->fromChunk]=&m->m;
	meshUpdated=m->meshUpdated;
	doWhat=m->doWhat;
      }
      //Assemble the current chunks into a serial mesh
      delete _meshptr;
      _meshptr=FEM_Mesh_assemble(_nchunks,cmsgs);
      //Blow away the old chunks
      for (i=0;i<_nchunks;i++) {
      	delete cmsgs[i];
      	cmsgs[i]=NULL;
      }

      //Now that the mesh is assembled, handle it
      TCharm::setState(inInit);
      meshUpdated.call();
      TCharm::setState(inDriver);
      
      if (doWhat==FEM_MESH_UPDATE) { /*repartition the mesh*/
	FEM_Mesh_Update u(femChunks);
	mesh_split(_nchunks,&u);
      } else if (doWhat==FEM_MESH_FINALIZE) { /*just broadcast meshUpdatedComplete*/
        femChunks.meshUpdatedComplete();
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

/*******************************************************
  Code used only by serial framework clients-- they want
  to set the mesh, partition, and get the mesh out piece by piece.
*/
class FEM_Mesh_Storer : public FEM_Mesh_Output {
  int nchunks;
  FEM_Mesh **chks;
public:
  FEM_Mesh_Storer(int nc) :nchunks(nc) {
    typedef FEM_Mesh* FEM_Mesh_Ptr;
    chks=new FEM_Mesh_Ptr[nc];
  }
  ~FEM_Mesh_Storer() {
    for (int i=0;i<nchunks;i++) delete chks[i];
    delete[] chks;
  }
  void accept(int chunkNo,FEM_Mesh *chk) {
    chks[chunkNo]=chk;
  }
  void useChunk(int i) {
    if (i<0 || i>=nchunks) 
    	FEM_Abort("FEM_Serial_begin","Invalid index %d: must be between 0 and %d",i,nchunks);
    FEM_Mesh_write(chks[i],i,nchunks,NULL);
    _meshptr=chks[i];
  }
};

static FEM_Mesh_Storer *meshChunkStore=NULL;
CDECL void FEM_Serial_split(int npieces) {
  FEMAPI("FEM_Serial_split");
  meshChunkStore=new FEM_Mesh_Storer(npieces);
  mesh_split(npieces,meshChunkStore);
  SaveTCharmGlobals(npieces);
}
FDECL void FTN_NAME(FEM_SERIAL_SPLIT,fem_serial_split)(int *npieces)
{ 
  FEM_Serial_split(*npieces); 
}

CDECL void FEM_Serial_begin(int chunkNo) {
  FEMAPI("FEM_Serial_begin");
  if (!meshChunkStore) FEM_Abort("Can't call FEM_Serial_begin before FEM_Serial_split!");
  meshChunkStore->useChunk(chunkNo);
}
FDECL void FTN_NAME(FEM_SERIAL_BEGIN,fem_serial_begin)(int *pieceNo)
{
  FEM_Serial_begin(*pieceNo-1);
}


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

FEMchunk *FEMchunk::lookup(const char *callingRoutine) {
	if(TCharm::getState()!=inDriver) 
		FEM_Abort(callingRoutine,"Can only be called from driver (from parallel context)");
	return CtvAccess(_femptr);
}

static FEM_Mesh *setMesh(void) {
  if(TCharm::getState()==inDriver) {
    FEMchunk *cptr = CtvAccess(_femptr);
    if (cptr->updated_mesh==NULL) {
      cptr->updated_mesh=new UpdateMeshChunk;
    }
    return &cptr->updated_mesh->m;
  } else {
    //Called from init, finalize, or meshUpdate
    if (_meshptr==NULL) {
      _meshptr=new FEM_Mesh;
    }
    return _meshptr;
  }
}

static const FEM_Mesh *getMesh(void) {
  if(TCharm::getState()==inDriver) {
    FEMchunk *cptr = CtvAccess(_femptr);
    return &cptr->getMesh();
  } else {
    //Called from init, finalize, or meshUpdate
    if (_meshptr==NULL) {
      FEM_Abort("FEM: Cannot get mesh-- it was never set!\n");
    }
    return _meshptr;
  }
}
const FEM_Mesh *FEM_Get_FEM_Mesh(void) {
	return getMesh();
}

CDECL int FEM_Mesh_default_read(void)  /* return default fem_mesh used for read (get) calls below */
{
	return 102;
}
CDECL int FEM_Mesh_default_write(void) /* return default fem_mesh used for write (set) calls below */
{
	return 103;
}

FEM_Mesh *FEMchunk::meshLookup(int fem_mesh,const char *callingRoutine) {
	switch (fem_mesh) {
	case 102: return cur_mesh;
	case 103: return ::setMesh();
	default: //Invalid mesh ID:
		FEM_Abort(callingRoutine,"Invalid fem_mesh ID %d",fem_mesh);
		return NULL; //<- for whining compilers
	};
}


FEM_Mesh *FEM_Mesh_lookup(int fem_mesh,const char *callingRoutine) {
	switch (fem_mesh) {
	case 102: return (FEM_Mesh *)getMesh();
	case 103: return setMesh();
	default: //Invalid mesh ID:
		FEM_Abort(callingRoutine,"Invalid fem_mesh ID %d",fem_mesh);
		return NULL; //<- for whining compilers
	};
}


/****** Custom Partitioning API *******/
static void Set_Partition(int *elem2chunk,int indexBase) {
	if (_elem2chunk!=NULL) delete[] _elem2chunk;
	_elem2chunk=CkCopyArray(elem2chunk,getMesh()->nElems(),indexBase);
}

//C bindings:
CDECL void FEM_Set_partition(int *elem2chunk) {
	FEMAPI("FEM_Set_partition");
	Set_Partition(elem2chunk,0);
}

//Fortran bindings:
FDECL void FTN_NAME(FEM_SET_PARTITION,fem_set_partition)
	(int *elem2chunk) 
{
	FEMAPI("FEM_Set_partition");
	Set_Partition(elem2chunk,1);
}


/******************************* CHUNK *********************************/

FEMchunk::FEMchunk(const FEMinit &init_)
	:super(init_.threads), init(init_), thisproxy(thisArrayID)
{
  initFields();

  updated_mesh=NULL;
  cur_mesh=NULL;

  updateCount=1; //Number of mesh updates

//Field updates:
  reductionBuf=NULL;
  listCount=0;listSuspended=false;
}
FEMchunk::FEMchunk(CkMigrateMessage *msg)
	:super(msg), thisproxy(thisArrayID)
{
  updated_mesh=NULL;
  cur_mesh=NULL;
  reductionBuf=NULL;	
  listCount=0;listSuspended=false;
}

FEMchunk::~FEMchunk()
{
	delete updated_mesh;
	delete cur_mesh;
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
  super::ckJustMigrated(); //Call superclass
  initFields();
}

void
FEMchunk::run(marshallMeshChunk &msg)
{
  setMesh(msg);
  thread->ready();
}

void
FEMchunk::run(void)
{
  setMesh();
  thread->ready();
}

void
FEMchunk::setMesh(FEM_Mesh *msg)
{
  if (cur_mesh!=NULL) delete cur_mesh;
  if(msg==0) { /*Read mesh from file*/
    cur_mesh=FEM_Mesh_read(thisIndex,init.numElements,NULL);
  } else {
    cur_mesh=msg;
  }
  cur_mesh->registerIDXL(this);
}


void
FEMchunk::reduce_field(int fid, const void *nodes, void *outbuf, int op)
{
  // first reduce over local nodes
  const IDXL_Layout &dt = layouts.get(fid,"FEM_Reduce_field");
  const byte *src = (const byte *) nodes;
  reduction_initialize(dt,outbuf,op);
  reduction_combine_fn fn=reduction_combine(dt,op);
  for(int i=0; i<cur_mesh->node.size(); i++) {
    if(getPrimary(i)) {
      fn((byte *)outbuf, src, &dt);
    }
    src += dt.userBytes();
  }
  // and now reduce over partitions
  reduce(fid, outbuf, outbuf, op);
}

void
FEMchunk::reduce(int fid, const void *inbuf, void *outbuf, int op)
{
  const IDXL_Layout &dt = layouts.get(fid,"FEM_Reduce");
  int len = dt.compressedBytes();
  if(numElements==1) {
    memcpy(outbuf,inbuf,len);
    return;
  }
  CkReduction::reducerType rtype;
  switch(op) {
    case FEM_SUM:
      switch(dt.type) {
        case FEM_INT: rtype = CkReduction::sum_int; break;
        case FEM_REAL: rtype = CkReduction::sum_float; break;
        case FEM_DOUBLE: rtype = CkReduction::sum_double; break;
      }
      break;
    case FEM_PROD:
      switch(dt.type) {
        case FEM_INT: rtype = CkReduction::product_int; break;
        case FEM_REAL: rtype = CkReduction::product_float; break;
        case FEM_DOUBLE: rtype = CkReduction::product_double; break;
      }
      break;
    case FEM_MAX:
      switch(dt.type) {
        case FEM_INT: rtype = CkReduction::max_int; break;
        case FEM_REAL: rtype = CkReduction::max_float; break;
        case FEM_DOUBLE: rtype = CkReduction::max_double; break;
      }
      break;
    case FEM_MIN:
      switch(dt.type) {
        case FEM_INT: rtype = CkReduction::min_int; break;
        case FEM_REAL: rtype = CkReduction::min_float; break;
        case FEM_DOUBLE: rtype = CkReduction::min_double; break;
      }
      break;
  }
  contribute(len, (void *)inbuf, rtype);
  reductionBuf = outbuf;
  thread->suspend();
}

void
FEMchunk::reductionResult(int length,const char *data)
{
  memcpy(reductionBuf, data, length);
  reductionBuf=NULL;
  thread->resume();
}

//Called by user to ask us to contribute our updated mesh chunk
void 
FEMchunk::updateMesh(int doWhat) {
  if (updated_mesh==NULL)
    CkAbort("FEM_Update_Mesh> You must first set the mesh before updating it!\n");
  
  if (cur_mesh) {
     updated_mesh->m.copyOldGlobalno(*cur_mesh);
  }
  
  updated_mesh->doWhat=doWhat;
  updated_mesh->updateCount=updateCount++;
  updated_mesh->fromChunk=thisIndex;

  //Send the mesh off to the coordinator
  CProxy_FEMcoordinator coord(init.coordinator);
  coord.updateMesh(updated_mesh);
  delete updated_mesh;
  updated_mesh=NULL;
  if (doWhat!=FEM_MESH_OUTPUT)
    thread->suspend();//Sleep until repartitioned mesh arrives
}

//Called by coordinator with a new, repartitioned mesh chunk for us
void 
FEMchunk::meshUpdated(marshallMeshChunk &msg) {
  setMesh(msg); //Read in the new mesh
  thread->resume();  //Start computing again
}

void
FEMchunk::readField(int fid, void *nodes, const char *fname)
{
  const IDXL_Layout &dt = layouts.get(fid,"FEM_Read_field");
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
  for(i=0;i<cur_mesh->node.size();i++) {
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
static FEMchunk *getCurChunk(void) 
{
  FEMchunk *cptr=CtvAccess(_femptr);
  if (cptr==NULL) 
    CkAbort("Routine can only be called from driver()!\n");
  return cptr;
}

CDECL void FEM_Update_mesh(FEM_Update_mesh_fn callFn,int userValue,int doWhat) 
{ 
  FEMAPI("FEM_Update_mesh");
  FEMchunk *chk=getCurChunk();
  chk->updated_mesh->meshUpdated=CallMeshUpdated(callFn,userValue);
  chk->updateMesh(doWhat); 
}

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
  FEMAPI("FEM_Reduce_field");
  getCurChunk()->reduce_field(fid, nodes, outbuf, op);
}

CDECL void
FEM_Reduce(int fid, const void *inbuf, void *outbuf, int op)
{
  FEMAPI("FEM_Reduce");
  getCurChunk()->reduce(fid, inbuf, outbuf, op);
}

CDECL void
FEM_Read_field(int fid, void *nodes, const char *fname)
{
  FEMAPI("FEM_Read_field");
  getCurChunk()->readField(fid, nodes, fname);
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
  TCharmAPIRoutine apiRoutineSentry;
  if(TCharm::getState()==inDriver) {
    FEMchunk *cptr = getCurChunk();
    CkPrintf("[%d] %s\n", cptr->thisIndex, str);
  } else {
    CkPrintf("%s\n", str);
  }
}

static void do_print_partition(int idxBase) {
  FEMAPI("FEM_Print_Partition");
  if(TCharm::getState()==inDriver) {
    FEMchunk *cptr = getCurChunk();
    cptr->print(idxBase);
  } else {
    if (_meshptr==NULL)
      CkPrintf("[%d] No serial mesh available.\n",FEM_My_partition());
    else
      _meshptr->print(idxBase);
  }
}

CDECL void 
FEM_Print_partition(void)
{
  do_print_partition(0);
}

CDECL int FEM_Get_comm_partners(void)
{
	FEMAPI("FEM_Get_Comm_Partners");
	return getCurChunk()->getComm().size();
}
CDECL int FEM_Get_comm_partner(int partnerNo)
{
	FEMAPI("FEM_Get_Comm_Partner");
	return getCurChunk()->getComm().getLocalList(partnerNo).getDest();
}
CDECL int FEM_Get_comm_count(int partnerNo)
{
	FEMAPI("FEM_Get_Comm_Count");
	return getCurChunk()->getComm().getLocalList(partnerNo).size();
}
CDECL void FEM_Get_comm_nodes(int partnerNo,int *nodeNos)
{
	FEMAPI("FEM_Get_comm_nodes");
	const int *nNo=getCurChunk()->getComm().getLocalList(partnerNo).getVec();
	int len=FEM_Get_comm_count(partnerNo);
	for (int i=0;i<len;i++)
		nodeNos[i]=nNo[i];
}

/************************ Fortran Bindings *********************************/
FDECL void FTN_NAME(FEM_UPDATE_MESH,fem_update_mesh)
  (FEM_Update_mesh_fortran_fn callFn,int *userValue,int *doWhat) 
{ 
  FEMAPI("FEM_Update_mesh");
  FEMchunk *chk=getCurChunk();
  chk->updated_mesh->meshUpdated=CallMeshUpdated(callFn,*userValue);
  chk->updateMesh(*doWhat);
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
  TCharmAPIRoutine apiRoutineSentry;
  char *tmpstr = new char[len+1]; CHK(tmpstr);
  memcpy(tmpstr,str,len);
  tmpstr[len] = '\0';
  FEM_Print(tmpstr);
  delete[] tmpstr;
}

FDECL void FTN_NAME(FEM_PRINT_PARTITION,fem_print_partition)
  (void)
{
  do_print_partition(1);
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
	int partnerNo=*pNo-1;
	const int *nNo=getCurChunk()->getComm().getLocalList(partnerNo).getVec();
	int len=FEM_Get_comm_count(partnerNo);
	for (int i=0;i<len;i++)
		nodeNos[i]=nNo[i]+1;
}

/******************** Ghost Layers *********************/
FEM_Ghost &FEM_Set_FEM_Ghost(void) {
	if(TCharm::getState()==inDriver)
		CkAbort("FEM: Cannot call ghost or symmetry routines from driver!");
	return ghosts;
}

CDECL void FEM_Add_ghost_layer(int nodesPerTuple,int doAddNodes)
{
	FEMAPI("FEM_Add_ghost_layer");
	curGhostLayer=FEM_Set_FEM_Ghost().addLayer();
	curGhostLayer->nodesPerTuple=nodesPerTuple;
	curGhostLayer->addNodes=(doAddNodes!=0);
	curGhostLayer->elem.makeLonger(getMesh()->elem.size());
}
FDECL void FTN_NAME(FEM_ADD_GHOST_LAYER,fem_add_ghost_layer)
	(int *nodesPerTuple,int *doAddNodes)
{ FEM_Add_ghost_layer(*nodesPerTuple,*doAddNodes); }

CDECL void FEM_Add_ghost_elem(int elType,int tuplesPerElem,const int *elem2tuple)
{
	FEMAPI("FEM_Add_ghost_elem");
	if (curGhostLayer==NULL)
		CkAbort("You must call FEM_Add_Ghost_Layer before calling FEM_Add_Ghost_elem!\n");
	curGhostLayer->elem[elType].add=true;
	curGhostLayer->elem[elType].tuplesPerElem=tuplesPerElem;
	curGhostLayer->elem[elType].elem2tuple=CkCopyArray(elem2tuple,
		          tuplesPerElem*curGhostLayer->nodesPerTuple,0);
}
FDECL void FTN_NAME(FEM_ADD_GHOST_ELEM,fem_add_ghost_elem)
	(int *FelType,int *FtuplesPerElem,const int *elem2tuple)
{
	FEMAPI("FEM_add_ghost_elem");
	int elType=*FelType;
	int tuplesPerElem=*FtuplesPerElem;
	if (curGhostLayer==NULL)
		CkAbort("You must call FEM_Add_Ghost_Layer before calling FEM_Add_Ghost_elem!\n");
	getMesh()->chkET(elType);
	curGhostLayer->elem[elType].add=true;
	curGhostLayer->elem[elType].tuplesPerElem=tuplesPerElem;
	curGhostLayer->elem[elType].elem2tuple=CkCopyArray(elem2tuple,
		          tuplesPerElem*curGhostLayer->nodesPerTuple,1);
}

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
	const FEM_Comm &cnt=cur_mesh->getCount(elemType).getGhostSend();
	
//Send off a list to each neighbor
	int nChk=cnt.size();
	CkVec<int> *outIdx=new CkVec<int>[nChk];
	//Loop over (the shared entries in) the input list
	for (int i=0;i<inLen;i++) {
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
	for (int chk=0;chk<nChk;chk++) {
		thisproxy[cnt.getLocalList(chk).getDest()].recvList(
			elemType,thisIndex,
			outIdx[chk].size(), outIdx[chk].getVec()
		);
	}
	delete[] outIdx;
	
//Check if the replies have all arrived
	if (!finishListExchange(cur_mesh->getCount(elemType).getGhostRecv())){
		listSuspended=true;
		thread->suspend(); //<- sleep until all lists arrive
	}
	else //We have everything we need-- reset and continue
		listCount=0;
}

void FEMchunk::recvList(int elemType,int fmChk,int nIdx,const int *idx)
{
	int i;
	const FEM_Entity &e=cur_mesh->getCount(elemType);
	int firstGhost=e.size();
	const FEM_Comm_List &l=e.getGhostRecv().getList(fmChk);
	for (i=0;i<nIdx;i++)
		listTmp.push_back(firstGhost+l[idx[i]]);
	listCount++;
	finishListExchange(cur_mesh->getCount(elemType).getGhostRecv());
}
bool FEMchunk::finishListExchange(const FEM_Comm &l)
{
	if (listCount<l.size()) return false; //Not finished yet!
	if (listSuspended) {
		listSuspended=false;
		thread->resume();
		listCount=0;
	}
	return true;
}

//List exchange API
CDECL void FEM_Exchange_ghost_lists(int elemType,int nIdx,const int *localIdx)
{
	FEMAPI("FEM_Exchange_Ghost_Lists");
	getCurChunk()->exchangeGhostLists(elemType,nIdx,localIdx,0);
}
FDECL void FTN_NAME(FEM_EXCHANGE_GHOST_LISTS,fem_exchange_ghost_lists)
	(int *elemType,int *nIdx,const int *localIdx)
{
	FEMAPI("FEM_exchange_ghost_lists");
	getCurChunk()->exchangeGhostLists(*elemType,*nIdx,localIdx,1);
}
CDECL int FEM_Get_ghost_list_length(void) 
{
	FEMAPI("FEM_Get_Ghost_List_Length");
	return getCurChunk()->getList().size();
}
FDECL int FTN_NAME(FEM_GET_GHOST_LIST_LENGTH,fem_get_ghost_list_length)(void)
{ return FEM_Get_ghost_list_length();}

CDECL void FEM_Get_ghost_list(int *dest)
{
	FEMAPI("FEM_Get_Ghost_List");
	int i,len=FEM_Get_ghost_list_length();
	const int *src=getCurChunk()->getList().getVec();
	for (i=0;i<len;i++) dest[i]=src[i];
	getCurChunk()->emptyList();
}
FDECL void FTN_NAME(FEM_GET_GHOST_LIST,fem_get_ghost_list)
	(int *dest)
{
	FEMAPI("FEM_get_ghost_list");
	int i,len=FEM_Get_ghost_list_length();
	const int *src=getCurChunk()->getList().getVec();
	for (i=0;i<len;i++) dest[i]=src[i]+1;
	getCurChunk()->emptyList();
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
FEMchunk::print(int idxBase)
{
  CkPrintf("-------------------- Chunk %d --------------------\n",thisIndex);
  cur_mesh->print(idxBase);
  CkPrintf("\n\n");
}  
  

/***************** Mesh Utility Classes **********/

void
FEMchunk::pup(PUP::er &p)
{
//Pup superclass
  super::pup(p);

// Pup the mesh fields
  bool hasUpdate=(updated_mesh!=NULL);
  p(hasUpdate);
  if(p.isUnpacking())
  {
    cur_mesh=new FEM_Mesh;
    if (hasUpdate) updated_mesh=new UpdateMeshChunk;
  }
  if (hasUpdate) updated_mesh->pup(p);
  cur_mesh->pup(p);
  p|updateCount;

//Pup all other fields
  init.pup(p);

  p(doneCalled);
}

#include "fem.def.h"
