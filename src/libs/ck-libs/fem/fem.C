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
	char **argv=CkGetArgv();
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
  FEM_DataMsg *dmsg = new (&datasize, 0) FEM_DataMsg(0,0,0,datasize); CHK(dmsg);
  memcpy(dmsg->data, data, datasize);
  CProxy_FEMchunk &proxy=*(CProxy_FEMchunk *)proxy_v;
  // broadcast the reduction results to all array elements
  proxy.reductionResult(dmsg);
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
static void mesh_split(int _nchunks,MeshChunkOutput *out) {
    int *elem2chunk=_elem2chunk;
    if (elem2chunk==NULL) 
    {//Partition the elements ourselves
    	elem2chunk=new int[_meshptr->nElems()];
    	fem_partition(_meshptr,_nchunks,elem2chunk);
    }
    //Build communication lists and split mesh data
    fem_split(_meshptr,_nchunks,elem2chunk,ghosts,out);
    //Blow away old partitioning
    delete[] elem2chunk; _elem2chunk=NULL;
    delete _meshptr; _meshptr=NULL;
}

static const char *meshFileNames="fem_mesh_vp%d_%d.dat";

static FILE *openMeshFile(int chunkNo,int nchunks,bool forRead)
{
    char fname[256];
    sprintf(fname, meshFileNames, nchunks, chunkNo);
    FILE *fp = fopen(fname, forRead?"r":"w");
    CkPrintf("FEM> %s %s...\n",forRead?"Reading":"Writing",fname);  
    if(fp==0) {
      CkAbort(forRead?"FEM: unable to open input file"
      	:"FEM: unable to create output file.\n");
    }
    return fp;
}

class MeshChunkOutputWriter : public MeshChunkOutput {
  int nchunks;
public:
  MeshChunkOutputWriter(int nc) :nchunks(nc) {}
  void accept(int chunkNo,MeshChunk *chk);
};

void MeshChunkOutputWriter::accept(int chunkNo,MeshChunk *chk)
{
	FILE *fp=openMeshFile(chunkNo,nchunks,false);
	chk->write(fp); /*closes file*/
	delete chk;
}

class MeshChunkOutputSender : public MeshChunkOutput {
	CProxy_FEMchunk dest;
public:
	MeshChunkOutputSender(const CProxy_FEMchunk &dest_)
		:dest(dest_) {}
	void accept(int chunkNo,MeshChunk *chk)
	{
		dest[chunkNo].run(chk);
		delete chk;
	}
};

static void RestoreTCharmGlobals(int _nchunks) {
  PUP::fromTextFile pg(openMeshFile(_nchunks,_nchunks,true));
  TCharmReadonlys::pupAllReadonlys(pg);
}
static void SaveTCharmGlobals(int _nchunks) {
  PUP::toTextFile pg(openMeshFile(_nchunks,_nchunks,false));
  TCharmReadonlys::pupAllReadonlys(pg);
}

FDECL void FTN_NAME(FEM_ATTACH,fem_attach)(int *flags) 
{
	FEM_Attach(*flags);
}

CDECL void FEM_Attach(int flags)
{
	FEMAPI("FEM_Attach");
	//Make sure the threads array exists
	TCharmSetupCookie *tc=TCharmSetupCookie::get();
	if (!tc->hasThreads())
		CkAbort("You must create a thread array with TCharmCreate before calling FEM_Attach!\n");
	int _nchunks=tc->getNumElements();
	
	if (flags&FEM_INIT_WRITE) 
	{ //First save the user's globals (if any):
		SaveTCharmGlobals(_nchunks);
		
		//Split the mesh and write it out
		MeshChunkOutputWriter w(_nchunks);
		mesh_split(_nchunks,&w);
		CkExit();
	}
	if (flags&FEM_INIT_READ)
	{ //Restore the user's globals on PE 0-- they'll get copied elsewhere
	  RestoreTCharmGlobals(_nchunks);
	}
	
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
	if (_meshptr!=NULL) 
	{ //Partition the serial mesh online
		MeshChunkOutputSender s(chunks);
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
	MeshChunk **cmsgs; //Messages from/for array elements
	int updateCount; //Number of mesh updates so far
	CkQ<MeshChunk *> futureUpdates;
	CkQ<MeshChunk *> curUpdates; 
	int numdone; //Length of curUpdates 
public:
	FEMcoordinator(int nChunks_) 
		:nChunks(nChunks_)
	{
		cmsgs=new MeshChunk*[nChunks]; CHK(cmsgs);		
		numdone=0;
		updateCount=1;
	}
	~FEMcoordinator() {
		delete[] cmsgs;
	}
	
	void setArray(const CkArrayID &fem_) {femChunks=fem_;}
	void updateMesh(marshallMeshChunk &chk);
};

class MeshChunkOutputUpdate : public MeshChunkOutput {
	CProxy_FEMchunk dest;
public:
	MeshChunkOutputUpdate(const CProxy_FEMchunk &dest_)
		:dest(dest_) {}
	void accept(int chunkNo,MeshChunk *chk)
	{
		dest[chunkNo].meshUpdated(chk);
		delete chk;
	}
};

//Called by a chunk on FEM_Update_Mesh
void FEMcoordinator::updateMesh(marshallMeshChunk &chk)
{
  MeshChunk *msg=chk;
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
      	MeshChunk *m=curUpdates.deq();
	cmsgs[m->fromChunk]=m;
      }
      //Save what to do with the mesh
      int callMeshUpdated=cmsgs[0]->callMeshUpdated;
      int doRepartition=cmsgs[0]->doRepartition;
      //Assemble the current chunks into a serial mesh
      delete _meshptr;
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
	MeshChunkOutputUpdate u(femChunks);
	mesh_split(_nchunks,&u);
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
class MeshChunkOutputStorer : public MeshChunkOutput {
  int nchunks;
  MeshChunk **chks;
public:
  MeshChunkOutputStorer(int nc) :nchunks(nc) {
    typedef MeshChunk* MeshChunkPtr;
    chks=new MeshChunkPtr[nc];
  }
  ~MeshChunkOutputStorer() {
    for (int i=0;i<nchunks;i++) delete chks[i];
    delete[] chks;
  }
  void accept(int chunkNo,MeshChunk *chk) {
    chks[chunkNo]=chk;
  }
  void useChunk(int i) {
    if (i<0 || i>=nchunks) CkAbort("Invalid index passed to FEM_Serial_Begin!");
    FILE *fp=openMeshFile(i,nchunks,false);
    chks[i]->write(fp); /*closes file*/
    _meshptr=&chks[i]->m;
  }
};

static MeshChunkOutputStorer *meshChunkStore=NULL;
CDECL void FEM_Serial_Split(int npieces) {
  FEMAPI("FEM_Serial_Split");
  meshChunkStore=new MeshChunkOutputStorer(npieces);
  mesh_split(npieces,meshChunkStore);
  SaveTCharmGlobals(npieces);
}
FDECL void FTN_NAME(FEM_SERIAL_SPLIT,fem_serial_split)(int *npieces)
{ 
  FEM_Serial_Split(*npieces); 
}

CDECL void FEM_Serial_Begin(int chunkNo) {
  FEMAPI("FEM_Serial_Begin");
  if (!meshChunkStore) CkAbort("Can't call FEM_Serial_Begin before FEM_Serial_Split!");
  meshChunkStore->useChunk(chunkNo);
}
FDECL void FTN_NAME(FEM_SERIAL_BEGIN,fem_serial_begin)(int *pieceNo)
{
  FEM_Serial_Begin(*pieceNo-1);
}


/********************** Mesh Creation ************************/
/*Utility*/

//Make a heap-allocated copy of this (len-item) array, changing the index as spec'd
static int *copyArray(const int *src,int len,int indexBase)
{
	int *ret=new int[len];
	for (int i=0;i<len;i++) ret[i]=src[i]-indexBase;
	return ret;
}

void FEM_Item::allocate(int nItems,int dataPer)
{
	udata.allocate(dataPer,nItems);
	if (sym) sym->setSize(nItems);
}

void FEM_Elem::allocate(int nItems,int dataPer,int nodesPer)
{
	FEM_Item::allocate(nItems,dataPer);
	conn.allocate(nodesPer,nItems);
}

/***** Mesh getting and setting state ****/

static FEM_Mesh *setMesh(void) {
  if(TCharm::getState()==inDriver) {
    FEMchunk *cptr = CtvAccess(_femptr);
    if (cptr->updated_mesh==NULL)
      cptr->updated_mesh=new MeshChunk;
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
    return &cptr->getMesh();
  } else {
    //Called from init, finalize, or meshUpdate
    if (_meshptr==NULL) {
      CkAbort("FEM: Cannot get mesh-- it was never set!\n");
    }
    return _meshptr;
  }
}

/********** Symmetries **********/
static FEM_Ghost &setGhost(void) {
	if(TCharm::getState()==inDriver)
		CkAbort("FEM: Cannot call ghost or symmetry routines from driver!");
	return ghosts;
}


void FEM_Ghost::setSymmetries(int nNodes_,int *can,const int *sym_src)
{
	nodeCanon=can;
	allocSym(nNodes_);
	for (int i=0;i<nNodes;i++) 
		nodeSymmetries[i]=(FEM_Symmetries_t)sym_src[i];
}

void FEM_Item::setSymmetries(int r,FEM_Symmetries_t s)
{
	if (!sym) {
		if (s==0) return; //Don't bother allocating for nothing
		sym=new sym_t;
	}
	sym->insert(r,s);
}

CDECL void FEM_Set_Sym_Nodes(const int *canon,const int *sym)
{
	FEMAPI("FEM_Set_Sym_Nodes");
	int n=setMesh()->node.size();
	setGhost().setSymmetries(n,copyArray(canon,n,0),sym);
}
FDECL void FTN_NAME(FEM_SET_SYM_NODES,fem_set_sym_nodes)
	(const int *canon,const int *sym)
{
	FEMAPI("FEM_Set_Sym_Nodes");
	int n=setMesh()->node.size();
	setGhost().setSymmetries(n,copyArray(canon,n,1),sym);
}

CDECL void FEM_Get_Sym(int elTypeOrMinusOne,int *destSym)
{
	FEMAPI("FEM_Get_Sym");
	const FEM_Item &l=getMesh()->getCount(elTypeOrMinusOne);
	int n=l.size();
	for (int i=0;i<n;i++) destSym[i]=l.getSymmetries(i);
}
FDECL void FTN_NAME(FEM_GET_SYM,fem_get_sym)
	(int *elTypeOrZero,int *destSym)
{
	FEM_Get_Sym(*elTypeOrZero-1,destSym);
}

#if 0
/*
 Build symmetries based on faces.
 This is only useful once you have face-face correspondences,
 which we don't immediately have.  Thus this is commented out for now...
*/
static FEM_Symmetries_t makeSym(int symNo) 
{ //Convert a user-input symmetry number (0,1,2,3, ...) to a 
  //  FEM_Symmetries_t (1,2,4,8,...)
	if (symNo<0) CkAbort("FEM_Add_Sym_Faces> symmetry number is negative");
	if (symNo>=(8*sizeof(FEM_Symmetries_t)))
		CkAbort("FEM_Add_Sym_Faces> symmetry number is too large");
	return FEM_Symmetries_t(1<<symNo);
}

static void combineCanon(int &a,int &b) 
{ //Combine entries in canonicalization array by taking minimum	
	if (a<b) {b=a;}
	else /*a>=b*/ {a=b;}
}

void FEM_Ghost::addFaces(int nNodes_,int nFaces,int nodePerFace,
		int symA,const int *fA,int symB,const int *fB,int idxBase)
{
	initSym(nNodes_);
	FEM_Symmetries_t mA=makeSym(symA);
	FEM_Symmetries_t mB=makeSym(symB);
	
	int nF=nFaces*nodePerFace;
	for (int n=0;n<nF;n++) {
		int a=nodeCheck(fA[n]-idxBase), b=nodeCheck(fB[n]-idxBase);
		//Check for invalid nodes:
		if (a==-1 || b==-1) 
		{ //User marked this as an invalid node: skip it
			break;
		}
		if (a==-2 || b==-2) { //nodeCheck marked as invalid: abort!
			CkAbort("FEM_Add_Sym_Faces> Bad node index passed in faces array!");
		}
		
		//a is same as b-- paste them together
		combineCanon(nodeCanon[a],nodeCanon[b]);
		//a belongs to symmetry mA, likewise b:
		nodeSymmetries[a]=mA;
		nodeSymmetries[b]=mB;
	}
}

CDECL void FEM_Add_Sym_Faces(int nFaces,int nodePerFace,
	int symA,const int *fA,int symB,const int *fB)
{
	FEMAPI("FEM_Add_Sym_Faces");
	setGhost().addFaces(setMesh()->node.size(),nFaces,nodePerFace,
		symA,fA,symB,fB,0);
}
FDECL void FTN_NAME(FEM_ADD_SYM_FACES,fem_add_sym_faces)
	(int *nFaces,int *nodePerFace,
	int *symA,const int *fA,int *symB,const int *fB)
{
	FEMAPI("FEM_Add_Sym_Faces");
	setGhost().addFaces(setMesh()->node.size(),*nFaces,*nodePerFace,
		*symA-1,fA,*symB-1,fB,1);
}
#endif

/********** Sparse data ********/
void FEM_Sparse::allocate(int n_) //Allocate storage for data and nodes of n tuples
{
	nodes.allocate(n_);
	data.allocate(n_);
}
const FEM_Sparse &FEM_Mesh::getSparse(int uniqueID) const
{
	if (uniqueID<0 || uniqueID>=sparse.size())
		CkAbort("FEM: Invalid sparse identifier!");
	return *sparse[uniqueID];
}
FEM_Sparse &FEM_Mesh::setSparse(int uniqueID)
{
	if (uniqueID<0 || uniqueID>=sparse.size())
		CkAbort("FEM: Invalid sparse identifier!");
	return *sparse[uniqueID];
}
void FEM_Mesh::setSparse(int uniqueID,FEM_Sparse *s)
{
	if (uniqueID<sparse.size()) CkAbort("FEM: Cannot re-use sparse identifiers!");
	if (uniqueID>sparse.size()) CkAbort("FEM: Must use sparse identifiers in sequence!");
	/*now uniqeID==sparse.size()*/
	sparse.push_back(s);
}
void FEM_Sparse::set(int rows,const int *n,int idxBase,const char *d)
{
	allocate(rows);
	nodes.set(n,idxBase);
	data.set(d);
}
void FEM_Sparse::get(int *n,int idxBase,char *d) const
{
	nodes.get(n,idxBase);
	data.get(d);
}

CDECL void FEM_Set_Sparse(int uniqueID,int nTuples,
  	const int *nodes,int nodesPerTuple,
  	const void *data,int dataPerTuple,int dataType)
{
	FEMAPI("FEM_Set_Sparse");
	int bytesPerTuple=dataPerTuple*DType::type_size(dataType);
	FEM_Sparse *s=new FEM_Sparse(nodesPerTuple,bytesPerTuple);
	s->set(nTuples,nodes,0,(char *)data);
	setMesh()->setSparse(uniqueID,s);
}
FDECL void FTN_NAME(FEM_SET_SPARSE,fem_set_sparse)
	(int *uniqueID,int *nTuples,
  	const int *nodes,int *nodesPerTuple,
  	const void *data,int *dataPerTuple,int *dataType)
{
	FEMAPI("FEM_Set_Sparse");
	int bytesPerTuple=(*dataPerTuple)*DType::type_size(*dataType);
	FEM_Sparse *s=new FEM_Sparse(*nodesPerTuple,bytesPerTuple);
	s->set(*nTuples,nodes,1,(char *)data);
	setMesh()->setSparse(*uniqueID-1,s);
}
CDECL void FEM_Set_Sparse_Elem(int uniqueID,const int *rec2elem)
{
	FEMAPI("FEM_Set_Sparse_Elem");
	FEM_Sparse &s=setMesh()->setSparse(uniqueID);
	s.setElem(copyArray(rec2elem,2*s.size(),0));
}
FDECL void FTN_NAME(FEM_SET_SPARSE_ELEM,fem_set_sparse_elem)
	(const int *uniqueID,int *rec2elem)
{
	FEMAPI("FEM_Set_Sparse_Elem");
	FEM_Sparse &s=setMesh()->setSparse(*uniqueID-1);
	s.setElem(copyArray(rec2elem,2*s.size(),1));
}

CDECL int  FEM_Get_Sparse_Length(int uniqueID)
{
	FEMAPI("FEM_Get_Sparse_Length");
	return getMesh()->getSparse(uniqueID).size();
}
CDECL void FEM_Get_Sparse(int uniqueID,int *tuples,void *data)
{
	FEMAPI("FEM_Get_Sparse");
	getMesh()->getSparse(uniqueID).get(tuples,0,(char *)data);
}
FDECL int  FTN_NAME(FEM_GET_SPARSE_LENGTH,fem_get_sparse_length)(int *uniqueID)
{
	return FEM_Get_Sparse_Length(*uniqueID-1);
}
FDECL void FTN_NAME(FEM_GET_SPARSE,fem_get_sparse)(int *uniqueID,int *tuples,void *data)
{
	FEMAPI("FEM_Get_Sparse");
	getMesh()->getSparse(*uniqueID-1).get(tuples,1,(char *)data);
}

/****** Ghost Renumbering *******/
#if 1
static FEM_Mesh &getRenumber(void) {
	if(TCharm::getState()!=inDriver)
		CkAbort("FEM: Can only call renumbering routines from driver!");
	FEMchunk *cptr = CtvAccess(_femptr);
	return cptr->cur_mesh->m;
}

//Clear any previous binding for element type elType:
CDECL void FEM_Composite_Elem(int elType) {
	FEMAPI("FEM_Composite_Elem");
	FEM_Mesh &m=getRenumber();
	if (elType<m.elem.size())
		m.elem.reinit(elType);
	else
		m.elem.makeLonger(elType);
}
FDECL void FTN_NAME(FEM_COMPOSITE_ELEM,fem_composite_elem)(int *elType) {
	FEM_Composite_Elem(*elType-1);
}


//Add all of src's items into dest, shifting by idxShift
static void combineList(const FEM_Comm &src,FEM_Comm &dest,
	int idxShift)
{
	int sNo,sMax=src.size();
	for (sNo=0;sNo<sMax;sNo++) {
		const FEM_Comm_List &s=src.getLocalList(sNo);
		FEM_Comm_List &d=dest.addList(s.getDest());
		for (int i=0;i<s.size();i++) {
			d.push_back(s[i]+idxShift);
		}
	}
	//FIXME: Combine map hashtables as well (eventually)
}

//Combine the ghost records for srcType and destType,
//  shifting srcType's interior elements to start at "startInt"
//  and its ghost elements to start at "startGhost"
CDECL void FEM_Combine_Elem(
	int srcType,int destType,
	int startInt,int startGhost) 
{
	FEMAPI("FEM_Combine_Elem");
	FEM_Mesh &m=getRenumber();
	const FEM_Item &src=m.getCount(srcType);
	FEM_Item &dest=m.setCount(destType);
	//Interior nodes are all sent off, shifted to startInt:
	combineList(src.ghostSend,dest.ghostSend,startInt);
	//Ghost nodes are all received:
	//   First shift ghost nodes to zero, then to startGhost:
	combineList(src.ghostRecv,dest.ghostRecv,
		startGhost-src.getGhostStart());
}
FDECL void FTN_NAME(FEM_COMBINE_ELEM,fem_combine_elem)(
	int *srcType,int *destType,int *startInt,int *startGhost) 
{
	FEM_Combine_Elem(*srcType-1,*destType-1,*startInt-1,*startGhost-1);
}

#endif

/****** Custom Partitioning API *******/
static void Set_Partition(int *elem2chunk,int indexBase) {
	if (_elem2chunk!=NULL) delete[] _elem2chunk;
	_elem2chunk=copyArray(elem2chunk,getMesh()->nElems(),indexBase);
}

//C bindings:
CDECL void FEM_Set_Partition(int *elem2chunk) {
	FEMAPI("FEM_Set_Partition");
	Set_Partition(elem2chunk,0);
}

//Fortran bindings:
FDECL void FTN_NAME(FEM_SET_PARTITION,fem_set_partition)
	(int *elem2chunk) 
{
	FEMAPI("FEM_Set_Partition");
	Set_Partition(elem2chunk,1);
}

/***** Mesh-Setting API: C bindings*/
CDECL void FEM_Set_Node(int nNodes,int dataPer) 
{
	FEMAPI("FEM_Set_Node");
	setMesh()->node.allocate(nNodes,dataPer);
}
CDECL void FEM_Set_Node_Data(const double *data) 
{
	FEMAPI("FEM_Set_Node_Data");
	setMesh()->node.setUdata().set(data);
}

CDECL void FEM_Set_Elem(int elType,int nElem,int dataPer,int nodePer) {
	FEMAPI("FEM_Set_Elem");
	FEM_Mesh *m=setMesh();
	m->elem.makeLonger(elType);
	m->elem[elType].allocate(nElem,dataPer,nodePer);
}
CDECL void FEM_Set_Elem_Data(int elType,const double *data) 
{
	FEMAPI("FEM_Set_Elem_Data");
	setMesh()->setElem(elType).setUdata().set(data);
}
CDECL void FEM_Set_Elem_Conn(int elType,const int *conn) {
	FEMAPI("FEM_Set_Elem_Conn");
	setMesh()->setElem(elType).setConn().set(conn);
}

/*Convenience routine: for use when you only have one kind of element
and no userdata.*/
CDECL void FEM_Set_Mesh(int nelem, int nnodes, int ctype, int *conn)
{
	FEMAPI("FEM_Set_Mesh");
	FEM_Set_Node(nnodes,0);
	FEM_Set_Elem(0,nelem,0,ctype);
	FEM_Set_Elem_Conn(0,conn);
}

FDECL void FTN_NAME(FEM_SET_NODE,fem_set_node)
	(int *nNodes,int *dataPer) 
{
	FEMAPI("FEM_Set_Node");
	FEM_Set_Node(*nNodes,*dataPer);
}

FDECL void FTN_NAME(FEM_SET_NODE_DATA_R,fem_set_node_data_r)
	(double *data) 
{
	FEMAPI("FEM_Set_Node_Data_r");
	setMesh()->node.setUdata().set(data);
}
FDECL void FTN_NAME(FEM_SET_NODE_DATA_C,fem_set_node_data_c)
	(double *data) 
{
	FEMAPI("FEM_Set_Node_Data_c");
	setMesh()->node.setUdata().setTranspose(data);
}

FDECL void FTN_NAME(FEM_SET_ELEM,fem_set_elem)
	(int *elType,int *nElem,int *dataPer,int *nodePer)  
{
	FEMAPI("FEM_set_elem");
	FEM_Set_Elem(*elType-1,*nElem,*dataPer,*nodePer);
}
FDECL void FTN_NAME(FEM_SET_ELEM_DATA_R,fem_set_elem_data_r)
	(int *elType,double *data)
{
	FEMAPI("FEM_set_elem_data_r");
	setMesh()->setCount(*elType-1).setUdata().set(data);
}
FDECL void FTN_NAME(FEM_SET_ELEM_DATA_C,fem_set_elem_data_c)
	(int *elType,double *data)
{
	FEMAPI("FEM_set_elem_data_c");
	setMesh()->setCount(*elType-1).setUdata().setTranspose(data);
}

FDECL void FTN_NAME(FEM_SET_ELEM_CONN_R,fem_set_elem_conn_r)
	(int *elType,int *conn_r)
{
	FEMAPI("FEM_set_elem_conn_r");
	setMesh()->setElem(*elType-1).setConn().set(conn_r,1);
}
FDECL void FTN_NAME(FEM_SET_ELEM_CONN_C,fem_set_elem_conn_c)
	(int *elType,int *conn_c)
{
	FEMAPI("FEM_set_elem_conn_c");
	setMesh()->setElem(*elType-1).setConn().setTranspose(conn_c,1);
}

/*Convenience routine: for use when you only have one kind of element*/
FDECL void FTN_NAME(FEM_SET_MESH,fem_set_mesh)
	(int *nelem, int *nnodes, int *ctype, int *conn)
{
	FEMAPI("FEM_Set_Mesh");
	int elType=1,zero=0;
	FTN_NAME(FEM_SET_NODE,fem_set_node) (nnodes,&zero);
	FTN_NAME(FEM_SET_ELEM,fem_set_elem) (&elType,nelem,&zero,ctype);
	FTN_NAME(FEM_SET_ELEM_CONN_C,fem_set_elem_conn_c) (&elType,conn);
}

/***** Mesh-Getting API: C bindings*/

CDECL void FEM_Get_Node(int *nNodes,int *dataPer) 
{
	FEMAPI("FEM_Get_Node");
	const FEM_Mesh *m=getMesh();
	if (nNodes!=NULL) *nNodes=m->node.size();
	if (dataPer!=NULL) *dataPer=m->node.getDataPer();
}
CDECL void FEM_Get_Node_Data(double *data) 
{
	FEMAPI("FEM_Get_Node_Data");
	getMesh()->node.getUdata().get(data);
}

CDECL void FEM_Get_Elem(int elType,int *nElem,int *dataPer,int *nodePer) 
{
	FEMAPI("FEM_Get_Elem");
	const FEM_Mesh *m=getMesh();
	m->chkET(elType);
	if (nElem!=NULL) *nElem=m->elem[elType].size();
	if (dataPer!=NULL) *dataPer=m->elem[elType].getDataPer();
	if (nodePer!=NULL) *nodePer=m->elem[elType].getNodesPer();
}
CDECL void FEM_Get_Elem_Data(int elType,double *data) 
{
	FEMAPI("FEM_Get_Elem_Data");
	getMesh()->getCount(elType).getUdata().get(data);
}
CDECL void FEM_Get_Elem_Conn(int elType,int *conn) {
	FEMAPI("FEM_Get_Elem_Conn");
	getMesh()->getElem(elType).getConn().get(conn);
}

FDECL void FTN_NAME(FEM_GET_NODE,fem_get_node)
	(int *nNodes,int *dataPer) 
{
	FEMAPI("FEM_Get_node");
	FEM_Get_Node(nNodes,dataPer);
}
FDECL void FTN_NAME(FEM_GET_NODE_DATA_R,fem_get_node_data_r)
	(double *data) 
{
	FEMAPI("FEM_Get_node_data_r");
	getMesh()->node.getUdata().get(data);
}
FDECL void FTN_NAME(FEM_GET_NODE_DATA_C,fem_get_node_data_c)
	(double *data) 
{
	FEMAPI("FEM_Get_node_data_c");
	getMesh()->node.getUdata().getTranspose(data);
}

FDECL void FTN_NAME(FEM_GET_ELEM,fem_get_elem)
	(int *elType,int *nElem,int *dataPer,int *nodePer)  
{
	FEMAPI("FEM_Get_elem");
	FEM_Get_Elem(*elType-1,nElem,dataPer,nodePer);
}
FDECL void FTN_NAME(FEM_GET_ELEM_DATA_R,fem_get_elem_data_r)
	(int *elType,double *data) 
{
	FEMAPI("FEM_Get_elem_data_r");
	getMesh()->getCount(*elType-1).getUdata().get(data);
}
FDECL void FTN_NAME(FEM_GET_ELEM_DATA_C,fem_get_elem_data_c)
	(int *elType,double *data) 
{
	FEMAPI("FEM_Get_elem_data_c");
	getMesh()->getCount(*elType-1).getUdata().getTranspose(data);
}

FDECL void FTN_NAME(FEM_GET_ELEM_CONN_R,fem_get_elem_conn_r)
	(int *elType,int *conn)
{
	FEMAPI("FEM_Get_elem_conn_r");
	getMesh()->getElem(*elType-1).getConn().get(conn,1);
}
FDECL void FTN_NAME(FEM_GET_ELEM_CONN_C,fem_get_elem_conn_c)
	(int *elType,int *conn)
{
	FEMAPI("FEM_Get_elem_conn_c");
	getMesh()->getElem(*elType-1).getConn().getTranspose(conn,1);
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


/************************************************
"Gather" routines extract data distributed (nodeIdx)
through the user's array (in) and collect it into a message (out).
 */
#define gather_args (int nVal,int valLen, \
		    const int *nodeIdx,int nodeScale, \
		    const char *in,char *out)

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

//Gather (into out) the given data type from in for each node
static void gather(const DType &dt,
		   int nNodes,const int *nodes,
		   const void *v_in,void *v_out)
{
  const char *in=(const char *)v_in;
  char *out=(char *)v_out;
  in += dt.init_offset;
  //Try for a more specialized version if possible:
  if (dt.base_type == FEM_DOUBLE) {
      switch(dt.vec_len) {
      case 1: gather_double1(nNodes,dt.length(),nodes,dt.distance,in,out); return;
      case 2: gather_double2(nNodes,dt.length(),nodes,dt.distance,in,out); return;
      case 3: gather_double3(nNodes,dt.length(),nodes,dt.distance,in,out); return;
      }
  }
  //Otherwise, use the general version
  gather_general(nNodes,dt.length(),nodes,dt.distance,in,out);
}

/************************************************
"Scatter" routines are the opposite of gather.
 */
#define scatter_args (int nVal,int valLen, \
		    const int *nodeIdx,int nodeScale, \
		    const char *in,char *out)

static void scatter_general scatter_args
{
  for(int i=0;i<nVal;i++) {
      void *dest = (void *)(out+nodeIdx[i]*nodeScale);
      memcpy(dest,in, valLen);
      in +=valLen;
  }
}

#define scatter_doubles(n,copy) \
static void scatter_double##n scatter_args \
{ \
  const double *src=(const double *)in; \
  for(int i=0;i<nVal;i++) { \
      double *od = (double *)(out+nodeIdx[i]*nodeScale); \
      copy \
      src+=n; \
  } \
}

scatter_doubles(1,od[0]=src[0];)
scatter_doubles(2,od[0]=src[0];od[1]=src[1];)
scatter_doubles(3,od[0]=src[0];od[1]=src[1];od[2]=src[2];)

//Scatter (into out) the given data type from in for each node
static void scatter(const DType &dt,
		   int nNodes,const int *nodes,
		   const void *v_in,void *v_out)
{
  const char *in=(const char *)v_in;
  char *out=(char *)v_out;
  out += dt.init_offset;
  //Try for a more specialized version if possible:
  if (dt.base_type == FEM_DOUBLE) {
      switch(dt.vec_len) {
      case 1: scatter_double1(nNodes,dt.length(),nodes,dt.distance,in,out); return;
      case 2: scatter_double2(nNodes,dt.length(),nodes,dt.distance,in,out); return;
      case 3: scatter_double3(nNodes,dt.length(),nodes,dt.distance,in,out); return;
      }
  }
  //Otherwise, use the general version
  scatter_general(nNodes,dt.length(),nodes,dt.distance,in,out);
}


/***********************************************
"ScatterAdd" routines add the message data (in) to the
shared nodes distributed through the user's data (out).
 */
#define scatteradd_args (int nVal, \
		    const int *nodeIdx,int nodeScale, \
		    const char *in,char *out)

#define scatteradd_doubles(n,copy) \
static void scatteradd_double##n scatteradd_args \
{ \
  const double *id=(const double *)in; \
  for(int i=0;i<nVal;i++) { \
      double *targ = (double *)(out+nodeIdx[i]*nodeScale); \
      copy \
      id+=n; \
  } \
}

scatteradd_doubles(1,targ[0]+=id[0];)
scatteradd_doubles(2,targ[0]+=id[0];targ[1]+=id[1];)
scatteradd_doubles(3,targ[0]+=id[0];targ[1]+=id[1];targ[2]+=id[2];)

//ScatterAdd (into out) the given data type, from in, for each node
static void scatteradd(const DType &dt,
		   int nNodes,const int *nodes,
		   const void *v_in,void *v_out)
{
  const char *in=(const char *)v_in;
  char *out=(char *)v_out;
  out += dt.init_offset;
  //Try for a more specialized version if possible:
  if (dt.base_type == FEM_DOUBLE) {
      switch(dt.vec_len) {
      case 1: scatteradd_double1(nNodes,nodes,dt.distance,in,out); return;
      case 2: scatteradd_double2(nNodes,nodes,dt.distance,in,out); return;
      case 3: scatteradd_double3(nNodes,nodes,dt.distance,in,out); return;
      }
  }

  /*Otherwise we need the slow, general version*/
  combineFn fn=combine(dt,FEM_SUM);
  int length=dt.length();
  for(int i=0;i<nNodes;i++) {
    void *cnode = (void*) (out+nodes[i]*dt.distance);
    fn(dt.vec_len, cnode, in);
    in += length;
  }
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

  updated_mesh=NULL;
  cur_mesh=NULL;

  updateCount=1; //Number of mesh updates

//Field updates:
  messages = CmmNew();
  updateSeqnum = 1;
  updateComm=NULL;
  nRecd=0;
  updateBuf=NULL;
  reductionBuf=NULL;
  listCount=0;listSuspended=false;
}
FEMchunk::FEMchunk(CkMigrateMessage *msg)
	:ArrayElement1D(msg), thisproxy(thisArrayID)
{
  updated_mesh=NULL;
  cur_mesh=NULL;
  messages=NULL;
  updateBuf=NULL;
  reductionBuf=NULL;	
  listCount=0;listSuspended=false;
}

FEMchunk::~FEMchunk()
{
	CmmFree(messages);
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
  ArrayElement1D::ckJustMigrated(); //Call superclass
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


//Update my shared nodes based on these values
void FEMchunk::update_node(FEM_DataMsg *msg)
{
  const DType &dt=dtypes[msg->dtype];
  const FEM_Comm_List &l = updateComm->getList(msg->from);
  if (l.size()*dt.length()!=msg->length)
	  CkAbort("Wrong-length message recv'd by FEM update node!  Have communication lists been corrupted?\n");
  scatteradd(dt,l.size(),l.getVec(),msg->data,updateBuf);
}

void FEMchunk::update_ghost(FEM_DataMsg *msg)
{
  const DType &dt=dtypes[msg->dtype];
  const FEM_Comm_List &l = updateComm->getList(msg->from);
  if (l.size()*dt.length()!=msg->length)
	  CkAbort("Wrong-length message recv'd by FEM update ghost!  Have communication lists been corrupted?\n");
  scatter(dt,l.size(),l.getVec(),msg->data,updateBuf);
}

//Received a message for the current update
void
FEMchunk::recvUpdate(FEM_DataMsg *msg)
{
  if (updateType==NODE_UPDATE) update_node(msg);
  else if (updateType==GHOST_UPDATE) update_ghost(msg);
  else CkAbort("FEM> Bad update type!\n");
  delete msg;
  nRecd++;
}

void
FEMchunk::recv(FEM_DataMsg *dm)
{
  if (dm->tag == updateSeqnum) {
    recvUpdate(dm); // update the appropriate field value
    if(nRecd==updateComm->size()) {
      thread->resume();
    }
    else if (nRecd>updateComm->size())
      CkAbort("FEM> Recv'd too many messages for an update!\n");   
  } else if (dm->tag > updateSeqnum) {
    //Message for a future update
    CmmPut(messages, 1, &(dm->tag), dm);
  } else 
    CkAbort("FEM> Recv'd message from an update that was finished!\n");
}

void 
FEMchunk::beginUpdate(void *buf,int fid,
     const FEM_Comm *sendComm,const FEM_Comm *recvComm,updateType_t t)
{
  updateBuf=buf;
  updateComm=recvComm;
  updateType=t;
  updateSeqnum++;
  nRecd=0;

  //Send off our values to those processors that need them:
  const DType &dt=dtypes[fid];
  for(int p=0;p<sendComm->size();p++) {
    const FEM_Comm_List &l=sendComm->getLocalList(p);
    int msgLen=l.size()*dt.length();
    FEM_DataMsg *msg = new (&msgLen, 0) FEM_DataMsg(updateSeqnum, 
	      thisIndex, fid,msgLen); CHK(msg);
    gather(dt,l.size(),l.getVec(),buf,msg->data);
    thisproxy[l.getDest()].recv(msg);
  }  
}
void 
FEMchunk::waitForUpdate(void)
{
  // if any of the field values have been received already,
  // process them
  FEM_DataMsg *dm;
  while ((dm = (FEM_DataMsg*)CmmGet(messages, 1, &updateSeqnum, 0))!=NULL) {
    recvUpdate(dm);
  }
  // if any field values are still required, sleep till they arrive
  if (nRecd < updateComm->size()) thread->suspend();
  updateBuf=NULL;
}

void
FEMchunk::update(int fid, void *nodes)
{
  beginUpdate(nodes,fid,&cur_mesh->comm,&cur_mesh->comm,NODE_UPDATE);
  waitForUpdate();
}

void
FEMchunk::updateGhost(int fid, int elemType, void *nodes)
{
  const FEM_Item &cnt=cur_mesh->m.getCount(elemType);
  beginUpdate(nodes,fid,&cnt.ghostSend,&cnt.ghostRecv,GHOST_UPDATE);
  waitForUpdate();
}



void
FEMchunk::reduce_field(int fid, const void *nodes, void *outbuf, int op)
{
  // first reduce over local nodes
  const DType &dt = dtypes[fid];
  const void *src = (const void *) ((const char *) nodes + dt.init_offset);
  initialize(dt,outbuf,op);
  combineFn fn=combine(dt,op);
  for(int i=0; i<cur_mesh->m.node.size(); i++) {
    if(getPrimary(i)) {
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
  reductionBuf = outbuf;
  thread->suspend();
}

void
FEMchunk::reductionResult(FEM_DataMsg *msg)
{
  //msg->from used as length
  memcpy(reductionBuf, msg->data, msg->length);
  reductionBuf=NULL;
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
  int newNode=updated_mesh->m.node.size();
  int oldNode=cur_mesh->m.node.size();
  updated_mesh->elemNums=new int[newElemTot];
  updated_mesh->nodeNums=new int[newNode];
  updated_mesh->isPrimary=new int[newNode];

  //Copy over the old global node numbers, and fabricate the rest  
  int comNode=oldNode; if (comNode>newNode) comNode=newNode;
  memcpy(updated_mesh->nodeNums,cur_mesh->nodeNums,comNode*sizeof(int));
  memcpy(updated_mesh->isPrimary,cur_mesh->isPrimary,comNode*sizeof(int));
  for (i=comNode;i<newNode;i++) {
    updated_mesh->nodeNums[i]=-1;//New nodes have no global number
    updated_mesh->isPrimary[i]=1;//New nodes are not shared
  }

  //Copy over the old global element numbers, and fabricate the rest
  i=0;
  for (t=0; t<cur_mesh->m.elem.size() && t<updated_mesh->m.elem.size() ;t++) {
    int oldElemStart=cur_mesh->m.nElems(t);
    int newElemStart=updated_mesh->m.nElems(t);
    int oldElems=cur_mesh->m.elem[t].size();
    int newElems=updated_mesh->m.elem[t].size();
    int comElems=oldElems;
    if (comElems>newElems) comElems=newElems;
    memcpy(&updated_mesh->elemNums[newElemStart],
	   &cur_mesh->elemNums[oldElemStart],comElems*sizeof(int));
    for (i=newElemStart+comElems;i<newElemStart+newElems;i++)
      updated_mesh->elemNums[i]=-1;//New elements have no global number
  }
  for (;i<newElemTot;i++)
    updated_mesh->elemNums[i]=-1;//New element types have no global number

  //Send the mesh off to the coordinator
  CProxy_FEMcoordinator coord(init.coordinator);
  coord.updateMesh(updated_mesh);
  delete updated_mesh;
  updated_mesh=NULL;
  if (doRepartition)
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
  for(i=0;i<cur_mesh->m.node.size();i++) {
    // skip lines to the next local node
    for(j=curline;j<cur_mesh->nodeNums[i];j++)
      fgets(str,80,fp);
    curline = cur_mesh->nodeNums[i]+1;
    fgets(str,80,fp);
    int curnode, numchars;
    sscanf(str,"%d%n",&curnode,&numchars);
    pos = str + numchars;
    if(curnode != cur_mesh->nodeNums[i]) {
      CkError("Expecting info for node %d, got %d\n", cur_mesh->nodeNums[i], curnode);
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
void MeshChunk::write(FILE *fp)
{
  PUP::toTextFile p(fp);
  pup(p);
}
void MeshChunk::read(FILE *fp)
{
  PUP::fromTextFile p(fp);
  pup(p);  
}

void
FEMchunk::setMesh(MeshChunk *msg)
{
  if (cur_mesh!=NULL) delete cur_mesh;
  if(msg==0) { /*Read mesh from file*/
    cur_mesh=new MeshChunk;
    FILE *fp=openMeshFile(thisIndex,init.numElements,true);
    cur_mesh->read(fp); /*closes file*/
  } else {
    cur_mesh=msg;
  }
}

/******************************* C Bindings **********************************/
static FEMchunk *getCurChunk(void) 
{
  FEMchunk *cptr=CtvAccess(_femptr);
  if (cptr==NULL) 
    CkAbort("Routine can only be called from driver()!\n");
  return cptr;
}
static MeshChunk *getCurMesh(void)
{
  return getCurChunk()->cur_mesh;
}

CDECL void FEM_Update_Mesh(int callMeshUpdated,int doRepartition) 
{ 
  FEMAPI("FEM_Update_Mesh");
  getCurChunk()->updateMesh(callMeshUpdated,doRepartition); 
}

CDECL int FEM_Register(void *_ud,FEM_PupFn _pup_ud)
{
  FEMAPI("FEM_Register");
  return TCharmRegister(_ud,_pup_ud);
}

CDECL void *FEM_Get_Userdata(int n)
{
  FEMAPI("FEM_Get_Userdata");
  return TCharmGetUserdata(n);
}

CDECL void FEM_Barrier(void) {TCharmBarrier();}
FDECL void FTN_NAME(FEM_BARRIER,fem_barrier)(void) {TCharmBarrier();}

CDECL void
FEM_Migrate(void)
{
  TCharmMigrate();
}

CDECL int *
FEM_Get_Node_Nums(void)
{
  FEMAPI("FEM_Get_Node_Nums");
  return getCurChunk()->getNodeNums();
}

CDECL int *
FEM_Get_Elem_Nums(void)
{
  FEMAPI("FEM_Get_Elem_Nums");
  return getCurChunk()->getElemNums();
}

CDECL void 
FEM_Done(void)
{
  TCharmDone();
}

CDECL int 
FEM_Create_Simple_Field(int base_type, int vec_len)
{
  FEMAPI("FEM_Create_Field");
  return getCurChunk()->new_DT(DType(base_type, vec_len));
}

CDECL int 
FEM_Create_Field(int base_type, int vec_len, int init_offset, int distance)
{
  FEMAPI("FEM_Create_Field");
  return getCurChunk()->new_DT(DType(base_type, vec_len, init_offset, distance));
}

CDECL void
FEM_Update_Field(int fid, void *nodes)
{
  FEMAPI("FEM_Update_Field");
  getCurChunk()->update(fid, nodes);
}

CDECL void
FEM_Reduce_Field(int fid, const void *nodes, void *outbuf, int op)
{
  FEMAPI("FEM_Reduce_Field");
  getCurChunk()->reduce_field(fid, nodes, outbuf, op);
}

CDECL void
FEM_Reduce(int fid, const void *inbuf, void *outbuf, int op)
{
  FEMAPI("FEM_Reduce");
  getCurChunk()->reduce(fid, inbuf, outbuf, op);
}

CDECL void
FEM_Read_Field(int fid, void *nodes, const char *fname)
{
  FEMAPI("FEM_Read_Field");
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
  return TCharmWallTimer();
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

CDECL void 
FEM_Print_Partition(void)
{
  FEMAPI("FEM_Print_Partition");
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
	FEMAPI("FEM_Get_Comm_Partners");
	return getCurChunk()->getComm().size();
}
CDECL int FEM_Get_Comm_Partner(int partnerNo)
{
	FEMAPI("FEM_Get_Comm_Partner");
	return getCurChunk()->getComm().getLocalList(partnerNo).getDest();
}
CDECL int FEM_Get_Comm_Count(int partnerNo)
{
	FEMAPI("FEM_Get_Comm_Count");
	return getCurChunk()->getComm().getLocalList(partnerNo).size();
}
CDECL void FEM_Get_Comm_Nodes(int partnerNo,int *nodeNos)
{
	FEMAPI("FEM_Get_Comm_Nodes");
	const int *nNo=getCurChunk()->getComm().getLocalList(partnerNo).getVec();
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

FDECL int FTN_NAME(FEM_CREATE_SIMPLE_FIELD,fem_create_simple_field)
  (int *bt, int *vl)
{
  return FEM_Create_Simple_Field(*bt, *vl);
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

FDECL double FTN_NAME(FEM_TIMER,fem_timer)
  (void)
{
  return FEM_Timer();
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
	const int *nNo=getCurChunk()->getComm().getLocalList(partnerNo).getVec();
	int len=FEM_Get_Comm_Count(partnerNo);
	for (int i=0;i<len;i++)
		nodeNos[i]=nNo[i]+1;
}

FDECL void FTN_NAME(FEM_GET_ELEM_NUMBERS,fem_get_elem_numbers)
	(int *gNo)
{
	FEMAPI("FEM_Get_Elem_Numbers");
	const int *no=getCurChunk()->getElemNums();
	int n=getMesh()->nElems();
	for (int i=0;i<n;i++) gNo[i]=no[i]+1;
}
FDECL void FTN_NAME(FEM_GET_NODE_NUMBERS,fem_get_node_numbers)
	(int *gNo)
{
	FEMAPI("FEM_Get_Node_Numbers");
	const int *no=getCurChunk()->getNodeNums();
	int n=getMesh()->node.size();
	for (int i=0;i<n;i++) gNo[i]=no[i]+1;
}

/******************** Ghost Layers *********************/
CDECL void FEM_Add_Ghost_Layer(int nodesPerTuple,int doAddNodes)
{
	FEMAPI("FEM_Add_Ghost_Layer");
	curGhostLayer=setGhost().addLayer();
	curGhostLayer->nodesPerTuple=nodesPerTuple;
	curGhostLayer->addNodes=(doAddNodes!=0);
	curGhostLayer->elem.makeLonger(getMesh()->elem.size());
}
FDECL void FTN_NAME(FEM_ADD_GHOST_LAYER,fem_add_ghost_layer)
	(int *nodesPerTuple,int *doAddNodes)
{ FEM_Add_Ghost_Layer(*nodesPerTuple,*doAddNodes); }

CDECL void FEM_Add_Ghost_Elem(int elType,int tuplesPerElem,const int *elem2tuple)
{
	FEMAPI("FEM_Add_Ghost_Elem");
	if (curGhostLayer==NULL)
		CkAbort("You must call FEM_Add_Ghost_Layer before calling FEM_Add_Ghost_Elem!\n");
	curGhostLayer->elem[elType].add=true;
	curGhostLayer->elem[elType].tuplesPerElem=tuplesPerElem;
	curGhostLayer->elem[elType].elem2tuple=copyArray(elem2tuple,
		          tuplesPerElem*curGhostLayer->nodesPerTuple,0);
}
FDECL void FTN_NAME(FEM_ADD_GHOST_ELEM,fem_add_ghost_elem)
	(int *FelType,int *FtuplesPerElem,const int *elem2tuple)
{
	FEMAPI("FEM_add_ghost_elem");
	int elType=*FelType-1;
	int tuplesPerElem=*FtuplesPerElem;
	if (curGhostLayer==NULL)
		CkAbort("You must call FEM_Add_Ghost_Layer before calling FEM_Add_Ghost_Elem!\n");
	getMesh()->chkET(elType);
	curGhostLayer->elem[elType].add=true;
	curGhostLayer->elem[elType].tuplesPerElem=tuplesPerElem;
	curGhostLayer->elem[elType].elem2tuple=copyArray(elem2tuple,
		          tuplesPerElem*curGhostLayer->nodesPerTuple,1);
}


CDECL int FEM_Get_Node_Ghost(void) {
	FEMAPI("FEM_Get_Node_Ghost");
	return getMesh()->node.getGhostStart();
}
FDECL int FTN_NAME(FEM_GET_NODE_GHOST,fem_get_node_ghost)(void) {
	return 1+FEM_Get_Node_Ghost();
}

CDECL int FEM_Get_Elem_Ghost(int elType) {
	FEMAPI("FEM_Get_Elem_Ghost");
	return getMesh()->getCount(elType).getGhostStart();
}
CDECL int FTN_NAME(FEM_GET_ELEM_GHOST,fem_get_elem_ghost)(int *elType) {
	return 1+FEM_Get_Elem_Ghost(*elType-1);
}

CDECL void FEM_Update_Ghost_Field(int fid, int elemType, void *data)
{
	FEMAPI("FEM_Update_Ghost_Field");
	getCurChunk()->updateGhost(fid,elemType,data);
}
FDECL void FTN_NAME(FEM_UPDATE_GHOST_FIELD,fem_update_ghost_field)
	(int *fid, int *elemType, void *data)
{
	FEM_Update_Ghost_Field(*fid,*elemType-1,data);
}

/*********** Mesh modification **********
It's the *user's* responsibility to ensure no update_field calls
are outstanding when the mesh is begin modified.  This typically
means keeping track of outstanding communication, and/or wrapping things 
in FEM_Barrier calls.
*/

//Add a new node at the given local index.  Copy the communication
// list from the intersection of the communication lists of the given nodes.
//FIXME: this ignores ghost communication lists, which may change too
static void addNode(int localIdx,int nBetween,int *betweenNodes,int idxbase)
{
	FEMchunk *chk=getCurChunk();
	FEM_Comm &c=chk->cur_mesh->comm;
	//Find the commRecs for the surrounding nodes
	const FEM_Comm_Rec *tween[20];
	int w;
	for (w=0;w<nBetween;w++) {
		tween[w]=c.getRec(betweenNodes[w]-idxbase);
		if (tween[w]==NULL) 
			return; //An unshared node! Thus a private-only addition
	}
	//Make a new commRec as the interesection of the surrounding nodes--
	// we loop over the first node's comm. list
	for (int zs=tween[0]->getShared()-1;zs>=0;zs--) {
		int chk=tween[0]->getChk(zs);
		//Make sure this processor shares all our nodes
		for (w=0;w<nBetween;w++)
			if (!tween[w]->hasChk(chk))
				break;
		if (w==nBetween) //The new node is shared with chk
			c.addNode(localIdx,chk);
	}
}

CDECL void FEM_Add_Node(int localIdx,int nBetween,int *betweenNodes)
{
	FEMAPI("FEM_Add_Node");
	addNode(localIdx,nBetween,betweenNodes,0);
}
FDECL void FTN_NAME(FEM_ADD_NODE,fem_add_node)
	(int *localIdx,int *nBetween,int *betweenNodes)
{
	FEMAPI("FEM_add_node");
	addNode(*localIdx-1,*nBetween,betweenNodes,1);	
}

//Item list exchange:
// FIXME: I assume only one outstanding list exchange.
// This implies the presence of a global barrier before or after the exchange.
void FEMchunk::exchangeGhostLists(int elemType,
	     int inLen,const int *inList,int idxbase)
{
	const FEM_Comm &cnt=cur_mesh->m.getCount(elemType).ghostSend;
	
//Send off a list to each neighbor
	int nChk=cnt.size();
	CkVec<int> *outIdx=new CkVec<int>[nChk];
	//Loop over (the shared entries in) the input list
	for (int i=0;i<inLen;i++) {
		int localNo=inList[i]-idxbase;
		const FEM_Comm_Rec *rec=cnt.getRec(localNo);
		if (NULL==rec) continue; //This item isn't shared
		//This item is shared-- add its comm. idx to each chk
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
	if (!finishListExchange(cur_mesh->m.getCount(elemType).ghostRecv)){
		listSuspended=true;
		thread->suspend(); //<- sleep until all lists arrive
	}
	else //We have everything we need-- reset and continue
		listCount=0;
}

void FEMchunk::recvList(int elemType,int fmChk,int nIdx,const int *idx)
{
	int i;
	const FEM_Comm_List &l=cur_mesh->m.getCount(elemType).ghostRecv.
		getList(fmChk);
	for (i=0;i<nIdx;i++)
		listTmp.push_back(l[idx[i]]);
	listCount++;
	finishListExchange(cur_mesh->m.getCount(elemType).ghostRecv);
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
CDECL void FEM_Exchange_Ghost_Lists(int elemType,int nIdx,const int *localIdx)
{
	FEMAPI("FEM_Exchange_Ghost_Lists");
	getCurChunk()->exchangeGhostLists(elemType,nIdx,localIdx,0);
}
FDECL void FTN_NAME(FEM_EXCHANGE_GHOST_LISTS,fem_exchange_ghost_lists)
	(int *elemType,int *nIdx,const int *localIdx)
{
	FEMAPI("FEM_exchange_ghost_lists");
	getCurChunk()->exchangeGhostLists(*elemType-1,*nIdx,localIdx,1);
}
CDECL int FEM_Get_Ghost_List_Length(void) 
{
	FEMAPI("FEM_Get_Ghost_List_Length");
	return getCurChunk()->getList().size();
}
FDECL int FTN_NAME(FEM_GET_GHOST_LIST_LENGTH,fem_get_ghost_list_length)(void)
{ return FEM_Get_Ghost_List_Length();}

CDECL void FEM_Get_Ghost_List(int *dest)
{
	FEMAPI("FEM_Get_Ghost_List");
	int i,len=FEM_Get_Ghost_List_Length();
	const int *src=getCurChunk()->getList().getVec();
	for (i=0;i<len;i++) dest[i]=src[i];
	getCurChunk()->emptyList();
}
FDECL void FTN_NAME(FEM_GET_GHOST_LIST,fem_get_ghost_list)
	(int *dest)
{
	FEMAPI("FEM_get_ghost_list");
	int i,len=FEM_Get_Ghost_List_Length();
	const int *src=getCurChunk()->getList().getVec();
	for (i=0;i<len;i++) dest[i]=src[i]+1;
	getCurChunk()->emptyList();
}


/********* Debugging mesh printouts *******/
# define ARRSTART 0

void FEM_Item::print(const char *type,const l2g_t &l2g)
{
  CkPrintf("Number of %ss = %d\n", type, size());
  CkPrintf("User data doubles per %s = %d\n", type, getDataPer());
  if (getDataPer()!=0)
    for (int i=0;i<size();i++) {
      CkPrintf("\t%s[%d]%s userdata:",type,l2g.no(i)+ARRSTART,
	       isGhostIndex(i)?"(GHOST)":"");
      for (int j=0;j<getDataPer();j++)
      	CkPrintf("\t%f",udata(i,j));
      CkPrintf("\n");
    }
}

void FEM_Elem::print(const char *type,const l2g_t &l2g)
{
  CkPrintf("Number of %ss = %d\n", type, size());
  CkPrintf("User data doubles per %s = %d\n", type, getDataPer());
  if (getDataPer()!=0)
    for (int i=0;i<size();i++) {
      CkPrintf("\t%s[%d]%s userdata:",type,l2g.el(i)+ARRSTART,
	       isGhostIndex(i)?"(GHOST)":"");
      for (int j=0;j<getDataPer();j++)
      	CkPrintf("\t%f",udata(i,j));
      CkPrintf("\n");
    }
  CkPrintf("Nodes per %s = %d\n", type, getNodesPer());
    for (int i=0;i<size();i++) {
      CkPrintf("\t%s[%d] nodes:",type,l2g.el(i)+ARRSTART);
      for (int j=0;j<getNodesPer();j++)
      	CkPrintf("\t%d",l2g.no(conn(i,j))+ARRSTART);
      CkPrintf("\n");
    }	
}

void FEM_Mesh::print(const l2g_t &l2g)
{
  node.print("node",l2g);
  CkPrintf("%d kinds of element:\n",elem.size());
  if (elem.size()==1)
	elem[0].print("element",l2g);
  else 
  	for (int elType=0;elType<elem.size();elType++) {
  		char elName[50];
  		sprintf(elName,"El type %d",elType+ARRSTART);
  		elem[elType].print(elName,l2g);
  	}
}

void FEM_Comm::print(const l2g_t &l2g)
{
  CkPrintf("We communicate with %d other chunks:\n",size());
  for (int p=0;p<size();p++) {
    const FEM_Comm_List &l=getLocalList(p);
    CkPrintf("  With chunk %d, we share %d nodes:\n",l.getDest(),l.size());
    for (int n=0;n<l.size();n++)
      CkPrintf("\t%d",l2g.no(l[n]));
    CkPrintf("\n");
  }
}

class l2g_arr:public l2g_t {
public:
	const int *elMap,*noMap;
	l2g_arr(const int *NelMap,const int *NnoMap)
	  {elMap=NelMap;noMap=NnoMap;}
	//Return the global number associated with this local element
	virtual int el(int localNo) const {
		if (localNo==-1) return -1;
		return elMap[localNo];
	}
	//Return the global number associated with this local node
	virtual int no(int localNo) const {
		if (localNo==-1) return -1;
		return noMap[localNo];
	}
};

void
FEMchunk::print(void)
{
  int i;
  CkPrintf("-------------------- Chunk %d --------------------\n",thisIndex);
  cur_mesh->m.print(l2g_arr(cur_mesh->elemNums,cur_mesh->nodeNums));
  cur_mesh->comm.print(l2g_arr(cur_mesh->elemNums,cur_mesh->nodeNums));
  CkPrintf("[%d] Element global numbers:\n", thisIndex);
  for(i=0;i<cur_mesh->m.nElems();i++) 
    CkPrintf("%d\t",cur_mesh->elemNums[i]);
  CkPrintf("\n[%d] Node global numbers (* is primary):\n",thisIndex);
  for(i=0;i<cur_mesh->m.node.size();i++) 
    CkPrintf("%d%s\t",cur_mesh->nodeNums[i],getPrimary(i)?"*":"");
  CkPrintf("\n\n");
}  
  

/***************** Mesh Utility Classes **********/

int FEM_Mesh::chkET(int elType) const {
	if ((elType<0)||(elType>=elem.size())) {
		CkError("FEM Error! Bad element type %d!\n",elType);
		CkAbort("FEM Error! Bad element type used!\n");
	}
	return elType;
}


/*CommRec: lists the chunks that share a single item.
 */
FEM_Comm_Rec::FEM_Comm_Rec(int item_) {
	item=item_; 
}
FEM_Comm_Rec::~FEM_Comm_Rec() {}
void FEM_Comm_Rec::pup(PUP::er &p)
{
	p(item); 
	shares.pup(p);
}
void FEM_Comm_Rec::add(int chk,int idx) 
{
	int n=shares.size();
#ifndef CMK_OPTIMIZE
	if (chk<0 || chk>1000)
		CkAbort("FEM FEM_Comm_Rec::add> Tried to add absurd chunk number!\n");
#endif
	shares.setSize(n+1); //Grow slowly, to save memory
	shares.push_back(FEM_Comm_Share(chk,idx));
}

/*CommMap: map item number to FEM_Comm_Rec.  
 */
FEM_Comm_Map::FEM_Comm_Map() { }
void FEM_Comm_Map::pup(PUP::er &p) 
{
	int keepGoing=1;
	p.comment("Communication map object: maps item number to shared elements");
	if (!p.isUnpacking()) 
	{ //Pack the table, by iterating through its elements:
		CkHashtableIterator *it=map.iterator();
		FEM_Comm_Rec **rec;
		while (NULL!=(rec=(FEM_Comm_Rec **)it->next())) {
			p(keepGoing);
			(*rec)->pup(p);
		}
		keepGoing=0; //Zero marks end of list
		p(keepGoing);
		delete it;
	}
	else { //Unpack the table, inserting each element:
		while (1) {
			p(keepGoing);
			if (!keepGoing) break; //No more
			FEM_Comm_Rec *rec=new FEM_Comm_Rec;
			rec->pup(p);
			map.put(rec->getItem())=rec;
		}
	}
}
FEM_Comm_Map::~FEM_Comm_Map() {
	CkHashtableIterator *it=map.iterator();
	FEM_Comm_Rec **rec;
	while (NULL!=(rec=(FEM_Comm_Rec **)it->next()))
		delete *rec;
	delete it;
}

//Add a comm. entry for this item
void FEM_Comm_Map::add(int item,int chk,int idx)
{
	FEM_Comm_Rec *rec;
	if (NULL!=(rec=map.get(item))) 
	{ //Already have a record in the table
		rec->add(chk,idx);
	}
	else
	{ //Make new record for this item
		rec=new FEM_Comm_Rec(item);
		rec->add(chk,idx);
		map.put(item)=rec;
	}
}

//Look up this item's FEM_Comm_Rec.  Returns NULL if item is not shared.
const FEM_Comm_Rec *FEM_Comm_Map::get(int item) const
{
	return map.get(item);
}

FEM_Comm_List::FEM_Comm_List()
{
	pe=-1;
}
FEM_Comm_List::FEM_Comm_List(int otherPe)
{
	pe=otherPe;
}
FEM_Comm_List::~FEM_Comm_List() {}
void FEM_Comm_List::pup(PUP::er &p)
{
	p(pe);
	shared.pup(p);
}

FEM_Comm::FEM_Comm(void) {}
FEM_Comm::~FEM_Comm() {}
void FEM_Comm::pup(PUP::er &p)  //For migration
{
	comm.pup(p);
	map.pup(p);
}

void FEM_Comm::add(int myChunk,int myLocalNo,
	 int hisChunk,int hisLocalNo,FEM_Comm &his)
{
	FEM_Comm_List *myList=getListN(hisChunk);
	if (myList==NULL) 
	{//These two PEs have never communicated before-- must add to both lists
		//Figure out our names in the other guy's table
		myList=new FEM_Comm_List(hisChunk);
		FEM_Comm_List *hisList=new FEM_Comm_List(myChunk);
		comm.push_back(myList);
		his.comm.push_back(hisList);
	}
	FEM_Comm_List *hisList=his.getListN(myChunk);
	
	//Add our local numbers to our maps and lists
	map.add(myLocalNo,hisChunk,myList->size());
	myList->push_back(myLocalNo);
	his.map.add(hisLocalNo,myChunk,hisList->size());
	hisList->push_back(hisLocalNo);
}

FEM_Mesh::FEM_Mesh() {
}
void FEM_Mesh::pup(PUP::er &p)  //For migration
{
	p.comment(" ------------- Node Data ---------- ");
	node.pup(p);

	p.comment(" Number of element types:");
	elem.pup(p);
	
	p.comment("-------------- Sparse Data ------------");
	sparse.pup(p);
}
void FEM_Item::pup(PUP::er &p) {
	p.comment("<ghostStart>");
	p|ghostStart;
	p.comment(" Ghosts to send out: ");
	ghostSend.pup(p);
	p.comment(" Ghosts to recv: ");
	ghostRecv.pup(p);
	p.comment(" User data array: ");
	udata.pup(p);
	p.comment(" Symmetries array: ");
	int hasSym=0; if (sym) hasSym=1; p|hasSym;
	if (hasSym) {
		if (p.isUnpacking()) sym=new sym_t;
		sym->pup(p);
	}
}
void FEM_Elem::pup(PUP::er &p) {
	p.comment(" ------------- Element data ---------- ");
	FEM_Item::pup(p);
	p.comment(" element connectivity array: ");
	conn.pup(p);
}
FEM_Mesh::~FEM_Mesh()
{ 
	//Node and element destructors get called automatically
}

int FEM_Mesh::nElems(int t_max) const //Return total number of elements before type t_max
{
#ifndef CMK_OPTIMIZE
	if (t_max<0 || t_max>elem.size()) {
		CkPrintf("FEM> Invalid element type %d used!\n");
		CkAbort("FEM> Invalid element type");
	}
#endif
	int ret=0;
	for (int t=0;t<t_max;t++) ret+=elem[t].size();
	return ret;
}

int FEM_Mesh::getGlobalElem(int elType,int elNo) const
{
	int base=nElems(elType); //Global number of first element of this type
#ifndef CMK_OPTIMIZE
	if (elNo<0 || elNo>=elem[elType].size()) {
		CkPrintf("FEM> Element number %d is invalid-- element type %d has only %d elements\n",
			elNo,elType,elem[elType].size());
		CkAbort("FEM> Invalid element number, probably passed via FEM_Set_Sparse_Elem");
	}
#endif
	return base+elNo;
}


MeshChunk::MeshChunk(void)
{
	elemNums=NULL;
	nodeNums=NULL;
	isPrimary=NULL;
}
MeshChunk::~MeshChunk()
{
	delete [] elemNums;
	delete [] nodeNums;
	delete [] isPrimary;
}

void MeshChunk::pup(PUP::er &p) {
	p.comment("v1.1, Charm++ Finite Element Mesh file");	
	m.pup(p);
	p.comment(" ----------- Shared nodes ------------ ");	
	comm.pup(p);
	p.comment(" ----------- Global numbers ------------ ");	
	bool hasNums=(elemNums!=NULL);
	p(hasNums);
	if (hasNums) {
		if(p.isUnpacking()) { allocate(); }
		p(elemNums,m.nElems());
		p(nodeNums,m.node.size());
		p(isPrimary,m.node.size());
	}
	p.comment(" Misc. data: ");	
	p(updateCount); p(fromChunk);
	p(callMeshUpdated); p(doRepartition);
}

void
FEMchunk::pup(PUP::er &p)
{
//Pup superclass
  ArrayElement1D::pup(p);

// Pup the mesh fields
  bool hasUpdate=(updated_mesh!=NULL);
  p(hasUpdate);
  if(p.isUnpacking())
  {
    cur_mesh=new MeshChunk;
    if (hasUpdate) updated_mesh=new MeshChunk;
  }
  if (hasUpdate) updated_mesh->pup(p);
  cur_mesh->pup(p);

//Pup all other fields
  p(updateCount);
  messages=CmmPup(&p,messages);
  init.pup(p);
  p(ntypes);
  p((void*)dtypes, MAXDT*sizeof(DType));

  p(updateSeqnum);
  p(nRecd);

  p(doneCalled);
}


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
