/*Charm++ Finite Element Framework:
C++ implementation file

This is the main internal implementation file for FEM.
Orion Sky Lawlor, olawlor@acm.org, 9/28/00
*/
#ifndef __CHARM_FEM_IMPL_H
#define __CHARM_FEM_IMPL_H

#include <stdio.h>

#include "charm-api.h"
#include "tcharm.h"
#include "ckvector3d.h"
#include "fem.h"

#include "fem_mesh.h"
#include "idxl_layout.h"
#include "idxl.h"

/** \addtogroup fem_impl FEM Framework Library Implementation */
/*\@{*/

// Verbose abort routine used by FEM framework:
void FEM_Abort(const char *msg);
void FEM_Abort(const char *callingRoutine,const char *sprintf_msg,int int0=0,int int1=0,int int2=0);

class FEMinit {
 public:
	int numElements;
	CkArrayID threads;
	int flags;
	CkChareID coordinator;
	FEMinit() {}
	FEMinit(int ne_,const CkArrayID &t_,int f_,const CkChareID &c_)
		:numElements(ne_),threads(t_),flags(f_),coordinator(c_) {}
	void pup(PUP::er &p) {
		p|numElements;
		p|threads;
		p|flags;
		p|coordinator;
	}
};
PUPmarshall(FEMinit);

/*This class describes a local-to-global index mapping, used in FEM_Print.
The default is the identity mapping.*/
class l2g_t {
public:
	//Return the global number associated with this local element
	virtual int el(int t,int localNo) const {return localNo;}
	//Return the global number associated with this local node
	virtual int no(int localNo) const {return localNo;}
};

/* Map (user-assigned) numbers to T's */
template <class T>
class NumberedVec {
	CkPupPtrVec<T, CkPupAlwaysAllocatePtr<T> > vec;
	
public:
	//Extend the vector to have up to this element
	void makeLonger(int toHaveElement)
	{
		int oldSize=vec.size(), newSize=toHaveElement+1;
		if (oldSize>=newSize) return; //Nothing to do
		vec.setSize(newSize);
		vec.length()=newSize;
		for (int j=oldSize;j<newSize;j++)
			vec[j]=new T;
	}
	//Reinitialize element i:
	void reinit(int doomedEl) {
		vec[doomedEl].destroy();
		vec[doomedEl]=new T;
	}
	
	int size(void) const {return vec.size();}
	
	//Same old bracket operators, but return the actual object, not a pointer:
	T &operator[](int i) {
		if (i>=vec.size()) makeLonger(i);
		return *( vec[i] );
	}
	const T &operator[](int i) const {return *( vec[i] );}
	
	void pup(PUP::er &p) {
		vec.pup(p);
	}
	friend void operator|(PUP::er &p,NumberedVec<T> &v) {v.pup(p);}
};


//Smart pointer-to-new[]'d array-of-T
template <class T>
class ArrayPtrT : public CkNoncopyable {
	T *sto;
public:
	ArrayPtrT() {sto=NULL;}
	ArrayPtrT(int *src) {sto=src;}
	~ArrayPtrT() {if (sto) delete[] sto;}
	void operator=(T *src) {
		if (sto) delete[] sto;
		sto=src;
	}
	operator T *(void) {return sto;}
	operator const T *(void) const {return sto;}
};
typedef ArrayPtrT<int> intArrayPtr;


/// Describes how to call the serial meshUpdated routine:
class CallMeshUpdated {
	int val; //Value to pass to function below
	FEM_Update_mesh_fn cfn; //if 0, skip meshUpdated call
	FEM_Update_mesh_fortran_fn ffn; //if 0, skip f90 meshUpdated call
public:
	CallMeshUpdated() 
		:val(0), cfn(0), ffn(0) {}
	CallMeshUpdated(FEM_Update_mesh_fn cfn_,int val_) 
		:val(val_), cfn(cfn_), ffn(0) {}
	CallMeshUpdated(FEM_Update_mesh_fortran_fn ffn_,int val_) 
		:val(val_), cfn(0), ffn(ffn_) {}
	/// Call the user's meshUpdated function:
	void call(void) 
	{ 
		if (cfn) { cfn(val); }
		if (ffn) { ffn(&val); }
	}
};
PUPmarshallBytes(CallMeshUpdated);

class UpdateMeshChunk {
public:
	FEM_Mesh m; //The mesh to update
	
	int updateCount; //Mesh update serial number
	int fromChunk; //Source chunk
	CallMeshUpdated meshUpdated;
	int doWhat; //If 0, do nothing; if 1, repartition; if 2, resume
	
	UpdateMeshChunk() {
		updateCount=fromChunk=doWhat=0;
	}
	void pup(PUP::er &p) {
		m.pup(p);
		p.comment(" UpdateMesh data: ");	
		p(updateCount); p(fromChunk);
		p|meshUpdated;
		p(doWhat);
	}
};
PUPmarshallBytes(UpdateMeshChunk);

/* Unmarshall into a heap-allocated copy */
template<class T>
class marshallNewHeapCopy {
	T *cur;
public:
	//Used on send side:
	marshallNewHeapCopy(T *readFrom) :cur(readFrom) {}
	marshallNewHeapCopy(const marshallNewHeapCopy &h) :cur(h.cur) {}
	marshallNewHeapCopy(void) { //Used on recv side:
		cur=new T;
	}
	
	void pup(PUP::er &p) {
		cur->pup(p);
	}
	operator T *() {return cur;}
	friend void operator|(PUP::er &p,marshallNewHeapCopy<T> &h) {h.pup(p);}
};
typedef marshallNewHeapCopy<UpdateMeshChunk> marshallUpdateMeshChunk;
typedef marshallNewHeapCopy<FEM_Mesh> marshallMeshChunk;


#include "fem.decl.h"

#define CHK(p) do{if((p)==0)CkAbort("FEM>Memory Allocation failure.");}while(0)

class FEMchunk : public IDXL_Chunk
{
  typedef IDXL_Chunk super;
public:
// updated_mesh keeps the still-being-assembled next mesh chunk.
// It is created and written by the FEM_Set routines called from driver.
  UpdateMeshChunk *updated_mesh;
  int updateCount; //Number of mesh updates

  //The current finite-element mesh
  FEM_Mesh *cur_mesh;
  
private:
  FEMinit init;

  CProxy_FEMchunk thisproxy;
  TCharm *thread;
  IDXL_Comm_t comm;
  
  void *reductionBuf; //Place to return reduction result
  
  typedef enum {INVALID_UPDATE,NODE_UPDATE,GHOST_UPDATE} updateType_t;
  
  CkVec<int> listTmp;//List of local entities 
  int listCount; //Number of lists received
  bool listSuspended;
  bool finishListExchange(const FEM_Comm &l);

  void initFields(void);
  void setMesh(FEM_Mesh *msg=0);

 public:

  int tsize;
  int doneCalled;

  FEMchunk(const FEMinit &init);
  FEMchunk(CkMigrateMessage *msg);
  ~FEMchunk();

  void ckJustMigrated(void);

  void run(void);
  void run(marshallMeshChunk &);
  void reductionResult(int length,const char *data);
  void updateMesh(int doWhat);
  void meshUpdated(marshallMeshChunk &);
  void meshUpdatedComplete(void) {thread->resume();}
  
  void exchangeGhostLists(int elemType,int inLen,const int *inList,int idxbase);
  void recvList(int elemType,int fmChk,int nIdx,const int *idx);
  const CkVec<int> &getList(void) {return listTmp;}
  void emptyList(void) {listTmp.length()=0;}
  
  void reduce_field(int idxl_datatype, const void *nodes, void *outbuf, int op);
  void reduce(int idxl_datatype, const void *inbuf, void *outbuf, int op);
  void readField(int idxl_datatype, void *nodes, const char *fname);
  void print(int idxBase);
  
  FEM_Mesh &getMesh(void) { return *cur_mesh; }
  int getPrimary(int nodeNo) { return cur_mesh->node.getPrimary(nodeNo); }
  const FEM_Comm &getComm(void) const {return cur_mesh->node.shared;}
  static FEMchunk *lookup(const char *callingRoutine);
  FEM_Mesh *meshLookup(int fem_mesh,const char *callingRoutine);

  void pup(PUP::er &p);
};


//Describes a single layer of ghost elements
class ghostLayer : public CkNoncopyable {
public:
	int nodesPerTuple; //Number of shared nodes needed to connect elements
	bool addNodes; //Add ghost nodes to the chunks
	class elemGhostInfo {
	public:
		bool add; //Add this kind of ghost element to the chunks
		int tuplesPerElem; //# of tuples surrounding this element
		intArrayPtr elem2tuple; //The tuples around this element [nodesPerTuple * tuplesPerElem]
		elemGhostInfo(void) {add=false;tuplesPerElem=0;}
		~elemGhostInfo(void) {}
		void pup(PUP::er &p) {CkAbort("FEM> Shouldn't call elemGhostInfo::pup!\n");}
	};
	NumberedVec<elemGhostInfo> elem;
};

//Accumulates all symmetries of the mesh before splitting:
class FEM_Initial_Symmetries; /*Defined in symmetries.C*/

//Describes all ghost elements
class FEM_Ghost : public CkNoncopyable {
	CkVec<ghostLayer *> layers;
	
	FEM_Initial_Symmetries *sym;
public:
	FEM_Ghost() {sym=NULL;}
	~FEM_Ghost() {for (int i=0;i<getLayers();i++) delete layers[i];}
	
	int getLayers(void) const {return layers.size();}
	ghostLayer *addLayer(void) {
		ghostLayer *l=new ghostLayer();
		layers.push_back(l);
		return l;
	}
	const ghostLayer &getLayer(int layerNo) const {return *layers[layerNo];}
	
	void setSymmetries(int nNodes_,int *new_can,const int *sym_src);
	void addLinearPeriodic(int nFaces_,int nPer,
		const int *facesA,const int *facesB,int idxBase,
		int nNodes_,const CkVector3d *nodeLocs);
	const int *getCanon(void) const;
	const FEM_Symmetries_t *getSymmetries(void) const;
	const FEM_Sym_List &getSymList(void) const;
};

//Declare this at the start of every API routine:
#define FEMAPI(routineName) TCHARM_API_TRACE(routineName,"fem")



/*Partition this mesh's elements into n chunks,
 writing each element's 0-based chunk number to elem2chunk.
*/
void FEM_Mesh_partition(const FEM_Mesh *mesh,int nchunks,int *elem2chunk);

/*A way to stream out partitioned chunks of a mesh.
  By streaming, we can send the chunks as they are built,
  dramatically reducing the memory needed by the framework.
*/
class FEM_Mesh_Output {
 public:
	virtual ~FEM_Mesh_Output() {} /*<- for whining compilers*/
	//Transfer ownership of this mesh chunk
	virtual void accept(int chunkNo,FEM_Mesh *msg) =0;
};

/*After partitioning, create a sub-mesh for each chunk's elements,
including communication lists between chunks.
*/
void FEM_Mesh_split(FEM_Mesh *mesh,int nchunks,const int *elem2chunk,
	       const FEM_Ghost &ghosts,FEM_Mesh_Output *out);


//Make a new[]'d copy of this (len-entry) array, changing the index as spec'd
int *CkCopyArray(const int *src,int len,int indexBase);
const FEM_Mesh *FEM_Get_FEM_Mesh(void);
FEM_Ghost &FEM_Set_FEM_Ghost(void);

/*\@}*/

#endif


