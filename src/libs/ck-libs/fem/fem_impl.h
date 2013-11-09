/*Charm++ Finite Element Framework:
C++ implementation file

This is the main internal implementation file for FEM.
It includes all the other headers, and contains quite
a lot of assorted leftovers and utility routines.

Orion Sky Lawlor, olawlor@acm.org, 9/28/00
*/
#ifndef __CHARM_FEM_IMPL_H
#define __CHARM_FEM_IMPL_H

#include <stdio.h>

#include "charm-api.h"
#include "ckvector3d.h"
#include "pup_mpi.h"
#define checkMPI pup_checkMPI
#include "tcharm.h"
#include "fem.h"
#include "map.h"

#include "fem_mesh.h"
#include "idxl_layout.h"
#include "idxl.h"

/** \addtogroup fem_impl FEM Framework Library Implementation */
/*\@{*/

/* A stupid, stupid number: the maximum value of user-defined "elType" fields. 
  This should be dynamic, so any use of this should be considered a bug.
*/
#define FEM_MAX_ELTYPE 20

// Verbose abort routine used by FEM framework:
void FEM_Abort(const char *msg);
void FEM_Abort(const char *caller,const char *sprintf_msg,int int0=0,int int1=0,int int2=0);


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
		vec.resize(newSize);
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
	T& operator[](int i) {return sto[i];}
	const T& operator[](int i) const {return sto[i];}
};
typedef ArrayPtrT<int> intArrayPtr;



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
typedef marshallNewHeapCopy<FEM_Mesh> marshallMeshChunk;


/// Keeps a list of dynamically-allocated T objects,
///  indexed by a user-carried, persistent "int".
template<class T>
class FEM_T_List {
	CkPupPtrVec<T> list; // Vector of T's
protected:
	int FIRST_DT; // User index of first T
	int size(void) const {return list.size();}
	
	/// If this isn't a valid, allocated index, abort.
	inline void check(int l,const char *caller) const {
		if (l<FIRST_DT || l>=FIRST_DT+list.size() || list[l-FIRST_DT]==NULL) 
			badIndex(l,caller);
	}
	
	void badIndex(int l,const char *caller) const {
		if (l<FIRST_DT || l>FIRST_DT+list.size()) bad(l,0,caller);
		else bad(l,1,caller);
	}
public:
	FEM_T_List(int FIRST_DT_) :FIRST_DT(FIRST_DT_) {}
	virtual ~FEM_T_List() {}
	void pup(PUP::er &p) { p|list; }
	
	/// This routine is called when we're passed an invalid T index.
	virtual void bad(int l,int bad_code,const char *caller) const =0;
	
	/// Insert a new T (allocated with "new"), returning the user index:
	int put(T *t) {
		for (int i=0;i<list.size();i++) 
			if (list[i]==NULL) {
				list[i]=t;
				return FIRST_DT+i;
			}
		int ret=list.size();
		list.push_back(t);
		return FIRST_DT+ret;
	}
	
	/// Get this T given its user index.
	inline T *lookup(int l,const char *caller) const {
		check(l,caller);
		return list[l-FIRST_DT];
	}
	
	/// Free this T
	void destroy(int l,const char *caller) {
		check(l,caller);
		list[l-FIRST_DT].destroy();
	}
	
	/// Clear all stored T's:
	void empty(void) {
		for (int i=0;i<list.size();i++) list[i].destroy();
	}
};
class FEM_Mesh_list : public FEM_T_List<FEM_Mesh> {
	typedef FEM_T_List<FEM_Mesh> super;
public:
	FEM_Mesh_list() :super(FEM_MESH_FIRST) { }
	
	virtual void bad(int l,int bad_code,const char *caller) const;
};

#define CHK(p) do{if((p)==0)CkAbort("FEM>Memory Allocation failure.");}while(0)

/**
  FEM global data object.  Keeps track of the global
  list of meshes, and the default read and write meshes.
  
  This class was once an array element, back when the 
  FEM framework was built directly on Charm++.
  
  There's only one of this object per thread, and it's
  kept in a thread-private variable.
*/
class FEMchunk 
{
public:
  FEM_Mesh_list meshes; ///< Global list of meshes.
  int default_read; ///< Index of default read mesh.
  int default_write; ///< Index of default write mesh.
  
  /// Default communicator to use
  FEM_Comm_t defaultComm;

  /// Global index (rank) in default communicator
  int thisIndex;

#if CMK_ERROR_CHECKING /* Do an extensive self-check */
  void check(const char *where);
#else  /* Skip the check, for speed. */
  inline void check(const char *where) { }
#endif

private:
  CkVec<int> listTmp;//List of local entities, for ghost list exchange
 
  void initFields(void);

 public:
  FEMchunk(FEM_Comm_t defaultComm_);
  FEMchunk(CkMigrateMessage *msg);
  void pup(PUP::er &p);
  ~FEMchunk();
  
  /// Return this thread's single static FEMchunk instance:
  static FEMchunk *get(const char *caller);
  
  inline FEM_Mesh *lookup(int fem_mesh,const char *caller) {
     return meshes.lookup(fem_mesh,caller);
  }

  inline FEM_Mesh *getMesh(const char *caller) 
  	{return meshes.lookup(default_read,caller);}
  inline FEM_Mesh *setMesh(const char *caller) 
  	{return meshes.lookup(default_write,caller);}

  void print(int fem_mesh,int idxBase);
  int getPrimary(int nodeNo) { return getMesh("getPrimary")->node.getPrimary(nodeNo); }
  const FEM_Comm &getComm(void) {return getMesh("getComm")->node.shared;}

  // Basically everything below here should be moved to IDXL:
  void exchangeGhostLists(int elemType,int inLen,const int *inList,int idxbase);
  void recvList(int elemType,int fmChk,int nIdx,const int *idx);
  const CkVec<int> &getList(void) {return listTmp;}
  void emptyList(void) {listTmp.length()=0;}
  
  void reduce_field(int idxl_datatype, const void *nodes, void *outbuf, int op);
  void reduce(int idxl_datatype, const void *inbuf, void *outbuf, int op);
  void readField(int idxl_datatype, void *nodes, const char *fname);  
};

/// Describes a single layer of ghost elements.
class FEM_Ghost_Layer : public CkNoncopyable {
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
		void pup(PUP::er &p) {//CkAbort("FEM> Shouldn't call elemGhostInfo::pup!\n");
		}
	};
	elemGhostInfo elem[FEM_MAX_ELTYPE];
	virtual void pup(PUP::er &p){
		p | nodesPerTuple;
		p | addNodes;
		for(int i=0;i<FEM_MAX_ELTYPE;i++){
			p | elem[i].add;
			p | elem[i].tuplesPerElem;
			if(elem[i].tuplesPerElem == 0){
				continue;
			}
			int *arr;
			if(p.isUnpacking()){
				arr = new int[nodesPerTuple*elem[i].tuplesPerElem];
			}else{
				arr = elem[i].elem2tuple;
			}
			p(arr,nodesPerTuple*elem[i].tuplesPerElem);
			if(p.isUnpacking()){
				elem[i].elem2tuple = arr;
			}
		}
	}
};

/// Describes a set of required adjacent elements for this kind of element,
///  stored as an explicit adjacency list.
class FEM_Ghost_Stencil {
	/// Our element type.
	int elType;
	/// Number of elements we describe.
	int n;
	/// If true, add ghost nodes as well as elements
	bool addNodes;
	
	/// Last adjacency entry (plus one), indexed by element.
	///  That is, element i's data is at [ends[i-1],ends[i])
	intArrayPtr ends;
	
	/** Adjacency entries for each element.
	  Stored as a series of pairs: elType, elNum.
	  The first pair for element i starts at
	  	2*(ends[i-1]) 
	  the last pair for element i starts at
	  	2*(ends[i]-1)
	  This array then has, in total, 2*ends[n-1] elements.
	*/
	intArrayPtr adj;
public:
	/**
	  Create a stencil with this number of elements, 
	  and these adjecent elements.
	*/
	FEM_Ghost_Stencil(int elType_, int n_,bool addNodes_,
		const int *ends_,
		const int *adj_,
		int idxBase);
	
	/// Make sure this stencil makes sense for this mesh.
	void check(const FEM_Mesh &mesh) const;
	
	/// Return the type of element we describe
	inline int getType(void) const {return elType;}
	
	/**
	  Return a pair consisting of the i'th element's
	  j'th neighbor: the return value's first int is an element type,
	  the second int is an element number.  Returns NULL if i doesn't
	  have a j'th neighbor.
	*/
	inline const int *getNeighbor(int i,int j) const {
		int start=0;
		if (i>0) start=ends[i-1];
		if (j>=ends[i]-start) return 0;
		return &adj[2*(start+j)];
	}
	
	inline bool wantNodes(void) const {return addNodes;}
};

/// Describes a way to grow a set of ghosts.
class FEM_Ghost_Region {
public:	
	FEM_Ghost_Layer *layer;
	FEM_Ghost_Stencil *stencil;
	
	FEM_Ghost_Region() {layer=0; stencil=0;}
	FEM_Ghost_Region(FEM_Ghost_Layer *l) {layer=l; stencil=0;}
	FEM_Ghost_Region(FEM_Ghost_Stencil *s) {layer=0; stencil=s;}
};


//Accumulates all symmetries of the mesh before splitting:
class FEM_Initial_Symmetries; /*Defined in symmetries.C*/

/// Describes all the data needed for partitioning a mesh.
class FEM_Partition : public CkNoncopyable {
	/// Maps element number to (0-based) chunk number, allocated with new[]
	int *elem2chunk;
	
	/// Describes the different regions of ghost elements:
	CkVec<FEM_Ghost_Region> regions;
	FEM_Ghost_Layer *lastLayer;
	
	/// Describes the problem domain's spatial symmetries.
	FEM_Initial_Symmetries *sym;
public:
	FEM_Partition();
	~FEM_Partition();
	
// Manipulate partitioning information
	void setPartition(const int *elem2chunk, int nElem, int idxBase);
	const int *getPartition(FEM_Mesh *src,int nChunks) const;
	
// Manipulate ghost layers
	FEM_Ghost_Layer *addLayer(void) {
		lastLayer=new FEM_Ghost_Layer();
		regions.push_back(lastLayer);
		return lastLayer;
	}
	FEM_Ghost_Layer *curLayer(void) {
		if (lastLayer==0) CkAbort("Must call FEM_Add_ghost_layer before FEM_Add_ghost_elem\n");
		return lastLayer;
	}
	
// Manipulate ghost stencils
	void addGhostStencil(FEM_Ghost_Stencil *s) {
		regions.push_back(s);
		lastLayer=0;
	}
	void markGhostStencilLayer(void) {
		regions.push_back(FEM_Ghost_Region());
	}
	
// Read back ghost regions
	int getRegions(void) const {return regions.size();}
	const FEM_Ghost_Region &getRegion(int regNo) const {return regions[regNo];}
	
// Manipulate spatial symmetries:
	void setSymmetries(int nNodes_,int *new_can,const int *sym_src);
	void addLinearPeriodic(int nFaces_,int nPer,
		const int *facesA,const int *facesB,int idxBase,
		int nNodes_,const CkVector3d *nodeLocs);
	const int *getCanon(void) const;
	const FEM_Symmetries_t *getSymmetries(void) const;
	const FEM_Sym_List &getSymList(void) const;


};
// Access the latest partition:
FEM_Partition &FEM_curPartition(void);

//Declare this at the start of every API routine:
#define FEMAPI(routineName) TCHARM_API_TRACE(routineName,"fem")
//#define FEMAPI(routineName) printf("%s\n", routineName);


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
void FEM_Mesh_split(FEM_Mesh *mesh,int nchunks,
	const FEM_Partition &partition,FEM_Mesh_Output *out);


//Make a new[]'d copy of this (len-entry) array, changing the index as spec'd
int *CkCopyArray(const int *src,int len,int indexBase);


// Isaac's stuff:
// Describes Element Faces. For use with finding element->element adjacencies
// based on FEM_Ghost_Layer
class FEM_ElemAdj_Layer : public CkNoncopyable {
 public:
  int initialized;
  int nodesPerTuple; //Number of shared nodes for a pair of elements
 
 class elemAdjInfo {
  public:
    //  int recentElType; // should not be here, but if it is it should be pup'ed
    int tuplesPerElem; //# of tuples surrounding this element, i.e. number of faces on an element
    intArrayPtr elem2tuple; //The tuples around this element [nodesPerTuple * tuplesPerElem]
    elemAdjInfo(void) {/*add=false;*/tuplesPerElem=0;}
    ~elemAdjInfo(void) {}
    void pup(PUP::er &p) {//CkAbort("FEM> Shouldn't call elemGhostInfo::pup!\n");
    }
  };
 
 elemAdjInfo elem[FEM_MAX_ELTYPE];

 FEM_ElemAdj_Layer() {initialized=0;}

  virtual void pup(PUP::er &p){
    p | nodesPerTuple;
	p | initialized;
    for(int i=0;i<FEM_MAX_ELTYPE;i++){
      p | elem[i].tuplesPerElem;
	  if(elem[i].tuplesPerElem == 0){
	continue;
      }
      int *arr;
      if(p.isUnpacking()){
	arr = new int[nodesPerTuple*elem[i].tuplesPerElem];
      }else{
	arr = elem[i].elem2tuple;
      }
      p(arr,nodesPerTuple*elem[i].tuplesPerElem);
      if(p.isUnpacking()){
	elem[i].elem2tuple = arr;
      }
    }
  }
};


/*\@}*/

#endif


