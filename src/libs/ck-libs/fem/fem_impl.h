/*Charm++ Finite Element Framework:
C++ implementation file

This is the under-the-hood implementation file for FEM.
Orion Sky Lawlor, olawlor@acm.org, 9/28/00
*/
#ifndef _FEM_IMPL_H
#define _FEM_IMPL_H

#include <stdio.h>

#include "charm-api.h"
#include "tcharm.h"
#include "ckvector3d.h"
#include "fem.h"

// temporary Datatype representation
// will go away once MPI user-defined datatypes are ready
struct DType {
  int base_type; //FEM_* datatype
  int vec_len; //Number of items of this datatype
  int init_offset; // offset of field in bytes from the beginning of data
  int distance; // distance in bytes between successive field values
  DType(void) {}
  DType( const int b,  const int v=1,  const int i=0,  const int d=0)
    : base_type(b), vec_len(v), init_offset(i) 
  {
    distance = (d ? d : length());
  }
  //Default copy constructor, assignment operator

  //Return the total number of bytes required by this FEM_* data type
  static int type_size(int dataType) {
    switch(dataType) {
      case FEM_BYTE : return 1; break;
      case FEM_INT : return sizeof(int); break;
      case FEM_REAL : return sizeof(float); break;
      case FEM_DOUBLE : return sizeof(double); break;
      default: CkAbort("Unrecognized data type field passed to FEM framework!\n");
    }
    return -1;
  }
  
  //Return the total number of bytes required by the data stored in this DType
  int length(const int nitems=1) const {
    return type_size(base_type) * vec_len * nitems;
  }
};

//This datatype is how the framework stores symmetries internally.
//  Each bit of this type describes a different symmetry.
//  There must be enough bits to accomidate several simulatanious symmetries.
typedef unsigned char FEM_Symmetries_t;

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
	virtual int el(int localNo) const {return localNo;}
	//Return the global number associated with this local node
	virtual int no(int localNo) const {return localNo;}
};

/* Map (user-assigned) numbers to T's */
template <class T>
class NumberedVec : public CkPupPtrVec<T> {
	typedef CkPupPtrVec<T> super;
	
public:
	//Extend the vector to have up to this element
	void makeLonger(int toHaveElement)
	{
		int oldSize=size(), newSize=toHaveElement+1;
		if (oldSize>=newSize) return; //Nothing to do
		setSize(newSize);
		length()=newSize;
		for (int j=oldSize;j<newSize;j++)
			(*(super *)this)[j]=new T;
	}
	//Reinitialize element i:
	void reinit(int doomedEl) {
		delete (*(super *)this)[doomedEl];
		(*(super *)this)[doomedEl]=new T;
	}
	
	//Same old bracket operators, but return the actual object, not a pointer:
	T &operator[](int i) {return *( (*(super *)this)[i] );}
	const T &operator[](int i) const {return *( (*(const super *)this)[i] );}
};


/*Inner class used by FEM_Comm_Rec:*/
class FEM_Comm_Share {
 public:
  int chk; //Chunk we're shared with
  int idx; //Our index in the local comm. list for that chunk
  FEM_Comm_Share(int x=0) {chk=idx=-1;}
  FEM_Comm_Share(int c,int i) :chk(c), idx(i) {}
  void pup(PUP::er &p) {p(chk); p(idx);}
};
PUPmarshall(FEM_Comm_Share);

/* List the chunks that share an item */
class FEM_Comm_Rec {
	int item; //Index of item we describe
	CkPupVec<FEM_Comm_Share> shares;
public:
	FEM_Comm_Rec(int item_=-1);
	~FEM_Comm_Rec();

	void pup(PUP::er &p);
	
	inline int getItem(void) const {return item;}
	inline int getShared(void) const {return shares.size();}
	inline int getChk(int shareNo) const {return shares[shareNo].chk;}
	inline int getIdx(int shareNo) const {return shares[shareNo].idx;}
	bool hasChk(int chk) const {
		for (int i=0;i<getShared();i++)
			if (getChk(i)==chk) return true;
		return false;
	}
	void add(int chk,int idx);
};

/* Map an item to its FEM_Comm_Rec (if any) */
class FEM_Comm_Map {
	CkHashtableT<CkHashtableAdaptorT<int>,FEM_Comm_Rec *> map;
public:
	FEM_Comm_Map();
	void pup(PUP::er &p);
	~FEM_Comm_Map();

	//Add a FEM_Comm_. entry for this item
	void add(int item,int chk,int idx);
	
	//Look up this item's FEM_Comm_Rec.  Returns NULL if item is not shared.
	const FEM_Comm_Rec *get(int item) const;
};

/* Lists the items we share with one other chunk. */
class FEM_Comm_List {
	int pe; //Global number of other chunk	
	CkPupBasicVec<int> shared; //Local indices of shared items
public:
	FEM_Comm_List();
	FEM_Comm_List(int otherPe);
	~FEM_Comm_List();
	int getDest(void) const {return pe;}
	int size(void) const {return shared.size();}
	int operator[](int idx) const {return shared[idx]; }
	const int *getVec(void) const {return &shared[0];}
	int push_back(int localIdx) {
		int ret=shared.size();
		shared.push_back(localIdx);
		return ret;
	}
	void pup(PUP::er &p);
};

/*This class describes all the shared items of a given chunk.
It provides both item->chunks that share it (map)
and chunk->items shared with it (comm)
*/
class FEM_Comm : public CkNoncopyable {
	FEM_Comm_Map map; //Indexed by local item number
	CkPupPtrVec<FEM_Comm_List> comm; //Indexed by (local) chunk number
	
	//Return the Comm_List associated with processor pe
	FEM_Comm_List *getListN(int pe) { 
		for (int i=0;i<comm.size();i++)
			if (comm[i]->getDest()==pe)
				return comm[i];
		return NULL; 
	}
public:
	FEM_Comm(void);
	~FEM_Comm();
	int totalShared() const;//Return total number of shared nodes
	void pup(PUP::er &p); //For migration
	
	int size(void) const {return comm.size();}
	//Return the i'th (local) chunk we communicate with
	const FEM_Comm_List &getLocalList(int idx) const { return *comm[idx]; }
	int findLocalList(int pe) const {
		for (int i=0;i<comm.size();i++) 
			if (comm[i]->getDest()==pe)
				return i;
		return -1;
	}
	
	//Return the Comm_List associated with processor pe
	const FEM_Comm_List &getList(int pe) const { 
		const FEM_Comm_List *ret=((FEM_Comm *)this)->getListN(pe);
		if (ret==NULL) CkAbort("FEM> Communication lists corrupted (unexpected message)");
		return *ret; 
	}
	//Return the FEM_Comm_List for processor pe, adding if needed
	FEM_Comm_List &addList(int pe) {
		FEM_Comm_List *ret=getListN(pe);
		if (ret==NULL) { //Have to add a new list:
			ret=new FEM_Comm_List(pe);
			comm.push_back(ret);
		}
		return *ret;
	}
	
	//Look up an item's FEM_Comm_Rec
	const FEM_Comm_Rec *getRec(int item) const {return map.get(item);}
	
	//This item is shared with the given (local) chunk
	void addNode(int localNo,int sharedWithChk) {
		map.add(localNo,sharedWithChk,
			comm[sharedWithChk]->push_back(localNo));
	}
	
	//Used in creating comm. lists:
	//Add this local number to both lists:
	void add(int myChunk,int myLocalNo,
		 int hisChunk,int hisLocalNo,FEM_Comm &hisList);
	
	void print(const l2g_t &l2g);
};


/*
This is a simple 2D table.  The operations are mostly row-centric.
*/
template <class T>
class BasicTable2d : public CkNoncopyable {
protected:
	int rows; //Number of entries in table
	int cols; //Size of each entry in table
	T *table; //Data in table [rows * cols]
public:
	BasicTable2d(T *src,int cols_,int rows_) 
		:rows(rows_), cols(cols_), table(src) {}
	
	//"size" of the table is the number of rows:
	inline int size(void) const {return rows;}
	//Width of the table is the number of columns:
	inline int width(void) const {return cols;}
	
	T *getData(void) {return table;}
	const T *getData(void) const {return table;}
	
//Element-by-element operations:
	T operator() (int r,int c) const {return table[c+r*cols];}
	T &operator() (int r,int c) {return table[c+r*cols];}
	
//Row-by-row operations
	//Get a pointer to a row of the table:
	inline T *getRow(int r) {return &table[r*cols];}
	inline const T *getRow(int r) const {return &table[r*cols];}
	inline T *operator[](int r) {return getRow(r);}
	inline const T *operator[](int r) const {return getRow(r);}
	inline void setRow(int r,const T *src,T idxBase=0) {
		T *dest=getRow(r);
		for (int c=0;c<cols;c++) dest[c]=src[c]-idxBase;
	}
	inline void setRow(int r,T value) {
		T *dest=getRow(r);
		for (int c=0;c<cols;c++) dest[c]=value;
	}
	
//These affect the entire table:
	void set(const T *src,T idxBase=0) {
		for (int r=0;r<rows;r++) 
		for (int c=0;c<cols;c++)
			table[c+r*cols]=src[c+r*cols]-idxBase;
	}
	void setTranspose(const T *srcT,int idxBase=0) {
		for (int r=0;r<rows;r++) 
		for (int c=0;c<cols;c++)
			table[c+r*cols]=srcT[r+c*rows]-idxBase;
	}
	void get(T *dest,T idxBase=0) const {
		for (int r=0;r<rows;r++) 
		for (int c=0;c<cols;c++)
			dest[c+r*cols]=table[c+r*cols]+idxBase;
	}
	void getTranspose(T *destT,int idxBase=0) const {
		for (int r=0;r<rows;r++) 
		for (int c=0;c<cols;c++)
			destT[r+c*rows]=table[c+r*cols]+idxBase;
	}
	void set(T value) {
		for (int r=0;r<rows;r++) setRow(r,value);
	}
};

//As above, but heap-allocatable and resizable.
// T must not require a copy constructor.
template <class T>
class AllocTable2d : public BasicTable2d<T> {
	int max; //Maximum number of rows that can be used without reallocation
public:
	AllocTable2d(int cols_=0,int rows_=0) 
		:BasicTable2d<T>(NULL,cols_,rows_), max(0)
	{
		if (rows>0) allocate(rows);
	}
	~AllocTable2d() {delete[] table;}
	void allocate(int rows_) { //Make room for this many rows
		allocate(width(),rows_,rows_);
	}
	void allocate(int cols_,int rows_,int max_=0) { //Make room for this many cols & rows
		int oldRows=rows;
		T *oldTable=table;
		if (max_==0) max_=rows_;
		cols=cols_;
		rows=rows_;
		max=max_;
		table=new T[max*cols];
		if (oldTable!=NULL) { //Preserve old table entries:
			memcpy(table,oldTable,sizeof(T)*cols*oldRows);
			delete[] oldTable;
		}
	}
	
	//Pup routine and operator|:
	void pup(PUP::er &p) {
		p|rows; p|cols;
		if (table==NULL) allocate(rows);
		p(table,rows*cols); //T better be a basic type, or this won't compile!
	}
	friend void operator|(PUP::er &p,AllocTable2d<T> &t) {t.pup(p);}

	//Add a row to the table (by analogy with std::vector):
	T *push_back(void) {
		if (rows>=max) 
		{ //Not already enough room for the new row:
			int newMax=max+(max/4)+16; //Grow 25% longer
			allocate(cols,rows,newMax);
		}
		rows++;
		return getRow(rows-1);
	}
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


/* Describes an FEM item-- a set of nodes or elements */
class FEM_Item : public CkNoncopyable {
public:
	typedef AllocTable2d<double> udata_t;
	typedef CkPupBasicVec<FEM_Symmetries_t> sym_t;
protected:
	int ghostStart; //Index of first ghost object
	udata_t udata; //Uninterpreted item-associated user data (one row per item)
	sym_t *sym; //Symmetries of each item (or NULL if all 0)
public:
	FEM_Comm ghostSend; //Non-ghosts we send out
	FEM_Comm ghostRecv; //Ghosts we recv into
	
	FEM_Item() //Default constructor
	  {ghostStart=0;sym=NULL;}
	~FEM_Item() {if (sym) delete sym;}
	void pup(PUP::er &p);
	void print(const char *type,const l2g_t &l2g);
	
	//Manipulate the user data array:
	void allocate(int nItems,int dataPer);
	udata_t &setUdata(void) {return udata;}
	const udata_t &getUdata(void) const {return udata;}
	inline int size(void) const {return udata.size();}
	inline int getDataPer(void) const {return udata.width();}
	void udataIs(int r,const double *src) {udata.setRow(r,src);}
	const double *udataFor(int r) const {return udata.getRow(r);}
	
	//Get ghost info:
	void startGhosts(int atIndex) { ghostStart=atIndex; }
	int getGhostStart(void) const {	return ghostStart; }
	int isGhostIndex(int idx) const { return idx>=ghostStart; }
	
	//Symmetry array:
	const FEM_Symmetries_t *getSymmetries(void) const {
		if (sym==NULL) return NULL;
		else return sym->getVec();
	}
	FEM_Symmetries_t getSymmetries(int r) const { 
		if (sym==NULL) return FEM_Symmetries_t(0);
		else return (*sym)[r]; 
	}
	void setSymmetries(int r,FEM_Symmetries_t s);
};

/* Describes one kind of FEM elements */
class FEM_Elem:public FEM_Item {
public:
	typedef AllocTable2d<int> conn_t;
private:
	conn_t conn; //Connectivity data (one row per element)
public:
	FEM_Elem():FEM_Item() {}
	
	void pup(PUP::er &p);
	void print(const char *type,const l2g_t &l2g);
	
	void allocate(int nItems,int dataPer,int nodesPer);
	conn_t &setConn(void) {return conn;}
	const conn_t &getConn(void) const {return conn;}
	int getConn(int elem,int nodeNo) const {return conn(elem,nodeNo);}
	int getNodesPer(void) const {return conn.width();}
	int *connFor(int i) {return conn.getRow(i);}
	const int *connFor(int i) const {return conn.getRow(i);}
	void connIs(int i,const int *src) {conn.setRow(i,src);}
};

/*Describes a set of records of sparse data that are all the
same size and all associated with the same number of nodes.
Sparse data is associated with some subset of the nodes in the mesh,
and gets copied to every chunk that has all those nodes.  The canonical
use of sparse data is to describe boundary conditions.
*/
class FEM_Sparse : public CkNoncopyable {
	AllocTable2d<int> nodes; //Each row is the nodes surrounding a tuple
	AllocTable2d<char> data; //Each row is the user data for a tuple
	intArrayPtr elem; //*OPTIONAL* partitioning based on elements (2*size() ints)
public:
	void allocate(int n_); //Allocate storage for data and nodes of n tuples
	
	FEM_Sparse() { } //Used during pup
	FEM_Sparse(int nodesPer_,int bytesPer_) 
		:nodes(nodesPer_), data(bytesPer_) { }
	
	//Return the number of records:
	inline int size(void) const {return data.size();}
	//Return the size of each record:
	inline int getNodesPer(void) const {return nodes.width();}
	inline int getDataPer(void) const {return data.width();}
	
	//Examine/change a single record:
	inline const int *getNodes(int i) const {return nodes.getRow(i);}
	inline const char *getData(int i) const {return data.getRow(i);}
	inline int *getNodes(int i) {return nodes.getRow(i);}
	inline char *getData(int i) {return data.getRow(i);}
	inline void setNodes(int i,const int *d,int idxBase) {nodes.setRow(i,d,idxBase);}
	inline void setData(int i,const char *d) {data.setRow(i,d);}
	
	//Allocate and set the entire table:
	void set(int records,const int *n,int idxBase,const char *d);
	
	//Get the entire table:
	void get(int *n,int idxBase,char *d) const;
	
	//Set the optional element-partitioning array
	void setElem(int *elem_) {elem=elem_;}
	int *getElem(void) {return elem;}
	const int *getElem(void) const {return elem;}
	
	void pup(PUP::er &p) {p|nodes; p|data;}
};
PUPmarshall(FEM_Sparse);


/*Describes one kind of symmetry condition
*/
class FEM_Sym_Desc : public PUP::able {
public:
	virtual ~FEM_Sym_Desc();

	//Apply this symmetry to this location vector
	virtual CkVector3d applyLoc(const CkVector3d &loc) const =0;
	
	//Apply this symmetry to this relative (vel or acc) vector
	virtual CkVector3d applyVec(const CkVector3d &vec) const =0;
	
	//Make a new copy of this class:
	virtual FEM_Sym_Desc *clone(void) const =0;
	
	//Allows Desc's to be pup'd via | operator:
	friend inline void operator|(PUP::er &p,FEM_Sym_Desc &a) {a.pup(p);}
	friend inline void operator|(PUP::er &p,FEM_Sym_Desc* &a) {
		PUP::able *pa=a;  p(&pa);  a=(FEM_Sym_Desc *)pa;
	}
};

//Describes a linear-periodic (space shift) symmetry:
class FEM_Sym_Linear : public FEM_Sym_Desc {
	CkVector3d shift; //Offset to add to locations
public:
	FEM_Sym_Linear(const CkVector3d &shift_) :shift(shift_) {}
	FEM_Sym_Linear(CkMigrateMessage *m) {}
	
	//Apply this symmetry to this location vector
	CkVector3d applyLoc(const CkVector3d &loc) const {return loc+shift;}
	
	//Apply this symmetry to this relative (vel or acc) vector
	virtual CkVector3d applyVec(const CkVector3d &vec) const {return vec;}
	
	virtual FEM_Sym_Desc *clone(void) const {
		return new FEM_Sym_Linear(shift);
	}
	
	virtual void pup(PUP::er &p);
	PUPable_decl(FEM_Sym_Linear);
};

/*
Describes all the different kinds of symmetries that apply to
this mesh.
*/
class FEM_Sym_List {
	//This lists the different kinds of symmetry
	CkPupAblePtrVec<FEM_Sym_Desc> sym; 
	
	FEM_Sym_List(const FEM_Sym_List &src); //NOT DEFINED: copy constructor
public:
	FEM_Sym_List();
	void operator=(const FEM_Sym_List &src); //Assignment operator
	~FEM_Sym_List();
	
	//Add a new kind of symmetry to this list, returning
	// the way objects with that symmetry should be marked.
	FEM_Symmetries_t add(FEM_Sym_Desc *desc);
	
	//Apply all the listed symmetries to this location
	void applyLoc(CkVector3d *loc,FEM_Symmetries_t sym) const;
	
	//Apply all the listed symmetries to this relative vector
	void applyVec(CkVector3d *vec,FEM_Symmetries_t sym) const;
	
	void pup(PUP::er &p);
};

/*This class describes the nodes and elements in
  a finite-element mesh or submesh*/
class FEM_Mesh : public CkNoncopyable {
	CkPupPtrVec<FEM_Sparse> sparse;
	FEM_Sym_List symList;
public:
	void setSymList(const FEM_Sym_List &src) {symList=src;}
	const FEM_Sym_List &getSymList(void) const {return symList;}

	int nSparse(void) const {return sparse.size();}
	void setSparse(int uniqueID,FEM_Sparse *s);
	FEM_Sparse &setSparse(int uniqueID);
	const FEM_Sparse &getSparse(int uniqueID) const;
	
	FEM_Item node; //Describes the nodes in the mesh
	NumberedVec<FEM_Elem> elem; //Describes the different types of elements in the mesh
	
	//Set up our fields based on this mesh:
	void makeRoom(const FEM_Mesh &src) {
		elem.makeLonger(src.elem.size()-1);
		setSymList(src.getSymList());
	}
	
	//Return this type of element, given an element type
	FEM_Item &setCount(int elTypeOrMinusOne) {
		if (elTypeOrMinusOne==-1) return node;
		else return elem[chkET(elTypeOrMinusOne)];
	}
	const FEM_Item &getCount(int elTypeOrMinusOne) const {
		if (elTypeOrMinusOne==-1) return node;
		else return elem[chkET(elTypeOrMinusOne)];
	}
	FEM_Elem &setElem(int elType) {return elem[chkET(elType)];}
	const FEM_Elem &getElem(int elType) const {return elem[chkET(elType)];}
	int chkET(int elType) const; //Check this element type-- abort if it's bad
	
	FEM_Mesh();
	~FEM_Mesh();
	
	int nElems() const //Return total number of elements (of all types)
	  {return nElems(elem.size());}
	int nElems(int t) const;//Return total number of elements before type t
	int getGlobalElem(int elType,int elNo) const;
	void pup(PUP::er &p); //For migration
	void print(const l2g_t &l2g);//Write human-readable description to CkPrintf
};

//Describes a single chunk of the finite-element mesh
class MeshChunk : public CkNoncopyable {
 public:
	FEM_Mesh m; //The chunk mesh
	FEM_Comm comm; //Shared nodes
	int *elemNums; // Maps local elem#-> global elem#  [m.nElems()]
	int *nodeNums; // Maps local node#-> global node#  [m.node.n]
	int *isPrimary; // Indicates us as owner of node  [m.node.n]
	//These fields are (only) used during an updateMesh
	int updateCount,fromChunk;
	int callMeshUpdated; //if 0, skip meshUpdated call; else pass to mesh_updated
	int doWhat; //If 0, do nothing; if 1, repartition; if 2, resume

	MeshChunk(void);
	~MeshChunk();
	//Allocates elemNums, nodeNums, and isPrimary
        void allocate() {
          elemNums=new int[m.nElems()];
          nodeNums=new int[m.node.size()];
          isPrimary=new int[m.node.size()];
        }
	void pup(PUP::er &p); //For send/recv

	void read(FILE *fp);
	void write(FILE *fp);
};

/* Unmarshall into a heap-allocated copy */
template<class T>
class marshallNewHeapCopy {
	T *cur;
public:
	marshallNewHeapCopy(T *readFrom) :cur(readFrom) {}
	marshallNewHeapCopy(const marshallNewHeapCopy &h) :cur(h.cur) {}
	marshallNewHeapCopy(void) {
		cur=new T;
	}
	
	void pup(PUP::er &p) {
		cur->pup(p);
	}
	operator T *() {return cur;}
};
typedef marshallNewHeapCopy<MeshChunk> marshallMeshChunk;
PUPmarshall(marshallMeshChunk);

#include "fem.decl.h"

#define CHK(p) do{if((p)==0)CkAbort("FEM>Memory Allocation failure.");}while(0)

class FEM_DataMsg : public CMessage_FEM_DataMsg
{
 public:
  int from; //Source's chunk number
  int dtype; //Field ID of data
  int length; //Length in bytes of below array
  int tag; //Sequence number
  void *data;
  double alignPad; //Makes sure this structure is double-aligned
  
  FEM_DataMsg(int t, int f, int d,int l) : 
    from(f), dtype(d), length(l), tag(t) { data = (void*) (this+1); }
  FEM_DataMsg(void) { data = (void*) (this+1); }
  static void *pack(FEM_DataMsg *);
  static FEM_DataMsg *unpack(void *);
  static void *alloc(int, size_t, int*, int);
};

/* Maximum number of fields that can be registered */
#define MAXDT 20

class FEMchunk : public ArrayElement1D
{
public:
// updated_mesh keeps the still-being-assembled next mesh chunk.
// It is created and written by the FEM_Set routines called from driver.
  MeshChunk *updated_mesh;
  int updateCount; //Number of mesh updates

  //The current finite-element mesh
  MeshChunk *cur_mesh;
  
private:
  FEMinit init;

  CProxy_FEMchunk thisproxy;
  TCharm *thread;
  
  DType dtypes[MAXDT];
  int ntypes;

  CmmTable messages; // update messages to be processed
  int updateSeqnum; // sequence number for last update operation

  //Describes the current data we're waiting for:
  typedef enum {INVALID_UPDATE,NODE_UPDATE,GHOST_UPDATE} updateType_t;
  updateType_t updateType;

  const FEM_Comm *updateComm; //Communicator we're blocked on
  int nRecd; //Number of messages received for this update
  void *updateBuf; //User data addr for current update

  void beginUpdate(void *buf,int fid,
		  const FEM_Comm *sendComm,const FEM_Comm *recvComm,updateType_t t);
  void recvUpdate(FEM_DataMsg *);
  void update_node(FEM_DataMsg *);
  void update_ghost(FEM_DataMsg *);
  void waitForUpdate(void);

  void *reductionBuf; //Place to return reduction result

  CkVec<int> listTmp;//List of local items 
  int listCount; //Number of lists received
  bool listSuspended;
  bool finishListExchange(const FEM_Comm &l);

  void initFields(void);
  void setMesh(MeshChunk *msg=0);

 public:

  int tsize;
  int doneCalled;

  FEMchunk(const FEMinit &init);
  FEMchunk(CkMigrateMessage *msg);
  ~FEMchunk();

  void ckJustMigrated(void);

  void run(void);
  void run(marshallMeshChunk &);
  void reductionResult(FEM_DataMsg *);
  void updateMesh(int callMeshUpdated,int doRepartition);
  void meshUpdated(marshallMeshChunk &);
  void meshUpdatedComplete(void) {thread->resume();}

  int new_DT(const DType &d) {
    if(ntypes>=MAXDT) {
      CkAbort("FEM_Create_Field> Too many registered datatypes!");
    }
    dtypes[ntypes] = d;
    ntypes++;
    return ntypes-1;
  }
  
  void update(int fid, void *nodes);
  void updateGhost(int fid, int elemType, void *nodes);
  void recv(FEM_DataMsg *);
  
  void exchangeGhostLists(int elemType,int inLen,const int *inList,int idxbase);
  void recvList(int elemType,int fmChk,int nIdx,const int *idx);
  const CkVec<int> &getList(void) {return listTmp;}
  void emptyList(void) {listTmp.length()=0;}
  
  void reduce_field(int fid, const void *nodes, void *outbuf, int op);
  void reduce(int fid, const void *inbuf, void *outbuf, int op);
  void readField(int fid, void *nodes, const char *fname);
  void print(void);
  FEM_Mesh &getMesh(void) { return cur_mesh->m; }
  int *getNodeNums(void) { return cur_mesh->nodeNums; }
  int *getElemNums(void) { return cur_mesh->elemNums; }
  int getPrimary(int nodeNo) { return cur_mesh->isPrimary[nodeNo]; }

  const FEM_Comm &getComm(void) const {return cur_mesh->comm;}

  void pup(PUP::er &p);
};

/*Partition this mesh's elements into n chunks,
 writing each element's 0-based chunk number to elem2chunk.
*/
void fem_partition(const FEM_Mesh *mesh,int nchunks,int *elem2chunk);

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

/*A way to stream out partitioned chunks of a mesh.
  By streaming, we can send the chunks as they are built,
  dramatically reducing the memory needed by the framework.
*/
class MeshChunkOutput {
 public:
	virtual ~MeshChunkOutput() {} /*<- for whining compilers*/
	//Transfer ownership of this mesh chunk
	virtual void accept(int chunkNo,MeshChunk *msg) =0;
};

/*After partitioning, create a sub-mesh for each chunk's elements,
including communication lists between chunks.
*/
void fem_split(FEM_Mesh *mesh,int nchunks,const int *elem2chunk,
	       const FEM_Ghost &ghosts,MeshChunkOutput *out);

/*The inverse of fem_split: reassemble split chunks into a single mesh*/
FEM_Mesh *fem_assemble(int nchunks,MeshChunk **msgs);


//Make a new[]'d copy of this (len-item) array, changing the index as spec'd
int *CkCopyArray(const int *src,int len,int indexBase);
const FEM_Mesh *FEM_Get_FEM_Mesh(void);
FEM_Ghost &FEM_Set_FEM_Ghost(void);

#endif


