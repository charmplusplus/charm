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
	
	//Same old bracket operators, but return the actual object, not a pointer:
	T &operator[](int i) {return *( (*(super *)this)[i] );}
	const T &operator[](int i) const {return *( (*(const super *)this)[i] );}
};


/*Inner class used by commRec:*/
class commShare {
 public:
  int chk;  //Local number of chunk we're shared with
  int idx; //Our index in the comm. list for that chunk
  commShare(int x=0) {chk=idx=-1;}
  commShare(int c,int i) :chk(c), idx(i) {}
  void pup(PUP::er &p) {p(chk); p(idx);}
};
PUPmarshall(commShare);

/* List the chunks that share an item */
class commRec {
	int item; //Index of item we describe
	CkPupVec<commShare> shares;
public:
	commRec(int item_=-1);
	~commRec();

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

/* Map an item to its commRec (if any) */
class commMap {
	CkHashtableT<CkHashtableAdaptorT<int>,commRec *> map;
public:
	commMap();
	void pup(PUP::er &p);
	~commMap();

	//Add a comm. entry for this item
	void add(int item,int chk,int idx);
	
	//Look up this item's commRec.  Returns NULL if item is not shared.
	const commRec *get(int item);
};

/* Lists the items we share with one other chunk */
class commList {
	int pe; //Global number of other chunk
	int us; //Other chunk's local number for our chunk	
	CkPupBasicVec<int> shared; //Local indices of shared items
public:
	commList();
	commList(int otherPe,int myPe);
	~commList();
	int getDest(void) const {return pe;}
	int getOurName(void) const {return us;}
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

/*This class describes all the shared items of a given chunk*/
class commCounts : public CkNoncopyable {
	commMap map;
	CkPupPtrVec<commList> comm;
	commList *getList(int forChunk,int *hisNum);
public:
	commCounts(void);
	~commCounts();
	int totalShared() const;//Return total number of shared nodes
	void pup(PUP::er &p); //For migration
	
	int size(void) const {return comm.size();}
	const commList &operator[](int idx) const {return *comm[idx];}
	
	//Look up an item's commRec
	const commRec *getRec(int item) {return map.get(item);}
	//This item is shared with the given (local) chunk
	void addNode(int localNo,int sharedWithChk) {
		map.add(localNo,sharedWithChk,
			comm[sharedWithChk]->push_back(localNo));
	}
	
	//Used in creating comm. lists:
	//Add this local number to both lists:
	void add(int myChunk,int myLocalNo,
		 int hisChunk,int hisLocalNo,commCounts &hisList);
	
	void print(const l2g_t &l2g);
};

/* Describes an FEM item-- a set of nodes or elements */
class FEM_Item : public CkNoncopyable {
public:
	int n;//Number of objects
	int dataPer;//Doubles of user data per object
	double *udata;//User's data-- [dataPer x n]

	int ghostStart; //Index of first ghost object
	commCounts ghostSend; //Non-ghosts we send out
	commCounts ghostRecv; //Ghosts we recv into
	
	FEM_Item() //Default constructor
	  {n=dataPer=0;ghostStart=0;udata=NULL;}
	~FEM_Item() 
	  {delete [] udata;udata=NULL;}
	void pup(PUP::er &p);
	void print(const char *type,const l2g_t &l2g);
	
	int size(void) const {return n;}
	int getDataPer(void) const {return dataPer;}

	int isGhostIndex(int idx) const {
		return idx>=ghostStart;
	}
		
	void setUdata_r(const double *Nudata);
	void setUdata_c(const double *NudataTranspose);
	void getUdata_r(double *Nudata) const;
	void getUdata_c(double *NudataTranspose) const;
	void udataIs(int i,const double *src)
	  {memcpy((void *)&udata[dataPer*i],(const void *)src,sizeof(double)*dataPer);}
	const double *udataFor(int i) const {return &udata[dataPer*i];}
	void allocUdata(void) //Create a new udata array
	  {delete[] udata;udata=new double[udataCount()];}
	int udataCount() const //Return total entries in user data array 
	  {return n*dataPer;}
};

/* Describes one kind of FEM elements */
class FEM_Elem:public FEM_Item {
public:
	//There's a separate elemCount for each kind of element
	int nodesPer;//Number of nodes per element
	int *conn;//Connectivity array-- [nodesPer x n]
	
	FEM_Elem():FEM_Item() {nodesPer=-1;conn=NULL;}
	~FEM_Elem() //Free all stored memory
	  {delete [] conn;conn=NULL;}
	
	int getNodesPer(void) const {return nodesPer;}
	
	void allocConn(void)
	{delete[] conn; conn=new int[connCount()];}
	void pup(PUP::er &p);
	void print(const char *type,const l2g_t &l2g);
	int *connFor(int i) {return &conn[nodesPer*i];}
	const int *connFor(int i) const {return &conn[nodesPer*i];}
	int connCount() const //Return total entries in connectivity array
	  {return n*nodesPer;}
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
		:cols(cols_), rows(rows_), table(src) {}
	
	//"size" of the table is the number of rows:
	inline int size(void) const {return rows;}
	//Width of the table is the number of columns:
	inline int width(void) const {return cols;}
	
	//Get a pointer to a row of the table:
	inline T *getRow(int r) {return &table[r*cols];}
	inline const T *getRow(int r) const {return &table[r*cols];}
	inline T *operator[](int r) {return getRow(r);}
	inline const T *operator[](int r) const {return getRow(r);}
	void setRow(int r,const T *src) {
		T *dest=getRow(r);
		for (int i=0;i<cols;i++) dest[i]=src[i];
	}
	//As above, but with an index shift
	void setRow(int r,const T *src,T idxBase) {
		T *dest=getRow(r);
		for (int i=0;i<cols;i++) dest[i]=src[i]-idxBase;
	}
};

//As above, but heap-allocatable
template <class T>
class AllocTable2d : public BasicTable2d<T> {
	T *allocTable; //Heap-allocated table 
public:
	AllocTable2d(int cols_=0,int rows_=0) 
		:BasicTable2d<T>(NULL,cols_,rows_)
	{
		allocTable=NULL;
		if (rows>0) allocate(rows);
	}
	~AllocTable2d() {delete[] allocTable;}
	
	void allocate(int rows_) { //Make room for n rows
		if (allocTable!=NULL) delete[] allocTable;
		rows=rows_;
		allocTable=table=new T[rows*cols];
	}
	
	//Pup routine and operator|:
	void pup(PUP::er &p) {
		p|rows; p|cols;
		if (table==NULL) allocate(rows);
		p(table,rows*cols); //T better be a basic type, or this won't compile!
	}
	friend void operator|(PUP::er &p,AllocTable2d<T> &t) {t.pup(p);}
};


//Smart pointer-to-new[]'d array-of-ints
class intArrayPtr : public CkNoncopyable {
	int *sto;
public:
	intArrayPtr() {sto=NULL;}
	intArrayPtr(int *src) {sto=src;}
	~intArrayPtr() {if (sto) delete[] sto;}
	void operator=(int *src) {
		if (sto) delete[] sto;
		sto=src;
	}
	operator int *(void) {return sto;}
	operator const int *(void) const {return sto;}
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

/*This class describes the nodes and elements in
  a finite-element mesh or submesh*/
class FEM_Mesh : public CkNoncopyable {
	CkPupPtrVec<FEM_Sparse> sparse;
public:
	int nSparse(void) const {return sparse.size();}
	void setSparse(int uniqueID,FEM_Sparse *s);
	FEM_Sparse &setSparse(int uniqueID);
	const FEM_Sparse &getSparse(int uniqueID) const;
	
	FEM_Item node; //Describes the nodes in the mesh
	NumberedVec<FEM_Elem> elem; //Describes the different types of elements in the mesh
	
	//Return this type of element, given an element type
	FEM_Item &getCount(int elTypeOrMinusOne) {
		if (elTypeOrMinusOne==-1) return node;
		else return elem[chkET(elTypeOrMinusOne)];
	}
	const FEM_Item &getCount(int elTypeOrMinusOne) const {
		if (elTypeOrMinusOne==-1) return node;
		else return elem[chkET(elTypeOrMinusOne)];
	}
	FEM_Elem &getElem(int elType) {return elem[chkET(elType)];}
	const FEM_Elem &getElem(int elType) const {return elem[chkET(elType)];}
	int chkET(int elType) const; //Check this element type-- abort if it's bad
	
	FEM_Mesh();
	~FEM_Mesh();
	void copyType(const FEM_Mesh &from);//Copies nElemTypes and *Per fields
	
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
	commCounts comm; //Shared nodes
	int *elemNums; // Maps local elem#-> global elem#  [m.nElems()]
	int *nodeNums; // Maps local node#-> global node#  [m.node.n]
	int *isPrimary; // Indicates us as owner of node  [m.node.n]
	//These fields are (only) used during an updateMesh
	int updateCount,fromChunk;
	int callMeshUpdated,doRepartition;

	MeshChunk(void);
	~MeshChunk();
	//Allocates elemNums, nodeNums, and isPrimary
        void allocate() {
          elemNums=new int[m.nElems()];
          nodeNums=new int[m.node.n];
          isPrimary=new int[m.node.n];
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
  int from; //Source's local chunk number on dest. chunk
  int dtype; //Field ID of data
  int length; //Length in bytes of below array
  void *data;
  int tag; //Sequence number
  FEM_DataMsg(int t, int f, int d,int l) : 
    from(f), dtype(d), tag(t), length(l) { data = (void*) (this+1); }
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

  commCounts *updateComm; //Communicator we're blocked on
  int nRecd; //Number of messages received for this update
  void *updateBuf; //User data addr for current update

  void beginUpdate(void *buf,int fid,
		   commCounts *sendComm,commCounts *recvComm,updateType_t t);
  void recvUpdate(FEM_DataMsg *);
  void update_node(FEM_DataMsg *);
  void update_ghost(FEM_DataMsg *);
  void waitForUpdate(void);

  void *reductionBuf; //Place to return reduction result

  CkVec<int> listTmp;//List of local items 
  int listCount; //Number of lists received
  bool listSuspended;
  bool finishListExchange(const commCounts &l);

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

  const commCounts &getComm(void) const {return cur_mesh->comm;}

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

//Declare this at the start of every API routine:
#define FEMAPI(routineName) TCHARM_API_TRACE(routineName,"fem")

/*A way to stream out partitioned chunks of a mesh.
  By streaming, we can send the chunks as they are built,
  dramatically reducing the memory needed by the framework.
*/
class MeshChunkOutput {
 public:
	//Transfer ownership of this mesh chunk
	virtual void accept(int chunkNo,MeshChunk *msg) =0;
};

/*After partitioning, create a sub-mesh for each chunk's elements,
including communication lists between chunks.
*/
void fem_split(const FEM_Mesh *mesh,int nchunks,int *elem2chunk,
	       int nGhostLayers,const ghostLayer *g,MeshChunkOutput *out);

/*The inverse of fem_split: reassemble split chunks into a single mesh*/
FEM_Mesh *fem_assemble(int nchunks,MeshChunk **msgs);

#endif


