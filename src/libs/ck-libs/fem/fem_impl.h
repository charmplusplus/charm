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

//Utility class (don't instantiate)
template <class T>
class CkVecPupbase : public CkVec<T> {
 protected:
	int pupbase(PUP::er &p) {
		int len=length();
		p(len);
		if (p.isUnpacking()) {
			setSize(len);
			length()=len;
		}
		return len;
	}
};

//A vector of derived types, which must be pupped separately
template <class T>
class CkPupVec : public CkVecPupbase<T> {
 public:
	void pup(PUP::er &p) {
		int len=pupbase(p);
		for (int i=0;i<len;i++)
			p|(*this)[i];
	}
};

//A vector of basic types, which can be pupped as an array
// (more restricted but efficient version of above)
template <class T>
class CkPupBasicVec : public CkVecPupbase<T> {
 public:
	void pup(PUP::er &p) {
		int len=pupbase(p);
		p(getVec(),len);
	}
};

//A vector of heap-allocated objects of type T
template <class T>
class CkPupPtrVec : public CkVecPupbase<T *> {
 public:
	~CkPupPtrVec() {
		for (int i=0;i<size();i++)
			delete (*this)[i];
	}
	void pup(PUP::er &p) {
		int len=pupbase(p);
		for (int i=0;i<len;i++) {
			if (p.isUnpacking()) (*this)[i]=new T;
			(*this)[i]->pup(p);
		}
	}
};


/*This class describes a local-to-global index mapping.
The default is the identity mapping.*/
class l2g_t {
public:
	//Return the global number associated with this local element
	virtual int el(int localNo) const {return localNo;}
	//Return the global number associated with this local node
	virtual int no(int localNo) const {return localNo;}
};

/* List the chunks that share an item */
class commRec {
	int item; //Index of item we describe
public:
	class share {
	public:
		int chk;  //Local number of chunk we're shared with
		int idx; //Our index in the comm. list for that chunk
		share(int x=0) {chk=idx=-1;}
		share(int c,int i) :chk(c), idx(i) {}
		void pup(PUP::er &p) {p(chk); p(idx);}
	};
private:
	CkPupVec<share> shares;
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
PUPmarshall(commRec::share);


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
	void write(FILE *fp) const;
	void read(FILE *fp);
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
	
	void write(FILE *fp) const;
	void read(FILE *fp);
	void print(const l2g_t &l2g);
};

/*This class describes the nodes and elements in
  a finite-element mesh or submesh*/
class FEM_Mesh : public CkNoncopyable {
public:
	class count : public CkNoncopyable {
	public:
		int n;//Number of objects
		int dataPer;//Doubles of user data per object
		double *udata;//User's data-- [dataPer x n]

		int ghostStart; //Index of first ghost object
		commCounts ghostSend; //Non-ghosts we send out
		commCounts ghostRecv; //Ghosts we recv into
		
		count() //Default constructor
		  {n=dataPer=0;ghostStart=0;udata=NULL;}
		~count() 
		  {delete [] udata;udata=NULL;}
		void pup(PUP::er &p);
		void print(const char *type,const l2g_t &l2g);

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

	count node; //Describes the nodes in the mesh
	
#define FEM_MAX_ELEMTYPES 20 
	class elemCount:public count {
	public:
		//There's a separate elemCount for each kind of element
		int nodesPer;//Number of nodes per element
		int *conn;//Connectivity array-- [nodesPer x n]
		
		elemCount():count() {nodesPer=-1;conn=NULL;}
		~elemCount() //Free all stored memory
		  {delete [] conn;conn=NULL;}
		void allocConn(void)
		{delete[] conn; conn=new int[connCount()];}
		void pup(PUP::er &p);
		void print(const char *type,const l2g_t &l2g);
		int *connFor(int i) {return &conn[nodesPer*i];}
		const int *connFor(int i) const {return &conn[nodesPer*i];}
		int connCount() const //Return total entries in connectivity array
		  {return n*nodesPer;}
	};
	int nElemTypes;//Length of array below
	elemCount elem[FEM_MAX_ELEMTYPES];

	count &getCount(int elTypeOrMinusOne);
	
	FEM_Mesh();
	~FEM_Mesh();
	void copyType(const FEM_Mesh &from);//Copies nElemTypes and *Per fields
	
	int nElems() const //Return total number of elements (of all types)
	  {return nElems(nElemTypes);}
	int nElems(int t) const;//Return total number of elements before type t
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

	void read(FILE *fp) 
		{readNodes(fp); readElems(fp); readComm(fp); }
	void write(FILE *fp) const 
		{writeNodes(fp); writeElems(fp); writeComm(fp); }
private:
	void readNodes(FILE *fp);
	void readElems(FILE *fp);
	void readComm(FILE *fp);
	void writeNodes(FILE *fp) const;
	void writeElems(FILE *fp) const;
	void writeComm(FILE *fp) const;
};

/* Unmarshall into a heap-allocated copy */
template<class T>
class marshallNewHeapCopy : public CkNoncopyable {
	T *cur;
public:
	marshallNewHeapCopy(T *readFrom) :cur(readFrom) {}
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
#include "fem.h"

#define CHK(p) do{if((p)==0)CkAbort("FEM>Memory Allocation failure.");}while(0)

// temporary Datatype representation
// will go away once MPI user-defined datatypes are ready
struct DType {
  int base_type;
  int vec_len;
  int init_offset; // offset of field in bytes from the beginning of data
  int distance; // distance in bytes between successive field values
  DType(void) {}
  DType(const DType& dt) : 
    base_type(dt.base_type), vec_len(dt.vec_len), init_offset(dt.init_offset),
    distance(dt.distance) {}
  void operator=(const DType& dt) {
    base_type = dt.base_type; 
    vec_len = dt.vec_len; 
    init_offset = dt.init_offset;
    distance = dt.distance;
  }
  DType( const int b,  const int v=1,  const int i=0,  const int d=0) : 
    base_type(b), vec_len(v), init_offset(i) {
    distance = (d ? d : length());
  }
  int length(const int nitems=1) const {
    int blen;
    switch(base_type) {
      case FEM_BYTE : blen = 1; break;
      case FEM_INT : blen = sizeof(int); break;
      case FEM_REAL : blen = sizeof(float); break;
      case FEM_DOUBLE : blen = sizeof(double); break;
    }
    return blen * vec_len * nitems;
  }
};

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

#define MAXDT 20
#define FEM_MAXUDATA 20

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

  int new_DT(int base_type, int vec_len=1, int init_offset=0, int distance=0) {
    if(ntypes==MAXDT) {
      CkAbort("FEM: registered datatypes limit exceeded.");
    }
    dtypes[ntypes] = DType(base_type, vec_len, init_offset, distance);
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
class ghostLayer {
public:
	int nodesPerTuple; //Number of shared nodes needed to connect elements
	bool addNodes; //Add ghost nodes to the chunks
	class elemGhostInfo {
	public:
		bool add; //Add this kind of ghost element to the chunks
		int tuplesPerElem; //# of tuples surrounding this element
		const int *elem2tuple; //The tuples around this element [nodesPerTuple * tuplesPerElem]
		elemGhostInfo(void) {add=false;}
	};
	elemGhostInfo elem[FEM_MAX_ELEMTYPES];
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


