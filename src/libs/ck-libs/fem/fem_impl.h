/*Charm++ Finite Element Framework:
C++ implementation file

This is the under-the-hood implementation file for FEM.
Orion Sky Lawlor, olawlor@acm.org, 9/28/00
*/
#ifndef _FEM_IMPL_H
#define _FEM_IMPL_H

#include <stdio.h>

class collision{ }; //<- needed by parCollide.decl.h
#include "fem.decl.h"
#include "ampiimpl.h"
#include "fem.h"

extern CkChareID _mainhandle;
extern CkArrayID _femaid;
extern int _nchunks;

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

class DataMsg : public CMessage_DataMsg
{
 public:
  int from;
  int dtype;
  void *data;
  int tag;
  DataMsg(int t, int f, int d) : 
    from(f), dtype(d), tag(t) { data = (void*) (this+1); }
  DataMsg(void) { data = (void*) (this+1); }
  static void *pack(DataMsg *);
  static DataMsg *unpack(void *);
  static void *alloc(int, size_t, int*, int);
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

/*This class describes the nodes and elements in
  a finite-element mesh or submesh*/
class FEM_Mesh {
public:
	class count {
	public:
		int n;//Number of objects
		int dataPer;//Doubles of user data per object
		double *udata;//User's data-- [dataPer x n]
		
		count() //Default constructor
		  {n=dataPer=0;udata=NULL;}
		void deallocate(void) //Free all stored memory
		  {delete [] udata;udata=NULL;}
		void pup(PUP::er &p);
		int size() const //Return total array storage size, in bytes
		  {return sizeof(double)*udataCount();}
		void print(const char *type,const l2g_t &l2g);
		
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
		void allocConn(void)
		{delete[] conn; conn=new int[connCount()];}
		void deallocate(void) //Free all stored memory
		  {count::deallocate();delete [] conn;conn=NULL;}
		void pup(PUP::er &p);
		int size() const //Return total array storage size, in bytes
		  {return count::size()+sizeof(int)*connCount();}
		void print(const char *type,const l2g_t &l2g);
		int *connFor(int i) {return &conn[nodesPer*i];}
		const int *connFor(int i) const {return &conn[nodesPer*i];}
		int connCount() const //Return total entries in connectivity array
		  {return n*nodesPer;}
	};
	int nElemTypes;//Length of array below
	elemCount elem[FEM_MAX_ELEMTYPES];
	
	FEM_Mesh() //Default constructor
	  {nElemTypes=0;}
	void copyType(const FEM_Mesh &from);//Copies nElemTypes and *Per fields
	
	int nElems() const //Return total number of elements (of all types)
	  {return nElems(nElemTypes);}
	int nElems(int t) const;//Return total number of elements before type t
	int size() const; //Return total array storage size, in bytes
	void deallocate(void); //Free all stored memory
	void pup(PUP::er &p); //For migration
	void print(const l2g_t &l2g);//Write human-readable description to CkPrintf
};

/*This class describes the shared nodes 
  of a given processor*/
class commCounts {
public:
	int nPes;//Number of processors we share nodes with
	int *peNums; // Maps local pe->global pe #      [nPes]
	int *numNodesPerPe; // Maps local pe -> # shared nodes [nPes]
	int **nodesPerPe; // Lists shared nodes for each pe [nPes][nodesPerPe[i]]
	
	commCounts() {nPes=0;peNums=numNodesPerPe=NULL;nodesPerPe=NULL;}
	int sharedNodes() const;//Return total number of shared nodes
	void allocate(void); //Allocate arrays based on nPes
	void deallocate(void); //Free all stored memory
	void print(const l2g_t &l2g);//Write human-readable description
	void pup(PUP::er &p); //For migration
	int size() const; //Return total array storage size, in bytes
};

//Describes a local chunk of a mesh
class ChunkMsg : public CMessage_ChunkMsg {
 public:
        int isPacked;//Is this message one contiguous block?
	FEM_Mesh m; //The chunk mesh
	commCounts comm; //Shared nodes
	int *elemNums; // Maps local elem#-> global elem#  [m.nElems()]
	int *nodeNums; // Maps local node#-> global node#  [m.node.n]
	int *isPrimary; // Indicates us as owner of node  [m.node.n]
	//These fields are (only) used during an updateMesh
	int updateCount,fromChunk;
	int callMeshUpdated,doRepartition;

        ChunkMsg(void) { isPacked=0; }
        ~ChunkMsg() { deallocate(); }
	void deallocate(void); //Free all stored memory

	void pup(PUP::er &p); //For send/recv
	int size() const; //Return total storage size, in bytes
	static void *pack(ChunkMsg *);
	static ChunkMsg *unpack(void *);
};

#define MAXDT 20
#define FEM_MAXUDATA 20

class chunk : public ampi
{
//Stored_mesh keeps the initial mesh passed to run(), if any
//  If this is non-NULL, all the mesh fields below point into it.
//  Otherwise, the mesh fields are heap-allocated.
  ChunkMsg *stored_mesh;
public:
// updated_mesh keeps the still-being-assembled next mesh chunk.
// It is created and written by the FEM_Set routines called from driver.
  ChunkMsg *updated_mesh;
  int updateCount;

  FEM_Mesh m; //The current chunk mesh
private:
  commCounts comm; //Shared nodes
  int *elemNums; // Maps local elem#-> global elem#  [m.nElems()]
  int *nodeNums; // Maps local node#-> global node#  [m.node.n]
  int *isPrimary; // Indicates us as owner of node  [m.node.n]
  int *gPeToIdx; // Maps global PE -> local PE [total Chunks]

  void deallocate(void); //Delete storage in above arrays
  
  DType dtypes[MAXDT];
  int ntypes;

  CmmTable messages; // messages to be processed
  int wait_for; // which tag is tid waiting for ? 0 if not waiting

  int seqnum; // sequence number for update operation
  int nRecd; // number of messages received for this seqnum
  void *curbuf; // data addr for current update operation

  int nudata;
  void *userdata[FEM_MAXUDATA];
  FEM_PupFn pup_ud[FEM_MAXUDATA];
 public:

  int tsize;
  int doneCalled;

  chunk(void);
  chunk(CkMigrateMessage *msg): ampi(msg) {stored_mesh=NULL;}
  ~chunk();
  
  void serialSwitch(ChunkMsg *);
  
  void run(void);
  void run(ChunkMsg*);
  void recv(DataMsg *);
  void reductionResult(DataMsg *);
  void updateMesh(int callMeshUpdated,int doRepartition);
  void meshUpdated(ChunkMsg *);

  int new_DT(int base_type, int vec_len=1, int init_offset=0, int distance=0) {
    if(ntypes==MAXDT) {
      CkAbort("FEM: registered datatypes limit exceeded.");
    }
    dtypes[ntypes] = DType(base_type, vec_len, init_offset, distance);
    ntypes++;
    return ntypes-1;
  }
  void update(int fid, void *nodes);
  void reduce_field(int fid, const void *nodes, void *outbuf, int op);
  void reduce(int fid, const void *inbuf, void *outbuf, int op);
  void readField(int fid, void *nodes, const char *fname);
  int id(void) { return thisIndex; }
  int total(void) { return numElements; }
  void print(void);
  int *get_nodenums(void) { return nodeNums; }
  int *get_elemnums(void) { return elemNums; }

  int register_userdata(void *_userdata,FEM_PupFn _pup_ud)
  {
    if(nudata==FEM_MAXUDATA)
      CkAbort("FEM> UserData registration limit exceeded.!\n");
    userdata[nudata] = _userdata;
    pup_ud[nudata] = _pup_ud;
    return nudata++;
  }
  int check_userdata(int n);
  void *get_userdata(int n) 
    { return userdata[check_userdata(n)]; }
    
  const commCounts &getComm(void) const {return comm;}

  void pup(PUP::er &p);
  void readyToMigrate(void)
  {
    // CkPrintf("[%d] going to sync\n", thisIndex);
    AtSync();
    thread_suspend();
  }
  void ResumeFromSync(void)
  {
    // CkPrintf("[%d] returned from sync\n", thisIndex);
    thread_resume();
  }

 private:
  CthThread tid; // waiting thread, 0 if no one is waiting
  void thread_suspend(void); //Thread will block until resume
  void thread_resume(void);  //Start thread running again
  void start_running(void)
  {
    ampi::prepareCtv();
    thisArray->the_lbdb->ObjectStart(ldHandle);
  }
  void stop_running(void)
  {
    thisArray->the_lbdb->ObjectStop(ldHandle);
  }

  FILE *fp;
  void update_field(DataMsg *);
  void send(int fid,const void *nodes);
  void readNodes();
  void readElems();
  void readComm();
  void readChunk(ChunkMsg *msg=0);
  void callDriver(void);
};



class main : public Chare
{
  ChunkMsg **cmsgs;//Array of _nchunks chunk messages
  int updateCount;
  CkQ<ChunkMsg *> futureUpdates;
  CkQ<ChunkMsg *> curUpdates;
  int numdone;
 public:
  main(CkArgMsg *);

  void updateMesh(ChunkMsg *);

  void done(void);
};

/*Partition this mesh's elements into n chunks,
 writing each element's 0-based chunk number to elem2chunk.
*/
void fem_partition(const FEM_Mesh *mesh,int nchunks,int *elem2chunk);

/*After partitioning, create a sub-mesh for each chunk's elements,
including communication lists between chunks.
*/
void fem_map(const FEM_Mesh *mesh,int nchunks,int *elem2chunk,ChunkMsg **msgs);

/*The inverse of fem_map: reassemble split chunks into a single mesh*/
FEM_Mesh *fem_assemble(int nchunks,ChunkMsg **msgs);

/*Decide how to declare C functions that are called from Fortran--
  some fortran compiles expect all caps; some all lowercase, 
  but with a trailing underscore.*/
#if CMK_FORTRAN_USES_ALLCAPS
# define FTN_NAME(caps,nocaps) caps  /*Declare name in all caps*/
#else
# if CMK_FORTRAN_USES_x__
#  define FTN_NAME(caps,nocaps) nocaps##_ /*No caps, extra underscore*/
# else
#  define FTN_NAME(caps,nocaps) nocaps /*Declare name without caps*/
# endif /*__*/
#endif /*ALLCAPS*/

#define CDECL extern "C" /*Function declaration for C linking*/
#define FDECL extern "C" /*Function declaration for Fortran linking*/

FDECL void FTN_NAME(INIT,init_) (void);
FDECL void FTN_NAME(DRIVER,driver_) (void);
FDECL void FTN_NAME(FINALIZE,finalize_) (void);
FDECL void FTN_NAME(MESH_UPDATED,mesh_updated_) (int *param);

#endif


