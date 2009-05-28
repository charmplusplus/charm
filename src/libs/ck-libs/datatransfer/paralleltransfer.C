/**
 * Conservative, accurate parallel cell-centered;
 * and interpolation-based node-centered data transfer.
 * Orion Sky Lawlor, olawlor@acm.org, 2003/3/24
 */
#include "tetmesh.h"
#include "transfer.h"
#include "bbox.h"
#include "paralleltransfer.h"
#include "charm++.h" /* for CmiAbort */
#include "GenericElement.h"

#define OSL_com_debug 0
#define COORD_PER_POINT 3 /* coordinates per point (e.g., 3d) */
#define POINT_PER_TET 4 /* points per element (e.g., 4 for tets) */

class progress_t {
public:
  virtual ~progress_t();
  virtual void p(const char *where) {}
};

/** Provides access to a local element. */
class ConcreteLocalElement : public ConcreteElementNodeData {
  const TetMesh &mesh;
  const int *conn;
  const xfer_t *ptVals; // contains valsPerPt values for each point
  int valsPerPt;
public:
  ConcreteLocalElement(const TetMesh &mesh_,const xfer_t *ptVals_,int valsPerPt_)
    :mesh(mesh_), conn(0), ptVals(ptVals_), valsPerPt(valsPerPt_) { }
  void set(int tet) {
    conn=mesh.getTet(tet);
  }
  
  /** Return the location of the i'th node of this element. */
  virtual CPoint getNodeLocation(int i) const {
    return mesh.getPoint(conn[i]);
  }
  
  /** Return the vector of data associated with our i'th node. */
  virtual const double *getNodeData(int i) const {
    return &ptVals[conn[i]*valsPerPt];
  }
};


class parallelTransfer_c {
  collide_t voxels;
  int firstDest; //Numbers greater than this are destination tets
  /// Return true if this is the collision-box number of a destination tet.
  /// Only works for local numbers.
  inline bool isDest(int n) {return n>=firstDest;}
  
  MPI_Comm mpi_comm;
  int myRank,commSize;
  /// Return true if this collision record describes a local intersection
  inline bool isLocal(const int *coll) {return coll[1]==myRank;}
  
  int valsPerTet, valsPerPt;
  const TetMesh &srcMesh;
  const xfer_t *srcTet; // srcMesh.getTets()*valsPerTet source values
  const xfer_t *srcPt; // srcMesh.getPts()*valsPerTet source values
  const TetMesh &destMesh;
  xfer_t *destTet; // destMesh.getTets()*valsPerTet partial values
  xfer_t *destPt; // destMesh.getPts()*valsPerPt values
  double *destVolumes; // destMesh.getTets() partial volumes
  
  ConcreteLocalElement theLocalElement;
  
  /// A source cell, with values sVals, overlaps with this dest cell
  ///  with this much shared volume.  Transfer cell-centered values.
  void accumulateCellValues(const xfer_t *sCellVals,int dest,double sharedVolume) {
    for (int v=0;v<valsPerTet;v++) {
      destTet[dest*valsPerTet+v]+=sharedVolume*sCellVals[v];
    }
    destVolumes[dest]+=sharedVolume;
  }
  
  /// This source element, with values sPt, overlaps with this dest element.
  ///   Transfer any possible node-centered values.
  void transferNodeValues(const ConcreteElementNodeData &srcElement, int dest)
  {
    GenericElement el(POINT_PER_TET);
    for (int dni=0;dni<POINT_PER_TET;dni++) {
      int dn=destMesh.getTet(dest)[dni];
      CkVector3d dnLoc=destMesh.getPoint(dn);
      CVector natc;
      if (el.element_contains_point(dnLoc,srcElement,natc)) 
	{ /* this source element overlaps our destination */
	  el.interpolate_natural(valsPerPt,srcElement,natc,&destPt[dn*valsPerPt]);
	}
    }
  }
public:
  parallelTransfer_c(collide_t voxels_,MPI_Comm mpi_comm_,
		     int valsPerTet_,int valsPerPt_,
		     const xfer_t *srcTet_,const xfer_t *srcPt_,const TetMesh &srcMesh_,
		     xfer_t *destTet_,xfer_t *destPt_,const TetMesh &destMesh_)
    :voxels(voxels_), mpi_comm(mpi_comm_), 
     valsPerTet(valsPerTet_), valsPerPt(valsPerPt_),
     srcMesh(srcMesh_),srcTet(srcTet_),srcPt(srcPt_),
     destMesh(destMesh_),destTet(destTet_),destPt(destPt_),
     theLocalElement(srcMesh, srcPt,valsPerPt)
  {
    MPI_Comm_rank(mpi_comm,&myRank);
    MPI_Comm_size(mpi_comm,&commSize);
    destVolumes=new double[destMesh.getTets()];
    for (int d=0;d<destMesh.getTets();d++) {
      destVolumes[d]=0;
      for (int v=0;v<valsPerTet;v++) {
	destTet[d*valsPerTet+v]=(xfer_t)0;
      }
    }
  }
  ~parallelTransfer_c() {
    delete[] destVolumes;
  }
  
  /** Perform a parallel data transfer from srcVals to destVals */
  void transfer(progress_t &progress);
};

static bbox3d getBox(int t,const TetMesh &mesh) {
  bbox3d ret; ret.empty();
  const int *conn=mesh.getTet(t);
  for (int i=0;i<4;i++)
    ret.add(mesh.getPoint(conn[i]));
  return ret;
}

// Return the number of xfer_t's (usual doubles) this many bytes corresponds to
inline int bytesToXfer(int nBytes) {
  return (nBytes+sizeof(xfer_t)-1)/sizeof(xfer_t);
}

// Copy n values from src to dest
template <class D,class S>
inline void copy(D *dest,const S *src,int n) {
  for (int i=0;i<n;i++) dest[i]=(D)src[i];
}

// Describes the amounts of user data associated with each entity in the mesh
class meshState {
public:
  int tetVal, ptVal; // Number of user data doubles per tet & pt
  meshState(int t,int p) :tetVal(t), ptVal(p) {}
};

/** Describes the outgoing mesh */
class sendState : public meshState {
public:
  const TetMesh &mesh; // Global source mesh
  const xfer_t *tetVals; // User data for each tet (tetVal * mesh->getTets())
  const xfer_t *ptVals;  // User data for each point (ptVal * mesh->getPoints())
  
  const xfer_t *tetData(int t) const {return &tetVals[t*tetVal];}
  const xfer_t * ptData(int p) const {return & ptVals[p* ptVal];}
  
  sendState(const TetMesh &m,int tv,int pv,const xfer_t *t,const xfer_t *p)
    :meshState(tv,pv), mesh(m), tetVals(t), ptVals(p) {}
};

// Keeps track of which elements are already in the message, and which aren't.
class entityPackList {
  int *mark; // Indexed by global number: -1 if not local, else local number
  int *locals; // Indexed by local number: gives global number
  void allocate(int nGlobal) {
    mark=new int[nGlobal];
    for (int i=0;i<nGlobal;i++) mark[i]=-1;
    locals=new int[nGlobal];
  }
public:
  int n; // Number of local entities (numbered 0..n-1)
  
  entityPackList() {n=0; mark=0; locals=0;}
  ~entityPackList() {if (mark) {delete[] mark;delete[] locals;}}
  
  /// Return true if we should add this global number,
  ///   false if it's already there.
  bool add(int global,int nGlobal) {
    if (mark==NULL) allocate(nGlobal);
    if (mark[global]==-1) {
      mark[global]=n;
      locals[n]=global;
      n++;
      return true;
    }
    return false;
  }
  /// Return the local number of this global entity:
  int getLocal(int global) const {return mark[global];}
  /// Return the global number of this local entity:
  int getGlobal(int local) const {return locals[local];}
};

/** Manages the on-the-wire mesh format when sending or receiving 
 tet mesh chunks.

 The on-the-wire mesh format looks like this:
   - list of points 0..nPoints-1
        x,y,z for point; ptVal values per point
   - list of tets 0..nTets-1
        p1,p2,p3,p4 point indices; tetVal values per tet
   - list of tet indices, in order they're shared in collision records
*/
class tetMeshChunk {
  xfer_t *buf; // Message buffer
  
  /* On-the-wire message header */
  struct header_t {
    int nXfer; // Total length of message, in xfer_t's
    int nSend; // Length of sendTets array, in ints
    int nTet; // Number of rows of tetData, 
    int nPt; // Number of rows of ptData
  };
  header_t *header; // Header of buffer
  inline int headerSize() const // size in doubles
  {return bytesToXfer(sizeof(header_t));}
  
  int *sendTets; // List of local numbers of tets, matching collision data
  inline int sendTetsSize(const meshState &s,int nSend) const
  {return bytesToXfer(nSend*sizeof(int));}
  
  xfer_t *tetData; // Tet connectivity and user data (local numbers)
  inline int tetDataRecordSize(const meshState &s) const
  {return bytesToXfer(POINT_PER_TET*sizeof(int))+s.tetVal;}
  inline int tetDataSize(const meshState &s,int nTet) const
  {return nTet*tetDataRecordSize(s); }
  
  xfer_t *ptData; // Point location and user data (local numbers)
  inline int ptDataRecordSize(const meshState &s) const
  {return COORD_PER_POINT+s.ptVal;}
  inline int ptDataSize(const meshState &s,int nPt) const
  {return nPt*ptDataRecordSize(s); }
public:
  tetMeshChunk() {buf=NULL;}
  ~tetMeshChunk() { if (buf) delete[] buf;}
  
  // Used by send side:
  /// Allocate a new outgoing message with this size: 
  void allocate(const meshState &s,int nSend,int nTet,int nPt) {
    int msgLen=headerSize()+sendTetsSize(s,nSend)+tetDataSize(s,nTet)+ptDataSize(s,nPt);
    buf=new xfer_t[msgLen];
    header=(header_t *)buf;
    header->nXfer=msgLen;
    header->nSend=nSend;
    header->nTet=nTet;
    header->nPt=nPt;
    setupPointers(s,msgLen);
  }
  
  /// Return the number of doubles in this message:
  int messageSizeXfer(void) const {return header->nXfer;}
  double *messageBuf(void) const {return buf;}
  
  // Used by receive side:
  /// Set up our pointers into this message, which
  ///  is then owned by and will be deleted by this object.
  void receive(const meshState &s,xfer_t *buf_,int msgLen) {
    buf=buf_; 
    setupPointers(s,msgLen);
  }
  
  // Used by both sides:
  /// Return the list of tets that are being sent:
  inline int *getSendTets(void) {return sendTets;}
  inline int nSendTets(void) const {return header->nSend;}
  
  /// Return the number of tets included in this message:
  inline int nTets(void) const {return header->nTet;}
  /// Return the connectivity array associated with this tet:
  inline int *getTetConn(const meshState &s,int t) {
    return (int *)&tetData[t*tetDataRecordSize(s)];
  }
  /// Return the user data associated with this tet:
  inline xfer_t *getTetData(const meshState &s,int t) {
    return &tetData[t*tetDataRecordSize(s)+bytesToXfer(POINT_PER_TET*sizeof(int))];
  }
  
  /// Return the number of points included in this message:
  inline int nPts(void) const {return header->nPt;}
  /// Return the coordinate data for this point:
  inline xfer_t *getPtLoc(const meshState &s,int n) {
    return &ptData[n*ptDataRecordSize(s)];
  }
  /// Return the user data for this point:
  inline xfer_t *getPtData(const meshState &s,int n) {
    return &ptData[n*ptDataRecordSize(s)+COORD_PER_POINT];
  }
  
private:
  // Point sendTets and the data arrays into message buffer:
  void setupPointers(const meshState &s,int msgLen) {
    xfer_t *b=buf; 
    header=(header_t *)b; b+=headerSize();
    sendTets=(int *)b; b+=sendTetsSize(s,header->nSend);
    tetData=b; b+=tetDataSize(s,header->nTet);
    ptData=b; b+=ptDataSize(s,header->nPt);
    int nRead=b-buf;
    if (nRead!=header->nXfer || nRead!=msgLen)
      CkAbort("Tet mesh header length mismatch!");
  }
};

/** Sends tets across the wire to one destination */
class tetSender {
  entityPackList packTets,packPts;
  int nSend; // Number of records we've been asked to send
  int nPut;
  tetMeshChunk ck;
  MPI_Request sendReq;	
public:
  tetSender(void) {
    nSend=0; nPut=0;
  }
  ~tetSender() {
  }
  
  /// This tet will be sent: count it
  void countTet(const sendState &s,int t) {
    nSend++;
    if (packTets.add(t,s.mesh.getTets())) {
      const int *conn=s.mesh.getTet(t);
      for (int i=0;i<POINT_PER_TET;i++)
	packPts.add(conn[i],s.mesh.getPoints());
    }
  }
  int getCount(void) const {return nSend;}
  
  /// Pack this tet up to be sent off (happens after all calls to "count")
  void putTet(const sendState &s,int t) {
    if (nPut==0) 
      { /* Allocate outgoing message buffer, and copy local data: */
	ck.allocate(s,nSend,packTets.n,packPts.n);
	for (int t=0;t<packTets.n;t++) 
	  { // Create local tet t, renumbering connectivity from global
	    int g=packTets.getGlobal(t); // Global number
	    copy(ck.getTetData(s,t),s.tetData(g),s.tetVal);

	    const int *gNode=s.mesh.getTet(g);
	    int *lNode=ck.getTetConn(s,t);
	    for (int i=0;i<POINT_PER_TET;i++)
	      lNode[i]=packPts.getLocal(gNode[i]);
	  }
	for (int p=0;p<packPts.n;p++) 
	  { // Create local node p from global node g
	    int g=packPts.getGlobal(p); // Global number
	    copy(ck.getPtData(s,p),s.ptData(g),s.ptVal);
	    copy(ck.getPtLoc(s,p),(const double *)s.mesh.getPoint(g),COORD_PER_POINT);
	  }
      }
    ck.getSendTets()[nPut++]=packTets.getLocal(t);
    
  }
  
  /// Send all put tets to this destination
  void isend(MPI_Comm comm,int src,int dest) {
    if (nSend==0) return; /* nothing to do */
    
#if OSL_com_debug 
    CkPrintf("%d sending %d records to %d\n", src,n,dest);
#endif
    MPI_Isend(ck.messageBuf(),ck.messageSizeXfer(),
	      PARALLELTRANSFER_MPI_DTYPE,
	      dest,PARALLELTRANSFER_MPI_TAG,comm,&sendReq);
  }
  /// Wait for sends to complete
  void wait(void) {
    if (nSend==0) return; /* nothing to do */
    MPI_Status sts;
    MPI_Wait(&sendReq,&sts);
  }
};



/** Provides access to an element received off the network. */
class ConcreteNetworkElement : public ConcreteElementNodeData {
  const meshState *s;
  tetMeshChunk &ck;
  int tet;
  const int *tetConn;
public:
  ConcreteNetworkElement(tetMeshChunk &ck_)
    :s(0), ck(ck_), tet(-1), tetConn(0) {}
  void setTet(const meshState &s_,int tet_) {
    s=&s_;
    tet=tet_;
    tetConn=ck.getTetConn(*s,tet);
  }
  
  /** Return the location of the i'th node of this element. */
  virtual CPoint getNodeLocation(int i) const {
    return CPoint(ck.getPtLoc(*s,tetConn[i]));
  }
  
  /** Return the vector of data associated with our i'th node. */
  virtual const double *getNodeData(int i) const {
    return ck.getPtData(*s,tetConn[i]);
  }
};

/** Receives tets from the wire */
class tetReceiver {
  int nRecv;
  tetMeshChunk ck; // Incoming message data
  int outCount;
  ConcreteNetworkElement outElement;
public:
  tetReceiver() :outElement(ck) { nRecv=0; }
  ~tetReceiver() { }
  
  void count(void) { nRecv++;}
  int getCount(void) const {return nRecv;}
  
  /** After all the "count" calls, grab data from network */
  void recv(const meshState &s,MPI_Comm comm,int src) {
    if (!nRecv) return; /* nothing to do */
    
    // Figure out how long the message we're sending is:
    MPI_Status sts;
    MPI_Probe(src,PARALLELTRANSFER_MPI_TAG,comm, &sts);
    int msgLen; MPI_Get_count(&sts, PARALLELTRANSFER_MPI_DTYPE, &msgLen);
    
    // Allocate and receive the message off the network
    xfer_t *buf=new xfer_t[msgLen];
    MPI_Recv(buf,msgLen,PARALLELTRANSFER_MPI_DTYPE,src,PARALLELTRANSFER_MPI_TAG,comm,&sts);
    ck.receive(s,buf,msgLen);
    
    outCount=0;
  }
  
  /// Extract the next tet (tet t of returned mesh) from the list.
  ///  Tets must be returned in the same order as presented in tetSender::putTet.
  ConcreteElementNodeData *getTet(const meshState &s,const xfer_t* &cells) { 
    int t=ck.getSendTets()[outCount++]; // Local number of this tet
    cells=ck.getTetData(s,t);
    outElement.setTet(s,t);
    return &outElement;
  }
};

/** Perform a parallel data transfer from srcVals to destVals */
void parallelTransfer_c::transfer(progress_t &progress) {
  int s,d; //Source and destination tets
  int p; //Processor
  int c; //Collision
  /* Convert input and output cells into bounding boxes:
     numbers 0..firstDest-1 are source tets (priority 1)
     numbers firstDest..lastDest-1 are dest tets (priority 2)
  */
  progress.p("Finding bounding boxes");
  firstDest=srcMesh.getTets();
  int lastDest=firstDest+destMesh.getTets();
  bbox3d *boxes=new bbox3d[lastDest];
  int *prio=new int[lastDest];
  progress.p("Finding bounding boxes: src");
  for (s=0;s<firstDest;s++) 
    { boxes[s]=getBox(s,srcMesh); prio[s]=1; }
  progress.p("Finding bounding boxes: dest");
  for (d=firstDest;d<lastDest;d++) 
    { boxes[d]=getBox(d-firstDest,destMesh); prio[d]=2; }
  
  /* Collide the bounding boxes */
  //printf("[%d] Rank %d: BEGIN colliding bounding boxes...\n",CkMyPe(), myRank);
  progress.p("Colliding bounding boxes");
  COLLIDE_Boxes_prio(voxels, lastDest,(const double *)boxes,prio);
  delete[] boxes; delete[] prio;
  //printf("[%d] Rank %d: DONE colliding bounding boxes...\n",CkMyPe(), myRank);
  
  /* Extract the list of collisions */
  progress.p("Extracting collision list");
  int nColl=COLLIDE_Count(voxels);
  int *coll=new int[3*nColl];
  COLLIDE_List(voxels,coll);
  
  /* Figure out the communication sizes with each PE */
  progress.p("Finding communication size");
  sendState ss(srcMesh,valsPerTet,valsPerPt, srcTet,srcPt);
  tetReceiver *recv=new tetReceiver[commSize];
  tetSender *send=new tetSender[commSize];
  for (c=0;c<nColl;c++) {
    const int *cr=&coll[3*c]; //Collision record:
    if (isLocal(cr)) continue;
    //Remote collision:
    if (isDest(cr[0])) /* collides our destination, so receive it */
      recv[cr[1]].count();
    else /* collides our source cell, so send it */ 
      send[cr[1]].countTet(ss,cr[0]);
  }
#if OSL_com_debug /* print out the communication table */
  printf("Rank %d: %d collisions, ",myRank,nColl);
  for (p=0;p<commSize;p++) 
    if (send[p].getCount() || recv[p].getCount())
      printf("(%d s%d r%d) ",p,send[p].getCount(),recv[p].getCount());
  CkPrintf("\n");
#endif
  
  /* Copy over outgoing data */
  progress.p("Creating outgoing messages");
  for (c=0;c<nColl;c++) {
    const int *cr=&coll[3*c]; //Collision record:
    if ((!isLocal(cr)) && (!isDest(cr[0]))) 
      send[cr[1]].putTet(ss,cr[0]);
  }
  
  /* Initiate send for outgoing data */
  progress.p("Isend");
  for (p=0;p<commSize;p++) send[p].isend(mpi_comm,myRank,p);
  
  /* Post receives for everything that hits our dest tets */
  progress.p("Recv");
  for (p=0;p<commSize;p++) recv[p].recv(ss,mpi_comm,p);
  
  /* Initiate send for outgoing data */
  progress.p("Wait");
  for (p=0;p<commSize;p++) send[p].wait();
  delete[] send;
  
  /* Do local and remote data transfer */
  progress.p("Transferring solution");
  for (c=0;c<nColl;c++) {
    const int *cr=&coll[3*c];  // Collision record:
    int dest=-1;  // Local destination tet number
    const xfer_t *sCell;  // Source cell-centered values
    ConcreteElementNodeData *srcElement=NULL;
    if (isLocal(cr)) { /* src and dest are local */
      int src=cr[0]; 
      dest=cr[2]-firstDest;
      // Ordering *should* be maintained by voxels:
      if (isDest(src) || dest<0) 
	CmiAbort("Collision library did not respect local priority");
      theLocalElement.set(src);
      srcElement=&theLocalElement;
      sCell=&srcTet[src*valsPerTet];
    }
    else if (isDest(cr[0])) { /* dest is local, src is remote */
      dest=cr[0]-firstDest;
      srcElement=recv[cr[1]].getTet(ss,sCell);
    }
    /* else isSrc, so it's send-only */
		
    if  (dest!=-1) {
      TetMeshElement destElement(dest,destMesh);
      double sharedVolume=getSharedVolumeTets(*srcElement,destElement);
      if (sharedVolume>0.0) { /* source and dest really overlap-- transfer */
	accumulateCellValues(sCell,dest,sharedVolume);
	transferNodeValues(*srcElement, dest);
      }
    }
  }
  delete[] recv;
  delete[] coll;

  /* Convert summed values from volume-weighted values to plain values */
  progress.p("Normalizing transfer");
  for (d=0;d<destMesh.getTets();d++) {
    double trueVolume=destMesh.getTetVolume(d);
    double volErr=fabs(destVolumes[d]-trueVolume);
    // double accumScale=1.0/trueVolume; // testing version: uncompensated
    double accumScale=1.0/destVolumes[d]; //Reverse volume weighting
    double relErr=volErr*accumScale;
    if (0 && (fabs(relErr)>1.0e-6 && volErr>1.0e-8)) {
      printf("WARNING: ------------- volume mismatch for cell %d -------------\n"
	     " True volume %g, but total is only %g (err %g)\n",
	     d,trueVolume,destVolumes[d],volErr);
      // abort();
    }
    // Compensate for partially-filled cells: divide out volume
    // WHY THIS CONDITION?
    //if (destVolumes[d]>1.0e-12)
      for (int v=0;v<valsPerTet;v++) 
	destTet[d*valsPerTet+v]*=accumScale;
  }
}

class VerboseProgress_t : public progress_t {
  MPI_Comm comm;
  int myRank;
  const char *module;
  const char *last;
  double start;
  void printLast(void) {
    double t=MPI_Wtime();
    double withoutBarrier=t-start; start=t;
    MPI_Barrier(comm);
    t=MPI_Wtime();
    double barrier=t-start; start=t;
    if (myRank==0 && last && ((withoutBarrier>0.1) || (barrier>0.1)))
      {
	CkPrintf("%s: %s took %.6f s (+%.6f s imbalance)\n",
		 module,last,withoutBarrier,barrier);
	fflush(stdout);
      }
  }
public:
  VerboseProgress_t(MPI_Comm comm_,const char *module_) {
    comm=comm_;
    MPI_Comm_rank(comm,&myRank);
    last=NULL;
    module=module_;
  }
  ~VerboseProgress_t() {
    if (last) printLast();
  }
  virtual void p(const char *where) {
    printLast();
    last=where;
  }
};

progress_t::~progress_t() {}


void ParallelTransfer(collide_t voxels,MPI_Comm mpi_comm,
		      int valsPerTet,int valsPerPt,
		      const xfer_t *srcTet,const xfer_t *srcPt,const TetMesh &srcMesh,
		      xfer_t *destTet,xfer_t *destPt,const TetMesh &destMesh)
{
  parallelTransfer_c t(voxels,mpi_comm,valsPerTet,valsPerPt,
		       srcTet,srcPt,srcMesh, destTet,destPt,destMesh);
  VerboseProgress_t p(mpi_comm,"ParallelTransfer");
  t.transfer(p);
}

