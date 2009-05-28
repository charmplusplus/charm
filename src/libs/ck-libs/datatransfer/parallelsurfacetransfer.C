/**
 * Conservative, accurate parallel cell-centered;
 * and interpolation-based node-centered data transfer.
 * Terry L. Wilmarth, wilmarth@uiuc.edu, 4 Oct 2006
 */
#include "triSurfMesh.h"
#include "prismMesh.h"
#include "transfer.h"
#include "bbox.h"
#include "parallelsurfacetransfer.h"
#include "charm++.h" /* for CmiAbort */
#include "GenericElement.h"

#define COORD_PER_POINT 3 
#define POINT_PER_PRISM 6 
#define POINT_PER_TRIANGLE 3

class surfProgress_t {
public:
  virtual ~surfProgress_t();
  virtual void p(const char *where) {}
};

/** Provides access to a local element. */
class ConcreteLocalElement : public ConcreteElementNodeData {
  const PrismMesh &mesh;
  const int *conn;
  const double *ptVals; // contains valsPerPt values for each point
  int valsPerPt;
public:
  ConcreteLocalElement(const PrismMesh &mesh_,const double *ptVals_,int valsPerPt_)
    :mesh(mesh_), conn(0), ptVals(ptVals_), valsPerPt(valsPerPt_) { }
  void set(int prism) {
    conn=mesh.getPrism(prism);
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


class parallelSurfaceTransfer_c {
  collide_t voxels;
  int firstDest; //Numbers greater than this are destination faces
  /// Return true if this is the collision-box number of a destination face.
  /// Only works for local numbers.
  inline bool isDest(int n) {return n>=firstDest;}
  
  MPI_Comm mpi_comm;
  int myRank,commSize;
  /// Return true if this collision record describes a local intersection
  inline bool isLocal(const int *coll) {return coll[1]==myRank;}
  
  int valsPerFace, valsPerPt;
  const PrismMesh &srcMesh;
  const double *srcFaceVals; 
  const double *srcPtVals; 
  const TriangleSurfaceMesh &destMesh;
  double *destFaceVals; 
  double *destPtVals;
  double *destAreas; 
  
  ConcreteLocalElement theLocalElement;
  
  /// A source face, with values sVals, overlaps with this dest face
  /// with this much shared area.  Transfer face-centered values.
  void accumulateCellValues(const double *sCellVals, int dest, double sharedArea) {
    for (int v=0;v<valsPerFace;v++) {
      destFaceVals[dest*valsPerFace+v]+=sharedArea*sCellVals[v];
    }
    destAreas[dest]+=sharedArea;
  }
  
  /// This source element, with values sPt, overlaps with this dest element.
  ///   Transfer any possible node-centered values.
  void transferNodeValues(const ConcreteElementNodeData &srcElement, int dest)
  {
    GenericElement el(POINT_PER_PRISM);
    for (int dni=0;dni<POINT_PER_TRIANGLE;dni++) {
      int dn=destMesh.getTriangle(dest)[dni];
      CkVector3d dnLoc=destMesh.getPoint(dn);
      CVector natc;
      if (el.element_contains_point(dnLoc,srcElement,natc)) 
	{ /* this source element overlaps our destination */
	  el.interpolate_natural(valsPerPt,srcElement,natc,&destPtVals[dn*valsPerPt]);
	}
    }
  }
public:
  parallelSurfaceTransfer_c(collide_t voxels_, MPI_Comm mpi_comm_, int valsPerFace_,
			    int valsPerPt_, const double *srcFaceVals_, 
			    const double *srcPtVals_, const PrismMesh &srcMesh_, 
			    double *destFaceVals_, double *destPtVals_,
			    const TriangleSurfaceMesh &destMesh_)
    :voxels(voxels_), mpi_comm(mpi_comm_), 
     valsPerFace(valsPerFace_), valsPerPt(valsPerPt_),
     srcMesh(srcMesh_),srcFaceVals(srcFaceVals_),srcPtVals(srcPtVals_),
     destMesh(destMesh_),destFaceVals(destFaceVals_),destPtVals(destPtVals_),
     theLocalElement(srcMesh,srcPtVals,valsPerPt)
  {
    MPI_Comm_rank(mpi_comm,&myRank);
    MPI_Comm_size(mpi_comm,&commSize);
    destAreas=new double[destMesh.getTriangles()];
    for (int d=0;d<destMesh.getTriangles();d++) {
      destAreas[d]=0;
      for (int v=0;v<valsPerFace;v++) {
	destFaceVals[d*valsPerFace+v]=(double)0;
      }
    }
  }
  ~parallelSurfaceTransfer_c() {
    delete[] destAreas;
  }
  
  /** Perform a parallel data transfer from srcVals to destVals */
  void transfer(surfProgress_t &surfProgress);
};

static bbox3d getPrismBox(int t,const PrismMesh &mesh) {
  bbox3d ret; ret.empty();
  const int *conn=mesh.getPrism(t);
  for (int i=0;i<POINT_PER_PRISM;i++)
    ret.add(mesh.getPoint(conn[i]));
  return ret;
}

static bbox3d getTriangleBox(int t,const TriangleSurfaceMesh &mesh) {
  bbox3d ret; ret.empty();
  const int *conn=mesh.getTriangle(t);
  for (int i=0;i<POINT_PER_TRIANGLE;i++)
    ret.add(mesh.getPoint(conn[i]));
  return ret;
}

// Return the number of double's (usual doubles) this many bytes corresponds to
inline int bytesToXfer(int nBytes) {
  return (nBytes+sizeof(double)-1)/sizeof(double);
}

// Copy n values from src to dest
template <class D,class S>
inline void copy(D *dest,const S *src,int n) {
  for (int i=0;i<n;i++) dest[i]=(D)src[i];
}

// Describes the amounts of user data associated with each entity in the mesh
class meshState {
public:
  int faceVal, ptVal; // Number of user data doubles per tet & pt
  meshState(int t,int p) :faceVal(t), ptVal(p) {}
};

/** Describes the outgoing mesh */
class sendState : public meshState {
public:
  const PrismMesh &mesh; // Global source mesh
  const double *faceVals; // User data for each tet (tetVal * mesh->getTets())
  const double *ptVals;  // User data for each point (ptVal * mesh->getPoints())
  
  const double *faceData(int t) const {return &faceVals[t*faceVal];}
  const double * ptData(int p) const {return & ptVals[p* ptVal];}
  
  sendState(const PrismMesh &m,int tv,int pv,const double *t,const double *p)
    :meshState(tv,pv), mesh(m), faceVals(t), ptVals(p) {}
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
class meshChunk {
  double *buf; // Message buffer
  
  /* On-the-wire message header */
  struct header_t {
    int nXfer; // Total length of message, in double's
    int nSend; // Length of sendTets array, in ints
    int nFace; // Number of rows of faceData, 
    int nPt; // Number of rows of ptData
  };
  header_t *header; // Header of buffer
  inline int headerSize() const // size in doubles
  {return bytesToXfer(sizeof(header_t));}
  
  int *sendFaces; // List of local numbers of faces, matching collision data
  inline int sendFacesSize(const meshState &s,int nSend) const
  {return bytesToXfer(nSend*sizeof(int));}
  
  double *faceData; // Face connectivity and user data (local numbers)
  inline int faceDataRecordSize(const meshState &s) const
  {return bytesToXfer(POINT_PER_PRISM*sizeof(int))+s.faceVal;}
  inline int faceDataSize(const meshState &s,int nFaces) const
  {return nFaces*faceDataRecordSize(s); }
  
  double *ptData; // Point location and user data (local numbers)
  inline int ptDataRecordSize(const meshState &s) const
  {return COORD_PER_POINT+s.ptVal;}
  inline int ptDataSize(const meshState &s,int nPt) const
  {return nPt*ptDataRecordSize(s); }
public:
  meshChunk() {buf=NULL;}
  ~meshChunk() { if (buf) delete[] buf;}
  
  // Used by send side:
  /// Allocate a new outgoing message with this size: 
  void allocate(const meshState &s,int nSend,int nFace,int nPt) {
    int msgLen=headerSize()+sendFacesSize(s,nSend)+faceDataSize(s,nFace)+ptDataSize(s,nPt);
    buf=new double[msgLen];
    header=(header_t *)buf;
    header->nXfer=msgLen;
    header->nSend=nSend;
    header->nFace=nFace;
    header->nPt=nPt;
    setupPointers(s,msgLen);
  }
  
  /// Return the number of doubles in this message:
  int messageSizeXfer(void) const {return header->nXfer;}
  double *messageBuf(void) const {return buf;}
  
  // Used by receive side:
  /// Set up our pointers into this message, which
  ///  is then owned by and will be deleted by this object.
  void receive(const meshState &s,double *buf_,int msgLen) {
    buf=buf_; 
    setupPointers(s,msgLen);
  }
  
  // Used by both sides:
  /// Return the list of tets that are being sent:
  inline int *getSendFaces(void) {return sendFaces;}
  inline int nSendFaces(void) const {return header->nSend;}
  
  /// Return the number of tets included in this message:
  inline int nFaces(void) const {return header->nFace;}
  /// Return the connectivity array associated with this tet:
  inline int *getFaceConn(const meshState &s,int t) {
    return (int *)&faceData[t*faceDataRecordSize(s)];
  }
  /// Return the user data associated with this tet:
  inline double *getFaceData(const meshState &s,int t) {
    return &faceData[t*faceDataRecordSize(s)+bytesToXfer(POINT_PER_PRISM*sizeof(int))];
  }
  
  /// Return the number of points included in this message:
  inline int nPts(void) const {return header->nPt;}
  /// Return the coordinate data for this point:
  inline double *getPtLoc(const meshState &s,int n) {
    return &ptData[n*ptDataRecordSize(s)];
  }
  /// Return the user data for this point:
  inline double *getPtData(const meshState &s,int n) {
    return &ptData[n*ptDataRecordSize(s)+COORD_PER_POINT];
  }
  
private:
  // Point sendFaces and the data arrays into message buffer:
  void setupPointers(const meshState &s,int msgLen) {
    double *b=buf; 
    header=(header_t *)b; b+=headerSize();
    sendFaces=(int *)b; b+=sendFacesSize(s,header->nSend);
    faceData=b; b+=faceDataSize(s,header->nFace);
    ptData=b; b+=ptDataSize(s,header->nPt);
    int nRead=b-buf;
    if (nRead!=header->nXfer || nRead!=msgLen)
      CkAbort("Mesh header length mismatch!");
  }
};

/** Sends faces across the wire to one destination */
class faceSender {
  entityPackList packFaces,packPts;
  int nSend; // Number of records we've been asked to send
  int nPut;
  meshChunk ck;
  MPI_Request sendReq;	
public:
  faceSender(void) {
    nSend=0; nPut=0;
  }
  ~faceSender() {
  }
  
  /// This face will be sent: count it
  void countFace(const sendState &s,int t) {
    nSend++;
    if (packFaces.add(t,s.mesh.getPrisms())) {
      const int *conn=s.mesh.getPrism(t);
      for (int i=0;i<POINT_PER_PRISM;i++)
	packPts.add(conn[i],s.mesh.getPoints());
    }
  }
  int getCount(void) const {return nSend;}
  
  /// Pack this face up to be sent off (happens after all calls to "count")
  void putFace(const sendState &s,int t) {
    if (nPut==0) 
      { /* Allocate outgoing message buffer, and copy local data: */
	ck.allocate(s,nSend,packFaces.n,packPts.n);
	for (int t=0;t<packFaces.n;t++) 
	  { // Create local face t, renumbering connectivity from global
	    int g=packFaces.getGlobal(t); // Global number
	    copy(ck.getFaceData(s,t),s.faceData(g),s.faceVal);

	    const int *gNode=s.mesh.getPrism(g);
	    int *lNode=ck.getFaceConn(s,t);
	    for (int i=0;i<POINT_PER_PRISM;i++)
	      lNode[i]=packPts.getLocal(gNode[i]);
	  }
	for (int p=0;p<packPts.n;p++) 
	  { // Create local node p from global node g
	    int g=packPts.getGlobal(p); // Global number
	    copy(ck.getPtData(s,p),s.ptData(g),s.ptVal);
	    copy(ck.getPtLoc(s,p),(const double *)s.mesh.getPoint(g),COORD_PER_POINT);
	  }
      }
    ck.getSendFaces()[nPut++]=packFaces.getLocal(t);
    
  }
  
  /// Send all put faces to this destination
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
  meshChunk &ck;
  int face;
  const int *faceConn;
public:
  ConcreteNetworkElement(meshChunk &ck_)
    :s(0), ck(ck_), face(-1), faceConn(0) {}
  void setFace(const meshState &s_,int face_) {
    s=&s_;
    face=face_;
    faceConn=ck.getFaceConn(*s,face);
  }
  
  /** Return the location of the i'th node of this element. */
  virtual CPoint getNodeLocation(int i) const {
    return CPoint(ck.getPtLoc(*s,faceConn[i]));
  }
  
  /** Return the vector of data associated with our i'th node. */
  virtual const double *getNodeData(int i) const {
    return ck.getPtData(*s,faceConn[i]);
  }
};

/** Receives faces from the wire */
class faceReceiver {
  int nRecv;
  meshChunk ck; // Incoming message data
  int outCount;
  ConcreteNetworkElement outElement;
public:
  faceReceiver() :outElement(ck) { nRecv=0; }
  ~faceReceiver() { }
  
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
    double *buf=new double[msgLen];
    MPI_Recv(buf,msgLen,PARALLELTRANSFER_MPI_DTYPE,src,PARALLELTRANSFER_MPI_TAG,comm,&sts);
    ck.receive(s,buf,msgLen);
    
    outCount=0;
  }
  
  /// Extract the next face (face t of returned mesh) from the list.
  ///  Faces must be returned in the same order as presented in faceSender::putFace.
  ConcreteElementNodeData *getFace(const meshState &s,const double* &faces) { 
    int t=ck.getSendFaces()[outCount++]; // Local number of this face
    faces=ck.getFaceData(s,t);
    outElement.setFace(s,t);
    return &outElement;
  }
};

/** Perform a parallel data transfer from srcVals to destVals */
void parallelSurfaceTransfer_c::transfer(surfProgress_t &surfProgress) {
  int s,d; //Source and destination tets
  int p; //Processor
  int c; //Collision
  /* Convert input and output faces into bounding boxes:
     numbers 0..firstDest-1 are source prisms (priority 1)
     numbers firstDest..lastDest-1 are dest triangles (priority 2)
  */
  surfProgress.p("Finding bounding boxes");
  firstDest=srcMesh.getPrisms();
  int lastDest=firstDest+destMesh.getTriangles();
  bbox3d *boxes=new bbox3d[lastDest];
  int *prio=new int[lastDest];
  surfProgress.p("Finding bounding boxes: src");
  for (s=0;s<firstDest;s++) 
    { boxes[s]=getPrismBox(s,srcMesh); prio[s]=1; }
  surfProgress.p("Finding bounding boxes: dest");
  for (d=firstDest;d<lastDest;d++) 
    { boxes[d]=getTriangleBox(d-firstDest,destMesh); prio[d]=2; }
  
  /* Collide the bounding boxes */
  printf("[%d] Rank %d: BEGIN colliding bounding boxes...\n",CkMyPe(), myRank);
  surfProgress.p("Colliding bounding boxes");
  COLLIDE_Boxes_prio(voxels, lastDest,(const double *)boxes,prio);
  delete[] boxes; delete[] prio;
  printf("[%d] Rank %d: DONE colliding bounding boxes...\n",CkMyPe(), myRank);
  
  /* Extract the list of collisions */
  surfProgress.p("Extracting collision list");
  int nColl=COLLIDE_Count(voxels);
  int *coll=new int[3*nColl];
  COLLIDE_List(voxels,coll);
  
  /* Figure out the communication sizes with each PE */
  surfProgress.p("Finding communication size");
  sendState ss(srcMesh,valsPerFace,valsPerPt, srcFaceVals,srcPtVals);
  faceReceiver *recv=new faceReceiver[commSize];
  faceSender *send=new faceSender[commSize];
  for (c=0;c<nColl;c++) {
    const int *cr=&coll[3*c]; //Collision record:
    if (isLocal(cr)) continue;
    //Remote collision:
    if (isDest(cr[0])) /* collides our destination, so receive it */
      recv[cr[1]].count();
    else /* collides our source face, so send it */ 
      send[cr[1]].countFace(ss,cr[0]);
  }
#if OSL_com_debug /* print out the communication table */
  printf("Rank %d: %d collisions, ",myRank,nColl);
  for (p=0;p<commSize;p++) 
    if (send[p].getCount() || recv[p].getCount())
      printf("(%d s%d r%d) ",p,send[p].getCount(),recv[p].getCount());
  CkPrintf("\n");
#endif
  
  /* Copy over outgoing data */
  surfProgress.p("Creating outgoing messages");
  for (c=0;c<nColl;c++) {
    const int *cr=&coll[3*c]; //Collision record:
    if ((!isLocal(cr)) && (!isDest(cr[0]))) 
      send[cr[1]].putFace(ss,cr[0]);
  }
  
  /* Initiate send for outgoing data */
  surfProgress.p("Isend");
  for (p=0;p<commSize;p++) send[p].isend(mpi_comm,myRank,p);
  
  /* Post receives for everything that hits our dest tets */
  surfProgress.p("Recv");
  for (p=0;p<commSize;p++) recv[p].recv(ss,mpi_comm,p);
  
  /* Initiate send for outgoing data */
  surfProgress.p("Wait");
  for (p=0;p<commSize;p++) send[p].wait();
  delete[] send;
  
  /* Do local and remote data transfer */
  surfProgress.p("Transferring solution");
  for (c=0;c<nColl;c++) {
    const int *cr=&coll[3*c];  // Collision record:
    int dest=-1;  // Local destination tet number
    const double *sFace;  // Source face-centered values
    ConcreteElementNodeData *srcElement=NULL;
    if (isLocal(cr)) { /* src and dest are local */
      int src=cr[0]; 
      dest=cr[2]-firstDest;
      // Ordering *should* be maintained by voxels:
      if (isDest(src) || dest<0) 
	CmiAbort("Collision library did not respect local priority");
      theLocalElement.set(src);
      srcElement=&theLocalElement;
      sFace=&srcFaceVals[src*valsPerFace];
    }
    else if (isDest(cr[0])) { /* dest is local, src is remote */
      dest=cr[0]-firstDest;
      srcElement=recv[cr[1]].getFace(ss,sFace);
    }
    /* else isSrc, so it's send-only */
		
    if  (dest!=-1) {
      Triangle3DElement destElement(dest,destMesh);
      double sharedArea=getSharedArea(*srcElement,destElement);
      if (sharedArea>0.0) { /* source and dest really overlap-- transfer */
	accumulateCellValues(sFace,dest,sharedArea);
	transferNodeValues(*srcElement, dest);
      }
    }
  }
  delete[] recv;
  delete[] coll;

  /* Convert summed values from volume-weighted values to plain values */
  surfProgress.p("Normalizing transfer");
  for (d=0;d<destMesh.getTriangles();d++) {
    double trueArea=destMesh.getArea(d);
    double areaErr=fabs(destAreas[d]-trueArea);
    // double accumScale=1.0/trueArea; // testing version: uncompensated
    double accumScale=1.0/destAreas[d]; //Reverse volume weighting
    double relErr=areaErr*accumScale;
    if (0 && (fabs(relErr)>1.0e-6 && areaErr>1.0e-8)) {
      printf("WARNING: ------------- area mismatch for face %d -------------\n"
	     " True area %g, but total is only %g (err %g)\n",
	     d,trueArea,destAreas[d],areaErr);
      // abort();
    }
    // Compensate for partially-filled faces: divide out volume
    // WHY THIS CONDITION?
    //if (destAreas[d]>1.0e-12)
      for (int v=0;v<valsPerFace;v++) 
	destFaceVals[d*valsPerFace+v]*=accumScale;
  }
}

class VerboseSurfProgress_t : public surfProgress_t {
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
  VerboseSurfProgress_t(MPI_Comm comm_,const char *module_) {
    comm=comm_;
    MPI_Comm_rank(comm,&myRank);
    last=NULL;
    module=module_;
  }
  ~VerboseSurfProgress_t() {
    if (last) printLast();
  }
  virtual void p(const char *where) {
    printLast();
    last=where;
  }
};

surfProgress_t::~surfProgress_t() {}


void ParallelSurfaceTransfer(collide_t voxels, MPI_Comm mpi_comm, int valsPerFace, 
			     int valsPerPt, const double *srcFaceVals, 
			     const double *srcPtVals,
			     const PrismMesh &srcMesh, double *destFaceVals, 
			     double *destPtVals, const TriangleSurfaceMesh &destMesh)
{
  printf("BEGIN ParallelSurfaceTransfer...\n");
  parallelSurfaceTransfer_c t(voxels, mpi_comm, valsPerFace, valsPerPt, srcFaceVals, 
		       srcPtVals, srcMesh, destFaceVals, destPtVals, destMesh);
  VerboseSurfProgress_t p(mpi_comm,"ParallelSurfaceTransfer");
  t.transfer(p);
  printf("END ParallelSurfaceTransfer.\n");
}

