/**
 * Conservative, accurate parallel cell-centered data transfer.
 * Orion Sky Lawlor, olawlor@acm.org, 2003/3/24
 */
#include "tetmesh.h"
#include "collidec.h"
#include "bbox.h"
#include "mpi.h"
#include "charm++.h" /* for CmiAbort */

#define OSL_COMM_DEBUG 0

class progress_t {
public:
	virtual ~progress_t();
	virtual void p(const char *where) =0;
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

	int valsPerTet;
	const TetMesh &srcMesh;
	const double *srcVals; // srcMesh.getTets()*valsPerTet source values
	const TetMesh &destMesh;
	double *destVals; // destMesh.getTets()*valsPerTet partial values
	double *destVolumes; // destMesh.getTets() partial volumes
	
	/// A source cell, with values sVals, overlaps with this dest cell
	///  with this shared volume.
	void addValues(const double *sVals,int dest,double sharedVolume) {
		for (int v=0;v<valsPerTet;v++) 
			destVals[dest*valsPerTet+v]+=sharedVolume*sVals[v];
		destVolumes[dest]+=sharedVolume;
	}
public:
	parallelTransfer_c(collide_t voxels_,MPI_Comm mpi_comm_,
		int valsPerTet_,
		const double *srcVals_,const TetMesh &srcMesh_,
		double *destVals_,const TetMesh &destMesh_)
		:voxels(voxels_), mpi_comm(mpi_comm_), 
		 valsPerTet(valsPerTet_), 
		 srcVals(srcVals_),srcMesh(srcMesh_),
		 destVals(destVals_),destMesh(destMesh_) 
	{
		MPI_Comm_rank(mpi_comm,&myRank);
		MPI_Comm_size(mpi_comm,&commSize);
		destVolumes=new double[destMesh.getTets()];
		for (int d=0;d<destMesh.getTets();d++) {
			destVolumes[d]=0;
			for (int v=0;v<valsPerTet;v++) {
				destVals[d*valsPerTet+v]=0;
			}
		}
	}
	~parallelTransfer_c() {
		delete[] destVolumes;
	}
	
	/**
	 Perform a parallel data transfer from srcVals to destVals
	*/
	void transfer(progress_t &progress);
};

static bbox3d getBox(int t,const TetMesh &mesh) {
	bbox3d ret; ret.empty();
	const int *conn=mesh.getTet(t);
	for (int i=0;i<4;i++)
		ret.add(mesh.getPoint(conn[i]));
	return ret;
}

/// MPI Tag for data transfer:
#define DATATRANSFER_TAG 0xDA7A

/**
 The on-the-wire mesh format looks like this:
   list of points 0..nPoints-1
      x,y,z for point; ptVal values per point
   list of tets 0..nTets-1
      p1,p2,p3,p4 point indices; tetVal values per tet
   list of tet indices, in order they're shared
*/
#define COORD_PER_POINT 3
#define POINT_PER_TET 4
#define DOUBLES_PER_TET (COORD_PER_POINT*POINT_PER_TET+valsPerTet)

// Return the number of doubles this many bytes corresponds to:
inline int bytesToDoubles(int nBytes) {
	return (nBytes+sizeof(double)-1)/sizeof(double);
}

// Copy n values from src to dest
template <class T>
inline void copy(T *dest,const T *src,int n) {
	for (int i=0;i<n;i++) dest[i]=src[i];
}

/** Describes the amounts of user data associated with each entity in the mesh */
class meshState {
public:
	int tetVal, ptVal; // Number of user data doubles per tet & pt
	meshState(int t,int p) :tetVal(t), ptVal(p) {}
};

/** Describes the outgoing mesh */
class sendState : public meshState {
public:
	const TetMesh &mesh; // Global source mesh
	const double *tetVals; // User data doubles for each tet (tetVal * mesh->getTets())
	const double *ptVals;  // User data doubles for each point (ptVal * mesh->getPoints())
	
	const double *tetData(int t) const {return &tetVals[t*tetVal];}
	const double * ptData(int p) const {return & ptVals[p* ptVal];}
	
	sendState(const TetMesh &m,int tv,int pv,const double *t,const double *p)
		:meshState(tv,pv), mesh(m), tetVals(t), ptVals(p) {}
};

/** Keeps track of which elements are already in the message,
   and which aren't.
 */
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

/** On-the-wire mesh format when sending/receiving tet mesh chunks: */
class tetMeshChunk {
	double *buf; // Message buffer
	
	/* On-the-wire message header */
	struct header_t {
		int nDoubles; // Total length of message, in doubles
		int nSend; // Length of sendTets array, in ints
		int nTet; // Number of rows of tetData, 
		int nPt; // Number of rows of ptData
	};
	header_t *header; // Header of buffer
	inline int headerSize() const // size in doubles
		{return bytesToDoubles(sizeof(header_t));}
	
	int *sendTets; // List of local numbers of tets, matching collision data
	inline int sendTetsSize(const meshState &s,int nSend) const
		{return bytesToDoubles(nSend*sizeof(int));}
	
	double *tetData; // Tet connectivity and user data (local numbers)
	inline int tetDataRecordSize(const meshState &s) const
		{return bytesToDoubles(POINT_PER_TET)+s.tetVal;}
	inline int tetDataSize(const meshState &s,int nTet) const
		{return nTet*tetDataRecordSize(s); }
	
	double *ptData; // Point location and user data (local numbers)
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
		buf=new double[msgLen];
		header=(header_t *)buf;
		header->nDoubles=msgLen;
		header->nSend=nSend;
		header->nTet=nTet;
		header->nPt=nPt;
		setupPointers(s,msgLen);
	}
	
	/// Return the number of doubles in this message:
	int messageSizeDoubles(void) const {return header->nDoubles;}
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
	inline int *getSendTets(void) {return sendTets;}
	inline int nSendTets(void) const {return header->nSend;}
	
	/// Return the number of tets included in this message:
	inline int nTets(void) const {return header->nTet;}
	/// Return the connectivity array associated with this tet:
	inline int *getTetConn(const meshState &s,int n) {
		return (int *)&tetData[n*tetDataRecordSize(s)];
	}
	/// Return the user data associated with this tet:
	inline double *getTetData(const meshState &s,int n) {
		return &tetData[n*tetDataRecordSize(s)+bytesToDoubles(POINT_PER_TET)];
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
	// Point sendTets and the data arrays into message buffer:
	void setupPointers(const meshState &s,int msgLen) {
		double *b=buf; 
		header=(header_t *)b; b+=headerSize();
		sendTets=(int *)b; b+=sendTetsSize(s,header->nSend);
		tetData=b; b+=tetDataSize(s,header->nTet);
		ptData=b; b+=ptDataSize(s,header->nPt);
		int nRead=b-buf;
		if (nRead!=header->nDoubles || nRead!=msgLen)
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

#if OSL_COMM_DEBUG 
		CkPrintf("%d sending %d records to %d\n", src,n,dest);
#endif
		MPI_Isend(ck.messageBuf(),ck.messageSizeDoubles(),MPI_DOUBLE,
			dest,DATATRANSFER_TAG,comm,&sendReq);
	}
	/// Wait for sends to complete
	void wait(void) {
		if (nSend==0) return; /* nothing to do */
		MPI_Status sts;
		MPI_Wait(&sendReq,&sts);
	}
};


/** Receives tets from the wire */
class tetReceiver {
	int nRecv;
	tetMeshChunk ck; // Incoming message data
	int outCount;
	TetMesh mesh; // HACK: temporary output mesh
public:
	tetReceiver() { nRecv=0; }
	~tetReceiver() { }
	
	void count(void) { nRecv++;}
	int getCount(void) const {return nRecv;}
	
	/** After all the "count" calls, grab data from network */
	void recv(const meshState &s,MPI_Comm comm,int src) {
		if (!nRecv) return; /* nothing to do */
		
		// Figure out how long the message we're sending is:
		MPI_Status sts;
		MPI_Probe(src,DATATRANSFER_TAG,comm, &sts);
		int msgLen; MPI_Get_count(&sts, MPI_DOUBLE, &msgLen);
		
		// Allocate and receive the message off the network
		double *buf=new double[msgLen];
		MPI_Recv(buf,msgLen,MPI_DOUBLE,src,DATATRANSFER_TAG,comm,&sts);
		ck.receive(s,buf,msgLen);
		
		// Unpack the message into our mesh:
		mesh.allocate(ck.nTets(), ck.nPts());
		for (int t=0;t<mesh.getTets();t++) 
			copy(mesh.getTet(t), ck.getTetConn(s,t), POINT_PER_TET);
		for (int p=0;p<mesh.getPoints();p++)
			copy((double *)mesh.getPoint(p), ck.getPtLoc(s,p), COORD_PER_POINT);
		outCount=0;
	}
	
	/// Extract the next tet (tet t of returned mesh) from the list.
	///  Tets must be returned in the same order as presented in tetSender::putTet.
	TetMesh *getTet(const meshState &s,int &t,const double* &vals) { 
		t=ck.getSendTets()[outCount++]; // Local number of this tet
		vals=ck.getTetData(s,t);
		return &mesh; 
	}
};

/**
 Perform a parallel data transfer from srcVals to destVals
*/
void parallelTransfer_c::transfer(progress_t &progress) {
	int s,d; //Source and destination tets
	int p; //Processor
	int c; //Collision
	int v; //Value
/* Convert input and output cells into bounding boxes:
	    numbers 0..firstDest-1 are source tets (priority 1)
	    numbers firstDest..lastDest-1 are dest tets (priority 2)
	 */
	progress.p("Finding bounding boxes");
	firstDest=srcMesh.getTets();
	int lastDest=firstDest+destMesh.getTets();
	bbox3d *boxes=new bbox3d[lastDest];
	int *prio=new int[lastDest];
	for (s=0;s<firstDest;s++) 
		{ boxes[s]=getBox(s,srcMesh); prio[s]=1; }
	for (d=firstDest;d<lastDest;d++) 
		{ boxes[d]=getBox(d-firstDest,destMesh); prio[d]=2; }
	
/* Collide the bounding boxes */
	progress.p("Colliding bounding boxes");
	COLLIDE_Boxes_prio(voxels, lastDest,(const double *)boxes,prio);
	delete[] boxes; delete[] prio;
	
/* Extract the list of collisions */
	progress.p("Extracting collision list");
	int nColl=COLLIDE_Count(voxels);
	int *coll=new int[3*nColl];
	COLLIDE_List(voxels,coll);
	
/* Figure out the communication sizes with each PE */
	progress.p("Finding communication size");
	sendState ss(srcMesh,valsPerTet,0, srcVals,0);
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
#if 1 //OSL_COMM_DEBUG /* print out the communication table */
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
		const int *cr=&coll[3*c]; //Collision record:
		int src,dest=-1; // Source and destination tets
		const TetMesh *sMesh; // Source and destination meshes
		const double *sVals; // Source and destination values
		if (isLocal(cr)) { /* src and dest are local */
			src=cr[0], dest=cr[2]-firstDest;
			// Ordering *should* be maintained by voxels:
			if (isDest(src) || dest<0) CmiAbort("Collision library did not respect local priority");
			sMesh=&srcMesh; sVals=&srcVals[src*valsPerTet];
		}
		else if (isDest(cr[0])) { /* dest is local, src is remote */
			src=-1, dest=cr[0]-firstDest;
			sMesh=recv[cr[1]].getTet(ss,src,sVals);
		}
		/* else isSrc, so it's send-only */
		
		if  (dest!=-1) {
			double sharedVolume=getSharedVolume(src,*sMesh,dest,destMesh);
			if (sharedVolume>0) 
				addValues(sVals,dest,sharedVolume);
		}
	}
	delete[] recv;
	delete[] coll;
	
/* Convert summed values from volume-weighted values to plain values */
	progress.p("Normalizing transfer");
	for (d=0;d<destMesh.getTets();d++) {
		double trueVolume=destMesh.getTetVolume(d);
		double volErr=fabs(destVolumes[d]-trueVolume);
		double accumScale=1.0/destVolumes[d]; //Reverse volume weighting
		double relErr=volErr*accumScale;
		if (fabs(relErr)>1.0e-5 && volErr>1.0e-7) {
			printf("WARNING: ------------- volume mismatch for cell %d -------------\n"
				" True volume %g, but total is only %g (err %g)\n",
				d,trueVolume,destVolumes[d],volErr);
#if OSL_CG3D_DEBUG
			// abort();
#endif
		}
		for (int v=0;v<valsPerTet;v++) destVals[d*valsPerTet+v]*=accumScale;
	}
}

class VerboseProgress_t : public progress_t {
	MPI_Comm comm;
	int myRank;
	const char *last;
	double start;
	void printLast(void) {
		double t=MPI_Wtime();
		double withoutBarrier=t-start; start=t;
		MPI_Barrier(comm);
		t=MPI_Wtime();
		double barrier=t-start; start=t;
		if (myRank==0 && last)
			CkPrintf("took %.6f s (+%.6f s imbalance)\n",
				withoutBarrier,barrier);
	}
public:
	VerboseProgress_t(MPI_Comm comm_) {
		comm=comm_;
		MPI_Comm_rank(comm,&myRank);
		last=NULL;
	}
	~VerboseProgress_t() {
		if (last) printLast();
	}
	virtual void p(const char *where) {
		printLast();
		last=where;
		if (myRank==0) CkPrintf("%s ",where);
	}
};

progress_t::~progress_t() {}


void parallelTransfer(collide_t voxels,MPI_Comm mpi_comm,
	int valsPerTet,
	const double *srcVals,const TetMesh &srcMesh,
	double *destVals,const TetMesh &destMesh)
{
	parallelTransfer_c t(voxels,mpi_comm,valsPerTet,
		srcVals,srcMesh, destVals,destMesh);
	VerboseProgress_t p(mpi_comm);
	t.transfer(p);
}

