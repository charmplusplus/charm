/*Charm++ Finite Element Framework:
C++ implementation file

This code matches up the components of a symmetry condition--
either linear periodicity, rotational periodicity, or 
mirror symmetry.

Orion Sky Lawlor, olawlor@acm.org, 7/23/2002
*/
#include "ParFUM.h"
#include "ParFUM_internals.h"

/*******************
Trivial Union/Find data structure.
This has poor worst-case complexity-- you need tree balancing
or online path compression for a general-purpose library.
*/
class unionFind : private CkNoncopyable {
	int n; //Number of nodes
	int *parent; //Parent of each node (normally parent[i]=i;)
public:
	unionFind(int n_) {
		n=n_;
		parent=new int[n];
		for (int i=0;i<n;i++) parent[i]=i;
	}
	~unionFind(void) {
		delete[] parent;
	}
	//Release control of parent array
	int *detach(void) {
		int *ret=parent;
		parent=NULL;
		return ret;
	}
	
	//Return the set that node i belongs to (linear worst case, often constant)
	int find(int i) const {
		while (i!=parent[i]) i=parent[i];
		return i;
	}
	
	//Make a and b have the same root (linear worst case, often constant)
	void combine(int a,int b) {
		parent[find(a)]=find(b);
	}
	
	//Compress paths, so subsequent finds are quick
	// (O(n^2) worst case, often linear)
	void compress(void) {
		for (int i=0;i<n;i++) parent[i]=find(i);
	}
};


/*******************
A set of "faces"-- index-tuples into a table of nodes.
A faceSet is used to do the detailed loops that match
up faces and nodes.
*/
class faceSet {
	int nFaces; //Number of faces we represent
	int nPer; //Nodes per face (maximum)
	const int *idx; //Array of [nFaces*nPer] idxBase-based node indices
	int idxBase; //0 for C, 1 for Fortran
	const CkVector3d *loc; //Locations of nodes
	CkVector3d *faceCen; //Centroids of faces

	//Compute the average position of this face
	CkVector3d calcFaceLoc(int faceNo) const 
	{
		CkVector3d sum(0.0);
		int total=0;
		for (int i=0;i<nPer;i++) {
			int nodeNo=getNode(faceNo,i);
			if (nodeNo!=-1) {
				sum+=getNodeLoc(nodeNo);
				total++;
			}
		}
		return sum*(1.0/total);
	}
	//Increment this (face-local) node index until it's no longer -1
	int findValid(int faceNo,int idx) const {
		while (getNode(faceNo,idx)==-1) {
			idx++;
			if (idx>=nPer) idx=0;
		}
		return idx;
	}
	//Compute the minimum edge length on any of my edges
	double calculateFaceLenSq(int faceNo,double minLenSq) const
	{
		for (int i=0;i<nPer;i++) {
			int l=findValid(faceNo,i);
			int r=findValid(faceNo,(l+1)%nPer);
			double curLenSq=getNodeLoc(getNode(faceNo,l)).distSqr(getNodeLoc(getNode(faceNo,r)));
			if (curLenSq<minLenSq) minLenSq=curLenSq;
		}
		return minLenSq;
	}
	
public:
	faceSet(int nFaces_,int nPer_,
		const int *idx_,int idxBase_,
		const CkVector3d *loc_)
		:nFaces(nFaces_),nPer(nPer_),idx(idx_),idxBase(idxBase_),loc(loc_) 
	{
		//Caching the face centroids speeds up the O(nFaces) getLocFace routine by 5x.
		faceCen=new CkVector3d[nFaces];
		for (int f=0;f<nFaces;f++) faceCen[f]=calcFaceLoc(f);
	}
	~faceSet() {delete[] faceCen;}
	
	//Compute the minimum edge length on any of my faces
	double getMinEdgeLength(void) const {
		double minLenSq=1.0e30;
		for (int f=0;f<nFaces;f++) {
			minLenSq=calculateFaceLenSq(f,minLenSq);
		}
		return sqrt(minLenSq);
	}
	
	int getFaces(void) const {return nFaces;}
	int getNodesPer(void) const {return nPer;}
	
	inline int getNode(int faceNo,int nodeNo) const {
		return idx[faceNo*nPer+nodeNo]-idxBase;
	}
	inline const CkVector3d &getNodeLoc(int nodeNo) const {return loc[nodeNo];}
	
	//Compute the average position of this face
	inline CkVector3d getFaceLoc(int faceNo) const { return faceCen[faceNo];}
	
	//Compute the face closest to this location
	// FIXME: this uses an O(nFaces) algorithm; using boxes it could be made O(1)
	int getLocFace(const CkVector3d &loc,double minTol) const {
		double minTolSq=minTol*minTol;
		int min=-1;
		for (int f=0;f<nFaces;f++) {
			double distSq=loc.distSqr(faceCen[f]);
			if (distSq<minTolSq) {
				//return f;
				if (min!=-1) 
					CkAbort("FEM_Linear_Periodicity> Several 'closest' faces found!");
				min=f;
			}
		}
		if (min==-1)
			CkAbort("FEM_Linear_Periodicity> No matching face found!");
		return min;
	}
	//Compute the node on this face closest to this location
	// This is O(nPer), but nPer is tiny (<10) so it's quite fast enough.
	int getLocNode(int faceNo,const CkVector3d &loc,double minTol) const {
		double minTolSq=minTol*minTol;
		int min=-1;
		for (int n=0;n<nPer;n++) {
			double distSq=loc.distSqr(getNodeLoc(getNode(faceNo,n)));
			if (distSq<minTolSq) {
				//return n;
				if (min!=-1) 
					CkAbort("FEM_Linear_Periodicity> Several 'closest' nodes found!");
				min=n;
			}
		}
		if (min==-1)
			CkAbort("FEM_Linear_Periodicity> No matching node found!");
		return getNode(faceNo,min);
	}
};

class matchingDest {
public:
	virtual void facesIdentical(int fa,int fb) =0;
	virtual void nodesIdentical(int na,int nb) =0;
};

class linearOffsetMatcher {
	faceSet a; //Set of "A" faces
	faceSet b; //Set of "B" faces
	int nNodes;
	int nFaces;
	double minTol; //Matching tolerance
	CkVector3d a2b_del; //Add this to "A" coordinates to get "B" coordinates
	//Map an "A" position into a "B" position
	CkVector3d a2b(const CkVector3d &a_pos) const {
		return a_pos+a2b_del;
	}
public:
	linearOffsetMatcher(
		int nFaces_,int nPer,const int *facesA,const int *facesB,int idxBase,
		int nNodes_,const CkVector3d *nodeLocs);
	void match(matchingDest &dest);
	CkVector3d getA2B(void) const {return a2b_del;}
};

linearOffsetMatcher::linearOffsetMatcher(
	int nFaces_,int nPer,const int *facesA,const int *facesB,int idxBase,
	int nNodes_,const CkVector3d *nodeLocs):
	a(nFaces_,nPer,facesA,idxBase,nodeLocs),
	b(nFaces_,nPer,facesB,idxBase,nodeLocs),
	nNodes(nNodes_),nFaces(nFaces_)
{
	//Compute the offset from A to B by taking the difference of their centroids:
	CkVector3d a_sum(0.0), b_sum(0.0);
	for (int f=0;f<nFaces;f++) {
		a_sum+=a.getFaceLoc(f); 
		b_sum+=b.getFaceLoc(f);
	}
	a2b_del=(b_sum-a_sum)*(1.0/nFaces);
	
	//Derive the matching tolerance from the edge length (for either face set)
	minTol=a.getMinEdgeLength()*0.001;
}

void linearOffsetMatcher::match(matchingDest &dest) {
	int nPer=a.getNodesPer();
	//FIXME: getLocFace is O(nFaces), so this loop is O(nFaces^2)!
	//  The only thing that saves us is that nFaces is small--
	//  for nFaces=10000, this loop takes less than 10 seconds.
	CkPrintf("FEM_Add_Linear_Periodic> Performing O(n^2) face matching loop (n=%d), (tol=%.1g)\n",nFaces,minTol);
	for (int fa=0;fa<nFaces;fa++) {
		CkVector3d bfaceCen=a2b(a.getFaceLoc(fa));
		int fb=b.getLocFace(bfaceCen,minTol); //<- the slow step
		dest.facesIdentical(fa,fb);
		for (int i=0;i<nPer;i++) {
			int na=a.getNode(fa,i);
			if (na!=-1) {
			  int nb=b.getLocNode(fb,a2b(a.getNodeLoc(na)),minTol);
			  dest.nodesIdentical(na,nb);
			}
		}
	}
	CkPrintf("FEM_Add_Linear_Periodic> Faces matched\n");
}

class unionFindDest : public matchingDest {
	unionFind &dest;
public:
	unionFindDest(unionFind &dest_) :dest(dest_) {}
	virtual void facesIdentical(int fa,int fb) { /*don't care*/ }
	virtual void nodesIdentical(int na,int nb) {
		dest.combine(na,nb);
	}
};

/*************
Symmetry representation and API
*/

//Describes all symmetries of the mesh
class FEM_Initial_Symmetries
{
	int nNodes;
public:
	ArrayPtrT<FEM_Symmetries_t> nodeSymmetries; //Symmetries each node belongs to
	intArrayPtr nodeCanon; //Map global node to canonical number
	unionFind *find; //Alternate representation for nodeCanon
	
	FEM_Sym_List symList;
	
	//Make sure nodeSymmetries is allocated
	void alloc(int nNodes_) {
		nNodes=nNodes_;
		if (nodeSymmetries==NULL) {
			nodeSymmetries=new FEM_Symmetries_t[nNodes];
			for (int i=0;i<nNodes;i++) 
				nodeSymmetries[i]=(FEM_Symmetries_t)0;
		}
	}
	int nodeCheck(int n) {
		if (n<-1 || n>=nNodes) return -2;
		return n;
	}
	
	FEM_Initial_Symmetries() {
		find=NULL;
	}
	~FEM_Initial_Symmetries() {
	    delete find;
	}
};

void FEM_Partition::setSymmetries(int nNodes_,int *new_can,const int *sym_src)
{
	if (sym!=NULL) CkAbort("Cannot call FEM_Set_Symmetries after adding other symmetries!");
	sym=new FEM_Initial_Symmetries;
	sym->nodeCanon=new_can;
	sym->alloc(nNodes_);
	for (int i=0;i<nNodes_;i++) 
		sym->nodeSymmetries[i]=(FEM_Symmetries_t)sym_src[i];
}
void FEM_Partition::addLinearPeriodic(int nFaces,int nPer,
	const int *facesA,const int *facesB,int idxBase,
	int nNodes,const CkVector3d *nodeLocs)
{
	if (sym==NULL) sym=new FEM_Initial_Symmetries;
	sym->alloc(nNodes);
	if (sym->find==NULL) sym->find=new unionFind(nNodes);
	
	//Figure out how the faces differ:
	linearOffsetMatcher matcher(nFaces,nPer,facesA,facesB,idxBase,nNodes,nodeLocs);
	CkVector3d a2b=matcher.getA2B();
	
	//Update the canon array by matching individual nodes:
	unionFindDest dest(*(sym->find));
	matcher.match(dest);
	
	//Mark nodeSymmetries for the nodes of both faces
	FEM_Symmetries_t sa=sym->symList.add(new FEM_Sym_Linear(-a2b));
	FEM_Symmetries_t sb=sym->symList.add(new FEM_Sym_Linear(a2b));
	for (int f=0;f<nFaces;f++) 
	for (int i=0;i<nPer;i++) {
		sym->nodeSymmetries[facesA[f*nPer+i]] |= sa;
		sym->nodeSymmetries[facesB[f*nPer+i]] |= sb;
	}
}

const int *FEM_Partition::getCanon(void) const {
	if (sym==NULL) return NULL;
	if (sym->nodeCanon==NULL && sym->find!=NULL) 
	{ //Need to transfer the canon array out of unionFind:
		sym->find->compress();
		sym->nodeCanon=sym->find->detach();
	}
	return sym->nodeCanon; //<- may be NULL, too
}
const FEM_Symmetries_t *FEM_Partition::getSymmetries(void) const {
	if (sym==NULL) return NULL;
	return sym->nodeSymmetries; //<- may be NULL, too
}
const FEM_Sym_List &FEM_Partition::getSymList(void) const {
	if (sym==NULL) {
		const static FEM_Sym_List emptyList;
		return emptyList;
	}
	else return sym->symList;
}

/* Runtime */

//Declaring these here prevents code bloat
FEM_Sym_List::FEM_Sym_List() {}
void FEM_Sym_List::operator=(const FEM_Sym_List &src) {
	for (int i=0;i<src.sym.size();i++)
		sym.push_back((FEM_Sym_Desc *)src.sym[i]->clone());
}
void FEM_Sym_List::pup(PUP::er &p) {
	p|sym;
}
FEM_Sym_List::~FEM_Sym_List() {}

//Add a new kind of symmetry to this list, returning
// the way objects with that symmetry should be marked.
FEM_Symmetries_t FEM_Sym_List::add(FEM_Sym_Desc *desc)
{
	int nSym=sym.size();
	sym.push_back(desc);
	return (FEM_Symmetries_t)(1<<nSym);
}

//Apply all the listed symmetries to this location
void FEM_Sym_List::applyLoc(CkVector3d *loc,FEM_Symmetries_t symToApply) const
{
	int nSym=sym.size();
	for (int i=0;i<nSym;i++)
		if (symToApply&(1<<i))
			*loc=sym[i]->applyLoc(*loc);
}

//Apply all the listed symmetries to this relative vector
void FEM_Sym_List::applyVec(CkVector3d *vec,FEM_Symmetries_t symToApply) const
{
	int nSym=sym.size();
	for (int i=0;i<nSym;i++)
		if (symToApply&(1<<i))
			*vec=sym[i]->applyVec(*vec);
}

FEM_Sym_Desc::~FEM_Sym_Desc() {}

void FEM_Sym_Linear::pup(PUP::er &p) {
	typedef PUP::able PUP_able;
	PUP_able::pup(p);
	p|shift;
}

/********** High-Level (Faces & coordinates) API **********/
CDECL void FEM_Add_linear_periodicity(
	int nFaces,int nPer,
	const int *facesA,const int *facesB,
	int nNodes,const double *nodeLocs
	)
{
	FEMAPI("FEM_Add_Linear_Periodicity");
	FEM_curPartition().addLinearPeriodic(nFaces,nPer,
		facesA,facesB,0,nNodes,(const CkVector3d *)nodeLocs);
}
FDECL void FTN_NAME(FEM_ADD_LINEAR_PERIODICITY,fem_add_linear_periodicity)(
	int *nFaces,int *nPer,
	const int *facesA,const int *facesB,
	int *nNodes,const double *nodeLocs
	)
{
	FEMAPI("fem_add_linear_periodicity");
	FEM_curPartition().addLinearPeriodic(*nFaces,*nPer,
		facesA,facesB,1,*nNodes,(const CkVector3d *)nodeLocs);
}

CDECL void FEM_Sym_coordinates(int elType,double *d_locs)
{
	const char *caller="FEM_Sym_coordinates"; FEMAPI(caller);
	
	const FEM_Mesh *m=FEM_chunk::get(caller)->getMesh(caller);
	const FEM_Entity &real_c=m->getCount(elType);
	const FEM_Entity &c=real_c.getGhost()[0];
	const FEM_Symmetries_t *sym=c.getSymmetries();
	if (sym==NULL) return; //Nothing to do-- no symmetries apply
	CkVector3d *locs=real_c.size()+(CkVector3d *)d_locs;
	int n=c.size();
	const FEM_Sym_List &sl=m->getSymList();
	for (int i=0;i<n;i++) 
		if (sym[i]!=(FEM_Symmetries_t)0)
			sl.applyLoc(&locs[i],sym[i]);
}
FDECL void FTN_NAME(FEM_SYM_COORDINATES,fem_sym_coordinates)
	(int *elType,double *locs)
{
	FEM_Sym_coordinates(zeroToMinusOne(*elType),locs);
}


/********** Low-Level (canonicalization array) API **********/

CDECL void FEM_Set_sym_nodes(const int *canon,const int *sym)
{
	const char *caller="FEM_Set_sym_nodes"; FEMAPI(caller);
	int n=FEM_chunk::get(caller)->setMesh(caller)->node.size();
	FEM_curPartition().setSymmetries(n,CkCopyArray(canon,n,0),sym);
}
FDECL void FTN_NAME(FEM_SET_SYM_NODES,fem_set_sym_nodes)
	(const int *canon,const int *sym)
{
	const char *caller="FEM_Set_sym_nodes"; FEMAPI(caller);
	int n=FEM_chunk::get(caller)->setMesh(caller)->node.size();
	FEM_curPartition().setSymmetries(n,CkCopyArray(canon,n,1),sym);
}

/*******************************************************************/
#if STANDALONE
//Standalone matching code test (not for inclusion in library)

  const int nFaces=10000;
  const int nPer=4;
  const int aPoints=10000; //Possible points on A faces
  const int bPoints=aPoints; //Possible points on B faces
  const int a2bPoints=aPoints;


static CrnStream rs;

void print(const char *str) {
	printf("%s",str);
}
void print(const CkVector3d &src,int digits=3) {
	printf("(%.*f,%.*f,%.*f) ",digits,src.x,digits,src.y,digits,src.z);
	FEM_Print("");
}

//Return random number on [0,max]
double randVal(double max) {
	return max*CrnDouble(&rs);
}
CkVector3d randVec(const CkVector3d &scale) {
	return CkVector3d(randVal(scale.x),randVal(scale.y),randVal(scale.z));
}
//Return random integer on [0,max)
int randNo(int max) {
	return (int)(randVal(max-0.000001));
}

class verbosematchingDest : public matchingDest {
public:
	virtual void facesIdentical(int fa,int fb) {
		if (fa!=fb)
			CkPrintf("--------- ERROR! Face %d %d\n",fa,fb);
	}
	virtual void nodesIdentical(int na,int nb) {
		if (na+a2bPoints!=nb)
			CkPrintf("--------- ERROR! Node %d %d\n",na,nb);
	}
};

//Fabricate some test faces (random subset of points)
void makeFaces(int *facesA,int *facesB) {
  for (int f=0;f<nFaces;f++) {
    int *fa=&facesA[f*nPer];
    int *fb=&facesB[f*nPer];
    
    bool doSkip=true;
    for (int i=0;i<nPer;i++) {
      int na=randNo(aPoints);
      bool repeat=true; //Did we hit a repeated point?
      do {
        repeat=false;
	for (int j=0;j<i;j++)
	  if (na==fa[j]) { //It's a repetition-- try again
	    na=randNo(aPoints);
	    repeat=true;
	    break;
	  }
      } while (repeat); 
      if (doSkip && randVal(1.0)<0.1) {
        fa[i]=fb[i]=-1; //Skip this node
	doSkip=false;
      }
      else {
        fa[i]=na;
        fb[i]=na+a2bPoints;
      }
    }
  }
}

void testUnion(int n,int *parent) {
	for (int i=0;i<n;i++) {
		if (parent[i]!=i) 
		{ //An actual match-- check it:
			if ((parent[i]+a2bPoints!=i) && (i+a2bPoints!=parent[i]))
				CkAbort("Union/Find mismatch!");
		}
	}
	delete[] parent;
}

int main(int argc,char *argv[])
{
  CkPrintf("init called\n");
  CrnInitStream(&rs,0,0);
  int facesA[nFaces*nPer],facesB[nFaces*nPer];
  const int nPoints=aPoints+bPoints;
  CkVector3d pts[nPoints];
  FEM_Print("Fabricating points:");
  CkVector3d a2bLocs(randVec(CkVector3d(10.0,90.0,50.0)));
  print("true offset vector "); print(a2bLocs,15); 
  int i;
  //Fabricate some test points (just purely random)
  for (i=0;i<aPoints;i++) {
    pts[i]=randVec(CkVector3d(5.0,5.0,5.0));
    pts[i+a2bPoints]=pts[i]+a2bLocs;
  }
  makeFaces(facesA,facesB);
  
  //Actually do the match:
  //verbosematchingDest dest;
  unionFind uf(nPoints); unionFindDest dest(uf);
  FEM_Print("Finding offset");
  linearOffsetMatcher matcher(nFaces,nPer,facesA,facesB,0, nPoints,pts);
  FEM_Print("Beginning match");
  matcher.match(dest);
  FEM_Print("Compressing paths");
  uf.compress();
  FEM_Print("Testing union");
  testUnion(nPoints,uf.detach());
  FEM_Print("Done");
}

#endif
