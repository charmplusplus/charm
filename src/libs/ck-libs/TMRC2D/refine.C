// Simple sample implementation of refinement interface.
// Orion Sky Lawlor, olawlor@acm.org, 4/9/2002
// Modified by Terry Wilmarth, wilmarth@cse.uiuc.edu, 4/16/2002
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include "charm++.h"
#include "charm-api.h"
#include "tcharm.h"
#include "mpi.h"
#include "tri.h"
#include "refine.h"

/********************* Attach *****************/
CDECL void REFINE2D_Init(void) {
  TCHARM_API_TRACE("REFINE2D_Init", "refine");
  TCharm *tc=TCharm::get();
  
  // Create the refinement array
  MPI_Comm comm=MPI_COMM_WORLD; /* fixme: make user pass in communicator */
  int rank; MPI_Comm_rank(comm,&rank);
  CkArrayID refArrayID;
  if (rank==0) 
  { /* Master creates the (empty) array and broadcasts arrayID: */
    CkArrayOptions opts;
    opts.bindTo(tc->getProxy());
    refArrayID=CProxy_chunk::ckNew(new chunkMsg, opts);
  }
  MPI_Bcast(&refArrayID,sizeof(refArrayID),MPI_BYTE, 0,comm);
  mesh=refArrayID; /* set up readonly proxy */
  
  // Now everybody inserts their element into the array
  chunkMsg *cm = new chunkMsg;
  cm->nChunks = tc->getNumElements();
  cm->myThreads = tc->getProxy();
  mesh[rank].insert(cm);
  tc->suspend(); /* will resume from chunk constructor */
}
FDECL void FTN_NAME(REFINE2D_INIT,refine2d_init)(void)
{
  REFINE2D_Init();
}

/******************** NewMesh *******************/
CDECL void REFINE2D_NewMesh(int meshID,int nEl,int nGhost,int nnodes,const int *conn,const int *gid,const int *boundaries, const int *edgeBounds, const int *edgeConn, int nEdges)
{
  TCHARM_API_TRACE("REFINE2D_NewMesh", "refine");
  if (!CtvAccess(_refineChunk))
    CkAbort("Forgot to call REFINE_Attach!!\n");
	
  CtvAccess(_refineChunk)->newMesh(meshID, nEl, nGhost, conn, gid, nnodes, 
				   boundaries, nEdges, edgeConn, edgeBounds, 0);
  MPI_Barrier(MPI_COMM_WORLD);
  CkWaitQD();
}
/*
FDECL void FTN_NAME(REFINE2D_NEWMESH,refine2d_newmesh)
(int *nEl,int *nGhost,int nnodes,const int *conn,const int *gid,const int *boundaries, const int **edgeBoundaries)
{
  TCHARM_API_TRACE("REFINE2D_NewMesh", "refine");
  if (!CtvAccess(_refineChunk))
    CkAbort("Forgot to call REFINE_Attach!!\n"); 
  
  CtvAccess(_refineChunk)->newMesh(*nEl, *nGhost,conn, gid, nnodes, 
                                   boundaries, edgeBoundaries, 1);
  MPI_Barrier(MPI_COMM_WORLD);
  CkWaitQD();
}*/

/********************** Splitting ******************/
class refineResults {
  std::vector<refineData> res;
  int nResults;
  
  //Return anything on [0,2] than a or b
  int otherThan(int a,int b) {
    if (a==b) CkAbort("Opposite node is moving!");
    for (int i=0;i<3;i++)
      if (i!=a && i!=b) return i;
    CkAbort("Logic error in refine.C::otherThan");
    return -1;
  }
public:
  refineResults(void) {nResults=0;}
	refineData createRefineData(int tri, int A, int B, int C,  
		int D, int _new, double frac,int flag, int origEdgeB, int newEdge1B, 
		int newEdge2B){
		refineData d;
		d.tri = tri;
		d.A = A;
		d.B = B;
		d.C = C;
		d.D = D;
		d._new = _new;
		d.frac = frac;
		d.flag = flag;
		d.origEdgeB = origEdgeB;
		d.newEdge1B = newEdge1B;
		d.newEdge2B = newEdge2B;
		return d;
	}
	void add(refineData &d){
		nResults++;
		res.push_back(d);
	};
	
  int countResults(void) const {return nResults;}
	
  void extract(int i, refineData *d) {
		*d =  res[i];	
  }
};
void FEM_Modify_IDXL(FEM_Refine_Operation_Data *data,refineData &d);

class resultsRefineClient : public refineClient {
  refineResults *res;
	FEM_Refine_Operation_Data *data;
public:
  resultsRefineClient(refineResults *res_,FEM_Refine_Operation_Data *data_) :res(res_),data(data_) {}
 /* void split(int tri, int side, int node, double frac) {
#if 0
    //Convert from "tri.C edges" to sensible edges
    if (side==1) side=2;
    else if (side==2) side=1;
#endif
    res->add(tri, side, node, frac);
  }*/
	//for the explanation of A,B,C,D look at diagram in refine.h
  void split(int tri, int A, int B, int C,  
	     int D, int _new, double frac,int flag, int origEdgeB, int newEdge1B, 
	     int newEdge2B) {
   	refineData d = res->createRefineData(tri,A,B,C,D, _new,frac,flag,origEdgeB,
		newEdge1B,newEdge2B);
		
		FEM_Modify_IDXL(data,d);

		res->add(d);
  }
};

class coarsenResults {
  // coarsenData is defined in refine.h
  std::vector<coarsenData> res;
public:
  coarsenResults(){}
  coarsenData addCollapse(int elementID, int nodeToKeep, int nodeToDelete,
			  double nX, double nY, int flag, int boundFlag, 
			  double frac)
  {
    coarsenData d;
    d.type = COLLAPSE;
    d.data.cdata.elemID = elementID;
    d.data.cdata.nodeToKeep = nodeToKeep;
    d.data.cdata.nodeToDelete = nodeToDelete;
    d.data.cdata.newX = nX;
    d.data.cdata.newY = nY;
    d.data.cdata.flag = flag;
    d.data.cdata.boundaryFlag = boundFlag;
    d.data.cdata.frac = frac;
    return d;
  };
  coarsenData addUpdate(int nodeID, double newX, double newY, int boundaryFlag)
  {
    coarsenData d;
    d.type = UPDATE;
    d.data.udata.nodeID = nodeID;
    d.data.udata.newX = newX;
    d.data.udata.newY = newY;
    d.data.udata.boundaryFlag = boundaryFlag;
    //		res.push_back(d);
    return d;
  };
  coarsenData addReplaceDelete(int elemID, int relnodeID, int oldNodeID,
			       int newNodeID)
  {
    coarsenData d;
    d.type = REPLACE;
    d.data.rddata.elemID = elemID;
    d.data.rddata.relnodeID = relnodeID;
    d.data.rddata.oldNodeID = oldNodeID;
    d.data.rddata.newNodeID = newNodeID;
    //		res.push_back(d);
    return d;
  };
  int countResults(){return res.size();}
  /*void extract(int i,int *conn,int *tri,int *nodeToThrow,int *nodeToKeep,double *nx,double *ny,int *flag,int idxbase){
    int t;
    t = res[i].elemID;
    *tri = t +idxbase;
    *nodeToKeep = conn[3*t+res[i].nodeToKeep];
    int n1 = res[i].collapseEdge;
    int n2 = (res[i].collapseEdge+1)%3;
    if(res[i].nodeToKeep == n1){
    *nodeToThrow = conn[3*t+n2];
    }else{
    *nodeToThrow = conn[3*t+n1];
    }
    *nx = res[i].nx;
    *ny = res[i].ny;
    *flag = res[i].flag;
    }*/
  void extract(int i, coarsenData *output){
    *output = res[i];
  }
};

// Modifying the code so that instead of being stored the results are
// processed immediately. The coarsen client is used simply to create
// and return an appropriate coarsenData structure.
class FEM_Operation_Data;
void FEM_Coarsen_Operation(FEM_Operation_Data *coarsen_data, 
			   coarsenData &operation);

class resultsCoarsenClient : public refineClient {
  coarsenResults *res;
  FEM_Operation_Data *data;
public:
  resultsCoarsenClient(coarsenResults *res_, FEM_Operation_Data *data_=NULL) 
    : res(res_),data(data_){};
  void collapse(int elementID, int nodeToKeep, int nodeToDelete, double nX,
		double nY, int flag, int b, double frac)
  {
    coarsenData d = res->addCollapse(elementID, nodeToKeep, nodeToDelete, nX,
				     nY, flag, b, frac);
    FEM_Coarsen_Operation(data,d);
  }
  void nodeUpdate(int nodeID, double newX, double newY, int boundaryFlag)
  {
    coarsenData d = res->addUpdate(nodeID,newX,newY,boundaryFlag);
    FEM_Coarsen_Operation(data,d);
  }
  void nodeReplaceDelete(int elementID, int relnodeID, int oldNodeID, 
			 int newNodeID)
  {
    coarsenData d = res->addReplaceDelete(elementID, relnodeID, oldNodeID,
					  newNodeID);
    FEM_Coarsen_Operation(data,d);
  }
};


// this function should be called from a thread
CDECL void REFINE2D_Split(int nNode,double *coord,int nEl,double *desiredArea,FEM_Refine_Operation_Data *refine_data)
{
  TCHARM_API_TRACE("REFINE2D_Split", "refine");
  chunk *C = CtvAccess(_refineChunk);
  if (!C)
    CkAbort("REFINE2D_Split failed> Did you forget to call REFINE2D_Attach?");
  C->refineResultsStorage=new refineResults;
  resultsRefineClient client(C->refineResultsStorage,refine_data);

  C->updateNodeCoords(nNode, coord, nEl);
  C->multipleRefine(desiredArea, &client);
}

CDECL void REFINE2D_Coarsen(int nNode, double *coord, int nEl,
			    double *desiredArea, FEM_Operation_Data *data)
{
  TCHARM_API_TRACE("REFINE2D_Coarsen", "coarsen");
  chunk *C = CtvAccess(_refineChunk);
  if (!C)
    CkAbort("REFINE2D_Split failed> Did you forget to call REFINE2D_Attach?");
  C->coarsenResultsStorage=new coarsenResults;
  resultsCoarsenClient client(C->coarsenResultsStorage,data);

  C->updateNodeCoords(nNode, coord, nEl);
  C->multipleCoarsen(desiredArea, &client);
}


FDECL void FTN_NAME(REFINE2D_SPLIT,refine2d_split)
   (int *nNode,double *coord,int *nEl,double *desiredArea,FEM_Refine_Operation_Data *data)
{
  REFINE2D_Split(*nNode,coord,*nEl,desiredArea,data);
}

static refineResults *getResults(void) {
  chunk *C = CtvAccess(_refineChunk);
  if (!C)
    CkAbort("Did you forget to call REFINE2D_Init?");
  refineResults *ret=C->refineResultsStorage;
  if (ret==NULL)
    CkAbort("Did you forget to call REFINE2D_Begin?");
  return ret;
}

CDECL int REFINE2D_Get_Split_Length(void)
{
  TCHARM_API_TRACE("REFINE2D_Get_Split_Length", "refine");
  return getResults()->countResults();
}
FDECL int FTN_NAME(REFINE2D_GET_SPLIT_LENGTH,refine2d_get_split_length)(void)
{
  return REFINE2D_Get_Split_Length();
}

CDECL void REFINE2D_Get_Split
    (int splitNo,refineData *d)
{
  TCHARM_API_TRACE("REFINE2D_Get_Split", "refine");
  refineResults *r=getResults();
  r->extract(splitNo,d);
}
FDECL void FTN_NAME(REFINE2D_GET_SPLIT,refine2d_get_split)
    (int *splitNo,refineData *d)
{
  TCHARM_API_TRACE("REFINE2D_Get_Split", "refine");
  refineResults *r=getResults();
  r->extract(*splitNo-1,d);
}

static coarsenResults *getCoarsenResults(void) {
  chunk *C = CtvAccess(_refineChunk);
  if (!C)
    CkAbort("Did you forget to call REFINE2D_Init?");
  coarsenResults *ret=C->coarsenResultsStorage;
  if (ret==NULL)
    CkAbort("Did you forget to call REFINE2D_Coarsen?");
  return ret;
}


CDECL int REFINE2D_Get_Collapse_Length(){
  return getCoarsenResults()->countResults();
}
/*
CDECL void REFINE2D_Get_Collapse(int i,int *conn,int *tri,int *nodeToThrow,int *nodeToKeep,double *nx,double *ny,int *flag,int idxbase){
	return getCoarsenResults()->extract(i,conn,tri,nodeToThrow,nodeToKeep,nx,ny,flag,idxbase);
}*/

CDECL void REFINE2D_Get_Collapse(int i,coarsenData *output){
	getCoarsenResults()->extract(i,output);
}

/********************* Check *****************/
static int checkElement(chunk *C,const element &e,const int *uc,int idxBase)
{
  int nMismatch=0;
  //Make sure the nodes are equal:
  /* This check no longer applies...
  for (int j=0;j<3;j++) {
    if (e.nodes[j].cid!=C->cid)
    	CkAbort("REFINE2D_Check> Triangle has non-local node!\n");
    if (uc[j]-idxBase!=e.nodes[j].idx) {
      //Very bad-- mismatching indices
      nMismatch++;
    }
  }
  */
  return nMismatch;
}

static void checkConn(int nEl,const int *conn,int idxBase,int nNode)
{
  chunk *C = CtvAccess(_refineChunk);
  if (!C) CkAbort("Did you forget to call REFINE2D_Attach?");
  //This check stolen from updateNodeCoords:
  if (nEl != C->numElements || nNode != C->numNodes) {
    CkPrintf("ERROR: inconsistency in REFINE2D_Check on chunk %d:\n"
       "  your nEl (%d); my numElements (%d)\n"
       "  your nNode (%d); my numNodes (%d)\n",
       C->cid, nEl, C->numElements, nNode,C->numNodes);
    CkAbort("User code/library numbering inconsistency in REFINE2D");
  }
  int i;
  int nErrs=0;
  for (i=0;i<nEl;i++) {
    const element &e=C->theElements[i];
    const int *uc=&conn[3*i];
    int elErrs=checkElement(C,e,uc,idxBase);
    nErrs+=elErrs;
    if (elErrs!=0) { //A bad element-- print out debugging info:
      /* This check is no longer valid...
	 CkError("REFINE2D Chunk %d>   Your triangle %d: %d %d %d\n"
	 "REFINE2D Chunk %d> but my triangle %d: %d %d %d\n",
	 C->cid+idxBase,i+idxBase, uc[0], uc[1], uc[2],
	 C->cid+idxBase,i+idxBase,
	 e.nodes[0].idx+idxBase,e.nodes[1].idx+idxBase,e.nodes[2].idx+idxBase);
      */
    }
  }
  if (nErrs!=0) {
    CkAbort("REFINE2D_Check> Major errors found. Exiting.");
  }
}

CDECL void REFINE2D_Check(int nEl,const int *conn,int nNodes) {
  checkConn(nEl,conn,0,nNodes);
}
FDECL void FTN_NAME(REFINE2D_CHECK,refine2d_check)
  (int *nEl,const int *conn,int *nNodes)
{
  checkConn(*nEl,conn,1,*nNodes);
}



