/*
Simple sample implementation of refinement interface.
Orion Sky Lawlor, olawlor@acm.org, 4/9/2002
Modified by Terry Wilmarth, wilmarth@cse.uiuc.edu, 4/16/2002
 */
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
CDECL void REFINE2D_NewMesh(int nEl,int nGhost,int nnodes,const int *conn,const int *gid,const int *boundaries)
{
  TCHARM_API_TRACE("REFINE2D_NewMesh", "refine");
  if (!CtvAccess(_refineChunk))
    CkAbort("Forgot to call REFINE_Attach!!\n");

/* FIXME: global barrier is a silly way to avoid early edge numbering messages */
  MPI_Barrier(MPI_COMM_WORLD);
  CtvAccess(_refineChunk)->newMesh(nEl,nGhost,conn, gid, nnodes, boundaries, 0);
  CkWaitQD(); //Wait for all edge numbering messages to finish
}
FDECL void FTN_NAME(REFINE2D_NEWMESH,refine2d_newmesh)
(int *nEl,int *nGhost,const int *conn,const int *gid)
{
  TCHARM_API_TRACE("REFINE2D_NewMesh", "refine");
  if (!CtvAccess(_refineChunk))
    CkAbort("Forgot to call REFINE_Attach!!\n"); 
  
  CtvAccess(_refineChunk)->newMesh(*nEl, *nGhost,conn, gid, 1);
  CkWaitQD(); //Wait for all edge numbering messages to finish
}

/********************** Splitting ******************/
class refineResults {
	int nResults;
	class resRec {
	public:
		int t,s,n;
		double f;
		int flag;
		resRec(int t_,int s_,int n_,double f_) 
			:t(t_), s(s_), n(n_), f(f_) {flag =0;}
		resRec(int t_,int s_,int n_,double f_,int flag_) 
			:t(t_), s(s_), n(n_), f(f_),flag(flag_) {}
	};
	std::vector<resRec> res;
	
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
	void add(int tri_,int side_,int n_,double frac_) {
		nResults++;
		res.push_back(resRec(tri_,side_,n_,frac_));
	}
	void add(int tri_,int side_,int n_,double frac_,int flag) {
		nResults++;
		res.push_back(resRec(tri_,side_,n_,frac_,flag));
	}

	int countResults(void) const {return nResults;}
	void extract(int i,const int *conn,int *triDest,int *A,int *B,int *C,double *fracDest,int idxBase,int *flags) {
		if ((i<0) || (i>=(int)res.size()))
			CkAbort("Invalid index in REFINE2D_Get_Splits");
		
		int tri=res[i].t;
		*triDest=tri+idxBase;
		int edgeOfTri=res[i].s;
		int movingNode=res[i].n;
		
		int c=(edgeOfTri+2)%3; //==opnode
		*A=conn[3*tri+movingNode]; //==othernode
		*B=conn[3*tri+otherThan(c,movingNode)];
		*C=conn[3*tri+c];
		*fracDest=res[i].f;
		*flags = res[i].flag;
		if (i==(int)res.size()-1) {
		  delete this;
		  chunk *C = CtvAccess(_refineChunk);
		  C->refineResultsStorage=NULL;
		}
	}
};


class resultsRefineClient : public refineClient {
  refineResults *res;
public:
  resultsRefineClient(refineResults *res_) :res(res_) {}
  void split(int tri, int side, int node, double frac) {
  #if 0
    //Convert from "tri.C edges" to sensible edges
    if (side==1) side=2;
    else if (side==2) side=1;
  #endif
    res->add(tri, side, node, frac);
  }
	void split(int tri, int side, int node, double frac,int flag) {
    res->add(tri, side, node, frac,flag);
  }

};

class coarsenResults {
	/*
		coarsenData is defined in refine.h
	*/
	std::vector<coarsenData> res;
	
public:
	coarsenResults(){}
  void addCollapse(int elementID,int nodeToKeep,int nodeToDelete,double nX,double nY,int flag){
		coarsenData d;
		d.type = COLLAPSE;
		d.data.cdata.elemID = elementID;
		d.data.cdata.nodeToKeep = nodeToKeep;
		d.data.cdata.nodeToDelete = nodeToDelete;
		d.data.cdata.newX = nX;
		d.data.cdata.newY = nY;
		d.data.cdata.flag = flag;
		res.push_back(d);
	};

	void addUpdate(int nodeID,double newX,double newY){
		coarsenData d;
		d.type = UPDATE;
		d.data.udata.nodeID = nodeID;
		d.data.udata.newX = newX;
		d.data.udata.newY = newY;
		res.push_back(d);
	};
	
	void addReplaceDelete(int elemID,int relnodeID,int oldNodeID,int newNodeID){
		coarsenData d;
		d.type = REPLACE;
		d.data.rddata.elemID = elemID;
		d.data.rddata.relnodeID = relnodeID;
		d.data.rddata.oldNodeID = oldNodeID;
		d.data.rddata.newNodeID = newNodeID;
		res.push_back(d);
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
	void extract(int i,coarsenData *output){
		*output = res[i];
	}
};


class resultsCoarsenClient : public refineClient {
  coarsenResults *res;
public:
  resultsCoarsenClient(coarsenResults *res_) : res(res_){};
  void collapse(int elementID,int nodeToKeep,int nodeToDelete,double nX,double nY,int flag){
		res->addCollapse(elementID,nodeToKeep,nodeToDelete,nX,nY,flag);
  }
	void nodeUpdate(int nodeID, double newX, double newY){
		res->addUpdate(nodeID,newX,newY);
	}
	void nodeReplaceDelete(int elementID, int relnodeID, int oldNodeID, int newNodeID){
		res->addReplaceDelete(elementID,relnodeID,oldNodeID,newNodeID);
	}
};


// this function should be called from a thread
CDECL void REFINE2D_Split(int nNode,double *coord,int nEl,double *desiredArea)
{
  TCHARM_API_TRACE("REFINE2D_Split", "refine");
  chunk *C = CtvAccess(_refineChunk);
  if (!C)
    CkAbort("REFINE2D_Split failed> Did you forget to call REFINE2D_Attach?");
  C->refineResultsStorage=new refineResults;
  resultsRefineClient client(C->refineResultsStorage);

  C->updateNodeCoords(nNode, coord, nEl);
  C->multipleRefine(desiredArea, &client);
  CkWaitQD();
}

CDECL void REFINE2D_Coarsen(int nNode,double *coord,int nEl,double *desiredArea)
{
  TCHARM_API_TRACE("REFINE2D_Coarsen", "coarsen");
  chunk *C = CtvAccess(_refineChunk);
  if (!C)
    CkAbort("REFINE2D_Split failed> Did you forget to call REFINE2D_Attach?");
  C->coarsenResultsStorage=new coarsenResults;
  resultsCoarsenClient client(C->coarsenResultsStorage);

  C->updateNodeCoords(nNode, coord, nEl);
  C->multipleCoarsen(desiredArea, &client);
  CkWaitQD();
}


FDECL void FTN_NAME(REFINE2D_SPLIT,refine2d_split)
   (int *nNode,double *coord,int *nEl,double *desiredArea)
{
  REFINE2D_Split(*nNode,coord,*nEl,desiredArea);
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
    (int splitNo,const int *conn,int *triDest,int *A,int *B,int *C,double *fracDest,int *flags)
{
  TCHARM_API_TRACE("REFINE2D_Get_Split", "refine");
  refineResults *r=getResults();
  r->extract(splitNo,conn,triDest,A,B,C,fracDest,0,flags);
}
FDECL void FTN_NAME(REFINE2D_GET_SPLIT,refine2d_get_split)
    (int *splitNo,const int *conn,int *triDest,int *A,int *B,int *C,double *fracDest, int *flags)
{
  TCHARM_API_TRACE("REFINE2D_Get_Split", "refine");
  refineResults *r=getResults();
  r->extract(*splitNo-1,conn,triDest,A,B,C,fracDest,1,flags);
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
  C->sanityCheck();
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
  for (i=0;i<nEl;i++) 
  {
    const element &e=C->theElements[i];
    const int *uc=&conn[3*i];
    int elErrs=checkElement(C,e,uc,idxBase);
    nErrs+=elErrs;
    if (elErrs!=0) 
    { //A bad element-- print out debugging info:
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



