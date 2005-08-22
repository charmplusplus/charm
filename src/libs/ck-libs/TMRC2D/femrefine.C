#include "ckvector3d.h"
#include "charm-api.h"
#include "refine.h"
#include "fem.h"
#include "fem_mesh.h"
#include "femrefine.h"
#include "ckhashtable.h"
#include <assert.h>


#define DEBUGINT(x) x
//#define DEBUGINT(x) 


class intdual{
private:
  int x,y;
public:
  intdual(int _x,int _y){
    if(_x <= _y){
      x = _x; y=_y;
    }else{
      x = _y; y= _x;
    }
  }
  inline int getx(){return x;};
  inline int gety(){return y;};
  inline CkHashCode hash() const {
    return (CkHashCode)(x+y);
  }
  static CkHashCode staticHash(const void *k,size_t){
    return ((intdual *)k)->hash();
  }
  inline int compare(intdual &t) const{
    return (t.getx() == x && t.gety() == y);
  }
  static int staticCompare(const void *a,const void *b,size_t){
    return ((intdual *)a)->compare((*(intdual *)b));
  }
};

void FEM_REFINE2D_Init(){
  REFINE2D_Init();	
}

FDECL void FTN_NAME(FEM_REFINE2D_INIT,fem_refine2d_init)(void)
{
  FEM_REFINE2D_Init();
}


void FEM_REFINE2D_Newmesh(int meshID,int nodeID,int elemID,int nodeBoundary){
  int nelems = FEM_Mesh_get_length(meshID,elemID);
  int nghost = FEM_Mesh_get_length(meshID,elemID+FEM_GHOST);
  int total = nghost + nelems;
  int *tempMesh = new int[3*total];
  int nnodes = FEM_Mesh_get_length(meshID,nodeID);
  int *tempBoundaries=NULL;
  if(nodeBoundary){
    tempBoundaries = new int[nnodes];
  }
  FEM_Mesh_data(meshID,elemID,FEM_CONN,&tempMesh[0],0,nelems,FEM_INDEX_0,3);
  FEM_Mesh_data(meshID,elemID+FEM_GHOST,FEM_CONN,&tempMesh[3*nelems],0,nghost,FEM_INDEX_0,3);
  
  for(int t=nelems;t<total;t++){
    for(int j=0;j<3;j++){
      if(FEM_Is_ghost_index(tempMesh[3*t+j])){
	tempMesh[3*t+j] += nelems;
      }
    }	
  }
  
  /*Set up the global ID's, for refinement*/
  int myID = FEM_My_partition();
  int *gid=new int[2*total];
  for (int i=0;i<nelems;i++) {
    gid[2*i+0]=myID; //Local element-- my chunk
    gid[2*i+1]=i; //Local number
  }
  int gid_fid=FEM_Create_field(FEM_INT,2,0,2*sizeof(int));
  FEM_Update_ghost_field(gid_fid,0,gid);
  
  if(nodeBoundary){
    FEM_Mesh_data(meshID,nodeID,FEM_BOUNDARY,tempBoundaries,0,nnodes,FEM_INT,1);
    printf("NODE BOUNDARIES-------------\n");
    for(int i=0;i<nnodes;i++){
      printf("%d %d \n",i,tempBoundaries[i]);
    }
    printf("------------------------\n");
  }
  
  /*Set up refinement framework*/
  /*FIX ME!  PASS IN EDGE BOUNDARIES! */
  const int *edgeConn = NULL;	
  const int *edgeBounds = NULL;
  int nEdges = 0;
  REFINE2D_NewMesh(meshID,nelems,total,nnodes,(int *)tempMesh,gid,tempBoundaries,edgeBounds, edgeConn, nEdges);
  if(tempBoundaries){
    delete [] tempBoundaries;
  }
  delete [] gid;
  delete [] tempMesh;
}

FDECL void FTN_NAME(FEM_REFINE2D_NEWMESH,fem_refine2d_newmesh)(int *meshID,int *nodeID,int *elemID, int *nodeBoundary)
{
  FEM_REFINE2D_Newmesh(*meshID,*nodeID,*elemID,*nodeBoundary);
}

class FEM_Refine_Operation_Data{
public:
  int meshID,nodeID;
  int cur_nodes;
  CkVec<double> *coordVec;
  double *coord;
  FEM_Entity *node;
  CkVec<FEM_Attribute *> *attrs;
  FEM_Entity *elem;
  CkVec<FEM_Attribute *> *elemattrs;
  int sparseID,elemID;
  CkHashtableT<intdual,int> *newnodes;
  CkHashtableT<intdual,int> *nodes2sparse;
  FEM_Attribute *sparseConnAttr, *sparseBoundaryAttr;
  int nSplits;
  AllocTable2d<int> *connTable;
  FEM_Entity *sparse;
  CkVec<FEM_Attribute *> *sparseattrs;
  AllocTable2d<int> *validEdge;
  FEM_Refine_Operation_Data(){};
};

void FEM_Refine_Operation(FEM_Refine_Operation_Data *data,refineData &d);
static int countValidEntities(int *validData,int total);

void FEM_REFINE2D_Split(int meshID,int nodeID,double *coord,int elemID,double *desiredAreas,int sparseID){
  int nnodes = FEM_Mesh_get_length(meshID,nodeID);
  int nelems = FEM_Mesh_get_length(meshID,elemID);
  int actual_nodes = nnodes, actual_elems = nelems;
  
  FEM_Refine_Operation_Data refine_data;
  refine_data.meshID = meshID;
  refine_data.nodeID = nodeID;
  refine_data.sparseID = sparseID;
  refine_data.elemID = elemID;
  refine_data.cur_nodes = FEM_Mesh_get_length(meshID,nodeID);
  
  /*Copy the cordinates of the nodes into a vector, 
    the cordinates of the new nodes will be inserted
    into this vector and will be used to sort all the
    nodes on the basis of the distance from origin
  */
  CkVec<double> coordVec;
  for(int i=0;i<nnodes*2;i++){
    coordVec.push_back(coord[i]);
  }
  refine_data.coordVec = &coordVec;
  refine_data.coord = coord;
  
  /*find out the attributes of the node */
  FEM_Entity *e=refine_data.node = FEM_Entity_lookup(meshID,nodeID,"REFINE2D_Mesh");
  CkVec<FEM_Attribute *> *attrs = refine_data.attrs = e->getAttrVec();
  /* 
  FEM_DataAttribute *boundaryAttr = (FEM_DataAttribute *)e->lookup(FEM_BOUNDARY,"split");
  if(boundaryAttr != NULL){
	AllocTable2d<int> &boundaryTable = boundaryAttr->getInt();
	printf(" Node Boundary flags \n");
	for(int i=0;i<nnodes;i++){
		printf("Node %d flag %d \n",i,boundaryTable[i][0]);
	}
  }
  */
  FEM_Entity *elem = refine_data.elem = FEM_Entity_lookup(meshID,elemID,"REFIN2D_Mesh_elem");
  CkVec<FEM_Attribute *> *elemattrs = refine_data.elemattrs = elem->getAttrVec();
  
  FEM_Attribute *connAttr = elem->lookup(FEM_CONN,"REFINE2D_Mesh");
  if(connAttr == NULL){
    CkAbort("Grrrr element without connectivity \n");
  }
  AllocTable2d<int> &connTable = ((FEM_IndexAttribute *)connAttr)->get();
  refine_data.connTable = &connTable;
  
  //hashtable to store the new node number as a function of the two old numbers
  CkHashtableT<intdual,int> newnodes(nnodes);
  refine_data.newnodes = &newnodes;
  
  /* Get the FEM_BOUNDARY data of sparse elements and load it into a hashtable
    indexed by the 2 node ids that make up the edge. The data in the hashtable
    is the index number of the sparse element */
  FEM_Entity *sparse;
  CkVec<FEM_Attribute *> *sparseattrs;
  FEM_Attribute *sparseConnAttr, *sparseBoundaryAttr;
  AllocTable2d<int> *sparseConnTable, *sparseBoundaryTable;
  CkHashtableT<intdual,int> nodes2sparse(nelems);
  refine_data.nodes2sparse = &nodes2sparse;
  
  if(sparseID != -1){
    sparse = refine_data.sparse = FEM_Entity_lookup(meshID,sparseID,"REFINE2D_Mesh_sparse");
    refine_data.sparseattrs = sparseattrs = sparse->getAttrVec();
    refine_data.sparseConnAttr = sparseConnAttr = sparse->lookup(FEM_CONN,"REFINE2D_Mesh_sparse");
    sparseConnTable = &(((FEM_IndexAttribute *)sparseConnAttr)->get());
    refine_data.sparseBoundaryAttr = sparseBoundaryAttr = sparse->lookup(FEM_BOUNDARY,"REFINE2D_Mesh_sparse");
    if(sparseBoundaryAttr == NULL){
      CkAbort("Specified sparse elements without boundary conditions");
    }
    FEM_DataAttribute *validEdgeAttribute = (FEM_DataAttribute *)sparse->lookup(FEM_VALID,"REFINE2D_Mesh_sparse");
    if(validEdgeAttribute){
      refine_data.validEdge = &(validEdgeAttribute->getInt());
    }else{
      refine_data.validEdge = NULL;
    }
    /* since the default value in the hashtable is 0, to distinguish between 
       uninserted keys and the sparse element with index 0, the index of the 
       sparse elements is incremented by 1 while inserting. */
    //		printf("[%d] Sparse elements\n",FEM_My_partition());
    for(int j=0;j<sparse->size();j++){
      if(refine_data.validEdge == NULL || (*(refine_data.validEdge))[j][0]){
	int *cdata = (*sparseConnTable)[j];
	//		printf("%d < %d,%d > \n",j,cdata[0],cdata[1]);
	nodes2sparse.put(intdual(cdata[0],cdata[1])) = j+1;
      }	
    }
  }else{
    printf("Edge boundary conditions not passed into FEM_REFINE2D_Split \n");
  }
  
  //count the actual number of nodes and elements
  if(refine_data.node->lookup(FEM_VALID,"refine2D_splilt") != NULL){
    AllocTable2d<int> &validNodeTable = ((FEM_DataAttribute *)(refine_data.node->lookup(FEM_VALID,"refine2D_splilt")))->getInt();
    actual_nodes = countValidEntities(validNodeTable.getData(),nnodes);
  }
  if(refine_data.elem->lookup(FEM_VALID,"refine2D_splilt") != NULL){
    AllocTable2d<int> &validElemTable = ((FEM_DataAttribute *)(refine_data.elem->lookup(FEM_VALID,"refine2D_splilt")))->getInt();
    actual_elems = countValidEntities(validElemTable.getData(),nelems);
  }
  
  DEBUGINT(printf("%d %d \n",nnodes,nelems));	
  REFINE2D_Split(actual_nodes,coord,actual_elems,desiredAreas,&refine_data);
  
  int nSplits= refine_data.nSplits = REFINE2D_Get_Split_Length();
  DEBUGINT(printf("called REFINE2D_Split nSplits = %d \n",nSplits));
  
  if(nSplits == 0){
    return;
  }
  
  for(int split = 0;split < nSplits;split++){
    refineData op;
    REFINE2D_Get_Split(split,&op);
    FEM_Refine_Operation(&refine_data,op);
  }
  
  DEBUGINT(printf("Cordinate list length %d according to FEM %d\n",coordVec.size()/2,FEM_Mesh_get_length(meshID,nodeID)));
  IDXL_Sort_2d(FEM_Comm_shared(meshID,nodeID),coordVec.getVec());
  int read = FEM_Mesh_is_get(meshID) ;
  assert(read);
}

extern void splitEntity(IDXL_Side &c,
	int localIdx,int nBetween,int *between,int idxbase);

void FEM_Modify_IDXL(FEM_Refine_Operation_Data *data,refineData &op){
  FEM_Node *node = (FEM_Node *)data->node;
  int between[2];
  between[0] = op.A;
  between[1] = op.B;
  splitEntity(node->shared,op.D,2,between,0);
};

void FEM_Refine_Operation(FEM_Refine_Operation_Data *data,refineData &op){
  
  int meshID = data->meshID;
  int nodeID = data->nodeID;
  int sparseID = data->sparseID;
  CkVec<FEM_Attribute *> *attrs = data->attrs;
  CkVec<FEM_Attribute *> *elemattrs = data->elemattrs;
  CkHashtableT<intdual,int> *newnodes=data->newnodes;
  CkHashtableT<intdual,int> *nodes2sparse = data->nodes2sparse;
  FEM_Attribute *sparseConnAttr = data->sparseConnAttr, *sparseBoundaryAttr = data->sparseBoundaryAttr;
  AllocTable2d<int> *connTable = data->connTable;
  double *coord = data->coord;
  AllocTable2d<int> *sparseConnTable, *sparseBoundaryTable;
  CkVec<FEM_Attribute *> *sparseattrs = data->sparseattrs;
  /*
  FEM_DataAttribute *boundaryAttr = (FEM_DataAttribute *)data->node->lookup(FEM_BOUNDARY,"split");
  if(boundaryAttr != NULL){
  AllocTable2d<int> &boundaryTable = boundaryAttr->getInt();
  printf(" Node Boundary flags \n");
  for(int i=0;i<data->node->size();i++){
  printf("Node %d flag %d \n",i,boundaryTable[i][0]);
  }
  }
  */
  
  int tri=op.tri,A=op.A,B=op.B,C=op.C,D=op.D;
  double frac=op.frac;
  // current number of nodes in the mesh
  int *connData = connTable->getData();
  int flags=op.flag;
    
  if((flags & 0x1) || (flags & 0x2)){
    //new node 
    DEBUGINT(CkPrintf("---- Adding node %d\n",D));			
    /*	lastA=A; lastB=B; lastD=D;*/
    if (A>=data->cur_nodes) CkAbort("Calculated A is invalid!");
    if (B>=data->cur_nodes) CkAbort("Calculated B is invalid!");
    if(D >= data->cur_nodes){
      data->node->setLength(D+1);
      data->cur_nodes = D+1;
    }	
    for(int i=0;i<attrs->size();i++){
      FEM_Attribute *a = (FEM_Attribute *)(*attrs)[i];
      if(a->getAttr()<FEM_ATTRIB_TAG_MAX){
	FEM_DataAttribute *d = (FEM_DataAttribute *)a;
	d->interpolate(A,B,D,frac);
      }else{
	/* Boundary value of new node D should be boundary value of the edge
	   (sparse element) that contains the two nodes A & B. */
	if(a->getAttr() == FEM_BOUNDARY){
	  if(sparseID != -1){
	    int sidx = nodes2sparse->get(intdual(A,B))-1;
	    if(sidx == -1){
	      CkAbort("no sparse element between these 2 nodes, are they really connected ??");
	    }
	    sparseBoundaryTable = &(((FEM_DataAttribute *)sparseBoundaryAttr)->getInt());
	    
	    int boundaryVal = ((*sparseBoundaryTable)[sidx])[0];
	    
	    (((FEM_DataAttribute *)a)->getInt()[D])[0] = boundaryVal;
	  }else{
	    /* if sparse elements don't exist, just do simple interpolation */
	    FEM_DataAttribute *d = (FEM_DataAttribute *)a;
	    d->interpolate(A,B,D,frac);
	  }
	}
      }	
    }
    int AandB[2];
    AandB[0]=A;
    AandB[1]=B;
    /* Add a new node D between A and B */
    IDXL_Add_entity(FEM_Comm_shared(meshID,nodeID),D,2,AandB);
    double Dx = coord[2*A]*(1-frac)+frac*coord[2*B];
    double Dy = coord[2*A+1]*(1-frac)+frac*coord[2*B+1];
    data->coordVec->push_back(Dx);
    data->coordVec->push_back(Dy);
    newnodes->put(intdual(A,B))=D;
    /* add the new sparse element <D,B> and modify the connectivity of the old
       one from <A,B> to <A,D> and change the hashtable to reflect that change
    */
    if(sparseID != -1){
      int oldsidx = nodes2sparse->get(intdual(A,B))-1;
      int newsidx = data->sparse->size();
      data->sparse->setLength(newsidx+1);
      for(int satt = 0;satt<sparseattrs->size();satt++){
	if((*sparseattrs)[satt]->getAttr() == FEM_CONN){
	  /* change the conn of the old sparse to A,D and new one to B,D */
	  sparseConnTable = &(((FEM_IndexAttribute *)sparseConnAttr)->get());
	  int *oldconn = (*sparseConnTable)[oldsidx];
	  int *newconn = (*sparseConnTable)[newsidx];
	  oldconn[0] = A;
	  oldconn[1] = D;
	  newconn[0] = D;
	  newconn[1] = B;
	  //printf("<%d,%d> edge being split into <%d,%d> <%d,%d> \n",A,B,A,D,D,B);
	}else{
	  /* apart from conn copy everything else */
	  FEM_Attribute *attr = (FEM_Attribute *)(*sparseattrs)[satt];
	  attr->copyEntity(newsidx,*attr,oldsidx);
	}
      }
      /* modify the hashtable - delete the old edge and the new ones */
      nodes2sparse->remove(intdual(A,B));
      nodes2sparse->put(intdual(A,D)) = oldsidx+1;
      nodes2sparse->put(intdual(D,B)) = newsidx+1;
    }
  }
  //add a new triangle
  /*TODO: replace  FEM_ELEM with parameter*/
  int newTri =	op._new;
  int cur_elements  = FEM_Mesh_get_length(data->meshID,data->elemID);
  DEBUGINT(CkPrintf("---- Adding triangle %d after splitting %d \n",newTri,tri));
  if(newTri >= cur_elements){
    data->elem->setLength(newTri+1);
  }	
  D = newnodes->get(intdual(A,B));
  
  for(int j=0;j<elemattrs->size();j++){
    if((*elemattrs)[j]->getAttr() == FEM_CONN){
      DEBUGINT(CkPrintf("elem attr conn code %d \n",(*elemattrs)[j]->getAttr()));
      //it is a connectivity attribute.. get the connectivity right
      FEM_IndexAttribute *connAttr = (FEM_IndexAttribute *)(*elemattrs)[j];
      AllocTable2d<int> &table = connAttr->get();
      DEBUGINT(CkPrintf("Table of connectivity attribute starts at %p width %d \n",table[0],connAttr->getWidth()));
      int *oldRow = table[tri];
      int *newRow = table[newTri];
      for (int i=0;i<3;i++){
	if (oldRow[i]==A){
	  oldRow[i]=D;	
	  DEBUGINT(CkPrintf("In triangle %d %d replaced by %d \n",tri,A,D));
	}
      }	
      for (int i=0; i<3; i++) {
	if (oldRow[i] == B){
	  newRow[i] = D;
	}	
	else if (oldRow[i] == C){
	  newRow[i] = C;
	}	
	else if (oldRow[i] == D){
	  newRow[i] = A;
	}	
      }
      DEBUGINT(CkPrintf("New Triangle %d  (%d %d %d) conn %p\n",newTri,newRow[0],newRow[1],newRow[2],newRow));
    }else{
      FEM_Attribute *elattr = (FEM_Attribute *)(*elemattrs)[j];
      if(elattr->getAttr() < FEM_ATTRIB_FIRST){ 
	elattr->copyEntity(newTri,*elattr,tri);
      }
    }
  }
  
  if(sparseID != -1){ /* add sparse element (edge between C and D) */
    int cdidx = data->sparse->size();
    data->sparse->setLength(cdidx+1);
    for(int satt = 0; satt < sparseattrs->size();satt++){
      if((*sparseattrs)[satt]->getAttr() == FEM_CONN){
	sparseConnTable = &(((FEM_IndexAttribute *)sparseConnAttr)->get());
	int *cdconn = (*sparseConnTable)[cdidx];
	cdconn[0]=C;
	cdconn[1]=D;
      }
      if((*sparseattrs)[satt]->getAttr() == FEM_BOUNDARY){
	/* An edge connecting C and D has to be an internal edge */
	sparseBoundaryTable = &(((FEM_DataAttribute *)sparseBoundaryAttr)->getInt());
	((*sparseBoundaryTable)[cdidx])[0] = 0;
      }
    }
    if(data->validEdge){
      (*(data->validEdge))[cdidx][0] = 1;
    }
    nodes2sparse->put(intdual(C,D)) = cdidx+1;
  }
}


FDECL void FTN_NAME(FEM_REFINE2D_SPLIT,fem_refine2d_split)(int *meshID,int *nodeID,double *coord,int *elemID,double *desiredAreas){
	FEM_REFINE2D_Split(*meshID,*nodeID,coord,*elemID,desiredAreas);
}

FDECL void FTN_NAME(FEM_REFINE2D_SPLIT_EDGE,fem_refine2d_split_edge)(int *meshID,int *nodeID,double *coord,int *elemID,double *desiredAreas,int *sparseID){
	FEM_REFINE2D_Split(*meshID,*nodeID,coord,*elemID,desiredAreas,*sparseID);
}

FDECL void FTN_NAME(CMIMEMORYCHECK,cmimemorycheck)(){
	CmiMemoryCheck();
}

class FEM_Operation_Data{
public:
  double *coord;
  int *connData,*nodeBoundaryData;
  int *validNodeData,*validElemData;
  FEM_Node *node;
  CkHashtableT<intdual,int> *nodes2sparse;
  AllocTable2d<int> *validEdge;
  AllocTable2d<int> *sparseConnTable;
  FEM_Attribute *sparseBoundaryAttr;
  FEM_Operation_Data(){
  };
};
void FEM_Coarsen_Operation(FEM_Operation_Data *coarsen_data, coarsenData &operation);

void FEM_REFINE2D_Coarsen(int meshID,int nodeID,double *coord,int elemID,double *desiredAreas,int sparseID){
  int nnodes = FEM_Mesh_get_length(meshID,nodeID);
  int nelems = FEM_Mesh_get_length(meshID,elemID);
  int nodeCount=0,elemCount=0;
  
  /* The attributes of the different entities */
  FEM_Entity *node=FEM_Entity_lookup(meshID,nodeID,"REFINE2D_Mesh");
  CkVec<FEM_Attribute *> *attrs = node->getAttrVec();
  FEM_Attribute *validNodeAttr = node->lookup(FEM_VALID,"FEM_COARSEN");
  FEM_Attribute *nodeBoundaryAttr = node->lookup(FEM_BOUNDARY,"FEM_COARSEN");
  if(!nodeBoundaryAttr){
    printf("Warning:- no boundary flags for nodes \n");
  }
  
  FEM_Entity *elem = FEM_Entity_lookup(meshID,elemID,"REFIN2D_Mesh_elem");
  CkVec<FEM_Attribute *> *elemattrs = elem->getAttrVec();
  FEM_Attribute *validElemAttr = elem->lookup(FEM_VALID,"FEM_COARSEN");
  
  FEM_Attribute *connAttr = elem->lookup(FEM_CONN,"REFINE2D_Mesh");
  if(connAttr == NULL){
    CkAbort("Grrrr element without connectivity \n");
  }
  AllocTable2d<int> &connTable = ((FEM_IndexAttribute *)connAttr)->get();
  AllocTable2d<int>&validNodeTable = ((FEM_DataAttribute *)validNodeAttr)->getInt();
  AllocTable2d<int>&validElemTable = ((FEM_DataAttribute *)validElemAttr)->getInt();
  FEM_Operation_Data *coarsen_data = new FEM_Operation_Data;
  coarsen_data->node = (FEM_Node *)node;
  coarsen_data->coord = coord;
  int *connData = coarsen_data->connData= connTable.getData();
  int *validNodeData = coarsen_data->validNodeData = validNodeTable.getData();
  int *validElemData = coarsen_data->validElemData = validElemTable.getData();
  /* Extract the data for node boundaries */
  int *nodeBoundaryData= coarsen_data->nodeBoundaryData =NULL;
  if(nodeBoundaryAttr){
    AllocTable2d<int> &nodeBoundaryTable = ((FEM_DataAttribute *)nodeBoundaryAttr)->getInt();
    nodeBoundaryData = coarsen_data->nodeBoundaryData = nodeBoundaryTable.getData();
  }
  
  for(int k=0;k<nnodes;k++){
    int valid = validNodeData[k];
    if(validNodeData[k]){
      nodeCount++;
    }
  }
  for(int k=0;k<nelems;k++){
    if(validElemData[k]){
      elemCount++;
    }
  }
  
  /* populate the hashtable from nodes2edges with the valid edges */
  CkHashtableT<intdual,int> nodes2sparse(nelems);
  coarsen_data->nodes2sparse = &nodes2sparse;
  FEM_Attribute *sparseBoundaryAttr;
  if(sparseID != -1){
    FEM_Entity *sparse = FEM_Entity_lookup(meshID,sparseID,"Coarsen_sparse");
    FEM_DataAttribute *validEdgeAttribute = (FEM_DataAttribute *)sparse->lookup(FEM_VALID,"Coarsen_sparse");
    FEM_IndexAttribute *sparseConnAttribute = (FEM_IndexAttribute *)sparse->lookup(FEM_CONN,"Coarsen_sparse");
    AllocTable2d<int> *sparseConnTable = coarsen_data->sparseConnTable = &(sparseConnAttribute->get());
    coarsen_data->sparseBoundaryAttr = sparseBoundaryAttr = sparse->lookup(FEM_BOUNDARY,"REFINE2D_Mesh_sparse");
    if(sparseBoundaryAttr == NULL){
      CkAbort("Specified sparse elements without boundary conditions");
    }

    
    if(validEdgeAttribute){
      coarsen_data->validEdge = &(validEdgeAttribute->getInt());
    }else{
      coarsen_data->validEdge = NULL;
    }
    /* since the default value in the hashtable is 0, to distinguish between 
       uninserted keys and the sparse element with index 0, the index of the 
       sparse elements is incremented by 1 while inserting. */
    printf("[%d] Sparse elements\n",FEM_My_partition());
    for(int j=0;j<sparse->size();j++){
      if(coarsen_data->validEdge == NULL || (*(coarsen_data->validEdge))[j][0]){
	int *cdata = (*sparseConnTable)[j];
	printf("%d < %d,%d > \n",j,cdata[0],cdata[1]);
	nodes2sparse.put(intdual(cdata[0],cdata[1])) = j+1;
      }	
    }
  }else{
    coarsen_data->validEdge = NULL;
  }
  
  DEBUGINT(printf("coarsen %d %d \n",nodeCount,elemCount));	
  REFINE2D_Coarsen(nodeCount,coord,elemCount,desiredAreas,coarsen_data);
  int nCollapses = REFINE2D_Get_Collapse_Length();
  
  
  for(int i=0;i<nnodes;i++){
    int valid = validNodeData[i];
  }
  delete coarsen_data;
}  

void interpolateNode(FEM_Node *node,int A,int B,int D,double frac);

void FEM_Coarsen_Operation(FEM_Operation_Data *coarsen_data, coarsenData &operation){
  double *coord = coarsen_data->coord;
  int tri,nodeToThrow,nodeToKeep,n1,n2;
  int *connData = coarsen_data->connData;
  int *validNodeData = coarsen_data->validNodeData;
  int *validElemData = coarsen_data->validElemData;
  int *nodeBoundaryData = coarsen_data->nodeBoundaryData;
  FEM_Attribute *sparseBoundaryAttr = coarsen_data->sparseBoundaryAttr;
  AllocTable2d<int> *sparseBoundaryTable;

  switch(operation.type){
  case COLLAPSE: 
    {
      tri = operation.data.cdata.elemID;
      nodeToKeep = operation.data.cdata.nodeToKeep;
      nodeToThrow = operation.data.cdata.nodeToDelete;
      int opNode = 0;
      for (int i=0; i<3; i++) {
	if ((connData[3*tri+i] != nodeToThrow) &&
	    (connData[3*tri+i] != nodeToKeep)) {
	  opNode = connData[3*tri+i];
	  break;
	}
      }
      CkPrintf("Collapse %d, nodeToKeep %d, nodeToThrow %d, opNode %d\n",
	       tri, nodeToKeep, nodeToThrow, opNode);
      sparseBoundaryTable = &(((FEM_DataAttribute *)sparseBoundaryAttr)->getInt());
      int delEdgeIdx = coarsen_data->nodes2sparse->get(intdual(nodeToThrow,opNode))-1;
      int keepEdgeIdx = coarsen_data->nodes2sparse->get(intdual(nodeToKeep,opNode))-1;
      int delBC = ((*sparseBoundaryTable)[delEdgeIdx])[0];
      int keepBC = ((*sparseBoundaryTable)[keepEdgeIdx])[0];
      if (delBC > keepBC) {
	((*sparseBoundaryTable)[keepEdgeIdx])[0] = delBC;
      }
      
      if(operation.data.cdata.flag & 0x1 || operation.data.cdata.flag & 0x2){
	interpolateNode(coarsen_data->node,nodeToKeep,nodeToThrow,nodeToKeep,operation.data.cdata.frac);
	coord[2*nodeToKeep] = operation.data.cdata.newX;
	coord[2*nodeToKeep+1] = operation.data.cdata.newY;
	validNodeData[nodeToThrow] = 0;
	validNodeData[nodeToKeep] = 1;
	DEBUGINT(printf("---------Collapse <%d,%d> invalidating node %d and element %d \n",nodeToKeep,nodeToThrow,nodeToThrow,tri));
	if(coarsen_data->validEdge){	
	  int sidx = coarsen_data->nodes2sparse->get(intdual(nodeToKeep,nodeToThrow))-1;
	  coarsen_data->nodes2sparse->remove(intdual(nodeToKeep,nodeToThrow));
	  (*(coarsen_data->validEdge))[sidx][0] = 0;
	  DEBUGINT(printf("---- Deleting edge %d between nodes %d and %d \n",sidx,nodeToKeep,nodeToThrow));
	  if (delEdgeIdx >=0) {
	    coarsen_data->nodes2sparse->remove(intdual(opNode,nodeToThrow));
	    (*(coarsen_data->validEdge))[delEdgeIdx][0] = 0;
	    DEBUGINT(printf("---- Deleting edge %d between nodes %d and %d \n",delEdgeIdx,opNode,nodeToThrow));
	  }
	}	
      }
      else {
	if (delEdgeIdx >=0) {
	  coarsen_data->nodes2sparse->remove(intdual(opNode,nodeToThrow));
	  (*(coarsen_data->validEdge))[delEdgeIdx][0] = 0;
	  DEBUGINT(printf("---- Deleting edge %d between nodes %d and %d \n",delEdgeIdx,opNode,nodeToThrow));
	}	
      }
      validElemData[tri] = 0;
      connData[3*tri] = connData[3*tri+1] = connData[3*tri+2] = -1;
    }
    break;
  case	UPDATE:
    if(validNodeData[operation.data.udata.nodeID]){
      coord[2*(operation.data.udata.nodeID)] = operation.data.udata.newX;
      coord[2*(operation.data.udata.nodeID)+1] = operation.data.udata.newY;
      if(nodeBoundaryData){
	nodeBoundaryData[operation.data.udata.nodeID]=operation.data.udata.boundaryFlag;
      }
    }else{
      DEBUGINT(printf("[%d] WEIRD -- update operation for invalid node %d \n",CkMyPe(),operation.data.udata.nodeID));
    }
    break;
  case REPLACE:
    if(validElemData[operation.data.rddata.elemID]){
      if(connData[3*operation.data.rddata.elemID+operation.data.rddata.relnodeID] == operation.data.rddata.oldNodeID){
	connData[3*operation.data.rddata.elemID+operation.data.rddata.relnodeID] = operation.data.rddata.newNodeID;
	if(validNodeData[operation.data.rddata.oldNodeID]){
	  validNodeData[operation.data.rddata.oldNodeID]=0;
	}
	if(coarsen_data->validEdge){
	  //remove the edges containing oldNodeID and add edges containing newNodeID in the nodes2sparse hashtable
	  //update the connectivity information of the edges
	  for(int i=0;i<3;i++){
	    //find the two nodes apart from the one being replaced
	    if(i != operation.data.rddata.relnodeID){
	      int otherNode = connData[3*operation.data.rddata.elemID+i];
	      int edgeIdx = coarsen_data->nodes2sparse->get(intdual(operation.data.rddata.oldNodeID,otherNode))-1;
	      if(edgeIdx >= 0){
		//The edge connectivity has not been updated on this processor
		coarsen_data->nodes2sparse->remove(intdual(operation.data.rddata.oldNodeID,otherNode));
		DEBUGINT(printf("---- Deleting edge %d between nodes %d and %d \n",edgeIdx,operation.data.rddata.oldNodeID,otherNode));
		coarsen_data->nodes2sparse->put(intdual(operation.data.rddata.newNodeID,otherNode)) = edgeIdx+1;
		DEBUGINT(printf("---- Adding edge %d between nodes %d and %d \n",edgeIdx,operation.data.rddata.newNodeID,otherNode));
		(*(coarsen_data->sparseConnTable))[edgeIdx][0] = otherNode;
		(*(coarsen_data->sparseConnTable))[edgeIdx][1] = operation.data.rddata.newNodeID;
	      }	
	    }
	  }
	}
      }else{
	DEBUGINT(printf("[%d] WEIRD -- REPLACE operation for element %d specifies different node number %d \n",CkMyPe(),operation.data.rddata.elemID,operation.data.rddata.oldNodeID));
      }
    }else{
      DEBUGINT(printf("[%d] WEIRD -- REPLACE operation for invalid element %d \n",CkMyPe(),operation.data.rddata.elemID));
    }
    DEBUGINT(printf("---------Replace invalidating node %d with %d in element %d\n",operation.data.rddata.oldNodeID,operation.data.rddata.newNodeID,operation.data.rddata.elemID));
    break;
  default:
    DEBUGINT(printf("[%d] WEIRD -- COARSENDATA type == invalid \n",CkMyPe()));
    CmiAbort("COARSENDATA type == invalid");
  }			
};


FDECL void FTN_NAME(FEM_REFINE2D_COARSEN,fem_refine2d_coarsen)(int *meshID,int *nodeID,double *coord,int *elemID,double *desiredAreas,int *sparseID){
  FEM_REFINE2D_Coarsen(*meshID,*nodeID,coord,*elemID,desiredAreas,*sparseID);
}

void interpolateNode(FEM_Node *node,int A,int B,int D,double frac){
  CkVec<FEM_Attribute *>*attrs = node->getAttrVec();
  for(int i=0;i<attrs->size();i++){
    FEM_Attribute *a = (FEM_Attribute *)(*attrs)[i];
    if(a->getAttr()<FEM_ATTRIB_TAG_MAX){
      FEM_DataAttribute *d = (FEM_DataAttribute *)a;
      AllocTable2d<double> *doubleData;
      if(a->getAttr() == FEM_DATA+6){
	doubleData = &d->getDouble();
	printf("A %d %.6lf %.6lf B %d %.6lf %.6lf frac %.6lf ",A,(*doubleData)[A][0],(*doubleData)[A][1],B,(*doubleData)[B][0],(*doubleData)[B][1],frac);
      }
      d->interpolate(A,B,D,frac);
      if(a->getAttr() == FEM_DATA+6){
	printf("D %d %.6lf %.6lf \n",D,(*doubleData)[D][0],(*doubleData)[D][1]);
      }
    }	
  }	
}


static int countValidEntities(int *validData,int total){
  int sum =0 ;
  for(int i=0;i<total;i++){
    sum += validData[i];
  }
  return sum;
}
