#include "ckvector3d.h"
#include "charm-api.h"
#include "refine.h"
#include "fem.h"
#include "fem_mesh.h"
#include "femrefine.h"
#include "ckhashtable.h"
#include <assert.h>

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
		int getx(){return x;};
		int gety(){return y;};
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
	int	nghost = FEM_Mesh_get_length(meshID,elemID+FEM_GHOST);
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
	}
	
  /*Set up refinement framework*/
  REFINE2D_NewMesh(nelems,total,nnodes,(int *)tempMesh,gid,tempBoundaries);
	if(tempBoundaries){
		delete [] tempBoundaries;
	}
	delete [] gid;
	delete [] tempMesh;
}

FDECL void FTN_NAME(FEM_REFINE2D_NEWMESH,fem_refine2d_newmesh)(int *meshID,int *nodeID,int *elemID){
	FEM_REFINE2D_Newmesh(*meshID,*nodeID,*elemID);
}



void FEM_REFINE2D_Split(int meshID,int nodeID,double *coord,int elemID,double *desiredAreas,int sparseID){
	int nnodes = FEM_Mesh_get_length(meshID,nodeID);
	int nelems = FEM_Mesh_get_length(meshID,elemID);

/*	for(int k=0;k<nnodes;k++){
		printf(" node %d ( %.6f %.6f )\n",k,coord[2*k+0],coord[2*k+1]);
	}*/
	printf("%d %d \n",nnodes,nelems);	
	REFINE2D_Split(nnodes,coord,nelems,desiredAreas);
	
  
	int nSplits=REFINE2D_Get_Split_Length();
	printf("called REFINE2D_Split nSplits = %d \n",nSplits);

	if(nSplits == 0){
		return;
	}

	/*Copy the cordinates of the nodes into a vector, 
		the cordinates of the new nodes will be inserted
		into this vector and will be used to sort all the
		nodes on the basis of the distance from origin
	*/
	CkVec<double> coordVec;
	for(int i=0;i<nnodes*2;i++){
		coordVec.push_back(coord[i]);
	}
	
	/*find out the attributes of the node 
	*/
	FEM_Entity *e=FEM_Entity_lookup(meshID,nodeID,"REFINE2D_Mesh");
	CkVec<FEM_Attribute *> *attrs = e->getAttrVec();
	
	FEM_Entity *elem = FEM_Entity_lookup(meshID,elemID,"REFIN2D_Mesh_elem");
	CkVec<FEM_Attribute *> *elemattrs = elem->getAttrVec();

	FEM_Attribute *connAttr = elem->lookup(FEM_CONN,"REFINE2D_Mesh");
	if(connAttr == NULL){
		CkAbort("Grrrr element without connectivity \n");
	}
	AllocTable2d<int> &connTable = ((FEM_IndexAttribute *)connAttr)->get();

	//hashtable to store the new node number as a function of the two old numbers
	CkHashtableT<intdual,int> newnodes;
	
	/*
		Get the FEM_BOUNDARY data of sparse elements and load it into a hashtable
		indexed by the 2 node ids that make up the edge. The data in the hashtable
		is the index number of the sparse element
	*/
	FEM_Entity *sparse;
	CkVec<FEM_Attribute *> *sparseattrs;
	FEM_Attribute *sparseConnAttr, *sparseBoundaryAttr;
	AllocTable2d<int> *sparseConnTable, *sparseBoundaryTable;
	CkHashtableT<intdual,int> nodes2sparse;
	if(sparseID != -1){
		sparse = FEM_Entity_lookup(meshID,sparseID,"REFINE2D_Mesh_sparse");
		sparseattrs = sparse->getAttrVec();
		sparseConnAttr = sparse->lookup(FEM_CONN,"REFINE2D_Mesh_sparse");
		sparseConnTable = &(((FEM_IndexAttribute *)sparseConnAttr)->get());
		sparseBoundaryAttr = sparse->lookup(FEM_BOUNDARY,"REFINE2D_Mesh_sparse");
		if(sparseBoundaryAttr == NULL){
			 CkAbort("Specified sparse elements without boundary conditions");
		}
		/*
			since the default value in the hashtable is 0, to 
			distinguish between uninserted keys and the sparse element
			with index 0, the index of the sparse elements is incremented
			by 1 while inserting.
		*/
//		printf("[%d] Sparse elements\n",FEM_My_partition());
		for(int j=0;j<sparse->size();j++){
			int *cdata = (*sparseConnTable)[j];
	//		printf("%d < %d,%d > \n",j,cdata[0],cdata[1]);
			nodes2sparse.put(intdual(cdata[0],cdata[1])) = j+1;
		}
	}	
	
  for (int splitNo=0;splitNo<nSplits;splitNo++){
    int tri,A,B,C,D;
    double frac;
		// current number of nodes in the mesh
		int cur_nodes = FEM_Mesh_get_length(meshID,nodeID);
		int *connData = connTable.getData();
		int flags;


		REFINE2D_Get_Split(splitNo,(int *)(connData),&tri,&A,&B,&C,&frac,&flags);
		if((flags & 0x1) || (flags & 0x2)){
			//new node 
			D = cur_nodes;
      CkPrintf("---- Adding node %d\n",D);					
		/*	lastA=A;
			lastB=B;
			lastD=D;*/
      if (A>=cur_nodes) CkAbort("Calculated A is invalid!");
      if (B>=cur_nodes) CkAbort("Calculated B is invalid!");
			e->setLength(cur_nodes+1);
			for(int i=0;i<attrs->size();i++){
				FEM_Attribute *a = (FEM_Attribute *)(*attrs)[i];
				if(a->getAttr()<FEM_ATTRIB_TAG_MAX){
					FEM_DataAttribute *d = (FEM_DataAttribute *)a;
					d->interpolate(A,B,D,frac);
				}else{
						/*The boundary value of a new node should be the 
						boundary value of the edge(sparse element) that contains
						the two nodes */
						if(a->getAttr() == FEM_BOUNDARY){
							if(sparseID != -1){
								int sidx = nodes2sparse.get(intdual(A,B))-1;
								if(sidx == -1){
									CkAbort("no sparse element between these 2 nodes, are they really connected ??");
								}
								sparseBoundaryTable = &(((FEM_DataAttribute *)sparseBoundaryAttr)->getInt());
								int boundaryVal = ((*sparseBoundaryTable)[sidx])[0];
								(((FEM_DataAttribute *)a)->getInt()[D])[0] = boundaryVal;
							}else{
								/*
									if sparse elements don't exist then just do simple
									interpolation
								*/
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
				coordVec.push_back(Dx);
				coordVec.push_back(Dy);
				newnodes.put(intdual(A,B))=D;
			/*
				add the new sparse element <D,B> and modify the connectivity of the old one
				from <A,B> to <A,D> and change the hashtable to reflect that change
			*/
			if(sparseID != -1){
				int oldsidx = nodes2sparse.get(intdual(A,B))-1;
				int newsidx = sparse->size();
				sparse->setLength(newsidx+1);
				for(int satt = 0;satt<sparseattrs->size();satt++){
					if((*sparseattrs)[satt]->getAttr() == FEM_CONN){
						/*
							change the conn of the old sparse to A,D
							and new one to B,D
						*/
						sparseConnTable = &(((FEM_IndexAttribute *)sparseConnAttr)->get());
						int *oldconn = (*sparseConnTable)[oldsidx];
						int *newconn = (*sparseConnTable)[newsidx];
						oldconn[0] = A;
						oldconn[1] = D;
						
						newconn[0] = D;
						newconn[1] = B;
						
//						printf("<%d,%d> edge being split into <%d,%d> <%d,%d> \n",A,B,A,D,D,B);
					}else{
						/*
							apart from conn copy everything else
						*/
						FEM_Attribute *attr = (FEM_Attribute *)(*sparseattrs)[satt];
						attr->copyEntity(newsidx,*attr,oldsidx);
					}
				}
				/*
					modify the hashtable - delete the old edge
					and the new ones
				*/
				nodes2sparse.remove(intdual(A,B));
				nodes2sparse.put(intdual(A,D)) = oldsidx+1;
				nodes2sparse.put(intdual(D,B)) = newsidx+1;
			}
				
		}
		//add a new triangle
		/*TODO: replace  FEM_ELEM with parameter*/
		int newTri =  FEM_Mesh_get_length(meshID,elemID);
    CkPrintf("---- Adding triangle %d after splitting %d \n",newTri,tri);
		elem->setLength(newTri+1);
		D = newnodes.get(intdual(A,B));
		for(int j=0;j<elemattrs->size();j++){
			if((*elemattrs)[j]->getAttr() == FEM_CONN){
				CkPrintf("elem attr conn code %d \n",(*elemattrs)[j]->getAttr());
				//it is a connectivity attribute.. get the connectivity right
				FEM_IndexAttribute *connAttr = (FEM_IndexAttribute *)(*elemattrs)[j];
				AllocTable2d<int> &table = connAttr->get();
				CkPrintf("Table of connectivity attribute starts at %p width %d \n",table[0],connAttr->getWidth());
				int *oldRow = table[tri];
				int *newRow = table[newTri];
				for (int i=0;i<3;i++){
		      if (oldRow[i]==A){
						oldRow[i]=D;	
						CkPrintf("In triangle %d %d replaced by %d \n",tri,A,D);
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
				CkPrintf("New Triangle %d  (%d %d %d) conn %p\n",newTri,newRow[0],newRow[1],newRow[2],newRow);
			}else{
				FEM_Attribute *elattr = (FEM_Attribute *)(*elemattrs)[j];
				if(elattr->getAttr() < FEM_ATTRIB_FIRST){ 
					elattr->copyEntity(newTri,*elattr,tri);
				}
			}
		}
		if(sparseID != -1){
			/*
				add the sparse element (edge between C and D)
			*/
			int cdidx = sparse->size();
			sparse->setLength(cdidx+1);
			for(int satt = 0; satt < sparseattrs->size();satt++){
					if((*sparseattrs)[satt]->getAttr() == FEM_CONN){
						sparseConnTable = &(((FEM_IndexAttribute *)sparseConnAttr)->get());
						int *cdconn = (*sparseConnTable)[cdidx];
						cdconn[0]=C;
						cdconn[1]=D;
					}
					if((*sparseattrs)[satt]->getAttr() == FEM_BOUNDARY){
						/*
							An edge connecting C and D has to be an internal edge
						*/
						sparseBoundaryTable = &(((FEM_DataAttribute *)sparseBoundaryAttr)->getInt());
						((*sparseBoundaryTable)[cdidx])[0] = 0;
					}
			}
			nodes2sparse.put(intdual(C,D)) = cdidx+1;
		}
		
	}
	printf("Cordinate list length %d \n",coordVec.size()/2);
	IDXL_Sort_2d(FEM_Comm_shared(meshID,nodeID),coordVec.getVec());
	int read = FEM_Mesh_is_get(meshID) ;
	assert(read);
	
/*	IDXL_t sharedid = FEM_Comm_shared(meshID,nodeID);
	IDXL_Side_t sharedsendid = IDXL_Get_send(sharedid);
	if(
	int numshared = IDXL_Get_count(sharedsendid,0);
	int *list = new int[numshared];
	
	char str[100];
	sprintf(str,"shared_%d",FEM_My_partition());
	FILE *fp = fopen(str,"a");
	IDXL_Get_list(sharedsendid,0,list);
	fprintf(fp,"*********************************\n");
	for(int i=0;i<numshared;i++){
		fprintf(fp,"id %d (%.6lf %.6lf) \n",list[i],coordVec[2*list[i]],coordVec[2*list[i]+1]);
	}
	fprintf(fp,"*********************************\n");	
	fclose(fp);
	delete [] list;*/
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

void FEM_REFINE2D_Coarsen(int meshID,int nodeID,double *coord,int elemID,double *desiredAreas){
	int nnodes = FEM_Mesh_get_length(meshID,nodeID);
	int nelems = FEM_Mesh_get_length(meshID,elemID);
	int nodeCount=0,elemCount=0;
		
	/*
		The attributes of the different entities
	*/
	
	FEM_Entity *node=FEM_Entity_lookup(meshID,nodeID,"REFINE2D_Mesh");
	CkVec<FEM_Attribute *> *attrs = node->getAttrVec();
	FEM_Attribute *validNodeAttr = node->lookup(FEM_VALID,"FEM_COARSEN");
	
	FEM_Entity *elem = FEM_Entity_lookup(meshID,elemID,"REFIN2D_Mesh_elem");
	CkVec<FEM_Attribute *> *elemattrs = elem->getAttrVec();
	FEM_Attribute *validElemAttr = elem->lookup(FEM_VALID,"FEM_COARSEN");

	FEM_Attribute *connAttr = elem->lookup(FEM_CONN,"REFINE2D_Mesh");
	if(connAttr == NULL){
		CkAbort("Grrrr element without connectivity \n");
	}
	AllocTable2d<int> &connTable = ((FEM_IndexAttribute *)connAttr)->get();
	AllocTable2d<unsigned char>&validNodeTable = ((FEM_DataAttribute *)validNodeAttr)->getChar();
	AllocTable2d<unsigned char>&validElemTable = ((FEM_DataAttribute *)validElemAttr)->getChar();
	int *connData = connTable.getData();
	unsigned char *validNodeData = validNodeTable.getData();
	unsigned char *validElemData = validElemTable.getData();


	for(int k=0;k<nnodes;k++){
		if(validNodeData[k]){
			nodeCount++;
		}
	}
	for(int k=0;k<nelems;k++){
		if(validElemData[k]){
			elemCount++;
		}
	}

	printf("coarsen %d %d \n",nodeCount,elemCount);	
	REFINE2D_Coarsen(nodeCount,coord,elemCount,desiredAreas);
	int nCollapses = REFINE2D_Get_Collapse_Length();


	
	for(int collapseNo=0;collapseNo < nCollapses;collapseNo++){
		int tri,nodeToThrow,nodeToKeep,n1,n2;
		coarsenData operation;
		REFINE2D_Get_Collapse(collapseNo,&operation);
		switch(operation.type){
			case COLLAPSE:
				tri = operation.data.cdata.elemID;
				nodeToKeep = operation.data.cdata.nodeToKeep;
				nodeToThrow = operation.data.cdata.nodeToDelete;
	/*			n1 = operation.data.cdata.collapseEdge;
				n2 = (n1 + 1)%3;
				if(operation.data.cdata.nodeToKeep == n1){
					nodeToThrow = connData[3*tri+n2];
				}else{
					nodeToThrow = connData[3*tri+n1];
				}*/
				if(operation.data.cdata.flag & 0x1 || operation.data.cdata.flag & 0x2){
					coord[2*nodeToKeep] = operation.data.cdata.newX;
					coord[2*nodeToKeep+1] = operation.data.cdata.newY;
					validNodeData[nodeToThrow] = 0;
					printf("---------Collapse <%d,%d> invalidating node %d \n",nodeToKeep,nodeToThrow,nodeToThrow);
				}
				validElemData[tri] = 0;
				connData[3*tri] = connData[3*tri+1] = connData[3*tri+2] = -1;
				break;
			case 	UPDATE:
				if(validNodeData[operation.data.udata.nodeID]){
					coord[2*(operation.data.udata.nodeID)] = operation.data.udata.newX;
					coord[2*(operation.data.udata.nodeID)+1] = operation.data.udata.newY;
				}else{
					printf("[%d] WEIRD -- update operation for invalid node %d \n",CkMyPe(),operation.data.udata.nodeID);
				}
				break;
			case REPLACE:
				if(validElemData[operation.data.rddata.elemID]){
					if(connData[3*operation.data.rddata.elemID+operation.data.rddata.relnodeID] == operation.data.rddata.oldNodeID){
						connData[3*operation.data.rddata.elemID+operation.data.rddata.relnodeID] = operation.data.rddata.newNodeID;
						if(validNodeData[operation.data.rddata.oldNodeID]){
							validNodeData[operation.data.rddata.oldNodeID]=0;
						}
					}else{
						printf("[%d] WEIRD -- REPLACE operation for element %d specifies different node number %d \n",CkMyPe(),operation.data.rddata.elemID,operation.data.rddata.oldNodeID);
					}
				}else{
						printf("[%d] WEIRD -- REPLACE operation for invalid element %d \n",CkMyPe(),operation.data.rddata.elemID);
				}
				printf("---------Replace invalidating node %d with %d in element %d\n",operation.data.rddata.oldNodeID,operation.data.rddata.newNodeID,operation.data.rddata.elemID);
				break;
				default:
					printf("[%d] WEIRD -- COARSENDATA type == invalid \n",CkMyPe());
					CmiAbort("COARSENDATA type == invalid");
		}
	/*	int tri,nodeToThrow,nodeToKeep,flag,idxbase;
		double nx,ny;
		//temp sol
		idxbase = 0;
		REFINE2D_Get_Collapse(collapseNo,connData,&tri,&nodeToThrow,&nodeToKeep,&nx,&ny,&flag,idxbase);
		if(flag & 0x1 || flag & 0x2){
			coord[2*nodeToKeep]=nx;
			coord[2*nodeToKeep+1] = ny;
			validNodeData[nodeToThrow]=0;
		}
		validElemData[tri] = 0;
		connData[3*tri] = -1;
		connData[3*tri+1] = -1;
		connData[3*tri+2] = -1;*/
	}
	
}  

