#include "ckvector3d.h"
#include "charm-api.h"
#include "refine.h"
#include "fem.h"
#include "fem_mesh.h"
#include "femrefine.h"
#include "ckhashtable.h"

class intdual{
	private:
		int x,y;
	public:
		intdual(int _x,int _y){
			x = _x; y=_y;
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




void FEM_REFINE2D_Newmesh(int meshID,int nodeID,int elemID){
	int nelems = FEM_Mesh_get_length(meshID,elemID);
	int	nghost = FEM_Mesh_get_length(meshID,elemID+FEM_GHOST);
	int total = nghost + nelems;
	int *tempMesh = new int[3*total];
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
	
  /*Set up refinement framework*/
  REFINE2D_NewMesh(nelems,total,(int *)tempMesh,gid);
	delete [] gid;
	delete [] tempMesh;
}

FDECL void FTN_NAME(FEM_REFINE2D_NEWMESH,fem_refine2d_newmesh)(int *meshID,int *nodeID,int *elemID){
	FEM_REFINE2D_Newmesh(*meshID,*nodeID,*elemID);
}



void FEM_REFINE2D_Split(int meshID,int nodeID,double *coord,int elemID,double *desiredAreas){
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
		      if (oldRow[i]==A) oldRow[i]=D;	
					CkPrintf("In triangle %d %d replaced by %d \n",tri,A,D);
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
					elattr->copyEntity(newTri,*elattr,tri);
			}
		}

		
	}
	printf("Cordinate list length %d \n",coordVec.size()/2);
	IDXL_Sort_2d(FEM_Comm_shared(meshID,nodeID),coordVec.getVec());
	
}

FDECL void FTN_NAME(FEM_REFINE2D_SPLIT,fem_refine2d_split)(int *meshID,int *nodeID,double *coord,int *elemID,double *desiredAreas){
	FEM_REFINE2D_Split(*meshID,*nodeID,coord,*elemID,desiredAreas);
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
		int tri,nodeToThrow,nodeToKeep,flag,idxbase;
		double nx,ny;
		//temp sol
		idxbase = 0;
		REFINE2D_Get_Collapse(collapseNo,connData,&tri,&nodeToThrow,&nodeToKeep,&nx,&ny,&flag,idxbase);
		if(flag & 0x1 || flag & 0x2){
			coord[2*nodeToKeep]=nx;
			coord[2*nodeToKeep+1] = ny;
			validNodeData[nodeToThrow]=0;
			connData[3*tri] = -1;
			connData[3*tri+1] = -1;
			connData[3*tri+2] = -1;
		}
		validElemData[tri] = 0;
	}
	
}  

