#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "tcharmc.h"
#include "fem.h"

#define TEST_SPARSE 1 /* Test out sparse data */
#define TEST_GHOST 1 /* Test out ghost layers */
#define TEST_SERIAL 0 /* Test out bizarre FEM_Serial_split */
#define TEST_MESHUPDATE 1 /* Test out FEM_Update_mesh */
#define TEST_SET_PARTITION 0  /* Test out FEM_Set_partition */

//These values are the sum of the node values
// after each timestep.
double *reduceValues=NULL;

//Number of time steps to simulate
int tsteps=10;
const int dim=29;//Length (elements) of one side of the FEM mesh
const int np=4; //Nodes per element for a quad

//Sum of sparse data[0] for both sets
double sparseSum[2];

extern "C" void
pupMyGlobals(pup_er p,void *ignored) 
{
	// CkPrintf("pupMyGlobals on PE %d\n",CkMyPe());
	pup_int(p,&tsteps);
	if (reduceValues==NULL)
		reduceValues=new double[tsteps];
	pup_doubles(p,reduceValues,tsteps);
	pup_doubles(p,sparseSum,2);
}

void tryPrint(int fem_mesh) {
  int nNodes=FEM_Mesh_get_length(fem_mesh,FEM_NODE);
  if (nNodes<50) 
    FEM_Mesh_print(fem_mesh);
}
void bad(const char *why) {
  fprintf(stderr,"FATAL ERROR: %s\n",why);
  exit(1);
}

/* Needed for linking in fem_alone mode: 
extern "C" void NONMANGLED_init(void) {init();}
extern "C" void NONMANGLED_driver(void) {driver();}
*/

void addGhostLayer(void) {
    //Set up ghost layers:
    if (0) 
    { /*Match across single nodes*/
     static const int quad2node[]={0,1,2,3};
     FEM_Add_ghost_layer(1,1);
     FEM_Add_ghost_elem(0,4,quad2node);
    } else if (1) 
    { /*Match edges*/
#if 1 /*Include ignored "-1" nodes as a test*/
     static const int quad2edge[]= {0,1,-1,  1,2,-1,  2,3,-1,  3,0,-1};
     FEM_Add_ghost_layer(3,1);
     FEM_Add_ghost_elem(0,4,quad2edge);
#else
     static const int quad2edge[]= {0,1,  1,2,  2,3,  3,0};
     FEM_Add_ghost_layer(2,1);
     FEM_Add_ghost_elem(0,4,quad2edge);
#endif
    }
}

extern "C" void
init(void)
{
  printf("init called for %d chunks\n",FEM_Num_partitions());
  tsteps=10;
  reduceValues=new double[tsteps];
  int *conn=new int[dim*dim*np];
  double *elements=new double[dim*dim];
  double *nodes=new double[(dim+1)*(dim+1)];
  double *noData=new double[(dim+1)*(dim+1)*tsteps];

  int nelems=dim*dim, nnodes=(dim+1)*(dim+1);

  //Describe the nodes and elements
  FEM_Set_node(nnodes,tsteps);
  FEM_Set_elem(0,nelems,1,np);

  //Create the connectivity array
  for(int y=0;y<dim;y++) for (int x=0;x<dim;x++) {
	   conn[(y*dim+x)*np+0]=(y  )*(dim+1)+(x  );
	   conn[(y*dim+x)*np+1]=(y+1)*(dim+1)+(x  );
	   conn[(y*dim+x)*np+2]=(y+1)*(dim+1)+(x+1);
	   conn[(y*dim+x)*np+3]=(y  )*(dim+1)+(x+1);
  }

  if (TEST_SPARSE) {
    //Create some random sparse data.  The first set
    // will run down the left side of the domain; the second set
    // down the diagonal.
    for (int sparseNo=0;sparseNo<2;sparseNo++) {
      int nSparse=dim;
      int *nodes=new int[2*nSparse];
      int *elems=new int[2*nSparse];
      double *data=new double[3*nSparse];
      sparseSum[sparseNo]=0.0;
      for (int y=0;y<nSparse;y++) {
  	int x=y*sparseNo;
  	nodes[2*y+0]=(y  )*(dim+1)+(x  );
  	nodes[2*y+1]=(y+1)*(dim+1)+(x+1);
  	elems[2*y+0]=0; //Always element type 0
  	elems[2*y+1]=y*dim+x;
  	double val=1.0+y*0.2+23.0*sparseNo;
  	sparseSum[sparseNo]+=val;
  	data[3*y+0]=data[3*y+1]=val;
  	data[3*y+2]=10.0;
      }
      FEM_Set_sparse(sparseNo,nSparse, nodes,2, data,3,FEM_DOUBLE);
      FEM_Set_sparse_elem(sparseNo,elems);
      delete[] nodes;
      delete[] elems;
      delete[] data;
    }
  }
  
  //Set the initial conditions
  for (int e=0;e<nelems;e++) elements[e]=0.0;
  elements[dim+2]=256;
  elements[dim+3]=256;
  FEM_Set_elem_data(0,elements);
  FEM_Set_elem_conn(0,conn);

  //Run the time loop over our serial mesh--
  // we'll use this data to check the parallel calculation.
  for (int t=0;t<tsteps;t++)
  {
    int i,j;

	//Nodes are sum of surrounding elements
    for(i=0;i<nnodes;i++) nodes[i] = 0.0;
    for(i=0;i<nelems;i++)
      for(j=0;j<np;j++)
        nodes[conn[i*np+j]] += elements[i]/np;

	//Elements are average of surrounding nodes
    for(i=0;i<nelems;i++) {
	  double sum=0;
      for(j=0;j<np;j++)
        sum += nodes[conn[i*np+j]];
      elements[i] = sum/np;
    }
	
    //Save the node values for this timestep
	for (i=0;i<nnodes;i++)
		 noData[i*tsteps+t]=nodes[i];	   

	//Compute the sum across all nodes
	double reduceSum=0;
	for (i=0;i<nnodes;i++) reduceSum+=nodes[i];
	reduceValues[t]=reduceSum;
  }
  FEM_Set_node_data(noData);
  FEM_Mesh_pup(FEM_Mesh_default_write(),0,pupMyGlobals,NULL);

  if (TEST_GHOST) {
    addGhostLayer();
    /* add a ghost stencil */
    int *ends=new int[nelems];
    int *adj=new int[dim];
    int magicElt=nelems-2;
    int curEnd=0;
    for (int e=0;e<nelems;e++) {
    	if (e==magicElt) 
	{ /* Make this element have the whole first row as ghosts */
		for (int i=0;i<dim;i++)
			adj[curEnd++]=i;
	}
	ends[e]=curEnd;
    }
    FEM_Add_ghost_stencil(nelems,1,ends,adj);
    delete []ends;
    delete []adj;
    /* Add another ghost layer */
    addGhostLayer();
  }
  delete[] conn;
  delete[] nodes;
  delete[] elements;
  delete[] noData;
  
  if (TEST_SET_PARTITION) {
    int *part=new int[nelems];
    int n=FEM_Num_partitions();
    for (int i=0;i<nelems;i++) part[i]=i%n;
    FEM_Set_partition(part);
  }

  if (TEST_SERIAL) //Test out the serial split routines
  {
    int nchunks=10;
    printf("Splitting into %d pieces:\n",nchunks);
    FEM_Serial_split(nchunks);
    for (int i=0;i<nchunks;i++) {
      FEM_Serial_begin(i);
      int node,elem,ignored;
      FEM_Get_node(&node,&ignored);
      FEM_Get_elem(0,&elem,&ignored,&ignored);
      printf(" partition[%d] has %d nodes, %d elems\n",i,node,elem);
    }
    FEM_Done();
  }
}

typedef struct _node {
  double val;
  unsigned char pad;
} Node;

typedef struct _element {
  short pad;
  double val;
} Element;

void testEqual(double is,double shouldBe,const char *what) {
	if (fabs(is-shouldBe)<0.000001) {
		//printf("[chunk %d] %s test passed.\n",FEM_My_partition(),what);
	} 
	else {/*test failed*/
		printf("[chunk %d] %s test FAILED-- expected %f, got %f\n",
                        FEM_My_partition(),what,shouldBe,is);
		exit(1);
	}
}

void testAssert(int shouldBe,const char *what,int myPartition=-1) 
{
	if (myPartition==-1) myPartition=FEM_My_partition();
	if (shouldBe) {
		// printf("[chunk %d] %s test passed.\n",myPartition,what);
	}
	else /*test failed-- should not be*/
	{
		printf("[chunk %d] %s test FAILED!\n",
			myPartition,what);
		exit(1);
	}
}

// Return an array of the real and ghost global entity numbers:
int *getGlobalNums(int mesh,int entity) {
	int nReal=FEM_Mesh_get_length(mesh,entity);
	int nGhost=FEM_Mesh_get_length(mesh,FEM_GHOST+entity);
	int nTot=nReal+nGhost;
	int *Gnum=new int[nTot];
	FEM_Mesh_get_data(mesh,entity,FEM_GLOBALNO,
	  &Gnum[0], 0,nReal, FEM_INDEX_0, 1);
	FEM_Mesh_get_data(mesh,FEM_GHOST+entity,FEM_GLOBALNO,
	  &Gnum[nReal], 0,nGhost, FEM_INDEX_0, 1);
	return Gnum;
}

extern "C" void
mesh_updated(int param);

extern "C" void
driver(void)
{
  int i,j;
  int nnodes,nelems,nnodeData,nelemData,np;
  int ngnodes, ngelems; //Counts including ghosts

 // FEM_Print("Starting driver...");
  FEM_Mesh_pup(FEM_Mesh_default_read(),0,pupMyGlobals,NULL);
  FEM_Get_node(&nnodes,&nnodeData);
  double *nodeData=new double[nnodeData*nnodes];
  FEM_Get_node_data(nodeData);  

  FEM_Get_elem(0,&nelems,&nelemData,&np);
  int *conn=new int[np*nelems];
  FEM_Get_elem_conn(0,conn);
  double *elData=new double[nelemData*nelems];
  FEM_Get_elem_data(0,elData);

  int myId = FEM_My_partition();
  tryPrint(FEM_Mesh_default_read());
  Node *nodes = new Node[nnodes];
  Element *elements = new Element[nelems];
  int doubleField=FEM_Create_simple_field(FEM_DOUBLE,1);
  int fid = FEM_Create_field(FEM_DOUBLE, 1, 
	(char *)(&nodes[0].val)-(char *)nodes, sizeof(Node));
  int efid = FEM_Create_field(FEM_DOUBLE, 1, 
	(char *)(&elements[0].val)-(char *)elements, sizeof(Element));
  
//Test out reduction
  double localSum = 1.0,globalSum;
  FEM_Reduce(fid, &localSum, &globalSum, FEM_SUM);
  testEqual(globalSum,(double)FEM_Num_partitions(),"reduce");
  
//Test readonly global
  testEqual(tsteps,nnodeData,"readonly");

//Test barrier
  FEM_Barrier();
  
  if (TEST_SPARSE) 
  { //Grab and check the sparse data:
    for (int sparseNo=0;sparseNo<2;sparseNo++) {
      int nSparse=FEM_Get_sparse_length(sparseNo);
      //printf("FEM Chunk %d has %d sparse entries (pass %d)\n",myId,nSparse,sparseNo);
      int *nodes=new int[2*nSparse];
      double *data=new double[3*nSparse];
      FEM_Get_sparse(sparseNo,nodes,data);
      //Clip off the ghost sparse elements:
      nSparse=FEM_Mesh_get_length(FEM_Mesh_default_read(), FEM_SPARSE+sparseNo);
      double sum=0.0;
      for (int y=0;y<nSparse;y++) {
    	testAssert(nodes[2*y]>=0 && nodes[2*y]<nnodes,"Sparse nodes");
    	testEqual(data[3*y+0],data[3*y+1],"Sparse data[0],[1]");
    	testEqual(data[3*y+2],10.0,"Sparse data[2]");
    	sum+=data[3*y];
      }
      double globalSum=0.0;
      FEM_Reduce(doubleField,&sum,&globalSum,FEM_SUM);
      testEqual(globalSum,sparseSum[sparseNo],"sparse data global sum");
      delete[] nodes;
      delete[] data;
    }
  }

//Set initial conditions
  for(i=0;i<nnodes;i++) {
    nodes[i].val = 0;
    nodes[i].pad=123;
  }
  for(i=0;i<nelems;i++) { 
    elements[i].val = elData[i]; 
    elements[i].pad=123;
  }

//Clip off ghost nodes/elements
  ngnodes=nnodes; ngelems=nelems;
  nnodes=FEM_Get_node_ghost();
  nelems=FEM_Get_elem_ghost(0);
  printf("Chunk %d: %d (%dg) nodes; %d (%dg) elems\n",
  	myId, nnodes,ngnodes, nelems,ngelems);
  
  if (TEST_GHOST) {//Update ghost field test
    
    // Write crap into our ghost values:
    for (i=0;i<nelems;i++) {elements[i].val=elData[i];}
    for (i=nelems;i<ngelems;i++) {elements[i].val=-1.0;}
    if (1) { //Use IDXL routines directly
      IDXL_t elComm=FEM_Comm_ghost(FEM_Mesh_default_read(),FEM_ELEM+0);
      /* Make sure each ghost is accessible */
      int gCount=ngelems-nelems;
      for (int g=0;g<gCount;g++) {
        int src=IDXL_Get_source(elComm,g);
	if (src==myId || src<0) 
		bad("Bad chunk number returned by IDXL_Get_source");
      }
      
      IDXL_t sumComm=IDXL_Create();
      IDXL_Combine(sumComm,elComm, 0, nelems); //Shift ghosts down to nelems..ngelems
      // if (myId==0) IDXL_Print(sumComm);
      IDXL_Comm_sendrecv(0,sumComm, efid, elements);
      IDXL_Destroy(sumComm);
    }
    else { //Use compatability FEM_Update_ghost_field
      FEM_Update_ghost_field(efid,0, elements);
    }
    
    // Now check to see if our ghosts got the values:
    for (i=0;i<ngelems;i++)
      testEqual(elements[i].pad,123,"update element ghost field pad");
    for (i=0;i<ngelems;i++)
      testEqual(elements[i].val,elData[i],"update element ghost field test");
  }
  
  int *elList=new int[100+ngelems];

//Time loop
  for (int t=0;t<tsteps;t++)
  {
	//Nodes are sum of surrounding elements
    for(i=0;i<nnodes;i++) nodes[i].val = 0.0;
    for(i=0;i<nelems;i++)
      for(j=0;j<np;j++)
        nodes[conn[i*np+j]].val += elements[i].val/np;

	//Update shared nodes
    FEM_Update_field(fid, nodes);

	//Elements are average of surrounding nodes
    for(i=0;i<nelems;i++) {
	  double sum=0;
      for(j=0;j<np;j++)
        sum += nodes[conn[i*np+j]].val;
      elements[i].val = sum/np;
    }

	//Check the update
    for(i=0;i<nnodes;i++)
        testEqual(nodes[i].val,nodeData[nnodeData*i+t], "update_field");

    double sum = 0.0;
    FEM_Reduce_field(fid, nodes, &sum, FEM_SUM);
    testEqual(sum,reduceValues[t],"reduce_field");
    
    if (TEST_GHOST) {//Update ghost field test
      //Communicate our ghost elements:
      for (i=nelems;i<ngelems;i++) elements[i].val=-1;
      FEM_Update_ghost_field(efid,0,elements);
      for (i=nelems;i<ngelems;i++) {
    	testAssert(elements[i].val!=-1,"update_ghost_field");
      }

      //Communicate our ghost nodes:
      FEM_Update_ghost_field(fid,-1,nodes);
      for(i=nnodes;i<ngnodes;i++)
    	  testEqual(nodes[i].val,nodeData[nnodeData*i+t],
    	     "update_ghost_node_field");


      //Make a list of elements with odd global numbers
      int *elGnum=getGlobalNums(FEM_Mesh_default_read(),FEM_ELEM+0);
      int elListLen=0;
      double thresh=2.0;
      for (i=0;i<nelems;i++)
    	      if (elGnum[i]%2) {
    		      //printf("[%d] List: Local %d, global %d)\n",myId,i,elGnum[i]);
    		      elList[elListLen++]=i;
    	      }
      
      //Get a list of ghost elements with odd global numbers
      FEM_Exchange_ghost_lists(0,elListLen,elList);
      elListLen=FEM_Get_ghost_list_length();
      FEM_Get_ghost_list(elList);
      //printf("[%d] My ghost list has %d entries\n",myId,elListLen);
      //Make sure everything on the list are actually ghosts and
      // actually have large values
      for (i=0;i<elListLen;i++) {
    	      testAssert(elList[i]<ngelems,"Ghost list ghost test (upper)");
    	      testAssert(elList[i]>=nelems,"Ghost list ghost test (lower)");
    	      testAssert(elGnum[elList[i]]%2,"Ghost list contents test");
      }
      delete[] elGnum;
    }

#if ENABLE_MIG /*Only works with -memory isomalloc*/
    printf("Before migrate: Thread %d on pe %d\n",myId,CkMyPe());
    FEM_Migrate();
    printf("After migrate: Thread %d on pe %d\n",myId,CkMyPe());
#endif

  }

  if (TEST_MESHUPDATE) {
/*Try reassembling the mesh*/
    int *noGnum=getGlobalNums(FEM_Mesh_default_read(),FEM_NODE);
    double *nodeOut=new double[nnodes];
    FEM_Set_node(nnodes,1);
    for (i=0;i<nnodes;i++) {
    	nodeOut[i]=0.1*noGnum[i];
    }
    FEM_Set_node_data(nodeOut);
    delete[] nodeOut;
    delete[] noGnum;
    int *elGnum=getGlobalNums(FEM_Mesh_default_read(),FEM_ELEM+0);
    double *elOut=new double[nelems];
    FEM_Set_elem(0,nelems,1,0);
    for (i=0;i<nelems;i++) {
    	elOut[i]=0.1*elGnum[i];
    }
    FEM_Set_elem_data(0,elOut);
    delete[] elOut;
    delete[] elGnum;
  if (TEST_SPARSE) 
  { //Grab and copy over the sparse data:
    for (int sparseNo=0;sparseNo<2;sparseNo++) {
      int nSparse=FEM_Get_sparse_length(sparseNo);
      int *nodes=new int[2*nSparse];
      double *data=new double[3*nSparse];
      FEM_Get_sparse(sparseNo,nodes,data);
      nSparse=FEM_Mesh_get_length(FEM_Mesh_default_read(), FEM_SPARSE+sparseNo);
      FEM_Set_sparse(sparseNo,nSparse,nodes,2,data,3,FEM_DOUBLE);
      delete[] nodes;
      delete[] data;
    }
  }
  
    FEM_Update_mesh(mesh_updated,123,2);
  }
  
  FEM_Print("All tests passed.");
}

extern "C" void
mesh_updated(int param)
{
  int nnodes,nelems,i,nodePer,dataPer;
  // printf("mesh_updated(%d) called.\n",param);
  testEqual(param,123,"mesh_updated param");

  tryPrint(FEM_Mesh_default_read());
  
  FEM_Get_node(&nnodes,&dataPer);
  // printf("Getting %d nodes (%d data per)\n",nnodes,dataPer);
  double *ndata=new double[nnodes*dataPer];
  FEM_Get_node_data(ndata);
  for (i=0;i<nnodes;i++) {
    testEqual(ndata[i],0.1*i,"mesh_updated node values");
  }
  delete[] ndata;
  
  FEM_Get_elem(0,&nelems,&dataPer,&nodePer);
  // printf("Getting %d elems (%d data per)\n",nelems,dataPer);
  double *ldata=new double[nelems*dataPer];
  FEM_Get_elem_data(0,ldata);
  for (i=0;i<nelems;i++) {
    testEqual(ldata[i],0.1*i,"mesh_updated elem values");
  }
  delete[] ldata;
}
