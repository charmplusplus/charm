#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "charm++.h"
#include "tcharmc.h"
#include "fem.h"

//These values are the sum of the node values
// after each timestep.
double *reduceValues=NULL;

//Number of time steps to simulate
int tsteps=10;
const int dim=20;//Length (elements) of one side of the FEM mesh
const int np=4; //Nodes per element for a quad

//Sum of sparse data[0] for both sets
double sparseSum[2];

extern "C" void
pupMyGlobals(pup_er p) 
{
	//CkPrintf("pupMyGlobals on PE %d\n",CkMyPe());
	pup_int(p,&tsteps);
	if (pup_isUnpacking(p))
		reduceValues=new double[tsteps];
	pup_doubles(p,reduceValues,tsteps);
	pup_doubles(p,sparseSum,2);
}

extern "C" void
TCharmUserNodeSetup(void)
{
	TCharmReadonlyGlobals(pupMyGlobals);
}

void printargs(void) {
  CkPrintf("Args for pe %d: ",CkMyPe());
  for (int i=0;i<CkGetArgc();i++) {
    CkPrintf("'%s' ",CkGetArgv()[i]);
  }
  CkPrintf("\n");
}

extern "C" void
init(void)
{
  CkPrintf("init called\n");
  printargs();
  tsteps=10;
  reduceValues=new double[tsteps];
  int *conn=new int[dim*dim*np];
  double *elements=new double[dim*dim];
  double *nodes=new double[(dim+1)*(dim+1)];
  double *noData=new double[(dim+1)*(dim+1)*tsteps];

  int nelems=dim*dim, nnodes=(dim+1)*(dim+1);
  for (int e=0;e<nelems;e++) elements[e]=0.3;

  //Describe the nodes and elements
  FEM_Set_Node(nnodes,tsteps);
  FEM_Set_Elem(0,nelems,1,np);

  //Create the connectivity array
  for(int y=0;y<dim;y++) for (int x=0;x<dim;x++) {
	   conn[(y*dim+x)*np+0]=(y  )*(dim+1)+(x  );
	   conn[(y*dim+x)*np+1]=(y+1)*(dim+1)+(x  );
	   conn[(y*dim+x)*np+2]=(y+1)*(dim+1)+(x+1);
	   conn[(y*dim+x)*np+3]=(y  )*(dim+1)+(x+1);
  }
  
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
    FEM_Set_Sparse(sparseNo,nSparse, nodes,2, data,3,FEM_DOUBLE);
    FEM_Set_Sparse_Elem(sparseNo,elems);
    delete[] nodes;
    delete[] elems;
    delete[] data;
  }
  
  //Set the initial conditions
  elements[3*dim+1]=256;
  elements[2*dim+1]=256;
  FEM_Set_Elem_Data(0,elements);
  FEM_Set_Elem_Conn(0,conn);

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
  FEM_Set_Node_Data(noData);

//Set up ghost layers:
  if (0) 
  { /*Match across single nodes*/
     static const int quad2node[]={0,1,2,3};
     FEM_Add_Ghost_Layer(1,1);
     FEM_Add_Ghost_Elem(0,4,quad2node);
  } else if (1) 
  { /*Match edges*/
#if 1 /*Include ignored "-1" nodes as a test*/
     static const int quad2edge[]= {0,1,-1,  1,2,-1,  2,3,-1,  3,0,-1};
     FEM_Add_Ghost_Layer(3,1);
     FEM_Add_Ghost_Elem(0,4,quad2edge);
#else
     static const int quad2edge[]= {0,1,  1,2,  2,3,  3,0};
     FEM_Add_Ghost_Layer(2,1);
     FEM_Add_Ghost_Elem(0,4,quad2edge);
#endif
/*Add a second layer
     FEM_Add_Ghost_Layer(2,0);
     FEM_Add_Ghost_Elem(0,4,quad2edge);
*/
  }
  delete[] conn;
  delete[] nodes;
  delete[] elements;
  delete[] noData;

#if 0 //Test out the serial split routines
  int nchunks=10;
  printf("Splitting into %d pieces:\n");
  FEM_Serial_Split(nchunks);
  for (int i=0;i<nchunks;i++) {
    FEM_Serial_Begin(i);
    int node,elem,ignored;
    FEM_Get_Node(&node,&ignored);
    FEM_Get_Elem(0,&elem,&ignored,&ignored);
    printf(" partition[%d] has %d nodes, %d elems\n",i,node,elem);
  }
  FEM_Done();
#endif
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
		//CkPrintf("[chunk %d] %s test passed.\n",FEM_My_Partition(),what);
	} 
	else {/*test failed*/
		CkPrintf("[chunk %d] %s test FAILED-- expected %f, got %f (pe %d)\n",
                        FEM_My_Partition(),what,shouldBe,is,CkMyPe());
		CkAbort("FEM Test failed\n");
	}
}

void testAssert(int shouldBe,const char *what,int myPartition=-1) 
{
	if (myPartition==-1) myPartition=FEM_My_Partition();
	if (shouldBe) {
		// CkPrintf("[chunk %d] %s test passed.\n",myPartition,what);
	}
	else /*test failed-- should not be*/
	{
		CkPrintf("[chunk %d] %s test FAILED! (pe %d)\n",
			myPartition,what,CkMyPe());
		CkAbort("FEM Test failed\n");
	}
}

extern "C" void
driver(void)
{
  int nnodes,nelems,nnodeData,nelemData,np;
  int ngnodes, ngelems; //Counts including ghosts

  FEM_Print("Starting driver...");
printargs();
  FEM_Get_Node(&nnodes,&nnodeData);
  double *nodeData=new double[nnodeData*nnodes];
  FEM_Get_Node_Data(nodeData);  

  FEM_Get_Elem(0,&nelems,&nelemData,&np);
  int *conn=new int[np*nelems];
  FEM_Get_Elem_Conn(0,conn);
  double *elData=new double[nelemData*nelems];
  FEM_Get_Elem_Data(0,elData);

  int myId = FEM_My_Partition();
  //FEM_Print_Partition();
  Node *nodes = new Node[nnodes];
  Element *elements = new Element[nelems];
  int doubleField=FEM_Create_Simple_Field(FEM_DOUBLE,1);
  int fid = FEM_Create_Field(FEM_DOUBLE, 1, 
	(char *)(&nodes[0].val)-(char *)nodes, sizeof(Node));
  int efid = FEM_Create_Field(FEM_DOUBLE, 1, 
	(char *)(&elements[0].val)-(char *)elements, sizeof(Element));
  int i;
  
//Test out reduction
  double localSum = 1.0,globalSum;
  FEM_Reduce(fid, &localSum, &globalSum, FEM_SUM);
  testEqual(globalSum,(double)FEM_Num_Partitions(),"reduce");
  
//Test readonly global
  testEqual(tsteps,nnodeData,"readonly");

//Test barrier
  FEM_Barrier();

//Grab and check the sparse data:
  for (int sparseNo=0;sparseNo<2;sparseNo++) {
    int nSparse=FEM_Get_Sparse_Length(sparseNo);
    //CkPrintf("FEM Chunk %d has %d sparse entries (pass %d)\n",myId,nSparse,sparseNo);
    int *nodes=new int[2*nSparse];
    double *data=new double[3*nSparse];
    FEM_Get_Sparse(sparseNo,nodes,data);
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
  nnodes=FEM_Get_Node_Ghost();
  nelems=FEM_Get_Elem_Ghost(0);

//Update ghost field test
  for (i=nelems;i<ngelems;i++) {elements[i].val=-1.0;}
  FEM_Update_Ghost_Field(efid,0,elements);
  for (i=0;i<ngelems;i++)
	  testEqual(elements[i].pad,123,"update element ghost field pad");
  for (i=0;i<ngelems;i++)
	  testEqual(elements[i].val,elData[i],"update element ghost field test");

  int *elList=new int[100+ngelems];

//Time loop
  for (int t=0;t<tsteps;t++)
  {
    int i,j;

	//Nodes are sum of surrounding elements
    for(i=0;i<nnodes;i++) nodes[i].val = 0.0;
    for(i=0;i<nelems;i++)
      for(j=0;j<np;j++)
        nodes[conn[i*np+j]].val += elements[i].val/np;

	//Update shared nodes
    FEM_Update_Field(fid, nodes);

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
    FEM_Reduce_Field(fid, nodes, &sum, FEM_SUM);
    testEqual(sum,reduceValues[t],"reduce_field");

    //Communicate our ghost elements:
    for (i=nelems;i<ngelems;i++) elements[i].val=-1;
    FEM_Update_Ghost_Field(efid,0,elements);
    for (i=nelems;i<ngelems;i++) {
      testAssert(elements[i].val!=-1,"update_ghost_field");
    }

    //Communicate our ghost nodes:
    FEM_Update_Ghost_Field(fid,-1,nodes);
    for(i=nnodes;i<ngnodes;i++)
        testEqual(nodes[i].val,nodeData[nnodeData*i+t],
	   "update_ghost_node_field");


    //Make a list of elements with odd global numbers
    const int *elGnum=FEM_Get_Elem_Nums();
    int elListLen=0;
    double thresh=2.0;
    for (i=0;i<nelems;i++) 
	    if (elGnum[i]%2) {
	    	    //CkPrintf("[%d] List: Local %d, global %d)\n",myId,i,elGnum[i]);
		    elList[elListLen++]=i;
	    }

    //Get a list of ghost elements with odd global numbers
    FEM_Exchange_Ghost_Lists(0,elListLen,elList);
    elListLen=FEM_Get_Ghost_List_Length();
    FEM_Get_Ghost_List(elList);
    //CkPrintf("[%d] My ghost list has %d entries\n",myId,elListLen);
    //Make sure everything on the list are actually ghosts and
    // actually have large values
    for (i=0;i<elListLen;i++) {
	    testAssert(elList[i]<ngelems,"Ghost list ghost test (upper)");
	    testAssert(elList[i]>=nelems,"Ghost list ghost test (lower)");
	    testAssert(elGnum[elList[i]]%2,"Ghost list contents test");
    }

#if ENABLE_MIG /*Only works with -memory isomalloc*/
    CkPrintf("Before migrate: Thread %d on pe %d\n",myId,CkMyPe());
    FEM_Migrate();
    CkPrintf("After migrate: Thread %d on pe %d\n",myId,CkMyPe());
#endif

  }

#if 1
/*Try reassembling the mesh*/
    const int *noGnum=FEM_Get_Node_Nums();
    double *nodeOut=new double[nnodes];
    FEM_Set_Node(nnodes,1);
    for (i=0;i<nnodes;i++) {
    	nodeOut[i]=0.1*noGnum[i];
    }
    FEM_Set_Node_Data(nodeOut);
    delete[] nodeOut;
    const int *elGnum=FEM_Get_Elem_Nums();
    double *elOut=new double[nelems];
    FEM_Set_Elem(0,nelems,1,0);
    for (i=0;i<nelems;i++) {
    	elOut[i]=0.1*elGnum[i];
    }
    FEM_Set_Elem_Data(0,elOut);
    delete[] elOut;
    FEM_Update_Mesh(123,2);
#endif

  FEM_Print("All tests passed.");
}

extern "C" void
mesh_updated(int param)
{
  int nnodes,nelems,i,nodePer,dataPer;
  CkPrintf("mesh_updated(%d) called.\n",param);
  testEqual(param,123,"mesh_updated param");
  
  FEM_Get_Node(&nnodes,&dataPer);
  CkPrintf("Getting %d nodes (%d data per)\n",nnodes,dataPer);
  double *ndata=new double[nnodes*dataPer];
  FEM_Get_Node_Data(ndata);
  for (int i=0;i<nnodes;i++) {
    testEqual(ndata[i],0.1*i,"mesh_updated node values");
  }
  delete[] ndata;
  
  FEM_Get_Elem(0,&nelems,&dataPer,&nodePer);
  CkPrintf("Getting %d elems (%d data per)\n",nelems,dataPer);
  double *ldata=new double[nelems*dataPer];
  FEM_Get_Elem_Data(0,ldata);
  for (int i=0;i<nelems;i++) {
    testEqual(ldata[i],0.1*i,"mesh_updated elem values");
  }
  delete[] ldata;
}

extern "C" void
finalize(void)
{
  CkPrintf("finalize called\n");
}
