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
int tsteps=0;

extern "C" void
pupMyGlobals(pup_er p) 
{
	CkPrintf("pupMyGlobals on PE %d\n",CkMyPe());
	pup_int(p,&tsteps);
	if (pup_isUnpacking(p))
		reduceValues=new double[tsteps];
	pup_doubles(p,reduceValues,tsteps);
}

extern "C" void
TCharmUserNodeSetup(void)
{
	TCharmReadonlyGlobals(pupMyGlobals);
}

extern "C" void
init(void)
{
  CkPrintf("init called\n");
  tsteps=10;
  reduceValues=new double[tsteps];
  const int dim=10;//Length of one side of the FEM mesh
  const int np=4; //Nodes per element
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
     static const int quad2edge[]= {0,1,  1,2,  2,3,  3,0};
     FEM_Add_Ghost_Layer(2,1);
     FEM_Add_Ghost_Elem(0,4,quad2edge);
/*Add a second layer
     FEM_Add_Ghost_Layer(2,0);
     FEM_Add_Ghost_Elem(0,4,quad2edge);
*/
  }
  delete[] conn;
  delete[] nodes;
  delete[] elements;
  delete[] noData;
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
	if (shouldBe)
		CkPrintf("[chunk %d] %s test passed.\n",myPartition,what);
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
  int fid = FEM_Create_Field(FEM_DOUBLE, 1, 
	(char *)(&nodes[0].val)-(char *)nodes, sizeof(Node));
  int efid = FEM_Create_Field(FEM_DOUBLE, 1, 
	(char *)(&elements[0].val)-(char *)elements, sizeof(Element));
  int i;

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

//Test readonly global
  testEqual(tsteps,nnodeData,"readonly");

//Test barrier
  FEM_Print("Going to barrier");
  FEM_Barrier();
  FEM_Print("Back from barrier");

  int *elList=new int[ngelems];

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
    FEM_Update_Ghost_Field(efid,0,elements);

    //Communicate our ghost nodes:
    FEM_Update_Ghost_Field(fid,-1,nodes);
    for(i=nnodes;i<ngnodes;i++)
        testEqual(nodes[i].val,nodeData[nnodeData*i+t],
	   "update_ghost_node_field");


    //Make a list of elements with large values
    int elListLen=0;
    double thresh=2.0;
    for (i=0;i<nelems;i++) 
	    if (elements[i].val>thresh)
		    elList[elListLen++]=i;

    FEM_Barrier();

#if 0 /* This seems to hang (why?) */
    //Get a list of ghost elements with large values
    FEM_Exchange_Ghost_Lists(0,elListLen,elList);
    elListLen=FEM_Get_Ghost_List_Length();
    FEM_Print("GHOSTHANG> In");
    FEM_Get_Ghost_List(elList);
    FEM_Print("GHOSTHANG>     Back");
    CkPrintf("[%d] My ghost list has %d entries\n",myId,elListLen);
    //Make sure everything on the list are actually ghosts and
    // actually have large values
    for (i=0;i<elListLen;i++) {
	    testAssert(elList[i]<ngelems,"Ghost list ghost test (upper)");
	    testAssert(elList[i]>=nelems,"Ghost list ghost test (lower)");
	    testAssert(elements[elList[i]].val>thresh,"Ghost list contents test");
    }
#endif

    double *nodeOut=new double[nnodes];
    FEM_Set_Node(nnodes,1);
    for (i=0;i<nnodes;i++) nodeOut[i]=nodes[i].val;
    FEM_Set_Node_Data(nodeOut);
    delete[] nodeOut;
    FEM_Update_Mesh(1+t,0);
  }

  double localSum = 1.0,globalSum;
  FEM_Reduce(fid, &localSum, &globalSum, FEM_SUM);
  testEqual(globalSum,(double)FEM_Num_Partitions(),"reduce");
  FEM_Print("All tests passed.\n");

  FEM_Done();
}

extern "C" void
mesh_updated(int param)
{
  int nnodes,dataPer;
  CkPrintf("mesh_updated(%d) called.\n",param);
  FEM_Get_Node(&nnodes,&dataPer);
  double *ndata=new double[nnodes*dataPer];
  FEM_Get_Node_Data(ndata);
  double sum=0;
  for (int i=0;i<nnodes;i++)
    sum+=ndata[i];
  testAssert(sum==reduceValues[param-1],"mesh_updated",0);
}

extern "C" void
finalize(void)
{
  CkPrintf("finalize called\n");
}
