#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "charm++.h"
#include "fem.h"

//These values are the sum of the node values
// after each timestep.
//If you change the mesh size, initial conditions, etc.
//  you'll have to update this array with the correct values,
//  which are (conveniently) printed out by init.
const double reduceValues[4]={512.00, 512.00, 465.00, -1};

extern "C" void
init(void)
{
  CkPrintf("init called\n");
  const int tsteps=3;//Number of time steps to simulate
  const int dim=5;//Length of one side of the FEM mesh
  const int np=4; //Nodes per element
  int conn[dim*dim*np];
  double elements[dim*dim]={0.0};
  double nodes[(dim+1)*(dim+1)]={0.0};
  double noData[(dim+1)*(dim+1)*tsteps];

  int nelems=dim*dim, nnodes=(dim+1)*(dim+1);

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
  CkPrintf("const double reduceValues[%d]={",tsteps+1);
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
    CkPrintf("%.2f, ",reduceSum);
  }
  FEM_Set_Node_Data(noData);
  CkPrintf("-1};\n");
}

typedef struct _node {
  double val;
} Node;

typedef struct _element {
  double val;
} Element;

extern "C" void
driver(void)
{
  int nnodes,nelems,nnodeData,nelemData,np;

  FEM_Get_Node(&nnodes,&nnodeData);
  double *nodeData=new double[nnodeData*nnodes];
  FEM_Get_Node_Data(nodeData);  

  FEM_Get_Elem(0,&nelems,&nelemData,&np);
  int *conn=new int[np*nelems];
  FEM_Get_Elem_Conn(0,conn);
  double *elData=new double[nelemData*nelems];
  FEM_Get_Elem_Data(0,elData);

  int tsteps=nnodeData;

  int myId = FEM_My_Partition();
  FEM_Print_Partition();
  Node *nodes = new Node[nnodes];
  Element *elements = new Element[nelems];
  int i;

//Set initial conditions
  for(i=0;i<nnodes;i++) {
    nodes[i].val = 0;
  }
  for(i=0;i<nelems;i++) { elements[i].val = elData[i]; }
  int fid = FEM_Create_Field(FEM_DOUBLE, 1, 0, sizeof(Node));

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
    int failed = 0;
    for(i=0;i<nnodes;i++)
        if(nodes[i].val != nodeData[nnodeData*i+t]) 
			 failed = 1;
    
    if(failed==0) {
      CkPrintf("[chunk %d] update_field %d test passed.\n", myId, t);
    } else {
      CkPrintf("[chunk %d] update_field %d test failed.\n", myId, t);
    }

    double sum = 0.0;
    FEM_Reduce_Field(fid, nodes, &sum, FEM_SUM);
    if(sum==reduceValues[t]) {
      CkPrintf("[chunk %d] reduce_field test passed.\n", myId);
    } else {
      CkPrintf("[chunk %d] reduce_field test failed (sum=%f at %d).\n", myId, sum, t);
    }

    FEM_Set_Node(nnodes,1);
    FEM_Set_Node_Data((double *)nodes);
    FEM_Update_Mesh(1+t,0);
  }

  double localSum = 1.0,globalSum;
  FEM_Reduce(fid, &localSum, &globalSum, FEM_SUM);
  if(globalSum==(double)FEM_Num_Partitions()) {
    CkPrintf("[chunk %d] reduce test passed.\n", myId);
  } else {
    CkPrintf("[chunk %d] reduce test failed.\n", myId);
  }

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
  if (sum!=reduceValues[param-1])
    CkPrintf("mesh_updated test FAILED! expected %f; got %f!\n",reduceValues[param-1],sum);
  else
    CkPrintf("mesh_updated test passed for %d\n",param);
  delete[] ndata;
}

extern "C" void
finalize(void)
{
  CkPrintf("finalize called\n");
}
