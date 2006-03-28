/*
 Charm++ Finite-Element Framework Program:
 
 Refine to 1 Billion Elements bwahahaha

*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "charm++.h"
#include "mpi.h"
#include "ckvector3d.h"
#include "charm-api.h"
#include "ParFUM.h"
#include "ParFUM_internals.h"

#include "vector2d.h"


extern void _registerParFUM(void);

//One element's connectivity information
typedef int connRec[3];

double start_time;

void print_mem_usage(){
  unsigned long memory_used = CmiMemoryUsage();
  unsigned long max_mem_used = 0;
  int myId=FEM_My_partition();

  CkAssert(MPI_SUCCESS == MPI_Reduce(&memory_used, &max_mem_used, 1, MPI_UNSIGNED_LONG, MPI_MAX ,0, MPI_COMM_WORLD));
  if(myId==0)
    CkPrintf("Max Memory Usage on a node=%ld MB\n", max_mem_used / 1024 / 1024);
}


extern "C" void
init(void)
{
  CkPrintf("init started\n");
  const char *eleName="xxx.1.ele";
  const char *nodeName="xxx.1.node";
  int nPts=0; //Number of nodes
  vector2d *pts=0; //Node coordinates
  int *bound=0; //Node coordinates

  CkPrintf("Reading node coordinates from %s\n",nodeName);
  //Open and read the node coordinate file
  {
    char line[1024];
    FILE *f=fopen(nodeName,"r");
    if (f==NULL) CkAbort("die!\n");
    fgets(line,1024,f);
    if (1!=sscanf(line,"%d",&nPts))  CkAbort("die!\n");
    pts=new vector2d[nPts];
    bound = new int[nPts];
    for (int i=0;i<nPts;i++) {
      int ptNo;
      if (NULL==fgets(line,1024,f))  CkAbort("die!\n");
      if (4!=sscanf(line,"%d%lf%lf%d",&ptNo,&pts[i].x,&pts[i].y,&bound[i])) 
        CkAbort("die!\n");
    }
    fclose(f);
  }
 
  int nEle=0;
  connRec *ele=NULL;
  CkPrintf("Reading elements from %s\n",eleName);
  //Open and read the element connectivity file
  {
    char line[1024];
    FILE *f=fopen(eleName,"r");
    if (f==NULL) CkAbort("Can't open element file!");
    fgets(line,1024,f);
    if (1!=sscanf(line,"%d",&nEle))  CkAbort("CkAbort!\n");
    ele=new connRec[nEle];
    for (int i=0;i<nEle;i++) {
      int elNo;
      if (NULL==fgets(line,1024,f))  CkAbort("CkAbort!\n");
      if (4!=sscanf(line,"%d%d%d%d",&elNo,&ele[i][0],&ele[i][1],&ele[i][2])) 
	CkAbort("Can't parse element input line!");  
      ele[i][0]--; //Fortran to C indexing
      ele[i][1]--; //Fortran to C indexing
      ele[i][2]--; //Fortran to C indexing
    }
    fclose(f);
  }
 


  int mesh=FEM_Mesh_default_write(); // Tell framework we are writing to the mesh

  CkPrintf("Passing node coords to framework\n");

  /*   Old versions used FEM_Set_node() and FEM_Set_node_data()
   *   New versions use the more flexible FEM_Set_Data()
   */

  FEM_Mesh_data(mesh,        // Add nodes to the current mesh
                FEM_NODE,        // We are registering nodes
                FEM_DATA+0,      // Register the point locations which are normally 
                                 // the first data elements for an FEM_NODE
                (double *)pts,   // The array of point locations
                0,               // 0 based indexing
                nPts,            // The number of points
                FEM_DOUBLE,      // Coordinates are doubles
                2);              // Points have dimension 2 (x,y)


  //add boundaries to the mesh
  FEM_Mesh_data(mesh,        // Add nodes to the current mesh
                FEM_NODE,        // We are registering nodes
                FEM_DATA+1,      // Register the point locations which are normally 
                                 // the first data elements for an FEM_NODE
                (int *)bound,   // The array of point locations
                0,               // 0 based indexing
                nPts,            // The number of points
                FEM_INT,      // Coordinates are doubles
                1);              // Points have dimension 2 (x,y)
 

  CkPrintf("Passing elements to framework\n");

  /*   Old versions used FEM_Set_elem() and FEM_Set_elem_conn() 
   *   New versions use the more flexible FEM_Set_Data()
   */

  FEM_Mesh_data(mesh,      // Add nodes to the current mesh
                FEM_ELEM+0,      // We are registering elements with type 0
                                 // The next type of element could be registered with FEM_ELEM+1
                FEM_CONN,        // Register the connectivity table for this
                                 // data elements for this type of FEM entity
                (int *)ele,      // The array of point locations
                0,               // 0 based indexing
                nEle,            // The number of elements
                FEM_INDEX_0,     // We use zero based node numbering
                3);              // Elements have degree 3, since triangles are defined 
                                 // by three nodes
 
   
  delete[] ele;
  delete[] pts;
  delete[] bound;

  // Add the nodal ghost layer. Not edge based.
  const int trianglefaces[3] = {0,1,2};
  FEM_Add_ghost_layer(1,1);
  FEM_Add_ghost_elem(0,3,trianglefaces);

}


// A driver() function 
// driver() is required in all FEM programs
extern "C" void
driver(void)
{
  int nnodes,nelems,ignored;
  int myId=FEM_My_partition();

  int mesh=FEM_Mesh_default_read(); // Tell framework we are reading data from the mesh

  if(myId==0)
    printf("In driver()\n");

  _registerParFUM();

  if(myId==0)
    start_time = CmiWallTimer();


  FEM_Mesh_allocate_valid_attr(mesh, FEM_ELEM+0);
  FEM_Mesh_allocate_valid_attr(mesh, FEM_NODE);


  long int elems;
  long int total_element_count;
  FEM_ADAPT_Init(mesh);  // setup the valid and adjs
  FEM_Mesh *meshP = FEM_Mesh_lookup(FEM_Mesh_default_read(),"driver");
  FEM_Adapt_Algs *adaptAlgs= meshP->getfmMM()->getfmAdaptAlgs();
  adaptAlgs->FEM_Adapt_Algs_Init(FEM_DATA+0,FEM_DATA+1);
  adaptAlgs->SetMeshSize(0, 1, NULL);

  double sizing = 1000000.0;
  int i=0;
  while(1){
    i++;
    // Terry's refine doesn't work
    //FEM_ADAPT_Refine(mesh, 0, 0, sizing, NULL);
      
    // Nilesh's refine works
    adaptAlgs->simple_refine(sizing);
    
    MPI_Barrier(MPI_COMM_WORLD);
    elems = FEM_count_valid(mesh, FEM_ELEM+0);
    total_element_count = 0;
    CkAssert(MPI_SUCCESS == MPI_Reduce(&elems, &total_element_count, 1, MPI_LONG, MPI_SUM ,0, MPI_COMM_WORLD));
    if(myId==0)
      CkPrintf("after iteration %d: Total Elements=%ld Time=%lf Sizing=%.20lf\n", i, total_element_count, CmiWallTimer()-start_time, sizing);
    
    print_mem_usage();
    
    if(total_element_count > 200*1000)
      CkExit();

    sizing = 0.7 * sizing;
  }    

}
