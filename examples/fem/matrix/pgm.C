/*
Test out iterative (matrix-based) FEM interface.

Orion Sky Lawlor, olawlor@acm.org, 1/27/2003 
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "charm++.h"
#include "tcharmc.h"
#include "fem.h"
#include "netfem.h"
#include "ifemc.h"

#if CMK_HAS_SLEEP
#include <unistd.h>
#endif

//Number of time steps to simulate
int tsteps=10;
int dim=10;//Elements per side of the FEM mesh
const int np=4; //Nodes per element for a quad
const int width=3; //Number of dimensions to solve for (1, 2, or 3)

//Index of a node at x,y
int nodeDex(int x,int y) {return x+y*(dim+1);}
//Index of element connecting x,y to x+1,y+1
int elemDex(int x,int y) {return x+y*dim;}

// An imposed displacment boundary condition
struct boundaryCondition {
	int node; //node this BC applies to
	int dof; //Coordinate axis this BC applies to
	double disp; //Imposed displacement value
};
int dof(const boundaryCondition &b) {
  return b.node*width+b.dof;
}

// Save or restore this list of boundary conditions to/from the FEM framework
void fem_boundaryConditions(int mesh,int entity,int n,boundaryCondition *bc) 
{
	int distance=sizeof(boundaryCondition);
	FEM_Mesh_data_offset(mesh,entity,FEM_CONN, bc, 0,n, 
		IDXL_INDEX_0, 1,offsetof(boundaryCondition,node),distance,0);
	
	FEM_Mesh_data_offset(mesh,entity,FEM_DATA+0, bc, 0,n, 
		IDXL_INT, 1,offsetof(boundaryCondition,dof),distance,0);
	
	FEM_Mesh_data_offset(mesh,entity,FEM_DATA+1, bc, 0,n, 
		IDXL_DOUBLE, 1, offsetof(boundaryCondition,disp),distance,0);
}

extern "C" void
init(void)
{
  CkPrintf("init called\n");
  int argc=CkGetArgc();
  char **argv=CkGetArgv();
  if (argc>1) dim=atoi(argv[1]);
  int nelems=dim*dim, nnodes=(dim+1)*(dim+1);
  CkPrintf("Generating %d elements, %d node serial mesh\n",nelems,nnodes);
  
  //Describe the nodes and elements
  int mesh=FEM_Mesh_default_write();
  
  int x,y,e;
  
//Create node coordinates and initial conditions
  double *nodes=new double[3*nnodes];
  double domainX=2.0; //Size of domain, meters
  double domainY=2.0;
  for(y=0;y<dim+1;y++) 
  for (x=0;x<dim+1;x++) {
    nodes[nodeDex(x,y)*width+0]=domainX/float(dim)*x;
    if (width>1) 
      nodes[nodeDex(x,y)*width+1]=domainY/float(dim)*y;
    if (width>2)
      nodes[nodeDex(x,y)*width+2]=0.1;
  }
  FEM_Mesh_data(mesh,FEM_NODE,FEM_DATA, nodes,
  	0,nnodes, FEM_DOUBLE,width);

  int i,j,n,c;
  int length=width*nnodes;
  
  // Prepare net forces (boundary conditions):
  double *netforce=new double[length];
  double domainDensity=0.7*1.0e3; //Domain's density, Kg/m^3
  double domainThickness=0.02; // Domain thickness, meters
  double domainVolume=domainThickness*domainX*domainY;
  double domainMass=domainDensity*domainVolume; //Entire domain's mass, Kg
  double nodeMass=domainMass/((dim)*(dim)); //Node's mass, Kg
  for (n=0;n<nnodes;n++) {
    for (c=0;c<width;c++) {
      double f=0;
      if (c==width-1) f=9.8*nodeMass; //Weight of node, in Newtons
      netforce[n*width+c]=f;
    }
  }
  FEM_Mesh_data(mesh,FEM_NODE,FEM_DATA+1, netforce,
  	0,nnodes, FEM_DOUBLE,width);
  
  // Prepare imposed displacements (boundary conditions)
  int nBC=2*width;
  boundaryCondition *bc=new boundaryCondition[nBC];
  for (c=0;c<width;c++) {
    bc[0*width+c].node=0; //Left end: all displacments==0.1
    bc[0*width+c].dof=c;
    bc[0*width+c].disp=0.1;
    bc[1*width+c].node=nnodes-1; //Right end: all == 0.2
    bc[1*width+c].dof=c;
    bc[1*width+c].disp=0.2;
  }
  fem_boundaryConditions(mesh,FEM_SPARSE+0,nBC,bc);

//Create the connectivity array
  int *conn=new int[nelems*np];
  for (y=0;y<dim;y++) 
  for (x=0;x<dim;x++) {
  	   e=elemDex(x,y);
	   conn[e*np+0]=nodeDex(x  ,y  );
	   conn[e*np+1]=nodeDex(x+1,y  );
	   conn[e*np+2]=nodeDex(x+1,y+1);
	   conn[e*np+3]=nodeDex(x  ,y+1);
  }
  FEM_Mesh_conn(mesh,FEM_ELEM+0, conn,
  	0,nelems, np);
  delete[] conn;

}

extern "C"
int compareInts(const void *a,const void *b) {
  return *(int *)a - *(int *)b;
}

/// User-defined class: contains my part of the FEM problem
class myMesh {
  int myId;
  int mesh;

  int nnodes;
  double *coord; //Node coordinates
  double *netforce; //(knowns) Node net forces (plus forces from fixed displacements)
  double *disp; //(unknowns) Node displacements 
  
  int nBC; //Number of fixed displacement boundary conditions
  boundaryCondition *bc;
  
  int nelems;
  int *conn; //Element->node mapping
  
  int mvps; //counter for matrix-vector-products
  
  // Return the undisplaced distance between these two nodes
  double dist(int n1,int n2) {
    double sum=0;
    for (int i=0;i<width;i++) {
      double del=coord[n1*width+i]-coord[n2*width+i];
      sum+=del*del;
    }
    return sqrt(sum);
  }
public:
  myMesh(int mesh_)
  	:mesh(mesh_)
  {
    myId = FEM_My_partition();
    nnodes=FEM_Mesh_get_length(mesh,FEM_NODE);
    coord=new double[width*nnodes];
    FEM_Mesh_data(mesh,FEM_NODE,FEM_DATA, coord,
  	0,nnodes, FEM_DOUBLE,width);
    
    netforce=new double[width*nnodes];
    FEM_Mesh_data(mesh,FEM_NODE,FEM_DATA+1, netforce,
  	0,nnodes, FEM_DOUBLE,width);
    disp=new double[width*nnodes];
    
    nelems=FEM_Mesh_get_length(mesh,FEM_ELEM);
    conn=new int[np*nelems];
    FEM_Mesh_conn(mesh,FEM_ELEM+0, conn,
  	0,nelems, np);

    nBC=FEM_Mesh_get_length(mesh,FEM_SPARSE+0);
    bc=new boundaryCondition[nBC];
    fem_boundaryConditions(mesh,FEM_SPARSE+0,nBC,bc);
    
    mvps=0;
  }
  ~myMesh() {
    delete[] coord;
    delete[] netforce;
    delete[] disp;
    delete[] bc;
    delete[] conn;
  }
  
  // Solve our system using this solver
  void solve(ILSI_Solver s);
  
  // Compute the forces we exert under these displacements.
  //  If skipBoundaries is true, it's like zeroing out the rows and 
  //  columns corresponding to boundary nodes.
  void applyElementForces(const double *disp, double *force) 
  {
     if ((myId==0) && (mvps++)%32==0) //FEM_My_partition()==0) 
     	CkPrintf("MatrixVectorProduct (d=%g,%g,%g)\n",disp[0],disp[1],disp[2]);
     
     //Zero out node forces
     int i,length=width*nnodes; //i loops over entries of our vectors
     for (i=0;i<length;i++) force[i]=0;
     
     /**
      * Add in (stupid) forces from local elements:
      *    n1 - A - n2
      *    |        |
      *    D        B
      *    |        |
      *    n4 - C - n3
      */
     for (int e=0;e<nelems;e++) {
       int n1=conn[np*e+0];
       int n2=conn[np*e+1];
       int n3=conn[np*e+2];
       int n4=conn[np*e+3];
       const double k=1.0e2; //Spring stiffness, N/m^2 (?)
       double kA=k*(1.0/dist(n1,n2)); //Spring constant, N/m
       double kB=k*(1.0/dist(n2,n3)); //Spring constant, N/m
       double kC=k*(1.0/dist(n3,n4)); //Spring constant, N/m
       double kD=k*(1.0/dist(n4,n1)); //Spring constant, N/m
       for (int c=0;c<width;c++) {
         double f;
	 f=-kA*(disp[n1*width+c]-disp[n2*width+c]);
         force[n1*width+c]+=f; force[n2*width+c]-=f;
	 
         f=-kB*(disp[n2*width+c]-disp[n3*width+c]);
         force[n2*width+c]+=f; force[n3*width+c]-=f;
	 
         f=-kC*(disp[n3*width+c]-disp[n4*width+c]);
         force[n3*width+c]+=f; force[n4*width+c]-=f;
	 
         f=-kD*(disp[n4*width+c]-disp[n1*width+c]);
         force[n4*width+c]+=f; force[n1*width+c]-=f;
       }
     }
     
     // Communicate forces from remote elements
     IDXL_Layout_t layout=IDXL_Layout_create(IDXL_DOUBLE,width);
     FEM_Update_field(layout,force);
     IDXL_Layout_destroy(layout);
  }
  
  void netfem(int ts) {
    NetFEM f=NetFEM_Begin(FEM_My_partition(),ts,width,NetFEM_POINTAT);
      NetFEM_Nodes(f,nnodes,coord,"Node Locs");
        NetFEM_Vector(f,disp,"Displacement");
        NetFEM_Vector(f,netforce,"Net force");
      NetFEM_Elements(f,nelems,np,conn,"Elements");
    NetFEM_End(f);
  }
};

extern "C" 
void mesh_matrix_product(void *ptr,
        int length,int width,const double *src, double *dest)
{
	myMesh *m=(myMesh *)ptr;
	m->applyElementForces(src,dest);
}


// Solve our system using this solver
void myMesh::solve(ILSI_Solver s) {
//Split up boundary conditions into DOF and value arrays:
  int *bcDOF=new int[nBC];
  double *bcValue=new double[nBC];
  int i;
  for (i=0;i<nBC;i++) {
    bcDOF[i]=dof(bc[i]);
    bcValue[i]=bc[i].disp;
  }
  
//Pick a reasonable initial guess
  for (i=0;i<width*nnodes;i++) disp[i]=0;

//Solve the system
  ILSI_Param param;
  ILSI_Param_new(&param);
  param.maxResidual=1.0e-6;
  // param.maxIterations=10000;
  if (myId==0) CkPrintf("Solving...\n");
  IFEM_Solve_shared_bc(ILSI_CG_Solver,&param,
  	mesh, FEM_NODE, nnodes,width,
	nBC,bcDOF,bcValue,
	mesh_matrix_product,this,netforce,disp);
  
  delete[] bcDOF; delete[] bcValue;
  
//Compute the actual forces applied to each node:
  applyElementForces(disp,netforce);
  
  if (myId==0) {
    CkPrintf("Solved-- %d iterations, %g residual\n",
    	(int)param.iterations,param.residual);
    if (nnodes<10)
      for (int n=0;n<nnodes;n++) {
        int c;
        printf("  node %d  disp= ",n);
	for (c=0;c<width;c++) printf(" %g",disp[width*n+c]);
	printf("  ");
        printf("  force= ",n);
	for (c=0;c<width;c++) printf(" %g",netforce[width*n+c]);
	printf("\n");
      }
  }
}

extern "C" void
driver(void)
{
  FEM_Print("Starting driver...");
  int myId = FEM_My_partition();

  int fem_mesh=FEM_Mesh_default_read();
  myMesh mesh(fem_mesh);
  
  // Solve a little problem on the mesh:
  mesh.solve(ILSI_CG_Solver);

  if (0) { //Loop for netfem access
    if (myId==0)
      FEM_Print("Waiting for NetFEM client to connect (hit ctrl-c to exit)");
    int ts=0;
    while(1) {
#if CMK_HAS_SLEEP
      sleep(1);
#endif
      mesh.netfem(ts);
      ts++;
      FEM_Barrier();
    }
  }
}

extern "C" void
mesh_updated(int param)
{
  CkPrintf("mesh_updated(%d) called.\n",param);
}
