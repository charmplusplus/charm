/*
Triangle Mesh Refinement (TMR) demo code.

Reads in a Triangle mesh.
Passes it to the REFINE2D subsystem.
Asks for refinements occasionally.
*/
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include "charm++.h"
#include "fem.h"
/*My HEADERS*/
#include "ckvector3d.h"
#include "charm-api.h"
#include "fem_mesh.h"


#include "netfem.h"
#include "refine.h"
#include "femrefine.h"
#include "pgm.h"

//The material constants c, as computed by fortran mat_const
// I think the units here are Pascals (N/m^2)
const double matConst[4]={3.692e9,  1.292e9,  3.692e9,  1.200e9 };

//Material density, Kg/m^3
const double density=5.0*1000.0;
//Plate thickness, meters
const double thickness=0.0001;

//The timestep, in seconds
// This aught to be adjusted when the mesh changes!
const double dt=1.0e-12;

static void die(const char *str) {
  CkError("Fatal error: %s\n",str);
  CkExit();
}

/**schak*/
void resize_nodes(void *data,int *len,int *max);
void resize_elems(void *data,int *len,int *max);
void resize_edges(void *data,int *len,int *max);

#define NANCHECK 1 /*Check for NaNs at each timestep*/

extern "C" void
init(void)
{
  CkPrintf("init started\n");

  const char *eleName="xxx.1.ele";
  const char *nodeName="xxx.1.node";
	const char *edgeName="xxx.1.edge";
  int nPts=0; //Number of nodes
  vector2d *pts=0; //Node coordinates

  CkPrintf("Reading node coordinates from %s\n",nodeName);
  //Open and read the node coordinate file
  {
    char line[1024];
    FILE *f=fopen(nodeName,"r");
    if (f==NULL) die("Can't open node file!");
    fgets(line,1024,f);
    if (1!=sscanf(line,"%d",&nPts)) die("Can't read number of points!");
    pts=new vector2d[nPts];
    for (int i=0;i<nPts;i++) {
      int ptNo;
      if (NULL==fgets(line,1024,f)) die("Can't read node input line!");
      if (3!=sscanf(line,"%d%lf%lf",&ptNo,&pts[i].x,&pts[i].y)) 
	die("Can't parse node input line!");
    }
    fclose(f);
  }
  CkPrintf("Passing node coords to framework\n");

	//Register the node entity and its data arrays that will be used later. This
	// needs to be done so that the width of the data segments are set correctly and
	// can be used later
	FEM_Register_entity(FEM_Mesh_default_write(),FEM_NODE,NULL,nPts,nPts,resize_nodes);
	for(int k=0;k<=4;k++){
		if(k != 0){
			vector2d *t = new vector2d[nPts];
			FEM_Register_array(FEM_Mesh_default_write(),FEM_NODE,FEM_DATA+k,t,FEM_DOUBLE,2);
		}else{
			FEM_Register_array(FEM_Mesh_default_write(),FEM_NODE,FEM_DATA+k,pts,FEM_DOUBLE,2);
		}
	}
	double *td = new double[nPts];
	FEM_Register_array(FEM_Mesh_default_write(),FEM_NODE,FEM_DATA+5,td,FEM_DOUBLE,1);
	/*boundary value for nodes*/
	int *nodeBoundary = new int[nPts];
	FEM_Register_array(FEM_Mesh_default_write(),FEM_NODE,FEM_BOUNDARY,nodeBoundary,FEM_INT,1);
	

  int nEle=0;
  int *ele=NULL;
  CkPrintf("Reading elements from %s\n",eleName);
  //Open and read the element connectivity file
  {
    char line[1024];
    FILE *f=fopen(eleName,"r");
    if (f==NULL) die("Can't open element file!");
    fgets(line,1024,f);
    if (1!=sscanf(line,"%d",&nEle)) die("Can't read number of elements!");
    ele=new int[3*nEle];
    for (int i=0;i<nEle;i++) {
      int elNo;
      if (NULL==fgets(line,1024,f)) die("Can't read element input line!");
      if (4!=sscanf(line,"%d%d%d%d",&elNo,&ele[3*i+0],&ele[3*i+1],&ele[3*i+2])) 
	die("Can't parse element input line!");
      ele[3*i+0]--; //Fortran to C indexing
      ele[3*i+1]--; //Fortran to C indexing
      ele[3*i+2]--; //Fortran to C indexing
      
    }
    fclose(f);
  }
  
  CkPrintf("Passing elements to framework\n");
  
  // Register the Element entity and its connectivity array. Register the
  // data arrays to set up the widths correctly at the beginning
  FEM_Register_entity(FEM_Mesh_default_write(),FEM_ELEM,NULL,nEle,nEle,resize_nodes);
  FEM_Register_array(FEM_Mesh_default_write(),FEM_ELEM,FEM_CONN,ele,FEM_INDEX_0,3);
  
  for(int k=0;k<3;k++){
    void *t = new double[nEle];
    FEM_Register_array(FEM_Mesh_default_write(),FEM_ELEM,FEM_DATA+k,t,FEM_DOUBLE,1);
  }
  
  /*Build the ghost layer for refinement border*/
  FEM_Add_ghost_layer(2,0); /*2 nodes/tuple, do not add ghost nodes*/
  //FEM_Add_ghost_layer(2,1); /*2 nodes/tuple, do not add ghost nodes*/
  const static int tri2edge[6]={0,1, 1,2, 2,0};
  FEM_Add_ghost_elem(0,3,tri2edge);
  
	//open and read the .edge (edge connectivity file) needed for boundary values
	int nEdge;
	int *edgeConn;
	int *edgeBoundary;
	{
		char line[1024];
    FILE *f=fopen(edgeName,"r");
    if (f==NULL) die("Can't open edge file!");
    fgets(line,1024,f);
    if (1!=sscanf(line,"%d",&nEdge)) die("Can't read number of elements!");
    edgeConn = new int[2*nEdge];
		edgeBoundary = new int[nEdge];
		for(int i=0;i<nEdge;i++){
			int edgeNo;
			if (NULL==fgets(line,1024,f)) die("Can't read edge input line!");
			if (4 != sscanf(line,"%d%d%d%d",&edgeNo,&edgeConn[i*2+0],&edgeConn[i*2+1],&edgeBoundary[i])){
				die("Can't parse edge input line!");
			}
			edgeConn[i*2+0]--;
			edgeConn[i*2+1]--;		
		}
		fclose(f);
	}
	printf("Number of edges %d \n",nEdge);
	FEM_Register_entity(FEM_Mesh_default_write(),FEM_SPARSE,NULL,nEdge,nEdge,resize_edges);
	FEM_Register_array(FEM_Mesh_default_write(),FEM_SPARSE,FEM_CONN,edgeConn,FEM_INDEX_0,2);
	FEM_Register_array(FEM_Mesh_default_write(),FEM_SPARSE,FEM_BOUNDARY,edgeBoundary,FEM_INT,1);
  CkPrintf("Finished with init\n");
}

struct myGlobals {
  int nnodes,maxnodes;
  int nelems,maxelems;
	int nedges,maxedges;
  int *conn; //Element connectivity table

  vector2d *coord; //Undeformed coordinates of each node
  vector2d *R_net, *d, *v, *a; //Physical fields of each node
  double *m_i; //Inverse of mass at each node
  int m_i_fid; //Field ID for m_i
 	int *nodeBoundary; //node boundary
	
  double *S11, *S22, *S12; //Stresses for each element
	
	
	int *edgeConn;//edge Connectivity table
	int *edgeBoundary;//edge boundary value
};

void pup_myGlobals(pup_er p,myGlobals *g) 
{
  FEM_Print("-------- called pup routine -------");
  pup_int(p,&g->nnodes);
  pup_int(p,&g->nelems);
	pup_int(p,&g->nedges);
  pup_int(p,&g->maxelems);
  pup_int(p,&g->maxnodes);
	pup_int(p,&g->maxedges);
	
  int nnodes=g->nnodes, nelems=g->nelems;
  if (pup_isUnpacking(p)) {
    g->coord=new vector2d[g->maxnodes];
    g->conn=new int[3*g->maxelems];
    g->R_net=new vector2d[g->maxnodes]; //Net force
    g->d=new vector2d[g->maxnodes];//Node displacement
    g->v=new vector2d[g->maxnodes];//Node velocity
    g->a=new vector2d[g->maxnodes];
    g->m_i=new double[g->maxnodes];
		g->nodeBoundary = new int[g->maxnodes];
		
    g->S11=new double[g->maxelems];
    g->S22=new double[g->maxelems];
    g->S12=new double[g->maxelems];
		
		g->edgeConn = new int[2*g->maxedges];
		g->edgeBoundary = new int[g->maxedges];
  }
  pup_doubles(p,(double *)g->coord,2*nnodes);
  pup_ints(p,(int *)g->conn,3*nelems);
  pup_doubles(p,(double *)g->R_net,2*nnodes);
  pup_doubles(p,(double *)g->d,2*nnodes);
  pup_doubles(p,(double *)g->v,2*nnodes);
  pup_doubles(p,(double *)g->a,2*nnodes);
  pup_doubles(p,(double *)g->m_i,nnodes);
	pup_ints(p,(int *)g->nodeBoundary,nnodes);
	
  pup_doubles(p,(double *)g->S11,nelems);
  pup_doubles(p,(double *)g->S22,nelems);
  pup_doubles(p,(double *)g->S12,nelems);
	
	pup_ints(p,(int *)g->edgeConn,2*g->nedges);
	pup_ints(p,(int *)g->edgeBoundary,g->nedges);
	
  if (pup_isDeleting(p)) {
    delete[] g->coord;
    delete[] g->conn;
    delete[] g->R_net;
    delete[] g->d;
    delete[] g->v;
    delete[] g->a;
    delete[] g->m_i;
		delete[] g->nodeBoundary;
		delete[] g->S11;
		delete[] g->S22;
		delete[] g->S12;
		delete[] g->edgeConn;
		delete[] g->edgeBoundary;
  }
}


//Return the signed area of triangle i
double calcArea(myGlobals &g, int i)
{
	int n1=g.conn[3*i+0];
	int n2=g.conn[3*i+1];
	int n3=g.conn[3*i+2];
	vector2d a=g.coord[n1];
	vector2d b=g.coord[n2];
	vector2d c=g.coord[n3];
	c-=a; b-=a;
	double area=0.5*(b.x*c.y-c.x*b.y);
	return area;
}

// Check the quality of triangle i
void checkTriangle(myGlobals &g, int i)
{
	double area=calcArea(g,i);
	if (area<0) {
		CkError("Triangle %d of chunk %d is inverted! (area=%g)\n",
			i,FEM_My_partition(),area);
		CkAbort("Inverted triangle");
	}
	if (area<1.0e-15) {
		CkError("Triangle %d of chunk %d is a sliver!\n",i,FEM_My_partition());
		CkAbort("Sliver triangle");
	}
}

//Compute forces on constant-strain triangles:
void CST_NL(const vector2d *coor,const int *lm,vector2d *R_net,
	    const vector2d *d,const double *c,
	    int numnp,int numel,
	    double *S11o,double *S22o,double *S12o);

//Update node position, velocity, accelleration based on net force.
void advanceNodes(const double dt,int nnodes,const vector2d *coord,
		  vector2d *R_net,vector2d *a,vector2d *v,vector2d *d,
		  const double *m_i,bool dampen)
{
  const vector2d z(0,0);

  const double shearForce=1.0e-11/(dt*dt);

  bool someNaNs=false;
  int i;
  for (i=0;i<nnodes;i++) {
    vector2d R_n=R_net[i];
#if NANCHECK
    if (((R_n.x-R_n.x)!=0)) {
	    CkPrintf("R_net[%d]=NaN at (%.4f,%.4f)   ",i,coord[i].x,coord[i].y);
			CmiAbort("nan node");
	    someNaNs=true;
    }
    if (fabs(d[i].x)>1.0) {
	    CkPrintf("d[%d] %f large at (%.4f,%.4f)   ",i,d[i].x,coord[i].x,coord[i].y);
	    someNaNs=true;
    }
#endif
    R_net[i]=z;
//Apply boundary conditions (HACK: hardcoded!)
    if (1) {
       if (coord[i].x<0.00001)
	       R_n.y+=shearForce/m_i[i]; //Bottom edge pushed hard down
       if (coord[i].y>0.02-0.00001)
	       R_n=z; //Top edge held in place
    }
//Update displacement and velocity
    vector2d aNew=R_n*m_i[i];
    v[i]+=(dt*0.5)*(aNew+a[i]);
    d[i]+=dt*v[i]+(dt*dt*0.5)*aNew;
    a[i]=aNew;   
    //if (coord[i].y>0.02-0.00001) d[i].y=0.0; //Top edge in horizontal slot
  }
  if (dampen)
    for (i=0;i<nnodes;i++)
	  v[i]*=0.9; //Dampen velocity slightly (prevents eventual blowup)
	  
  if (someNaNs) {
	  CkPrintf("Nodes all NaN!\n");
	  CkAbort("Node forces NaN!");
  }
}

//Fill out the m_i array with the inverse of the node masses
void calcMasses(myGlobals &g) {
	int i;
	double *m_i=g.m_i;
	//Zero out node masses
	for (i=0;i<g.nnodes;i++) m_i[i]=0.0;
	//Add mass from surrounding triangles:
	for (i=0;i<g.nelems;i++) {
		int n1=g.conn[3*i+0];
		int n2=g.conn[3*i+1];
		int n3=g.conn[3*i+2];
		double area=calcArea(g,i);
		if (1 || i%100==0) CkPrintf("Triangle %d (%d %d %d) has area %.3g\n",i,n1,n2,n3,area);
		double mass=0.333*density*(thickness*area);
		m_i[n1]+=mass;
		m_i[n2]+=mass;
		m_i[n3]+=mass;
	}
	//Include mass from other processors
	FEM_Update_field(g.m_i_fid,m_i);
	//Invert masses to get m_i
	for (i=0;i<g.nnodes;i++) {
		double mass=m_i[i];
		if (mass<1.0e-10) m_i[i]=1.0; //Disconnected node (!)
		else m_i[i]=1.0/mass;
	}
}

void init_myGlobal(myGlobals *g){
	g->coord = g->R_net = g->d = g->v = g->a = NULL;
	g->m_i = NULL;
	g->conn = NULL;
	g->S11 = g->S22 = g->S12 = NULL;
	g->edgeConn = g->edgeBoundary = NULL;
}


void resize_nodes(void *data,int *len,int *max){
	printf("[%d] resize nodes called len %d max %d\n",FEM_My_partition(),*len,*max);
	FEM_Register_entity(FEM_Mesh_default_read(),FEM_NODE,data,*len,*max,resize_nodes);
	myGlobals *g = (myGlobals *)data;

	vector2d *coord=g->coord,*R_net=g->R_net,*d=g->d,*v=g->v,*a=g->a;
	double *m_i=g->m_i;
	int *nodeBoundary = g->nodeBoundary;
	
	
	g->coord=new vector2d[*max];
	g->coord[0].x = 0.9;
	g->coord[0].y = 0.8;
	g->maxnodes = *max;
  g->R_net=new vector2d[g->maxnodes]; //Net force
  g->d=new vector2d[g->maxnodes];//Node displacement
  g->v=new vector2d[g->maxnodes];//Node velocity
  g->a=new vector2d[g->maxnodes];//Node accelleration
  g->m_i=new double[g->maxnodes];//Node mass
	g->nodeBoundary = new int[g->maxnodes];//Node boundary
	
	if(coord != NULL){
		for(int k=0;k<*len;k++){
			printf("before resize node %d ( %.6f %.6f ) \n",k,coord[k].x,coord[k].y);
		}
	}	
	
	FEM_Register_array(FEM_Mesh_default_read(),FEM_NODE,FEM_DATA,(void *)g->coord,FEM_DOUBLE,2);
	FEM_Register_array(FEM_Mesh_default_read(),FEM_NODE,FEM_DATA+1,(void *)g->R_net,FEM_DOUBLE,2);
	FEM_Register_array(FEM_Mesh_default_read(),FEM_NODE,FEM_DATA+2,(void *)g->d,FEM_DOUBLE,2);
	FEM_Register_array(FEM_Mesh_default_read(),FEM_NODE,FEM_DATA+3,(void *)g->v,FEM_DOUBLE,2);
	FEM_Register_array(FEM_Mesh_default_read(),FEM_NODE,FEM_DATA+4,(void *)g->a,FEM_DOUBLE,2);
	FEM_Register_array_layout(FEM_Mesh_default_read(),FEM_NODE,FEM_DATA+5,(void *)g->m_i,g->m_i_fid);
	FEM_Register_array(FEM_Mesh_default_read(),FEM_NODE,FEM_BOUNDARY,(void *)g->nodeBoundary,FEM_INT,1);

	for(int k=0;k<*len;k++){
		printf("after resize node %d ( %.6f %.6f )\n",k,g->coord[k].x,g->coord[k].y);
	}


	if(coord != NULL){
		delete [] coord;
		delete [] R_net;
		delete [] d;
		delete [] v;
		delete [] a;
		delete [] m_i;
		delete [] nodeBoundary;
	}

};


void resize_elems(void *data,int *len,int *max){
	printf("[%d] resize elems called len %d max %d\n",FEM_My_partition(),*len,*max);
	FEM_Register_entity(FEM_Mesh_default_read(),FEM_ELEM,data,*len,*max,resize_elems);
	myGlobals *g = (myGlobals *)data;
	
	int *conn=g->conn;
	double *S11 = g->S11,*S22 = g->S22,*S12 = g->S12;
	
	g->conn = new int[3*(*max)];
	g->maxelems = *max;
 
 	g->S11=new double[g->maxelems];
  g->S22=new double[g->maxelems];
  g->S12=new double[g->maxelems];
	
	FEM_Register_array(FEM_Mesh_default_read(),FEM_ELEM,FEM_CONN,(void *)g->conn,FEM_INDEX_0,3);	
	CkPrintf("Connectivity array starts at %p \n",g->conn);
	FEM_Register_array(FEM_Mesh_default_read(),FEM_ELEM,FEM_DATA,(void *)g->S11,FEM_DOUBLE,1);	
	FEM_Register_array(FEM_Mesh_default_read(),FEM_ELEM,FEM_DATA+1,(void *)g->S22,FEM_DOUBLE,1);	
	FEM_Register_array(FEM_Mesh_default_read(),FEM_ELEM,FEM_DATA+2,(void *)g->S12,FEM_DOUBLE,1);	
	
	if(conn != NULL){
		delete [] conn;
		delete [] S11;
		delete [] S22;
		delete [] S12;
	}
};

void resize_edges(void *data,int *len,int *max){
	printf("[%d] resize edges called len %d max %d\n",FEM_My_partition(),*len,*max);
	FEM_Register_entity(FEM_Mesh_default_read(),FEM_SPARSE,data,*len,*max,resize_edges);
	myGlobals *g = (myGlobals *)data;
	
	int *conn = g->edgeConn;
	int *bound = g->edgeBoundary;

	g->maxedges = *max;	
	g->edgeConn = new int[2*(*max)];
	g->edgeBoundary = new int[(*max)];
	
	FEM_Register_array(FEM_Mesh_default_read(),FEM_SPARSE,FEM_CONN,(void *)g->edgeConn,FEM_INDEX_0,2);	
	FEM_Register_array(FEM_Mesh_default_read(),FEM_SPARSE,FEM_BOUNDARY,(void *)g->edgeBoundary,FEM_INT,1);
	if(conn != NULL){
		delete [] conn;
		delete [] bound;	
	}
}



void repeat_after_split(void *data){
	myGlobals *g = (myGlobals *)data;
	g->nelems = FEM_Mesh_get_length(FEM_Mesh_default_read(),FEM_ELEM);
	g->nnodes = FEM_Mesh_get_length(FEM_Mesh_default_read(),FEM_NODE);
	for(int k=0;k<g->nnodes;k++){
		printf(" node %d ( %.6f %.6f )\n",k,g->coord[k].x,g->coord[k].y);
	}

	calcMasses(*g);
};


extern "C" void
driver(void)
{
  int ignored;
  int i;  
  int myChunk=FEM_My_partition();

/*Add a refinement object to FEM array*/
CkPrintf("[%d] begin init\n",myChunk);
  FEM_REFINE2D_Init();
CkPrintf("[%d] end init\n",myChunk);

  myGlobals g;
  FEM_Register(&g,(FEM_PupFn)pup_myGlobals);

	init_myGlobal(&g);
  
 	g.nnodes = FEM_Mesh_get_length(FEM_Mesh_default_read(),FEM_NODE);
	int maxNodes = g.nnodes;
  g.maxnodes=2*maxNodes;
  
	g.m_i_fid=FEM_Create_field(FEM_DOUBLE,1,0,sizeof(double));

	
	resize_nodes((void *)&g,&g.nnodes,&maxNodes);
  
	
	int nghost=0;
  
	g.nelems=FEM_Mesh_get_length(FEM_Mesh_default_read(),FEM_ELEM);
  g.maxelems=g.nelems;

	resize_elems((void *)&g,&g.nelems,&g.maxelems);
	
	g.nedges = FEM_Mesh_get_length(FEM_Mesh_default_read(),FEM_SPARSE);
	g.maxedges = g.nedges;
	resize_edges((void *)&g,&g.nedges,&g.maxedges);

	FEM_REFINE2D_Newmesh(FEM_Mesh_default_read(),FEM_NODE,FEM_ELEM);
  
  
  //Initialize associated data
  for (i=0;i<g.maxnodes;i++){
    g.R_net[i]=g.d[i]=g.v[i]=g.a[i]=vector2d(0.0);
	}

//Apply a small initial perturbation to positions
  for (i=0;i<g.nnodes;i++) {
	  const double max=1.0e-15/15.0; //Tiny perturbation
	  g.d[i].x+=max*(i&15);
	  g.d[i].y+=max*((i+5)&15);
  }

  int fid=FEM_Create_field(FEM_DOUBLE,2,0,sizeof(vector2d));
  
  for (i=0;i<g.nelems;i++){
    checkTriangle(g,i);
	}	

  //Timeloop
  if (CkMyPe()==0){
    CkPrintf("Entering timeloop\n");
	}	
  int tSteps=0x70FF00FF;
  calcMasses(g);
  double startTime=CkWallTimer();
  double curArea=2.0e-5;
  for (int t=0;t<tSteps;t++) {
    if (1) { //Structural mechanics
    	//Compute forces on nodes exerted by elements
			CST_NL(g.coord,g.conn,g.R_net,g.d,matConst,g.nnodes,g.nelems,g.S11,g.S22,g.S12);
	
	    //Communicate net force on shared nodes
			FEM_Update_field(fid,g.R_net);

	    //Advance node positions
			advanceNodes(dt,g.nnodes,g.coord,g.R_net,g.a,g.v,g.d,g.m_i,(t%4)==0);
    
    }

    //Debugging/perf. output
    double curTime=CkWallTimer();
    double total=curTime-startTime;
    startTime=curTime;
    if (CkMyPe()==0 && (t%64==0))
	    CkPrintf("%d %.6f sec for loop %d \n",CkNumPes(),total,t);
 /*   if (0 && t%16==0) {
	    CkPrintf("    Triangle 0:\n");
	    for (int j=0;j<3;j++) {
		    int n=g.conn[0][j];
		    CkPrintf("    Node %d: coord=(%.4f,%.4f)  d=(%.4g,%.4g)\n",
			     n,g.coord[n].x,g.coord[n].y,g.d[n].x,g.d[n].y);
	    }
    }*/
//    if (t%512==0)
//      FEM_Migrate();

    if (t%128==0) { //Refinement:
      vector2d *loc=new vector2d[2*g.nnodes];
      for (i=0;i<g.nnodes;i++) {
				loc[i]=g.coord[i];//+g.d[i];
      }
      double *areas=new double[g.nelems];
      curArea=curArea*0.98;
      for (i=0;i<g.nelems;i++) {
      #if 0
        double origArea=8e-8; //Typical triangle size
	if (fabs(g.S12[i])>1.0e8)
		areas[i]=origArea*0.9; //Refine stuff that's stressed
	else
		areas[i]=origArea; //Leave everything else big
      #endif
        areas[i]=curArea;
      }
      
      CkPrintf("[%d] Starting refinement step: %d nodes, %d elements to %.3g\n",
	       myChunk,g.nnodes,g.nelems,curArea);
			
			FEM_REFINE2D_Split(FEM_Mesh_default_read(),FEM_NODE,(double *)loc,FEM_ELEM,areas,FEM_SPARSE);
			repeat_after_split((void *)&g);

			
      g.nelems = FEM_Mesh_get_length(FEM_Mesh_default_read(),FEM_ELEM);
			g.nnodes = FEM_Mesh_get_length(FEM_Mesh_default_read(),FEM_NODE);
             
      CkPrintf("[%d] Done with refinement step: %d nodes, %d elements\n",
	       myChunk,g.nnodes,g.nelems);
      
    }
    
    if (1) { //Publish data to the net
	    NetFEM n=NetFEM_Begin(myChunk,t,2,NetFEM_POINTAT);
	    
	    NetFEM_Nodes(n,g.nnodes,(double *)g.coord,"Position (m)");
	    NetFEM_Vector(n,(double *)g.d,"Displacement (m)");
	    NetFEM_Vector(n,(double *)g.v,"Velocity (m/s)");
	    
	    NetFEM_Elements(n,g.nelems,3,(int *)g.conn,"Triangles");
		NetFEM_Scalar(n,g.S11,1,"X Stress (pure)");
		NetFEM_Scalar(n,g.S22,1,"Y Stress (pure)");
		NetFEM_Scalar(n,g.S12,1,"Shear Stress (pure)");
	    
	    NetFEM_End(n);
    }
  }

  if (CkMyPe()==0)
    CkPrintf("Driver finished\n");
}


