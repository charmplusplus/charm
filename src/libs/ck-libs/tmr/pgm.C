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
#include "netfem.h"
#include "refine.h"
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


#define NANCHECK 1 /*Check for NaNs at each timestep*/

extern "C" void
init(void)
{
  CkPrintf("init started\n");

  const char *eleName="xxx.1.ele";
  const char *nodeName="xxx.1.node";
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
  FEM_Set_node(nPts,2);
  FEM_Set_node_data((double *)pts);
  delete[] pts;

  int nEle=0;
  connRec *ele=NULL;
  CkPrintf("Reading elements from %s\n",eleName);
  //Open and read the element connectivity file
  {
    char line[1024];
    FILE *f=fopen(eleName,"r");
    if (f==NULL) die("Can't open element file!");
    fgets(line,1024,f);
    if (1!=sscanf(line,"%d",&nEle)) die("Can't read number of elements!");
    ele=new connRec[nEle];
    for (int i=0;i<nEle;i++) {
      int elNo;
      if (NULL==fgets(line,1024,f)) die("Can't read element input line!");
      if (4!=sscanf(line,"%d%d%d%d",&elNo,&ele[i][0],&ele[i][1],&ele[i][2])) 
	die("Can't parse element input line!");
      ele[i][0]--; //Fortran to C indexing
      ele[i][1]--; //Fortran to C indexing
      ele[i][2]--; //Fortran to C indexing
      
    }
    fclose(f);
  }
  
  CkPrintf("Passing elements to framework\n");

  FEM_Set_elem(0,nEle,0,3);
  FEM_Set_elem_conn(0,(int *)ele);
  delete[] ele;

/*Build the ghost layer for refinement border*/
  FEM_Add_ghost_layer(2,0); /*2 nodes/tuple, do not add ghost nodes*/
  const static int tri2edge[6]={0,1, 1,2, 2,0};
  FEM_Add_ghost_elem(0,3,tri2edge);

  CkPrintf("Finished with init\n");

}

struct myGlobals {
  int nnodes,maxnodes;
  int nelems,maxelems;
  connRec *conn; //Element connectivity table

  vector2d *coord; //Undeformed coordinates of each node
  vector2d *R_net, *d, *v, *a; //Physical fields of each node
  double *m_i; //Inverse of mass at each node
  int m_i_fid; //Field ID for m_i
  
  double *S11, *S22, *S12; //Stresses for each element
};

void pup_myGlobals(pup_er p,myGlobals *g) 
{
  FEM_Print("-------- called pup routine -------");
  pup_int(p,&g->nnodes);
  pup_int(p,&g->nelems);
  pup_int(p,&g->maxelems);
  pup_int(p,&g->maxnodes);
  int nnodes=g->nnodes, nelems=g->nelems;
  if (pup_isUnpacking(p)) {
    g->coord=new vector2d[g->maxnodes];
    g->conn=new connRec[g->maxelems];
    g->R_net=new vector2d[g->maxnodes]; //Net force
    g->d=new vector2d[g->maxnodes];//Node displacement
    g->v=new vector2d[g->maxnodes];//Node velocity
    g->a=new vector2d[g->maxnodes];
    g->m_i=new double[g->maxnodes];
    g->S11=new double[g->maxelems];
    g->S22=new double[g->maxelems];
    g->S12=new double[g->maxelems];
  }
  pup_doubles(p,(double *)g->coord,2*nnodes);
  pup_ints(p,(int *)g->conn,3*nelems);
  pup_doubles(p,(double *)g->R_net,2*nnodes);
  pup_doubles(p,(double *)g->d,2*nnodes);
  pup_doubles(p,(double *)g->v,2*nnodes);
  pup_doubles(p,(double *)g->a,2*nnodes);
  pup_doubles(p,(double *)g->m_i,nnodes);
  pup_doubles(p,(double *)g->S11,nelems);
  pup_doubles(p,(double *)g->S22,nelems);
  pup_doubles(p,(double *)g->S12,nelems);
  if (pup_isDeleting(p)) {
    delete[] g->coord;
    delete[] g->conn;
    delete[] g->R_net;
    delete[] g->d;
    delete[] g->v;
    delete[] g->a;
    delete[] g->m_i;
	delete[] g->S11;
	delete[] g->S22;
	delete[] g->S12;
  }
}


//Return the signed area of triangle i
double calcArea(myGlobals &g, int i)
{
	int n1=g.conn[i][0];
	int n2=g.conn[i][1];
	int n3=g.conn[i][2];
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

class myRefineClient {
  myGlobals &g;
  int lastA,lastB,lastD;
  int lastSplit(int A,int B) {
    if (A==lastA && B==lastB) return lastD;
    if (A==lastB && B==lastA) return lastD;
    return -1;
  }

public:
  myRefineClient(myGlobals &g_) :g(g_) {
    lastA=lastB=lastD=-1;
  }
  void split(int triNo,int A,int B,int C, double frac) {
    CkPrintf("---- Splitting edge %d-%d (%d), of triangle %d at %.2f\n",
    	A,B,C, triNo, frac);
    checkTriangle(g,triNo);
    
    //Figure out what we're adding:
    connRec &oldConn=g.conn[triNo];
    int D; //New node
    if (-1==(D=lastSplit(A,B))) 
    { //This edge wasn't just split-- create a new node
      D=g.nnodes++;
      CkPrintf("---- Adding node %d\n",D);
      if (g.nnodes>g.maxnodes) CkAbort("Added too many nodes to mesh!\n");
      lastA=A; lastB=B; lastD=D;
      if (A>=g.nnodes) CkAbort("Calculated A is invalid!");
      if (B>=g.nnodes) CkAbort("Calculated B is invalid!");
      
      //Interpolate node's physical quantities
      g.coord[D]=g.coord[A]*(1-frac)+g.coord[B]*frac;
      vector2d z(0,0);
      g.d[D]=g.d[A]*(1-frac)+g.d[B]*frac;
      g.v[D]=g.v[A]*(1-frac)+g.v[B]*frac;
      g.a[D]=g.a[A]*(1-frac)+g.a[B]*frac;
      g.R_net[D]=z;
      //m_i will be reconstructed after all insertions

      //Create new node's communication list:
      int AandB[2];
      AandB[0]=A;
      AandB[1]=B;
      /* Add a new node D between A and B */
      IDXL_Add_entity(
      	FEM_Comm_shared(FEM_Mesh_default_read(),FEM_NODE),
	D,2,AandB);
    }

  //Add the new triangle
    int newTri=g.nelems++;
    CkPrintf("---- Adding triangle %d\n",newTri);
    if (g.nelems>g.maxelems) CkAbort("Added too many elements to mesh!\n");
    connRec &newConn=g.conn[newTri];
    
  //Update the element connectivity:
    int i;
    //Replace A by D in the old triangle
    for (i=0;i<3;i++)
      if (oldConn[i]==A) oldConn[i]=D;
    //Insert new triangle CAD
    // OLD WAY: 
    //newConn[0]=C; newConn[1]=A; newConn[2]=D;
    // NEW WAY: preserves orientation and makes connectivity consistent
    //          with what TMR framework has
    for (i=0; i<3; i++) {
      if (oldConn[i] == B)
	newConn[i] = D;
      else if (oldConn[i] == C)
	newConn[i] = C;
      else if (oldConn[i] == D)
	newConn[i] = A;
    }
      
    
    checkTriangle(g,triNo);
    checkTriangle(g,newTri);
  }
};

//Compute forces on constant-strain triangles:
void CST_NL(const vector2d *coor,const connRec *lm,vector2d *R_net,
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
	    someNaNs=true;
    }
    if (fabs(d[i].x)>1.0) {
	    CkPrintf("d[%d] large at (%.4f,%.4f)   ",i,coord[i].x,coord[i].y);
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
		int n1=g.conn[i][0];
		int n2=g.conn[i][1];
		int n3=g.conn[i][2];
		double area=calcArea(g,i);
		if (1 || i%100==0) CkPrintf("Triangle %d has area %.3g\n",i,area);
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

extern "C" void
driver(void)
{
  int ignored;
  int i;  
  int myChunk=FEM_My_partition();

/*Add a refinement object to FEM array*/
CkPrintf("[%d] begin init\n",myChunk);
  REFINE2D_Init();
CkPrintf("[%d] end init\n",myChunk);

  myGlobals g;
  FEM_Register(&g,(FEM_PupFn)pup_myGlobals);
  
  FEM_Get_node(&g.nnodes,&ignored);
  g.maxnodes=10000+3*g.nnodes; //Silly: large maximum instead of slow additions
  g.coord=new vector2d[g.maxnodes];
  FEM_Get_node_data((double *)g.coord);  

  int nghost=0;    
  FEM_Get_elem(0,&nghost,&ignored,&ignored);
  g.nelems=FEM_Get_elem_ghost(0);
  g.maxelems=20000+6*g.nelems;
  g.conn=new connRec[(g.maxelems>nghost)?g.maxelems:nghost];
  FEM_Get_elem_conn(0,(int *)g.conn);
  
  /*Set up the global ID's, for refinement*/
  int *gid=new int[2*nghost];
  for (i=0;i<g.nelems;i++) {
    gid[2*i+0]=myChunk; //Local element-- my chunk
    gid[2*i+1]=i; //Local number
  }
  int gid_fid=FEM_Create_field(FEM_INT,2,0,2*sizeof(int));
  FEM_Update_ghost_field(gid_fid,0,gid);

  /*Set up refinement framework*/
  REFINE2D_NewMesh(g.nelems,nghost,(int *)g.conn,gid);
  delete[] gid;
  
  g.S11=new double[g.maxelems];
  g.S22=new double[g.maxelems];
  g.S12=new double[g.maxelems];
  
  //Initialize associated data
  g.R_net=new vector2d[g.maxnodes]; //Net force
  g.d=new vector2d[g.maxnodes];//Node displacement
  g.v=new vector2d[g.maxnodes];//Node velocity
  g.a=new vector2d[g.maxnodes];//Node accelleration
  g.m_i=new double[g.maxnodes];//Node mass
  g.m_i_fid=FEM_Create_field(FEM_DOUBLE,1,0,sizeof(double));
  for (i=0;i<g.maxnodes;i++)
    g.R_net[i]=g.d[i]=g.v[i]=g.a[i]=vector2d(0.0);

//Apply a small initial perturbation to positions
  for (i=0;i<g.nnodes;i++) {
	  const double max=1.0e-15/15.0; //Tiny perturbation
	  g.d[i].x+=max*(i&15);
	  g.d[i].y+=max*((i+5)&15);
  }

  int fid=FEM_Create_field(FEM_DOUBLE,2,0,sizeof(vector2d));
  
  for (i=0;i<g.nelems;i++)
    checkTriangle(g,i);

  //Timeloop
  if (CkMyPe()==0)
    CkPrintf("Entering timeloop\n");
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
    if (0 && t%16==0) {
	    CkPrintf("    Triangle 0:\n");
	    for (int j=0;j<3;j++) {
		    int n=g.conn[0][j];
		    CkPrintf("    Node %d: coord=(%.4f,%.4f)  d=(%.4g,%.4g)\n",
			     n,g.coord[n].x,g.coord[n].y,g.d[n].x,g.d[n].y);
	    }
    }
//    if (t%512==0)
//      FEM_Migrate();

    if (t%128==0) { //Refinement:
      vector2d *loc=new vector2d[2*g.nnodes];
      for (i=0;i<g.nnodes;i++) {
	loc[i]=g.coord[i];//+g.d[i];
      }
      double *areas=new double[g.nelems];
      curArea=curArea*0.99;
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
      REFINE2D_Split(g.nnodes,(double *)loc,g.nelems,areas);
      delete[] areas;
      delete[] loc;
      myRefineClient c(g);
      int nSplits=REFINE2D_Get_Split_Length();
      for (int splitNo=0;splitNo<nSplits;splitNo++) {
        int tri,A,B,C;
        double frac;
        REFINE2D_Get_Split(splitNo,(int *)(g.conn),&tri,&A,&B,&C,&frac);      
        c.split(tri,A,B,C,frac);
      
        //Since the connectivity changed, update the masses
        calcMasses(g);
      }
      
      REFINE2D_Check(g.nelems,(int *)g.conn,g.nnodes);
             
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


