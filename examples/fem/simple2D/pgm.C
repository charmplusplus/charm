/*
 Charm++ Finite-Element Framework Program:
   Performs simple 2D structural simulation on Triangle-style inputs.
   
 init: 
    read node input file
    pass nodes into FEM framework
    read element input file
    pass elements into FEM framework
 
 driver:
    extract mesh chunk from framework
    calculate masses of nodes
    timeloop
      compute forces within my chunk
      communicate
      apply boundary conditions and integrate
      pass data to NetFEM
 
 Among the hideous things about this program are:
   -Hardcoded material properties, timestep, and boundary conditions
   -Limited to 2D
 
 Converted from f90 Structural Materials program by 
 	Orion Sky Lawlor, 2001, olawlor@acm.org

 Updated to new FEM interface by
    Isaac Dooley, 2005
 */

#include "pgm.h"


extern "C" void
init(void)
{
  CkPrintf("init started\n");
  double startTime=CmiWallTimer();
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
 


  int fem_mesh=FEM_Mesh_default_write(); // Tell framework we are writing to the mesh

  CkPrintf("Passing node coords to framework\n");

  /*   Old versions used FEM_Set_node() and FEM_Set_node_data()
   *   New versions use the more flexible FEM_Set_Data()
   */

  FEM_Mesh_data(fem_mesh,        // Add nodes to the current mesh
                FEM_NODE,        // We are registering nodes
                FEM_DATA+0,      // Register the point locations which are normally 
                                 // the first data elements for an FEM_NODE
                (double *)pts,   // The array of point locations
                0,               // 0 based indexing
                nPts,            // The number of points
                FEM_DOUBLE,      // Coordinates are doubles
                2);              // Points have dimension 2 (x,y)
 

  CkPrintf("Passing elements to framework\n");

  /*   Old versions used FEM_Set_elem() and FEM_Set_elem_conn() 
   *   New versions use the more flexible FEM_Set_Data()
   */

  FEM_Mesh_data(fem_mesh,      // Add nodes to the current mesh
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

  CkPrintf("Finished with init (Reading took %.f s)\n",CmiWallTimer()-startTime);

}


// A driver() function 
// driver() is required in all FEM programs
extern "C" void
driver(void)
{
  int nnodes,nelems,ignored;
  int i, myId=FEM_My_partition();
  myGlobals g;
  FEM_Register(&g,(FEM_PupFn)pup_myGlobals);
  
  
  int mesh=FEM_Mesh_default_read(); // Tell framework we are reading data from the mesh
  
  // Get node data
  nnodes=FEM_Mesh_get_length(mesh,FEM_NODE); // Get number of nodes
  g.coord=new vector2d[nnodes];
  // Get node positions
  FEM_Mesh_data(mesh, FEM_NODE, FEM_DATA+0, (double*)g.coord, 0, nnodes, FEM_DOUBLE, 2);  


  // Get element data
  nelems=FEM_Mesh_get_length(mesh,FEM_ELEM+0); // Get number of elements
  g.nnodes=nnodes; g.nelems=nelems;
  g.conn=new connRec[nelems];
  g.S11=new double[nelems];
  g.S22=new double[nelems];
  g.S12=new double[nelems];
  // Get connectivity for elements
  FEM_Mesh_data(mesh, FEM_ELEM+0, FEM_CONN, (int *)g.conn, 0, nelems, FEM_INDEX_0, 3);  


  //Initialize associated data
  g.R_net=new vector2d[nnodes]; //Net force
  g.d=new vector2d[nnodes];//Node displacement
  g.v=new vector2d[nnodes];//Node velocity
  g.a=new vector2d[nnodes];//Node acceleration
  for (i=0;i<nnodes;i++)
    g.R_net[i]=g.d[i]=g.v[i]=g.a[i]=vector2d(0.0);

//Apply a small initial perturbation to positions
  for (i=0;i<nnodes;i++) {
	  const double max=1.0e-10/15.0; //Tiny perturbation
	  g.d[i].x+=max*(i&15);
	  g.d[i].y+=max*((i+5)&15);
  }

  int fid=FEM_Create_simple_field(FEM_DOUBLE,2);

  //Timeloop
  if (myId==0)
    CkPrintf("Entering timeloop\n");
  int tSteps=5000;
  double startTime, totalStart;
  startTime=totalStart=CkWallTimer();
  for (int t=0;t<tSteps;t++) {
    if (1) { //Structural mechanics

    //Compute forces on nodes exerted by elements
	CST_NL(g.coord,g.conn,g.R_net,g.d,matConst,nnodes,nelems,g.S11,g.S22,g.S12);

    //Communicate net force on shared nodes
	FEM_Update_field(fid,g.R_net);

    //Advance node positions
	advanceNodes(dt,nnodes,g.coord,g.R_net,g.a,g.v,g.d,0);
    }

    //Debugging/perf. output
    double curTime=CkWallTimer();
    double total=curTime-startTime;
    startTime=curTime;
    if (myId==0 && (t%64==0)) {
	    CkPrintf("%d %.6f sec for loop %d \n",CkNumPes(),total,t);
    	    if (0) {
	      CkPrintf("    Triangle 0:\n");
	      for (int j=0;j<3;j++) {
		    int n=g.conn[0][j];
		    CkPrintf("    Node %d: coord=(%.4f,%.4f)  d=(%.4g,%.4g)\n",
			     n,g.coord[n].x,g.coord[n].y,g.d[n].x,g.d[n].y);
	      }
    	    }
    }
    /* perform migration-based load balancing */
    if (t%1024==0)
      FEM_Migrate();
    
    if (t%1024==0) { //Publish data to the net
#if ! CMK_MULTICORE
	    NetFEM n=NetFEM_Begin(FEM_My_partition(),t,2,NetFEM_POINTAT);
	    
	    NetFEM_Nodes(n,nnodes,(double *)g.coord,"Position (m)");
	    NetFEM_Vector(n,(double *)g.d,"Displacement (m)");
	    NetFEM_Vector(n,(double *)g.v,"Velocity (m/s)");
	    
	    NetFEM_Elements(n,nelems,3,(int *)g.conn,"Triangles");
		NetFEM_Scalar(n,g.S11,1,"X Stress (pure)");
		NetFEM_Scalar(n,g.S22,1,"Y Stress (pure)");
		NetFEM_Scalar(n,g.S12,1,"Shear Stress (pure)");
	    
	    NetFEM_End(n);
#endif
    }
  }

  if (myId==0) {
    double elapsed=CkWallTimer()-totalStart;
    CkPrintf("Driver finished: average %.6f s/step\n",elapsed/tSteps);
  }
}


// A PUP function to allow for migration and load balancing of mesh partitions.
// The PUP function is not needed if no migration or load balancing is desired.
void pup_myGlobals(pup_er p,myGlobals *g) 
{
  FEM_Print("-------- called pup routine -------");
  pup_int(p,&g->nnodes);
  pup_int(p,&g->nelems);
  int nnodes=g->nnodes, nelems=g->nelems;
  if (pup_isUnpacking(p)) {
    g->coord=new vector2d[nnodes];
    g->conn=new connRec[nelems];
    g->R_net=new vector2d[nnodes]; //Net force
    g->d=new vector2d[nnodes];//Node displacement
    g->v=new vector2d[nnodes];//Node velocity
    g->a=new vector2d[nnodes];
	g->S11=new double[nelems];
	g->S22=new double[nelems];
	g->S12=new double[nelems];
  }
  pup_doubles(p,(double *)g->coord,2*nnodes);
  pup_ints(p,(int *)g->conn,3*nelems);
  pup_doubles(p,(double *)g->R_net,2*nnodes);
  pup_doubles(p,(double *)g->d,2*nnodes);
  pup_doubles(p,(double *)g->v,2*nnodes);
  pup_doubles(p,(double *)g->a,2*nnodes);
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
	delete[] g->S11;
	delete[] g->S22;
	delete[] g->S12;
  }
}



//Update node position, velocity, acceleration based on net force.
void advanceNodes(const double dt,int nnodes,const vector2d *coord,
		  vector2d *R_net,vector2d *a,vector2d *v,vector2d *d,bool dampen)
{
  const double nodeMass=1.0e-6; //Node mass, kilograms (HACK: hardcoded)
  const double xm=1.0/nodeMass; //Inverse of node mass
  const vector2d z(0,0);

  const double shearForce=1.0e-6/(dt*dt*xm);

  bool someNaNs=false;
  int i;
  for (i=0;i<nnodes;i++) {
    vector2d R_n=R_net[i];
#if NANCHECK
    if (((R_n.x-R_n.x)!=0)) {
	    CkPrintf("%d (%.4f,%.4f)   ",i,coord[i].x,coord[i].y);
	    someNaNs=true;
    }
#endif
    R_net[i]=z;
//Apply boundary conditions (HACK: hardcoded!)
    if (1) {
       if (coord[i].x<0.00001)
	       continue; //Left edge will NOT move
       if (coord[i].y>0.02-0.00001)
	       R_n.x+=shearForce; //Top edge pushed hard to right
    }
//Update displacement and velocity
    vector2d aNew=R_n*xm;
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
