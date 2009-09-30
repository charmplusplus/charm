/**
 * Main implementation file for FEM version of crack propagation code.
 */
#include <stddef.h>
#include "charm++.h" // for CkWallTimer, CkPrintf, etc.
#include "crack.h"
#include "netfem.h"

void crack_abort(const char *why)
{
  CkAbort(why);
}

extern "C" void
init(void)
{
  //Read and send off mesh data:
  CkPrintf("Reading mesh...\n");
  MeshData mesh;
  readMesh(&mesh,"crck_bar.inp");
  CkPrintf("Uploading mesh to FEM...\n");
  sendMesh(&mesh,FEM_Mesh_default_write());
  deleteMesh(&mesh);
  CkPrintf("Partitioning mesh...\n");
}

/// This structure contains all the data held by one chunk of the mesh.
struct GlobalData {
  int myid;      // my chunk of the mesh
  
  MeshData mesh;
  
  double *itimes; // Length of each timestep, for performance tuning
};

static void mypup(pup_er p, GlobalData *gd)
{
  pup_int(p,&gd->myid);
  pupMesh(p,&gd->mesh);
  if(pup_isUnpacking(p))
  {
    gd->itimes = new double[config.nTime];
  }
  pup_doubles(p, gd->itimes, config.nTime);
  if(pup_isDeleting(p))
  {
    delete[] gd->itimes;
  }
}

extern "C" double CmiCpuTimer(void);
static void
_DELAY_(int microsecs)
{
  double upto = CmiCpuTimer() + 1.e-6 * microsecs;
  while(upto > CmiCpuTimer());
}


void uploadNetFEM(MeshData *m,int timeStep) {
   NetFEM n=NetFEM_Begin(FEM_My_partition(),timeStep,2,NetFEM_POINTAT);
   NetFEM_Nodes_field(n,m->nn,NetFEM_Field(Node,pos),m->nodes,"Position (m)");
    NetFEM_Vector_field(n,m->nodes,NetFEM_Field(Node,disp),"Displacement (m)");
    NetFEM_Vector_field(n,m->nodes,NetFEM_Field(Node,vel),"Velocity (m/s)");
    NetFEM_Scalar_field(n,m->nodes,1,NetFEM_Field(Node,xM),"Mass (Kg)");
   
   NetFEM_Elements_field(n,m->ne,3,NetFEM_Field(Vol,conn),0, m->vols,"Triangles");
    NetFEM_Scalar_field(n,m->vols,3, NetFEM_Field(Vol,s11l), "S11");
    NetFEM_Scalar_field(n,m->vols,3, NetFEM_Field(Vol,s12l), "S12");
    NetFEM_Scalar_field(n,m->vols,3, NetFEM_Field(Vol,s22l), "S22");
   
   NetFEM_Elements_field(n,m->nc,6,NetFEM_Field(Coh,conn),0, m->cohs,"Cohesive");
    NetFEM_Scalar_field(n,m->vols,3, NetFEM_Field(Coh,Sthresh), "Sthresh");
   
   NetFEM_End(n);
}

extern "C" void
driver(void)
{
  //Read our configuration data.  We should 
  //  really use TCHARM_Readonly_globals here, rather than
  //  re-reading the input file for each chunk.
  readConfig("cohesive.inp","crck_bar.inp");
  
// Set up my chunk's global data:
  GlobalData gd_storage;
  GlobalData *gd=&gd_storage;
  int myid = FEM_My_partition();
  int numparts = FEM_Num_partitions();
  FEM_Register((void*)gd, (FEM_PupFn)mypup); //allows migration
  
  gd->myid = myid;
  gd->myid = myid;
  gd->itimes = new double[config.nTime];
  for(int i=0;i<config.nTime;i++) gd->itimes[i] = 0.0;
  
// Get the FEM mesh:
  if (myid==0)
    CkPrintf("Extracting partitioned mesh...\n");
  int fem_mesh=FEM_Mesh_default_read();
  recvMesh(&gd->mesh,fem_mesh);
  
  // Prepare for communication:
  int rfield = FEM_Create_field(FEM_DOUBLE, 4, offsetof(Node, Rin), 
                                sizeof(Node));

// Prepare for the timeloop:
  if (myid==0)
    CkPrintf("Running timeloop...\n");
  NodeSlope sl;
  nodeSetup(&sl);
  int t;
  for(t=0;t<config.nTime;t++) //Timeloop:
  {
    double startTime = CkWallTimer(); //Start timing:
    
    nodeBeginStep(&gd->mesh);
    lst_NL(&gd->mesh);
    lst_coh2(&gd->mesh);
    
    //We have the node forces from local elements, but not remote:
    //  ask FEM to add in the remote node forces.
    FEM_Update_field(rfield, gd->mesh.nodes);
    
    nodeFinishStep(&gd->mesh, &sl, t);
    
    if (1) { //Output data to NetFEM:
      uploadNetFEM(&gd->mesh,t);
    }
    
    if(0 && myid==79 && t>35) // Add fake load imbalance
    {
      int biter = (t < 40 ) ? t : 40;
      _DELAY_((biter-35)*19000);
    }
    
    if(t%1000==999) // Migrate, for load balance
    {
       FEM_Migrate();
    }
    
    // Keep track of how long that timestep took, across the whole machine:
    gd->itimes[t] = CkWallTimer()-startTime;
  }
  
  if (0) 
  {
    // Do a debugging printout of how long each step took:
    int tfield = FEM_Create_field(FEM_DOUBLE, 1, 0, 0);
    for(t=0;t<config.nTime;t++)
    {
        double thisStep;
        // Sum across the whole machine:
        FEM_Reduce(tfield, &gd->itimes[t], &thisStep, FEM_SUM);
        if(gd->myid==0)
          CkPrintf("Iteration\t%d\t%.9f\n",t,thisStep/numparts);
    }
  }
}

