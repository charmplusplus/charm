#include <stdio.h>
#include "collidecharm.h"
#include "mpicollide.decl.h"
#include "mpiCollisionClient.h"
//header from Charm to enable Interoperation
#include "mpi-interoperate.h"

/* readonly */ CProxy_MpiCollisionClient myClient;
/* readonly */ CollideHandle collision_detector;
/* readonly */ CProxy_MainCollide mainProxy;

/*mainchare*/
class MainCollide : public CBase_MainCollide
{
public:
  MainCollide(CkArgMsg *m)
  {
	mainProxy = thisProxy;	 
	double grid_dim_size = 1.0;
	if(m->argc >1 ) grid_dim_size=atof(m->argv[1]);
	// create CProxy_collideMgr
	myClient = CProxy_MpiCollisionClient::ckNew();
	// origin of grid and grid cell size
	//vector3d origin(0,0,0), grid_size(grid_dim_size, grid_dim_size, grid_dim_size);
	vector3d origin(0,0,0), grid_size(2,100,2);
	CollideGrid3d grid(origin, grid_size);
	collision_detector = CollideCreate(grid, myClient);
        //Start the computation
        CkPrintf("Running collide on %d processors\n", CkNumPes());

  };

  MainCollide(CkMigrateMessage *m) {}

  void done(int numColls)
  {
	  CkPrintf("number of collisions:%d \n", numColls);
    CkExit();
  };
};

//C++ function invoked from MPI, marks the begining of Charm++
void detectCollision(CollisionList *&colls,int nBoxes, bbox3d *boxes, int *prio)
{
	CollideRegister(collision_detector, CkMyPe());
	myClient.ckLocalBranch()->setResultPointer(colls);
	CollideBoxesPrio(collision_detector, CkMyPe(), nBoxes, boxes, prio);
  if(CkMyPe() == 0) {
    CkPrintf("calling collision detection\n");
  }
  CsdScheduler(-1);
}

#include "mpicollide.def.h"
