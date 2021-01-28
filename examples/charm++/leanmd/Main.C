#include "time.h"

#include "defs.h"
#include "leanmd.decl.h"
#include "Main.h"
#include "Cell.h"
#include "Compute.h"

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_Cell cellArray;
/* readonly */ CProxy_Compute computeArray;

/* readonly */ int cellArrayDimX;
/* readonly */ int cellArrayDimY;
/* readonly */ int cellArrayDimZ;
/* readonly */ int finalStepCount; 

// Entry point of Charm++ application
Main::Main(CkArgMsg* m) {
  CkPrintf("\nLENNARD JONES MOLECULAR DYNAMICS START UP ...\n");

  //set variable values to a default set
  cellArrayDimX = CELLARRAY_DIM_X;
  cellArrayDimY = CELLARRAY_DIM_Y;
  cellArrayDimZ = CELLARRAY_DIM_Z;

  mainProxy = thisProxy;

  int numPes = CkNumPes();
  int currPe = -1, pe;
  int cur_arg = 1;

  CkPrintf("\nInput Parameters...\n");

  //read user parameters
  //number of cells in each dimension
  if (m->argc > cur_arg) {
    cellArrayDimX=atoi(m->argv[cur_arg++]);
    cellArrayDimY=atoi(m->argv[cur_arg++]);
    cellArrayDimZ=atoi(m->argv[cur_arg++]);
    CkPrintf("Cell Array Dimension X:%d Y:%d Z:%d of size %d %d %d\n",cellArrayDimX,cellArrayDimY,cellArrayDimZ,CELL_SIZE_X,CELL_SIZE_Y,CELL_SIZE_Z);
  }

  //number of steps in simulation
  if (m->argc > cur_arg) {
    finalStepCount=atoi(m->argv[cur_arg++]);
    CkPrintf("Final Step Count:%d\n",finalStepCount);
  }

  //initializing the 3D cell array
  cellArray = CProxy_Cell::ckNew(cellArrayDimX,cellArrayDimY,cellArrayDimZ);
  CkPrintf("\nCells: %d X %d X %d .... created\n", cellArrayDimX, cellArrayDimY, cellArrayDimZ);

  //initializing the 6D compute array
  computeArray = CProxy_Compute::ckNew();
  for (int x=0; x<cellArrayDimX; x++)
    for (int y=0; y<cellArrayDimY; y++)
      for (int z=0; z<cellArrayDimZ; z++)
        cellArray(x, y, z).createComputes();

  thisProxy.run();
  delete m;
}

//constructor for chare object migration
Main::Main(CkMigrateMessage* msg): CBase_Main(msg) { 
}

//pup routine incase the main chare moves, pack important information
void Main::pup(PUP::er &p) {
}

#include "leanmd.def.h"
