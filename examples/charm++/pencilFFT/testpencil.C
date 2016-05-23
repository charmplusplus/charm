
#include "charm++.h"
#include "pencilfft/pencil_api.h"
#include "testpencil.decl.h"

#define NUM_FFT_ITER   200
#define START_TIMING   1
#define MAX_ITERATIONS 5

LineFFTInfo   info;
int           iteration;
double        startTime;

void red_handler (void *param, int size, void *data) {

  iteration ++;
  //printf ("Iteration Complete\n", iteration);
  
  if (iteration == START_TIMING)
    startTime = CkWallTimer ();
  
  if (iteration == MAX_ITERATIONS) {

    double endTime = CkWallTimer();

    CkAssert (MAX_ITERATIONS > START_TIMING);
    CkPrintf ("Time to perform a pair of (%d, %d, %d) 3D FFT operations %g ms\n", 
	      info.sizeX, info.sizeY, info.sizeZ,
	      (endTime - startTime) * 1000.0/ 
	      (NUM_FFT_ITER * (MAX_ITERATIONS - START_TIMING)));
    CkExit ();
  }
  
  startLineFFTArray (&info);
}

class main : public CBase_main {
public:
  main (CkArgMsg *m);
  main (CkMigrateMessage *m) {}
};

main::main (CkArgMsg *m) {
  int sizeX=0, sizeY=0, sizeZ=0;
  int grainX=0, grainY=0, grainZ=0;

  if (m->argc <= 1) {
    sizeX = sizeY = sizeZ = 16;
    grainX = grainY = grainZ = 4;
  }
  else if (m->argc == 7) {
    sizeX = atoi (m->argv[1]);
    sizeY = atoi (m->argv[2]);
    sizeZ = atoi (m->argv[3]);

    grainX = atoi (m->argv[4]);
    grainY = atoi (m->argv[5]);
    grainZ = atoi (m->argv[6]);

    CkAssert ((sizeX % grainX) == 0);
    CkAssert ((sizeY % grainY) == 0);
    CkAssert ((sizeZ % grainZ) == 0);
  }
  else {
    sizeX = sizeY = sizeZ = atoi (m->argv[1]);
    grainX = grainY = grainZ = atoi (m->argv[2]);    
    CkAssert ((sizeX % grainX) == 0);
  }

  CkPrintf ("Calling Configure\n");
  configureLineFFTInfo (&info, sizeX, sizeY, sizeZ, 
			grainX, grainY, grainZ,
			NULL, 
			ARRAY_REDUCTION, 
			true,
			NUM_FFT_ITER);

  CkPrintf ("Calling Create\n");
  createLineFFTArray (&info);

  info.xProxy.setReductionClient (red_handler, NULL);

  CkPrintf ("Calling Start\n");
  startLineFFTArray (&info);

  delete m;
};


#include "testpencil.def.h"
