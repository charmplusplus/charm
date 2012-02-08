
#include "charm++.h"
#include "pencilfft/pencil_api.h"
#include "patch.h"

#define MAX_ITERATIONS 100

LineFFTInfo   info;
PatchInfo     patchInfo;

int           iteration;
double        startTime;

CProxy_Patch  patch_proxy;

class main : public CBase_main {
public:
  main (CkArgMsg *m);
  main (CkMigrateMessage *m) {}
  
  void iterationsDone (CkReductionMsg *msg) {
    double endTime = CmiWallTimer();    
    CkPrintf ("PME Time to perform a pair of (%d, %d, %d) 3D FFT operations %g ms\n", 
	      info.sizeX, info.sizeY, info.sizeZ,
	      (endTime - startTime) * 1000.0/ (MAX_ITERATIONS));
    CkExit ();  
  }
  
  void finishedInitialization (CkReductionMsg *msg) {    
    int *gmsgcounts = (int *) msg->getData();
    int size  = msg->getSize () / sizeof(int);
    
    int ny = info.sizeY / info.grainY;
    int nz = info.sizeZ / info.grainZ;    
    
    CkAssert (size == ny * nz);    
   
    CkCallback cb(CkIndex_main::finishedSetNumGrid (NULL), 
		  CProxy_main(ckGetChareID()));
 
    for (int z = 0; z < nz; z ++)
      for (int y = 0; y < ny; y ++) {
#if __TEST_PME_VERBOSE__       
	CkPrintf ("Finished intialization nmsgs[%d,%d] = %d", z,y, 
		  gmsgcounts [z *ny + y]);
#endif
	info.xProxy (y,z).setNumGridMessages(gmsgcounts[z * ny + y], cb);
      }
  }
  
  void finishedSetNumGrid (CkReductionMsg *msg) {
#if __TEST_PME_VERBOSE__       
    CkPrintf ("Finished SetNumGrid Reduction\n");
#endif

    startTime = CmiWallTimer ();
    patch_proxy.startTimeStep ();
  }
};

main::main (CkArgMsg *m) {
  int sizeX=0, sizeY=0, sizeZ=0;
  int grainX=0, grainY=0, grainZ=0;
  int patch_nx=0, patch_ny=0, patch_nz=0;

  CkAssert (CkNumPes() > 1); //test needs a power of two processors > 1

  if (m->argc <= 1) {
    if (CkNumPes () <= 64) {
      sizeX = sizeY = sizeZ = 16;
      grainX = grainY = grainZ = 4;
      patch_nx = 2;
    }
    else {
      sizeX = sizeY = sizeZ = 128;
      grainX = grainY = grainZ = 8;
      patch_nx = 8;
    }
  }
  else {
    CkAssert (m->argc == 4);
    sizeX = sizeY = sizeZ = atoi (m->argv[1]);
    grainX = grainY = grainZ = atoi (m->argv[2]);    
    patch_nx = atoi (m->argv[3]);

    CkAssert ((sizeX % grainX) == 0);
  }

  CkAssert ((CkNumPes() & (CkNumPes() - 1)) == 0); //must run on power of 2

  if (CkNumPes () >= 8 * patch_nx * patch_nx) {
    patch_ny = patch_nx;
    patch_nz = (CkNumPes()) / (patch_nx * patch_nx);
  }
  else {
    if (patch_nx >= CkNumPes())
      patch_ny = patch_nx/2;
    else
      patch_ny = patch_nx;

    patch_nz = CkNumPes() / (patch_nx * patch_ny);
  }
    
  CkAssert (patch_nx * patch_ny * patch_nz == CkNumPes());

  CkPrintf ("PME TEST: Calling Configure\n");
  configureLineFFTInfo (&info, sizeX, sizeY, sizeZ, 
			grainX, grainY, grainZ,
			NULL, SEND_GRID, true);

  CkPrintf ("PME Test: Calling CreateLineFFTArray\n");
  createLineFFTArray (&info);
  
  patchInfo.nx = patch_nx;
  patchInfo.ny = patch_ny;
  patchInfo.nz = patch_nz;  
  patchInfo.fftinfo = info;
  patchInfo.niterations = MAX_ITERATIONS;

  CkCallback cb_start(CkIndex_main::finishedInitialization (NULL), 
		      CProxy_main (ckGetChareID()));
  CkCallback cb_done (CkIndex_main::iterationsDone (NULL), 
		      CProxy_main (ckGetChareID()));

  patchInfo.cb_start = cb_start;
  patchInfo.cb_done  = cb_done;

  patch_proxy = CProxy_Patch::ckNew();

  for (int z = 0; z < patch_nz; z ++) 
    for (int y = 0; y < patch_ny; y ++) 
      for (int x = 0; x < patch_nx; x ++) {
	patchInfo.my_x = x;
	patchInfo.my_y = y;
	patchInfo.my_z = z;
	
	int idx = z * patch_nx * patch_ny + y * patch_nx + x;

	patch_proxy[idx].initialize (patchInfo);
      }

  delete m;
};


#include "testpme.def.h"
