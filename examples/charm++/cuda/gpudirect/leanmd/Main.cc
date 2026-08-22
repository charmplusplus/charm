#include <string>
#include <cstring>
#include <algorithm>
#include "time.h"
#include "Main.h"
#include "Cell.h"
#include "Compute.h"

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_Cell cellArray;
/* readonly */ CProxy_Compute computeArray;
/* readonly */ CProxy_StreamPool streamPool;

/* readonly */ int cellArrayDimX;
/* readonly */ int cellArrayDimY;
/* readonly */ int cellArrayDimZ;
/* readonly */ int finalStepCount; 
/* readonly */ int firstLdbStep; 
/* readonly */ int ldbPeriod;
/* readonly */ int checkptFreq; 
/* readonly */ int checkptStrategy;
/* readonly */ std::string logs;
/* readonly */ int densityMode;
/* readonly */ int computeMapMode;
/* readonly */ int maxCellParts;
/* readonly */ int densityReportFreq;

// Entry point of Charm++ application
Main::Main(CkArgMsg* m) {
  CkPrintf("\nLENNARD JONES MOLECULAR DYNAMICS START UP ...\n");

  //set variable values to a default set
  cellArrayDimX = CELLARRAY_DIM_X;
  cellArrayDimY = CELLARRAY_DIM_Y;
  cellArrayDimZ = CELLARRAY_DIM_Z;
  finalStepCount = DEFAULT_FINALSTEPCOUNT;
  firstLdbStep = DEFAULT_FIRST_LDB;
  ldbPeriod = DEFAULT_LDB_PERIOD;
  checkptFreq = DEFAULT_FT_PERIOD;

  mainProxy = thisProxy;

  // Created before any Cell or Compute, so a branch exists wherever they land.
  streamPool = CProxy_StreamPool::ckNew();

  int numPes = CkNumPes();
  int currPe = -1, pe;

  // Named options first: CmiGetArg* strips what it matches out of argv, so the
  // positional parsing below still sees a clean argument list.
  densityMode = DENSITY_UNIFORM;
  computeMapMode = COMPUTEMAP_RR;
  {
    char* opt = NULL;
    if (CmiGetArgStringDesc(m->argv, "-density", &opt,
                            "atom density profile: uniform|gradient|clump")) {
      if (!strcmp(opt, "gradient")) densityMode = DENSITY_GRADIENT;
      else if (!strcmp(opt, "clump")) densityMode = DENSITY_CLUMP;
      else if (strcmp(opt, "uniform"))
        CkAbort("unknown -density '%s' (expected uniform, gradient or clump)", opt);
    }
    if (CmiGetArgStringDesc(m->argv, "-computemap", &opt,
                            "Compute placement: rr|local")) {
      if (!strcmp(opt, "local")) computeMapMode = COMPUTEMAP_LOCAL;
      else if (strcmp(opt, "rr"))
        CkAbort("unknown -computemap '%s' (expected rr or local)", opt);
    }
    densityReportFreq = 0;
    CmiGetArgIntDesc(m->argv, "-densityreport", &densityReportFreq,
                     "report atoms per x-slab every N steps (0 = off)");
    m->argc = CmiGetArgc(m->argv);
  }

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

  //step after which load balancing starts
  if (m->argc > cur_arg) {
    firstLdbStep=atoi(m->argv[cur_arg++]);
    CkPrintf("First LB Step:%d\n",firstLdbStep);
  }

  //periodicity of load balancing
  if (m->argc > cur_arg) {
    ldbPeriod=atoi(m->argv[cur_arg++]);
    CkPrintf("LB Period:%d\n",ldbPeriod);
  }

  //periodicity of checkpointing
  if (m->argc > cur_arg) {
    checkptFreq=atoi(m->argv[cur_arg++]);
    CkPrintf("FT Period:%d\n",checkptFreq);
  }

  checkptStrategy = 1;
  //choose the checkpointing strategy use in disk checkpointing
  if (m->argc > cur_arg) {
  	checkptStrategy = 0;
    logs = m->argv[cur_arg];
  }

  // Both knobs have to be reported: a run's numbers mean nothing without them,
  // and the defaults reproduce the stock benchmark exactly.
  static const char* const kDensityName[] = {"uniform", "gradient", "clump"};
  static const char* const kMapName[] = {"round-robin", "local"};
  CkPrintf("Atom density profile:%s\n", kDensityName[densityMode]);
  CkPrintf("Compute placement:%s\n", kMapName[computeMapMode]);

  // Device buffers are sized off the fullest cell rather than each cell's own
  // count: migration moves atoms between cells, and under an imbalanced profile a
  // sparse cell sitting next to dense ones takes in far more than it started with.
  maxCellParts = 0;
  for (int x = 0; x < cellArrayDimX; x++)
    for (int y = 0; y < cellArrayDimY; y++)
      for (int z = 0; z < cellArrayDimZ; z++)
        maxCellParts = std::max(maxCellParts, cellParticleCount(x, y, z));
  CkPrintf("Atoms in the fullest cell:%d\n", maxCellParts);

  CProxy_CellMap cellMap = CProxy_CellMap::ckNew(cellArrayDimX, 
    cellArrayDimY, cellArrayDimZ);
  CkArrayOptions opts(cellArrayDimX, cellArrayDimY, cellArrayDimZ);
  opts.setMap(cellMap);
  //create a 3D Patch array (with a uniform distribution)
  cellArray = CProxy_Cell::ckNew(opts);

  //create an empty 6D computer array to be filled in by Cells
  computeArray = CProxy_Compute::ckNew();

  cellArray.createComputes();
  CkPrintf("\nCells: %d X %d X %d .... created\n", cellArrayDimX, 
    cellArrayDimY, cellArrayDimZ);

  delete m;
}

// Atoms per x-slab. The density profiles vary along x, so a decaying spread here
// is the physical statement that the imbalance they create is transient: atoms
// diffuse from the dense slabs into the sparse ones until the box is uniform
// again, and from then on there is nothing for a balancer to do.
void Main::densityReport(int n, int* counts) {
  int lo = counts[0], hi = counts[0];
  long total = 0;
  std::string line;
  for (int x = 0; x < n; x++) {
    lo = std::min(lo, counts[x]);
    hi = std::max(hi, counts[x]);
    total += counts[x];
    line += " " + std::to_string(counts[x]);
  }
  const double avg = (double)total / n;
  CkPrintf("[density] slabs:%s  min=%d max=%d max/min=%.2f max/avg=%.3f\n",
           line.c_str(), lo, hi, lo ? (double)hi / lo : 0.0, avg ? hi / avg : 0.0);
}

//constructor for chare object migration
Main::Main(CkMigrateMessage* msg): CBase_Main(msg) { 
}

//pup routine incase the main chare moves, pack important information
void Main::pup(PUP::er &p) {
  CBase_Main::pup(p);
  __sdag_pup(p);
}

#include "leanmd.def.h"
