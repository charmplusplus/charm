#include "pose.h"
#include "pgm.h"
#include "Pgm.def.h"
#include "Worker_sim.h"

main::main(CkArgMsg *m) { 
  CkGetChareID(&mainhandle);

  int numWorkers, patternNum;

  if ((m->argc < 3) || (strcmp(m->argv[1], "-help") == 0)) {
    CkPrintf("\nUsage: pgm <#workerObjs> <pattern#> [<pattern parameters>]\n");
    CkPrintf("\n");
    CkPrintf("Pattern 0: Simple ring test\n");
    CkPrintf("   Pattern parameters: none\n");
    CkPrintf("Pattern 1: Short communcation bursts (~100 GVT ticks apart), long message send (~1M ticks), repeat\n");
    CkPrintf("   Pattern parameters: none\n");
    CkPrintf("Pattern 2: Simultaneous ring with elapse statements\n");
    CkPrintf("   Pattern parameters: none\n");
    CkPrintf("Pattern 3: Simultaneous ring without elapse statements\n");
    CkPrintf("   Pattern parameters: none\n");
    CkPrintf("\n");
    CkPrintf("For more details on the patterns, see the README file.\n\n");
    CkExit();
  }

  numWorkers = atoi(m->argv[1]);
  if (numWorkers < 1) {
    CkPrintf("Number of Worker objects (%d) must be positive\n", numWorkers);
    CkExit();
  }

  patternNum = atoi(m->argv[2]);
  if ((patternNum < 0) || (patternNum >= NUM_AVAILABLE_PATTERNS)) {
    CkPrintf("Invalid pattern number: %d\n", patternNum);
    CkPrintf("Options: 0-%d\n", NUM_AVAILABLE_PATTERNS - 1);
    CkExit();
  }

  CkPrintf("Pattern %d selected with %d Worker objects spread across %d PE(s)\n", patternNum, numWorkers, CkNumPes());

  POSE_init();

  int mappedPE;
  WorkerInitMsg *initMsg;
  for (int workerNum = 0; workerNum < numWorkers; workerNum++) {
    initMsg = new WorkerInitMsg(workerNum, numWorkers, patternNum);
    initMsg->Timestamp(0);
    mappedPE = workerNum % CkNumPes();
    dbPrintf("Placing Worker %d on PE %d\n", workerNum, mappedPE);
    (*(CProxy_Worker *) &POSE_Objects)[workerNum].insert(initMsg, mappedPE);
  }
  POSE_Objects.doneInserting();
}
