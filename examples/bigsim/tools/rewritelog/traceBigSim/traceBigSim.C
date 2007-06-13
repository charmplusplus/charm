#include "charm++.h"
#include "traceBigSim.h"

CkpvDeclare(FILE*, bgfp);
CkpvDeclare(unsigned long, bgTraceCounter);


CkpvDeclare(bool, insideTraceBracket);

void initBigSimTrace()
{
  CkpvInitialize(FILE *, bgfp);
  CkpvInitialize(bool, insideTraceBracket);
  CkpvInitialize(unsigned long, bgTraceCounter);
  CkpvAccess(bgTraceCounter) = 0;
  CkpvAccess(insideTraceBracket) = false;

#ifdef CMK_BLUEGENE_CHARM
  //   for bigsim emulation, write to files, one for each processor
  char fname[128];
  sprintf(fname, "param.%d", CkMyPe());
  CkpvAccess(bgfp) = fopen(fname, "w");
  if (CkpvAccess(bgfp) == NULL) 
    CmiAbort("Failed to generated trace param file!");
#else
  //   for Mambo simulation, write to screen for now
  CkpvAccess(bgfp) = stdout;
#endif




#ifdef BIG_SIM_PAPI
	CkPrintf("PAPI: number of available counters: %d\n", PAPI_num_counters());
	CkAssert(PAPI_num_counters() >= 0);
#endif

}

void finalizeBigSimTrace()
{
#ifdef CMK_BLUEGENE_CHARM
  fclose(CkpvAccess(bgfp));
#endif  
}
