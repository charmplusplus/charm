#include "charm++.h"
#include "traceBigSim.h"

CkpvDeclare(FILE*, bgfp);
CkpvDeclare(unsigned long, bgTraceCounter);


void initBigSimTrace()
{
  CkpvInitialize(FILE *, bgfp);
  CkpvInitialize(unsigned long, bgTraceCounter);
  CkpvAccess(bgTraceCounter) = 0;

#ifdef CMK_BIGSIM_CHARM
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
}

void finalizeBigSimTrace()
{
#ifdef CMK_BIGSIM_CHARM
  fclose(CkpvAccess(bgfp));
#endif  
}
