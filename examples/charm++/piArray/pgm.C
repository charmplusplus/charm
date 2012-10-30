#include "pgm.h"

CkChareID mainhandle;

#define USE_REDUCTION 0

void reductionHandler(void *param, int dataSize, void *data)
{
  CProxy_main m(mainhandle);
  int d = *(int*)data;
  m.results(d);
}

main::main(CkArgMsg * m)
{ 
  if(m->argc != 3)
  {
    CkPrintf("Usage: pgm <nsamples> <nchares>\n");
    CkAbort("");
  }

  ns = atoi(m->argv[1]);
  nc = atoi(m->argv[2]);
  delete m;

  starttime = CkWallTimer();

  //FIXME
  //CkGroupID gid = CkCreatePropMap();
  //CProxy_piPart  arr = CProxy_piPart::ckNew(nc, gid);
  CProxy_piPart  arr = CProxy_piPart::ckNew(nc);

  CkPrintf("At time %lf, array created.\n", (CkWallTimer()-starttime));

#if USE_REDUCTION
  arr.setReductionClient(reductionHandler,(void *)NULL);
#endif
  arr.compute(ns);
  responders = nc;
  count = 0;
  mainhandle = thishandle; // readonly initialization
  CkPrintf("At time %lf, main exits.\n", (CkWallTimer()-starttime));
}

void main::results(int cnt) 
{ 
  count += cnt;
#if ! USE_REDUCTION
  if (0 == --responders) 
#endif
  {
    endtime = CkWallTimer();
    CkPrintf("At time %lf, pi=: %f \n", (endtime-starttime), 4.0*count/(ns*nc));
    CkExit();
  }
}

piPart::piPart()
{ 
  CrnSrand((int)(long) this);
}

void
piPart::compute(int ns)
{
  int i;
  int count=0;
  for (i= 0; i<ns; i++) {
    double x = CrnDrand();
    double y = CrnDrand();
    if ((x*x + y*y) <= 1.0) count++;
  }
#if ! USE_REDUCTION
  CProxy_main mainproxy(mainhandle);
  mainproxy.results(count);
#else
  contribute(sizeof(int), (void *)&count, CkReduction::sum_int);
#endif
}

#include "pgm.def.h"
