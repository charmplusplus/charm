#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "speed.h"

void 
test::measure(int nc)
{
  nchunks = nc;
  int s = LDProcessorSpeed();
  CkPrintf("[%d] speed = %d \n", CkMyPe(), s);
  CProxy_test grp(thisgroup);
  grp.recv(CkMyPe(), s);
}

void
test::recv(int pe, int speed)
{
  speeds[pe] = speed;
  numrecd++;
  if(numrecd==CkNumPes())
  {
    distrib();
    if(CkMyPe()==0)
      CkExit();
  }
}

double *rem;

int cmp(const void *first, const void *second)
{
  int fi = *((const int *)first);
  int si = *((const int *)second);
  return ((rem[fi]==rem[si]) ? 0 : ((rem[fi]<rem[si]) ? 1 : (-1)));
}

void
test::distrib(void)
{
  double total = 0.0;
  int i;
  for(i=0;i<numrecd;i++)
    total += (double) speeds[i];
  double *nspeeds = new double[numrecd];
  for(i=0;i<numrecd;i++)
    nspeeds[i] = (double) speeds[i] / total;
  int *cp = new int[numrecd];
  for(i=0;i<numrecd;i++)
    cp[i] = (int) (nspeeds[i]*nchunks);
  int nr = 0;
  for(i=0;i<numrecd;i++)
    nr += cp[i];
  nr = nchunks - nr;
  if(nr != 0)
  {
    rem = new double[numrecd];
    for(i=0;i<numrecd;i++)
      rem[i] = (double)nchunks*nspeeds[i] - cp[i];
    int *pes = new int[numrecd];
    for(i=0;i<numrecd;i++)
      pes[i] = i;
    qsort(pes, numrecd, sizeof(int), cmp);
    for(i=0;i<nr;i++)
      cp[pes[i]]++;
  }
  int minspeed = INT_MAX;
  for(i=0;i<numrecd;i++)
    if(minspeed > speeds[i])
      minspeed = speeds[i];
  double *rel = new double[numrecd];
  for(i=0;i<numrecd;i++)
    rel[i] = (double)speeds[i] /(double)minspeed;
  int *rr = new int[numrecd];
  for(i=0;i<numrecd;i++)
    rr[i] = 0;
  int j = 0;
  for(i=0;i<nchunks;i++)
  {
    rr[j]++;
    j = (j+1)%numrecd;
  }
  double rrtime = 0.0;
  double proptime = 0.0;
  for(i=0;i<numrecd;i++)
  {
    double ptime = (double)rr[i]/rel[i];
    if(rrtime < ptime)
      rrtime = ptime;
    ptime = (double)cp[i]/rel[i];
    if(proptime < ptime)
      proptime = ptime;
  }
  if(CkMyPe()==0)
  {
    char *str = new char[1024];
    char stmp[32];
    sprintf(str, "Distrib: ");
    for(i=0;i<numrecd;i++)
    {
      sprintf(stmp, "%d=>%d ", i, cp[i]);
      strcat(str, stmp);
    }
    CkPrintf("%s\n", str);
    CkPrintf("Expected perf improvement using prop map: %lf percent\n",
              ((rrtime-proptime))*100.0/rrtime);
  }
}

main::main(CkArgMsg* m)
{
  if(m->argc < 2) {
    CkPrintf("Usage: charmrun pgm +pN <nchunks>\n");
    CkAbort("");
  }
  int nchunks = atoi(m->argv[1]);
  CkGroupID gid = CProxy_test::ckNew();
  CProxy_test grp(gid);
  grp.measure(nchunks);
  delete m;
}

#include "speed.def.h"

