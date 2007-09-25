#include <unistd.h>
#include <math.h>
#include "pose.h"
#include "phold.h"
#include "PHOLD.def.h"
#include "Worker_sim.h"

main::main(CkArgMsg *m)
{ 
  CkGetChareID(&mainhandle);

  int numLPs=-1, numMsgs=-1, locality=1, grainSize=-1, msgDist, tsIncFn, 
    moveFn, tScale;
  double granularity=-1.0;
  char grainString[20];
  char *text;

  if(m->argc<7) {
    CkPrintf("Usage: phold <#LPs> <#initMsgs> <initMsgDist (RANDOM)> <timestampIncFn (RANDOM)> <moveFn (RANDOM)> [ -g[f|m|c|z] | -t<granularity> ] <spacialLocality (%)> <timescale (>=100)>\n");
    CkExit();
  }
  numLPs = atoi(m->argv[1]);
  map = (int *)malloc(numLPs*sizeof(int));
  numMsgs = atoi(m->argv[2]);
  if (strcmp(m->argv[3], "RANDOM") == 0)
    msgDist = RANDOM;
  else {
    CkPrintf("Invalid message distribution: %s\n", m->argv[3]);
    CkExit();
  }
  if (strcmp(m->argv[4], "RANDOM") == 0)
    tsIncFn = RANDOM;
  else {
    CkPrintf("Invalid timestamp increment function: %s\n", m->argv[4]);
    CkExit();
  }
  if (strcmp(m->argv[5], "RANDOM") == 0)
    moveFn = RANDOM;
  else {
    CkPrintf("Invalid movement function: %s\n", m->argv[5]);
    CkExit();
  }
  locality = atoi(m->argv[7]);
  tScale = atoi(m->argv[8]);

  strcpy(grainString, m->argv[6]);
  text = "";
  if (strcmp(grainString, "-gf") == 0) {
    grainSize = FINE; text = "FINE"; }
  else if (strcmp(grainString, "-gm") == 0) {
    grainSize = MEDIUM_GS; text = "MEDIUM"; }
  else if (strcmp(grainString, "-gc") == 0) {
    grainSize = COARSE; text = "COARSE"; }
  else if (strcmp(grainString, "-gz") == 0) {
    grainSize = MIX_GS; text = "MIXED"; }
  else if (strncmp(grainString, "-t", 2) == 0)
    granularity = atof(&(grainString[2]));

  CkPrintf(">>> PHOLD: %d LPs, %d initial messages distributed: %s ...\n", 
	   numLPs, numMsgs, m->argv[3]);
  CkPrintf(">>> ...timestamp incremented: %s  movement function: %s ...\n",
	   m->argv[4], m->argv[5]);
  CkPrintf(">>> ...grainsize: %s %3.8e  locality: %d  time scale: %d.\n", 
	   text, granularity, locality, tScale);

  CkPrintf("Procs %d nodes %d\n",CkNumPes(), CkNumNodes());

  POSE_init();

  // create all the workers
  WorkerData *wd;
  int dest;
  srand48(42);
  buildMap(numLPs, UNIFORM);
  for (int i=0; i<numLPs; i++) {
    wd = new WorkerData;
    wd->numObjs = numLPs;
    wd->numMsgs = numMsgs;
    wd->grainSize = grainSize;
    wd->granularity = granularity;
    wd->locality = locality;
    wd->tscale = tScale;
    dest = map[i];
    wd->Timestamp(0);
    (*(CProxy_worker *) &POSE_Objects)[i].insert(wd, dest);
  }
  POSE_Objects.doneInserting();
}

void main::buildMap(int numLPs, int dist)
{
  int i, j=0, k;
  if (dist == RANDOM)
    for (i=0; i<numLPs; i++) map[i] = lrand48() % CkNumPes();
  else if (dist == UNIFORM)
    for (i=0; i<numLPs; i++) map[i] = (i / (numLPs/CkNumPes()))%CkNumPes();
  else if (dist == IMBALANCED) {
    int min = (numLPs/CkNumPes())/2;
    if (min < 1) min = 1;
    for (k=0; k<CkNumPes(); k++)
      for (i=0; i<min; i++) {
	map[j] = k;
	j++;
      }
    i=CkNumPes()/2;
    for (k=j; k<numLPs; k++) {
      map[k] = i;
      i++;
      if (i == CkNumPes()) i = CkNumPes()/2;
    }
  }
}

int main::getAnbr(int numLPs, int locale, int dest)
{
  int here = (lrand48() % 101) <= locale;
  int idx;
  if (CkNumPes() == 1) return lrand48() % numLPs;
  if (here) {
    idx = lrand48() % numLPs;
    while (map[idx] != dest)
      idx = lrand48() % numLPs;
    return idx;
  }
  else {
    idx = lrand48() % numLPs;
    while (map[idx] == dest)
      idx = lrand48() % numLPs;
    return idx;
  }
}

