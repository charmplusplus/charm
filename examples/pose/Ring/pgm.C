#include <unistd.h>
#include <math.h>
#include "pose.h"
#include "pgm.h"
#include "Pgm.def.h"
#include "Worker_sim.h"

main::main(CkArgMsg *m)
{ 
  CkGetChareID(&mainhandle);

  int numObjs=-1, numMsgs=-1, msgSize=-1, locality=-1, grainSize=-1;
  int density=-1, i, totalObjs=-1;
  double granularity=-1.0;
  char grainString[20];
  char *text;

  if(m->argc<6) {
    CkPrintf("Usage: simb <#objsPerProc> <#msgsPerObj> <msgSize(MIXED,SMALL,MEDIUM,LARGE)> [-g[f|m|c|z] | -t<granularity> ] <density(msgsPerVTU)>\n");
    CkExit();
  }
  numObjs = atoi(m->argv[1]);
  totalObjs = numObjs * CkNumPes();
  map = (int *)malloc(totalObjs*sizeof(int));
  numMsgs = atoi(m->argv[2]);
  if (strcmp(m->argv[3], "MIXED") == 0)
    msgSize = MIX_MS;
  else if (strcmp(m->argv[3], "SMALL") == 0)
    msgSize = SMALL;
  else if (strcmp(m->argv[3], "MEDIUM") == 0)
    msgSize = MEDIUM;
  else if (strcmp(m->argv[3], "LARGE") == 0)
    msgSize = LARGE;
  else {
    CkPrintf("Invalid message size: %s\n", m->argv[8]);
    CkExit();
  }

  CkPrintf(">>> simb run with %d objects per processor each to send %d messages of %s size...\n", numObjs, numMsgs, m->argv[3]);

  strcpy(grainString, m->argv[4]);
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
  density = atoi(m->argv[5]);

  CkPrintf(">>> ...Each event has %s granularity of %f on average...\n>>> ...Events are concentrated at approximately %d per Virtual Time Unit(VTU).\n", text, granularity, density);

  POSE_init();

  WorkerData *wd;
  wd = new WorkerData;
  wd->numObjs = numObjs;
  wd->numMsgs = numMsgs;
  wd->msgSize = msgSize;
  wd->grainSize = grainSize;
  wd->granularity = granularity;
  wd->density = density;
  wd->Timestamp(0);
  //Create the map group
  CProxy_BlockMap myMap=CProxy_BlockMap::ckNew();
  //Make a new array using that map
  CkArrayOptions opts(totalObjs);
  opts.setMap(myMap);
  POSE_Objects.doneInserting();
  (*(CProxy_worker *) &POSE_Objects) = CProxy_worker::ckNew(wd, opts);
  /*
  // create all the workers
  WorkerData *wd;
  int dest, j;
  srand48(42);
  buildMap(totalObjs, UNIFORM);
  for (i=0; i<totalObjs; i++) {
    wd = new WorkerData;
    wd->numObjs = numObjs;
    wd->numMsgs = numMsgs;
    wd->msgSize = msgSize;
    wd->grainSize = grainSize;
    wd->granularity = granularity;
    wd->density = density;

    dest = map[i];
    wd->Timestamp(0);
    //wd->dump();
    (*(CProxy_worker *) &POSE_Objects)[i].insert(wd, dest);
  }

  SmallWorkMsg *sm = new SmallWorkMsg;
  memset(sm->data, 0, SM_MSG_SZ*sizeof(int));
  sm->fromPE = -1;
  sm->Timestamp(0);
  (*(CProxy_worker *) &POSE_Objects).workSmall(sm);
  */
  POSE_startTimer();
}

void main::buildMap(int numObjs, int dist)
{
  int i, j=0, k;
  if (dist == RANDOM)
    for (i=0; i<numObjs; i++) map[i] = lrand48() % CkNumPes();
  else if (dist == UNIFORM)
    for (i=0; i<numObjs; i++) map[i] = i / (numObjs/CkNumPes());
  else if (dist == IMBALANCED) {
    int min = (numObjs/CkNumPes())/2;
    if (min < 1) min = 1;
    for (k=0; k<CkNumPes(); k++)
      for (i=0; i<min; i++) {
	map[j] = k;
	j++;
      }
    i=CkNumPes()/2;
    for (k=j; k<numObjs; k++) {
      map[k] = i;
      i++;
      if (i == CkNumPes()) i = CkNumPes()/2;
    }
  }
}

int main::getAnbr(int numObjs, int locale, int dest)
{
  int here = (lrand48() % 101) <= locale;
  int idx;
  if (CkNumPes() == 1) return lrand48() % numObjs;
  if (here) {
    idx = lrand48() % numObjs;
    while (map[idx] != dest)
      idx = lrand48() % numObjs;
    return idx;
  }
  else {
    idx = lrand48() % numObjs;
    while (map[idx] == dest)
      idx = lrand48() % numObjs;
    return idx;
  }
}

