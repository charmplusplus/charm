#include <unistd.h>
#include <math.h>
#include "pose.h"
#include "pgm.h"
#include "Pgm.def.h"
#include "Worker_sim.h"

main::main(CkArgMsg *m)
{
  CkGetChareID(&mainhandle);
  //  CProxy_main M(mainhandle);

  int numObjs=-1, numMsgs=-1, msgSize=-1, distribution=-1, connectivity=-1,
    locality=-1, grainSize=-1, elapsePattern=-1, offsetPattern=-1,
    sendPattern=-1, pattern=-1, i;
  double granularity=-1.0;
  char grainString[20];
  char *text;

  if(m->argc<10) {
    CkPrintf("Usage: asim <numObjs> <numMsgs> <msgSize> <distribution> <connectivitiy> <locality> <endTime> [ -g[f|m|c|z] | -t<granularity> ] <pattern>\n");
    CkExit();
  }
  numObjs = atoi(m->argv[1]);
  map = (int *)malloc(numObjs*sizeof(int));
  numMsgs = atoi(m->argv[2]);
  msgSize = atoi(m->argv[3]);
  text = "";
  if (msgSize == MIX_MS) { text = "MIXED"; }
  else if (msgSize == SMALL) { text = "SMALL"; }
  else if (msgSize == MEDIUM) { text = "MEDIUM"; }
  else if (msgSize == LARGE) { text = "LARGE"; }

  CkPrintf("asim run with: %d objects  %d messages  %s message size\n",
	   numObjs, numMsgs, text);

  if (strcmp(m->argv[4], "RANDOM") == 0)
    distribution = RANDOM;
  else if (strcmp(m->argv[4], "IMBALANCED") == 0)
    distribution = IMBALANCED;
  else if (strcmp(m->argv[4], "UNIFORM") == 0)
    distribution = UNIFORM;
  else {
    CkPrintf("Invalid distribution type: %s\n", m->argv[4]);
    CkExit();
  }

  if (strcmp(m->argv[5], "SPARSE") == 0)
    connectivity = SPARSE;
  else if (strcmp(m->argv[5], "HEAVY") == 0)
    connectivity = HEAVY;
  else if (strcmp(m->argv[5], "FULL") == 0)
    connectivity = FULL;
  else {
    CkPrintf("Invalid connectivity type: %s\n", m->argv[5]);
    CkExit();
  }

  locality = atoi(m->argv[6]);
  strcpy(grainString, m->argv[8]);
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

  CkPrintf("%s distribution  %s connectivity  %d%% locality  %d endtime  %s grainsize %f granularity\n",
	   m->argv[4], m->argv[5], locality, atoi(m->argv[7]), text, granularity);

  pattern = atoi(m->argv[9]);
  elapsePattern = pattern / 100;
  pattern -= elapsePattern*100;
  offsetPattern = pattern / 10;
  pattern -= offsetPattern*10;
  sendPattern = pattern;

  CkPrintf("  %d elapsePattern  %d offsetPattern  %d sendPattern\n",
	   elapsePattern, offsetPattern, sendPattern);

#if USE_LONG_TIMESTAMPS
  long long endtime = atoll(m->argv[7]);
  if(endtime == -1)
    POSE_init();
  else
    POSE_init(endtime);
#else
  int endtime = atoll(m->argv[7]);
  if(endtime == -1)
    POSE_init();
  else
    POSE_init(endtime);
#endif

  // create all the workers
  WorkerData *wd;
  int dest, j;
  srand48(42);
  buildMap(numObjs, distribution);
  for (i=0; i<numObjs; i++) {
    wd = new WorkerData;
    dest = map[i];

    wd->numObjs = numObjs;
    wd->numMsgs = numMsgs;
    wd->msgSize = msgSize;
    wd->distribution = distribution;
    wd->connectivity = connectivity;
    wd->locality = locality;
    wd->grainSize = grainSize;
    wd->elapsePattern = elapsePattern;
    wd->offsetPattern = offsetPattern;
    wd->sendPattern = sendPattern;

    wd->granularity = granularity;

    // compute elapseTimes, numSends, offsets, neighbors, numNbrs
    if (connectivity == SPARSE) wd->numNbrs = 4;
    else if (connectivity == HEAVY) wd->numNbrs = 25;
    else if (connectivity == FULL) wd->numNbrs = 100;

    if (elapsePattern == 1)
      for (j=0; j<5; j++) wd->elapseTimes[j] = (lrand48() % 2);
    else if (elapsePattern == 2)
      for (j=0; j<5; j++) wd->elapseTimes[j] = (lrand48() % 48) + 3;
    else if (elapsePattern == 3)
      for (j=0; j<5; j++) wd->elapseTimes[j] = (lrand48() % 50) + 51;
    else if (elapsePattern == 4)
      for (j=0; j<5; j++) wd->elapseTimes[j] = (lrand48() % 400) + 101;
    else if (elapsePattern == 5)
      for (j=0; j<5; j++) wd->elapseTimes[j] = (lrand48() % 500) + 501;

    if (offsetPattern == 1)
      for (j=0; j<5; j++) wd->offsets[j] = (lrand48() % 2);
    else if (offsetPattern == 2)
      for (j=0; j<5; j++) wd->offsets[j] = (lrand48() % 48) + 3;
    else if (offsetPattern == 3)
      for (j=0; j<5; j++) wd->offsets[j] = (lrand48() % 50) + 51;
    else if (offsetPattern == 4)
      for (j=0; j<5; j++) wd->offsets[j] = (lrand48() % 400) + 101;
    else if (offsetPattern == 5)
      for (j=0; j<5; j++) wd->offsets[j] = (lrand48() % 500) + 501;

    if (sendPattern == 1)
      for (j=0; j<5; j++) wd->numSends[j] = (lrand48()%numMsgs)/4;
    else if (sendPattern == 2)
      for (j=0; j<5; j++) wd->numSends[j] = (lrand48()%numMsgs)/3;
    else if (sendPattern == 3)
      for (j=0; j<5; j++) wd->numSends[j] = (lrand48()%numMsgs)/2;
    else if (sendPattern == 4)
      for (j=0; j<5; j++) wd->numSends[j] = (lrand48()%numMsgs);

    for (j=0; j<wd->numNbrs; j++)
      wd->neighbors[j] = getAnbr(numObjs, locality, dest);

    wd->Timestamp(0);
    //wd->dump();
    if (distribution == RANDOM)
      (*(CProxy_worker *) &POSE_Objects)[i].insert(wd);
    else
      (*(CProxy_worker *) &POSE_Objects)[i].insert(wd, dest);
  }
  POSE_Objects.doneInserting();
}

void main::buildMap(int numObjs, int dist)
{
  int i, j=0, k;
  if (dist == RANDOM)
    for (i=0; i<numObjs; i++) map[i] = lrand48() % CkNumPes();
  else if (dist == UNIFORM)
    for (i=0; i<numObjs; i++) map[i] = i % CkNumPes();
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

