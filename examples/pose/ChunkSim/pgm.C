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

  int numTeams=-1, numWorkers=-1, numObjs=-1, numMsgs=-1, msgSize=-1, 
    distribution=-1, connectivity=-1, locality=-1, offsetPattern=-1, 
    sendPattern=-1, pattern=-1, i;
  char *text;

  if(m->argc<11) {
    CkPrintf("Usage: asim <numTeams> <numWorkers> <numObjs> <numMsgs> <msgSize> <distribution> <connectivitiy> <locality> <endTime> <pattern>\n");
    CkExit();
  }
  numTeams = atoi(m->argv[1]);
  numWorkers = atoi(m->argv[2]);
  numObjs = atoi(m->argv[3]);
  map = (int *)malloc(numTeams*sizeof(int));
  numMsgs = atoi(m->argv[4]);
  msgSize = atoi(m->argv[5]);
  text = "";
  if (msgSize == MIX_MS) { text = "MIXED"; }
  else if (msgSize == SMALL) { text = "SMALL"; }
  else if (msgSize == MEDIUM) { text = "MEDIUM"; }
  else if (msgSize == LARGE) { text = "LARGE"; }

  CkPrintf("asim run with: %d objects  %d messages  %s message size\n",
	   numObjs, numMsgs, text);

  if (strcmp(m->argv[6], "RANDOM") == 0)
    distribution = RANDOM;
  else if (strcmp(m->argv[6], "IMBALANCED") == 0)
    distribution = IMBALANCED;
  else if (strcmp(m->argv[6], "UNIFORM") == 0)
    distribution = UNIFORM;
  else {
    CkPrintf("Invalid distribution type: %s\n", m->argv[4]);
    CkExit();
  }

  if (strcmp(m->argv[7], "SPARSE") == 0)
    connectivity = SPARSE;
  else if (strcmp(m->argv[7], "HEAVY") == 0)
    connectivity = HEAVY;
  else if (strcmp(m->argv[7], "FULL") == 0)
    connectivity = FULL;
  else { 
    CkPrintf("Invalid connectivity type: %s\n", m->argv[5]);
    CkExit();
  }

  locality = atoi(m->argv[8]);
  POSE_endtime = atoi(m->argv[9]);

  CkPrintf("%s distribution  %s connectivity  %d%% locality  %d endtime\n",
	   m->argv[6], m->argv[7], locality, POSE_endtime);

  pattern = atoi(m->argv[10]);
  offsetPattern = pattern / 10;
  pattern -= offsetPattern*10;
  sendPattern = pattern;

  CkPrintf("  %d offsetPattern  %d sendPattern\n", offsetPattern, sendPattern);

  POSE_init();
  POSE_useET(atoi(m->argv[7]));
  POSE_useID();

  // create all the teams of workers
  WorkerData *wd;
  TeamData *td;
  int dest, j, k, wid=0;
  srand48(42);
  buildMap(numTeams, distribution);
  for (k=0; k<numTeams; k++) { // create numTeams teams
    dest = map[k];
    td = new TeamData;
    td->numTeams = numTeams;
    td->numWorkers = numWorkers;
    td->numObjs = numObjs;
    td->Timestamp(0);
    (*(CProxy_team *) &POSE_Objects)[k].insert(td, dest);
  }
  for (k=0; k<numTeams; k++) { // for each of the numTeams teams, 
    for (i=0; i<numWorkers; i++) {  // create numWorkers workers
      wd = new WorkerData;
      wd->workerID = wid;
      wid++;
      wd->numWorkers = numWorkers;
      wd->numObjs = numObjs;
      wd->numMsgs = numMsgs;
      wd->msgSize = msgSize;
      wd->distribution = distribution;
      wd->connectivity = connectivity;
      wd->locality = locality;
      wd->offsetPattern = offsetPattern;
      wd->sendPattern = sendPattern;

      // compute numSends, offsets, neighbors, numNbrs
      if (connectivity == SPARSE) wd->numNbrs = 4; 
      else if (connectivity == HEAVY) wd->numNbrs = 25;
      else if (connectivity == FULL) wd->numNbrs = 100;
    
      if (offsetPattern == 1) 
	for (j=0; j<5; j++) wd->offsets[j] = 5;
      else if (offsetPattern == 2)
	for (j=0; j<5; j++) wd->offsets[j] = lrand48() % 11;
      else if (offsetPattern == 3)
	for (j=0; j<5; j++) wd->offsets[j] = (lrand48() % 40) + 51;
      else if (offsetPattern == 4)
	for (j=0; j<5; j++) wd->offsets[j] = (lrand48() % 50) + 101;
      else if (offsetPattern == 5)
	for (j=0; j<5; j++) wd->offsets[j] = lrand48() % 201;
      
      if (sendPattern == 1) 
	for (j=0; j<5; j++) wd->numSends[j] = (lrand48()%numMsgs) % 2;
      else if (sendPattern == 2)
	for (j=0; j<5; j++) wd->numSends[j] = (lrand48()%numMsgs) % (numMsgs+4/4) + 1;
      else if (sendPattern == 3)
	for (j=0; j<5; j++) wd->numSends[j] = (lrand48()%numMsgs) % (numMsgs+2/2) + 1;
      else if (sendPattern == 4)
	for (j=0; j<5; j++) wd->numSends[j] = (lrand48()%numMsgs) % numMsgs + 1;
      
      for (j=0; j<wd->numNbrs; j++) 
	wd->neighbors[j] = getAnbr(numObjs, locality, dest);

      (*(CProxy_team *) &POSE_Objects)[k].addWorker(wd);
    }
  }
  POSE_start();
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

