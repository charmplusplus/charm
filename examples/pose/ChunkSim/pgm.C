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

  int numTeams=-1, numWorkers=-1, i;

  if(m->argc!=4) {
    CkPrintf("Usage: asim <numTeams> <numWorkers> <endTime>\n");
    CkExit();
  }
  numTeams = atoi(m->argv[1]);
  numWorkers = atoi(m->argv[2]);
  map = (int *)malloc(numTeams*sizeof(int));

  CkPrintf("asim run with: %d teams of %d workers", numTeams, numWorkers);
  CkPrintf("%d endtime\n", atoi(m->argv[3]));

  POSE_init(atoi(m->argv[3]));
  CkPrintf("POSE_endtime = %d\n", POSE_endtime);

  // create all the teams of workers
  TeamData *td;
  int dest, j, k, wid=0;
  srand48(42);
  buildMap(numTeams, UNIFORM);
  for (k=0; k<numTeams; k++) { // create numTeams teams
    dest = map[k];
    td = new TeamData;
    td->teamID = k;
    td->numTeams = numTeams;
    td->numWorkers = numWorkers*numTeams;
    td->Timestamp(0);
    (*(CProxy_team *) &POSE_Objects)[k].insert(td, dest);
  }
}

void main::buildMap(int numObjs, int dist)
{
  int i, j=0, k;
  if (dist == RANDOM)
    for (i=0; i<numObjs; i++) map[i] = lrand48() % CkNumPes();
  else if (dist == UNIFORM) {
    i=0;
    for (j=0; j<CkNumPes(); j++)
      for (k=0; k<numObjs/CkNumPes(); k++) {
	map[i] = j; 
	i++;
      }
    while (i< numObjs) map[i] = CkNumPes()-1;
  }
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

