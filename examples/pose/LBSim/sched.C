#include "sched.h"
#include "sched.def.h"
#include "pose.h"
#include "SchedSIM_sim.h"

main::main(CkArgMsg *m) {
  int i;
  SchedulerData *sd;
  eventMsg *m1;

  // check and grab arguments
  if (m->argc != 6)
    CmiAbort("Usage: sched <maxObjects> <connectivity> <value> <endTime> <topology>\n");
  maxObjects = atoi(m->argv[1]);
  connectivity = atoi(m->argv[2]);
  n = atoi(m->argv[3]);
  POSE_endtime= atoi(m->argv[4]);
 
  // start simulation
  POSE_init(POSE_endtime);
  //POSE_useID();
    for (i=0; i<maxObjects; i++) {
    sd = new SchedulerData;
    sd->maxObjects = maxObjects;
    sd->connectivity = connectivity;
    //printf("lbtopo %s\n",m->argv[5]);

    
    sd->lbtopolen=strlen(m->argv[5]);
    sd->lbtopo=new char[sd->lbtopolen];
    strcpy(sd->lbtopo,m->argv[5]);
    sd->id = i;
    sd->data=0;
    sd->sum=0;
    if(i==0) sd->data=n;
    sd->Timestamp(0);
    (*(CProxy_schedulerObject *) &POSE_Objects)[i].insert(sd);
  }
    POSE_Objects.doneInserting();
}




