worker::worker(WorkerData *m)
{
  int i;
  numObjs = m->numObjs;
  numMsgs = m->numMsgs;
  tscale = m->tscale;
  locality = m->locality;
  grainSize = m->grainSize;
  granularity = m->granularity;
  delete m;
  POSE_srand(myHandle);
  WorkMsg *wm;
  if (myHandle == 0) { // populate system with messages
    for (int i=0; i<numMsgs; i++) {
      wm = new WorkMsg;
      wm->fromPE = -1;
      POSE_invoke(work(wm), worker, POSE_rand()%numObjs, 1+POSE_rand()%(tscale/100));
    }
  }
}

worker::worker()
{
}

worker& worker::operator=(const worker& obj)
{
  int i;
  rep::operator=(obj);
  numObjs = obj.numObjs;
  numMsgs = obj.numMsgs;
  tscale = obj.tscale;
  locality = obj.locality;
  grainSize = obj.grainSize;
  granularity = obj.granularity;
  return *this;
}

void worker::work(WorkMsg *m)
{
  //  CkPrintf("%d receiving work at %d\n", parent->thisIndex, ovt);
  WorkMsg *wm;
  int nbr=-1, away, sign, offset;

  // fake computation
  if (granularity > 0.0) POSE_busy_wait(granularity);
  else if (grainSize == FINE) POSE_busy_wait(FINE_GRAIN);
  else if (grainSize == MEDIUM_GS) POSE_busy_wait(MEDIUM_GRAIN);
  else if (grainSize == COARSE) POSE_busy_wait(COARSE_GRAIN);
  else if (grainSize == MIX_GS) {
    int gsIdx = POSE_rand() % 3;
    if (gsIdx == 0) POSE_busy_wait(FINE_GRAIN);
    else if (gsIdx == 1) POSE_busy_wait(MEDIUM_GRAIN);
    else POSE_busy_wait(COARSE_GRAIN);
  }
 
 // generate an event
  if (OVT() < tscale) {
    wm = new WorkMsg;
    wm->fromPE = myHandle;
    offset = 1 + POSE_rand() % (tscale/100);
    while ((nbr < 0) || (nbr >= numObjs)) {
      away = (POSE_rand() % (numObjs - ((int)((((float)locality)/100.0) * (float)numObjs)))) + 1;
      away = away/2;
      sign = POSE_rand() % 2;
      if (sign) nbr = myHandle+away;
      else nbr = myHandle-away;
    }
    POSE_invoke(work(wm), worker, nbr, offset);
    //CkPrintf("%d sending work to %d at %d. Sent=%d\n",myHandle,nbr,ovt,sent);
  }
}

void worker::work_anti(WorkMsg *m)
{
  restore(this);
}

void worker::work_commit(WorkMsg *m)
{
}
