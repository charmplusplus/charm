#include "pose.h"
#include "ldbal.def.h"

CProxy_LBgroup TheLBG;
CProxy_LBstrategy TheLBstrategy;

LBgroup::LBgroup(void)
{
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    localStats = (localStat *)CkLocalBranch(theLocalStats);
#endif
  busy = reportTo = 0;
}

// local methods
int LBgroup::computeObjectLoad(POSE_TimeType ovt, POSE_TimeType eet, double rbOh, int sync, POSE_TimeType gvt)
{
  // an object can score a max of 100 if it is the heaviest an object can be
  int offset = eet - gvt; 

  if (!offset)  return 100;
  else if (offset < pose_config.spec_window)  return 90;
  else if (eet < 0)  return 50;
  else  return 80;
}

int LBgroup::computePeLoad()
{
  PVT *localPVT = (PVT *)CkLocalBranch(ThePVT);
  POSE_TimeType gvt = localPVT->getGVT();
  peLoad = 0;
  for (int i=0; i<objs.numSpaces; i++)
    if (objs.objs[i].present) {
      objs.objs[i].execPrio = 
	computeObjectLoad(objs.objs[i].ovt, objs.objs[i].eet, 
			  objs.objs[i].rbOh, objs.objs[i].sync, gvt);
      peLoad += objs.objs[i].execPrio;
    }
  return peLoad;
}

int LBgroup::findHeaviestUnder(int loadDiff, int prioLoad, int **mvObjs,
			       int pe, int *contrib)
{
  int objIdx = -1, i;
  int maxContribUnder = 0, newContrib;

  if (mvObjs[pe][0] > 0) {
    objIdx = mvObjs[pe][mvObjs[pe][0]];
    mvObjs[pe][0]--;
    *contrib = objs.objs[objIdx].execPrio;
    //CkPrintf("-");
    return objIdx;
  }

  for (i=0; i<objs.numSpaces; i++)
    if (objs.objs[i].present) {
      newContrib = objs.objs[i].execPrio;
      if ((newContrib <= loadDiff) && (newContrib > maxContribUnder)) {
	objIdx = i;
	maxContribUnder = newContrib;
      }
    }

  *contrib = maxContribUnder;
  //  if (objIdx >= 0) CkPrintf("|");
  return objIdx;
}

int LBgroup::objRegister(int arrayIdx, int sync, sim *myPtr)
{
  int i = objs.Insert(sync, arrayIdx, myPtr); // add to object list
  return(i*1000 + CkMyPe());                  // return a unique index
}

void LBgroup::objRemove(int arrayIdx)
{
  int idx = (arrayIdx-CkMyPe())/1000; // calculate local idx from unique idx
  objs.Delete(idx);                   // delete the object
}

void LBgroup::objUpdate(int ldIdx, POSE_TimeType ovt, POSE_TimeType eet, int ne, double rbOh, 
			int *srVec)
{
  int idx = (ldIdx-CkMyPe())/1000; // calculate local idx from unique idx
  objs.UpdateEntry(idx, ovt, eet, ne, rbOh, srVec); // update data for object
}

// entry methods
void LBgroup::calculateLocalLoad(void)
{

#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    localStats->TimerStart(LB_TIMER);
#endif
  if (busy) {
    TheLBG[CkMyPe()].calculateLocalLoad();
#if !CMK_TRACE_DISABLED
    if(pose_config.stats)
      localStats->TimerStop();
#endif
    return;
  }
  busy = 1;

  CProxy_LBstrategy rootLB(TheLBstrategy);
  LoadReport *lr = new LoadReport;
  
  objs.ResetComm();
  objs.RequestReport();

  lr->PE = CkMyPe();
  lr->peLoad = computePeLoad();

  CkPrintf("PE%d reporting load of %d.\n", CkMyPe(), lr->peLoad);

  // reduce loads to strategy
  rootLB[reportTo].recvLoadReport(lr);
  reportTo = (reportTo + 1) % CkNumPes();
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    localStats->TimerStop();
#endif
}

void LBgroup::balance(BalanceSpecs *bs)
{
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    localStats->TimerStart(LB_TIMER);
#endif
  int myIndex = bs->indexArray[CkMyPe()], i, start, end, objIdx;
  int underLoad, contrib;
  destMsg *dm;

  if (bs->sortArray[0].peLoad + pose_config.lb_diff < bs->sortArray[CkNumPes()-1].peLoad){
    // we only want to move objects that are not tightly coupled with other 
    // local objects via communication, so we build a list of movable objects
    // the very first time this processor is overloaded
    static int **movableObjs = NULL;
    if (!movableObjs) { // allocate space for the movable objects
      movableObjs = (int **)malloc(CkNumPes()*sizeof(int *));
      for (i=0; i<CkNumPes(); i++)
      movableObjs[i] = (int *)malloc(21*sizeof(int));
    }
    
    if ((peLoad > bs->avgLoad) &&
	(bs->sortArray[myIndex].peLoad - 
	 bs->sortArray[bs->sortArray[myIndex].startPEidx].peLoad 
	 > pose_config.lb_threshold)) {
      CkPrintf("[%d] overload: checking balancing prospects.\n", CkMyPe());
      // Load up the table with movable objects
      for (i=0; i<CkNumPes(); i++)
	movableObjs[i][0] = 0;
      for (i=0; i<objs.numSpaces; i++)
	if (objs.objs[i].present && 
	    (objs.objs[i].localComm < objs.objs[i].remoteComm)) {
	  objs.objs[i].maxComm = objs.objs[i].comm[0];
	  objs.objs[i].maxCommPE = 0;
	  for (int j=0; j<CkNumPes(); j++)
	    if (objs.objs[i].comm[j] > objs.objs[i].maxComm) {
	      objs.objs[i].maxComm = objs.objs[i].comm[j];
	      objs.objs[i].maxCommPE = j;
	    }
	  if (objs.objs[i].maxComm > objs.objs[i].localComm) {
	    //CkPrintf("[%d] Obj %3d@[%3d] ", CkMyPe(), i, objs.objs[i].index);
	    //for (int k=0; k<CkNumPes(); k++) CkPrintf("%3d ", objs.objs[i].comm[k]);
	    //CkPrintf("%3d %3d %3d %3d %3d\n", objs.objs[i].totalComm, objs.objs[i].localComm, objs.objs[i].remoteComm, objs.objs[i].maxComm, objs.objs[i].maxCommPE);
	    if (movableObjs[objs.objs[i].maxCommPE][0] < 20) {
	      movableObjs[objs.objs[i].maxCommPE][0]++;
	      movableObjs[objs.objs[i].maxCommPE][movableObjs[objs.objs[i].maxCommPE][0]] = i;
	    }
	  }
	}
      // done building movable objects table; now, try to migrate objects away
      start = bs->sortArray[myIndex].startPEidx;
      end = bs->sortArray[myIndex].endPEidx;
      if (start != -1) {
	for (i=start; i<=end; i++) {
	  //CkPrintf("start=%d end=%d avgPeLd=%f i=%d [i].peLoad=%f\n", start, end, bs->avgPeLd, i, bs->sortArray[i].peLoad);
	  if (bs->sortArray[myIndex].peLoad - bs->sortArray[i].peLoad 
	      < pose_config.lb_threshold)
	    break;
	  if (bs->avgLoad > bs->sortArray[i].peLoad)
	    underLoad = bs->avgLoad - bs->sortArray[i].peLoad;
	  else
	    underLoad = (bs->sortArray[myIndex].peLoad - bs->sortArray[i].peLoad)/2;
	  objIdx = findHeaviestUnder(underLoad, bs->sortArray[i].peLoad, 
				     movableObjs, bs->sortArray[i].PE, 
				     &contrib);
	  while (objIdx >= 0) {
	    dm = new destMsg;
	    dm->destPE = bs->sortArray[i].PE;
	    //CkPrintf("%d->%d ", CkMyPe(), dm->destPE);
	    CkPrintf("PE[%d] to migrate %d to PE %d: contrib %d to load %d with diff %d\n", CkMyPe(), objs.objs[objIdx].index, dm->destPE, contrib, bs->sortArray[i].peLoad, underLoad);
	    POSE_Objects[objs.objs[objIdx].index].Migrate(dm);
	    objs.objs[objIdx].present = 0;
	    underLoad -= contrib;
	    objIdx = findHeaviestUnder(underLoad, bs->sortArray[i].peLoad,
				       movableObjs, bs->sortArray[i].PE,
				       &contrib);
	  }
	}
      }
    }
    else CkPrintf("[%d] underload.\n", CkMyPe());
  }
  CkFreeMsg(bs);
  busy = 0;
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    localStats->TimerStop();
#endif
}

LBstrategy::LBstrategy(void)
{
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    localStats = (localStat *)CkLocalBranch(theLocalStats);
#endif
  peLoads = (int *)malloc(CkNumPes()*sizeof(int));
  for (int i=0; i<CkNumPes(); i++)
    peLoads[i] = -1;
}

// local methods

void LBstrategy::computeLoadMap(int avgLd, int ttlLd)
{
  BalanceSpecs *dm = new BalanceSpecs;
  int i, pe, start, count;
  int overLoad, underLoad;

  //  CkPrintf(":");
  dm->avgLoad = avgLd;
  dm->totalLoad = ttlLd;

  for (i=0; i<CkNumPes(); i++) {
    dm->indexArray[i] = -1;
    dm->sortArray[i].PE = dm->sortArray[i].startPEidx = 
      dm->sortArray[i].endPEidx = -1;
    dm->sortArray[i].peLoad = 0;
  }

  for (i=0; i<CkNumPes(); i++) {
    pe = findMinPE();
    dm->indexArray[pe] = i;
    dm->sortArray[i].PE = pe;
    dm->sortArray[i].peLoad = peLoads[pe];
    peLoads[pe] = -1;
  }

  pe = CkNumPes() - 1;
  start = 0;
  while (dm->sortArray[pe].peLoad > avgLd) {
    overLoad = dm->sortArray[pe].peLoad - avgLd;
    count = 0;
    while ((overLoad > (avgLd - dm->sortArray[start].peLoad)/2) 
	   && (start < pe)) {
      if (dm->sortArray[start].peLoad < avgLd) {
	underLoad = avgLd - dm->sortArray[start].peLoad;
	overLoad -= underLoad;
	count++;
	start++;
      }
      else {
	underLoad = (dm->sortArray[pe].peLoad - dm->sortArray[start].peLoad)/2;
	overLoad -= underLoad;
	count++;
	start++;
      }
    }
    if (count > 0) {
      dm->sortArray[pe].startPEidx = start - count;
      dm->sortArray[pe].endPEidx = start-1;
    }
    else
      dm->sortArray[pe].startPEidx = dm->sortArray[pe].endPEidx = -1;
    pe--;
  }

  CkPrintf("LB balance info: Total Load = %d; Avg load = %d\n", dm->totalLoad, dm->avgLoad);
  for (i=0; i<CkNumPes(); i++) CkPrintf("[%d] PE:%d PE Load:%d start:%d end:%d\n", i, dm->sortArray[i].PE, dm->sortArray[i].peLoad, dm->sortArray[i].startPEidx, dm->sortArray[i].endPEidx);
  TheLBG.balance(dm);
  CkPrintf("...DONE load balancing]\n");
}

int LBstrategy::findMinPE()
{
  int minPE = 0, i;
  int minLoad = peLoads[0];
  
  for (i=1; i<CkNumPes(); i++)
    if ((minLoad < 0) || 
	((peLoads[i] < minLoad) && (peLoads[i] >= 0))) {
      minLoad = peLoads[i];
      minPE = i;
    }
  return minPE;
}

// entry methods

void LBstrategy::recvLoadReport(LoadReport *lr)
{
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    localStats->TimerStart(LB_TIMER);
#endif
  int i, avgLd = 0, totalLd = 0;
  static int done=0;

  peLoads[lr->PE] = lr->peLoad;  
  CkFreeMsg(lr);
  done++;  

  if (done == CkNumPes()) {
    CkPrintf("[BEGIN load balancing on %d...\n", CkMyPe());
    for (i=0; i<CkNumPes(); i++)
      totalLd += peLoads[i];
    avgLd = totalLd / CkNumPes();

    CkPrintf("LB[%d] totalLd=%d avgLd=%d\n", CkMyPe(), totalLd, avgLd);
    computeLoadMap(avgLd, totalLd);

    done = 0;
  }
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    localStats->TimerStop();
#endif
}
