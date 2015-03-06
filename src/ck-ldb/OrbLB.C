/**
 * \addtogroup CkLdb
   Load balancer that use Orthogonal Recursive Bisection(ORB) to partition
   objects and map to processors. In OrbLB, objects are treated to be enclosed 
   by a rectangular box using their LDObjid as coordinates.

   Written by Gengbin Zheng

   ORB now takes background load into account
   3/26/2010:        added support for avail_vector
*/
/*@{*/

#include "OrbLB.h"

//#define DEBUG

CreateLBFunc_Def(OrbLB, "partition objects based on coordinates")

CkpvExtern(int, _lb_obj_index);

void OrbLB::init()
{
  lbname = "OrbLB";
#if CMK_LB_USER_DATA
  if (use_udata && CkpvAccess(_lb_obj_index) == -1)
    CkpvAccess(_lb_obj_index) = LBRegisterObjUserData(sizeof(CkArrayIndex));
#endif
}


OrbLB::OrbLB(const CkLBOptions &opt, bool userdata): CBase_OrbLB(opt)
{
  use_udata = userdata;
  init();
  if (CkMyPe() == 0)
    CkPrintf("[%d] OrbLB created\n",CkMyPe());
}

OrbLB::OrbLB(const CkLBOptions &opt): CBase_OrbLB(opt)
{
  use_udata = true;
  init();
  if (CkMyPe() == 0)
    CkPrintf("[%d] OrbLB created\n",CkMyPe());
}

bool OrbLB::QueryBalanceNow(int _step)
{
  return true;
}

void OrbLB::rec_divide(int n, Partition &p)
{
  int i;
  int midpos;
  int n1, n2;
  double load1, currentload;
  int maxdir, count;
  Partition p1, p2;

  if (_lb_args.debug()>=2) {
    CmiPrintf("rec_divide starts: partition n:%d count:%d load:%f (%d %d %d, %d %d %d)\n", n, p.count, p.load, p.origin[0], p.origin[1], p.origin[2], p.corner[0], p.corner[1], p.corner[2]);
  }

  if (n==1) {		// we are done in this branch
    partitions[currentp++] = p;
    return;
  }
/*
  if (p.origin.x==p.corner.x && p.origin.y==p.corner.y && p.origin.z==p.corner.z) 
     CmiAbort("AlgRecBisection failed in recursion.\n"); 
*/
  if (_lb_args.debug()>=2) {
    CmiPrintf("{\n");
  }

  // divide into n1 and n2 two subpartitions
  n2 = n/2;
  n1 = n-n2;

  // subpartition n1 should have this load
  load1 = (1.0*n1/n) * p.load;
  if (_lb_args.debug()>=2)
    CmiPrintf("goal: n1: %d with load1: %f; n2: %d load2: %f\n", n1, load1, n2, p.load-load1);

  p1 = p;
  p1.refno = ++refno;
  p1.bkpes.resize(0);

  p2 = p;
  p2.refno = ++refno;
  p2.bkpes.resize(0);

  // determine the best division direction
  int maxSpan=-1;
  maxdir = XDIR;
  for (i=XDIR; i<=ZDIR; i++) {
    int myspan = p.corner[i] - p.origin[i];
    if (myspan > maxSpan) {
      maxdir = i;
      maxSpan = myspan;
    }
  }

  // other two dimensions
  int dir2 = (maxdir+1)%3;
  int dir3 = (maxdir+2)%3;

  currentload = 0.0;
  // counting background load
  if (!_lb_args.ignoreBgLoad()) {
    CmiAssert(p.bkpes.size() == n);
    // first n1 processors
    for (i=0; i<n1; i++) currentload += statsData->procs[p.bkpes[i]].bg_walltime;
  }

  count = 0;
  midpos = p.origin[maxdir];
  for (i=0; i<nObjs; i++) {
    // not belong to this partition
    if (computeLoad[vArray[maxdir][i].id].refno != p.refno) continue;
    if (vArray[maxdir][i].v<p.origin[maxdir]) continue;
    if (vArray[maxdir][i].v>p.corner[maxdir]) break;

    int cid = vArray[maxdir][i].id;	// this compute ID
    // check if this compute is within the partition
    if ( computeLoad[cid].v[dir2] >= p.origin[dir2] &&
	 computeLoad[cid].v[dir2] <= p.corner[dir2] &&
	 computeLoad[cid].v[dir3] >= p.origin[dir3] &&
	 computeLoad[cid].v[dir3] <= p.corner[dir3]  ) {
      // this compute is set to the first partition
      if (currentload <= load1) {
	computeLoad[cid].refno = p1.refno;
        currentload += computeLoad[cid].load;
        count ++;
	midpos = computeLoad[cid].v[maxdir];
      }
      else {	// or the next partition
	computeLoad[cid].refno = p2.refno;
      }
    }
  }
#ifdef DEBUG
//  CmiPrintf("X:cur:%d, prev:%d load:%f %f\n", cur, prev, currentload, prevload);
  CmiPrintf("DIR:%d %d load:%f\n", maxdir, midpos, currentload);
#endif

  p1.corner[maxdir] = midpos;
  p2.origin[maxdir] = midpos;

  p1.load = currentload;
  p1.count = count;
  p2.load = p.load - p1.load;
  p2.count = p.count - p1.count;

  // assign first n1 copy of background to p1, and rest to p2
  if (!_lb_args.ignoreBgLoad()) {
    for (i=0; i<n; i++)
      if (i<n1) p1.bkpes.push_back(p.bkpes[i]);
      else p2.bkpes.push_back(p.bkpes[i]);
  }

  if (_lb_args.debug()>=2) {
    CmiPrintf("p1: n:%d count:%d load:%f\n", n1, p1.count, p1.load);
    CmiPrintf("p2: n:%d count:%d load:%f\n", n2, p2.count, p2.load);
    CmiPrintf("}\n");
  }

  rec_divide(n1, p1);
  rec_divide(n2, p2);
}

void OrbLB::setVal(int x, int y, int z)
{
  int i;
  for (i=0; i<nObjs; i++) {
    computeLoad[i].tv = 1000000.0*computeLoad[i].v[x]+
			1000.0*computeLoad[i].v[y]+
			computeLoad[i].v[z];
  }
#if 0
  CmiPrintf("original:%d\n", x);
  for (i=0; i<numComputes; i++) 
    CmiPrintf("%d ", computeLoad[i].tv);
  CmiPrintf("\n");
#endif
}

int OrbLB::sort_partition(int x, int p, int r)
{
  double mid = computeLoad[vArray[x][p].id].tv;
  int i= p;
  int j= r;
  while (1) {
    while (computeLoad[vArray[x][j].id].tv > mid && j>i) j--;
    while (computeLoad[vArray[x][i].id].tv < mid && i<j) i++;
    if (i<j) {
      if (computeLoad[vArray[x][i].id].tv == computeLoad[vArray[x][j].id].tv)
      {
	if (computeLoad[vArray[x][i].id].tv != mid) CmiAbort("my god!\n");
	if (i-p < r-j) i++;
	else j--;
	continue;
      }
      VecArray tmp = vArray[x][i];
      vArray[x][i] = vArray[x][j];
      vArray[x][j] = tmp;
    }
    else
      return j;
  }
}

void OrbLB::qsort(int x, int p, int r)
{
  if (p<r) {
    int q = sort_partition(x, p, r);
//CmiPrintf("midpoint: %d %d %d\n", p,q,r);
    qsort(x, p, q-1);
    qsort(x, q+1, r);
  }
}

void OrbLB::quicksort(int x)
{
  int y = (x+1)%3;
  int z = (x+2)%3;
  setVal(x, y, z);
  qsort(x, 0, nObjs-1);

#if 0
  CmiPrintf("result for :%d\n", x);
  for (int i=0; i<nObjs; i++) 
    CmiPrintf("%d ", computeLoad[vArray[x][i].id].tv);
  CmiPrintf("\n");
#endif
}

void OrbLB::mapPartitionsToNodes()
{
  int i,j;
#if 1
  if (!_lb_args.ignoreBgLoad()) {
      // processor mapping has already been determined by the background load pe
    for (i=0; i<npartition; i++) partitions[i].node = partitions[i].bkpes[0];
  }
  else {
    int n = 0;
    for (i=0; i<P; i++) { 
      if (!statsData->procs[i].available) continue;
      partitions[n++].node = i;
    }
  }
#else
  PatchMap *patchMap = PatchMap::Object();

  int **pool = new int *[P];
  for (i=0; i<P; i++) pool[i] = new int[P];
  for (i=0; i<P; i++) for (j=0; j<P; j++) pool[i][j] = 0;

  // sum up the number of nodes that patches of computes are on
  for (i=0; i<numComputes; i++)
  {
    for (j=0; j<P; j++)
      if (computeLoad[i].refno == partitions[j].refno) 
      {
	int node1 = patchMap->node(computes[i].patch1);
	int node2 = patchMap->node(computes[i].patch2);
	pool[j][node1]++;
	pool[j][node2]++;
      }
  }
#ifdef DEBUG
  for (i=0; i<P; i++) {
    for (j=0; j<P; j++) CmiPrintf("%d ", pool[i][j]);
    CmiPrintf("\n");
  }
#endif
  while (1)
  {
    int index=-1, node=0, eager=-1;
    for (j=0; j<npartition; j++) {
      if (partitions[j].node != -1) continue;
      int wantmost=-1, maxnodes=-1;
      for (k=0; k<P; k++) if (pool[j][k] > maxnodes && !partitions[k].mapped) {wantmost=k; maxnodes = pool[j][k];}
      if (maxnodes > eager) {
	index = j; node = wantmost; eager = maxnodes;
      }
    }
    if (eager == -1) break;
    partitions[index].node = node;
    partitions[node].mapped = 1;
  }

  for (i=0; i<P; i++) delete [] pool[i];
  delete [] pool;
#endif

/*
  if (_lb_args.debug()) {
    CmiPrintf("partition load: ");
    for (i=0; i<npartition; i++) CmiPrintf("%f ", partitions[i].load);
    CmiPrintf("\n");
    CmiPrintf("partitions to nodes mapping: ");
    for (i=0; i<npartition; i++) CmiPrintf("%d ", partitions[i].node);
    CmiPrintf("\n");
  }
*/
  if (_lb_args.debug()) {
    CmiPrintf("After partitioning: \n");
    for (i=0; i<npartition; i++) {
      double bgload = 0.0;
      if (!_lb_args.ignoreBgLoad())
        bgload = statsData->procs[partitions[i].bkpes[0]].bg_walltime;
      CmiPrintf("[%d=>%d] (%d,%d,%d) (%d,%d,%d) load:%f count:%d objload:%f\n", i, partitions[i].node, partitions[i].origin[0], partitions[i].origin[1], partitions[i].origin[2], partitions[i].corner[0], partitions[i].corner[1], partitions[i].corner[2], partitions[i].load, partitions[i].count, partitions[i].load-bgload);
    }
    for (i=npartition; i<P; i++) CmiPrintf("[%d] --------- \n", i);
  }

}

void OrbLB::work(LDStats* stats)
{
#if CMK_LBDB_ON
  int i,j;

  statsData = stats;

  P = stats->nprocs();

  // calculate total number of migratable objects
  nObjs = stats->n_migrateobjs;
#ifdef DEBUG
  CmiPrintf("ORB: num objects:%d\n", nObjs);
#endif

  // create computeLoad and calculate tentative computes coordinates
  computeLoad = new ComputeLoad[nObjs];
  for (i=XDIR; i<=ZDIR; i++) vArray[i] = new VecArray[nObjs];

  // v[0] = XDIR  v[1] = YDIR v[2] = ZDIR
  // vArray[XDIR] is an array holding the x vector for all computes
  int objIdx = 0;
  for (i=0; i<stats->n_objs; i++) {
    LDObjData &odata = stats->objData[i];
    if (odata.migratable == 0) continue;
    computeLoad[objIdx].id = objIdx;
#if CMK_LB_USER_DATA
    int x, y, z;
    if (use_udata) {
      CkArrayIndex *idx =
        (CkArrayIndex *)odata.getUserData(CkpvAccess(_lb_obj_index));
      x = idx->data()[0];
      y = idx->data()[1];
      z = idx->data()[2];
    } else {
      x = odata.objID().id[0];
      y = odata.objID().id[1];
      z = odata.objID().id[2];
    }
    computeLoad[objIdx].v[XDIR] = x;
    computeLoad[objIdx].v[YDIR] = y;
    computeLoad[objIdx].v[ZDIR] = z;
#else
    computeLoad[objIdx].v[XDIR] = odata.objID().id[0];
    computeLoad[objIdx].v[YDIR] = odata.objID().id[1];
    computeLoad[objIdx].v[ZDIR] = odata.objID().id[2];
#endif
#if CMK_LB_CPUTIMER
    computeLoad[objIdx].load = _lb_args.useCpuTime()?odata.cpuTime:odata.wallTime;
#else
    computeLoad[objIdx].load = odata.wallTime;
#endif
    computeLoad[objIdx].refno = 0;
    computeLoad[objIdx].partition = NULL;
    for (int k=XDIR; k<=ZDIR; k++) {
        vArray[k][objIdx].id = objIdx;
        vArray[k][objIdx].v = computeLoad[objIdx].v[k];
    }
#ifdef DEBUG
    CmiPrintf("Object %d: %d %d %d load:%f\n", objIdx, computeLoad[objIdx].v[XDIR], computeLoad[objIdx].v[YDIR], computeLoad[objIdx].v[ZDIR], computeLoad[objIdx].load);
#endif
    objIdx ++;
  }
  CmiAssert(nObjs == objIdx);

  double t = CkWallTimer();

  quicksort(XDIR);
  quicksort(YDIR);
  quicksort(ZDIR);
#ifdef DEBUG
  CmiPrintf("qsort time: %f\n", CkWallTimer() - t);
#endif

  npartition = 0;
  for (i=0; i<P; i++)
    if (stats->procs[i].available == true) npartition++;
  partitions = new Partition[npartition];

  double totalLoad = 0.0;
  int minx, miny, minz, maxx, maxy, maxz;
  minx = maxx= computeLoad[0].v[XDIR];
  miny = maxy= computeLoad[0].v[YDIR];
  minz = maxz= computeLoad[0].v[ZDIR];
  for (i=1; i<nObjs; i++) {
    totalLoad += computeLoad[i].load;
    if (computeLoad[i].v[XDIR] < minx) minx = computeLoad[i].v[XDIR];
    else if (computeLoad[i].v[XDIR] > maxx) maxx = computeLoad[i].v[XDIR];
    if (computeLoad[i].v[YDIR] < miny) miny = computeLoad[i].v[YDIR];
    else if (computeLoad[i].v[YDIR] > maxy) maxy = computeLoad[i].v[YDIR];
    if (computeLoad[i].v[ZDIR] < minz) minz = computeLoad[i].v[ZDIR];
    else if (computeLoad[i].v[ZDIR] > maxz) maxz = computeLoad[i].v[ZDIR];
  }

  top_partition.origin[XDIR] = minx;
  top_partition.origin[YDIR] = miny;
  top_partition.origin[ZDIR] = minz;
  top_partition.corner[XDIR] = maxx;
  top_partition.corner[YDIR] = maxy; 
  top_partition.corner[ZDIR] = maxz;

  top_partition.refno = 0;
  top_partition.load = 0.0;
  top_partition.count = nObjs;

  // if we take background load into account
  if (!_lb_args.ignoreBgLoad()) {
    top_partition.bkpes.resize(0);
    double total = totalLoad;
    for (i=0; i<P; i++) {
      if (!stats->procs[i].available) continue;
      double bkload = stats->procs[i].bg_walltime;
      total += bkload;
    }
    double averageLoad = total / npartition;
    for (i=0; i<P; i++) {
      if (!stats->procs[i].available) continue;
      double bkload = stats->procs[i].bg_walltime;
      if (bkload < averageLoad) top_partition.bkpes.push_back(i);
      else CkPrintf("OrbLB Info> PE %d with %f background load will have 0 object.\n", i, bkload);
    }
    npartition = top_partition.bkpes.size();
    // formally add these bg load to total load
    for (i=0; i<npartition; i++) 
      totalLoad += stats->procs[top_partition.bkpes[i]].bg_walltime; 
    if (_lb_args.debug()>=2) {
      CkPrintf("BG load: ");
      for (i=0; i<P; i++)  CkPrintf(" %f", stats->procs[i].bg_walltime);
      CkPrintf("\n");
      CkPrintf("Partition BG load: ");
      for (i=0; i<npartition; i++)  CkPrintf(" %f", stats->procs[top_partition.bkpes[i]].bg_walltime);
      CkPrintf("\n");
    }
  }

  top_partition.load = totalLoad;

  currentp = 0;
  refno = 0;

  // recursively divide
  rec_divide(npartition, top_partition);

  // mapping partitions to nodes
  mapPartitionsToNodes();

  // this is for sanity check
  int *num = new int[P];
  for (i=0; i<P; i++) num[i] = 0;

  for (i=0; i<nObjs; i++)
  {
    for (j=0; j<npartition; j++)
      if (computeLoad[i].refno == partitions[j].refno)   {
        computeLoad[i].partition = partitions+j;
        num[j] ++;
    }
    CmiAssert(computeLoad[i].partition != NULL);
  }

  for (i=0; i<npartition; i++)
    if (num[i] != partitions[i].count) 
      CmiAbort("OrbLB: Compute counts don't agree!\n");

  delete [] num;

  // Save output
  objIdx = 0;
  for(int obj=0;obj<stats->n_objs;obj++) {
      stats->to_proc[obj] = stats->from_proc[obj];
      LDObjData &odata = stats->objData[obj];
      if (odata.migratable == 0) { continue; }
      int frompe = stats->from_proc[obj];
      int tope = computeLoad[objIdx].partition->node;
      if (frompe != tope) {
        if (_lb_args.debug() >= 3) {
              CkPrintf("[%d] Obj %d migrating from %d to %d\n",
                     CkMyPe(),obj,frompe,tope);
        }
	stats->to_proc[obj] = tope;
      }
      objIdx ++;
  }

  // free memory
  delete [] computeLoad;
  for (i=0; i<3; i++) delete [] vArray[i];
  delete [] partitions;

  if (_lb_args.debug() >= 1)
    CkPrintf("OrbLB finished time: %fs\n", CkWallTimer() - t);
#endif
}

#include "OrbLB.def.h"

/*@}*/
