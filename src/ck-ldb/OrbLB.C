/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkLdb
   Load balancer that use Orthogonal Recursive Bisection(ORB) to partition
   objects and map to processors. In OrbLB, objects are treated to be enclosed 
   by a rectangular box using their LDObjid as coordinates.
*/
/*@{*/

#include <charm++.h>

#if CMK_LBDB_ON

#include "cklists.h"

#include "OrbLB.h"

//#define DEBUG

void CreateOrbLB()
{
  loadbalancer = CProxy_OrbLB::ckNew();
}

static void lbinit(void) {
  LBRegisterBalancer("ORbLB", CreateOrbLB, "partition objects based on coordinates");
}

#include "OrbLB.def.h"

OrbLB::OrbLB()
{
  lbname = "OrbLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] OrbLB created\n",CkMyPe());
}

CmiBool OrbLB::QueryBalanceNow(int _step)
{
  return CmiTrue;
}

void OrbLB::rec_divide(int n, Partition &p)
{
  int i;
  int midpos;
  int n1, n2;
  double load1, currentload;
  int maxdir, count;
  Partition p1, p2;

#ifdef DEBUG
  CmiPrintf("rec_divide: partition n:%d count:%d load:%f (%d %d %d, %d %d %d)\n", n, p.count, p.load, p.origin[0], p.origin[1], p.origin[2], p.corner[0], p.corner[1], p.corner[2]);
#endif

  if (n==1) {
    partitions[currentp++] = p;
    return;
  }
/*
  if (p.origin.x==p.corner.x && p.origin.y==p.corner.y && p.origin.z==p.corner.z) 
     NAMD_die("AlgRecBisection failed in recursion.\n"); 
*/

  n2 = n/2;
  n1 = n-n2;

  load1 = (1.0*n1/n) * p.load;

  p1 = p;
  p1.refno = ++refno;
  p2 = p;
  p2.refno = ++refno;

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
//  CmiPrintf("X:cur:%d, prev:%d load:%f %f\n", cur, prev, currentload, prevload);
#ifdef DEBUG
  CmiPrintf("DIR:%d %d load:%f\n", maxdir, midpos, currentload);
#endif

  p1.corner[maxdir] = midpos;
  p2.origin[maxdir] = midpos;

  p1.load = currentload;
  p1.count = count;
  p2.load = p.load - p1.load;
  p2.count = p.count - p1.count;
#ifdef DEBUG
  CmiPrintf("p1: n:%d count:%d load:%f\n", n1, p1.count, p1.load);
  CmiPrintf("p2: n:%d count:%d load:%f\n", n2, p2.count, p2.load);
#endif
  rec_divide(n1, p1);
  rec_divide(n2, p2);
}

void OrbLB::setVal(int x, int y, int z)
{
  int i;
  for (i=0; i<nObjs; i++) {
    computeLoad[i].tv = 1000000*computeLoad[i].v[x]+
			1000*computeLoad[i].v[y]+
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
  int mid = computeLoad[vArray[x][p].id].tv;
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
  CmiPrintf("result:%d\n", x);
  for (int i=0; i<numComputes; i++) 
    CmiPrintf("%d ", computeLoad[vArray[x][i].id].tv);
  CmiPrintf("\n");
#endif
}

void OrbLB::mapPartitionsToNodes()
{
  int i,j,k;
#if 1
  for (i=0; i<P; i++) partitions[i].node = i;
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
    for (j=0; j<P; j++) {
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

  CmiPrintf("partitions to nodes mapping: ");
  for (i=0; i<P; i++) CmiPrintf("%d ", partitions[i].node);
  CmiPrintf("\n");
}

LBMigrateMsg* OrbLB::Strategy(CentralLB::LDStats* stats, int count)
{
  int i,j;

  P = count;
  // calculate total number of objects
  nObjs = 0;
  for (i=0; i<count; i++) nObjs += stats[i].n_objs;
#ifdef DEBUG
  CmiPrintf("ORB: num objects:%d\n", nObjs);
#endif

  // create computeLoad and calculate tentative computes coordinates
  computeLoad = new ComputeLoad[nObjs];
  for (i=XDIR; i<=ZDIR; i++) vArray[i] = new VecArray[nObjs];

  // v[0] = XDIR  v[1] = YDIR v[2] = ZDIR
  // vArray[XDIR] is an array holding the x vector for all computes
  int objIdx = 0;
  for (i=0; i<count; i++) {
    int osz = stats[i].n_objs;
    LDObjData *odata = stats[i].objData;
    for (j=0; j<osz; j++) {
      computeLoad[objIdx].id = objIdx;
      computeLoad[objIdx].v[XDIR] = odata[j].id().id[0];
      computeLoad[objIdx].v[YDIR] = odata[j].id().id[1];
      computeLoad[objIdx].v[ZDIR] = odata[j].id().id[2];
      computeLoad[objIdx].load = odata[j].wallTime;
      computeLoad[objIdx].refno = 0;
      computeLoad[objIdx].partition = NULL;
      for (int k=XDIR; k<=ZDIR; k++) {
        vArray[k][objIdx].id = objIdx;
        vArray[k][objIdx].v = computeLoad[objIdx].v[k];
      }
      objIdx ++;
    }
  }

  double t = CmiWallTimer();

  quicksort(XDIR);
  quicksort(YDIR);
  quicksort(ZDIR);
#ifdef DEBUG
  CmiPrintf("qsort time: %f\n", CmiWallTimer() - t);
#endif

  npartition = P;
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
  top_partition.load = totalLoad;

  currentp = 0;
  refno = 0;

  // recursively divide
  rec_divide(npartition, top_partition);

  CmiPrintf("After partitioning: \n");
  for (i=0; i<P; i++) {
    CmiPrintf("[%d] (%d,%d,%d) (%d,%d,%d) load:%f count:%d\n", i, partitions[i].origin[0], partitions[i].origin[1], partitions[i].origin[2], partitions[i].corner[0], partitions[i].corner[1], partitions[i].corner[2], partitions[i].load, partitions[i].count);
  }

  // mapping partitions to nodes
  mapPartitionsToNodes();

  // this is for debugging
  int *num = new int[P];
  for (i=0; i<P; i++) num[i] = 0;

  for (i=0; i<nObjs; i++)
  {
    for (j=0; j<count; j++)
      if (computeLoad[i].refno == partitions[j].refno)   {
        computeLoad[i].partition = partitions+j;
        num[j] ++;
    }
    CmiAssert(computeLoad[i].partition != NULL);
  }

  for (i=0; i<P; i++)
    if (num[i] != partitions[i].count) 
      CmiAbort("OrbLB: Compute counts don't agree!\n");

  delete [] num;

  CkVec<MigrateInfo*> migrateInfo;

  // Save output
  objIdx = 0;
  for(int pe=0;pe < count; pe++) {
    for(int obj=0;obj<stats[pe].n_objs;obj++) {
      if (pe != computeLoad[objIdx].partition->node) {
        //      CkPrintf("[%d] Obj %d migrating from %d to %d\n",
        //               CkMyPe(),obj,pe,to_procs[pe][obj]);
        MigrateInfo *migrateMe = new MigrateInfo;
        migrateMe->obj = stats[pe].objData[obj].handle;
        migrateMe->from_pe = pe;
        migrateMe->to_pe = computeLoad[objIdx].partition->node;
        migrateInfo.insertAtEnd(migrateMe);
      }
      objIdx ++;
    }
  }

  int migrate_count=migrateInfo.length();
  LBMigrateMsg* msg = new(&migrate_count,1) LBMigrateMsg;
  msg->n_moves = migrate_count;
  for(int i=0; i < migrate_count; i++) {
    MigrateInfo* item = (MigrateInfo*)migrateInfo[i];
    msg->moves[i] = *item;
    delete item;
    migrateInfo[i] = 0;
  }

  delete [] computeLoad;
  for (i=0; i<3; i++) delete [] vArray[i];
  delete [] partitions;

  CmiPrintf("OrbLB finished time: %f\n", CmiWallTimer() - t);

  return msg;
}


#if 0
/*@}*/
LBMigrateMsg* OrbLB::Strategy(CentralLB::LDStats* stats, int count)
{
  int obj, pe;

  //  CkPrintf("[%d] RefineLB strategy\n",CkMyPe());

  // remove non-migratable objects
  RemoveNonMigratable(stats, count);


  CkVec<MigrateInfo*> migrateInfo;

  // Save output
  objIdx = 0;
  for(int pe=0;pe < count; pe++) {
    for(int obj=0;obj<stats[pe].n_objs;obj++) {
      if (to_procs[pe][obj] != pe) {
	//	CkPrintf("[%d] Obj %d migrating from %d to %d\n",
	//		 CkMyPe(),obj,pe,to_procs[pe][obj]);
	MigrateInfo *migrateMe = new MigrateInfo;
	migrateMe->obj = stats[pe].objData[obj].handle;
	migrateMe->from_pe = pe;
	migrateMe->to_pe = to_procs[pe][obj];
	migrateInfo.insertAtEnd(migrateMe);
      }
    }
  }

  int migrate_count=migrateInfo.length();
  LBMigrateMsg* msg = new(&migrate_count,1) LBMigrateMsg;
  msg->n_moves = migrate_count;
  for(int i=0; i < migrate_count; i++) {
    MigrateInfo* item = (MigrateInfo*)migrateInfo[i];
    msg->moves[i] = *item;
    delete item;
    migrateInfo[i] = 0;
  }

  return msg;
};
#endif

#endif


/*@}*/
