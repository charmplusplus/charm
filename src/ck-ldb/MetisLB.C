#include <charm++.h>

#if CMK_LBDB_ON

#include "CkLists.h"

#include "MetisLB.h"
#include "MetisLB.def.h"

void CreateMetisLB()
{
  // CkPrintf("[%d] creating MetisLB %d\n",CkMyPe(),loadbalancer);
  loadbalancer = CProxy_MetisLB::ckNew();
  // CkPrintf("[%d] created MetisLB %d\n",CkMyPe(),loadbalancer);
}

MetisLB::MetisLB()
{
  if (CkMyPe() == 0)
    CkPrintf("[%d] MetisLB created\n",CkMyPe());
}

CmiBool MetisLB::QueryBalanceNow(int _step)
{
  // CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),_step);
  return CmiTrue;
}

static void printStats(int count, int numobjs, double *cputimes, 
                       int **comm, int *map)
{
  int i, j;
  double *petimes = new double[count];
  for(i=0;i<count;i++) {
    petimes[i] = 0.0;
  }
  for(i=0;i<numobjs;i++) {
    petimes[map[i]] += cputimes[i];
  }
  double maxpe = petimes[0], minpe = petimes[0];
  CkPrintf("\tPE\tTime\n");
  for(i=0;i<count;i++) {
    CkPrintf("\t%d\t%lf\n",i,petimes[i]);
    if(maxpe < petimes[i])
      maxpe = petimes[i];
    if(minpe > petimes[i])
      minpe = petimes[i];
  }
  delete[] petimes;
  CkPrintf("\tLoad Imbalance=%lf seconds\n", maxpe-minpe);
  int ncomm = 0;
  for(i=0;i<numobjs;i++) {
    for(j=0;j<numobjs;j++) {
      if(map[i] != map[j])
        ncomm += comm[i][j];
    }
  }
  CkPrintf("\tCommunication (off proc msgs) = %d\n", ncomm/2);
}

extern "C" void METIS_PartGraphRecursive(int*, int*, int*, int*, int*,
					 int*, int*, int*, int*,
					 int*, int*);
extern "C" void METIS_PartGraphKway(int*, int*, int*, int*, int*,
                                    int*, int*, int*, int*,
                                    int*, int*);
extern "C" void METIS_PartGraphVKway(int*, int*, int*, int*, int*,
				     int*, int*, int*, int*,
				     int*, int*);

// the following are to compute a partitioning with a given partition weights
// "W" means giving weights
extern "C" void METIS_WPartGraphRecursive(int*, int*, int*, int*, int*,
					  int*, int*, int*, float*, int*,
					  int*, int*);
extern "C" void METIS_WPartGraphKway(int*, int*, int*, int*, int*,
				     int*, int*, int*, float*, int*,
				     int*, int*);

// the following are for multiple constraint partition "mC"
extern "C" void METIS_mCPartGraphRecursive(int*, int*, int*, int*, int*, int*,
					 int*, int*, int*, int*,
					 int*, int*);
extern "C" void METIS_mCPartGraphKway(int*, int*, int*, int*, int*, int*,
                                    int*, int*, int*, int*, int*,
                                    int*, int*);
/*
CLBMigrateMsg* MetisLB::Strategy(CentralLB::LDStats* stats, int count,
				 int option=0)
*/
CLBMigrateMsg* MetisLB::Strategy(CentralLB::LDStats* stats, int count)
{
  // CkPrintf("entering MetisLB::Strategy...\n");
  // CkPrintf("[%d] MetisLB strategy\n",CkMyPe());
  CkVector migrateInfo;

  int i, j, m;
  int option = 0;
  int numobjs = 0;
  for (j=0; j < count; j++) {
    numobjs += stats[j].n_objs;
  }

  // allocate space for the computing data
  double *objtime = new double[numobjs];
  int *objwt = new int[numobjs];
  int *origmap = new int[numobjs];
  LDObjHandle *handles = new LDObjHandle[numobjs];
  for(i=0;i<numobjs;i++) {
    objtime[i] = 0.0;
    objwt[i] = 0;
    origmap[i] = 0;
  }

  int k=0;
  for (j=0; j<count; j++) {
    for (i=0; i<stats[j].n_objs; i++) {
      LDObjData *odata = stats[j].objData;
      /*
      origmap[odata[i].id.id[0]] = j;
      cputime[odata[i].id.id[0]] = odata[i].cpuTime;
      handles[odata[i].id.id[0]] = odata[i].handle;
      */
      origmap[k] = j;
      objtime[k] = odata[i].wallTime*stats[j].pe_speed;
      handles[k] = odata[i].handle;
      k++;
    }
  }

  // to convert the weights on vertices to integers
  double max_objtime = objtime[0];
  for(i=0; i<numobjs; i++) {
    if(max_objtime < objtime[i])
      max_objtime = objtime[i];
  }
  double ratio = 1000.0/max_objtime;
  for(i=0; i<numobjs; i++) {
      objwt[i] = (int)(objtime[i]*ratio);
  }
  int **comm = new int*[numobjs];
  for (i=0; i<numobjs; i++) {
    comm[i] = new int[numobjs];
    for (j=0; j<numobjs; j++)  {
      comm[i][j] = 0;
    }
  }

  for(j=0; j<count; j++) {
    LDCommData *cdata = stats[j].commData;
    const int csz = stats[j].n_comm;
    for(i=0; i<csz; i++) {
      if(cdata[i].from_proc || cdata[i].to_proc)
        continue;
      int senderID = cdata[i].sender.id[0];
      int recverID = cdata[i].receiver.id[0];
      comm[senderID][recverID] += cdata[i].messages;
      comm[recverID][senderID] += cdata[i].messages;
    }
  }
  // ignore messages sent from an object to itself
  for (i=0; i<numobjs; i++)
    comm[i][i] = 0;

  // construct the graph in CSR format
  int *xadj = new int[numobjs+1];
  int numedges = 0;
  for(i=0;i<numobjs;i++) {
    for(j=0;j<numobjs;j++) {
      if(comm[i][j] != 0)
        numedges++;
    }
  }
  int *adjncy = new int[numedges];
  int *edgewt = new int[numedges];
  xadj[0] = 0;
  int count4all = 0;
  for (i=0; i<numobjs; i++) {
    for (j=0; j<numobjs; j++) { 
      if (comm[i][j] != 0) { 
        adjncy[count4all] = j;
        edgewt[count4all++] = comm[i][j];
      }
    }
    xadj[i+1] = count4all;
  }

  //  CkPrintf("Pre-LDB Statistics step %d\n", step());
  //  printStats(count, numobjs, objtime, comm, origmap);

  int wgtflag = 3; // Weights both on vertices and edges
  int numflag = 0; // C Style numbering
  int options[5];
  options[0] = 0;
  int edgecut;
  int *newmap;
  int sameMapFlag = 1;

  if (count < 1) {
    CkPrintf("error: Number of Pe less than 1!");
  }
  else if (count == 1) {
    newmap = origmap;
    sameMapFlag = 1;
  }
  else {
    sameMapFlag = 0;
    newmap = new int[numobjs];
    for(i=0;i<(numobjs+1);i++)
      xadj[i] = 0;
    delete[] edgewt;
    edgewt = 0;
    wgtflag = 2;
    // CkPrintf("before calling Metis functions. option is %d.\n", option);
    if (0 == option) {

/*  I intended to follow the instruction in the Metis 4.0 manual
    which said that METIS_PartGraphKway is preferable to 
    METIS_PartGraphRecursive, when nparts > 8.
    However, it turned out that there is bug in METIS_PartGraphKway,
    and the function seg faulted when nparts = 4 or 9.
    So right now I just comment that function out and always use the other one.
*/
/*
      if (count > 8)
	METIS_PartGraphKway(&numobjs, xadj, adjncy, objwt, edgewt, 
			    &wgtflag, &numflag, &count, options, 
			    &edgecut, newmap);
      else
	METIS_PartGraphRecursive(&numobjs, xadj, adjncy, objwt, edgewt, 
				 &wgtflag, &numflag, &count, options, 
				 &edgecut, newmap);
*/
      METIS_PartGraphRecursive(&numobjs, xadj, adjncy, objwt, edgewt,
                                 &wgtflag, &numflag, &count, options,
                                 &edgecut, newmap);
      // CkPrintf("after calling Metis functions.\n");
    }
    else if (WEIGHTED == option) {
      float maxtotal_walltime = stats[0].total_walltime;
      for (m=1; m<count; m++) {
	if (maxtotal_walltime < stats[m].total_walltime)
	  maxtotal_walltime = stats[m].total_walltime;
      }
      float totaltimeAllPe = 0.0;
      for (m=0; m<count; m++) {
	totaltimeAllPe += stats[m].pe_speed * 
	  (maxtotal_walltime-stats[m].bg_walltime);
      }
      // set up the different weights
      float *tpwgts = new float[count];
      for (m=0; m<count; m++) {
	tpwgts[m] = stats[m].pe_speed * 
	  (maxtotal_walltime-stats[m].bg_walltime) / totaltimeAllPe;
      }
      if (count > 8)
	METIS_WPartGraphKway(&numobjs, xadj, adjncy, objwt, edgewt, 
			     &wgtflag, &numflag, &count, tpwgts, options, 
			     &edgecut, newmap);
      else
	METIS_WPartGraphRecursive(&numobjs, xadj, adjncy, objwt, edgewt, 
				  &wgtflag, &numflag, &count, tpwgts, options, 
				  &edgecut, newmap);
      delete[] tpwgts;
    }
    else if (MULTI_CONSTRAINT == option) {
      CkPrintf("Metis load balance strategy: ");
      CkPrintf("multiple constraints not implemented yet.\n");
    }
  }
  //  CkPrintf("Post-LDB Statistics step %d\n", step());
  //  printStats(count, numobjs, objtime, comm, newmap);

  for(i=0;i<numobjs;i++)
    delete[] comm[i];
  delete[] comm;
  delete[] objtime;
  delete[] xadj;
  delete[] adjncy;
  if(objwt) delete[] objwt;
  if(edgewt) delete[] edgewt;

  if(!sameMapFlag) {
    for(i=0; i<numobjs; i++) {
      if(origmap[i] != newmap[i]) {
	MigrateInfo* migrateMe = new MigrateInfo;
	migrateMe->obj = handles[i];
	migrateMe->from_pe = origmap[i];
	migrateMe->to_pe = newmap[i];
	migrateInfo.push_back(migrateMe);
      }
    }
  }

  delete[] origmap;
  if(newmap != origmap)
    delete[] newmap;

  int migrate_count=migrateInfo.size();
  //  CkPrintf("Migration Count = %d\n", migrate_count);
  CLBMigrateMsg* msg = new(&migrate_count,1) CLBMigrateMsg;
  msg->n_moves = migrate_count;
  for(i=0; i < migrate_count; i++) {
    MigrateInfo* item = (MigrateInfo*)migrateInfo[i];
    msg->moves[i] = *item;
    delete item;
    migrateInfo[i] = 0;
  }

  return msg;
}

#endif
