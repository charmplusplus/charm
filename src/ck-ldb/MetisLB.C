/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkLdb
*/
/*@{*/

#include <charm++.h>

#if CMK_LBDB_ON

#include "cklists.h"

#include "MetisLB.h"

CreateLBFunc_Def(MetisLB);

static void lbinit(void) {
//        LBSetDefaultCreate(CreateMetisLB);
  LBRegisterBalancer("MetisLB", CreateMetisLB, "Use Metis(tm) to partition object graph");
}

#include "MetisLB.def.h"

MetisLB::MetisLB(const CkLBOptions &opt): CentralLB(opt)
{
  lbname = "MetisLB";
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
LBMigrateMsg* MetisLB::Strategy(CentralLB::LDStats* stats, int count,
				 int option=0)
*/
void MetisLB::work(CentralLB::LDStats* stats, int count)
{
  // CkPrintf("entering MetisLB::Strategy...\n");
  // CkPrintf("[%d] MetisLB strategy\n",CkMyPe());
  int i, j, m;
  int option = 0;
  int numobjs = stats->n_objs;

  stats->makeCommHash();

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

  for (i=0; i<stats->n_objs; i++) {
      LDObjData &odata = stats->objData[i];
      if (!odata.migratable) 
        CmiAbort("MetisLB doesnot dupport nonmigratable object.\n");
      /*
      origmap[odata[i].id.id[0]] = j;
      cputime[odata[i].id.id[0]] = odata[i].cpuTime;
      handles[odata[i].id.id[0]] = odata[i].handle;
      */
      int frompe = stats->from_proc[i];
      origmap[i] = frompe;
      objtime[i] = odata.wallTime*stats->procs[frompe].pe_speed;
      handles[i] = odata.handle;
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

  const int csz = stats->n_comm;
  for(i=0; i<csz; i++) {
      LDCommData &cdata = stats->commData[i];
      if(cdata.from_proc() || cdata.receiver.get_type() != LD_OBJ_MSG)
        continue;
      int senderID = stats->getHash(cdata.sender);
      int recverID = stats->getHash(cdata.receiver.get_destObj());
      CmiAssert(senderID < numobjs);
      CmiAssert(recverID < numobjs);
      comm[senderID][recverID] += cdata.messages;
      comm[recverID][senderID] += cdata.messages;
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

  //CkPrintf("Pre-LDB Statistics step %d\n", step());
  //printStats(count, numobjs, objtime, comm, origmap);

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
      float maxtotal_walltime = stats->procs[0].total_walltime;
      for (m=1; m<count; m++) {
	if (maxtotal_walltime < stats->procs[m].total_walltime)
	  maxtotal_walltime = stats->procs[m].total_walltime;
      }
      float totaltimeAllPe = 0.0;
      for (m=0; m<count; m++) {
	totaltimeAllPe += stats->procs[m].pe_speed * 
	  (maxtotal_walltime-stats->procs[m].bg_walltime);
      }
      // set up the different weights
      float *tpwgts = new float[count];
      for (m=0; m<count; m++) {
	tpwgts[m] = stats->procs[m].pe_speed * 
	  (maxtotal_walltime-stats->procs[m].bg_walltime) / totaltimeAllPe;
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
  //CkPrintf("Post-LDB Statistics step %d\n", step());
  //printStats(count, numobjs, objtime, comm, newmap);

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
	CmiAssert(stats->from_proc[i] == origmap[i]);
	stats->to_proc[i] =  newmap[i];
      }
    }
  }

  delete[] origmap;
  if(newmap != origmap)
    delete[] newmap;
}

#endif

/*@}*/
