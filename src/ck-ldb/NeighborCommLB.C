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

#include "cklists.h"

#include "heap.h"
#include "NeighborCommLB.h"
#include "topology.h"

#define PER_MESSAGE_SEND_OVERHEAD   35e-6
#define PER_BYTE_SEND_OVERHEAD      8.5e-9
#define PER_MESSAGE_RECV_OVERHEAD   0.0
#define PER_BYTE_RECV_OVERHEAD      0.0

CreateLBFunc_Def(NeighborCommLB);

static void lbinit(void) {
//  LBSetDefaultCreate(CreateNeighborCommLB);
  LBRegisterBalancer("NeighborCommLB", 
                     CreateNeighborCommLB, 
                     AllocateNeighborCommLB, 
                     "The neighborhood load balancer with communication");
}

#include "NeighborCommLB.def.h"

NeighborCommLB::NeighborCommLB(const CkLBOptions &opt):NborBaseLB(opt)
{
  lbname = "NeighborCommLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] NeighborCommLB created\n",CkMyPe());
}

LBMigrateMsg* NeighborCommLB::Strategy(NborBaseLB::LDStats* stats, int count)
{
bool _lb_debug=1;
#if CMK_LBDB_ON
  //  CkPrintf("[%d] Strategy starting\n",CkMyPe());
  // Compute the average load to see if we are overloaded relative
  // to our neighbors
  double myload = myStats.total_walltime - myStats.idletime;
  double avgload = myload;
  int i;
  if (_lb_debug) 
    CkPrintf("[%d] Neighbor Count = %d\n", CkMyPe(), count);
  
  for(i=0; i < count; i++) {
    // Scale times we need appropriately for relative proc speeds
    const double scale =  ((double)myStats.pe_speed) 
      / stats[i].pe_speed;

    stats[i].total_walltime *= scale;
    stats[i].idletime *= scale;

    avgload += (stats[i].total_walltime - stats[i].idletime);
  }
  avgload /= (count+1);

  CkVec<MigrateInfo*> migrateInfo;

  if (myload > avgload) {
    if (_lb_debug) 
      CkPrintf("[%d] OVERLOAD My load is %f, average load is %f\n", CkMyPe(),myload,avgload);

    // First of all, explore the topology and get dimension
    LBTopology* topo;
    {
      LBtopoFn topofn;
      topofn = LBTopoLookup(_lbtopo);
      if (topofn == NULL) {
        char str[1024];
        CmiPrintf("NeighborCommLB> Fatal error: Unknown topology: %s. Choose from:\n", _lbtopo);
        printoutTopo();
        sprintf(str, "NeighborCommLB> Fatal error: Unknown topology: %s", _lbtopo);
        CmiAbort(str);
      }
      topo = topofn();
    }
    int dimension = topo->get_dimension();
    if (_lb_debug) 
      CkPrintf("[%d] Topology dimension = %d\n", CkMyPe(), dimension);
    if (dimension == -1) {
      char str[1024];
      CmiPrintf("NeighborCommLB> Fatal error: Unsupported topology: %s. Only some of the following are supported:\n", _lbtopo);
      printoutTopo();
      sprintf(str, "NeighborCommLB> Fatal error: Unsupported topology: %s", _lbtopo);
      CmiAbort(str);
    }

    // Position of this processor
    int *myProc = new int[dimension];
    topo->get_processor_coordinates(myStats.from_pe, myProc);
    if (_lb_debug) {
      char temp[1000];
      char* now=temp;
      sprintf(now, "[%d] Coordinates = [", CkMyPe());
      now += strlen(now);
      for(i=0;i<dimension;i++) {
        sprintf(now, "%d ", myProc[i]); 
        now +=strlen(now);
      }
      sprintf(now, "]\n");
      now += strlen(now);
      CkPrintf(temp);
    }

    // Then calculate the communication center of each object
    // The communication center is relative to myProc
    double **commcenter = new double*[myStats.n_objs];
    double *commamount = new double[myStats.n_objs];
    if(_lb_debug) {
      CkPrintf("[%d] Number of Objs = %d \n", CkMyPe(), myStats.n_objs);
    }
    {
      memset(commamount, 0, sizeof(double)*myStats.n_objs);
      for(i=0; i<myStats.n_objs;i++) {
        commcenter[i] = new double[dimension];
        memset(commcenter[i], 0, sizeof(double)*dimension);
      }

      //coordinates of procs
      int *destProc = new int[dimension];
      int *diff = new int[dimension];
      
      //for each comm entry
      for(i=0; i<myStats.n_comm;i++) {
        int j;
        //for each object //TODO use hashtable to accelerate
        for(j=0; j<myStats.n_objs;j++) 
          if((myStats.objData[j].handle.omhandle.id == myStats.commData[i].sender.omId)
              && (myStats.objData[j].handle.id == myStats.commData[i].sender.objId)) {
            double comm=
              PER_MESSAGE_SEND_OVERHEAD * myStats.commData[i].messages 
              + PER_BYTE_SEND_OVERHEAD * myStats.commData[i].bytes;
            commamount[j] += comm;
            int dest_pe = myStats.commData[i].receiver.proc();
            if(dest_pe==-1) continue;
            topo->get_processor_coordinates(dest_pe, destProc);
            topo->coordinate_difference(myProc, destProc, diff);
            int k;
            for(k=0;k<dimension;k++) {
              commcenter[j][k] += diff[k] * comm;
            }
          }
      }
      for(i=0; i<myStats.n_objs;i++) if (commamount[i]>0) {
        int k;
        double ratio = 1.0 /commamount[i];
        for(k=0;k<dimension;k++)
          commcenter[i][k] *= ratio;
      } else { //if no communication, set commcenter to myself
        int k;
        for(k=0;k<dimension;k++)
          commcenter[i][k] = myProc[k];
      }
      
      delete [] destProc;
      delete [] diff;
    }
    
    if(_lb_debug) {
      for(i=0;i<myStats.n_objs;i++) {
        char temp[1000];
        char* now=temp;
        sprintf(now, "[%d] Objs [%d] Comm Amount = %lf  ", CkMyPe(), i, commamount[i]);
        now += strlen(now);
        sprintf(now, "Comm Center = [");
        now += strlen(now);
        int j;
        for(j=0;j<dimension;j++) {
          sprintf(now, "%.2lf ", commcenter[j]); 
          now +=strlen(now);
        }
        sprintf(now, "]\n");
        now += strlen(now);
        CkPrintf(temp);
      }
    }
    
    // First, build heaps of my objects
    // Then assign objects to the least loaded other processors until either
    //   - The smallest remaining object would put me below average, or
    //   - I only have 1 object left, or
    //   - The smallest remaining object would put someone else 
    //     above average
    // Note: Object can only move towards its communication center!

    // My neighbors: 
    typedef struct _procInfo{
      int id;
      double load;
      int* difference;
    } procInfo;

    if(_lb_debug) {
      CkPrintf("[%d] Querying neighborhood topology...\n", CkMyPe() );
    }

    procInfo* neighbors = new procInfo[count];
    {
      int *destProc = new int[dimension];
      for(i=0; i < count; i++) {
        neighbors[i].id = stats[i].from_pe;
        neighbors[i].load = stats[i].total_walltime - stats[i].idletime;
        neighbors[i].difference = new int[dimension];
        topo->get_processor_coordinates(neighbors[i].id, destProc);
        topo->coordinate_difference(myProc, destProc, neighbors[i].difference);
      }
      delete[] destProc;
    }
    
    if(_lb_debug) {
      CkPrintf("[%d] Building obj heap...\n", CkMyPe() );
    }
    // My objects: build heaps
    maxHeap objs(myStats.n_objs);
    for(i=0; i < myStats.n_objs; i++) {
      InfoRecord* item = new InfoRecord;
      item->load = myStats.objData[i].wallTime;
      item->Id = i;
      objs.insert(item);
    }

    if(_lb_debug) {
      CkPrintf("[%d] Beginning distributing objects...\n", CkMyPe() );
    }

    // for each object
    while(objs.numElements()>0) {
      InfoRecord* obj;
      obj = objs.deleteMax();
      int bestDest = -1;
      for(i=0; i<count;i++) if(neighbors[i].load < myload && (bestDest==-1 || neighbors[i].load < neighbors[bestDest].load)) {
        double dotsum=0;
        int j;
        for(j=0; j<dimension; j++) dotsum += (commcenter[obj->Id][j] * neighbors[i].difference[j]);
        if(dotsum>0 || commamount[obj->Id]==0) {
          bestDest = i;
        }
      }
      // Best place for the object
      if(bestDest != -1) {
        if(_lb_debug) {
          CkPrintf("[%d] Obj[%d] will move to Proc[%d]\n", CkMyPe(), obj->Id, neighbors[bestDest].id);
        }
        //Migrate it
        MigrateInfo* migrateMe = new MigrateInfo;
        migrateMe->obj = myStats.objData[obj->Id].handle;
        migrateMe->from_pe = myStats.from_pe;
        migrateMe->to_pe = neighbors[bestDest].id;
        migrateInfo.insertAtEnd(migrateMe);
        //Modify loads
        myload -= obj->load;
        neighbors[bestDest].load += obj->load;
      }
      delete obj;
    }

    if(_lb_debug) {
      CkPrintf("[%d] Clearing Up...\n", CkMyPe());
    }

    for(i=0;i<count;i++) {
      delete[] neighbors[i].difference;
    }
    delete[] neighbors;
    
    delete[] myProc;

    for(i=0;i<myStats.n_objs;i++) {
      delete[] commcenter[i];
    }
    delete[] commcenter;
    delete[] commamount;        
  }  

  if(_lb_debug) {
    CkPrintf("[%d] Generating result...\n", CkMyPe());
  }

  // Now build the message to actually perform the migrations
  int migrate_count=migrateInfo.length();
  //  if (migrate_count > 0) {
  //    CkPrintf("PE %d migrating %d elements\n",CkMyPe(),migrate_count);
  //  }
  LBMigrateMsg* msg = new(migrate_count,CkNumPes(),CkNumPes(),0) LBMigrateMsg;
  msg->n_moves = migrate_count;
  for(i=0; i < migrate_count; i++) {
    MigrateInfo* item = (MigrateInfo*) migrateInfo[i];
    msg->moves[i] = *item;
    delete item;
    migrateInfo[i] = 0;
  }

  return msg;
#else
  return NULL;
#endif
};


/*@}*/
