#include <charm++.h>

#if CMK_LBDB_ON

#include "CkLists.h"

#include "heap.h"
#include "WSLB.h"
#include "WSLB.def.h"

void CreateWSLB()
{
  loadbalancer = CProxy_WSLB::ckNew();
}

WSLB::WSLB()
{
  if (CkMyPe() == 0)
    CkPrintf("[%d] WSLB created\n",CkMyPe());
}

NLBMigrateMsg* WSLB::Strategy(NeighborLB::LDStats* stats, int count)
{
  //  CkPrintf("[%d] Strategy starting\n",CkMyPe());
  // Compute the average load to see if we are overloaded relative
  // to our neighbors
  double myload = myStats.total_walltime - myStats.idletime;
  double avgload = myload;
  int i;
  for(i=0; i < count; i++) {
    // Scale times we need appropriately for relative proc speeds
    const double scale =  ((double)myStats.proc_speed) 
      / stats[i].proc_speed;

    stats[i].total_walltime *= scale;
    stats[i].idletime *= scale;

    avgload += (stats[i].total_walltime - stats[i].idletime);
  }
  avgload /= (count+1);

  CkVector migrateInfo;

  if (myload < avgload)
    CkPrintf("[%d] Underload My load is %f, average load is %f\n",
	     CkMyPe(),myload,avgload);
  else {
    CkPrintf("[%d] OVERLOAD My load is %f, average load is %f\n",
	     CkMyPe(),myload,avgload);

    // First, build heaps of other processors and my objects
    // Then assign objects to other processors until either
    //   - The smallest remaining object would put me below average, or
    //   - I only have 1 object left, or
    //   - The smallest remaining object would put someone else 
    //     above average

    // Build heaps
    minHeap procs(count);
    for(i=0; i < count; i++) {
      InfoRecord* item = new InfoRecord;
      item->load = stats[i].total_walltime - stats[i].idletime;
      item->Id =  stats[i].from_pe;
      procs.insert(item);
    }
      
    maxHeap objs(myStats.obj_data_sz);
    for(i=0; i < myStats.obj_data_sz; i++) {
      InfoRecord* item = new InfoRecord;
      item->load = myStats.objData[i].wallTime;
      item->Id = i;
      objs.insert(item);
    }

    int objs_here = myStats.obj_data_sz;
    do {
      if (objs_here <= 1) break;  // For now, always leave 1 object

      InfoRecord* p;
      InfoRecord* obj;

      // Get the lightest-loaded processor
      p = procs.deleteMin();
      if (p == 0) {
	CkPrintf("[%d] No destination PE found!\n",CkMyPe());
	break;
      }

      // Get the biggest object
      CmiBool objfound = CmiFalse;
      do {
	obj = objs.deleteMax();
	if (obj == 0) break;

	double new_p_load = p->load + obj->load;
	double my_new_load = myload - obj->load;
	if (new_p_load < my_new_load) {
//	if (new_p_load < avgload) {
	  objfound = CmiTrue;
	} else {
	  // This object is too big, so throw it away
//	  CkPrintf("[%d] Can't move object w/ load %f to proc %d load %f %f\n",
//		   CkMyPe(),obj->load,p->Id,p->load,avgload);
	  delete obj;
	}
      } while (!objfound);

      if (!objfound) {
	//	CkPrintf("[%d] No suitable object found!\n",CkMyPe());
	break;
      }

      const int me = CkMyPe();
      // Apparently we can give this object to this processor
      CkPrintf("[%d] Obj %d of %d migrating from %d to %d\n",
	       CkMyPe(),obj->Id,myStats.obj_data_sz,me,p->Id);

      MigrateInfo* migrateMe = new MigrateInfo;
      migrateMe->obj = myStats.objData[obj->Id].handle;
      migrateMe->from_pe = me;
      migrateMe->to_pe = p->Id;
      migrateInfo.push_back((void*)migrateMe);

      objs_here--;
      
      // We may want to assign more to this processor, so lets
      // update it and put it back in the heap
      p->load += obj->load;
      myload -= obj->load;
      procs.insert(p);
      
      // This object is assigned, so we delete it from the heap
      delete obj;

    } while(myload > avgload);

    // Now empty out the heaps
    while (InfoRecord* p=procs.deleteMin())
      delete p;
    while (InfoRecord* obj=objs.deleteMax())
      delete obj;
  }  

  // Now build the message to actually perform the migrations
  int migrate_count=migrateInfo.size();
  NLBMigrateMsg* msg = new(&migrate_count,1) NLBMigrateMsg;
  msg->n_moves = migrate_count;
  for(i=0; i < migrate_count; i++) {
    MigrateInfo* item = (MigrateInfo*) migrateInfo[i];
    msg->moves[i] = *item;
    delete item;
    migrateInfo[i] = 0;
  }

  return msg;
};

#endif
