/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "elements.h"
#include "ckheap.h"
#include "NeighborLB.h"

CreateLBFunc_Def(NeighborLB, "The neighborhood load balancer")

NeighborLB::NeighborLB(const CkLBOptions &opt):CBase_NeighborLB(opt)
{
  lbname = "NeighborLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] NeighborLB created\n",CkMyPe());
}

LBMigrateMsg* NeighborLB::Strategy(NborBaseLB::LDStats* stats, int n_nbrs)
{
#if CMK_LBDB_ON
  //  CkPrintf("[%d] Strategy starting\n",CkMyPe());
  // Compute the average load to see if we are overloaded relative
  // to our neighbors
  double myload = myStats.total_walltime - myStats.idletime;
  double avgload = myload;
  int i;
  for(i=0; i < n_nbrs; i++) {
    // Scale times we need appropriately for relative proc speeds
    const double scale =  ((double)myStats.pe_speed) 
      / stats[i].pe_speed;

    stats[i].total_walltime *= scale;
    stats[i].idletime *= scale;

    avgload += (stats[i].total_walltime - stats[i].idletime);
  }
  avgload /= (n_nbrs + 1);

  CkVec<MigrateInfo*> migrateInfo;

  if (myload > avgload) {
    if (_lb_args.debug()>1) 
      CkPrintf("[%d] OVERLOAD My load is %f, average load is %f\n", CkMyPe(),myload,avgload);

    // First, build heaps of other processors and my objects
    // Then assign objects to other processors until either
    //   - The smallest remaining object would put me below average, or
    //   - I only have 1 object left, or
    //   - The smallest remaining object would put someone else 
    //     above average

    // Build heaps
    minHeap procs(n_nbrs);
    for(i=0; i < n_nbrs; i++) {
      InfoRecord* item = new InfoRecord;
      item->load = stats[i].total_walltime - stats[i].idletime;
      item->Id =  stats[i].from_pe;
      procs.insert(item);
    }
      
    maxHeap objs(myStats.n_objs);
    for(i=0; i < myStats.n_objs; i++) {
      InfoRecord* item = new InfoRecord;
      item->load = myStats.objData[i].wallTime;
      item->Id = i;
      objs.insert(item);
    }

    int objs_here = myStats.n_objs;
    do {
      if (objs_here <= 1) break;  // For now, always leave 1 object

      InfoRecord* p;
      InfoRecord* obj;

      // Get the lightest-loaded processor
      p = procs.deleteMin();
      if (p == 0) {
	//	CkPrintf("[%d] No destination PE found!\n",CkMyPe());
	break;
      }

      // Get the biggest object
      bool objfound = false;
      do {
	obj = objs.deleteMax();
	if (obj == 0) break;

	double new_p_load = p->load + obj->load;
	double my_new_load = myload - obj->load;
	if (new_p_load < my_new_load) {
//	if (new_p_load < avgload) {
	  objfound = true;
	} else {
	  // This object is too big, so throw it away
//	  CkPrintf("[%d] Can't move object w/ load %f to proc %d load %f %f\n",
//		   CkMyPe(),obj->load,p->Id,p->load,avgload);
	  delete obj;
	}
      } while (!objfound);

      if (!objfound) {
        if (_lb_args.debug()>2) 
	  CkPrintf("[%d] No suitable object found!\n",CkMyPe());
	break;
      }

      const int me = CkMyPe();
      // Apparently we can give this object to this processor
      //      CkPrintf("[%d] Obj %d of %d migrating from %d to %d\n",
      //      	       CkMyPe(),obj->Id,myStats.n_objs,me,p->Id);

      MigrateInfo* migrateMe = new MigrateInfo;
      migrateMe->obj = myStats.objData[obj->Id].handle;
      migrateMe->from_pe = me;
      migrateMe->to_pe = p->Id;
      migrateInfo.insertAtEnd(migrateMe);

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
    InfoRecord* p;
    while (NULL!=(p=procs.deleteMin()))
      delete p;

    InfoRecord* obj;
    while (NULL!=(obj=objs.deleteMax()))
      delete obj;
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
}

#include "NeighborLB.def.h"

/*@}*/
